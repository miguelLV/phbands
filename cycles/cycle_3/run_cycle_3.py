from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import jn_zeros


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from phbands.bandstructure import compute_band_structure
from phbands.inplane.media import InPlaneMedium
from phbands.inplane.secular import InPlaneSecularProblem
from phbands.lattice import SquareLattice, build_path
from phbands.lattice_sums import LatticeSumConfig
from phbands.rootfinding import RootFindingConfig
from phbands.rootfinding.gap_detection import detect_bridge_gaps, detect_continuity_mismatches
from phbands.rootfinding.validation import equilibrate_matrix, refine_matrix_candidate


A = 1.0
M_MAX = 2
H_PRIMARY = 8
H_CHECK = 12
N_SCAN = 200
N_PER_SEGMENT = 40
OMEGA_BAR_MAX = 1.0
LOCAL_GRID_SIZE = 101
LOCAL_HALF_WIDTH_BAR = 0.02
SINGULAR_RATIO_CUTOFF = 2e-2
SIGMA_PRIMARY_MAX = 1e-4
SIGMA_CHECK_MAX = 5e-3
MAX_SHIFT_BAR = 2e-3
MATCH_EXISTING_BAR = 2e-4
RECIPROCAL_POLE_GUARD_BAR = 2e-3
BESSEL_POLE_GUARD = 1e-2
POLE_SUPPORT_RADIUS = 4
MIN_POLE_SUPPORT = 5
POLE_SUPPORT_DELTA_BAR = 1.5e-2
LOW_FREQUENCY_BAR = 1e-2
GAMMA_NEIGHBORHOOD = 0.2
HIGH_SYMMETRY_INDICES = [0, 40, 80, 119]
MULTIPLICITY_THRESHOLD = 1e-3
MULTIPLICITY_RADIUS = 3
CONTINUITY_RECOVERY = False
CONTINUITY_MATCH_TOLERANCE = 0.010
POLE_CHECK_H = None
RECIPROCAL_POLE_MIN_SUPPORT = 3
RECIPROCAL_POLE_SUPPORT_DELTA_PER_STEP = 0.025

SYSTEM_LABEL = "Pb/epoxy"
FILLING_FRACTION = 0.30
CYCLE_NUMBER = 3
OUTPUT_TAG = "cycle_3"
HOST = InPlaneMedium(1.18, 2.54, 2.54 / 2.19)
INCLUSION = InPlaneMedium(11.4, 2.16, 2.16 / 2.51)
RADIUS = float(np.sqrt(FILLING_FRACTION / np.pi))
OMEGA_SCALE = 2 * np.pi * HOST.c_t


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nearest_bessel_zero_distance(omega_bar: float):
    omega = omega_bar * OMEGA_SCALE
    distances = []
    for channel, argument in (("l", omega / HOST.c_l), ("t", omega / HOST.c_t)):
        for order in range(1, 2 * M_MAX + 2):
            zeros = jn_zeros(order, 4)
            position = int(np.argmin(np.abs(zeros - argument)))
            distances.append((abs(float(zeros[position] - argument)), channel, order, float(zeros[position])))
    return min(distances)


def nearest_reciprocal_pole_bar(k: np.ndarray, omega_bar: float):
    poles = []
    reciprocal_step = 2 * np.pi / A
    for hx in range(-H_CHECK, H_CHECK + 1):
        for hy in range(-H_CHECK, H_CHECK + 1):
            q = float(np.hypot(reciprocal_step * hx + k[0], reciprocal_step * hy + k[1]))
            poles.append((q / (2 * np.pi), "t"))
            poles.append(((HOST.c_l / HOST.c_t) * q / (2 * np.pi), "l"))
    pole_bar, channel = min(poles, key=lambda item: abs(item[0] - omega_bar))
    return abs(pole_bar - omega_bar), channel, pole_bar


def local_minima(values: np.ndarray):
    finite = np.isfinite(values)
    return np.where(
        finite[1:-1]
        & finite[:-2]
        & finite[2:]
        & (values[1:-1] <= values[:-2])
        & (values[1:-1] <= values[2:])
    )[0] + 1


def merge_windows(raw_windows):
    grouped = defaultdict(list)
    for window in raw_windows:
        grouped[int(window["k_index"])].append(window)
    merged = []
    for index, windows in sorted(grouped.items()):
        for window in sorted(windows, key=lambda item: item["lower_bar"]):
            if merged and merged[-1]["k_index"] == index and window["lower_bar"] <= merged[-1]["upper_bar"]:
                merged[-1]["lower_bar"] = min(merged[-1]["lower_bar"], window["lower_bar"])
                merged[-1]["upper_bar"] = max(merged[-1]["upper_bar"], window["upper_bar"])
                merged[-1]["reasons"] = sorted(set(merged[-1]["reasons"] + window["reasons"]))
            else:
                merged.append(dict(window))
    return merged


def bessel_support(index, frequency, bands):
    supported = {index}
    for neighbor in range(max(0, index - POLE_SUPPORT_RADIUS), min(len(bands), index + POLE_SUPPORT_RADIUS + 1)):
        if neighbor == index:
            continue
        if any(abs(value - frequency) <= POLE_SUPPORT_DELTA_BAR for value in bands[neighbor]):
            supported.add(neighbor)
    return len(supported)


def reciprocal_pole_support(index, frequency, bands):
    """Soporte de una rama permitiendo su pendiente local a lo largo de k."""
    supported = {index}
    for neighbor in range(max(0, index - 2), min(len(bands), index + 3)):
        if neighbor == index:
            continue
        tolerance = RECIPROCAL_POLE_SUPPORT_DELTA_PER_STEP * abs(neighbor - index)
        if any(abs(value - frequency) <= tolerance for value in bands[neighbor]):
            supported.add(neighbor)
    return len(supported)


def classify_initial_candidates(result, path):
    raw_bands = [[omega / OMEGA_SCALE for omega in roots] for roots in result.bands]
    accepted = [[] for _ in raw_bands]
    rejected = []
    for index, values in enumerate(raw_bands):
        k = path.k_points[index]
        for frequency in values:
            bessel_distance, channel, order, zero = nearest_bessel_zero_distance(frequency)
            unsupported_bessel = bool(
                bessel_distance <= BESSEL_POLE_GUARD
                and bessel_support(index, frequency, raw_bands) < MIN_POLE_SUPPORT
            )
            low_away = bool(frequency <= LOW_FREQUENCY_BAR and np.linalg.norm(k) > GAMMA_NEIGHBORHOOD)
            if unsupported_bessel or low_away:
                rejected.append(
                    {
                        "k_index": index,
                        "omega_bar": frequency,
                        "reason": "unsupported_bessel_pole" if unsupported_bessel else "low_frequency_away_from_gamma",
                        "bessel_channel": channel,
                        "bessel_order": order,
                        "bessel_zero": zero,
                    }
                )
            else:
                accepted[index].append(frequency)
        accepted[index].sort()
    return raw_bands, accepted, rejected


def activation_windows(bands, path, problem):
    raw_windows = []
    for trigger in detect_bridge_gaps(bands):
        raw_windows.append(
            {
                "k_index": trigger.k_index,
                "lower_bar": max(1e-6, trigger.frequency - LOCAL_HALF_WIDTH_BAR),
                "upper_bar": min(OMEGA_BAR_MAX, trigger.frequency + LOCAL_HALF_WIDTH_BAR),
                "reasons": [trigger.reason],
            }
        )
    if CONTINUITY_RECOVERY:
        for trigger in detect_continuity_mismatches(
            bands, match_tolerance=CONTINUITY_MATCH_TOLERANCE
        ):
            raw_windows.append(
                {
                    "k_index": trigger.k_index,
                    "lower_bar": max(1e-6, trigger.frequency - LOCAL_HALF_WIDTH_BAR),
                    "upper_bar": min(OMEGA_BAR_MAX, trigger.frequency + LOCAL_HALF_WIDTH_BAR),
                    "reasons": [trigger.reason],
                }
            )
    multiplicities = []
    for index in HIGH_SYMMETRY_INDICES:
        k = path.k_points[index]
        for frequency in bands[index]:
            matrix = equilibrate_matrix(problem.M(frequency * OMEGA_SCALE, k))
            singular = np.linalg.svd(matrix, compute_uv=False)
            second_ratio = float(singular[-2] / singular[0])
            if second_ratio > MULTIPLICITY_THRESHOLD:
                continue
            multiplicities.append({"k_index": index, "omega_bar": frequency, "second_singular_ratio": second_ratio})
            if index == 0:
                neighbors = range(1, 1 + MULTIPLICITY_RADIUS)
            elif index == len(path) - 1:
                neighbors = range(index - MULTIPLICITY_RADIUS, index)
            else:
                neighbors = [
                    neighbor
                    for offset in range(1, MULTIPLICITY_RADIUS + 1)
                    for neighbor in (index - offset, index + offset)
                    if 0 <= neighbor < len(path)
                ]
            for neighbor in neighbors:
                raw_windows.append(
                    {
                        "k_index": neighbor,
                        "lower_bar": max(1e-6, frequency - LOCAL_HALF_WIDTH_BAR),
                        "upper_bar": min(OMEGA_BAR_MAX, frequency + LOCAL_HALF_WIDTH_BAR),
                        "reasons": ["high_symmetry_multiplicity"],
                    }
                )
    return merge_windows(raw_windows), multiplicities


def search_windows(windows, bands, path, primary, check):
    pole_check = None
    if POLE_CHECK_H is not None:
        pole_check = InPlaneSecularProblem(
            SquareLattice(A), HOST, INCLUSION, RADIUS, M_MAX, LatticeSumConfig(POLE_CHECK_H)
        )
    candidates = []
    for number, window in enumerate(windows, 1):
        index = window["k_index"]
        k = path.k_points[index]
        frequency_grid = np.linspace(window["lower_bar"], window["upper_bar"], LOCAL_GRID_SIZE)
        omega_grid = frequency_grid * OMEGA_SCALE
        ratios = []
        for omega in omega_grid:
            singular = np.linalg.svd(primary.M(omega, k), compute_uv=False)
            ratios.append(float(singular[-1] / singular[0]))
        ratios = np.asarray(ratios)
        grid_step = float(omega_grid[1] - omega_grid[0])
        for position in local_minima(ratios):
            if ratios[position] > SINGULAR_RATIO_CUTOFF:
                continue
            seed = float(omega_grid[position])
            first = refine_matrix_candidate(lambda value: primary.M(value, k), seed, 1.5 * grid_step)
            second = refine_matrix_candidate(lambda value: check.M(value, k), first.omega, 1.5 * grid_step)
            frequency = second.omega / OMEGA_SCALE
            pole_distance, pole_channel, pole_bar = nearest_reciprocal_pole_bar(k, frequency)
            bessel_distance, bessel_channel, bessel_order, bessel_zero = nearest_bessel_zero_distance(frequency)
            shift = abs(second.omega - first.omega) / OMEGA_SCALE
            converged = bool(
                first.balanced_sigma_min <= SIGMA_PRIMARY_MAX
                and second.balanced_sigma_min <= SIGMA_CHECK_MAX
                and shift <= MAX_SHIFT_BAR
            )
            pole_convergence_pass = pole_distance > RECIPROCAL_POLE_GUARD_BAR
            pole_check_sigma = None
            pole_check_shift = None
            pole_support = reciprocal_pole_support(index, frequency, bands)
            if not pole_convergence_pass and pole_check is not None:
                third = refine_matrix_candidate(
                    lambda value: pole_check.M(value, k), second.omega, 1.5 * grid_step
                )
                pole_check_sigma = third.balanced_sigma_min
                pole_check_shift = abs(third.omega - second.omega) / OMEGA_SCALE
                frequency = third.omega / OMEGA_SCALE
                pole_convergence_pass = bool(
                    third.balanced_sigma_min <= SIGMA_CHECK_MAX
                    and pole_check_shift <= MAX_SHIFT_BAR
                    and pole_support >= RECIPROCAL_POLE_MIN_SUPPORT
                )
                pole_distance, pole_channel, pole_bar = nearest_reciprocal_pole_bar(k, frequency)
                bessel_distance, bessel_channel, bessel_order, bessel_zero = nearest_bessel_zero_distance(frequency)
            existing = any(abs(frequency - value) <= MATCH_EXISTING_BAR for value in bands[index])
            converged = converged and pole_convergence_pass
            candidates.append(
                {
                    "k_index": index,
                    "Kx": float(k[0]),
                    "Ky": float(k[1]),
                    "omega_bar": frequency,
                    "sigma_balanced_h8": first.balanced_sigma_min,
                    "sigma_balanced_h12": second.balanced_sigma_min,
                    "shift_h8_h12_bar": shift,
                    "nearest_reciprocal_pole_distance_bar": pole_distance,
                    "nearest_reciprocal_pole_channel": pole_channel,
                    "nearest_reciprocal_pole_bar": pole_bar,
                    "reciprocal_pole_support": pole_support,
                    "pole_check_sigma": pole_check_sigma,
                    "pole_check_shift_bar": pole_check_shift,
                    "pole_convergence_pass": pole_convergence_pass,
                    "nearest_bessel_zero_distance": bessel_distance,
                    "nearest_bessel_channel": bessel_channel,
                    "nearest_bessel_order": bessel_order,
                    "nearest_bessel_zero": bessel_zero,
                    "existing": existing,
                    "converged_preliminary": converged,
                    "window_reasons": "+".join(window["reasons"]),
                }
            )
        if number % 10 == 0 or number == len(windows):
            print(f"SVD local [{number}/{len(windows)}]")

    preliminary = [candidate for candidate in candidates if candidate["converged_preliminary"] and not candidate["existing"]]
    combined_for_support = [list(values) for values in bands]
    for candidate in preliminary:
        combined_for_support[candidate["k_index"]].append(candidate["omega_bar"])
    recovered = []
    for candidate in sorted(preliminary, key=lambda item: (item["k_index"], item["omega_bar"])):
        bessel_ok = bool(
            candidate["nearest_bessel_zero_distance"] > BESSEL_POLE_GUARD
            or bessel_support(candidate["k_index"], candidate["omega_bar"], combined_for_support) >= MIN_POLE_SUPPORT
        )
        candidate["bessel_support_pass"] = bessel_ok
        candidate["accepted"] = bessel_ok
        if not bessel_ok:
            continue
        if any(
            previous["k_index"] == candidate["k_index"]
            and abs(previous["omega_bar"] - candidate["omega_bar"]) <= MATCH_EXISTING_BAR
            for previous in recovered
        ):
            continue
        recovered.append(candidate)
    return candidates, recovered


def write_candidate_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_bands(path, bands, recovered, output, recovered_color=None, title=""):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for index, values in enumerate(bands):
        ax.scatter([path.distances[index]] * len(values), values, s=5, color="black")
    if recovered:
        ax.scatter(
            [path.distances[item["k_index"]] for item in recovered],
            [item["omega_bar"] for item in recovered],
            s=18 if recovered_color else 5,
            marker="x" if recovered_color else "o",
            color=recovered_color or "black",
            label="Recuperada por SVD" if recovered_color else None,
        )
    for position in path.tick_positions:
        ax.axvline(position, color="0.72", linewidth=0.6, zorder=0)
    ax.set_xticks(path.tick_positions)
    ax.set_xticklabels(path.tick_labels)
    ax.set_xlim(path.tick_positions[0], path.tick_positions[-1])
    ax.set_ylim(0.0, OMEGA_BAR_MAX)
    ax.set_ylabel(r"$\omega a/(2\pi c_{t,\mathrm{host}})$")
    ax.set_title(title)
    if recovered_color and recovered:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    total_started = time.perf_counter()
    path = build_path(
        [("M", 1.0, 1.0), ("Gamma", 0.0, 0.0), ("X", 1.0, 0.0), ("M", 1.0, 1.0)],
        A,
        N_PER_SEGMENT,
    )
    primary = InPlaneSecularProblem(
        SquareLattice(A), HOST, INCLUSION, RADIUS, M_MAX, LatticeSumConfig(H_PRIMARY)
    )
    check = InPlaneSecularProblem(
        SquareLattice(A), HOST, INCLUSION, RADIUS, M_MAX, LatticeSumConfig(H_CHECK)
    )

    scan_started = time.perf_counter()
    scan_result = compute_band_structure(
        primary,
        path,
        1e-6,
        OMEGA_BAR_MAX * OMEGA_SCALE,
        RootFindingConfig(n_scan=N_SCAN),
        verbose=True,
    )
    scan_elapsed = time.perf_counter() - scan_started

    filter_started = time.perf_counter()
    raw_bands, accepted_bands, rejected_initial = classify_initial_candidates(scan_result, path)
    windows, multiplicities = activation_windows(accepted_bands, path, primary)
    filter_elapsed = time.perf_counter() - filter_started

    recovery_started = time.perf_counter()
    local_candidates, recovered = search_windows(windows, accepted_bands, path, primary, check)
    recovery_elapsed = time.perf_counter() - recovery_started
    total_elapsed = time.perf_counter() - total_started

    rows = []
    for index, values in enumerate(accepted_bands):
        for frequency in values:
            rows.append(
                {
                    "source": "determinant_scan",
                    "k_index": index,
                    "Kx": float(path.k_points[index, 0]),
                    "Ky": float(path.k_points[index, 1]),
                    "path_distance": float(path.distances[index]),
                    "omega_bar": frequency,
                    "omega": frequency * OMEGA_SCALE,
                }
            )
    for candidate in recovered:
        rows.append(
            {
                "source": "local_svd_recovery",
                "k_index": candidate["k_index"],
                "Kx": candidate["Kx"],
                "Ky": candidate["Ky"],
                "path_distance": float(path.distances[candidate["k_index"]]),
                "omega_bar": candidate["omega_bar"],
                "omega": candidate["omega_bar"] * OMEGA_SCALE,
            }
        )
    rows.sort(key=lambda row: (row["k_index"], row["omega_bar"]))

    bands_csv = ROOT / f"bands_{OUTPUT_TAG}.csv"
    write_candidate_csv(bands_csv, rows)
    candidates_csv = ROOT / f"local_candidate_diagnostics_{OUTPUT_TAG}.csv"
    write_candidate_csv(candidates_csv, local_candidates)
    windows_json = ROOT / f"activation_windows_{OUTPUT_TAG}.json"
    windows_json.write_text(json.dumps(windows, indent=2), encoding="utf-8")
    rejected_json = ROOT / f"rejected_initial_candidates_{OUTPUT_TAG}.json"
    rejected_json.write_text(json.dumps(rejected_initial, indent=2), encoding="utf-8")

    final_image = ROOT / f"bands_{OUTPUT_TAG}.png"
    diagnostic_image = ROOT / f"bands_{OUTPUT_TAG}_sources.png"
    plot_bands(path, accepted_bands, recovered, final_image, title=f"Ciclo {CYCLE_NUMBER} — {SYSTEM_LABEL} in-plane, búsqueda híbrida completa")
    plot_bands(path, accepted_bands, recovered, diagnostic_image, recovered_color="red", title=f"Ciclo {CYCLE_NUMBER} — origen de las raíces")

    manifest = {
        "cycle": CYCLE_NUMBER,
        "purpose": "Corrida completa desde cero con determinante global y SVD local automatica",
        "input_band_data_reused": False,
        "interpolation_used_to_create_roots": False,
        "physics": {
            "system": SYSTEM_LABEL,
            "filling_fraction": FILLING_FRACTION,
            "radius_over_a": RADIUS,
            "host": {"rho": HOST.rho, "c_l": HOST.c_l, "c_t": HOST.c_t},
            "inclusion": {"rho": INCLUSION.rho, "c_l": INCLUSION.c_l, "c_t": INCLUSION.c_t},
            "normalization": "omega*a/(2*pi*c_t_host)",
        },
        "numerics": {
            "n_scan": N_SCAN,
            "n_per_segment": N_PER_SEGMENT,
            "m_max": M_MAX,
            "h_primary": H_PRIMARY,
            "h_check": H_CHECK,
            "local_grid_size": LOCAL_GRID_SIZE,
            "continuity_recovery": CONTINUITY_RECOVERY,
            "continuity_match_tolerance": CONTINUITY_MATCH_TOLERANCE,
            "pole_check_h": POLE_CHECK_H,
            "reciprocal_pole_min_support": RECIPROCAL_POLE_MIN_SUPPORT,
        },
        "result": {
            "raw_scan_candidates": sum(len(values) for values in raw_bands),
            "initial_accepted": sum(len(values) for values in accepted_bands),
            "initial_rejected": len(rejected_initial),
            "activation_windows": len(windows),
            "activated_k_indices": len({window["k_index"] for window in windows}),
            "local_candidates": len(local_candidates),
            "recovered_roots": len(recovered),
            "final_roots": len(rows),
            "multiplicity_events": multiplicities,
            "recovered": recovered,
        },
        "timing_s": {
            "global_scan": scan_elapsed,
            "classification_and_activation": filter_elapsed,
            "local_svd_and_validation": recovery_elapsed,
            "total": total_elapsed,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": {
            "bands_csv": bands_csv.name,
            "candidate_diagnostics": candidates_csv.name,
            "activation_windows": windows_json.name,
            "rejected_initial": rejected_json.name,
            "final_image": final_image.name,
            "diagnostic_image": diagnostic_image.name,
        },
    }
    manifest["output_sha256"] = {
        filename: sha256(ROOT / filename) for filename in manifest["outputs"].values()
    }
    manifest_path = ROOT / f"manifest_{OUTPUT_TAG}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"final_image": str(final_image), **manifest["result"], "timing_s": manifest["timing_s"]}, indent=2))


if __name__ == "__main__":
    main()
