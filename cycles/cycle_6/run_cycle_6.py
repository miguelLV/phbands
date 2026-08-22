"""Ciclo 6: bandas antiplanas (modo z) de cilindros de Au en Be."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from phbands.bandstructure import compute_band_structure
from phbands.lattice import SquareLattice, build_path
from phbands.lattice_sums import LatticeSumConfig
from phbands.media import HostMedium
from phbands.rootfinding import RootFindingConfig
from phbands.rootfinding.gap_detection import detect_bridge_gaps, detect_continuity_mismatches
from phbands.rootfinding.validation import equilibrate_matrix, refine_matrix_candidate
from phbands.scatterer import CavityScatterer, SolidCylinderScatterer
from phbands.secular import SecularProblem


ROOT = Path(__file__).resolve().parent
CYCLE_NUMBER = 6
OUTPUT_TAG = "cycle_6"
SYSTEM_LABEL = "Au/Be"
HOST_LABEL = "Be"
SCATTERER_KIND = "solid"
A = 1.0
FILLING_FRACTION = 0.0908
RADIUS = float(np.sqrt(FILLING_FRACTION / np.pi))

RHO_H, C_T_H = 1.85, 13.11 / 1.44
RHO_I, C_T_I = 19.5, 3.36 / 2.71
HOST = HostMedium(mu=RHO_H * C_T_H**2, rho=RHO_H)
INCLUSION = HostMedium(mu=RHO_I * C_T_I**2, rho=RHO_I)
OMEGA_SCALE = 2 * np.pi * HOST.c_t

M_MAX = 2
H_PRIMARY, H_CHECK, H_POLE_CHECK = 8, 12, 16
N_SCAN, N_PER_SEGMENT = 200, 40
OMEGA_BAR_MAX = 0.5
LOCAL_GRID_SIZE, LOCAL_HALF_WIDTH_BAR = 101, 0.02
SINGULAR_RATIO_CUTOFF = 2e-2
SIGMA_PRIMARY_MAX, SIGMA_CHECK_MAX = 1e-4, 5e-3
MAX_SHIFT_BAR, MATCH_EXISTING_BAR = 5e-4, 2e-4
RECIPROCAL_POLE_GUARD_BAR = 2e-3
RECIPROCAL_POLE_MIN_SUPPORT = 3
SUPPORT_DELTA_PER_STEP = 0.025
CONTINUITY_MATCH_TOLERANCE = 0.010
HIGH_SYMMETRY_INDICES = [0, 40, 80, 119]
MULTIPLICITY_THRESHOLD, MULTIPLICITY_RADIUS = 1e-3, 3
ADAPTIVE_MULTIPLICITY = False
MULTIPLICITY_BAND_RADIUS = 0.04
MULTIPLICITY_RESOLVED_STREAK = 2
MULTIPLICITY_WINDOW_HALF_WIDTH = 0.03
MULTIPLICITY_LOCAL_GRID_SIZE = 401
MULTIPLICITY_ULTRAFINE_GRID_SIZE = 1601
MULTIPLICITY_ULTRAFINE_K_RADIUS = 2
MULTIPLICITY_ULTRAFINE_HALF_WIDTH = 0.006
MULTIPLICITY_MATCH_EXISTING_BAR = 2e-6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def nearest_reciprocal_pole_bar(k: np.ndarray, omega_bar: float):
    reciprocal_step = 2 * np.pi / A
    poles = [
        float(np.hypot(reciprocal_step * hx + k[0], reciprocal_step * hy + k[1])) / (2 * np.pi)
        for hx in range(-H_POLE_CHECK, H_POLE_CHECK + 1)
        for hy in range(-H_POLE_CHECK, H_POLE_CHECK + 1)
    ]
    pole = min(poles, key=lambda value: abs(value - omega_bar))
    return abs(pole - omega_bar), pole


def branch_support(index, frequency, bands):
    supported = {index}
    for neighbor in range(max(0, index - 2), min(len(bands), index + 3)):
        if neighbor == index:
            continue
        tolerance = SUPPORT_DELTA_PER_STEP * abs(neighbor - index)
        if any(abs(value - frequency) <= tolerance for value in bands[neighbor]):
            supported.add(neighbor)
    return len(supported)


def classify_initial_candidates(result, path):
    raw = [[omega / OMEGA_SCALE for omega in roots] for roots in result.bands]
    accepted = [[] for _ in raw]
    rejected = []
    for index, values in enumerate(raw):
        for frequency in values:
            if frequency <= 1e-2 and np.linalg.norm(path.k_points[index]) > 0.2:
                rejected.append({"k_index": index, "omega_bar": frequency, "reason": "low_frequency_away_from_gamma"})
            else:
                accepted[index].append(frequency)
        accepted[index].sort()
    return raw, accepted, rejected


def activation_windows(bands, path, problem):
    raw = []
    triggers = list(detect_bridge_gaps(bands)) + list(
        detect_continuity_mismatches(bands, match_tolerance=CONTINUITY_MATCH_TOLERANCE)
    )
    for trigger in triggers:
        raw.append(
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
            matrix = problem.M(frequency * OMEGA_SCALE, k)
            singular = np.linalg.svd(
                matrix if ADAPTIVE_MULTIPLICITY else equilibrate_matrix(matrix),
                compute_uv=False,
            )
            if len(singular) < 2:
                continue
            ratio = float(singular[-2] / singular[0])
            if ratio > MULTIPLICITY_THRESHOLD:
                continue
            multiplicities.append({"k_index": index, "omega_bar": frequency, "second_singular_ratio": ratio})
            if ADAPTIVE_MULTIPLICITY:
                directions = [1] if index == 0 else [-1] if index == len(path) - 1 else [-1, 1]
                neighbors = []
                for direction in directions:
                    resolved_streak = 0
                    neighbor = index + direction
                    while 0 <= neighbor < len(path):
                        neighbors.append(neighbor)
                        nearby = sum(abs(value - frequency) <= MULTIPLICITY_BAND_RADIUS for value in bands[neighbor])
                        resolved_streak = resolved_streak + 1 if nearby >= 2 else 0
                        if resolved_streak >= MULTIPLICITY_RESOLVED_STREAK:
                            break
                        neighbor += direction
            else:
                offsets = range(1, MULTIPLICITY_RADIUS + 1)
                neighbors = (
                    [index + offset for offset in offsets] if index == 0
                    else [index - offset for offset in offsets] if index == len(path) - 1
                    else [neighbor for offset in offsets for neighbor in (index - offset, index + offset)]
                )
            for neighbor in neighbors:
                if 0 <= neighbor < len(path):
                    if ADAPTIVE_MULTIPLICITY and abs(neighbor - index) <= MULTIPLICITY_ULTRAFINE_K_RADIUS:
                        half_width = MULTIPLICITY_ULTRAFINE_HALF_WIDTH
                    else:
                        half_width = MULTIPLICITY_WINDOW_HALF_WIDTH if ADAPTIVE_MULTIPLICITY else LOCAL_HALF_WIDTH_BAR
                    raw.append(
                        {
                            "k_index": neighbor,
                            "lower_bar": max(1e-6, frequency - half_width),
                            "upper_bar": min(OMEGA_BAR_MAX, frequency + half_width),
                            "reasons": ["high_symmetry_multiplicity"],
                        }
                    )
    return merge_windows(raw), multiplicities


def search_windows(windows, bands, path, primary, check, pole_check):
    candidates = []
    for number, window in enumerate(windows, 1):
        index, k = window["k_index"], path.k_points[window["k_index"]]
        multiplicity_window = ADAPTIVE_MULTIPLICITY and "high_symmetry_multiplicity" in window["reasons"]
        near_high_symmetry = any(
            abs(index - symmetry_index) <= MULTIPLICITY_ULTRAFINE_K_RADIUS
            for symmetry_index in HIGH_SYMMETRY_INDICES
        )
        grid_size = (
            MULTIPLICITY_ULTRAFINE_GRID_SIZE
            if multiplicity_window and near_high_symmetry
            else MULTIPLICITY_LOCAL_GRID_SIZE
            if multiplicity_window
            else LOCAL_GRID_SIZE
        )
        grid_bar = np.linspace(window["lower_bar"], window["upper_bar"], grid_size)
        grid = grid_bar * OMEGA_SCALE
        ratios = np.asarray([
            (lambda s: float(s[-1] / s[0]))(np.linalg.svd(primary.M(omega, k), compute_uv=False))
            for omega in grid
        ])
        grid_step = float(grid[1] - grid[0])
        for position in local_minima(ratios):
            if ratios[position] > SINGULAR_RATIO_CUTOFF:
                continue
            first = refine_matrix_candidate(lambda value: primary.M(value, k), float(grid[position]), 1.5 * grid_step)
            second = refine_matrix_candidate(lambda value: check.M(value, k), first.omega, 1.5 * grid_step)
            frequency = second.omega / OMEGA_SCALE
            shift = abs(second.omega - first.omega) / OMEGA_SCALE
            pole_distance, pole_bar = nearest_reciprocal_pole_bar(k, frequency)
            support = branch_support(index, frequency, bands)
            pole_sigma = pole_shift = None
            pole_pass = pole_distance > RECIPROCAL_POLE_GUARD_BAR
            if not pole_pass:
                third = refine_matrix_candidate(lambda value: pole_check.M(value, k), second.omega, 1.5 * grid_step)
                pole_sigma = third.balanced_sigma_min
                pole_shift = abs(third.omega - second.omega) / OMEGA_SCALE
                frequency = third.omega / OMEGA_SCALE
                pole_distance, pole_bar = nearest_reciprocal_pole_bar(k, frequency)
                pole_pass = bool(
                    third.balanced_sigma_min <= SIGMA_CHECK_MAX
                    and pole_shift <= MAX_SHIFT_BAR
                    and support >= RECIPROCAL_POLE_MIN_SUPPORT
                )
            match_tolerance = MULTIPLICITY_MATCH_EXISTING_BAR if multiplicity_window else MATCH_EXISTING_BAR
            existing = any(abs(frequency - value) <= match_tolerance for value in bands[index])
            converged = bool(
                first.balanced_sigma_min <= SIGMA_PRIMARY_MAX
                and second.balanced_sigma_min <= SIGMA_CHECK_MAX
                and shift <= MAX_SHIFT_BAR
                and pole_pass
            )
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
                    "nearest_reciprocal_pole_bar": pole_bar,
                    "reciprocal_pole_support": support,
                    "pole_check_sigma_h16": pole_sigma,
                    "pole_check_shift_bar": pole_shift,
                    "pole_convergence_pass": pole_pass,
                    "existing": existing,
                    "match_existing_bar": match_tolerance,
                    "converged_preliminary": converged,
                    "window_reasons": "+".join(window["reasons"]),
                }
            )
        if number % 10 == 0 or number == len(windows):
            print(f"SVD local [{number}/{len(windows)}]")
    recovered = []
    for candidate in sorted(candidates, key=lambda item: (item["k_index"], item["omega_bar"])):
        accepted = candidate["converged_preliminary"] and not candidate["existing"]
        candidate["accepted"] = accepted
        if not accepted:
            continue
        match_tolerance = (
            MULTIPLICITY_MATCH_EXISTING_BAR
            if ADAPTIVE_MULTIPLICITY and "high_symmetry_multiplicity" in candidate["window_reasons"]
            else MATCH_EXISTING_BAR
        )
        if any(
            old["k_index"] == candidate["k_index"]
            and abs(old["omega_bar"] - candidate["omega_bar"]) <= match_tolerance
            for old in recovered
        ):
            continue
        recovered.append(candidate)
    return candidates, recovered


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_bands(path, bands, recovered, output, diagnostic=False):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for index, values in enumerate(bands):
        ax.scatter([path.distances[index]] * len(values), values, s=8, facecolors="none", edgecolors="black", linewidths=0.7)
    if recovered:
        ax.scatter(
            [path.distances[item["k_index"]] for item in recovered],
            [item["omega_bar"] for item in recovered],
            s=22 if diagnostic else 8,
            marker="x" if diagnostic else "o",
            facecolors="none" if not diagnostic else None,
            edgecolors="black" if not diagnostic else None,
            color="red" if diagnostic else None,
            label="Recuperada por SVD" if diagnostic else None,
        )
    for position in path.tick_positions:
        ax.axvline(position, color="0.72", linewidth=0.6, zorder=0)
    ax.set_xticks(path.tick_positions, path.tick_labels)
    ax.set_xlim(path.tick_positions[0], path.tick_positions[-1])
    ax.set_ylim(0, OMEGA_BAR_MAX)
    ax.set_ylabel(r"$\omega a/(2\pi c_{t,\mathrm{host}})$")
    ax.set_title(f"Ciclo {CYCLE_NUMBER} - {SYSTEM_LABEL} antiplano (modo z)")
    if diagnostic and recovered:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    total_started = time.perf_counter()
    path = build_path([("M", 1, 1), ("Gamma", 0, 0), ("X", 1, 0), ("M", 1, 1)], A, N_PER_SEGMENT)
    scatterer = (
        CavityScatterer(RADIUS, HOST)
        if SCATTERER_KIND == "cavity"
        else SolidCylinderScatterer(RADIUS, HOST, INCLUSION)
    )
    primary = SecularProblem(SquareLattice(A), scatterer, M_MAX, LatticeSumConfig(H_PRIMARY))
    check = SecularProblem(SquareLattice(A), scatterer, M_MAX, LatticeSumConfig(H_CHECK))
    pole_check = SecularProblem(SquareLattice(A), scatterer, M_MAX, LatticeSumConfig(H_POLE_CHECK))

    started = time.perf_counter()
    scan = compute_band_structure(primary, path, 1e-6, OMEGA_BAR_MAX * OMEGA_SCALE, RootFindingConfig(n_scan=N_SCAN), verbose=True)
    scan_elapsed = time.perf_counter() - started
    started = time.perf_counter()
    raw, accepted, rejected = classify_initial_candidates(scan, path)
    windows, multiplicities = activation_windows(accepted, path, primary)
    activation_elapsed = time.perf_counter() - started
    started = time.perf_counter()
    candidates, recovered = search_windows(windows, accepted, path, primary, check, pole_check)
    recovery_elapsed = time.perf_counter() - started

    rows = []
    for index, values in enumerate(accepted):
        for frequency in values:
            rows.append({"source": "determinant_scan", "k_index": index, "Kx": float(path.k_points[index, 0]), "Ky": float(path.k_points[index, 1]), "path_distance": float(path.distances[index]), "omega_bar": frequency, "omega": frequency * OMEGA_SCALE})
    for candidate in recovered:
        rows.append({"source": "local_svd_recovery", "k_index": candidate["k_index"], "Kx": candidate["Kx"], "Ky": candidate["Ky"], "path_distance": float(path.distances[candidate["k_index"]]), "omega_bar": candidate["omega_bar"], "omega": candidate["omega_bar"] * OMEGA_SCALE})
    rows.sort(key=lambda row: (row["k_index"], row["omega_bar"]))

    outputs = {
        "bands_csv": f"bands_{OUTPUT_TAG}.csv",
        "candidate_diagnostics": f"local_candidate_diagnostics_{OUTPUT_TAG}.csv",
        "activation_windows": f"activation_windows_{OUTPUT_TAG}.json",
        "rejected_initial": f"rejected_initial_candidates_{OUTPUT_TAG}.json",
        "final_image": f"bands_{OUTPUT_TAG}.png",
        "diagnostic_image": f"bands_{OUTPUT_TAG}_sources.png",
    }
    write_csv(ROOT / outputs["bands_csv"], rows)
    write_csv(ROOT / outputs["candidate_diagnostics"], candidates)
    (ROOT / outputs["activation_windows"]).write_text(json.dumps(windows, indent=2), encoding="utf-8")
    (ROOT / outputs["rejected_initial"]).write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    plot_bands(path, accepted, recovered, ROOT / outputs["final_image"])
    plot_bands(path, accepted, recovered, ROOT / outputs["diagnostic_image"], diagnostic=True)

    manifest = {
        "cycle": CYCLE_NUMBER,
        "purpose": (
            "Corrida antiplana desde cero con T_n de cavidad y rescate SVD automatico"
            if SCATTERER_KIND == "cavity"
            else "Corrida Au/Be antiplana desde cero con T_n de cilindro solido y rescate SVD automatico"
        ),
        "input_band_data_reused": False,
        "interpolation_used_to_create_roots": False,
        "physics": {"system": SYSTEM_LABEL, "polarization": "antiplane_z", "scatterer_kind": SCATTERER_KIND, "filling_fraction": FILLING_FRACTION, "radius_over_a": RADIUS, "host": {"rho": HOST.rho, "mu": HOST.mu, "c_t": HOST.c_t}, "inclusion": {"type": "cavity"} if SCATTERER_KIND == "cavity" else {"type": "solid", "rho": INCLUSION.rho, "mu": INCLUSION.mu, "c_t": INCLUSION.c_t}, "normalization": "omega*a/(2*pi*c_t_host)"},
        "numerics": {"n_scan": N_SCAN, "n_per_segment": N_PER_SEGMENT, "m_max": M_MAX, "h_primary": H_PRIMARY, "h_check": H_CHECK, "pole_check_h": H_POLE_CHECK, "local_grid_size": LOCAL_GRID_SIZE, "continuity_recovery": True, "adaptive_multiplicity": ADAPTIVE_MULTIPLICITY, "multiplicity_local_grid_size": MULTIPLICITY_LOCAL_GRID_SIZE if ADAPTIVE_MULTIPLICITY else None, "multiplicity_ultrafine_grid_size": MULTIPLICITY_ULTRAFINE_GRID_SIZE if ADAPTIVE_MULTIPLICITY else None, "multiplicity_ultrafine_half_width": MULTIPLICITY_ULTRAFINE_HALF_WIDTH if ADAPTIVE_MULTIPLICITY else None, "multiplicity_match_existing_bar": MULTIPLICITY_MATCH_EXISTING_BAR if ADAPTIVE_MULTIPLICITY else None, "multiplicity_resolved_streak": MULTIPLICITY_RESOLVED_STREAK if ADAPTIVE_MULTIPLICITY else None},
        "result": {"raw_scan_candidates": sum(map(len, raw)), "initial_accepted": sum(map(len, accepted)), "initial_rejected": len(rejected), "activation_windows": len(windows), "activated_k_indices": len({window["k_index"] for window in windows}), "local_candidates": len(candidates), "recovered_roots": len(recovered), "final_roots": len(rows), "multiplicity_events": multiplicities, "recovered": recovered},
        "timing_s": {"global_scan": scan_elapsed, "classification_and_activation": activation_elapsed, "local_svd_and_validation": recovery_elapsed, "total": time.perf_counter() - total_started},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "outputs": outputs,
    }
    manifest["output_sha256"] = {name: sha256(ROOT / name) for name in outputs.values()}
    (ROOT / f"manifest_{OUTPUT_TAG}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"final_image": str(ROOT / outputs["final_image"]), **manifest["result"], "timing_s": manifest["timing_s"]}, indent=2))


if __name__ == "__main__":
    main()
