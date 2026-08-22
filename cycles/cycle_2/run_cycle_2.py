from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
CYCLE_1 = ROOT.parent / "cycle_1"
sys.path.insert(0, str(REPOSITORY / "src"))

from phbands.inplane.media import InPlaneMedium
from phbands.inplane.secular import InPlaneSecularProblem
from phbands.lattice import SquareLattice, build_path
from phbands.lattice_sums import LatticeSumConfig
from phbands.rootfinding.validation import refine_matrix_candidate


A = 1.0
M_MAX = 2
H_PRIMARY = 8
H_CHECK = 12
GRID_SIZES = [51, 101, 201, 401]
DET_ACCEPT_RATIO = 1e-3
SINGULAR_RATIO_CUTOFF = 1e-2
SIGMA_PRIMARY_MAX = 1e-4
SIGMA_CHECK_MAX = 5e-3
MAX_SHIFT_BAR = 2e-3
MATCH_EXISTING_BAR = 3e-3
RECIPROCAL_POLE_GUARD_BAR = 1.5e-3

HOST = InPlaneMedium(1.18, 2.54, 2.54 / 2.19)
INCLUSION = InPlaneMedium(11.4, 2.16, 2.16 / 2.51)
RADIUS = float(np.sqrt(0.30 / np.pi))
OMEGA_SCALE = 2 * np.pi * HOST.c_t


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_minima(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    return np.where(
        finite[1:-1]
        & finite[:-2]
        & finite[2:]
        & (values[1:-1] <= values[:-2])
        & (values[1:-1] <= values[2:])
    )[0] + 1


def matrix_curves(problem, k, omega_grid):
    determinants = []
    singular_ratios = []
    for omega in omega_grid:
        matrix = problem.M(omega, k)
        singular = np.linalg.svd(matrix, compute_uv=False)
        determinants.append(abs(np.linalg.det(matrix)) ** 2)
        singular_ratios.append(singular[-1] / singular[0])
    return np.asarray(determinants), np.asarray(singular_ratios)


def refine_determinant(problem, k, omega, half_window):
    lower = max(1e-8, omega - half_window)
    upper = omega + half_window
    result = minimize_scalar(
        lambda value: abs(np.linalg.det(problem.M(value, k))) ** 2,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-9},
    )
    return float(result.x)


def deduplicate(values, tolerance):
    result = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
    return result


def nearest_reciprocal_pole_bar(k: np.ndarray, omega_bar: float) -> tuple[float, str, float]:
    poles = []
    reciprocal_step = 2 * np.pi / A
    for hx in range(-H_CHECK, H_CHECK + 1):
        for hy in range(-H_CHECK, H_CHECK + 1):
            q = float(np.hypot(reciprocal_step * hx + k[0], reciprocal_step * hy + k[1]))
            poles.append((q / (2 * np.pi), "t"))
            poles.append(((HOST.c_l / HOST.c_t) * q / (2 * np.pi), "l"))
    pole_bar, channel = min(poles, key=lambda item: abs(item[0] - omega_bar))
    return abs(pole_bar - omega_bar), channel, pole_bar


def main() -> None:
    path = build_path(
        [("M", 1.0, 1.0), ("Gamma", 0.0, 0.0), ("X", 1.0, 0.0), ("M", 1.0, 1.0)],
        A,
        40,
    )
    windows = json.loads((ROOT / "detected_gap_windows.json").read_text(encoding="utf-8"))
    with (CYCLE_1 / "candidate_validation_cycle_1.csv").open(newline="", encoding="utf-8") as stream:
        accepted_cycle_1 = [row for row in csv.DictReader(stream) if row["accepted"] == "True"]
    existing_by_k: dict[int, list[float]] = {}
    for row in accepted_cycle_1:
        existing_by_k.setdefault(int(row["k_index"]), []).append(float(row["omega_bar_refined"]))

    primary = InPlaneSecularProblem(
        SquareLattice(A), HOST, INCLUSION, RADIUS, M_MAX, LatticeSumConfig(H_PRIMARY)
    )
    check = InPlaneSecularProblem(
        SquareLattice(A), HOST, INCLUSION, RADIUS, M_MAX, LatticeSumConfig(H_CHECK)
    )
    started = time.perf_counter()
    diagnostics = []
    candidates = []
    plot_data = []

    for window in windows:
        index = int(window["k_index"])
        k = path.k_points[index]
        lower_bar = max(window["search_lower_bar"], 1e-6)
        upper_bar = window["search_upper_bar"]
        dense_bar = np.linspace(lower_bar, upper_bar, GRID_SIZES[-1])
        dense_omega = dense_bar * OMEGA_SCALE
        det_dense, ratio_dense = matrix_curves(primary, k, dense_omega)
        plot_data.append((index, dense_bar, det_dense, ratio_dense))

        resolution_summary = {}
        for size in GRID_SIZES:
            sample_indices = np.rint(np.linspace(0, GRID_SIZES[-1] - 1, size)).astype(int)
            det_values = det_dense[sample_indices]
            ratio_values = ratio_dense[sample_indices]
            det_scale = np.median(det_values[np.isfinite(det_values)])
            det_minima = [
                int(sample_indices[position])
                for position in local_minima(det_values)
                if det_values[position] / max(det_scale, 1e-300) < DET_ACCEPT_RATIO
            ]
            svd_minima = [
                int(sample_indices[position])
                for position in local_minima(ratio_values)
                if ratio_values[position] < SINGULAR_RATIO_CUTOFF
            ]
            resolution_summary[str(size)] = {
                "det_minima_bar": [float(dense_bar[position]) for position in det_minima],
                "svd_minima_bar": [float(dense_bar[position]) for position in svd_minima],
            }

        det_scale = np.median(det_dense[np.isfinite(det_dense)])
        seed_indices = {
            int(position)
            for position in local_minima(det_dense)
            if det_dense[position] / max(det_scale, 1e-300) < DET_ACCEPT_RATIO
        }
        seed_indices.update(
            int(position)
            for position in local_minima(ratio_dense)
            if ratio_dense[position] < SINGULAR_RATIO_CUTOFF
        )
        grid_step = float(dense_omega[1] - dense_omega[0])
        refined_seeds = []
        for position in sorted(seed_indices):
            omega_seed = float(dense_omega[position])
            refined_seeds.append(refine_determinant(primary, k, omega_seed, 1.1 * grid_step))
            refined_seeds.append(
                refine_matrix_candidate(lambda value: primary.M(value, k), omega_seed, 1.1 * grid_step).omega
            )
        refined_seeds = deduplicate(refined_seeds, tolerance=0.25 * grid_step)

        for omega_seed in refined_seeds:
            first = refine_matrix_candidate(lambda value: primary.M(value, k), omega_seed, 1.5 * grid_step)
            second = refine_matrix_candidate(lambda value: check.M(value, k), first.omega, 1.5 * grid_step)
            omega_bar = second.omega / OMEGA_SCALE
            shift_bar = abs(second.omega - first.omega) / OMEGA_SCALE
            already_present = any(
                abs(omega_bar - existing) <= MATCH_EXISTING_BAR for existing in existing_by_k.get(index, [])
            )
            pole_distance, pole_channel, pole_bar = nearest_reciprocal_pole_bar(k, omega_bar)
            reciprocal_pole_suspect = pole_distance <= RECIPROCAL_POLE_GUARD_BAR
            converged = bool(
                first.success
                and second.success
                and first.balanced_sigma_min <= SIGMA_PRIMARY_MAX
                and second.balanced_sigma_min <= SIGMA_CHECK_MAX
                and shift_bar <= MAX_SHIFT_BAR
                and not reciprocal_pole_suspect
            )
            candidate = {
                "k_index": index,
                "Kx": float(k[0]),
                "Ky": float(k[1]),
                "omega_bar": omega_bar,
                "sigma_balanced_h8": first.balanced_sigma_min,
                "sigma_balanced_h12": second.balanced_sigma_min,
                "shift_h8_h12_bar": shift_bar,
                "nearest_reciprocal_pole_distance_bar": pole_distance,
                "nearest_reciprocal_pole_channel": pole_channel,
                "nearest_reciprocal_pole_bar": pole_bar,
                "reciprocal_pole_suspect": reciprocal_pole_suspect,
                "already_present_cycle_1": already_present,
                "converged": converged,
                "new_recovered_root": bool(converged and not already_present),
            }
            candidates.append(candidate)

        diagnostics.append(
            {
                **window,
                "K": [float(k[0]), float(k[1])],
                "resolution_summary": resolution_summary,
                "dense_seed_count": len(seed_indices),
                "refined_seed_count": len(refined_seeds),
                "is_gamma_low_frequency_window": bool(np.linalg.norm(k) < 1e-12 and lower_bar < 1e-4),
            }
        )

    elapsed = time.perf_counter() - started
    recovered = [candidate for candidate in candidates if candidate["new_recovered_root"]]

    candidate_csv = ROOT / "candidate_diagnostics_cycle_2.csv"
    with candidate_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    diagnostic_json = ROOT / "resolution_diagnostics_cycle_2.json"
    diagnostic_json.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(len(plot_data), 2, figsize=(10, 3.0 * len(plot_data)), squeeze=False)
    for row, (index, frequency, determinant, singular_ratio) in enumerate(plot_data):
        axes[row, 0].plot(frequency, np.log10(np.maximum(determinant, 1e-300)), color="black", linewidth=0.8)
        axes[row, 0].set_ylabel(f"k[{index}]\nlog10 |det M|^2")
        axes[row, 1].plot(frequency, np.log10(np.maximum(singular_ratio, 1e-300)), color="tab:blue", linewidth=0.8)
        axes[row, 1].set_ylabel("log10 sigma_min/sigma_max")
        for candidate in candidates:
            if candidate["k_index"] == index and candidate["new_recovered_root"]:
                axes[row, 0].axvline(candidate["omega_bar"], color="red", linewidth=0.8)
                axes[row, 1].axvline(candidate["omega_bar"], color="red", linewidth=0.8)
        axes[row, 0].grid(alpha=0.2)
        axes[row, 1].grid(alpha=0.2)
    axes[-1, 0].set_xlabel(r"$\omega a/(2\pi c_t)$")
    axes[-1, 1].set_xlabel(r"$\omega a/(2\pi c_t)$")
    fig.suptitle("Ciclo 2 — diagnóstico directo de ventanas con pérdidas")
    fig.tight_layout()
    diagnostic_image = ROOT / "diagnostics_cycle_2.png"
    fig.savefig(diagnostic_image, dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(
        [float(row["path_distance"]) for row in accepted_cycle_1],
        [float(row["omega_bar_refined"]) for row in accepted_cycle_1],
        s=5,
        color="black",
        label="Ciclo 1",
    )
    if recovered:
        ax.scatter(
            [path.distances[item["k_index"]] for item in recovered],
            [item["omega_bar"] for item in recovered],
            s=18,
            color="red",
            marker="x",
            label="Raiz recuperada por barrido directo",
        )
    for position in path.tick_positions:
        ax.axvline(position, color="0.72", linewidth=0.6, zorder=0)
    ax.set_xticks(path.tick_positions)
    ax.set_xticklabels(path.tick_labels)
    ax.set_xlim(path.tick_positions[0], path.tick_positions[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(r"$\omega a/(2\pi c_{t,\mathrm{epoxy}})$")
    ax.set_title("Ciclo 2 — raíces obtenidas directamente")
    if recovered:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    band_image = ROOT / "bands_cycle_2.png"
    fig.savefig(band_image, dpi=180)
    plt.close(fig)

    manifest = {
        "cycle": 2,
        "purpose": "Determinar por que se pierden puntos mediante barridos directos de resolucion creciente",
        "interpolation_used_to_create_roots": False,
        "grid_sizes": GRID_SIZES,
        "reciprocal_pole_guard_bar": RECIPROCAL_POLE_GUARD_BAR,
        "windows": len(windows),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "result": {
            "elapsed_s": elapsed,
            "refined_candidates": len(candidates),
            "new_converged_roots": len(recovered),
            "new_roots": recovered,
        },
        "outputs": {
            "candidate_csv": candidate_csv.name,
            "diagnostics_json": diagnostic_json.name,
            "diagnostic_image": diagnostic_image.name,
            "band_image": band_image.name,
        },
    }
    manifest["output_sha256"] = {
        candidate_csv.name: sha256(candidate_csv),
        diagnostic_json.name: sha256(diagnostic_json),
        diagnostic_image.name: sha256(diagnostic_image),
        band_image.name: sha256(band_image),
    }
    manifest_path = ROOT / "manifest_cycle_2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"band_image": str(band_image), **manifest["result"]}, indent=2))


if __name__ == "__main__":
    main()
