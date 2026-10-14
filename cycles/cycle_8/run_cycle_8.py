"""Ciclo 8: certificacion adaptativa de cada raiz antiplana de una cavidad."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import platform
import time
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
from phbands.rootfinding.certification import (
    modal_overlap,
    null_mode_diagnostics,
    prolongated_residual,
)
from phbands.rootfinding.validation import refine_with_expansion
from phbands.scatterer import CavityScatterer
from phbands.secular import SecularProblem


_CYCLE6_PATH = Path(__file__).resolve().parents[1] / "cycle_6" / "run_cycle_6.py"
_SPEC = importlib.util.spec_from_file_location("phbands_cycle6_base", _CYCLE6_PATH)
BASE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(BASE)

ROOT = Path(__file__).resolve().parent
CYCLE_NUMBER = 8
OUTPUT_TAG = "cycle_8"
SYSTEM_LABEL = "cavidad-en-caucho"
A = 1.0
FILLING_FRACTION = 0.7
RADIUS = float(np.sqrt(FILLING_FRACTION / np.pi))
HOST = HostMedium(mu=0.920 * 0.03**2, rho=0.920)
OMEGA_SCALE = 2 * np.pi * HOST.c_t
M_MAX = 2
H_PRIMARY, H_CHECK, H_POLE_CHECK = 8, 12, 16
N_SCAN, N_PER_SEGMENT = 520, 40
OMEGA_BAR_MAX = 1.3
HIGH_SYMMETRY_INDICES = [0, 40, 80, 119]

CERTIFY_M_STEP = 1
CERTIFY_LEVELS = 3
CERTIFY_HALF_WINDOW_BAR = 0.01
CERTIFY_SIGMA_MAX = 1e-4
CERTIFY_RESIDUAL_MAX = 1e-4
CERTIFY_SHIFT_MAX_BAR = 7.5e-4
CERTIFY_REJECT_SHIFT_BAR = 5e-3
CERTIFY_OVERLAP_MIN = 0.90
CERTIFY_REJECT_OVERLAP = 0.50
CERTIFY_EDGE_MAX = 0.08
CERTIFY_PROLONGATED_MAX = 0.15
RECIPROCAL_POLE_GUARD_BAR = 2e-3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sync_base() -> None:
    for name in (
        "ROOT", "CYCLE_NUMBER", "OUTPUT_TAG", "SYSTEM_LABEL", "A",
        "FILLING_FRACTION", "RADIUS", "HOST", "OMEGA_SCALE", "M_MAX",
        "H_PRIMARY", "H_CHECK", "H_POLE_CHECK", "N_SCAN", "N_PER_SEGMENT",
        "OMEGA_BAR_MAX", "HIGH_SYMMETRY_INDICES",
    ):
        setattr(BASE, name, globals()[name])
    BASE.SCATTERER_KIND = "cavity"
    BASE.ADAPTIVE_MULTIPLICITY = True


def _deduplicate(rows: list[dict], tolerance: float = 2e-6) -> list[dict]:
    result = []
    for row in sorted(rows, key=lambda item: (item["k_index"], item["omega_bar"])):
        if result and row["k_index"] == result[-1]["k_index"] and abs(row["omega_bar"] - result[-1]["omega_bar"]) <= tolerance:
            if row["source"] == "local_svd_recovery":
                result[-1] = row
            continue
        result.append(row)
    return result


def _classification(record: dict) -> tuple[str, str]:
    sigmas = record["sigmas"]
    residuals = record["residuals"]
    shifts = record["shifts_bar"]
    overlaps = record["overlaps"]
    if not all(np.isfinite(sigmas + residuals + shifts + overlaps)):
        return "rejected", "nonfinite_numerics"
    tests = {
        # El nivel de escaneo solo localiza la rama. La certificacion se decide
        # por la estabilidad de los dos niveles multipolares mas altos.
        "sigma": max(sigmas[-2:]) <= CERTIFY_SIGMA_MAX,
        "null_residual": max(residuals[-2:]) <= CERTIFY_RESIDUAL_MAX,
        "frequency_shift": shifts[-1] <= CERTIFY_SHIFT_MAX_BAR,
        "modal_overlap": overlaps[-1] >= CERTIFY_OVERLAP_MIN,
        "edge_weight": record["edge_weights"][-1] <= CERTIFY_EDGE_MAX,
        "prolongated_residual": record["prolongated_residuals"][-1] <= CERTIFY_PROLONGATED_MAX,
        "away_from_pole": record["pole_distance_bar"] > RECIPROCAL_POLE_GUARD_BAR,
    }
    failed = [name for name, passed in tests.items() if not passed]
    if not failed:
        return "certified", "all_tests_passed"
    strongly_unstable = (
        sigmas[-1] > 5e-3
        or residuals[-1] > 5e-3
        or max(shifts, default=0.0) > CERTIFY_REJECT_SHIFT_BAR
        or min(overlaps, default=1.0) < CERTIFY_REJECT_OVERLAP
    )
    if record["pole_distance_bar"] <= RECIPROCAL_POLE_GUARD_BAR and strongly_unstable:
        return "rejected", "unstable_at_reciprocal_pole"
    return "uncertain", "+".join(failed)


def certify_candidate(candidate: dict, path, problems: list[SecularProblem]) -> dict:
    index = int(candidate["k_index"])
    k = path.k_points[index]
    frequency = float(candidate["omega_bar"])
    level_data = []
    window_expansions = []
    previous_omega = frequency * OMEGA_SCALE
    initial_half_window = CERTIFY_HALF_WINDOW_BAR * OMEGA_SCALE
    for problem in problems:
        refined, used_window = refine_with_expansion(
            lambda value, p=problem: p.M(value, k), previous_omega, initial_half_window
        )
        matrix = problem.M(refined.omega, k)
        mode = null_mode_diagnostics(matrix)
        level_data.append((refined, mode, matrix))
        window_expansions.append(round(used_window / initial_half_window))
        previous_omega = refined.omega
    frequencies = [item[0].omega / OMEGA_SCALE for item in level_data]
    shifts = [abs(frequencies[i] - frequencies[i - 1]) for i in range(1, len(frequencies))]
    overlaps = [modal_overlap(level_data[i - 1][1].vector, level_data[i][1].vector) for i in range(1, len(level_data))]
    prolonged = [
        prolongated_residual(level_data[i - 1][1].vector, problems[i].M(level_data[i - 1][0].omega, k))
        for i in range(1, len(level_data))
    ]
    pole_distance, pole = BASE.nearest_reciprocal_pole_bar(k, frequencies[-1])
    compact = {
        "sigmas": [item[1].sigma_min for item in level_data],
        "residuals": [item[1].relative_residual for item in level_data],
        "edge_weights": [item[1].edge_weight for item in level_data],
        "shifts_bar": shifts,
        "overlaps": overlaps,
        "prolongated_residuals": prolonged,
        "pole_distance_bar": pole_distance,
    }
    status, reason = _classification(compact)
    record = dict(candidate)
    record.update({
        "omega_bar_input": frequency,
        "omega_bar": frequencies[-1],
        "status": status,
        "reason": reason,
        "pole_bar": pole,
        "pole_distance_bar": pole_distance,
    })
    for level, (problem, data) in enumerate(zip(problems, level_data), 1):
        refined, mode, _ = data
        prefix = f"level_{level}_m{problem.m_max}_h{problem.lattice_sum_config.h_max}"
        record[f"{prefix}_omega_bar"] = refined.omega / OMEGA_SCALE
        record[f"{prefix}_sigma"] = mode.sigma_min
        record[f"{prefix}_residual"] = mode.relative_residual
        record[f"{prefix}_edge_weight"] = mode.edge_weight
        record[f"{prefix}_window_expansions"] = window_expansions[level - 1]
    for level, value in enumerate(shifts, 2):
        record[f"level_{level}_shift_bar"] = value
        record[f"level_{level}_overlap"] = overlaps[level - 2]
        record[f"level_{level}_prolongated_residual"] = prolonged[level - 2]
    return record


def plot_certification(path, records: list[dict], output: Path, diagnostic: bool) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    styles = {
        "certified": dict(color="black", marker="o", s=10, label="Certificada"),
        "uncertain": dict(color="#e69f00", marker="x", s=18, label="Incierta"),
        "rejected": dict(color="#cc3311", marker="+", s=20, label="Rechazada"),
    }
    statuses = ("certified", "uncertain", "rejected") if diagnostic else ("certified",)
    for status in statuses:
        selected = [row for row in records if row["status"] == status]
        if selected:
            ax.scatter(
                [path.distances[row["k_index"]] for row in selected],
                [row["omega_bar"] for row in selected],
                **styles[status],
            )
    for position in path.tick_positions:
        ax.axvline(position, color="0.72", linewidth=0.6, zorder=0)
    ax.set_xticks(path.tick_positions, path.tick_labels)
    ax.set_xlim(path.tick_positions[0], path.tick_positions[-1])
    ax.set_ylim(0, OMEGA_BAR_MAX)
    ax.set_ylabel(r"$\omega a/(2\pi c_{t,\mathrm{host}})$")
    ax.set_title(f"Ciclo 8 - {SYSTEM_LABEL} cavidad antiplana")
    if diagnostic:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    _sync_base()
    total_started = time.perf_counter()
    path = build_path([("M", 1, 1), ("Gamma", 0, 0), ("X", 1, 0), ("M", 1, 1)], A, N_PER_SEGMENT)
    scatterer = CavityScatterer(RADIUS, HOST)
    scan_problem = SecularProblem(SquareLattice(A), scatterer, M_MAX, LatticeSumConfig(H_PRIMARY))

    started = time.perf_counter()
    scan = compute_band_structure(scan_problem, path, 1e-6, OMEGA_BAR_MAX * OMEGA_SCALE, RootFindingConfig(n_scan=N_SCAN), verbose=True)
    scan_elapsed = time.perf_counter() - started
    raw, accepted, rejected_initial = BASE.classify_initial_candidates(scan, path)
    windows, multiplicities = BASE.activation_windows(accepted, path, scan_problem)
    check = SecularProblem(SquareLattice(A), scatterer, M_MAX, LatticeSumConfig(H_CHECK))
    pole_check = SecularProblem(SquareLattice(A), scatterer, M_MAX, LatticeSumConfig(H_POLE_CHECK))
    local_candidates, recovered = BASE.search_windows(windows, accepted, path, scan_problem, check, pole_check)

    candidates = []
    for index, values in enumerate(accepted):
        for value in values:
            candidates.append({"source": "determinant_scan", "k_index": index, "omega_bar": value})
    for item in recovered:
        candidates.append({"source": "local_svd_recovery", "k_index": item["k_index"], "omega_bar": item["omega_bar"]})
    candidates = _deduplicate(candidates)

    levels = [M_MAX + CERTIFY_M_STEP * index for index in range(CERTIFY_LEVELS)]
    h_levels = [H_PRIMARY, H_CHECK, H_POLE_CHECK]
    problems = [SecularProblem(SquareLattice(A), scatterer, m, LatticeSumConfig(h)) for m, h in zip(levels, h_levels)]
    certified = []
    started = time.perf_counter()
    for number, candidate in enumerate(candidates, 1):
        certified.append(certify_candidate(candidate, path, problems))
        if number % 25 == 0 or number == len(candidates):
            print(f"Certificacion [{number}/{len(candidates)}]")
    certification_elapsed = time.perf_counter() - started

    counts = {status: sum(row["status"] == status for row in certified) for status in ("certified", "uncertain", "rejected")}
    outputs = {
        "certification_csv": f"certification_{OUTPUT_TAG}.csv",
        "certified_bands_csv": f"bands_certified_{OUTPUT_TAG}.csv",
        "final_image": f"bands_certified_{OUTPUT_TAG}.png",
        "diagnostic_image": f"bands_certification_diagnostic_{OUTPUT_TAG}.png",
        "rejected_initial": f"rejected_initial_candidates_{OUTPUT_TAG}.json",
    }
    write_csv(ROOT / outputs["certification_csv"], certified)
    write_csv(ROOT / outputs["certified_bands_csv"], [row for row in certified if row["status"] == "certified"])
    (ROOT / outputs["rejected_initial"]).write_text(json.dumps(rejected_initial, indent=2), encoding="utf-8")
    plot_certification(path, certified, ROOT / outputs["final_image"], diagnostic=False)
    plot_certification(path, certified, ROOT / outputs["diagnostic_image"], diagnostic=True)
    manifest = {
        "cycle": CYCLE_NUMBER,
        "purpose": "Certificacion por raiz mediante SVD, convergencia multipolar, estabilidad modal y polos",
        "physics": {"system": SYSTEM_LABEL, "polarization": "antiplane_z", "scatterer_kind": "cavity", "filling_fraction": FILLING_FRACTION, "radius_over_a": RADIUS, "host": {"rho": HOST.rho, "mu": HOST.mu, "c_t": HOST.c_t}, "normalization": "omega*a/(2*pi*c_t_host)"},
        "numerics": {"n_scan": N_SCAN, "n_per_segment": N_PER_SEGMENT, "scan_m_max": M_MAX, "certification_m_levels": levels, "certification_h_levels": h_levels, "thresholds": {"sigma_max": CERTIFY_SIGMA_MAX, "residual_max": CERTIFY_RESIDUAL_MAX, "shift_max_bar": CERTIFY_SHIFT_MAX_BAR, "overlap_min": CERTIFY_OVERLAP_MIN, "edge_weight_max": CERTIFY_EDGE_MAX, "prolongated_residual_max": CERTIFY_PROLONGATED_MAX, "pole_guard_bar": RECIPROCAL_POLE_GUARD_BAR}},
        "result": {"raw_scan_candidates": sum(map(len, raw)), "recovered_candidates": len(recovered), "deduplicated_candidates": len(candidates), **counts, "multiplicity_events": multiplicities},
        "timing_s": {"global_scan": scan_elapsed, "certification": certification_elapsed, "total": time.perf_counter() - total_started},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "outputs": outputs,
    }
    manifest["output_sha256"] = {name: sha256(ROOT / name) for name in outputs.values()}
    (ROOT / f"manifest_{OUTPUT_TAG}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"final_image": str(ROOT / outputs["final_image"]), "diagnostic_image": str(ROOT / outputs["diagnostic_image"]), **manifest["result"], "timing_s": manifest["timing_s"]}, indent=2))


if __name__ == "__main__":
    main()
