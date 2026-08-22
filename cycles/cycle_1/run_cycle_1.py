from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import jn_zeros


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
CYCLE_0B = ROOT.parent / "cycle_0b"
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
N_SCAN = 200
N_PER_SEGMENT = 40
OMEGA_BAR_MAX = 1.0
SIGMA_PRIMARY_MAX = 1e-4
SIGMA_CHECK_MAX = 5e-3
MAX_SHIFT_BAR = 2e-3
CONTINUITY_RADIUS = 2
MAX_NEIGHBOR_DELTA_BAR = 3.5e-2
BESSEL_POLE_GUARD = 1e-2
POLE_SUPPORT_RADIUS = 4
MIN_POLE_SUPPORT = 5
POLE_SUPPORT_DELTA_BAR = 1.5e-2
LOW_FREQUENCY_BAR = 1e-2
GAMMA_NEIGHBORHOOD = 0.2

HOST = InPlaneMedium(1.18, 2.54, 2.54 / 2.19)
INCLUSION = InPlaneMedium(11.4, 2.16, 2.16 / 2.51)
RADIUS = float(np.sqrt(0.30 / np.pi))
OMEGA_SCALE = 2 * np.pi * HOST.c_t
SCAN_STEP = OMEGA_BAR_MAX * OMEGA_SCALE / (N_SCAN - 1)
HALF_WINDOW = 0.45 * SCAN_STEP


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nearest_bessel_zero_distance(omega: float) -> dict:
    distances = []
    for channel, kappa in (("l", omega / HOST.c_l), ("t", omega / HOST.c_t)):
        argument = kappa * A
        for order in range(1, 2 * M_MAX + 2):
            zeros = jn_zeros(order, 4)
            index = int(np.argmin(np.abs(zeros - argument)))
            distances.append((abs(float(zeros[index] - argument)), channel, order, float(zeros[index])))
    distance, channel, order, zero = min(distances)
    return {"distance": distance, "channel": channel, "order": order, "zero": zero}


def continuity_mask(records: list[dict]) -> list[bool]:
    by_k: dict[int, list[float]] = {}
    for record in records:
        if record["eligible"]:
            by_k.setdefault(record["k_index"], []).append(record["omega_bar_refined"])

    mask = []
    for record in records:
        if not record["eligible"]:
            mask.append(False)
            continue
        index = record["k_index"]
        frequency = record["omega_bar_refined"]
        matched = False
        for offset in range(1, CONTINUITY_RADIUS + 1):
            for neighbor in (index - offset, index + offset):
                tolerance = MAX_NEIGHBOR_DELTA_BAR * offset
                if any(abs(frequency - value) <= tolerance for value in by_k.get(neighbor, [])):
                    matched = True
                    break
            if matched:
                break
        mask.append(matched)
    return mask


def local_support(record: dict, records: list[dict]) -> int:
    """Numero de indices k cercanos con una frecuencia compatible."""
    support_indices = {record["k_index"]}
    for other in records:
        offset = abs(other["k_index"] - record["k_index"])
        if not 1 <= offset <= POLE_SUPPORT_RADIUS:
            continue
        if other["low_frequency_away_from_gamma"]:
            continue
        if abs(other["omega_bar_refined"] - record["omega_bar_refined"]) <= POLE_SUPPORT_DELTA_BAR:
            support_indices.add(other["k_index"])
    return len(support_indices)


def main() -> None:
    path = build_path(
        [("M", 1.0, 1.0), ("Gamma", 0.0, 0.0), ("X", 1.0, 0.0), ("M", 1.0, 1.0)],
        A,
        N_PER_SEGMENT,
    )
    with (CYCLE_0B / "bands_cycle_0b.csv").open(newline="", encoding="utf-8") as stream:
        candidates = list(csv.DictReader(stream))

    primary = InPlaneSecularProblem(
        SquareLattice(A), HOST, INCLUSION, RADIUS, M_MAX, LatticeSumConfig(H_PRIMARY)
    )
    check = InPlaneSecularProblem(
        SquareLattice(A), HOST, INCLUSION, RADIUS, M_MAX, LatticeSumConfig(H_CHECK)
    )

    started = time.perf_counter()
    records = []
    for number, row in enumerate(candidates, 1):
        index = int(row["k_index"])
        k = path.k_points[index]
        omega = float(row["omega"])
        first = refine_matrix_candidate(lambda value: primary.M(value, k), omega, HALF_WINDOW)
        second = refine_matrix_candidate(lambda value: check.M(value, k), first.omega, HALF_WINDOW)
        shift_bar = abs(second.omega - first.omega) / OMEGA_SCALE
        pole = nearest_bessel_zero_distance(second.omega)
        matrix_pass = bool(
            first.success
            and second.success
            and first.balanced_sigma_min <= SIGMA_PRIMARY_MAX
            and second.balanced_sigma_min <= SIGMA_CHECK_MAX
            and shift_bar <= MAX_SHIFT_BAR
        )
        reasons = []
        if first.balanced_sigma_min > SIGMA_PRIMARY_MAX:
            reasons.append("primary_residual")
        if second.balanced_sigma_min > SIGMA_CHECK_MAX:
            reasons.append("h_convergence_residual")
        if shift_bar > MAX_SHIFT_BAR:
            reasons.append("h_convergence_shift")
        records.append(
            {
                "k_index": index,
                "Kx": float(k[0]),
                "Ky": float(k[1]),
                "path_distance": float(path.distances[index]),
                "candidate_rank": int(row["candidate_rank"]),
                "omega_bar_original": float(row["omega_bar"]),
                "omega_bar_refined": second.omega / OMEGA_SCALE,
                "sigma_balanced_h8": first.balanced_sigma_min,
                "sigma_balanced_h12": second.balanced_sigma_min,
                "raw_sigma_min_h12": second.raw_sigma_min,
                "raw_sigma_max_h12": second.raw_sigma_max,
                "shift_h8_h12_bar": shift_bar,
                "nearest_bessel_zero_distance": pole["distance"],
                "nearest_bessel_channel": pole["channel"],
                "nearest_bessel_order": pole["order"],
                "nearest_bessel_zero": pole["zero"],
                "pole_suspect": pole["distance"] <= BESSEL_POLE_GUARD,
                "low_frequency_away_from_gamma": bool(
                    second.omega / OMEGA_SCALE <= LOW_FREQUENCY_BAR
                    and np.linalg.norm(k) > GAMMA_NEIGHBORHOOD
                ),
                "matrix_pass": matrix_pass,
                "rejection_reason": "+".join(reasons),
            }
        )
        if number % 50 == 0 or number == len(candidates):
            print(f"[{number}/{len(candidates)}] candidatos validados")

    for record in records:
        record["local_support"] = local_support(record, records)
        record["unsupported_pole"] = bool(
            record["pole_suspect"] and record["local_support"] < MIN_POLE_SUPPORT
        )
        record["eligible"] = bool(
            not record["unsupported_pole"] and not record["low_frequency_away_from_gamma"]
        )
    continuity = continuity_mask(records)
    for record, continuity_pass in zip(records, continuity):
        record["continuity_pass"] = continuity_pass
        record["accepted"] = bool(
            record["eligible"] and (record["matrix_pass"] or continuity_pass)
        )
        if record["unsupported_pole"]:
            record["rejection_reason"] = "unsupported_bessel_pole"
        elif record["low_frequency_away_from_gamma"]:
            record["rejection_reason"] = "low_frequency_away_from_gamma"
        elif not record["accepted"]:
            record["rejection_reason"] = "matrix_and_continuity"
        elif not record["matrix_pass"]:
            record["rejection_reason"] = "rescued_by_continuity"
        else:
            record["rejection_reason"] = ""
    elapsed = time.perf_counter() - started

    csv_path = ROOT / "candidate_validation_cycle_1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    accepted = [record for record in records if record["accepted"]]
    rejected = [record for record in records if not record["accepted"]]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(
        [record["path_distance"] for record in accepted],
        [record["omega_bar_refined"] for record in accepted],
        s=5,
        color="black",
    )
    for position in path.tick_positions:
        ax.axvline(position, color="0.72", linewidth=0.6, zorder=0)
    ax.set_xticks(path.tick_positions)
    ax.set_xticklabels(path.tick_labels)
    ax.set_xlim(path.tick_positions[0], path.tick_positions[-1])
    ax.set_ylim(0.0, OMEGA_BAR_MAX)
    ax.set_ylabel(r"$\omega a/(2\pi c_{t,\mathrm{epoxy}})$")
    ax.set_title("Ciclo 1 — Pb/epoxy in-plane, candidatos validados")
    fig.tight_layout()
    image_path = ROOT / "bands_cycle_1.png"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)

    reasons = Counter(record["rejection_reason"] for record in rejected)
    manifest = {
        "cycle": 1,
        "purpose": "Filtrar candidatos espurios mediante SVD equilibrada, convergencia en h y continuidad en k",
        "input": str(CYCLE_0B / "bands_cycle_0b.csv"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "validation": {
            "h_primary": H_PRIMARY,
            "h_check": H_CHECK,
            "half_window_bar": HALF_WINDOW / OMEGA_SCALE,
            "sigma_primary_max": SIGMA_PRIMARY_MAX,
            "sigma_check_max": SIGMA_CHECK_MAX,
            "max_shift_bar": MAX_SHIFT_BAR,
            "continuity_radius": CONTINUITY_RADIUS,
            "max_neighbor_delta_bar_per_index": MAX_NEIGHBOR_DELTA_BAR,
            "bessel_pole_guard": BESSEL_POLE_GUARD,
            "pole_support_radius": POLE_SUPPORT_RADIUS,
            "min_pole_support": MIN_POLE_SUPPORT,
            "pole_support_delta_bar": POLE_SUPPORT_DELTA_BAR,
            "low_frequency_bar": LOW_FREQUENCY_BAR,
            "gamma_neighborhood": GAMMA_NEIGHBORHOOD,
        },
        "result": {
            "elapsed_s": elapsed,
            "input_candidates": len(records),
            "accepted_candidates": len(accepted),
            "rejected_candidates": len(rejected),
            "rejection_reasons": dict(reasons),
        },
        "outputs": {"csv": csv_path.name, "image": image_path.name},
    }
    manifest["output_sha256"] = {
        csv_path.name: sha256(csv_path),
        image_path.name: sha256(image_path),
    }
    manifest_path = ROOT / "manifest_cycle_1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"image": str(image_path), **manifest["result"]}, indent=2))


if __name__ == "__main__":
    main()
