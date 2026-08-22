from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
CYCLE_1 = ROOT.parent / "cycle_1"
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(ROOT))

from phbands.inplane.media import InPlaneMedium
from phbands.inplane.secular import InPlaneSecularProblem
from phbands.lattice import SquareLattice, build_path
from phbands.lattice_sums import LatticeSumConfig
from phbands.rootfinding.validation import refine_matrix_candidate
from run_cycle_2 import (
    MAX_SHIFT_BAR,
    RECIPROCAL_POLE_GUARD_BAR,
    SIGMA_CHECK_MAX,
    SIGMA_PRIMARY_MAX,
    deduplicate,
    local_minima,
    matrix_curves,
    nearest_reciprocal_pole_bar,
)


HOST = InPlaneMedium(1.18, 2.54, 2.54 / 2.19)
INCLUSION = InPlaneMedium(11.4, 2.16, 2.16 / 2.51)
RADIUS = float(np.sqrt(0.30 / np.pi))
OMEGA_SCALE = 2 * np.pi * HOST.c_t
GRID_SIZE = 801
SINGULAR_RATIO_CUTOFF = 2e-2
MATCH_EXISTING_BAR = 8e-4
REGIONS = [
    {"name": "M_crossing", "indices": list(range(0, 5)), "lower_bar": 0.30, "upper_bar": 0.39},
    {"name": "Gamma_X_crossing", "indices": list(range(69, 76)), "lower_bar": 0.39, "upper_bar": 0.47},
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    path = build_path(
        [("M", 1.0, 1.0), ("Gamma", 0.0, 0.0), ("X", 1.0, 0.0), ("M", 1.0, 1.0)],
        1.0,
        40,
    )
    with (CYCLE_1 / "candidate_validation_cycle_1.csv").open(newline="", encoding="utf-8") as stream:
        accepted_cycle_1 = [row for row in csv.DictReader(stream) if row["accepted"] == "True"]
    existing_by_k: dict[int, list[float]] = {}
    for row in accepted_cycle_1:
        existing_by_k.setdefault(int(row["k_index"]), []).append(float(row["omega_bar_refined"]))

    primary = InPlaneSecularProblem(
        SquareLattice(1.0), HOST, INCLUSION, RADIUS, 2, LatticeSumConfig(8)
    )
    check = InPlaneSecularProblem(
        SquareLattice(1.0), HOST, INCLUSION, RADIUS, 2, LatticeSumConfig(12)
    )
    started = time.perf_counter()
    records = []
    curve_data = []

    for region in REGIONS:
        for index in region["indices"]:
            k = path.k_points[index]
            frequency_bar = np.linspace(region["lower_bar"], region["upper_bar"], GRID_SIZE)
            omega_grid = frequency_bar * OMEGA_SCALE
            determinant, singular_ratio = matrix_curves(primary, k, omega_grid)
            curve_data.append((region["name"], index, frequency_bar, determinant, singular_ratio))
            grid_step = float(omega_grid[1] - omega_grid[0])

            positions = [
                int(position)
                for position in local_minima(singular_ratio)
                if singular_ratio[position] <= SINGULAR_RATIO_CUTOFF
            ]
            refined = []
            for position in positions:
                result = refine_matrix_candidate(
                    lambda value: primary.M(value, k),
                    float(omega_grid[position]),
                    1.5 * grid_step,
                )
                refined.append(result.omega)
            refined = deduplicate(refined, tolerance=0.35 * grid_step)

            for omega in refined:
                first = refine_matrix_candidate(lambda value: primary.M(value, k), omega, 1.5 * grid_step)
                second = refine_matrix_candidate(lambda value: check.M(value, k), first.omega, 1.5 * grid_step)
                omega_bar = second.omega / OMEGA_SCALE
                pole_distance, pole_channel, pole_bar = nearest_reciprocal_pole_bar(k, omega_bar)
                pole_suspect = pole_distance <= RECIPROCAL_POLE_GUARD_BAR
                shift_bar = abs(second.omega - first.omega) / OMEGA_SCALE
                existing = any(
                    abs(omega_bar - value) <= MATCH_EXISTING_BAR
                    for value in existing_by_k.get(index, [])
                )
                converged = bool(
                    first.balanced_sigma_min <= SIGMA_PRIMARY_MAX
                    and second.balanced_sigma_min <= SIGMA_CHECK_MAX
                    and shift_bar <= MAX_SHIFT_BAR
                    and not pole_suspect
                )
                records.append(
                    {
                        "region": region["name"],
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
                        "reciprocal_pole_suspect": pole_suspect,
                        "already_present_cycle_1": existing,
                        "converged": converged,
                        "new_recovered_root": bool(converged and not existing),
                    }
                )
            print(f"{region['name']} k[{index}]: {len(positions)} minimos SVD")

    elapsed = time.perf_counter() - started
    recovered = [record for record in records if record["new_recovered_root"]]
    previous_manifest = json.loads((ROOT / "manifest_cycle_2.json").read_text(encoding="utf-8"))
    previous_recovered = previous_manifest["result"]["new_roots"]
    combined_recovered = [*previous_recovered, *recovered]

    csv_path = ROOT / "crossing_candidate_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    for axis, region in zip(axes, REGIONS):
        for name, index, frequency, _, singular_ratio in curve_data:
            if name == region["name"]:
                axis.plot(frequency, np.log10(np.maximum(singular_ratio, 1e-300)), linewidth=0.8, label=f"k[{index}]")
        for record in recovered:
            if record["region"] == region["name"]:
                axis.scatter(record["omega_bar"], -7.5, marker="x", color="red", s=35)
        axis.set_title(region["name"])
        axis.set_ylabel(r"$\log_{10}(\sigma_{min}/\sigma_{max})$")
        axis.grid(alpha=0.2)
        axis.legend(ncol=4, fontsize=7)
    axes[-1].set_xlabel(r"$\omega a/(2\pi c_t)$")
    fig.tight_layout()
    diagnostic_image = ROOT / "crossing_diagnostics.png"
    fig.savefig(diagnostic_image, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(
        [float(row["path_distance"]) for row in accepted_cycle_1],
        [float(row["omega_bar_refined"]) for row in accepted_cycle_1],
        s=5,
        color="black",
        label="Ciclo 1",
    )
    if combined_recovered:
        ax.scatter(
            [path.distances[record["k_index"]] for record in combined_recovered],
            [record["omega_bar"] for record in combined_recovered],
            marker="x",
            color="red",
            s=22,
            label="Raiz recuperada directamente",
        )
    for position in path.tick_positions:
        ax.axvline(position, color="0.72", linewidth=0.6, zorder=0)
    ax.set_xticks(path.tick_positions)
    ax.set_xticklabels(path.tick_labels)
    ax.set_xlim(path.tick_positions[0], path.tick_positions[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(r"$\omega a/(2\pi c_{t,\mathrm{epoxy}})$")
    ax.set_title("Ciclo 2 — raíces recuperadas por búsqueda directa")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    band_image = ROOT / "bands_cycle_2_crossings.png"
    fig.savefig(band_image, dpi=180)
    plt.close(fig)

    summary = {
        "purpose": "Diagnostico directo de perdidas cerca de los cruces indicados por el usuario",
        "interpolation_used_to_create_roots": False,
        "grid_size_per_k": GRID_SIZE,
        "regions": REGIONS,
        "elapsed_s": elapsed,
        "candidate_count": len(records),
        "new_converged_roots": len(recovered),
        "new_roots": recovered,
        "combined_cycle_2_recovered_roots": len(combined_recovered),
        "combined_roots": combined_recovered,
        "outputs": {"csv": csv_path.name, "diagnostic_image": diagnostic_image.name, "band_image": band_image.name},
    }
    summary["output_sha256"] = {
        csv_path.name: sha256(csv_path),
        diagnostic_image.name: sha256(diagnostic_image),
        band_image.name: sha256(band_image),
    }
    manifest = ROOT / "manifest_cycle_2_crossings.json"
    manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"band_image": str(band_image), **{key: summary[key] for key in ("elapsed_s", "candidate_count", "new_converged_roots", "new_roots")}}, indent=2))


if __name__ == "__main__":
    main()
