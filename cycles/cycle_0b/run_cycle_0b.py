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


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from phbands.bandstructure import compute_band_structure
from phbands.inplane.media import InPlaneMedium
from phbands.inplane.secular import InPlaneSecularProblem
from phbands.lattice import SquareLattice, build_path
from phbands.lattice_sums import LatticeSumConfig
from phbands.rootfinding import RootFindingConfig


A = 1.0
M_MAX = 2
H_MAX = 8
N_SCAN = 200
N_PER_SEGMENT = 40
OMEGA_BAR_MAX = 1.0
BASELINE_COMMIT = "73d4ee3f9b1c79d6362901d67a5f71fd2408bba3"

HOST = InPlaneMedium(1.18, 2.54, 2.54 / 2.19)
INCLUSION = InPlaneMedium(11.4, 2.16, 2.16 / 2.51)
FILLING_FRACTION = 0.30


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    path = build_path(
        [("M", 1.0, 1.0), ("Gamma", 0.0, 0.0), ("X", 1.0, 0.0), ("M", 1.0, 1.0)],
        A,
        N_PER_SEGMENT,
    )
    radius = float(np.sqrt(FILLING_FRACTION / np.pi))
    problem = InPlaneSecularProblem(
        SquareLattice(A),
        HOST,
        INCLUSION,
        radius,
        M_MAX,
        LatticeSumConfig(H_MAX),
    )
    omega_max = OMEGA_BAR_MAX * 2 * np.pi * HOST.c_t

    started = time.perf_counter()
    result = compute_band_structure(
        problem,
        path,
        1e-6,
        omega_max,
        RootFindingConfig(n_scan=N_SCAN),
        verbose=True,
    )
    elapsed = time.perf_counter() - started

    rows = []
    for i, (k, distance, roots) in enumerate(zip(path.k_points, path.distances, result.bands)):
        for rank, omega in enumerate(roots):
            rows.append(
                {
                    "k_index": i,
                    "Kx": float(k[0]),
                    "Ky": float(k[1]),
                    "path_distance": float(distance),
                    "candidate_rank": rank,
                    "omega": float(omega),
                    "omega_bar": float(omega / (2 * np.pi * HOST.c_t)),
                }
            )

    csv_path = ROOT / "bands_cycle_0b.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    xs, ys = result.as_scatter_arrays()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(xs, ys / (2 * np.pi * HOST.c_t), s=5, color="black")
    for position in path.tick_positions:
        ax.axvline(position, color="0.72", linewidth=0.6, zorder=0)
    ax.set_xticks(path.tick_positions)
    ax.set_xticklabels(path.tick_labels)
    ax.set_xlim(path.tick_positions[0], path.tick_positions[-1])
    ax.set_ylim(0.0, OMEGA_BAR_MAX)
    ax.set_ylabel(r"$\omega a/(2\pi c_{t,\mathrm{epoxy}})$")
    ax.set_title("Ciclo 0B — Pb/epoxy, modos in-plane")
    fig.tight_layout()
    image_path = ROOT / "bands_cycle_0b.png"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)

    manifest = {
        "cycle": "0B",
        "purpose": "Corregir A22 de la matriz T_n y alinear la comparacion con Mei Fig. 2",
        "baseline_commit": BASELINE_COMMIT,
        "source_tree": str(REPOSITORY),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "physics": {
            "system": "lead cylinders in epoxy",
            "polarization": "in-plane only",
            "filling_fraction": FILLING_FRACTION,
            "radius_over_a": radius,
            "host": {"rho": HOST.rho, "c_l": HOST.c_l, "c_t": HOST.c_t},
            "inclusion": {"rho": INCLUSION.rho, "c_l": INCLUSION.c_l, "c_t": INCLUSION.c_t},
        },
        "changes_from_cycle_0": [
            "A22 displacement element changed from +P_t to -P_t (Mei Appendix B)",
            "path changed to M-Gamma-X-M",
            "path distances now match the sampled k points",
            "omega_bar_max changed from 0.9 to 1.0",
            "only Pb/epoxy is evaluated",
        ],
        "numerics": {
            "m_max": M_MAX,
            "h_max": H_MAX,
            "n_scan": N_SCAN,
            "n_per_segment": N_PER_SEGMENT,
            "omega_bar_max": OMEGA_BAR_MAX,
            "root_method": "uniform scan + bounded minimization of |det(M)|^2",
        },
        "path": {"labels": list(path.tick_labels), "k_points": len(path)},
        "result": {
            "elapsed_s": elapsed,
            "candidate_count": len(rows),
            "candidates_per_k": [len(values) for values in result.bands],
        },
        "outputs": {"csv": csv_path.name, "image": image_path.name},
    }
    manifest_path = ROOT / "manifest_cycle_0b.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["output_sha256"] = {
        csv_path.name: sha256(csv_path),
        image_path.name: sha256(image_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"image": str(image_path), "csv": str(csv_path), **manifest["result"]}, indent=2))


if __name__ == "__main__":
    main()
