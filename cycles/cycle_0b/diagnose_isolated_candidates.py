from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from phbands.inplane.media import InPlaneMedium
from phbands.inplane.secular import InPlaneSecularProblem
from phbands.lattice import SquareLattice
from phbands.lattice_sums import LatticeSumConfig


HOST = InPlaneMedium(1.18, 2.54, 2.54 / 2.19)
INCLUSION = InPlaneMedium(11.4, 2.16, 2.16 / 2.51)
RADIUS = float(np.sqrt(0.30 / np.pi))
SCALE = 2 * np.pi * HOST.c_t


def matrix_metrics(problem, omega, k):
    matrix = problem.M(omega, k)
    singular = np.linalg.svd(matrix, compute_uv=False)
    return {
        "abs_det": float(abs(np.linalg.det(matrix))),
        "sigma_min": float(singular[-1]),
        "sigma_max": float(singular[0]),
        "sigma_ratio": float(singular[-1] / singular[0]),
    }


def main():
    with (ROOT / "bands_cycle_0b.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    selected = [
        row
        for row in rows
        if 0.59 < float(row["omega_bar"]) < 0.63 or float(row["omega_bar"]) < 0.01
    ]
    report = []
    for row in selected:
        omega_bar = float(row["omega_bar"])
        omega = omega_bar * SCALE
        k = np.array([float(row["Kx"]), float(row["Ky"])])
        entry = {
            "k_index": int(row["k_index"]),
            "k": k.tolist(),
            "omega_bar": omega_bar,
            "convergence": {},
        }
        for m_max, h_max in [(2, 6), (2, 8), (2, 10), (2, 12), (3, 8), (3, 12)]:
            problem = InPlaneSecularProblem(
                SquareLattice(1.0), HOST, INCLUSION, RADIUS, m_max, LatticeSumConfig(h_max)
            )
            entry["convergence"][f"m{m_max}_h{h_max}"] = matrix_metrics(problem, omega, k)
        report.append(entry)

    output = ROOT / "isolated_candidate_diagnostics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "candidate_count": len(report)}, indent=2))


if __name__ == "__main__":
    main()
