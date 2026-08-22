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


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
CYCLE_1 = ROOT.parent / "cycle_1"
sys.path.insert(0, str(REPOSITORY / "src"))

from phbands.inplane.media import InPlaneMedium
from phbands.inplane.secular import InPlaneSecularProblem
from phbands.lattice import SquareLattice, build_path
from phbands.lattice_sums import LatticeSumConfig
from phbands.rootfinding.gap_detection import detect_bridge_gaps
from phbands.rootfinding.validation import equilibrate_matrix


HOST = InPlaneMedium(1.18, 2.54, 2.54 / 2.19)
INCLUSION = InPlaneMedium(11.4, 2.16, 2.16 / 2.51)
RADIUS = float(np.sqrt(0.30 / np.pi))
OMEGA_SCALE = 2 * np.pi * HOST.c_t
KNOWN_GAPS = {
    1: 0.35956238320875616,
    2: 0.3573836773237277,
    3: 0.3539888021821088,
    31: 0.9005302845623716,
    71: 0.42918755706918454,
    72: 0.4354148285704355,
    73: 0.43417024055117204,
    104: 0.856568507875814,
}
HIGH_SYMMETRY_INDICES = [0, 40, 80, 119]
MULTIPLICITY_THRESHOLD = 1e-3
MULTIPLICITY_RADIUS = 3
WINDOW_HALF_WIDTH = 0.02
PROJECTED_LOCAL_SAMPLES = 101
SECONDS_PER_MATRIX_EVALUATION = 50.81 / (12 * 801)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_windows(raw_windows):
    grouped = defaultdict(list)
    for window in raw_windows:
        grouped[int(window["k_index"])].append(window)
    merged = []
    for index, windows in sorted(grouped.items()):
        ordered = sorted(windows, key=lambda item: item["lower_bar"])
        for window in ordered:
            if merged and merged[-1]["k_index"] == index and window["lower_bar"] <= merged[-1]["upper_bar"]:
                merged[-1]["lower_bar"] = min(merged[-1]["lower_bar"], window["lower_bar"])
                merged[-1]["upper_bar"] = max(merged[-1]["upper_bar"], window["upper_bar"])
                merged[-1]["reasons"] = sorted(set(merged[-1]["reasons"] + window["reasons"]))
                merged[-1]["evidence_count"] += window["evidence_count"]
            else:
                merged.append(dict(window))
    return merged


def main() -> None:
    started = time.perf_counter()
    path = build_path(
        [("M", 1.0, 1.0), ("Gamma", 0.0, 0.0), ("X", 1.0, 0.0), ("M", 1.0, 1.0)],
        1.0,
        40,
    )
    with (CYCLE_1 / "candidate_validation_cycle_1.csv").open(newline="", encoding="utf-8") as stream:
        accepted = [row for row in csv.DictReader(stream) if row["accepted"] == "True"]
    bands = [[] for _ in range(len(path))]
    for row in accepted:
        bands[int(row["k_index"])].append(float(row["omega_bar_refined"]))
    for values in bands:
        values.sort()

    bridge_triggers = detect_bridge_gaps(bands)
    raw_windows = [
        {
            "k_index": trigger.k_index,
            "lower_bar": max(0.0, trigger.frequency - WINDOW_HALF_WIDTH),
            "upper_bar": min(1.0, trigger.frequency + WINDOW_HALF_WIDTH),
            "reasons": [trigger.reason],
            "evidence_count": 1,
        }
        for trigger in bridge_triggers
    ]

    problem = InPlaneSecularProblem(
        SquareLattice(1.0), HOST, INCLUSION, RADIUS, 2, LatticeSumConfig(8)
    )
    multiplicities = []
    for index in HIGH_SYMMETRY_INDICES:
        k = path.k_points[index]
        for omega_bar in bands[index]:
            matrix = equilibrate_matrix(problem.M(omega_bar * OMEGA_SCALE, k))
            singular = np.linalg.svd(matrix, compute_uv=False)
            second_ratio = float(singular[-2] / singular[0])
            if second_ratio <= MULTIPLICITY_THRESHOLD:
                multiplicities.append(
                    {"k_index": index, "omega_bar": omega_bar, "second_singular_ratio": second_ratio}
                )
                direction = 1 if index == 0 else -1 if index == len(path) - 1 else 0
                if direction:
                    neighbors = [index + direction * offset for offset in range(1, MULTIPLICITY_RADIUS + 1)]
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
                            "lower_bar": max(0.0, omega_bar - WINDOW_HALF_WIDTH),
                            "upper_bar": min(1.0, omega_bar + WINDOW_HALF_WIDTH),
                            "reasons": ["high_symmetry_multiplicity"],
                            "evidence_count": 1,
                        }
                    )

    windows = merge_windows(raw_windows)
    activated_indices = sorted({window["k_index"] for window in windows})
    recovered_known = sorted(
        index
        for index, frequency in KNOWN_GAPS.items()
        if any(
            window["k_index"] == index
            and window["lower_bar"] <= frequency <= window["upper_bar"]
            for window in windows
        )
    )
    missed_known = sorted(set(KNOWN_GAPS).difference(recovered_known))
    projected_evaluations = len(windows) * PROJECTED_LOCAL_SAMPLES
    projected_seconds = projected_evaluations * SECONDS_PER_MATRIX_EVALUATION
    elapsed = time.perf_counter() - started

    windows_path = ROOT / "activation_windows_cycle_2c.json"
    windows_path.write_text(json.dumps(windows, indent=2), encoding="utf-8")
    report = {
        "cycle": "2C",
        "purpose": "Calibrar activacion automatica de SVD local sin crear ni eliminar raices",
        "roots_modified": False,
        "known_gaps": KNOWN_GAPS,
        "known_detected": recovered_known,
        "known_missed": missed_known,
        "known_recall": len(recovered_known) / len(KNOWN_GAPS),
        "bridge_trigger_count_before_merge": len(bridge_triggers),
        "activation_window_count": len(windows),
        "activated_k_indices": activated_indices,
        "activated_fraction": len(activated_indices) / len(path),
        "multiplicity_events": multiplicities,
        "projection": {
            "local_samples_per_window": PROJECTED_LOCAL_SAMPLES,
            "matrix_evaluations": projected_evaluations,
            "additional_seconds": projected_seconds,
        },
        "elapsed_s": elapsed,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": {"windows": windows_path.name, "image": "activation_map_cycle_2c.png"},
    }

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for index, values in enumerate(bands):
        ax.scatter([path.distances[index]] * len(values), values, s=5, color="black")
    for window in windows:
        x = path.distances[window["k_index"]]
        ax.plot([x, x], [window["lower_bar"], window["upper_bar"]], color="red", linewidth=2, alpha=0.7)
    for position in path.tick_positions:
        ax.axvline(position, color="0.75", linewidth=0.6, zorder=0)
    ax.set_xticks(path.tick_positions)
    ax.set_xticklabels(path.tick_labels)
    ax.set_xlim(path.tick_positions[0], path.tick_positions[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(r"$\omega a/(2\pi c_t)$")
    ax.set_title("Ciclo 2C — ventanas activadas automáticamente (rojo)")
    fig.tight_layout()
    image_path = ROOT / "activation_map_cycle_2c.png"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)

    report["output_sha256"] = {
        windows_path.name: sha256(windows_path),
        image_path.name: sha256(image_path),
    }
    manifest = ROOT / "manifest_cycle_2c.json"
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("known_detected", "known_missed", "known_recall", "activation_window_count", "activated_fraction", "projection", "elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()
