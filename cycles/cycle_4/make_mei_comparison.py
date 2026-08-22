from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def main() -> None:
    with (ROOT / "bands_cycle_4.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["k_index"]) <= 80]

    distance_by_index = {}
    for row in rows:
        distance_by_index[int(row["k_index"])] = float(row["path_distance"])
    tick_indices = [0, 40, 80]
    ticks = [distance_by_index[index] for index in tick_indices]

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.scatter(
        [float(row["path_distance"]) for row in rows],
        [float(row["omega_bar"]) for row in rows],
        s=7,
        color="black",
    )
    for position in ticks:
        ax.axvline(position, color="0.72", linewidth=0.6, zorder=0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["M", "Gamma", "X"])
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_ylim(0.0, 0.5)
    ax.set_ylabel(r"$\omega a/(2\pi c_{t,\mathrm{Be}})$")
    ax.set_title("Ciclo 4 — ventana comparable con Mei Fig. 1")
    fig.tight_layout()
    fig.savefig(ROOT / "bands_cycle_4_mei_window.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
