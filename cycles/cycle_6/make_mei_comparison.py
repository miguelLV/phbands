"""Vista conjunta de modos XY y Z en la ventana publicada por MEI."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if int(row["k_index"]) <= 80 and float(row["omega_bar"]) <= 0.5]


inplane = read_rows(REPOSITORY / "cycles" / "cycle_5" / "bands_cycle_5.csv")
antiplane = read_rows(ROOT / "bands_cycle_6.csv")

fig, ax = plt.subplots(figsize=(6.4, 5.2))
ax.scatter(
    [float(row["path_distance"]) for row in inplane],
    [float(row["omega_bar"]) for row in inplane],
    s=8,
    color="black",
    label="In-plane (XY)",
)
ax.scatter(
    [float(row["path_distance"]) for row in antiplane],
    [float(row["omega_bar"]) for row in antiplane],
    s=14,
    facecolors="none",
    edgecolors="black",
    linewidths=0.7,
    label="Antiplano (Z)",
)

gamma = max(float(row["path_distance"]) for row in inplane if int(row["k_index"]) == 40)
x_point = max(float(row["path_distance"]) for row in inplane if int(row["k_index"]) == 80)
for position in (0.0, gamma, x_point):
    ax.axvline(position, color="0.72", linewidth=0.6, zorder=0)
ax.set_xticks([0.0, gamma, x_point], ["M", "Gamma", "X"])
ax.set_xlim(0.0, x_point)
ax.set_ylim(0.0, 0.5)
ax.set_ylabel(r"$\omega a/(2\pi c_{t,\mathrm{Be}})$")
ax.set_title("Au/Be - comparación con la ventana de MEI")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "bands_cycle_6_mei_combined.png", dpi=180)
plt.close(fig)
