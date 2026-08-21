"""Reproduce la Fig. 2.5 de la tesis (docs/Tesis_MLetelier.pdf, p. 25):
estructura de bandas de ondas antiplanas en una matriz de caucho con una
red cuadrada de cavidades cilindricas, para razones de llenado nu=0.1 y
nu=0.7.

Caso (tesis, Sec. 2.2.1, p.25): rho_c = 920 kg/m^3, C_t0 = 30 m/s. Como
todo esta nondimensionalizado (a=1), se usa HostMedium(mu=1, rho=1) =>
c_t=1, kappa(omega)=omega, y se grafica omega_bar = omega/(2*pi) (porque
omega_bar = omega*a/(2*pi*C_t0) con a=C_t0=1 en estas unidades).

Camino reciproco: X-Gamma-M-X, igual que en la Fig. 2.5 de la tesis.

Parametros numericos calibrados en esta sesion (ver HANDOFF.md): con
m_max=3, h_max=10, n_scan=300 las raices coinciden a 3-4 decimales con
m_max=5,h_max=30,n_scan=600 (que tardaba ~30s por punto k), a ~1-2s por
punto -- suficiente para barrer el camino completo en minutos.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from phbands.bandstructure import compute_band_structure
from phbands.lattice import SquareLattice, build_path
from phbands.lattice_sums import LatticeSumConfig
from phbands.media import HostMedium
from phbands.rootfinding import RootFindingConfig
from phbands.scatterer.cavity import CavityScatterer
from phbands.secular import SecularProblem

A = 1.0
M_MAX = 3
H_MAX = 10
N_SCAN = 300
N_PER_SEGMENT = 40
OMEGA_BAR_MAX = 1.4  # margen sobre el ~1.2-1.3 que muestra la Fig. 2.5


def x_gamma_m_x_path(a: float, n_per_segment: int):
    """Camino X-Gamma-M-X (en unidades de pi/a), igual al eje horizontal
    de la Fig. 2.5 de la tesis (a diferencia de standard_GXM_path, que
    empieza y termina en Gamma)."""
    labeled = [
        ("X", 1.0, 0.0),
        ("Gamma", 0.0, 0.0),
        ("M", 1.0, 1.0),
        ("X", 1.0, 0.0),
    ]
    return build_path(labeled, a, n_per_segment)


def run_case(nu: float, path):
    radius = np.sqrt(nu / np.pi)  # nu = pi*r1^2/a^2, a=1
    host = HostMedium(mu=1.0, rho=1.0)
    lattice = SquareLattice(a=A)
    scatterer = CavityScatterer(radius=radius, host=host)
    problem = SecularProblem(
        lattice=lattice,
        scatterer=scatterer,
        m_max=M_MAX,
        lattice_sum_config=LatticeSumConfig(h_max=H_MAX),
    )
    root_config = RootFindingConfig(n_scan=N_SCAN)
    omega_max = OMEGA_BAR_MAX * 2 * np.pi  # omega = omega_bar * 2*pi*C_t0, C_t0=1

    t0 = time.time()
    result = compute_band_structure(
        problem, path, omega_min=1e-6, omega_max=omega_max,
        root_config=root_config, verbose=True,
    )
    elapsed = time.time() - t0
    print(f"nu={nu}: {elapsed:.1f}s para {len(path)} puntos k")
    return result


def main():
    lattice = SquareLattice(a=A)
    path = x_gamma_m_x_path(A, N_PER_SEGMENT)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, nu, label in zip(axes, (0.1, 0.7), ("(a)", "(b)")):
        result = run_case(nu, path)
        xs, ys = result.as_scatter_arrays()
        omega_bar = ys / (2 * np.pi)
        ax.scatter(xs, omega_bar, s=4, color="black")
        for tp in path.tick_positions:
            ax.axvline(tp, color="gray", linewidth=0.5, zorder=0)
        ax.set_xticks(path.tick_positions)
        ax.set_xticklabels(path.tick_labels)
        ax.set_xlabel("ka")
        ax.set_title(f"{label} nu = {nu}")
        ax.set_ylim(0, OMEGA_BAR_MAX)
    axes[0].set_ylabel(r"$\omega a / 2\pi C_{t0}$")
    fig.suptitle("Reproduccion Fig. 2.5 (cavidades en caucho, red cuadrada)")
    fig.tight_layout()

    out_path = "examples/fig_2_5_reproduction.png"
    fig.savefig(out_path, dpi=150)
    print(f"Figura guardada en {out_path}")


if __name__ == "__main__":
    main()
