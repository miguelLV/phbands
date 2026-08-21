"""Reproduce la Fig. 2.4 de la tesis (docs/Tesis_MLetelier.pdf, p. 24):
estructura de bandas de ondas elasticas polarizadas EN EL PLANO x-y
(longitudinal + transversal acopladas) para dos sistemas de materiales,
replicando los resultados de Mei et al. [53].

(a) Matriz de Berilio con incrustaciones cilindricas de Oro, nu=0.0908.
(b) Matriz de Epoxy con incrustaciones cilindricas de Plomo, nu=0.3.

A diferencia de reproduce_fig_2_5.py (cavidad antiplana, pipeline
validado contra el documento canonico), este caso usa el subpaquete
`phbands.inplane`, que sigue DIRECTAMENTE la tesis (Sec. 2.1.4, Anexo B)
-- no auditado independientemente, solo validado con el chequeo algebraico
`validate_zero_scattering_limit` (D=0 cuando inclusion=anfitrion).

Camino reciproco: X-Gamma-M-X, igual eje que en la Fig. 2.4.
Eje vertical: omega_bar = omega*a/(2*pi*C_t0) con C_t0 la velocidad
transversal del medio ANFITRION de cada panel (a=1, nondimensionalizado).
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from phbands.bandstructure import compute_band_structure
from phbands.inplane.media import InPlaneMedium
from phbands.inplane.secular import InPlaneSecularProblem
from phbands.lattice import SquareLattice
from phbands.lattice_sums import LatticeSumConfig
from phbands.rootfinding import RootFindingConfig

# Parametros numericos calibrados en esta sesion (ver HANDOFF.md): con
# m_max=2 (el mismo corte modal que uso la tesis), h_max=8, n_scan=200,
# las raices coinciden a 3-4 decimales con m_max=3,h_max=15,n_scan=300
# (mucho mas lento), a ~2s por punto k.
M_MAX = 2
H_MAX = 8
N_SCAN = 200
N_PER_SEGMENT = 40
OMEGA_BAR_MAX = 0.9  # margen sobre el ~0.7-0.85 que muestra la Fig. 2.4

A = 1.0

CASES = {
    "(a) Berilio + Oro": dict(
        host=InPlaneMedium(rho=1.85, c_l=13.11, c_t=13.11 / 1.44),
        inclusion=InPlaneMedium(rho=19.5, c_l=3.36, c_t=3.36 / 2.71),
        nu=0.0908,
    ),
    "(b) Epoxy + Plomo": dict(
        host=InPlaneMedium(rho=1.18, c_l=2.54, c_t=2.54 / 2.19),
        inclusion=InPlaneMedium(rho=11.4, c_l=2.16, c_t=2.16 / 2.51),
        nu=0.3,
    ),
}


def x_gamma_m_x_path(a: float, n_per_segment: int):
    from phbands.lattice import build_path
    labeled = [
        ("X", 1.0, 0.0),
        ("Gamma", 0.0, 0.0),
        ("M", 1.0, 1.0),
        ("X", 1.0, 0.0),
    ]
    return build_path(labeled, a, n_per_segment)


def run_case(host: InPlaneMedium, inclusion: InPlaneMedium, nu: float, path):
    radius = np.sqrt(nu / np.pi)  # nu = pi*r1^2/a^2, a=1
    lattice = SquareLattice(a=A)
    problem = InPlaneSecularProblem(
        lattice=lattice, host=host, inclusion=inclusion, radius=radius,
        m_max=M_MAX, lattice_sum_config=LatticeSumConfig(h_max=H_MAX),
    )
    root_config = RootFindingConfig(n_scan=N_SCAN)
    omega_max = OMEGA_BAR_MAX * 2 * np.pi * host.c_t

    t0 = time.time()
    result = compute_band_structure(
        problem, path, omega_min=1e-6, omega_max=omega_max,
        root_config=root_config, verbose=True,
    )
    elapsed = time.time() - t0
    print(f"{elapsed:.1f}s para {len(path)} puntos k")
    return result, host.c_t


def main():
    path = x_gamma_m_x_path(A, N_PER_SEGMENT)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (label, params) in zip(axes, CASES.items()):
        result, c_t0 = run_case(params["host"], params["inclusion"], params["nu"], path)
        xs, ys = result.as_scatter_arrays()
        omega_bar = ys / (2 * np.pi * c_t0)
        ax.scatter(xs, omega_bar, s=4, color="black")
        for tp in path.tick_positions:
            ax.axvline(tp, color="gray", linewidth=0.5, zorder=0)
        ax.set_xticks(path.tick_positions)
        ax.set_xticklabels(path.tick_labels)
        ax.set_xlabel("ka")
        ax.set_title(f"{label}  (nu={params['nu']})")
        ax.set_ylim(0, OMEGA_BAR_MAX)
    axes[0].set_ylabel(r"$\omega a / 2\pi C_{t0}$")
    fig.suptitle("Reproduccion Fig. 2.4 (polarizacion en el plano, red cuadrada)")
    fig.tight_layout()

    out_path = "examples/fig_2_4_reproduction.png"
    fig.savefig(out_path, dpi=150)
    print(f"Figura guardada en {out_path}")


if __name__ == "__main__":
    main()
