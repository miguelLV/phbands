"""Ensamblaje del problema secular para ondas en el plano (tesis,
ec. 2.82-2.86): det|T S - I| = 0.

Indice combinado (modo m, polarizacion alpha in {l,t}): a diferencia del
caso antiplano (un escalar T_n por modo), aqui cada modo tiene un bloque
2x2 (`boundary.transfer_matrix_D`) que mezcla l/t DENTRO del mismo modo
-- nunca entre modos distintos (ec. 2.66: b_m = D^{m'} delta_m^{m'} a_{m'}).
Por eso T_combined es bloque-diagonal por modo (bloques 2x2).

La suma de red, en cambio, NO mezcla polarizaciones (ec. 2.83/2.85: la
formula es identica a la del caso antiplano, solo evaluada en kappa_l o
kappa_t segun el canal) -- por eso S_combined es bloque-diagonal por
POLARIZACION, reusando `lattice_sums.S_matrix` sin modificarlo, una vez
por canal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..lattice import Lattice
from ..lattice_sums import LatticeSumConfig, S_matrix
from .boundary import transfer_matrix_D
from .media import InPlaneMedium


@dataclass
class InPlaneSecularProblem:
    lattice: Lattice
    host: InPlaneMedium
    inclusion: InPlaneMedium
    radius: float
    m_max: int
    lattice_sum_config: LatticeSumConfig | None = None

    @property
    def n_modes(self) -> int:
        return 2 * self.m_max + 1

    def M(self, omega: float, K: np.ndarray) -> np.ndarray:
        """M(omega,K) = T(omega) @ S(omega,K) - I, indexado por (m,alpha)
        intercalado como [m=-m_max,l; m=-m_max,t; m=-m_max+1,l; ...]."""
        modes = list(range(-self.m_max, self.m_max + 1))
        n_modes = self.n_modes
        N = 2 * n_modes

        kappa_l = self.host.kappa_l(omega)
        kappa_t = self.host.kappa_t(omega)
        S_l = S_matrix(kappa_l, K, self.lattice, self.m_max, self.lattice_sum_config)
        S_t = S_matrix(kappa_t, K, self.lattice, self.m_max, self.lattice_sum_config)

        S = np.zeros((N, N), dtype=complex)
        S[0::2, 0::2] = S_l
        S[1::2, 1::2] = S_t

        T = np.zeros((N, N), dtype=complex)
        for i, m in enumerate(modes):
            D_m = transfer_matrix_D(omega, m, self.radius, self.host, self.inclusion)
            T[2 * i:2 * i + 2, 2 * i:2 * i + 2] = D_m

        I = np.eye(N, dtype=complex)
        return T @ S - I

    def det(self, omega: float, K: np.ndarray) -> complex:
        return np.linalg.det(self.M(omega, K))
