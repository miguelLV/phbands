"""Ensamblaje del problema secular (Sec. 11-12 del documento canonico):

    a = S b          (geometria/periodicidad de la red)
    b = T a          (fisica local del dispersor)

Combinando:  a = S T a  =>  (I - S T) a = 0.
Solucion no trivial (Sec. 12):  det(I - S T) = 0.

Se trabaja sobre el sistema en `a` (no en `b`); ambas formas dan el mismo
conjunto de ceros del determinante (Sec. 12: det(I-ST) = det(I-TS)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import SquareLattice
from .lattice_sums import LatticeSumConfig, S_matrix
from .scatterer.base import Scatterer


@dataclass
class SecularProblem:
    """Agrupa todo lo necesario para evaluar M(omega,K) = I - S(omega,K) T(omega)
    para un sistema fisico dado (red + dispersor), a un corte modal m_max.
    """

    lattice: SquareLattice
    scatterer: Scatterer
    m_max: int
    lattice_sum_config: LatticeSumConfig | None = None

    @property
    def n_modes(self) -> int:
        return 2 * self.m_max + 1

    def M(self, omega: float, K: np.ndarray) -> np.ndarray:
        """M(omega,K) = I - S(kappa(omega),K) @ diag(T(omega))."""
        kappa = self.scatterer.host_kappa(omega)  # tipo de dispersor decide como obtener kappa
        S = S_matrix(kappa, K, self.lattice, self.m_max, self.lattice_sum_config)
        T_diag = self.scatterer.T_diag(omega, self.m_max)
        I = np.eye(self.n_modes, dtype=complex)
        return I - S @ np.diag(T_diag)

    def det(self, omega: float, K: np.ndarray) -> complex:
        return np.linalg.det(self.M(omega, K))
