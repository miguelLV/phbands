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
    symmetrized: bool = False

    @property
    def n_modes(self) -> int:
        return 2 * self.m_max + 1

    def M(self, omega: float, K: np.ndarray) -> np.ndarray:
        """M(omega,K) = I - S(kappa(omega),K) @ diag(T(omega)).

        Con `symmetrized=True` devuelve en cambio la forma equilibrada

            M_sym = I - T^{1/2} S T^{1/2}

        que tiene EXACTAMENTE el mismo determinante (con D=T^{1/2} diagonal
        e invertible, D (I - D S D) D^{-1} = I - D^2 S = I - T S, y
        det(I-TS)=det(I-ST)), por lo que no altera en nada la busqueda de
        raices por determinante. Lo que si cambia es el CONDICIONAMIENTO,
        y por tanto todos los diagnosticos basados en SVD.

        Motivo (ver docs/diagnostico_soluciones_espurias.md): en la forma
        I - S diag(T) los dos factores crecen/decrecen de forma opuesta y
        extrema -- S_{m-n} contiene H^{(1)}_{m-n}, que crece factorialmente
        con |m-n|, mientras T_n decae super-exponencialmente con |n|. En las
        esquinas de la matriz se multiplica un numero enorme por uno
        minusculo, y el sistema truncado adquiere direcciones casi-nulas
        espurias localizadas en los multipolos del borde +-m_max. Eso
        arruina el test `edge_weight` de `rootfinding.certification`, que
        deja de decrecer al aumentar m_max (se midio: 0.087 -> 0.88 -> 0.97
        para m_max 4 -> 6 -> 10, cuando deberia decrecer). Repartiendo
        T^{1/2} a ambos lados el decaimiento queda simetrico y el mismo
        test si converge (0.010 -> 0.0039 -> 1.3e-4 para los mismos m_max).

        La literatura de matriz-T / metodos multipolares identifica la
        normalizacion diagonal de la matriz del sistema como requisito de
        estabilidad al truncar; ver las Fuentes del documento citado.
        """
        kappa = self.scatterer.host_kappa(omega)  # tipo de dispersor decide como obtener kappa
        S = S_matrix(kappa, K, self.lattice, self.m_max, self.lattice_sum_config)
        T_diag = self.scatterer.T_diag(omega, self.m_max)
        I = np.eye(self.n_modes, dtype=complex)
        if self.symmetrized:
            root_T = np.sqrt(T_diag.astype(complex))
            return I - root_T[:, None] * S * root_T[None, :]
        return I - S @ np.diag(T_diag)

    def det(self, omega: float, K: np.ndarray) -> complex:
        return np.linalg.det(self.M(omega, K))
