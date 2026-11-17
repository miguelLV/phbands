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

from .lattice import Lattice
from .lattice_sums import LatticeSumConfig, S_matrix
from .scatterer.base import Scatterer


@dataclass
class SecularProblem:
    """Agrupa todo lo necesario para evaluar M(omega,K) = I - S(omega,K) T(omega)
    para un sistema fisico dado (red + dispersor), a un corte modal m_max.
    """

    lattice: Lattice
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


@dataclass
class BasisSecularProblem:
    """Problema secular para una red con BASE: varios dispersores por celda,
    el dispersor `j` en la posicion `positions[j]` (p.ej. panal de abejas).

    Bloques: a^alpha = sum_beta S^{alpha beta} b^beta, con
    S^{alpha alpha} la suma propia y S^{alpha beta} la suma cruzada con
    offset c = d_alpha - d_beta (`lattice_sums.S_matrix(..., offset=c)`), y
    b^beta = T^beta a^beta. Con un solo dispersor es identico a
    `SecularProblem`. Todos los dispersores comparten el medio anfitrion.
    """

    lattice: Lattice
    scatterers: tuple[Scatterer, ...]
    positions: tuple[tuple[float, float], ...]
    m_max: int
    lattice_sum_config: LatticeSumConfig | None = None
    symmetrized: bool = False

    def __post_init__(self) -> None:
        if len(self.scatterers) != len(self.positions):
            raise ValueError("un dispersor por posicion")

    @property
    def n_modes(self) -> int:
        return len(self.scatterers) * (2 * self.m_max + 1)

    def M(self, omega: float, K: np.ndarray) -> np.ndarray:
        kappa = self.scatterers[0].host_kappa(omega)
        nb = 2 * self.m_max + 1
        pos = [np.asarray(p, dtype=float) for p in self.positions]
        S = np.zeros((self.n_modes, self.n_modes), dtype=complex)
        self_block = S_matrix(kappa, K, self.lattice, self.m_max, self.lattice_sum_config)
        for i in range(len(pos)):
            for j in range(len(pos)):
                c = pos[i] - pos[j]
                block = (self_block if i == j else
                         S_matrix(kappa, K, self.lattice, self.m_max, self.lattice_sum_config, offset=c))
                S[i * nb:(i + 1) * nb, j * nb:(j + 1) * nb] = block
        T_diag = self._T_diag(omega)
        I = np.eye(self.n_modes, dtype=complex)
        if self.symmetrized:
            root_T = np.sqrt(T_diag.astype(complex))
            return I - root_T[:, None] * S * root_T[None, :]
        return I - S @ np.diag(T_diag)

    def _T_diag(self, omega: float) -> np.ndarray:
        # T no depende de K y el escaneo global repite la misma malla de omega
        # en cada punto k: se cachea (las capas pre-deformadas integran una EDO).
        cache = self.__dict__.setdefault("_T_cache", {})
        T = cache.get(omega)
        if T is None:
            if len(cache) > 20000:
                cache.clear()
            T = np.concatenate([sc.T_diag(omega, self.m_max) for sc in self.scatterers])
            cache[omega] = T
        return T

    def det(self, omega: float, K: np.ndarray) -> complex:
        return np.linalg.det(self.M(omega, K))
