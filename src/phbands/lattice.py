"""Red directa y reciproca para una red cuadrada 2D.

Convenciones canonicas (ver Modelo_fisico_matematico_canonico_v0_1.docx,
Sec. 3 y Sec. 14):

    R = a * (n_x, n_y),            n_x, n_y in Z          (red directa)
    G_h = (2*pi/a) * (h_x, h_y),   h_x, h_y in Z           (red reciproca)
    A_c = a**2                                              (area celda)

Para dos centros p, q:  R_qp = R_q - R_p  (apunta de p a q).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, NamedTuple, Sequence

import numpy as np


@lru_cache(maxsize=64)
def _cached_reciprocal_xy(a: float, h_max: int) -> tuple[np.ndarray, np.ndarray]:
    two_pi_a = 2.0 * np.pi / a
    h = np.arange(-h_max, h_max + 1)
    hx, hy = np.meshgrid(h, h, indexing="ij")
    Gx = (two_pi_a * hx).ravel()
    Gy = (two_pi_a * hy).ravel()
    return Gx, Gy


class PolarVector(NamedTuple):
    """Un vector 2D en forma (magnitud, angulo) junto a sus componentes."""

    x: float
    y: float
    r: float
    theta: float

    @classmethod
    def from_xy(cls, x: float, y: float) -> "PolarVector":
        return cls(x=x, y=y, r=float(np.hypot(x, y)), theta=float(np.arctan2(y, x)))


@dataclass(frozen=True)
class SquareLattice:
    """Red cuadrada de constante `a` (ver Sec. 14 del documento canonico)."""

    a: float

    @property
    def cell_area(self) -> float:
        """A_c = a^2."""
        return self.a**2

    def direct_vectors(self, n_max: int) -> list[PolarVector]:
        """Vectores de la red directa R = a*(nx,ny) con nx,ny in [-n_max,n_max],
        excluyendo el origen (R != 0, como exige la suma de red, Sec. 10).

        `n_max` es el numero de "capas" (shells) alrededor del origen que se
        incluyen; a mayor n_max, mayor alcance de la suma directa. Se usa
        principalmente para la suma de red *directa* (lenta, de referencia)
        de la Sec. 10, no para la representacion acelerada de la Sec. 20.
        """
        vecs: list[PolarVector] = []
        for nx in range(-n_max, n_max + 1):
            for ny in range(-n_max, n_max + 1):
                if nx == 0 and ny == 0:
                    continue
                vecs.append(PolarVector.from_xy(self.a * nx, self.a * ny))
        return vecs

    def reciprocal_vectors(self, h_max: int) -> list[PolarVector]:
        """Vectores de la red reciproca G_h = (2*pi/a)*(hx,hy),
        hx,hy in [-h_max,h_max], INCLUYENDO (0,0) (h=0 contribuye en la
        representacion acelerada, Sec. 20).

        NOTA: esta version en lista de PolarVector es conveniente pero
        lenta si se llama repetidamente (p.ej. en el interior de un
        root-finder). Para produccion usar `reciprocal_vectors_xy`, que
        cachea el resultado como arrays de numpy.
        """
        vecs: list[PolarVector] = []
        two_pi_a = 2.0 * np.pi / self.a
        for hx in range(-h_max, h_max + 1):
            for hy in range(-h_max, h_max + 1):
                vecs.append(PolarVector.from_xy(two_pi_a * hx, two_pi_a * hy))
        return vecs

    def reciprocal_vectors_xy(self, h_max: int) -> tuple[np.ndarray, np.ndarray]:
        """Igual que `reciprocal_vectors` pero devuelve (Gx, Gy) como arrays
        de numpy 1D, cacheado por (a, h_max) -- pensado para el interior de
        bucles calientes (suma de red evaluada miles de veces)."""
        return _cached_reciprocal_xy(self.a, h_max)


class HighSymmetryPath:
    """Camino en el espacio reciproco definido por una secuencia de puntos
    etiquetados (fracciones de pi/a), interpolados linealmente con un numero
    de puntos por segmento.

    NOTA: tu documento canonico marca el "camino de alta simetria y
    frecuencia adimensional definitiva" como TODAVIA NO CERRADO (Sec. 25.2).
    Aqui se usa la convencion estandar para red cuadrada:
        Gamma = (0,0),  X = (pi/a, 0),  M = (pi/a, pi/a)
    que es la mas comun en la literatura (Mei et al. 2003) y coincide con la
    reciproca G_h = 2*pi/a*(hx,hy) ya fijada en Sec. 14. Si mas adelante
    decides otra normalizacion, solo hay que ajustar `standard_GXM_path`.
    """

    def __init__(self, k_points: np.ndarray, distances: np.ndarray,
                 tick_positions: Sequence[float], tick_labels: Sequence[str]):
        self.k_points = k_points          # (N, 2) array
        self.distances = distances        # (N,) distancia acumulada a lo largo del camino
        self.tick_positions = tick_positions
        self.tick_labels = tick_labels

    def __len__(self) -> int:
        return len(self.k_points)

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.k_points)


def build_path(labeled_points: Sequence[tuple[str, float, float]],
               a: float, n_per_segment: int) -> HighSymmetryPath:
    """Construye un HighSymmetryPath a partir de puntos etiquetados dados en
    unidades de pi/a, p.ej. ('X', 1.0, 0.0) representa kx=pi/a, ky=0.
    """
    pts_k = np.array([[kx * np.pi / a, ky * np.pi / a] for _, kx, ky in labeled_points])
    labels = [lbl for lbl, _, _ in labeled_points]

    all_k = []
    all_dist = []
    tick_positions = []
    cum_dist = 0.0
    for i in range(len(pts_k) - 1):
        p0, p1 = pts_k[i], pts_k[i + 1]
        seg_k = np.linspace(p0, p1, n_per_segment, endpoint=(i == len(pts_k) - 2))
        seg_len = np.linalg.norm(p1 - p0)
        # La distancia debe corresponder al punto k realmente muestreado. En
        # segmentos con endpoint=False el ultimo punto no alcanza p1; repartir
        # igualmente [0, seg_len] producia una abscisa duplicada con el nodo
        # inicial del segmento siguiente.
        seg_dist = cum_dist + np.linalg.norm(seg_k - p0, axis=1)
        tick_positions.append(cum_dist)
        all_k.append(seg_k)
        all_dist.append(seg_dist)
        cum_dist += seg_len
    tick_positions.append(cum_dist)

    k_points = np.concatenate(all_k, axis=0)
    distances = np.concatenate(all_dist, axis=0)
    return HighSymmetryPath(k_points, distances, tick_positions, labels)


def standard_GXM_path(a: float, n_per_segment: int = 60) -> HighSymmetryPath:
    """Camino Gamma - X - M - Gamma estandar para red cuadrada."""
    labeled = [
        ("Gamma", 0.0, 0.0),
        ("X", 1.0, 0.0),
        ("M", 1.0, 1.0),
        ("Gamma", 0.0, 0.0),
    ]
    return build_path(labeled, a, n_per_segment)
