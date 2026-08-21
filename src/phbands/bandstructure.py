"""Barrido en k a lo largo de un camino de alta simetria, resolviendo el
problema secular en cada punto y coleccionando las bandas encontradas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lattice import HighSymmetryPath
from .rootfinding import BeynConfig, RootFindingConfig, find_roots, find_roots_beyn


@dataclass
class BandStructureResult:
    path: HighSymmetryPath
    omega_min: float
    omega_max: float
    # bandas[i] = lista de omegas encontrados en el punto k=path.k_points[i]
    bands: list[list[float]] = field(default_factory=list)

    def as_scatter_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Aplana el resultado a (distancia_k, omega) para graficar como
        nube de puntos (cada rama no se conecta explicitamente, igual que
        en las Figs. 2.4-2.5 de la tesis)."""
        xs, ys = [], []
        for dist, omegas in zip(self.path.distances, self.bands):
            for w in omegas:
                xs.append(dist)
                ys.append(w)
        return np.array(xs), np.array(ys)


def compute_band_structure(problem, path: HighSymmetryPath,
                            omega_min: float, omega_max: float,
                            root_config: RootFindingConfig | None = None,
                            verbose: bool = False) -> BandStructureResult:
    """Para cada punto k del camino, busca los omega en (omega_min,omega_max)
    que anulan det(M(omega,k)) (Sec. 12) y los guarda como las bandas.

    `problem` puede ser cualquier objeto con `.det(omega,K)` (duck typing:
    `SecularProblem` antiplano o `InPlaneSecularProblem`). Usa el
    root-finding v0.1 (escaneo+Brent, `rootfinding.scan_bracket`) --
    ver `compute_band_structure_beyn` para la variante v0.2 (integracion
    de contorno), mas robusta cerca de puntos de alta simetria pero que
    requiere `.M(omega,K)` en vez de solo `.det(omega,K)`.
    """
    result = BandStructureResult(path=path, omega_min=omega_min, omega_max=omega_max)
    for idx, K in enumerate(path.k_points):
        det_func = lambda w, K=K: problem.det(w, K)
        omegas = find_roots(det_func, omega_min, omega_max, root_config)
        result.bands.append(omegas)
        if verbose:
            print(f"[{idx + 1}/{len(path)}] K={K}  -> {len(omegas)} bandas encontradas")
    return result


def compute_band_structure_beyn(problem, path: HighSymmetryPath,
                                 omega_min: float, omega_max: float,
                                 beyn_config: BeynConfig | None = None,
                                 verbose: bool = False) -> BandStructureResult:
    """Igual que `compute_band_structure`, pero usando el root-finding v0.2
    (Beyn, `rootfinding.beyn`) en vez de escaneo+Brent. Requiere que
    `problem` exponga `.M(omega,K) -> ndarray` (ambas `SecularProblem` e
    `InPlaneSecularProblem` lo hacen)."""
    result = BandStructureResult(path=path, omega_min=omega_min, omega_max=omega_max)
    for idx, K in enumerate(path.k_points):
        M_func = lambda w, K=K: problem.M(w, K)
        omegas = find_roots_beyn(M_func, omega_min, omega_max, beyn_config)
        result.bands.append(omegas)
        if verbose:
            print(f"[{idx + 1}/{len(path)}] K={K}  -> {len(omegas)} bandas encontradas")
    return result
