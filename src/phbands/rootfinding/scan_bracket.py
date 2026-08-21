"""Estrategia de busqueda de raices v0.1: escaneo + bisecion/Brent sobre
|det(M(omega))|^2.

Diseñada para ser reemplazable: cualquier otra estrategia (p.ej. Beyn por
integracion de contorno) solo necesita exponer la misma firma
`find_roots(det_func, omega_min, omega_max, **kwargs) -> list[float]`.

Limitacion conocida (documentada, no oculta): si dos raices caen mas
cerca entre si que la resolucion de la malla de escaneo, pueden fusionarse
en un unico minimo y perderse una. Se mitiga con:
  - una malla fina configurable (`n_scan`),
  - refinamiento adaptativo cerca de cada minimo candidato,
  - un umbral de aceptacion relativo (no absoluto) para distinguir un cero
    genuino de una simple depresion local del ruido numerico.
Para frecuencias muy cercanas a una degeneracion (p.ej. en puntos de alta
simetria) se recomienda `n_scan` mas alto o, mas adelante, la estrategia
de contorno (Beyn) que no depende de resolucion de malla.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class RootFindingConfig:
    n_scan: int = 600          # puntos de la malla inicial de escaneo
    accept_ratio: float = 1e-3  # h(candidato)/mediana(h en la malla) debe ser menor a esto
    refine_tol: float = 1e-10   # tolerancia del refinamiento local (Brent)


def find_roots(det_func: Callable[[float], complex],
                omega_min: float, omega_max: float,
                config: RootFindingConfig | None = None) -> list[float]:
    """Busca los omega reales en (omega_min, omega_max) donde det_func(omega) = 0,
    escaneando |det_func(omega)|^2 y refinando cada minimo local candidato.
    """
    if config is None:
        config = RootFindingConfig()
    if omega_min <= 0:
        omega_min = max(omega_min, 1e-6)  # evitar omega=0 exacto (kappa=0 puede ser singular)

    omegas = np.linspace(omega_min, omega_max, config.n_scan)
    h = np.array([abs(det_func(w)) ** 2 for w in omegas])

    scale = np.median(h[np.isfinite(h)]) if np.any(np.isfinite(h)) else 1.0
    if scale <= 0:
        scale = 1.0

    roots: list[float] = []
    for i in range(1, len(omegas) - 1):
        if not (np.isfinite(h[i - 1]) and np.isfinite(h[i]) and np.isfinite(h[i + 1])):
            continue
        if h[i] <= h[i - 1] and h[i] <= h[i + 1]:
            lo, hi = omegas[i - 1], omegas[i + 1]
            result = minimize_scalar(
                lambda w: abs(det_func(w)) ** 2,
                bounds=(lo, hi), method="bounded",
                options={"xatol": config.refine_tol},
            )
            if not result.success:
                continue
            if result.fun / scale < config.accept_ratio:
                roots.append(float(result.x))

    # eliminar posibles duplicados muy cercanos (mismo minimo capturado dos veces)
    roots.sort()
    deduped: list[float] = []
    for r in roots:
        if not deduped or abs(r - deduped[-1]) > 10 * config.refine_tol:
            deduped.append(r)
    return deduped
