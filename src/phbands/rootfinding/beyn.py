"""Root-finding v0.2: metodo de Beyn por integracion de contorno (Beyn,
"An integral method for solving nonlinear eigenvalue problems", Linear
Algebra Appl. 2012).

Motivacion (ver HANDOFF.md): el escaneo+Brent de `scan_bracket.py` pierde
raices sistematicamente cerca de puntos de alta simetria -- degeneraciones
donde dos o mas raices caen mas cerca entre si de lo que la malla del
escaneo puede resolver, y se fusionan en un solo minimo detectado. Se
confirmo empiricamente que esto no es un simple problema de "subir
n_scan": en Gamma/X/M el numero de raices encontradas seguia creciendo
incluso a n_scan=20000 sin terminar de saturar.

El metodo de Beyn no depende de resolucion de malla: cuenta los ceros de
det(M(z)) dentro de un contorno cerrado del plano complejo evaluando
integrales de contorno de M(z)^-1 (generalizacion del principio del
argumento a problemas de autovalores no lineales). Raices arbitrariamente
cercanas dentro del mismo contorno se resuelven correctamente mientras el
subespacio de prueba `l` sea suficientemente grande (este modulo lo
auto-incrementa si detecta saturacion).

Diferencia de interfaz con `scan_bracket.find_roots`: aqui se recibe la
funcion MATRICIAL `M_func(z) -> ndarray`, no `det_func(z) -> complex` --
Beyn necesita resolver sistemas lineales con M(z), no solo su
determinante. Ver `bandstructure.compute_band_structure_beyn`.

Nota importante: este metodo soluciona la perdida de raices por
resolucion de malla, pero NO la convergencia de `m_max`/`h_max` (el corte
modal/de red debe seguir siendo suficiente para el problema fisico --
p.ej. cavidades casi tocandose en redes de llenado alto requieren m_max
mas alto independientemente de la estrategia de root-finding usada).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class BeynConfig:
    n_windows: int = 40       # ventanas (contornos) que cubren [omega_min,omega_max]
    overlap: float = 0.25     # fraccion de superposicion entre ventanas vecinas
    l_guess: int = 6          # dimension inicial del subespacio de prueba (>= raices esperadas por ventana)
    l_max: int = 40           # tope para el auto-incremento de l si una ventana esta saturada
    n_quad: int = 32          # puntos de cuadratura trapezoidal sobre el contorno circular
    svd_tol: float = 1e-10    # umbral relativo (respecto al mayor valor singular) para el rango numerico
    imag_tol: float = 1e-3    # tolerancia relativa (respecto al radio) para aceptar un autovalor como real
    dedup_tol: float = 1e-6   # tolerancia absoluta en omega para fusionar raices de ventanas solapadas
    refine: bool = True       # pulir cada raiz con un ajuste local (Brent) sobre |det|^2
    accept_ratio: float = 1e-6  # |det(M(candidato))| / escala_ventana debe ser menor a esto (filtra
                                 # autovalores espurios de B causados por ruido numerico en la SVD --
                                 # ver nota en find_roots)
    cross_validate: bool = True   # correr con 2 semillas independientes y quedarse solo con
                                   # las raices que ambas encuentran (ver find_roots)
    cross_validate_tol: float = 1e-4  # tolerancia absoluta en omega para emparejar raices entre semillas


def _contour_points(z_c: complex, rho: float, n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    """Puntos y pesos de un contorno circular para la cuadratura
    trapezoidal (exponencialmente exacta en contornos analiticos cerrados
    y suaves). Para z(t)=z_c+rho*e^{it}: (1/2pi i) dz = (1/2pi)(z-z_c) dt,
    por lo que el peso por punto es simplemente (z_j - z_c)."""
    t = np.arange(n_quad) * (2.0 * np.pi / n_quad)
    z = z_c + rho * np.exp(1j * t)
    weight = z - z_c
    return z, weight


def _beyn_window(M_func: Callable[[complex], np.ndarray], z_c: complex, rho: float,
                  l: int, n_quad: int, svd_tol: float,
                  rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Aplica el algoritmo de Beyn en un contorno circular (z_c, rho).
    Devuelve (autovalores encontrados, saturado). `saturado=True` si el
    rango numerico de A0 alcanzo `l` -- indicio de que hay mas raices en
    la ventana de las que el subespacio de prueba puede resolver; el
    llamador debe reintentar con `l` mas grande."""
    N = M_func(z_c).shape[0]
    V_hat = rng.standard_normal((N, l)) + 1j * rng.standard_normal((N, l))

    z_pts, w_pts = _contour_points(z_c, rho, n_quad)
    A0 = np.zeros((N, l), dtype=complex)
    A1 = np.zeros((N, l), dtype=complex)
    for z_j, w_j in zip(z_pts, w_pts):
        Minv_V = np.linalg.solve(M_func(z_j), V_hat)
        A0 += Minv_V * w_j
        A1 += Minv_V * (w_j * z_j)
    A0 /= n_quad
    A1 /= n_quad

    U, s, Wh = np.linalg.svd(A0, full_matrices=False)
    if s[0] <= 0:
        return np.array([]), False
    k = int(np.sum(s > svd_tol * s[0]))
    saturated = k >= l
    if k == 0:
        return np.array([]), saturated

    Uk = U[:, :k]
    Wk = Wh[:k, :].conj().T
    sk = s[:k]
    B = (Uk.conj().T @ A1 @ Wk) @ np.diag(1.0 / sk)
    eigvals = np.linalg.eigvals(B)
    return eigvals, saturated


def _polish(M_func: Callable[[complex], np.ndarray], w0: float, eps: float) -> float:
    """Afina una raiz candidata minimizando |det(M(omega))|^2 en un
    entorno pequeno -- Beyn da la raiz con precision de cuadratura/SVD,
    esto la lleva a precision de maquina (mismo mecanismo de refinamiento
    que usa `scan_bracket.find_roots`)."""
    result = minimize_scalar(
        lambda w: abs(np.linalg.det(M_func(w))) ** 2,
        bounds=(w0 - eps, w0 + eps), method="bounded",
        options={"xatol": 1e-12},
    )
    return float(result.x) if result.success else w0


def _find_roots_single_seed(M_func: Callable[[complex], np.ndarray],
                             omega_min: float, omega_max: float,
                             config: BeynConfig, seed: int) -> list[float]:
    """Barre [omega_min,omega_max] con ventanas (contornos circulares)
    superpuestas, aplicando Beyn en cada una; auto-incrementa `l` si una
    ventana esta saturada. Devuelve las raices reales encontradas,
    ordenadas y sin duplicados, para UNA sola semilla aleatoria (ver
    `find_roots` para la validacion cruzada entre semillas)."""
    if config is None:
        config = BeynConfig()
    if omega_min <= 0:
        omega_min = max(omega_min, 1e-6)

    rng = np.random.default_rng(seed)
    span = omega_max - omega_min
    window_size = span / config.n_windows
    step = window_size * (1.0 - config.overlap)

    found: list[float] = []
    start = omega_min
    while start < omega_max:
        end = min(start + window_size, omega_max)
        z_c = 0.5 * (start + end)
        rho = 0.5 * (end - start) / (1.0 - config.overlap)
        # no cruzar el corte de rama de H_n^(1) en Re(z)<=0 (relevante
        # solo para la primera ventana, cerca de omega_min)
        rho = min(rho, 0.999 * z_c)

        l = config.l_guess
        while True:
            eigvals, saturated = _beyn_window(M_func, z_c, rho, l, config.n_quad,
                                               config.svd_tol, rng)
            if not saturated or l >= config.l_max:
                break
            l = min(l * 2, config.l_max)

        if len(eigvals) == 0:
            start += step
            continue

        # Escala local de |det(M)| en la ventana (referencia para el
        # chequeo de aceptacion de abajo). El rango numerico de A0 via SVD
        # es una condicion NECESARIA pero no suficiente de que z sea un
        # cero de det(M) -- ruido numerico en las colas de la SVD puede
        # producir autovalores de B espurios (se observo empiricamente:
        # valores repetidos identicos en ventanas sin relacion). Se
        # verifica cada candidato contra la definicion real del problema,
        # igual que hace `scan_bracket.find_roots` con `accept_ratio`.
        scale = np.median([abs(np.linalg.det(M_func(start))),
                            abs(np.linalg.det(M_func(z_c))),
                            abs(np.linalg.det(M_func(end)))])
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0

        for z in eigvals:
            if abs(z.imag) > config.imag_tol * rho:
                continue  # descarta autovalores espurios no-reales
            w = float(z.real)
            if not (start - 1e-9 <= w <= end + 1e-9):
                continue
            det_w = abs(np.linalg.det(M_func(w)))
            if det_w / scale > config.accept_ratio:
                continue  # espurio: no es realmente un cero de det(M)
            found.append(w)

        start += step

    found.sort()
    deduped: list[float] = []
    for w in found:
        if not deduped or (w - deduped[-1]) > config.dedup_tol:
            deduped.append(w)

    if config.refine:
        polish_eps = max(window_size * 0.05, 1e-9)
        polished = sorted(_polish(M_func, w, polish_eps) for w in deduped)
        deduped = []
        for w in polished:
            if not deduped or (w - deduped[-1]) > config.dedup_tol:
                deduped.append(w)

    return deduped


def find_roots(M_func: Callable[[complex], np.ndarray],
               omega_min: float, omega_max: float,
               config: BeynConfig | None = None,
               seed: int = 0) -> list[float]:
    """Busca las raices reales de det(M(omega))=0 en (omega_min,omega_max)
    con el metodo de Beyn.

    Por defecto (`config.cross_validate=True`) corre el barrido dos veces
    con semillas aleatorias independientes (el vector de prueba `V_hat` de
    Beyn es aleatorio) y se queda solo con las raices que AMBAS corridas
    encuentran (emparejadas dentro de `cross_validate_tol`). Esto es
    necesario en la practica: se observo empiricamente que un umbral SVD
    demasiado laxo deja pasar autovalores de B que son ruido numerico, no
    ceros genuinos de det(M) -- y aunque `accept_ratio` ya filtra la
    mayoria, un cruce entre semillas independientes es la forma robusta de
    eliminar los que sobreviven por casualidad (el ruido no es reproducible
    entre semillas distintas; una raiz genuina si lo es)."""
    if config is None:
        config = BeynConfig()

    roots_a = _find_roots_single_seed(M_func, omega_min, omega_max, config, seed)
    if not config.cross_validate:
        return roots_a

    roots_b = _find_roots_single_seed(M_func, omega_min, omega_max, config, seed + 1)
    confirmed: list[float] = []
    for w in roots_a:
        if any(abs(w - wb) <= config.cross_validate_tol for wb in roots_b):
            confirmed.append(w)
    return confirmed
