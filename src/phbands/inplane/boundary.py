"""Matriz de transferencia (Anexo B de la tesis) para una inclusion
cilindrica elastica embebida (bonded), polarizacion en el plano.

Boundary conditions en r=R (ecs. 2.63a-c): continuidad de desplazamiento
(u_r, u_theta) y de esfuerzo (sigma_rr, sigma_r-theta), acoplando las
amplitudes de onda incidente `a_m=(a_lm,a_tm)`, dispersada
`b_m=(b_lm,b_tm)` e interior `c_m=(c_lm,c_tm)` (ec. 2.65a/2.65b):

    A_1 a_m + B_1 b_m = C_1 c_m     (continuidad de desplazamiento)
    A_2 a_m + B_2 b_m = C_2 c_m     (continuidad de esfuerzo)

con A_i,B_i,C_i matrices 2x2 (Anexo B, ecs. B.1a-B.1w). Resolviendo el
sistema se obtiene la matriz de dispersion D (ec. 2.67) tal que
b_m = D_m a_m -- el analogo, para este problema, del T_n escalar del caso
antiplano (`phbands.scatterer.cavity`).

Los bloques 2x2 siguen un patron: la fila "desplazamiento" mezcla una
funcion tipo-gradiente (P) para el canal longitudinal con una tipo-rotor
(Q) para el transversal, y viceversa (siguiendo phi_l=grad(...),
phi_t=rot(...), ec. 2.58); la fila "esfuerzo" analogamente mezcla U/V/W.
Ver B.1a-B.1l (bloques 1,2) y B.1m-B.1w (bloques 3,4).
"""

from __future__ import annotations

import numpy as np

from ..basis import H1, J
from .media import InPlaneMedium


def _P(C, k: complex, R: float, m: int) -> complex:
    """[k R C_{m-1}(kR) - m C_m(kR)] / R  (B.1a, B.1h, B.1e, B.1l tipo)."""
    z = k * R
    return (k * R * C(m - 1, z) - m * C(m, z)) / R


def _Q(C, k: complex, R: float, m: int) -> complex:
    """i m C_m(kR) / R  (B.1b, B.1g, B.1f, B.1k tipo)."""
    z = k * R
    return 1j * m * C(m, z) / R


def _U(C, k: complex, R: float, m: int, mu: float, lam: float) -> complex:
    """[(2*mu*m*(1+m) - k^2 R^2 (lam+2mu)) C_m(kR) - 2 mu k R C_{m-1}(kR)] / R^2
    (B.1m, B.1p tipo)."""
    z = k * R
    return ((2 * mu * m * (1 + m) - k**2 * R**2 * (lam + 2 * mu)) * C(m, z)
             - 2 * mu * k * R * C(m - 1, z)) / R**2


def _V(C, k: complex, R: float, m: int, mu: float) -> complex:
    """2i mu m (k R C_{m-1}(kR) - (m+1) C_m(kR)) / R^2 (B.1n, B.1r, B.1q, B.1v tipo)."""
    z = k * R
    return 2j * mu * m * (k * R * C(m - 1, z) - (m + 1) * C(m, z)) / R**2


def _W(C, k: complex, R: float, m: int, mu: float) -> complex:
    """mu [2 k R C_{m-1}(kR) + (k^2 R^2 - 2 m(m+1)) C_m(kR)] / R^2
    (B.1s, B.1w tipo)."""
    z = k * R
    return mu * (2 * k * R * C(m - 1, z) + (k**2 * R**2 - 2 * m * (m + 1)) * C(m, z)) / R**2


def _block_12(C, k_l: complex, k_t: complex, R: float, m: int) -> np.ndarray:
    """Bloque 2x2 de continuidad de desplazamiento (filas 1,2 del Anexo B)."""
    return np.array([
        [_P(C, k_l, R, m), _Q(C, k_t, R, m)],
        [_Q(C, k_l, R, m), _P(C, k_t, R, m)],
    ], dtype=complex)


def _block_34(C, k_l: complex, k_t: complex, R: float, m: int,
              mu: float, lam: float) -> np.ndarray:
    """Bloque 2x2 de continuidad de esfuerzo (filas 3,4 del Anexo B)."""
    return np.array([
        [_U(C, k_l, R, m, mu, lam), _V(C, k_t, R, m, mu)],
        [_V(C, k_l, R, m, mu), _W(C, k_t, R, m, mu)],
    ], dtype=complex)


def transfer_matrix_D(omega: float, m: int, radius: float,
                       host: InPlaneMedium, inclusion: InPlaneMedium) -> np.ndarray:
    """Matriz de dispersion D_m (2x2, ec. 2.67) para el modo m: b_m = D_m a_m.

    D = (C_2 C_1^-1 B_1 - B_2)^-1 (A_2 - C_2 C_1^-1 A_1)
    """
    k_l0 = host.kappa_l(omega)
    k_t0 = host.kappa_t(omega)
    k_ls = inclusion.kappa_l(omega)
    k_ts = inclusion.kappa_t(omega)
    R = radius

    A1 = _block_12(J, k_l0, k_t0, R, m)
    B1 = _block_12(H1, k_l0, k_t0, R, m)
    C1 = _block_12(J, k_ls, k_ts, R, m)

    A2 = _block_34(J, k_l0, k_t0, R, m, host.mu, host.lam)
    B2 = _block_34(H1, k_l0, k_t0, R, m, host.mu, host.lam)
    C2 = _block_34(J, k_ls, k_ts, R, m, inclusion.mu, inclusion.lam)

    C1_inv = np.linalg.inv(C1)
    lhs = C2 @ C1_inv @ B1 - B2
    rhs = A2 - C2 @ C1_inv @ A1
    return np.linalg.solve(lhs, rhs)


def validate_zero_scattering_limit(omega: float, m_values, radius: float,
                                    medium: InPlaneMedium, tol: float = 1e-8) -> dict:
    """Chequeo de correccion algebraica de `transfer_matrix_D` (no viene de
    la tesis, es un invariante fisico obvio que sirve de red de seguridad):
    si la "inclusion" tiene EXACTAMENTE las mismas propiedades que el
    medio anfitrion, no hay interfaz real -> no debe haber dispersion, es
    decir D_m debe ser identicamente cero para todo modo m.

    Algebraicamente: si A_1=C_1 y A_2=C_2 (mismos materiales), entonces
    (A_2 - C_2 C_1^-1 A_1) = A_2 - A_2 @ I = 0, sin importar el otro
    factor -- por lo que D=0 es una consecuencia directa de la formula
    (ec. 2.67), no una coincidencia numerica. Si esto falla, hay un error
    de transcripcion en `_block_12`/`_block_34` respecto al Anexo B.
    """
    max_norm = 0.0
    for m in m_values:
        D = transfer_matrix_D(omega, m, radius, medium, medium)
        max_norm = max(max_norm, float(np.max(np.abs(D))))
    return {"max_abs_D": max_norm, "passed": max_norm < tol}
