"""Suma de red S_l(kappa, K): la parte del problema que codifica la
geometria y periodicidad de la red (Sec. 9-11, 15-20 del documento
canonico). Esta es la contraparte de T: S es pura geometria/red, T es
pura fisica local del dispersor.

Se implementan DOS metodos, deliberadamente, para poder validar uno
contra el otro (ver `validate_against_direct_sum` mas abajo):

1. `direct_sum`: suma directa Ec. Sec. 10 (exacta mientras converja).
   H_l^{(1)}(kappa*R) decae solo como 1/sqrt(R) para kappa real, por lo
   que la suma directa es apenas CONDICIONALMENTE convergente en el eje
   real -- este es precisamente el problema que motiva a Ewald/Chin
   (ver tesis Sec. 2.2.1, p. 22). Por eso `direct_sum` solo se usa como
   referencia con Im(kappa) > 0 (convergencia geometrica exacta), NUNCA
   como metodo de produccion en el eje real.

2. `chin_accelerated`: representacion acelerada canonica auditada
   (Sec. 20), la que se usa en produccion (converge absolutamente,
   Wu & Zhang 2009 / Chin, Nicorovici & McPhedran 1994).

Ambas deben coincidir (dentro de tolerancia numerica) para el mismo
kappa complejo con Im(kappa) > 0: eso es exactamente lo que hace
`validate_against_direct_sum`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import hankel1, jv

from .lattice import SquareLattice


def direct_sum(l: int, kappa: complex, K: np.ndarray,
                lattice: SquareLattice, n_shells: int = 40) -> complex:
    """S_l via suma directa (Sec. 10):

        S_l(kappa, K) = sum_{R != 0} e^{i K.R} H_l^{(1)}(kappa R) e^{-i l Theta_R}

    Solo confiable para Im(kappa) > 0 (convergencia geometrica). Para
    kappa real se recomienda NO usar esta funcion en produccion: usar
    `chin_accelerated`.
    """
    vecs = lattice.direct_vectors(n_shells)
    total = 0.0 + 0.0j
    for v in vecs:
        phase_bloch = np.exp(1j * (K[0] * v.x + K[1] * v.y))
        phase_ang = np.exp(-1j * l * v.theta)
        total += phase_bloch * hankel1(l, kappa * v.r) * phase_ang
    return total


def chin_accelerated(l: int, kappa: complex, K: np.ndarray,
                      lattice: SquareLattice, h_max: int = 30) -> complex:
    """S_l via la representacion acelerada canonica (Sec. 20):

        S_l(kappa,K) = 1/J_{|l|+1}(kappa a) *
            [ -(H_1^{(1)}(kappa a) + 2i/(pi kappa a)) delta_{l,0}
              - (4 i^{l+1} kappa / A_c) * sum_h J_{|l|+1}(Q_h a) e^{-i l theta_h}
                                              / (Q_h (Q_h^2 - kappa^2)) ]

    valida para l entero (positivo, negativo o cero) segun Sec. 18-20.
    """
    a = lattice.a
    A_c = lattice.cell_area
    # Orden de Bessel que aparece en la formula: J_{|l|+1}, NO J_{|l|}
    # (Sec. 20 del documento canonico). Cuidado: es un off-by-one facil de
    # cometer al transcribir "|l|+1".
    order = abs(l) + 1

    Gx, Gy = lattice.reciprocal_vectors_xy(h_max)  # cacheado: no se reconstruye cada llamada
    # el vector Q_h incluye K: Q_h = G_h + K
    Qx = Gx + K[0]
    Qy = Gy + K[1]
    Qh = np.hypot(Qx, Qy)
    thetah = np.arctan2(Qy, Qx)

    denom_pole = Qh * (Qh**2 - kappa**2)
    # Caso especial Q_h -> 0 (ocurre en h=(0,0) cuando K=Gamma=(0,0)):
    # J_{order}(Qh*a)/Qh es una forma 0/0. El limite analitico es:
    #   order == 1 (l=0):  lim_{Qh->0} J_1(Qh*a)/Qh = a/2  -> termino = (a/2)/(-kappa^2)
    #   order  > 1        :  el limite es 0 (J_order ~ Qh^order decae mas rapido)
    near_zero = np.isclose(Qh, 0.0, atol=1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        recip_terms = jv(order, Qh * a) * np.exp(-1j * l * thetah) / denom_pole
    if np.any(near_zero):
        limit_value = (a / 2.0) / (-(kappa**2)) if order == 1 else 0.0
        recip_terms = np.where(near_zero, limit_value, recip_terms)
    recip_terms = np.where(np.isfinite(recip_terms), recip_terms, 0.0)
    recip_sum = np.sum(recip_terms)

    prefactor = 1.0 / jv(order, kappa * a)
    # OJO: `4j**(l+1)` en Python es (4j)**(l+1) == (4i)^(l+1), NO 4*i^(l+1).
    # Hay que escribir el factor i^(l+1) explicitamente como (1j)**(l+1).
    bracket = -4.0 * (1j ** (l + 1)) * kappa / A_c * recip_sum
    if l == 0:
        bracket += -(hankel1(1, kappa * a) + 2j / (np.pi * kappa * a))

    return prefactor * bracket


@dataclass
class LatticeSumConfig:
    """Parametros numericos de la suma de red acelerada."""

    h_max: int = 30       # capas de red reciproca sumadas


def S_matrix(kappa: complex, K: np.ndarray, lattice: SquareLattice,
             m_max: int, config: LatticeSumConfig | None = None) -> np.ndarray:
    """Ensambla la matriz S_{mn} = S_{m-n}(kappa,K), m,n in [-m_max,m_max],
    usando la representacion acelerada (Sec. 10-11: a = S b).

    Version vectorizada: calcula S_l para TODOS los l in [-2m_max,2m_max]
    de una sola vez (broadcasting sobre l y h), en vez de un bucle Python
    que reevalua la suma reciproca por separado para cada l. Debe dar
    exactamente los mismos valores que llamar `chin_accelerated` l por l
    (ver tests/validacion) -- aqui solo se optimiza velocidad.
    """
    if config is None:
        config = LatticeSumConfig()
    a = lattice.a
    A_c = lattice.cell_area
    modes = np.arange(-m_max, m_max + 1)
    n_modes = len(modes)
    l_values = np.arange(-2 * m_max, 2 * m_max + 1)   # shape (Nl,)
    orders = np.abs(l_values) + 1                      # shape (Nl,)

    Gx, Gy = lattice.reciprocal_vectors_xy(config.h_max)
    Qx = Gx + K[0]
    Qy = Gy + K[1]
    Qh = np.hypot(Qx, Qy)          # shape (Nh,)
    thetah = np.arctan2(Qy, Qx)    # shape (Nh,)
    denom_pole = Qh * (Qh**2 - kappa**2)  # shape (Nh,)
    near_zero = np.isclose(Qh, 0.0, atol=1e-12)

    # J_{orders}(Qh*a): shape (Nl, Nh) via broadcasting
    with np.errstate(divide="ignore", invalid="ignore"):
        Jvals = jv(orders[:, None], (Qh * a)[None, :])
        phase = np.exp(-1j * l_values[:, None] * thetah[None, :])  # (Nl,Nh)
        terms = Jvals * phase / denom_pole[None, :]

    if np.any(near_zero):
        # limite analitico en Qh->0 (ver chin_accelerated): a/2/(-kappa^2) si order==1, si no 0
        limit_order1 = (a / 2.0) / (-(kappa**2))
        limit_per_l = np.where(orders == 1, limit_order1, 0.0)  # (Nl,)
        terms = np.where(near_zero[None, :], limit_per_l[:, None], terms)

    terms = np.where(np.isfinite(terms), terms, 0.0)
    recip_sum = terms.sum(axis=1)  # shape (Nl,)

    prefactor = 1.0 / jv(orders, kappa * a)  # shape (Nl,)
    bracket = -4.0 * (1j ** (l_values + 1)) * kappa / A_c * recip_sum
    zero_idx = np.where(l_values == 0)[0]
    if zero_idx.size:
        bracket[zero_idx] += -(hankel1(1, kappa * a) + 2j / (np.pi * kappa * a))

    S_l_values = prefactor * bracket  # shape (Nl,), indexado por l_values

    # armar la matriz Toeplitz S_{mn} = S_{m-n}
    l_to_idx = {int(l): i for i, l in enumerate(l_values)}
    S = np.zeros((n_modes, n_modes), dtype=complex)
    for i, m in enumerate(modes):
        for j, n in enumerate(modes):
            S[i, j] = S_l_values[l_to_idx[int(m - n)]]
    return S


def validate_against_direct_sum(l: int, kappa_complex: complex, K: np.ndarray,
                                 lattice: SquareLattice,
                                 n_shells: int = 60, h_max: int = 40) -> dict:
    """Compara direct_sum vs chin_accelerated para un kappa con Im(kappa) > 0.
    Devuelve un dict con ambos valores y el error relativo -- pensado para
    usarse como chequeo de correccion del modulo (no es pytest, pero cumple
    la misma funcion mientras decides formalizar tests).
    """
    if kappa_complex.imag <= 0:
        raise ValueError("validate_against_direct_sum requiere Im(kappa) > 0 "
                          "para que la suma directa converja.")
    s_direct = direct_sum(l, kappa_complex, K, lattice, n_shells=n_shells)
    s_chin = chin_accelerated(l, kappa_complex, K, lattice, h_max=h_max)
    rel_err = abs(s_direct - s_chin) / max(abs(s_direct), 1e-30)
    return {"direct": s_direct, "chin": s_chin, "rel_err": rel_err}
