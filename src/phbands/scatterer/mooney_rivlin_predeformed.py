"""Dispersor antiplano con capa pre-deformada azimutalmente, Mooney-Rivlin
incompresible, SIN truncar en la pre-deformacion.

Geometria
---------
* ``r < a``: inclusion elastica sin deformar (``inclusion``), o cavidad si
  ``inclusion is None``.
* ``a < r < b``: la propia matriz (``host``), pre-deformada por el corte
  azimutal r = R, theta = Theta + f(R), z = Z con el perfil de equilibrio de
  Mooney-Rivlin

      f(r) = psi a^2/(b^2 - a^2) (b^2/r^2 - 1),   f(a) = psi,  f(b) = 0,
      gamma(r) = r f'(r) = c / r^2,   c = -2 psi a^2 b^2/(b^2 - a^2).

  Es el mismo perfil que usa ``CavityAzimuthalPredeformedScatterer`` (con
  r1 = a, r2 = b, D = psi a^2/(b^2 - a^2)).
* ``r > b``: matriz sin deformar.

Fisica (derivacion completa en derivacion_Tn_antiplano.md, Secs. 2.2-2.4)
-------------------------------------------------------------------------
W = C1 (I1 - 3) + C2 (I2 - 3), mu = 2 (C1 + C2), beta = C2/(C1 + C2).
La ecuacion incremental de Ogden (small-on-large) para u = w e_z es exacta en
la pre-deformacion:

    div(Q grad w) = rho w_tt,   Q = 2 C1 B + 2 C2 I,

con traccion sobre r = cte  S0dot_rz = mu W' + 2 i n C1 gamma W / r  (primer
indice = normal). El cambio de fase W = exp(i chi) Wt, chi = -n (1-beta) f(r),
la reduce a una ecuacion REAL (Sec. 2.3.1):

    Wt'' + Wt'/r + [k^2 - n^2/r^2 - q_n/r^6] Wt = 0,
    q_n = beta (1 - beta) n^2 c^2,

y la traccion a  S0dot_rz = exp(i chi) mu Wt'. La fase es comun a w y a la
traccion, asi que se cancela en las condiciones de continuidad: el sistema es
el del cilindro recubierto con Wt en la capa. Aqui Wt se integra numericamente
(no hay forma cerrada en funciones estandar para el perfil de equilibrio).

Propiedades exactas que sirven de test: T_0 y el caso neo-Hookean (beta = 0) o
beta = 1 no cambian respecto del caso sin deformar; T_{-n} = T_n; T es par en
psi y simetrico en beta <-> 1 - beta; |1 + 2 T_n| = 1 (sin perdidas).

Convencion temporal e^{-i omega t}; H_n = H_n^{(1)}; b_n = T_n a_n.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ..basis import H1, H1p, J, Jp
from ..media import HostMedium
from .base import Scatterer


class MooneyRivlinPredeformedScatterer(Scatterer):
    """Inclusion (o cavidad) de radio `a` con capa de matriz pre-deformada
    hasta `b`. `psi` es el giro del borde interno; `beta = C2/(C1+C2)`.
    """

    def __init__(self, a: float, b: float, psi: float, beta: float,
                 host: HostMedium, inclusion: HostMedium | None = None,
                 rtol: float = 1e-11, atol: float = 1e-14):
        if not 0 < a < b:
            raise ValueError("se requiere 0 < a < b (capa pre-deformada no vacia)")
        if not 0 <= beta <= 1:
            raise ValueError("beta = C2/(C1+C2) debe estar en [0, 1]")
        self.a = float(a)
        self.b = float(b)
        self.psi = float(psi)
        self.beta = float(beta)
        self.host = host
        self.inclusion = inclusion
        self.rtol = rtol
        self.atol = atol
        self.c = -2 * self.psi * self.a**2 * self.b**2 / (self.b**2 - self.a**2)
        # T_n depende de n solo via n^2; clave (omega, |n|).
        self._cache: dict[tuple[complex, int], complex] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def host_kappa(self, omega: float) -> complex:
        return self.host.kappa(omega)

    def q_n(self, n: int) -> float:
        """Intensidad del potencial centrifugo extra P_n = q_n / r^6."""
        return self.beta * (1 - self.beta) * n**2 * self.c**2

    def _layer_impedance(self, omega, n: int):
        """Integra Wt de a a b y devuelve (Wt(b), mu Wt'(b)).

        La condicion inicial en r = a es la que impone el nucleo, salvo un
        factor comun: cavidad -> traccion nula; inclusion -> (J_n, mu1 k1 J_n'/mu).
        """
        k = self.host_kappa(omega)
        mu = self.host.mu
        q = self.q_n(n)
        if self.inclusion is None:
            y0 = np.array([1.0, 0.0], dtype=complex)
        else:
            k1 = self.inclusion.kappa(omega)
            x1 = k1 * self.a
            y0 = np.array([J(n, x1), self.inclusion.mu * k1 * Jp(n, x1) / mu], dtype=complex)
        k2 = k * k

        def rhs(r, y):
            W, dW = y
            return [dW, -dW / r - (k2 - n * n / r**2 - q / r**6) * W]

        sol = solve_ivp(rhs, (self.a, self.b), y0, method="DOP853",
                        rtol=self.rtol, atol=self.atol)
        if not sol.success:
            raise RuntimeError(f"fallo la integracion de la capa: {sol.message}")
        Wb, dWb = sol.y[:, -1]
        return Wb, mu * dWb

    def _T_unstrained(self, omega, n: int) -> complex:
        k = self.host_kappa(omega)
        x = k * self.a
        if self.inclusion is None:
            return -Jp(n, x) / H1p(n, x)
        mu, mu1 = self.host.mu, self.inclusion.mu
        k1 = self.inclusion.kappa(omega)
        x1 = k1 * self.a
        num = mu * k * Jp(n, x) * J(n, x1) - mu1 * k1 * Jp(n, x1) * J(n, x)
        den = mu * k * H1p(n, x) * J(n, x1) - mu1 * k1 * Jp(n, x1) * H1(n, x)
        return -num / den

    def T_n(self, omega: float, n: int) -> complex:
        n = int(n)
        if n == 0 or self.q_n(n) == 0:
            # n = 0, psi = 0, neo-Hookean (beta=0) o beta=1: invariancia exacta.
            return complex(self._T_unstrained(omega, n))
        key = (complex(omega), abs(n))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        k = self.host_kappa(omega)
        mu = self.host.mu
        Wb, tb = self._layer_impedance(omega, n)
        x = k * self.b
        # Continuidad en r = b con J_n(kr) + T_n H_n(kr):
        #   Wb * s = J + T H,   tb * s = mu k (J' + T H')
        T = (tb * J(n, x) - mu * k * Jp(n, x) * Wb) / (mu * k * H1p(n, x) * Wb - tb * H1(n, x))
        T = complex(T)
        self._cache[key] = T
        return T
