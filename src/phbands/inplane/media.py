"""Medio elastico homogeneo isotropico para ondas en el plano (tesis,
Sec. 2.1.4). A diferencia del caso antiplano (`phbands.media.HostMedium`,
que solo necesita mu y rho), aqui el desplazamiento se descompone en
potenciales longitudinal y transversal desacoplados (ec. 2.58-2.60), cada
uno con su propio numero de onda -- por eso se parametriza directamente
por (rho, c_l, c_t), tal como se dan los materiales en la tesis
(Sec. 2.2.1 / Fig. 2.4).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InPlaneMedium:
    """Medio elastico isotropico homogeneo, sin pre-esfuerzo."""

    rho: float
    c_l: float   # velocidad de onda longitudinal, k_l0 = omega*sqrt(rho/(2mu+lam))
    c_t: float   # velocidad de onda transversal, k_t0 = omega*sqrt(rho/mu)

    @property
    def mu(self) -> float:
        """Modulo de cizalle mu = rho * c_t^2."""
        return self.rho * self.c_t**2

    @property
    def lam(self) -> float:
        """Parametro de Lame lambda = rho*(c_l^2 - 2*c_t^2)."""
        return self.rho * (self.c_l**2 - 2.0 * self.c_t**2)

    def kappa_l(self, omega: float) -> complex:
        return omega / self.c_l

    def kappa_t(self, omega: float) -> complex:
        return omega / self.c_t
