"""Dispersion antiplana por una inclusion cilindrica solida isotropica."""

from __future__ import annotations

from ..basis import H1, H1p, J, Jp
from ..media import HostMedium
from .base import Scatterer


class SolidCylinderScatterer(Scatterer):
    """Cilindro solido unido perfectamente a un anfitrion solido.

    El coeficiente resulta de imponer continuidad de ``u_z`` y de
    ``mu * partial_r u_z`` en la interfaz. Las derivadas de Bessel son
    respecto de su argumento completo.
    """

    def __init__(self, radius: float, host: HostMedium, inclusion: HostMedium):
        self.radius = float(radius)
        self.host = host
        self.inclusion = inclusion

    def T_n(self, omega: float, n: int) -> complex:
        kappa_h = self.host.kappa(omega)
        kappa_i = self.inclusion.kappa(omega)
        x_h = kappa_h * self.radius
        x_i = kappa_i * self.radius
        numerator = (
            self.host.mu * kappa_h * Jp(n, x_h) * J(n, x_i)
            - self.inclusion.mu * kappa_i * Jp(n, x_i) * J(n, x_h)
        )
        denominator = (
            self.host.mu * kappa_h * H1p(n, x_h) * J(n, x_i)
            - self.inclusion.mu * kappa_i * Jp(n, x_i) * H1(n, x_h)
        )
        return -numerator / denominator

    def host_kappa(self, omega: float) -> complex:
        return self.host.kappa(omega)
