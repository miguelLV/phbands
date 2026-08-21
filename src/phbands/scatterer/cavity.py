"""Cavidad cilindrica libre (traccion nula), Sec. 6.1 del documento canonico.

    T_n^cav(omega) = - J_n'(kappa R) / H_n^{(1)'}(kappa R)

con kappa = kappa(omega) del medio ANFITRION (la cavidad es un hueco: no
tiene material propio, solo importa el radio R y las propiedades del
medio que la rodea).
"""

from __future__ import annotations

from ..basis import H1p, Jp
from ..media import HostMedium
from .base import Scatterer


class CavityScatterer(Scatterer):
    """Cavidad cilindrica de radio `radius`, embebida en `host`."""

    def __init__(self, radius: float, host: HostMedium):
        self.radius = radius
        self.host = host

    def T_n(self, omega: float, n: int) -> complex:
        kappa = self.host.kappa(omega)
        z = kappa * self.radius
        return -Jp(n, z) / H1p(n, z)

    def host_kappa(self, omega: float) -> complex:
        return self.host.kappa(omega)
