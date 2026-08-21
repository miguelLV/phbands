"""Propiedades del medio anfitrion para el problema antiplano.

Ver Sec. 2 y Sec. 3 del documento canonico: para polarizacion antiplana
solo importan la densidad `rho` y el modulo de corte `mu` del material.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HostMedium:
    """Medio anfitrion homogeneo, isotropico, sin preesfuerzo."""

    mu: float    # modulo de corte
    rho: float   # densidad

    @property
    def c_t(self) -> float:
        """Velocidad de onda de corte c_t = sqrt(mu/rho) (Sec. 3)."""
        return float(np.sqrt(self.mu / self.rho))

    def kappa(self, omega: float) -> float:
        """Numero de onda fisico kappa = omega/c_t = omega*sqrt(rho/mu) (Sec. 3)."""
        return omega / self.c_t
