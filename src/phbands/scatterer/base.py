"""Interfaz abstracta para la respuesta local del dispersor T (Sec. 6 y 11
del documento canonico):

    b = T a

`T` codifica exclusivamente la fisica local (condiciones de borde,
materiales, interfaces); NUNCA la periodicidad de la red (eso es S). Esta
separacion es la que permite, mas adelante, sustituir la cavidad libre
(v0.1) por una inclusion elastica o una configuracion predeformada sin
tocar el resto del solver (lattice.py, lattice_sums.py, secular.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Scatterer(ABC):
    """Respuesta local abstracta del dispersor para el problema antiplano.

    Para un dispersor circular sin mezcla modal (el unico caso auditado en
    v0.1), la respuesta se reduce a b_n = T_n(omega) a_n para cada modo n
    por separado (matriz T diagonal).
    """

    @abstractmethod
    def T_n(self, omega: float, n: int) -> complex:
        """Coeficiente de dispersion T_n(omega) para el modo entero n."""
        raise NotImplementedError

    @abstractmethod
    def host_kappa(self, omega: float) -> complex:
        """Numero de onda kappa(omega) del medio ANFITRION (exterior).

        La suma de red S siempre se evalua con el kappa del medio exterior
        (independientemente de que material tenga el dispersor por dentro),
        porque S describe la propagacion entre centros a traves del
        anfitrion. Cada Scatterer concreto debe exponer este valor."""
        raise NotImplementedError

    def T_diag(self, omega: float, m_max: int) -> np.ndarray:
        """Vector con T_n(omega) para n en [-m_max, m_max], en el mismo
        orden de modos usado por `lattice_sums.S_matrix` y `basis.mode_range`.
        """
        modes = np.arange(-m_max, m_max + 1)
        return np.array([self.T_n(omega, int(n)) for n in modes], dtype=complex)
