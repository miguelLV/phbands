"""Funciones base cilindricas para el problema antiplano.

Convencion temporal canonica: e^{-i*omega*t}  (Sec. 3 y Sec. 5 del
documento canonico). Bajo esta convencion:

  - Base REGULAR:   J_n(kappa*r) * e^{i*n*theta}        (Sec. 5.1)
  - Base SALIENTE:  H_n^{(1)}(kappa*r) * e^{i*n*theta}   (Sec. 5.2)

J_n NO es una onda puramente entrante (contiene H^(1) y H^(2) a partes
iguales); en MST "incidente" designa el campo local regular, no una
solucion puramente inward. Ver Sec. 5.1.
"""

from __future__ import annotations

import numpy as np
from scipy.special import h1vp, hankel1, jv, jvp


def J(n: int | np.ndarray, z: complex | np.ndarray) -> np.ndarray:
    """Funcion de Bessel de primer tipo, orden n (base regular)."""
    return jv(n, z)


def Jp(n: int | np.ndarray, z: complex | np.ndarray) -> np.ndarray:
    """Derivada de J_n respecto de su argumento completo: dJ_n(z)/dz."""
    return jvp(n, z)


def H1(n: int | np.ndarray, z: complex | np.ndarray) -> np.ndarray:
    """Funcion de Hankel de primer tipo, orden n (base saliente bajo
    e^{-i*omega*t})."""
    return hankel1(n, z)


def H1p(n: int | np.ndarray, z: complex | np.ndarray) -> np.ndarray:
    """Derivada de H_n^{(1)} respecto de su argumento completo."""
    return h1vp(n, z)


def mode_range(m_max: int) -> np.ndarray:
    """Indices modales enteros m in [-m_max, m_max], en el orden en que se
    ensamblaran las matrices T y S (Sec. 3: m,n indices modales enteros)."""
    return np.arange(-m_max, m_max + 1)
