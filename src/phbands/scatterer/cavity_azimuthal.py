"""Cavidad cilindrica con pre-deformacion azimutal de fondo, ecuacion de onda
truncada a orden D (la fuente D^2 de la ecuacion completa se descarta
deliberadamente; es una decision de alcance, no un descuido).

Derivacion fundamentada en la teoria "small-on-large" de Ogden, siguiendo las
convenciones exactas de Parnell (2012), "Nonlinear pre-stress for cloaking from
antiplane elastic waves", arXiv:1203.3246, Apendice A -- el mismo formalismo que
cita la tesis. La re-derivacion completa (con D y D^2) reproduce la Ec. 3.39 de
la tesis termino por termino, verificado simbolicamente.

Ver HANDOFF_azimuthal_predeformed_Tn_v2.md para la derivacion completa y los
valores de referencia usados en los tests.

NOTA HISTORICA -- convencion de indices
---------------------------------------
Una version anterior de este modulo (derivada en
docs/archivo/HANDOFF_azimuthal_predeformed_Tn_v1_OBSOLETO.md) usaba la
componente de traccion `zeta_zr` (fila z) en la condicion de borde. Esa fila da
CERO identicamente para cualquier F, de modo que aquel resultado carecia de
significado fisico pese a parecer autoconsistente. La componente correcta es
`zeta_rz` (primer indice = normal de la cara, segundo = direccion de la fuerza).
Ogden ademas define {dW/dF}_ij = dW/dF_ji, con la transpuesta ya incorporada.

Fisica modelada
---------------
Cavidad de radio r1; el anillo r1 < r < r2 quedo pre-deformado azimutalmente con
angulo de rotacion f(r) = D (r2^2/r^2 - 1), donde

    D = psi * r1^2 / (r2^2 - r1^2)

Mas alla de r2 el medio es homogeneo y sin deformar.

    (nabla^2 + k0^2) u = D * L1[u]
    L1[u] = 4 (r2^2/r^2) [ (1/r) d^2u/(dr dtheta) - (1/r^2) du/dtheta ]

El resultado `T_n(D)` es una FORMULA CERRADA: no hay expansion perturbativa en
D. Vale para cualquier D dentro del regimen donde truncar la fuente a O(D) siga
siendo razonable.

Consecuencia notable, util como test: `T_n(D) - T_n(0)` resulta ser O(D^2), no
O(D). La correccion de primer orden se anula exactamente. Esto concuerda con un
resultado independiente obtenido con la ecuacion completa via sustitucion exacta
de coordenadas, que da T_n invariante para todo D.

Convencion temporal e^{-i omega t}: `Hn` es siempre H_n^{(1)}.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import h1vp, hankel1, jv, jvp

from ..media import HostMedium
from .base import Scatterer


def _complex_quad(f, a, b, **kwargs):
    """Cuadratura para integrandos complejos: parte real e imaginaria por
    separado. `f` acepta un float y devuelve un complex."""
    re, _ = quad(lambda x: f(x).real, a, b, **kwargs)
    im, _ = quad(lambda x: f(x).imag, a, b, **kwargs)
    return re + 1j * im


class CavityAzimuthalPredeformedScatterer(Scatterer):
    """Cavidad de radio `r1` con anillo pre-deformado [r1, r2] y parametro de
    deformacion D = psi * r1**2/(r2**2 - r1**2). Ecuacion de onda truncada a
    O(D); formula cerrada en D (no perturbativa).
    """

    def __init__(self, r1: float, r2: float, psi: float, host: HostMedium,
                 quad_kwargs: dict | None = None):
        if not (0 < r1 < r2):
            raise ValueError("se requiere 0 < r1 < r2 (anillo de pre-deformacion no vacio)")
        self.r1 = float(r1)
        self.r2 = float(r2)
        self.psi = float(psi)
        self.D = self.psi * self.r1**2 / (self.r2**2 - self.r1**2)
        self.host = host
        self.quad_kwargs = quad_kwargs or {"limit": 200}
        # Memoizacion de las 4 cuadraturas del anillo; ver `_annulus_integrals`.
        self._integral_cache: dict[tuple[complex, int], tuple] = {}

    def clear_cache(self) -> None:
        """Vacia la memoizacion de las integrales del anillo."""
        self._integral_cache.clear()

    # ------------------------------------------------------------------
    # interfaz Scatterer
    # ------------------------------------------------------------------

    def host_kappa(self, omega: float) -> complex:
        return self.host.kappa(omega)

    # ------------------------------------------------------------------
    # piezas de la derivacion
    # ------------------------------------------------------------------

    @staticmethod
    def _g_b(k0: complex, n: int, rp: float) -> complex:
        """g_b(r') = d/dr'[ J_n(k0 r')/r' ] = k0 J_n'(k0 r')/r' - J_n(k0 r')/r'^2."""
        return k0 * jvp(n, k0 * rp) / rp - jv(n, k0 * rp) / rp**2

    @staticmethod
    def _g_c(k0: complex, n: int, rp: float) -> complex:
        """g_c(r') = d/dr'[ H_n(k0 r')/r' ] = k0 H_n'(k0 r')/r' - H_n(k0 r')/r'^2."""
        return k0 * h1vp(n, k0 * rp) / rp - hankel1(n, k0 * rp) / rp**2

    def _V_integrals(self, k0: complex, n: int, lower: float):
        """Las cuatro integrales de variacion de parametros desde `lower` a r2:

            V_J^(b)(r) = int_[r,r2] J_n(k0 r') g_b(r')/r' dr'
            V_J^(c)(r) = int_[r,r2] J_n(k0 r') g_c(r')/r' dr'
            V_H^(b)(r) = int_[r,r2] H_n(k0 r') g_b(r')/r' dr'
            V_H^(c)(r) = int_[r,r2] H_n(k0 r') g_c(r')/r' dr'

        Devuelve (VJ_b, VJ_c, VH_b, VH_c). Sin memoizar: `lower` es variable
        (lo usa la verificacion de la EDO). El caso lower=r1, que es el unico
        que necesitan las condiciones de borde, se cachea en
        `_annulus_integrals`.
        """
        kw = self.quad_kwargs
        VJ_b = _complex_quad(lambda rp: jv(n, k0 * rp) * self._g_b(k0, n, rp) / rp,
                             lower, self.r2, **kw)
        VJ_c = _complex_quad(lambda rp: jv(n, k0 * rp) * self._g_c(k0, n, rp) / rp,
                             lower, self.r2, **kw)
        VH_b = _complex_quad(lambda rp: hankel1(n, k0 * rp) * self._g_b(k0, n, rp) / rp,
                             lower, self.r2, **kw)
        VH_c = _complex_quad(lambda rp: hankel1(n, k0 * rp) * self._g_c(k0, n, rp) / rp,
                             lower, self.r2, **kw)
        return VJ_b, VJ_c, VH_b, VH_c

    def _annulus_integrals(self, k0: complex, n: int):
        """`_V_integrals` evaluadas en el borde interno (lower = r1), memoizadas.

        Clave `(k0, |n|)`:

        * dependen solo de (k0, n), NO del vector de Bloch K. En un barrido de
          bandas el escaneo global recorre la misma malla de omega en cada punto
          k, asi que el mismo (k0, n) se vuelve a pedir una vez por punto k: el
          acierto de cache es muy alto.
        * son PARES en n: J_{-n} = (-1)^n J_n y H_{-n} = (-1)^n H_n aparecen en
          ambos factores de cada integrando y los signos se cancelan.

        La dependencia impar en n de la correccion entra por los factores
        explicitos `n` del prefactor y de los terminos de traccion.
        """
        key = (complex(k0), abs(int(n)))
        cached = self._integral_cache.get(key)
        if cached is not None:
            return cached
        result = self._V_integrals(k0, n, self.r1)
        self._integral_cache[key] = result
        return result

    def particular_solution(self, k0: complex, n: int, r: float,
                            bn: complex, cn: complex) -> complex:
        """Solucion particular R1(r) de la EDO radial, sin el factor D.

            R1(r) = 2 pi n r2^2 [ J_n(k0 r) V_H(r) - H_n(k0 r) V_J(r) ]

        con V_J, V_H construidas a partir de sigma = bn*g_b + cn*g_c. El ORDEN
        de los terminos importa: `Jn*V_H - Hn*V_J`, no al reves (la version con
        el signo cambiado falla la verificacion contra la EDO por un factor -1).

        Expuesto publicamente porque es lo que verifica el checklist #3
        sustituyendo directamente en
            R1'' + R1'/r + (k0^2 - n^2/r^2) R1 = S_n(r).
        """
        VJ_b, VJ_c, VH_b, VH_c = self._V_integrals(k0, n, r)
        V_J = bn * VJ_b + cn * VJ_c
        V_H = bn * VH_b + cn * VH_c
        return 2 * np.pi * n * self.r2**2 * (jv(n, k0 * r) * V_H - hankel1(n, k0 * r) * V_J)

    def source_term(self, k0: complex, n: int, r: float,
                    bn: complex, cn: complex) -> complex:
        """Lado derecho de la EDO radial:

            S_n(r) = (4 i n r2^2 / r^2) sigma(r)
            sigma(r) = d/dr[ (bn J_n(k0 r) + cn H_n(k0 r)) / r ]
                     = bn g_b(r) + cn g_c(r)
        """
        sigma = bn * self._g_b(k0, n, r) + cn * self._g_c(k0, n, r)
        return (4j * n * self.r2**2 / r**2) * sigma

    # ------------------------------------------------------------------
    # coeficiente de dispersion
    # ------------------------------------------------------------------

    def T_n(self, omega: float, n: int) -> complex:
        """T_n(D), forma cerrada.

        Pasos (ver HANDOFF v2):
          * Psi_b, Psi_c: solucion completa en el anillo. En r=r2 las integrales
            valen cero por construccion (van de r2 a r2), asi que Psi(r2) y
            Psi'(r2) NO dependen de D.
          * Phi: traccion zeta_rz/mu = du/dr + s(r)(1/r) du0/dtheta, con
            s(r) = r f'(r) = -2 D r2^2/r^2. El acoplamiento usa u0, coherente
            con truncar a O(D).
          * BC1 en r1 (traccion nula) fija rho = -Phi_b(r1)/Phi_c(r1); BC2 y BC3
            en r2 (continuidad de u y de la traccion) se eliminan para dar T_n.
        """
        k0 = self.host_kappa(omega)
        r1, r2, D = self.r1, self.r2, self.D

        if D == 0:
            # cavidad libre: identico a CavityScatterer
            z1 = k0 * r1
            return -jvp(n, z1) / h1vp(n, z1)

        VJ_b, VJ_c, VH_b, VH_c = self._annulus_integrals(k0, n)

        z1, z2 = k0 * r1, k0 * r2
        Jn_r1, Hn_r1 = jv(n, z1), hankel1(n, z1)
        Jn_r2, Hn_r2 = jv(n, z2), hankel1(n, z2)
        Jnp_r1, Hnp_r1 = jvp(n, z1), h1vp(n, z1)
        Jnp_r2, Hnp_r2 = jvp(n, z2), h1vp(n, z2)

        pref = 2 * np.pi * n * r2**2

        # Psi'(r1) (Leibniz, con la cancelacion exacta de los terminos en g(r1))
        # menos el acoplamiento de traccion en r1.
        Phi_b_r1 = (k0 * Jnp_r1 + D * pref * (k0 * Jnp_r1 * VH_b - k0 * Hnp_r1 * VJ_b)
                    - (2 * D * 1j * n * r2**2 / r1**3) * Jn_r1)
        Phi_c_r1 = (k0 * Hnp_r1 + D * pref * (k0 * Jnp_r1 * VH_c - k0 * Hnp_r1 * VJ_c)
                    - (2 * D * 1j * n * r2**2 / r1**3) * Hn_r1)

        # En r2 las integrales se anulan; solo queda el acoplamiento de traccion.
        Phi_b_r2 = k0 * Jnp_r2 - (2 * D * 1j * n / r2) * Jn_r2
        Phi_c_r2 = k0 * Hnp_r2 - (2 * D * 1j * n / r2) * Hn_r2

        rho = -Phi_b_r1 / Phi_c_r1
        A = Jn_r2 + rho * Hn_r2
        B = Phi_b_r2 + rho * Phi_c_r2

        return (A * k0 * Jnp_r2 - B * Jn_r2) / (B * Hn_r2 - A * k0 * Hnp_r2)
