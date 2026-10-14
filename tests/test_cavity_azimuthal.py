"""Checklist de validacion del T_n de cavidad con pre-deformacion azimutal.

Cubre los 5 puntos del checklist de HANDOFF_azimuthal_predeformed_Tn_v2.md.

Este archivo reemplaza por completo a la version anterior: aquella validaba una
derivacion con un error de convencion de indices tensoriales (usaba `zeta_zr`,
la fila z, que da cero identicamente). Ver
docs/archivo/HANDOFF_azimuthal_predeformed_Tn_v1_OBSOLETO.md.

Diferencia estructural importante respecto de aquella version: aqui `T_n(D)` es
una FORMULA CERRADA en D, no una expansion perturbativa, y la correccion de
primer orden en D resulta ser exactamente CERO (la primera correccion no nula es
O(D^2)). Los tests estan planteados alrededor de ese hecho.
"""

from __future__ import annotations

import numpy as np
import pytest

from phbands.media import HostMedium
from phbands.scatterer import CavityScatterer
from phbands.scatterer.cavity_azimuthal import CavityAzimuthalPredeformedScatterer


# Caso de referencia del HANDOFF v2: k0=1.3, r1=0.7, r2=1.4, n=2.
# Con mu=rho=1 se tiene c_t=1 y por tanto kappa(omega)=omega, asi que se puede
# fijar k0 pasando omega=k0 directamente.
K0_REF = 1.3
R1_REF, R2_REF = 0.7, 1.4
N_REF = 2

# T_n(D) tabulado en el HANDOFF v2
REFERENCIA_TN = {
    0.00: -0.0034799979695563678 + 0.058888772985079614j,
    0.01: -0.0035631240 + 0.0593495026j,
    0.10: -0.0317426871 + 0.0900141857j,
    0.35: -0.2149027253 + 0.0232286627j,
}


def unit_host() -> HostMedium:
    """Medio con c_t = 1, de modo que kappa(omega) = omega."""
    return HostMedium(mu=1.0, rho=1.0)


def psi_para(D: float, r1: float = R1_REF, r2: float = R2_REF) -> float:
    """Invierte D = psi r1^2/(r2^2-r1^2) para poder fijar D en los tests."""
    return D * (r2**2 - r1**2) / r1**2


def scatterer_con_D(D: float, **kwargs) -> CavityAzimuthalPredeformedScatterer:
    return CavityAzimuthalPredeformedScatterer(
        r1=R1_REF, r2=R2_REF, psi=psi_para(D), host=unit_host(), **kwargs
    )


# ---------------------------------------------------------------------------
# 1. D = 0 debe reducirse EXACTAMENTE a la cavidad libre
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("omega", [0.3, 1.3, 2.7, 5.0])
@pytest.mark.parametrize("n", [-3, -1, 0, 1, 2, 5])
def test_sin_deformacion_coincide_con_cavidad_libre(omega, n):
    """Con psi=0 (luego D=0) el dispersor debe degenerar en la cavidad libre.

    Es el camino corto de `T_n`, asi que este test cubre tanto `_Tn0` implicito
    como el corto-circuito `if D == 0`.
    """
    host = unit_host()
    deformado = CavityAzimuthalPredeformedScatterer(
        r1=R1_REF, r2=R2_REF, psi=0.0, host=host
    )
    libre = CavityScatterer(radius=R1_REF, host=host)
    assert deformado.D == 0.0
    assert deformado.T_n(omega, n) == pytest.approx(libre.T_n(omega, n), abs=1e-14)


def test_modo_n_cero_no_se_ve_afectado_por_la_deformacion():
    """La fuente L1 es proporcional a n (va como d/dtheta), asi que el modo n=0
    debe dar el resultado sin deformar aunque D != 0.

    No es un caso especial del codigo -- sale solo de la formula cerrada, lo que
    lo hace un buen test de que los prefactores en `n` estan en su sitio."""
    host = unit_host()
    deformado = scatterer_con_D(0.35)
    libre = CavityScatterer(radius=R1_REF, host=host)
    for omega in (0.7, 1.3, 3.1):
        assert deformado.T_n(omega, 0) == pytest.approx(libre.T_n(omega, 0), abs=1e-14)


# ---------------------------------------------------------------------------
# 2. Valores de referencia del HANDOFF v2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("D,esperado", sorted(REFERENCIA_TN.items()))
def test_reproduce_valores_de_referencia(D, esperado):
    """Los 4 valores tabulados de T_n(D) para k0=1.3, r1=0.7, r2=1.4, n=2.

    Son el ancla de la traduccion a codigo: si salen los cuatro, la formula
    cerrada esta transcrita correctamente."""
    s = scatterer_con_D(D)
    assert s.T_n(K0_REF, N_REF) == pytest.approx(esperado, abs=1e-6)


def test_D_se_calcula_segun_el_handoff():
    """D = psi * r1^2 / (r2^2 - r1^2)."""
    s = CavityAzimuthalPredeformedScatterer(
        r1=R1_REF, r2=R2_REF, psi=0.3, host=unit_host()
    )
    assert s.D == pytest.approx(0.3 * R1_REF**2 / (R2_REF**2 - R1_REF**2))


# ---------------------------------------------------------------------------
# 3. Verificacion INDEPENDIENTE de la solucion particular contra la EDO
# ---------------------------------------------------------------------------


def _residuo_edo(s, k0, n, r, bn, cn, h=1e-3):
    """Sustituye R1 en la EDO radial y devuelve (lhs, rhs).

        R1'' + R1'/r + (k0^2 - n^2/r^2) R1 = S_n(r)

    Las derivadas se toman por diferencias finitas centradas, de modo que esta
    comprobacion NO usa nada del resto de la derivacion: ni las condiciones de
    borde, ni el sistema 3x3, ni la formula final de T_n. Es una prueba pura de
    que la variacion de parametros esta bien implementada.
    """
    R = lambda x: s.particular_solution(k0, n, x, bn, cn)
    R_menos, R_centro, R_mas = R(r - h), R(r), R(r + h)
    primera = (R_mas - R_menos) / (2 * h)
    segunda = (R_mas - 2 * R_centro + R_menos) / h**2
    lhs = segunda + primera / r + (k0**2 - n**2 / r**2) * R_centro
    rhs = s.source_term(k0, n, r, bn, cn)
    return lhs, rhs


@pytest.mark.parametrize("r", [0.80, 0.95, 1.10, 1.25])
def test_solucion_particular_satisface_la_edo(r):
    """Checklist #3, el test mas importante segun el HANDOFF.

    Reconstruye R1(r) explicitamente dentro del anillo y confirma por
    diferencias finitas que resuelve la EDO radial. Tolerancia pedida: 1e-3
    relativo (se alcanza ~1e-6 con cuadratura ajustada).
    """
    s = CavityAzimuthalPredeformedScatterer(
        r1=R1_REF, r2=R2_REF, psi=0.1, host=unit_host(),
        quad_kwargs={"limit": 400, "epsabs": 1e-13, "epsrel": 1e-13},
    )
    lhs, rhs = _residuo_edo(s, K0_REF, N_REF, r, bn=1.0, cn=0.3 + 0.2j)
    assert abs(lhs - rhs) / abs(rhs) < 1e-3


def test_el_orden_de_los_terminos_de_variacion_de_parametros_es_el_correcto():
    """El HANDOFF avisa: la formula es `Jn*V_H - Hn*V_J`, en ese orden.

    Con el signo invertido la solucion particular resuelve la EDO con un factor
    exacto de -1, asi que basta exigir que el cociente lhs/rhs sea +1 (y no
    simplemente que su modulo valga 1) para detectar la inversion.
    """
    s = CavityAzimuthalPredeformedScatterer(
        r1=R1_REF, r2=R2_REF, psi=0.1, host=unit_host(),
        quad_kwargs={"limit": 400, "epsabs": 1e-13, "epsrel": 1e-13},
    )
    lhs, rhs = _residuo_edo(s, K0_REF, N_REF, 0.95, bn=1.0, cn=0.3 + 0.2j)
    cociente = lhs / rhs
    assert cociente.real == pytest.approx(1.0, abs=1e-3)
    assert cociente.imag == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 4. Firma O(D^2): la pendiente en D=0 es CERO
# ---------------------------------------------------------------------------


def test_correccion_es_orden_D_cuadrado_no_orden_D():
    """Checklist #4.

    `(T_n(D) - T_n(0))/D` NO converge a una constante no nula cuando D -> 0: se
    reduce a la mitad cada vez que D se reduce a la mitad. Es decir, la
    correccion de primer orden en D se anula exactamente y la primera no nula es
    O(D^2).

    Este comportamiento concuerda con un resultado independiente obtenido con la
    ecuacion completa (con D y D^2) mediante sustitucion exacta de coordenadas,
    que da T_n exactamente invariante en D. Un test que asumiera convergencia a
    una constante no nula estaria mal planteado.
    """
    T0 = scatterer_con_D(0.0).T_n(K0_REF, N_REF)

    def cociente(D: float) -> complex:
        return (scatterer_con_D(D).T_n(K0_REF, N_REF) - T0) / D

    valores = [cociente(D) for D in (0.01, 0.005, 0.0025, 0.00125)]

    # cada paso divide el cociente por ~2 (firma de O(D^2))
    for grande, pequeno in zip(valores, valores[1:]):
        assert abs(grande) / abs(pequeno) == pytest.approx(2.0, abs=0.15)

    # y explicitamente: NO tiende a una constante no nula
    assert abs(valores[-1]) < abs(valores[0]) / 4

    # valores tabulados en el HANDOFF (parte imaginaria, que es la dominante)
    assert valores[0].imag == pytest.approx(0.046, abs=2e-3)
    assert valores[1].imag == pytest.approx(0.023, abs=2e-3)
    assert valores[2].imag == pytest.approx(0.012, abs=2e-3)


def test_unitariedad_exacta_en_D_cero_y_violacion_cubica():
    """Un dispersor sin perdidas cumple |1 + 2 T_n| = 1.

    La pre-deformacion es estatica y no introduce perdidas, asi que la relacion
    deberia mantenerse; lo que la rompe es haber truncado la fuente a O(D). Sirve
    entonces de termometro independiente del rango de validez.

    El defecto resulta ser O(D^3), no O(D^2). Medido: al duplicar D crece por
    factores 7.29, 7.82, 7.96, 7.99 (exponentes 2.87, 2.97, 2.99, 3.00). La
    razon es que, escribiendo T = T0 + a*D^2 + ..., la contribucion de primer
    orden al modulo es 2*Re[(1+2T0)^* * 2a*D^2]; que el defecto sea cubico
    significa que ese termino se anula, es decir que la correccion O(D^2) es
    TANGENTE al circulo de unitariedad. Es una consistencia fisica no trivial de
    la derivacion, mas fuerte que el mero O(D^2) que uno esperaria a priori.
    """
    def violacion(D: float, n: int = N_REF) -> float:
        T = scatterer_con_D(D).T_n(K0_REF, n)
        return abs(abs(1.0 + 2.0 * T) - 1.0)

    assert violacion(0.0) < 1e-14

    # al duplicar D el defecto crece ~8x (cubico); se mide el exponente para no
    # depender de un solo par de puntos
    for D in (0.005, 0.01, 0.02):
        exponente = np.log2(violacion(2 * D) / violacion(D))
        assert exponente == pytest.approx(3.0, abs=0.2)


# ---------------------------------------------------------------------------
# 5. Sanidad de signos y del indice modal n
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("D", [0.1, 0.3])
def test_simetria_de_quiralidad(n, D):
    """Invertir el sentido de la rotacion equivale a reflejar theta, luego

        T_{-n}(D) = T_n(-D)

    Se cumple a precision de maquina. Es un invariante estructural exigente: lo
    rompe cualquier error de signo o de precedencia al escribir potencias de la
    unidad imaginaria (recordar el bug historico `4j**(l+1)`, que en Python es
    `(4j)**(l+1)` y no `4*(1j**(l+1))`).
    """
    directo = scatterer_con_D(D).T_n(K0_REF, -n)
    reflejado = scatterer_con_D(-D).T_n(K0_REF, n)
    assert directo == pytest.approx(reflejado, rel=1e-12)


def test_los_modos_mas_y_menos_n_dejan_de_ser_degenerados():
    """Sin deformar, T_n es par en n. La pre-deformacion azimutal rompe la
    simetria theta -> -theta, asi que T_n y T_{-n} deben diferir para D != 0.

    Es la firma fisica de la quiralidad del sistema; si coincidieran, la
    deformacion no estaria entrando en la solucion."""
    libre = CavityScatterer(radius=R1_REF, host=unit_host())
    assert libre.T_n(K0_REF, 2) == pytest.approx(libre.T_n(K0_REF, -2), rel=1e-12)

    s = scatterer_con_D(0.1)
    assert abs(s.T_n(K0_REF, 2) - s.T_n(K0_REF, -2)) > 1e-4


@pytest.mark.parametrize("n", [-4, -1, 0, 2, 6])
def test_valores_finitos_para_n_positivo_y_negativo(n):
    """No deben aparecer NaN/inf en ningun modo, de ningun signo."""
    s = scatterer_con_D(0.2)
    assert np.isfinite(s.T_n(K0_REF, n))


def test_T_diag_ordena_los_modos_como_el_resto_del_solver():
    """`T_diag` (heredado de Scatterer) entrega n = -m_max..m_max en el mismo
    orden que usa `lattice_sums.S_matrix`."""
    s = scatterer_con_D(0.2)
    m_max = 3
    diag = s.T_diag(K0_REF, m_max)
    assert len(diag) == 2 * m_max + 1
    for i, n in enumerate(range(-m_max, m_max + 1)):
        assert diag[i] == pytest.approx(s.T_n(K0_REF, n), rel=1e-12)


# ---------------------------------------------------------------------------
# extras: memoizacion y validacion de geometria
# ---------------------------------------------------------------------------


def test_las_integrales_del_anillo_son_pares_en_n():
    """Justifica que la memoizacion pueda usar |n| como clave: J_{-n} y H_{-n}
    aportan (-1)^n en ambos factores de cada integrando y los signos se
    cancelan."""
    s = scatterer_con_D(0.1)
    for n in (1, 2, 3):
        positivos = s._V_integrals(K0_REF, n, R1_REF)
        negativos = s._V_integrals(K0_REF, -n, R1_REF)
        for a, b in zip(positivos, negativos):
            assert a == pytest.approx(b, rel=1e-12)


def test_la_memoizacion_no_altera_el_resultado():
    """La segunda llamada sale de cache y debe ser identica a la primera."""
    s = scatterer_con_D(0.2)
    frio = s.T_n(K0_REF, N_REF)
    cacheado = s.T_n(K0_REF, N_REF)
    assert frio == cacheado
    s.clear_cache()
    assert s.T_n(K0_REF, N_REF) == pytest.approx(frio, rel=1e-12)


def test_geometria_invalida_es_rechazada():
    """El anillo de pre-deformacion debe ser no vacio: 0 < r1 < r2."""
    host = unit_host()
    for r1, r2 in [(1.4, 0.7), (0.7, 0.7), (-0.3, 1.0)]:
        with pytest.raises(ValueError):
            CavityAzimuthalPredeformedScatterer(r1=r1, r2=r2, psi=0.1, host=host)
