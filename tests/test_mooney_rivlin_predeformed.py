"""Validacion de MooneyRivlinPredeformedScatterer.

Referencias: derivacion_Tn_antiplano.md (Secs. 2.3-2.5) y tmatrix_antiplano.py
del proyecto. La prueba central (`test_gauge_reduction_matches_direct_equation`)
integra la ecuacion radial ORIGINAL, compleja y con la traccion cruzada
2 i n C1 gamma W / r, y la compara con el T_n que el dispersor obtiene de la
ecuacion real reducida: verifica de forma independiente el cambio de fase.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from phbands.basis import H1, H1p, J, Jp
from phbands.media import HostMedium
from phbands.scatterer import (
    CavityScatterer,
    MooneyRivlinPredeformedScatterer,
    SolidCylinderScatterer,
)

HOST = HostMedium(mu=1.0, rho=1.0)
INCLUSION = HostMedium(mu=3.0, rho=2.0)
A, B = 1.0, 1.5


def direct_T(n, omega, a, b, psi, beta, host, inclusion):
    """T_n resolviendo la ecuacion compleja sin reducir (sistema 4x4, Sec. 2.4)."""
    mu, rho = host.mu, host.rho
    C1 = (1 - beta) * mu / 2
    c = -2 * psi * a**2 * b**2 / (b**2 - a**2)
    gamma = lambda r: c / r**2
    dgamma = lambda r: -2 * c / r**3

    def rhs(r, y):
        W, dW = y
        g, dg = gamma(r), dgamma(r)
        d2W = (-dW / r + n * n * W / r**2 - rho * omega**2 * W / mu
               - (2 * C1 / mu) * (2j * n * g / r * dW + 1j * n * dg / r * W
                                  - n * n * g * g / r**2 * W))
        return [dW, d2W]

    def trac(r, W, dW):
        return mu * dW + 2j * n * C1 * gamma(r) * W / r

    ends = []
    for y0 in ([1, 0], [0, 1]):
        sol = solve_ivp(rhs, (a, b), np.array(y0, complex), method="DOP853",
                        rtol=1e-12, atol=1e-14)
        ends.append((np.array(y0, complex), sol.y[:, -1]))
    (Ua, Ub), (Va, Vb) = ends
    k, k1 = host.kappa(omega), inclusion.kappa(omega)
    x = k * b
    M = np.array([
        [J(n, k1 * a), -Ua[0], -Va[0], 0],
        [inclusion.mu * k1 * Jp(n, k1 * a), -trac(a, *Ua), -trac(a, *Va), 0],
        [0, Ub[0], Vb[0], -H1(n, x)],
        [0, trac(b, *Ub), trac(b, *Vb), -mu * k * H1p(n, x)],
    ], dtype=complex)
    rhs_vec = np.array([0, 0, J(n, x), mu * k * Jp(n, x)], dtype=complex)
    return np.linalg.solve(M, rhs_vec)[3]


@pytest.mark.parametrize("omega", [0.3, 1.0, 2.5, 5.0, 2.5 + 0.2j])
@pytest.mark.parametrize("n", [-3, -1, 1, 2, 4])
def test_gauge_reduction_matches_direct_equation(n, omega):
    s = MooneyRivlinPredeformedScatterer(A, B, 0.7, 0.3, HOST, INCLUSION)
    assert s.T_n(omega, n) == pytest.approx(
        direct_T(n, omega, A, B, 0.7, 0.3, HOST, INCLUSION), abs=1e-9)


def test_reference_value_from_derivation():
    # Sec. 2.4: mu=rho=a=1, mu1=3, rho1=2, b=1.5, C1=0.35, C2=0.15, psi=0.5, omega=2.5
    s = MooneyRivlinPredeformedScatterer(A, B, 0.5, 0.15 / 0.5, HOST, INCLUSION)
    assert s.T_n(2.5, 2) == pytest.approx(-0.2850 - 0.4514j, abs=1e-4)


@pytest.mark.parametrize("psi, beta", [(0.0, 0.5), (0.8, 0.0), (0.8, 1.0)])
def test_invariant_cases_reduce_to_unstrained(psi, beta):
    """psi=0, neo-Hookean (beta=0) y beta=1: la pre-deformacion es invisible."""
    solid = MooneyRivlinPredeformedScatterer(A, B, psi, beta, HOST, INCLUSION)
    cavity = MooneyRivlinPredeformedScatterer(A, B, psi, beta, HOST)
    for n in range(-3, 4):
        for omega in (0.5, 2.5):
            assert solid.T_n(omega, n) == pytest.approx(
                SolidCylinderScatterer(A, HOST, INCLUSION).T_n(omega, n), abs=1e-13)
            assert cavity.T_n(omega, n) == pytest.approx(
                CavityScatterer(A, HOST).T_n(omega, n), abs=1e-13)


def test_neo_hookean_invariance_is_not_a_shortcut():
    """beta pequeno pero no nulo debe tender continuamente al caso sin deformar."""
    s = MooneyRivlinPredeformedScatterer(A, B, 0.8, 1e-8, HOST)
    assert s.T_n(2.5, 2) == pytest.approx(CavityScatterer(A, HOST).T_n(2.5, 2), abs=1e-7)


def test_mode_zero_unchanged():
    s = MooneyRivlinPredeformedScatterer(A, B, 1.0, 0.5, HOST, INCLUSION)
    assert s.T_n(2.5, 0) == pytest.approx(
        SolidCylinderScatterer(A, HOST, INCLUSION).T_n(2.5, 0), abs=1e-14)


@pytest.mark.parametrize("inclusion", [None, INCLUSION])
def test_lossless_and_symmetries(inclusion):
    s = MooneyRivlinPredeformedScatterer(A, B, 1.0, 0.5, HOST, inclusion)
    mirror = MooneyRivlinPredeformedScatterer(A, B, -1.0, 0.5, HOST, inclusion)
    for n in range(1, 5):
        for omega in (0.3, 1.0, 2.5, 5.0):
            T = s.T_n(omega, n)
            assert abs(abs(1 + 2 * T) - 1) < 1e-10
            assert s.T_n(omega, -n) == pytest.approx(T, abs=1e-12)
            assert mirror.T_n(omega, n) == pytest.approx(T, abs=1e-10)
    for beta in (0.1, 0.3):
        p = MooneyRivlinPredeformedScatterer(A, B, 0.5, beta, HOST, inclusion)
        q = MooneyRivlinPredeformedScatterer(A, B, 0.5, 1 - beta, HOST, inclusion)
        assert p.T_n(2.5, 2) == pytest.approx(q.T_n(2.5, 2), abs=1e-10)


def test_small_twist_phase_shift_scaling():
    """Sec. 2.5: Delta delta_2 ~ -K_2 beta(1-beta) psi^2 con K_2 ~ 1.83
    (n=2, ka=2.5, b/a=1.5), independiente de beta."""
    phase = lambda T: np.angle(1 + 2 * T) / 2
    base = phase(SolidCylinderScatterer(A, HOST, INCLUSION).T_n(2.5, 2))
    for beta in (0.1, 0.3, 0.5):
        s = MooneyRivlinPredeformedScatterer(A, B, 0.05, beta, HOST, INCLUSION)
        K = -(phase(s.T_n(2.5, 2)) - base) / (0.05**2 * beta * (1 - beta))
        assert K == pytest.approx(1.826, abs=2e-3)


def test_rejects_bad_geometry():
    with pytest.raises(ValueError):
        MooneyRivlinPredeformedScatterer(1.0, 1.0, 0.3, 0.5, HOST)
    with pytest.raises(ValueError):
        MooneyRivlinPredeformedScatterer(1.0, 1.5, 0.3, 1.5, HOST)
