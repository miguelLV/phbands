"""Red con base (panal tipo grafeno): sumas cruzadas y bandas contra ondas planas."""

import numpy as np
import pytest
import scipy.linalg as sla
from scipy.optimize import minimize_scalar
from scipy.special import j1

from phbands.lattice import HexagonalLattice
from phbands.lattice_sums import LatticeSumConfig, S_matrix, cross_direct_sum
from phbands.media import HostMedium
from phbands.scatterer import SolidCylinderScatterer
from phbands.secular import BasisSecularProblem, SecularProblem


A = np.sqrt(3.0)  # distancia entre vecinos = 1
LATTICE = HexagonalLattice(A)
D_A, D_B = np.array([0.0, 0.5]), np.array([0.0, -0.5])
K_POINT = np.array([4.0 * np.pi / (3.0 * A), 0.0])
M_POINT = np.array([np.pi / A, np.pi / (A * np.sqrt(3.0))])


@pytest.mark.parametrize("K", [np.array([0.3, 0.5]), K_POINT])
@pytest.mark.parametrize("gauge", [(), (0.45,)])
def test_cross_sum_matches_direct_sum(K, gauge):
    kappa, m_max = 2.0 + 0.4j, 3
    S = S_matrix(kappa, K, LATTICE, m_max,
                 LatticeSumConfig(h_max=60, gauge_alternatives_over_a=gauge), offset=D_A - D_B)
    for i, m in enumerate(range(-m_max, m_max + 1)):
        direct = cross_direct_sum(m, kappa, K, LATTICE, D_A - D_B, n_shells=45)
        assert abs(S[i, m_max] - direct) <= 1e-5 * max(1.0, abs(direct))


def test_single_scatterer_basis_matches_secular_problem():
    host = HostMedium(mu=1.0, rho=1.0)
    sc = SolidCylinderScatterer(0.3, host, HostMedium(mu=2.0, rho=1.5))
    cfg = LatticeSumConfig(h_max=10, gauge_alternatives_over_a=(0.45,))
    one = SecularProblem(LATTICE, sc, 3, cfg, symmetrized=True)
    basis = BasisSecularProblem(LATTICE, (sc,), ((0.0, 0.0),), 3, cfg, symmetrized=True)
    K = np.array([0.4, 0.2])
    np.testing.assert_allclose(basis.M(3.0, K), one.M(3.0, K), atol=1e-13)


def _pwe(K, radius, inclusions, h_max=13, n_bands=4):
    Gx, Gy = LATTICE.reciprocal_vectors_xy(h_max)
    dGx, dGy = Gx[:, None] - Gx[None, :], Gy[:, None] - Gy[None, :]
    dG = np.hypot(dGx, dGy)
    f = np.pi * radius**2 / LATTICE.cell_area
    with np.errstate(invalid="ignore", divide="ignore"):
        form = np.where(dG < 1e-12, f, 2 * f * j1(dG * radius) / (dG * radius))
    eye = (dG < 1e-12).astype(complex)
    mu, rho = eye.copy(), eye.copy()
    for d, inc in zip((D_A, D_B), inclusions):
        phase = np.exp(-1j * (dGx * d[0] + dGy * d[1]))
        mu += (inc.mu - 1.0) * form * phase
        rho += (inc.rho - 1.0) * form * phase
    Qx, Qy = Gx + K[0], Gy + K[1]
    stiffness = mu * (Qx[:, None] * Qx[None, :] + Qy[:, None] * Qy[None, :])
    w2 = sla.eigh(stiffness, rho, eigvals_only=True, subset_by_index=[0, n_bands - 1])
    return np.sqrt(np.abs(w2))


def test_two_scatterer_bands_match_plane_wave_expansion():
    host = HostMedium(mu=1.0, rho=1.0)
    inclusions = (HostMedium(mu=2.0, rho=1.5), HostMedium(mu=3.0, rho=1.2))
    radius = 0.30
    problem = BasisSecularProblem(
        LATTICE, tuple(SolidCylinderScatterer(radius, host, inc) for inc in inclusions),
        (tuple(D_A), tuple(D_B)), 5, LatticeSumConfig(h_max=25, gauge_alternatives_over_a=(0.45,)),
        symmetrized=True,
    )

    def sigma_ratio(omega, K):
        s = np.linalg.svd(problem.M(omega, K), compute_uv=False)
        return s[-1] / s[0]

    for K in (M_POINT, K_POINT):
        for omega_pwe in _pwe(K, radius, inclusions):
            res = minimize_scalar(lambda w: sigma_ratio(w, K), method="bounded",
                                  bounds=(0.99 * omega_pwe, 1.01 * omega_pwe),
                                  options={"xatol": 1e-9})
            assert res.fun < 1e-6
            assert abs(res.x - omega_pwe) / omega_pwe < 4e-3
