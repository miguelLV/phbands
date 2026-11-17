"""Red hexagonal: geometria, suma de red y bandas contra ondas planas."""

import numpy as np
import pytest
import scipy.linalg as sla
from scipy.optimize import minimize_scalar
from scipy.special import j1

from phbands.lattice import HexagonalLattice, hexagonal_GMKG_path
from phbands.lattice_sums import LatticeSumConfig, chin_accelerated, direct_sum
from phbands.media import HostMedium
from phbands.scatterer import SolidCylinderScatterer
from phbands.secular import SecularProblem


LATTICE = HexagonalLattice(1.0)
K_POINT = np.array([4.0 * np.pi / 3.0, 0.0])


def test_reciprocal_basis_is_dual_and_cutoff_is_hexagonally_symmetric():
    Gx, Gy = LATTICE.reciprocal_vectors_xy(6)
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3.0) / 2.0])
    for a_i in (a1, a2):
        phase = (Gx * a_i[0] + Gy * a_i[1]) / (2 * np.pi)
        np.testing.assert_allclose(phase, np.round(phase), atol=1e-12)
    # cerrado bajo rotaciones de 60 grados
    c, s = np.cos(np.pi / 3), np.sin(np.pi / 3)
    rotated = {(round(c * x - s * y, 8), round(s * x + c * y, 8)) for x, y in zip(Gx, Gy)}
    original = {(round(x, 8), round(y, 8)) for x, y in zip(Gx, Gy)}
    assert rotated == original
    assert LATTICE.cell_area == pytest.approx(np.sqrt(3.0) / 2.0)


def test_high_symmetry_path_lengths():
    path = hexagonal_GMKG_path(1.0, 4)
    lengths = np.diff(path.tick_positions)
    np.testing.assert_allclose(lengths, [2 * np.pi / np.sqrt(3), 2 * np.pi / 3, 4 * np.pi / 3])
    np.testing.assert_allclose(path.k_points[-1], [0.0, 0.0], atol=1e-14)


@pytest.mark.parametrize("K", [np.array([0.7, 0.3]), K_POINT])
@pytest.mark.parametrize("l", [0, 1, 3, 4])
def test_accelerated_sum_matches_direct_sum(K, l):
    kappa = 3.0 + 0.5j
    s_direct = direct_sum(l, kappa, K, LATTICE, n_shells=50)
    s_chin = chin_accelerated(l, kappa, K, LATTICE, h_max=50)
    assert abs(s_direct - s_chin) <= 2e-5 * max(1.0, abs(s_direct))


def test_lattice_sum_vanishes_at_K_unless_order_is_multiple_of_three():
    for l in (1, 2, 4, 5):
        assert abs(chin_accelerated(l, 3.0 + 0.5j, K_POINT, LATTICE, h_max=50)) < 1e-4


def _pwe_frequencies(K, radius, host, inclusion, filling, h_max=14, n_bands=4):
    """Expansion en ondas planas (referencia independiente del MST)."""
    Gx, Gy = LATTICE.reciprocal_vectors_xy(h_max)
    dG = np.hypot(Gx[:, None] - Gx[None, :], Gy[:, None] - Gy[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        form = np.where(dG < 1e-12, filling, 2 * filling * j1(dG * radius) / (dG * radius))
    eye = dG < 1e-12
    mu = eye * host.mu + (inclusion.mu - host.mu) * form
    rho = eye * host.rho + (inclusion.rho - host.rho) * form
    Qx, Qy = Gx + K[0], Gy + K[1]
    stiffness = mu * (Qx[:, None] * Qx[None, :] + Qy[:, None] * Qy[None, :])
    w2 = sla.eigh(stiffness, rho, eigvals_only=True, subset_by_index=[0, n_bands - 1])
    return np.sqrt(np.abs(w2))


def test_bands_match_plane_wave_expansion():
    filling = 0.283
    radius = np.sqrt(filling * LATTICE.cell_area / np.pi)
    host = HostMedium(mu=1.0, rho=1.0)
    inclusion = HostMedium(mu=2.0, rho=1.5)
    problem = SecularProblem(
        LATTICE, SolidCylinderScatterer(radius, host, inclusion), 6,
        LatticeSumConfig(h_max=30, gauge_alternatives_over_a=(0.45,)), symmetrized=True,
    )

    def sigma_ratio(omega, K):
        s = np.linalg.svd(problem.M(omega, K), compute_uv=False)
        return s[-1] / s[0]

    M_point = np.array([np.pi, np.pi / np.sqrt(3.0)])
    for K in (M_point, K_POINT):
        for omega_pwe in _pwe_frequencies(K, radius, host, inclusion, filling):
            res = minimize_scalar(lambda w: sigma_ratio(w, K), method="bounded",
                                  bounds=(0.99 * omega_pwe, 1.01 * omega_pwe),
                                  options={"xatol": 1e-9})
            assert res.fun < 1e-6
            # la PWE converge lento con coeficientes discontinuos (~5e-4 a h_max=14)
            assert abs(res.x - omega_pwe) / omega_pwe < 3e-3
