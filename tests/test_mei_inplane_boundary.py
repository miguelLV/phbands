import numpy as np
import pytest
from scipy.special import hankel1, jv

from phbands.basis import H1, J
from phbands.inplane.boundary import _block_12


def _mei_displacement_block(C, k_l, k_t, radius, m):
    """Elementos A_11, A_21, A_12 y A_22 de Mei et al., Ap. B."""
    z_l = k_l * radius
    z_t = k_t * radius
    return np.array(
        [
            [
                (m * C(m, z_l) - z_l * C(m + 1, z_l)) / radius,
                1j * m * C(m, z_t) / radius,
            ],
            [
                1j * m * C(m, z_l) / radius,
                (z_t * C(m + 1, z_t) - m * C(m, z_t)) / radius,
            ],
        ],
        dtype=complex,
    )


@pytest.mark.parametrize("C", [J, H1])
@pytest.mark.parametrize("m", [-2, -1, 0, 1, 2])
def test_displacement_block_matches_mei_appendix_b(C, m):
    obtained = _block_12(C, 2.13, 3.47, 0.31, m)
    expected = _mei_displacement_block(C, 2.13, 3.47, 0.31, m)
    np.testing.assert_allclose(obtained, expected, rtol=2e-13, atol=2e-13)


def test_mei_reference_functions_are_independent_scipy_wrappers():
    # Red de seguridad: documenta que las funciones usadas por el test son
    # las funciones cilindricas estandar de SciPy para J y H^(1).
    z = 1.37
    for m in range(-2, 3):
        assert J(m, z) == pytest.approx(jv(m, z))
        assert H1(m, z) == pytest.approx(hankel1(m, z))
