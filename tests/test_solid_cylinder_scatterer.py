import numpy as np
import pytest

from phbands.basis import H1, H1p, J, Jp
from phbands.media import HostMedium
from phbands.scatterer import CavityScatterer, SolidCylinderScatterer


HOST = HostMedium(mu=153.338823784722, rho=1.85)
INCLUSION = HostMedium(mu=29.9760624174507, rho=19.5)
RADIUS = np.sqrt(0.0908 / np.pi)


@pytest.mark.parametrize("n", range(-3, 4))
def test_solid_cylinder_is_symmetric_in_modal_order(n):
    scatterer = SolidCylinderScatterer(RADIUS, HOST, INCLUSION)
    assert scatterer.T_n(8.3, n) == pytest.approx(scatterer.T_n(8.3, -n), abs=2e-14)


def test_identical_media_do_not_scatter():
    scatterer = SolidCylinderScatterer(RADIUS, HOST, HOST)
    for n in range(-3, 4):
        assert abs(scatterer.T_n(8.3, n)) < 2e-14


def test_soft_inclusion_limit_matches_free_cavity():
    almost_void = HostMedium(mu=1e-14, rho=1e-14)
    solid = SolidCylinderScatterer(RADIUS, HOST, almost_void)
    cavity = CavityScatterer(RADIUS, HOST)
    for n in range(-3, 4):
        assert solid.T_n(8.3, n) == pytest.approx(cavity.T_n(8.3, n), abs=2e-12)


def test_closed_form_satisfies_both_interface_conditions():
    scatterer = SolidCylinderScatterer(RADIUS, HOST, INCLUSION)
    omega = 11.7
    for n in range(-2, 3):
        kh, ki = HOST.kappa(omega), INCLUSION.kappa(omega)
        xh, xi = kh * RADIUS, ki * RADIUS
        b = scatterer.T_n(omega, n)
        c = (J(n, xh) + b * H1(n, xh)) / J(n, xi)
        displacement = J(n, xh) + b * H1(n, xh) - c * J(n, xi)
        traction = (
            HOST.mu * kh * (Jp(n, xh) + b * H1p(n, xh))
            - INCLUSION.mu * ki * c * Jp(n, xi)
        )
        assert abs(displacement) < 2e-13
        assert abs(traction) < 2e-11
