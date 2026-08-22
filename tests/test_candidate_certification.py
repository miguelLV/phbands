import numpy as np
import pytest

from phbands.rootfinding.certification import (
    central_projection,
    modal_overlap,
    null_mode_diagnostics,
    prolongated_residual,
)


def test_null_mode_reports_exact_singular_direction():
    matrix = np.diag([2.0, 1.0, 0.0]).astype(complex)
    result = null_mode_diagnostics(matrix)
    assert result.sigma_min == pytest.approx(0.0)
    assert result.relative_residual == pytest.approx(0.0)
    assert result.edge_weight == pytest.approx(1.0)


def test_modal_overlap_ignores_phase_and_extra_zero_modes():
    lower = np.array([1.0, 2.0j, -0.5], dtype=complex)
    higher = np.r_[0.0, np.exp(0.7j) * lower, 0.0]
    assert modal_overlap(lower, higher) == pytest.approx(1.0)
    assert np.linalg.norm(central_projection(higher, 3)) == pytest.approx(1.0)


def test_prolongated_residual_detects_unsatisfied_extra_equations():
    lower = np.array([0.0, 1.0, 0.0], dtype=complex)
    compatible = np.diag([1.0, 1.0, 0.0, 1.0, 1.0]).astype(complex)
    incompatible = compatible.copy()
    incompatible[0, 2] = 1.0
    assert prolongated_residual(lower, compatible) == pytest.approx(0.0)
    assert prolongated_residual(lower, incompatible) > 0.0
