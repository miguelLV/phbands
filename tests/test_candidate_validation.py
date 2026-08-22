import numpy as np
import pytest

from phbands.rootfinding.validation import (
    balanced_sigma_min,
    equilibrate_matrix,
    refine_matrix_candidate,
)


def test_equilibration_normalizes_nonzero_rows_and_columns():
    matrix = np.array([[1e12, 2e12], [3e-9, 8e-9]], dtype=complex)
    balanced = equilibrate_matrix(matrix)
    np.testing.assert_allclose(np.max(np.abs(balanced), axis=1), 1.0)
    np.testing.assert_allclose(np.max(np.abs(balanced), axis=0), 1.0)


def test_balanced_sigma_rejects_scale_only_false_singularity():
    badly_scaled_identity = np.diag([1e12, 1e-12]).astype(complex)
    assert np.linalg.svd(badly_scaled_identity, compute_uv=False)[-1] == pytest.approx(1e-12)
    assert balanced_sigma_min(badly_scaled_identity) == pytest.approx(1.0)


def test_refinement_recovers_known_matrix_root():
    root = 2.345

    def matrix_func(omega):
        return np.array([[omega - root, 0.0], [0.0, 2.0]], dtype=complex)

    result = refine_matrix_candidate(matrix_func, 2.34, half_window=0.02)
    assert result.success
    assert result.omega == pytest.approx(root, abs=2e-7)
    assert result.balanced_sigma_min < 1e-6
