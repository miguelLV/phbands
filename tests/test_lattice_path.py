import numpy as np

from phbands.lattice import build_path


def test_path_distances_match_sampled_k_points_without_node_overlap():
    path = build_path(
        [("M", 1.0, 1.0), ("Gamma", 0.0, 0.0), ("X", 1.0, 0.0), ("M", 1.0, 1.0)],
        a=1.0,
        n_per_segment=4,
    )

    increments = np.linalg.norm(np.diff(path.k_points, axis=0), axis=1)
    np.testing.assert_allclose(np.diff(path.distances), increments)
    assert np.all(np.diff(path.distances) > 0.0)
    np.testing.assert_allclose(path.tick_positions, [0.0, np.sqrt(2) * np.pi, (np.sqrt(2) + 1) * np.pi, (np.sqrt(2) + 2) * np.pi])
