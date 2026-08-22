from phbands.rootfinding.gap_detection import detect_bridge_gaps, detect_continuity_mismatches


def test_one_to_one_matching_detects_missing_crossing_multiplicity():
    bands = [
        [0.40, 0.44],
        [0.42],
        [0.44, 0.40],
    ]
    triggers = detect_bridge_gaps(bands, max_span=2, interior_tolerance=0.005)
    assert len(triggers) == 1
    assert triggers[0].k_index == 1
    assert triggers[0].frequency in (0.40, 0.44)


def test_complete_crossing_does_not_trigger():
    bands = [
        [0.40, 0.44],
        [0.419, 0.421],
        [0.44, 0.40],
    ]
    assert not detect_bridge_gaps(bands, max_span=2, interior_tolerance=0.005)


def test_continuity_mismatch_detects_lost_branch_without_count_deficit():
    bands = [
        [0.20, 0.30],
        [0.20, 0.41],
        [0.20, 0.30],
    ]
    triggers = detect_continuity_mismatches(bands, max_span=2, match_tolerance=0.01)
    assert any(trigger.k_index == 1 and abs(trigger.frequency - 0.30) < 1e-12 for trigger in triggers)


def test_continuity_mismatch_does_not_replace_physical_validation():
    bands = [[0.20, 0.30], [0.201, 0.299], [0.20, 0.30]]
    assert not detect_continuity_mismatches(bands, max_span=2, match_tolerance=0.01)
