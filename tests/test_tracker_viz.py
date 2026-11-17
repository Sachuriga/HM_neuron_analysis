import numpy as np
import pytest

pytest.importorskip("pynwb")
from hm_neuron.tracker_viz import visualize_nwb as V  # noqa: E402


def _map(peaks):
    """Rate map with one 4x4 blob per (row, col, peak_hz)."""
    r = np.zeros((40, 60))
    for iy, ix, hz in peaks:
        r[iy:iy + 4, ix:ix + 4] = hz
    return np.ma.masked_invalid(r)


def test_min_field_peak_is_1hz():
    assert V.MIN_FIELD_PEAK_HZ == 1.0
    assert len(V.place_fields(_map([(5, 5, 0.8)]))) == 0      # passed the old 0.5 Hz gate
    assert len(V.place_fields(_map([(5, 5, 0.8)]), min_peak_hz=0.5)) == 1
    assert len(V.place_fields(_map([(5, 5, 1.2)]))) == 1


def test_fields_threshold_and_size():
    # second blob at 40% of peak is below the 50% threshold
    assert len(V.place_fields(_map([(5, 5, 10.0), (20, 30, 4.0)]))) == 1
    assert len(V.place_fields(_map([(5, 5, 10.0), (20, 30, 6.0)]))) == 2
    small = np.ma.masked_invalid(np.zeros((40, 60)))
    small[5:7, 5:7] = 10.0                                   # 4 bins < 6
    assert len(V.place_fields(small)) == 0
