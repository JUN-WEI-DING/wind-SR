import numpy as np
import pytest

from windsr.data.store import open_store
from windsr.periods import TEST, TRAIN

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def store(data):
    return open_store(data, "stations").load()


def stuck_rule(x):
    """>= 12 h at one value above 0, or >= 48 h at exactly 0; a missing hour ends a run."""
    out = np.zeros(len(x), bool)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and np.isfinite(x[i]) and x[j + 1] == x[i]:
            j += 1
        n = j - i + 1
        if np.isfinite(x[i]) and ((x[i] > 0 and n >= 12) or (x[i] == 0 and n >= 48)):
            out[i:j + 1] = True
        i = j + 1
    return out


def test_stuck_mask_follows_the_rule(store):
    hours = np.concatenate([TRAIN.hours, TEST.hours])
    raw = store.wind_speed_raw.transpose("time", "station_id").values.astype(np.float64)
    mask = store.stuck_mask.transpose("time", "station_id").values
    inside = np.isin(store.time.values, hours)
    for j in range(raw.shape[1]):
        assert np.array_equal(stuck_rule(raw[:, j])[inside], mask[inside, j]), store.station_id.values[j]
