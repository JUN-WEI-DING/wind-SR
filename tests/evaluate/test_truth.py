import numpy as np
import pytest

from windsr.data.stations import stations
from windsr.data.store import open_store
from windsr.evaluate.truth import LEVEL_SHIFT_STATIONS, station_set_mask, truth
from windsr.periods import TEST, TRAIN

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def store(data):
    return open_store(data, "stations").load()


def on(store, var, period):
    return store[var].sel(time=period.hours).transpose("time", "station_id").values


@pytest.mark.parametrize("period", [TRAIN, TEST], ids=["train", "test"])
def test_truth_is_the_raw_speed_without_missing_codes_stuck_hours_and_calms(data, store, period):
    raw = on(store, "wind_speed_raw", period).astype(np.float64)
    stuck = on(store, "stuck_mask", period)
    y = truth(data, period)
    assert y.values.shape == (len(period.hours), 345) and y.values.dtype == np.float64
    assert np.array_equal(y.station_ids, store.station_id.values.astype(str))
    dropped = ~np.isfinite(raw) | (raw < 0) | stuck | (raw < 0.1)
    assert np.array_equal(np.isnan(y.values), dropped)
    assert np.array_equal(y.values[~dropped], raw[~dropped])
    assert not (y.values < 0.1).any() and stuck.sum() > 0


def test_sensitivity_set_leaves_out_the_18_level_shift_stations(data):
    ids = stations(data).ids
    keep = station_set_mask("sensitivity", ids)
    assert len(LEVEL_SHIFT_STATIONS) == 18 and set(ids[~keep]) == set(LEVEL_SHIFT_STATIONS)
    assert station_set_mask("main", ids).all()
