import numpy as np

from windsr.model.inputs import CHANNELS, clock_and_position

TIMES = np.array(["2023-01-01T06", "2024-03-01T00", "2023-12-31T23"], "datetime64[ns]")
CLOCK = CHANNELS[-6:]


def _ch(name):
    return clock_and_position(TIMES)[:, CLOCK.index(name)]


def test_six_channels_follow_the_ten_era5_channels_on_the_input_grid():
    assert CLOCK == ("hour_sin", "hour_cos", "doy_sin", "doy_cos", "lat", "lon") and len(CHANNELS) == 16
    x = clock_and_position(TIMES)
    assert x.shape == (3, 6, 57, 57) and x.dtype == np.float32


def test_hour_of_day_is_a_utc_clock():
    assert np.isclose(_ch("hour_sin")[0], 1, atol=1e-6).all() and np.isclose(_ch("hour_cos")[0], 0, atol=1e-6).all()
    assert np.isclose(_ch("hour_sin")[1], 0, atol=1e-6).all() and np.isclose(_ch("hour_cos")[1], 1, atol=1e-6).all()
    assert np.isclose(_ch("hour_sin")[2], np.sin(2 * np.pi * 23 / 24), atol=1e-6).all()


def test_day_of_year_counts_from_1_january_as_day_0():
    for k, day in enumerate((0, 31 + 29, 364)):
        assert np.isclose(_ch("doy_sin")[k], np.sin(2 * np.pi * day / 365.25), atol=1e-6).all(), day
        assert np.isclose(_ch("doy_cos")[k], np.cos(2 * np.pi * day / 365.25), atol=1e-6).all(), day


def test_clock_channels_are_constant_over_the_grid():
    x = clock_and_position(TIMES)[:, :4]
    assert (x == x[:, :, :1, :1]).all()


def test_position_spans_minus_one_to_one_along_its_axis():
    lat, lon = _ch("lat"), _ch("lon")
    assert np.allclose(lat[:, 0], -1) and np.allclose(lat[:, -1], 1) and np.allclose(lat[:, 28], 0, atol=1e-6)
    assert (lat == lat[:, :, :1]).all(), "latitude is constant along each row"
    assert np.allclose(lon[:, :, 0], -1) and np.allclose(lon[:, :, -1], 1) and np.allclose(lon[:, :, 28], 0, atol=1e-6)
    assert (lon == lon[:, :1]).all(), "longitude is constant down each column"
    assert np.allclose(np.diff(lat[0, :, 0]), 2 / 56), "linear in latitude"
