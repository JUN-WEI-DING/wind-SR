from pathlib import Path

import numpy as np
import pytest

from windsr.data.store import open_store
from windsr.grid import crop
from windsr.model.inputs import CHANNELS, Norm, era5_speed, fit_fields, icon, raw_channels
from windsr.periods import FIT, TEST, TRAIN, VALID, Period

pytestmark = pytest.mark.data
SAMPLE = Period("sample", TRAIN.hours[0], TRAIN.hours[0] + np.timedelta64(47, "h"))


def test_inputs_reproduce_era5_on_the_output_window(data):
    raw = raw_channels(data, SAMPLE)
    lr = open_store(data, "era5").sel(time=SAMPLE.hours)
    for name, var in {"u10": "u10", "v10": "v10", "u100": "u100", "ws10": "wind_speed_10m", "ws100": "wind_speed_100m"}.items():
        assert np.abs(crop(raw[:, CHANNELS.index(name)]) - lr[var].values).max() < 1e-5, name


def test_direction_terms_are_unit_vectors_pointing_where_the_wind_comes_from(data):
    raw = raw_channels(data, SAMPLE)
    s, c = raw[:, CHANNELS.index("sin10")], raw[:, CHANNELS.index("cos10")]
    calm = np.hypot(raw[:, CHANNELS.index("u10")], raw[:, CHANNELS.index("v10")]) < 0.1
    assert calm.any() and (s[calm] == 0).all() and (c[calm] == 0).all()
    assert np.allclose((s ** 2 + c ** 2)[~calm], 1, atol=1e-5)
    deg = (np.degrees(np.arctan2(s, c)) + 360) % 360
    diff = crop(deg) - open_store(data, "era5").wind_dir_10m.sel(time=SAMPLE.hours).values
    assert np.abs(((diff + 180) % 360 - 180)[~crop(calm)]).max() < 0.01


def test_inputs_end_with_the_utc_clock_and_the_grid_position(data):
    raw = raw_channels(data, SAMPLE)
    assert raw.shape == (48, 16, 57, 57)
    hour = SAMPLE.hours.astype("datetime64[h]").astype(np.int64) % 24
    assert np.allclose(raw[:, CHANNELS.index("hour_sin"), 30, 40], np.sin(2 * np.pi * hour / 24), atol=1e-6)
    assert np.allclose(raw[:, CHANNELS.index("hour_cos"), 0, 0], np.cos(2 * np.pi * hour / 24), atol=1e-6)
    lat, lon = raw[:, CHANNELS.index("lat")], raw[:, CHANNELS.index("lon")]
    assert np.allclose(lat[:, 0], -1) and np.allclose(lat[:, -1], 1) and np.allclose(lon[..., 0], -1)


def test_norm_uses_fit_hours_and_round_trips(data, tmp_path):
    raw = raw_channels(data, SAMPLE)
    norm = Norm.fit(raw, icon(data, SAMPLE))
    x = norm.inputs(raw)
    assert np.allclose(x[:, :6].mean((0, 2, 3)), 0, atol=1e-4) and np.allclose(x[:, :6].std((0, 2, 3)), 1, atol=1e-4)
    assert np.array_equal(x[:, 6:], raw[:, 6:])
    norm.save(tmp_path / "n.json")
    assert Norm.load(tmp_path / "n.json") == norm
    ws = era5_speed(raw)
    assert np.allclose(norm.to_ms(norm.to_scale(ws)), ws, atol=1e-5)
    lr = open_store(data, "era5").sel(time=SAMPLE.hours)
    for j, h in enumerate(("10m", "100m")):
        assert np.abs(ws[:, j] - lr[f"wind_speed_{h}"].values).max() < 1e-5, "ERA5 base is the scored ERA5 product"
        both = np.concatenate([ws[:, j].ravel(), icon(data, SAMPLE)[:, j].ravel()])
        assert np.isclose(norm.ws_mean[j], both.mean()) and np.isclose(norm.ws_std[j], both.std(), rtol=1e-5)


def test_fit_hours_give_the_reference_normalisation(data):
    norm, fields = fit_fields(data, FIT)
    assert norm == Norm.load(Path(__file__).parent / "norm_reference.json"), "fit_fields reproduces norm_reference.json"
    assert fields.base.shape == fields.target.shape == (len(fields.times), 2, 41, 41) == (len(FIT.hours), 2, 41, 41)
    assert norm.ws_mean[1] > norm.ws_mean[0], "100 m wind is faster on average"


def test_icon_has_no_missing_values_on_training_hours(data):
    for p in (FIT, VALID):
        t = icon(data, p)
        assert t.shape[1] == 2 and np.isfinite(t).all()
    assert not np.isin(TEST.hours, TRAIN.hours).any()
