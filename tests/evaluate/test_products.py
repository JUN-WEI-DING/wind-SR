import numpy as np
import pytest
import xarray as xr

from windsr.data.stations import Stations, stations
from windsr.data.store import open_store
from windsr.evaluate.products import model_product, prediction_grid, rivals
from windsr.grid import GRID_LAT, GRID_LON
from windsr.periods import TEST, TRAIN, VALID, Period

SITES = Stations(np.array(["a", "b"]), np.array([GRID_LAT[3] + 0.02, GRID_LAT[30]]),
                 np.array([GRID_LON[5], GRID_LON[20] - 0.03]))


def _store(tmp_path, lat, hours, value=0.0):
    da = xr.DataArray(np.full((len(hours), 41, 41), value, np.float32), dims=("time", "latitude", "longitude"),
                      coords={"time": hours, "latitude": lat, "longitude": GRID_LON})
    path = str(tmp_path / "pred.zarr")
    da.to_dataset(name="wind_speed_10m").to_zarr(path, mode="w")
    return path


def test_prediction_store_checks_orientation_and_hours(tmp_path):
    with pytest.raises(ValueError, match="time axis lacks"):
        prediction_grid(_store(tmp_path, GRID_LAT, TEST.hours[:48]), TEST)
    with pytest.raises(ValueError, match="ascending"):
        prediction_grid(_store(tmp_path, GRID_LAT[::-1], TEST.hours), TEST)
    path = _store(tmp_path, GRID_LAT, TRAIN.hours)
    assert prediction_grid(path, TRAIN).shape == (len(TRAIN.hours), 41, 41)
    sub = prediction_grid(path, VALID)
    assert np.array_equal(sub.time.values, VALID.hours), "a store over the training period serves the validation weeks"


def test_model_reads_each_period_from_the_store_that_holds_it_and_clips_at_zero(tmp_path):
    (tmp_path / "t").mkdir()
    (tmp_path / "v").mkdir()
    test, valid = _store(tmp_path / "t", GRID_LAT, TEST.hours, 2.0), _store(tmp_path / "v", GRID_LAT, VALID.hours, -1.0)
    load = model_product(test, valid)
    x = load(SITES, TEST, "bilinear")
    assert x.shape == (len(TEST.hours), 2) and x.dtype == np.float64 and (x == 2).all()
    assert (load(SITES, VALID, "nearest") == 0).all(), "negative speed is clipped"
    with pytest.raises(ValueError, match="holds every train hour"):
        load(SITES, TRAIN, "nearest")


@pytest.mark.data
def test_era5_land_has_a_value_at_every_station_and_zero_where_its_own_cell_is_sea(data):
    st = stations(data)
    period = Period("sample", TEST.hours[0], TEST.hours[0] + np.timedelta64(47, "h"))
    land = open_store(data, "era5_land").wind_speed.isel(time=0)
    iy = np.abs(st.lat[:, None] - land.latitude.values).argmin(1)
    ix = np.abs(st.lon[:, None] - land.longitude.values).argmin(1)
    sea = np.isnan(land.values[iy, ix])
    x = rivals(data)["era5_land"](st, period, "bilinear")
    assert int(sea.sum()) == 46
    assert np.isfinite(x).all() and (x[:, sea] == 0).all() and (x[:, ~sea] > 0).mean() > 0.99
    assert np.array_equal(x, rivals(data)["era5_land"](st, period, "nearest")), "always cell-based"


@pytest.mark.data
def test_era5_and_icon_take_the_nearest_cell_or_bilinear_interpolation(data):
    st = stations(data)
    period = Period("sample", TEST.hours[0], TEST.hours[0] + np.timedelta64(5, "h"))
    field = open_store(data, "era5").wind_speed_10m.sel(time=period.hours).values
    iy = np.abs(st.lat[:, None] - GRID_LAT).argmin(1)
    ix = np.abs(st.lon[:, None] - GRID_LON).argmin(1)
    load = rivals(data)["era5"]
    assert np.array_equal(load(st, period, "nearest"), field[:, iy, ix].astype(np.float64))
    assert not np.array_equal(load(st, period, "bilinear"), load(st, period, "nearest"))
