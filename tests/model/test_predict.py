import numpy as np
import pytest
import xarray as xr

pytest.importorskip("torch")

from windsr.data.stations import Stations
from windsr.data.taipower import FARMS
from windsr.evaluate.products import model_product
from windsr.evaluate.windfarm import store_wind
from windsr.grid import GRID_LAT, GRID_LON
from windsr.model.predict import write_store
from windsr.periods import TRAIN, Period


def test_store_holds_both_heights_and_serves_station_and_farm_scoring(tmp_path):
    period = Period("sample", TRAIN.hours[0], TRAIN.hours[0] + np.timedelta64(5, "h"))
    ws = np.empty((6, 2, 41, 41), np.float32)
    ws[:, 0] = GRID_LAT[:, None] - 15
    ws[:, 1] = GRID_LON[None, :] - 110
    path = write_store(tmp_path / "pred.zarr", period.hours, ws)
    ds = xr.open_zarr(path)
    assert set(ds.data_vars) == {"wind_speed_10m", "wind_speed_100m"}

    st = Stations(np.array(["a", "b", "c"]), np.array([22.01, 23.4, 25.07]), np.array([120.3, 121.0, 121.5]))
    got = model_product(path)(st, period, "nearest")
    want = GRID_LAT[np.abs(st.lat[:, None] - GRID_LAT).argmin(1)] - 15
    assert got.shape == (6, len(st.ids)) and np.allclose(got, want[None], atol=1e-5), "model_product reads 10 m"

    w10, w100 = store_wind(path, period)
    lat, lon = np.array([f.lat for f in FARMS]), np.array([f.lon for f in FARMS])
    assert np.allclose(w10, GRID_LAT[np.abs(lat[:, None] - GRID_LAT).argmin(1)] - 15, atol=1e-5)
    assert np.allclose(w100, GRID_LON[np.abs(lon[:, None] - GRID_LON).argmin(1)] - 110, atol=1e-5)
    with pytest.raises(Exception):
        write_store(tmp_path / "pred.zarr", period.hours, ws)
