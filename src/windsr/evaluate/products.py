"""The products at the stations: ERA5, ICON and ERA5-Land from DATA, and a model's prediction stores.

A product is a `Loader`: (stations, period, sampling) -> (time, station) float64 array of 10 m wind speed in m/s,
clipped at 0. Sampling is the nearest 0.125 deg cell or bilinear interpolation at the station location.

- era5, icon: `wind_speed_10m` of era5.zarr and icon.zarr.
- era5_land: the station's nearest cell of era5_land.zarr whatever the sampling, with a sea cell (NaN) taken as 0,
  so it has a value at every station.
- a model: `wind_speed_10m` of a store on the ascending 41x41 output grid, each period read from the first store
  whose time axis holds all its hours.
"""
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import xarray as xr

from ..data.stations import Stations
from ..data.store import open_store
from ..grid import GRID_LAT, GRID_LON, sample_grid
from ..periods import Period, on_hours

Sampling = Literal["nearest", "bilinear"]
Loader = Callable[[Stations, Period, Sampling], np.ndarray]
RIVALS = ("era5", "icon", "era5_land")


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, None)


def _grid_store(data: Path, name: str) -> Loader:
    def load(st: Stations, period: Period, sampling: Sampling) -> np.ndarray:
        da = open_store(data, name).wind_speed_10m
        return _clip(sample_grid(on_hours(da, period).load(), st, sampling))
    return load


def _era5_land(data: Path) -> Loader:
    def load(st: Stations, period: Period, sampling: Sampling) -> np.ndarray:
        da = on_hours(open_store(data, "era5_land").wind_speed, period).load()
        x = sample_grid(da, st, "nearest")
        return _clip(np.where(np.isnan(x), 0.0, x))
    return load


def rivals(data: Path) -> dict[str, Loader]:
    """The loaders of RIVALS."""
    return {"era5": _grid_store(data, "era5"), "icon": _grid_store(data, "icon"), "era5_land": _era5_land(data)}


def open_prediction(path: str) -> xr.Dataset:
    return xr.open_zarr(path) if path.endswith(".zarr") else xr.open_dataset(path)


def prediction_grid(path: str, period: Period, var: str = "wind_speed_10m") -> xr.DataArray:
    """(time, latitude, longitude) float64 field `var` of a prediction store on the hours of `period`."""
    da = open_prediction(path)[var].transpose("time", "latitude", "longitude")
    if not np.allclose(da.latitude.values, GRID_LAT, atol=1e-6, rtol=0) or not np.allclose(da.longitude.values, GRID_LON, atol=1e-6, rtol=0):
        raise ValueError(f"{path}: latitude/longitude are not the ascending 41x41 WindSR grid")
    t = da.time.values.astype("datetime64[ns]")
    missing = ~np.isin(period.hours, t)
    if missing.any():
        raise ValueError(f"{path}: time axis lacks {int(missing.sum())} of the {period.name} hours, first {period.hours[missing][0]}")
    da = da.assign_coords(time=t).sel(time=period.hours)
    return da.astype(np.float64).assign_coords(latitude=GRID_LAT, longitude=GRID_LON).load()


def model_product(*paths: str) -> Loader:
    """Model predictions in m/s from .zarr or .nc stores holding wind_speed_10m on time/latitude/longitude."""
    def covers(path: str, period: Period) -> bool:
        return bool(np.isin(period.hours, open_prediction(path).time.values.astype("datetime64[ns]")).all())

    def load(st: Stations, period: Period, sampling: Sampling) -> np.ndarray:
        path = next((p for p in paths if covers(p, period)), None)
        if path is None:
            raise ValueError(f"no store of {', '.join(paths)} holds every {period.name} hour")
        return _clip(sample_grid(prediction_grid(path, period), st, sampling))
    return load
