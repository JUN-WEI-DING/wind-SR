"""era5_land.zarr: Zenodo bm (ERA5-Land 10 m wind speed) with its sea fill of 0 set back to NaN.

The land cells are where the ERA5-Land u10 source (NSF NCAR GDEX dataset d633008) is finite.
"""
from pathlib import Path

import numpy as np
import xarray as xr

LAND_CELLS = 549
ATTRS = {"long_name": "ERA5-Land 10 metre wind speed, float32 hypot(u10, v10); NaN over the sea", "units": "m s**-1"}
NOTE = ("ERA5-Land hourly 10 m wind speed (Copernicus Climate Change Service, doi:10.24381/cds.e2161bac), float32 "
        "hypot of u10,v10, as published in Zenodo doi:10.5281/zenodo.17647355 (bm.zarr), with the sea fill of 0 set "
        "back to NaN; land mask and values checked bit-exact against ERA5-Land u10, v10 of NSF NCAR GDEX d633008")


def read_u10(nc_dir: str, times: np.ndarray) -> xr.DataArray:
    """GDEX ERA5-Land u10 NetCDF subsets (`*.u10_*.nc`) -> u10 (valid_time = `times`, latitude ascending,
    longitude) float32, NaN where missing."""
    files = sorted(Path(nc_dir).glob("*.u10_*.nc"))
    da = xr.concat([xr.open_dataset(f)["u10"].load() for f in files], "valid_time").sortby("valid_time")
    assert not da.valid_time.to_index().duplicated().any()
    return da.sortby("latitude").reindex(valid_time=times).astype("float32")


def land_mask(u10: np.ndarray) -> np.ndarray:
    """(time, latitude, longitude) source -> (latitude, longitude) bool, True where finite at every hour."""
    land = np.isfinite(u10).all(0)
    assert (np.isfinite(u10).any(0) == land).all(), "a cell is NaN in some hours only"
    return land


def sea_to_nan(ws: xr.DataArray, land: np.ndarray) -> xr.DataArray:
    """(time, latitude, longitude) wind speed whose sea cells are all 0 -> the same with the sea cells NaN."""
    assert (ws.values[:, ~land] == 0).all(), "sea cells must be all-zero in bm"
    return ws.where(xr.DataArray(land, dims=("latitude", "longitude"), coords={"latitude": ws.latitude, "longitude": ws.longitude}))


def build(zenodo_bm: str, nc_dir: str, out: str) -> None:
    bm = xr.open_zarr(zenodo_bm)
    src = read_u10(nc_dir, bm.time.values)
    slat, slon = np.round(src.latitude.values.astype(float), 4), np.round(src.longitude.values.astype(float), 4)
    iy = [int(np.where(slat == round(float(v), 4))[0][0]) for v in bm.latitude.values]
    ix = [int(np.where(slon == round(float(v), 4))[0][0]) for v in bm.longitude.values]
    land = land_mask(src.values[:, iy][:, :, ix])
    assert int(land.sum()) == LAND_CELLS, f"{int(land.sum())} land cells, expected {LAND_CELLS}"
    out_ws = sea_to_nan(bm.wind_speed.load(), land)
    out_ws.attrs = ATTRS
    ds = out_ws.to_dataset(name="wind_speed")
    ds.attrs = {"note": NOTE}
    ds.chunk({"time": 720}).to_zarr(out, mode="w", consolidated=True, zarr_format=2)
    print("era5_land built", dict(ds.sizes), "land cells", int(land.sum()))
