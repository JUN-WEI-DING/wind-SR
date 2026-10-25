"""ERA5 stores from daily ERA5 stores in the ARCO-ERA5 layout.

`era5_native.zarr` keeps u and v at 10 m and 100 m on the native 29x29 grid, 20.5-27.5 N, 117.5-124.5 E, unchanged.

`era5.zarr` interpolates them bilinearly to the labelled 0.125 deg coordinates of Zenodo lr. Speed is hypot(u, v) on
the native grid, then interpolated. Direction (from) comes from the interpolated u, v.
"""
import numpy as np
import xarray as xr

VARS = {"10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
        "100m_u_component_of_wind": "u100", "100m_v_component_of_wind": "v100"}
MS = "m s**-1"
SOURCE = ("ERA5 hourly data on single levels (Hersbach et al. 2020, Q. J. R. Meteorol. Soc. 146, 1999-2049; "
          "Copernicus Climate Change Service, doi:10.24381/cds.adbb2d47), read from the ARCO-ERA5 copy")


def _wind(h):
    return {
        f"u{h}": {"long_name": f"{h} metre U wind component", "units": MS},
        f"v{h}": {"long_name": f"{h} metre V wind component", "units": MS},
        f"wind_speed_{h}m": {"long_name": f"{h} metre wind speed: hypot(u{h}, v{h}) on the native 0.25 deg grid, then interpolated", "units": MS},
        f"wind_dir_{h}m": {"long_name": f"{h} metre wind direction (from), clockwise from north, computed from the interpolated u{h}, v{h}", "units": "degree"},
    }


GRID = {**_wind(10), **_wind(100)}
NATIVE = {k: v for k, v in GRID.items() if k[0] in "uv"}
NATIVE_NOTE = f"{SOURCE}; native 0.25 deg, 20.5-27.5 N, 117.5-124.5 E, values unchanged; latitude ascending"
GRID_NOTE = f"{SOURCE}; native 0.25 deg, bilinear to the labelled 0.125 deg coordinates; direction (from) computed from interpolated u,v"


def read_native(paths: list[str]) -> xr.Dataset:
    """Daily ERA5 stores in time order -> u10, v10, u100, v100 (time, latitude ascending, longitude), float32."""
    parts = [xr.open_zarr(p)[list(VARS)].sel(latitude=slice(27.5, 20.5), longitude=slice(117.5, 124.5)).load()
             .sortby("latitude").rename(VARS) for p in paths]
    native = xr.concat(parts, "time")
    return native.assign_coords(time=native.time.values.astype("datetime64[h]").astype("datetime64[ns]"),
                                latitude=native.latitude.values.astype(float), longitude=native.longitude.values.astype(float))


def interp1d(x: np.ndarray, a: np.ndarray, xn: np.ndarray, axis: int) -> np.ndarray:
    """Linear interpolation of `a` along `axis` from ascending `x` to `xn`, inside the range of `x`.

    The slope form (a_hi - a_lo) / (x_hi - x_lo) * (xn - x_lo) + a_lo, with a_hi - a_lo in the dtype of `a`, is the
    arithmetic of earlier scipy interp1d, which produced the published era5.zarr. scipy 1.18 sums weighted ends
    instead, which can differ in the last float32 bit.
    """
    assert x[0] <= xn.min() and xn.max() <= x[-1], "target outside the source grid"
    a = np.moveaxis(a, axis, 0)
    i = np.searchsorted(x, xn).clip(1, len(x) - 1)
    lo, hi = i - 1, i
    col = (-1,) + (1,) * (a.ndim - 1)
    out = (a[hi] - a[lo]) / (x[hi] - x[lo]).reshape(col) * (xn - x[lo]).reshape(col) + a[lo]
    return np.moveaxis(out, 0, axis)


def bilinear(a: np.ndarray, lat: np.ndarray, lon: np.ndarray, lat_new: np.ndarray, lon_new: np.ndarray) -> np.ndarray:
    """(time, lat, lon) -> (time, lat_new, lon_new) float32: linear along latitude, then along longitude."""
    return interp1d(lon, interp1d(lat, a, lat_new, 1), lon_new, 2).astype(np.float32)


def build(paths: list[str], zenodo_lr: str, out_native: str, out_grid: str) -> None:
    native = read_native(paths)
    nat = native.copy()
    for k in nat.variables:
        nat[k].encoding = {}
    for k, a in NATIVE.items():
        nat[k].attrs = a
    nat.attrs = {"note": NATIVE_NOTE}
    nat.chunk({"time": 720}).to_zarr(out_native, mode="w", consolidated=True, zarr_format=2)
    print("era5_native built", dict(nat.sizes))

    zen = xr.open_zarr(zenodo_lr)
    assert np.array_equal(native.time.values, zen.time.values), "native hours differ from Zenodo lr hours"
    lat, lon = native.latitude.values, native.longitude.values
    lat41, lon41 = zen.latitude.values, zen.longitude.values
    out = {}
    for h in ["10", "100"]:
        u, v = native["u" + h].values, native["v" + h].values
        out[f"u{h}"], out[f"v{h}"] = bilinear(u, lat, lon, lat41, lon41), bilinear(v, lat, lon, lat41, lon41)
        out[f"wind_speed_{h}m"] = bilinear(np.hypot(u, v), lat, lon, lat41, lon41)
        out[f"wind_dir_{h}m"] = ((np.degrees(np.arctan2(-out[f"u{h}"], -out[f"v{h}"])) + 360) % 360).astype("float32")
    dims = ("time", "latitude", "longitude")
    ds = xr.Dataset({k: (dims, a, GRID[k]) for k, a in out.items()},
                    coords={"time": native.time.values, "latitude": lat41, "longitude": lon41})
    ds.attrs = {"note": GRID_NOTE}
    ds.chunk({"time": 720}).to_zarr(out_grid, mode="w", consolidated=True, zarr_format=2)
    print("era5 built", dict(ds.sizes))
