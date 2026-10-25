"""The grids of WindSR, and sampling a gridded field at stations.

The output grid is 41x41 cells of 0.125 deg, 21.375-26.375 N and 118.125-123.125 E: the grid of ERA5
(`era5.zarr`), ICON and the network output. The network input is native ERA5 on 29x29 cells of 0.25 deg,
20.5-27.5 N and 117.5-124.5 E, upsampled to 57x57 cells of 0.125 deg on the same extent; rows ROW0 to ROW0 + N and
columns COL0 to COL0 + N of that fine grid are the output grid.
"""
import numpy as np
import xarray as xr

GRID_LAT = np.arange(41) * 0.125 + 21.375
GRID_LON = np.arange(41) * 0.125 + 118.125
NATIVE_LAT = 20.5 + 0.25 * np.arange(29)
NATIVE_LON = 117.5 + 0.25 * np.arange(29)
FINE_LAT = 20.5 + 0.125 * np.arange(57)
FINE_LON = 117.5 + 0.125 * np.arange(57)
ROW0, COL0, N = 7, 5, 41


def crop(x):
    """The output window of a (..., 57, 57) array or tensor: (..., 41, 41)."""
    return x[..., ROW0:ROW0 + N, COL0:COL0 + N]


assert np.allclose(crop(FINE_LAT[:, None] + 0 * FINE_LON)[:, 0], GRID_LAT)
assert np.allclose(crop(0 * FINE_LAT[:, None] + FINE_LON)[0], GRID_LON)


def nearest_cell(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Flat index (row * 41 + column) of the output-grid cell nearest each location."""
    iy = np.abs(lat[:, None] - GRID_LAT[None]).argmin(1)
    ix = np.abs(lon[:, None] - GRID_LON[None]).argmin(1)
    return iy * len(GRID_LON) + ix


def sample_grid(da: xr.DataArray, st, method: str) -> np.ndarray:
    """(time, latitude, longitude) field -> (time, station) float64 at the locations `st.lat`, `st.lon`, by the
    nearest cell or bilinear interpolation."""
    lat = xr.DataArray(st.lat, dims="station")
    lon = xr.DataArray(st.lon, dims="station")
    if method == "nearest":
        out = da.sel(latitude=lat, longitude=lon, method="nearest")
    else:
        out = da.interp(latitude=lat, longitude=lon, method="linear")
    return out.transpose("time", "station").values.astype(np.float64)
