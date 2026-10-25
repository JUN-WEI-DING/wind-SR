"""icon.zarr: Zenodo hr (ICON 10 m and 100 m wind speed on the 41x41 grid) with its one-column eastward shift from
2025-06-02T18 undone.

From that hour on, Zenodo hr column j holds the value of column j-1, so fixed[:, :, j] = zenodo[:, :, j+1].
Column 40 has no source for those hours and is NaN.
"""
import numpy as np
import xarray as xr

SWITCH = np.datetime64("2025-06-02T18")
HR_10M = {"long_name": "Wind speed at 10 m", "units": "m s-1"}
SOURCE = ("ICON global forecasts (DWD) from the Open-Meteo historical forecast API, model icon_global, as published "
          "in Zenodo doi:10.5281/zenodo.17647355 (hr.zarr)")
NOTE = ("From 2025-06-02T18 UTC to the end, the published ICON field (Zenodo hr) is displaced one column (0.125 deg) "
        "east. Undone as fixed[t, :, j] = zenodo[t, :, j+1] for j < 40; column 40 (longitude 123.125) has no source "
        "and is NaN for those hours. Hours before 2025-06-02T18 are unchanged.")


def undo_shift(a: np.ndarray, k0: int) -> None:
    """(time, latitude, longitude) array, in place: from hour index k0 on, column j takes column j+1 and the last
    column is NaN."""
    a[k0:, :, :-1] = a[k0:, :, 1:]
    a[k0:, :, -1] = np.nan


def build(zenodo_hr: str, out: str) -> None:
    ds = xr.open_zarr(zenodo_hr).load()
    k0 = int(np.searchsorted(ds.time.values, SWITCH))
    assert ds.time.values[k0] == SWITCH
    for v in ("wind_speed_10m", "wind_speed_100m"):
        undo_shift(ds[v].values, k0)
        ds[v].attrs["correction"] = NOTE
    ds["wind_speed_10m"].attrs.update(HR_10M)
    ds.attrs.update({"correction": NOTE, "source": SOURCE})
    for v in ds.variables.values():
        v.encoding = {}
    ds.chunk({"time": 720}).to_zarr(out, mode="w", consolidated=True, zarr_format=2)
    print("icon built", dict(ds.sizes), "shifted hours", len(ds.time) - k0)
