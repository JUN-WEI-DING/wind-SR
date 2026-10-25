"""Network inputs and targets from DATA.

Data shapes: the input is (hours, 16, 57, 57) float32 in CHANNELS order on the fine grid (`grid.py`); ERA5 and ICON
speed on the output window are (hours, 2, 41, 41) float32 at the HEIGHTS 10 m and 100 m.

The ten ERA5 channels are native ERA5 u, v at 10 m and 100 m upsampled bilinearly to the fine grid, the speed
upsampled from the native speed, and the direction the wind comes from as its sine and cosine (0 below 0.1 m/s).
Six channels follow: a UTC clock, the day of the year and the grid position. The first six channels are z-scored
on the fit hours; ERA5 and ICON speed share one scale per height, fitted on the fit hours of both.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..data.store import open_store
from ..grid import FINE_LAT, FINE_LON, GRID_LAT, GRID_LON, NATIVE_LAT, NATIVE_LON, crop
from ..periods import Period

ERA5_CHANNELS = ("u10", "v10", "u100", "v100", "ws10", "ws100", "sin10", "cos10", "sin100", "cos100")
CHANNELS = ERA5_CHANNELS + ("hour_sin", "hour_cos", "doy_sin", "doy_cos", "lat", "lon")
ZSCORED = 6
HEIGHTS = ("10m", "100m")
SPEED = (CHANNELS.index("ws10"), CHANNELS.index("ws100"))
CALM = 0.1


def upsample_matrix() -> np.ndarray:
    """(57, 29) bilinear weights from the 0.25 deg grid to the 0.125 deg grid on the same extent."""
    a = np.zeros((57, 29))
    for k in range(29):
        a[2 * k, k] = 1.0
    for k in range(28):
        a[2 * k + 1, k] = a[2 * k + 1, k + 1] = 0.5
    return a


def clock_and_position(times: np.ndarray) -> np.ndarray:
    """(hours, 6, 57, 57) float32, the last six CHANNELS for UTC `times`: sin and cos of 2 pi * hour / 24 and of
    2 pi * day / 365.25, with day 0 on 1 January, each constant over the grid; then latitude (along rows) and
    longitude (along columns) of the 57x57 grid, scaled linearly to [-1, 1] over its extent."""
    t = times.astype("datetime64[h]")
    day = t.astype("datetime64[D]")
    hour = (t - day).astype(np.float64)
    doy = (day - t.astype("datetime64[Y]")).astype(np.float64)
    scale = lambda c: 2 * (c - c[0]) / (c[-1] - c[0]) - 1
    out = np.empty((len(t), 6, 57, 57), np.float32)
    for k, v in enumerate((np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                           np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25))):
        out[:, k] = v[:, None, None]
    out[:, 4] = scale(FINE_LAT)[:, None]
    out[:, 5] = scale(FINE_LON)[None, :]
    return out


def raw_channels(data: Path, period: Period) -> np.ndarray:
    """(hours, 16, 57, 57) float32 in CHANNELS order: the ten ERA5 channels in physical units, then
    `clock_and_position`."""
    ds = open_store(data, "era5_native").sel(time=period.hours)
    assert np.allclose(ds.latitude.values, NATIVE_LAT) and np.allclose(ds.longitude.values, NATIVE_LON)
    a = upsample_matrix().astype(np.float32)
    up = lambda f: np.einsum("ij,tjk,lk->til", a, f, a, optimize=True).astype(np.float32)
    out = np.empty((len(period.hours), len(CHANNELS), 57, 57), np.float32)
    for h, (iu, iv, iws, isin, icos) in {"10": (0, 1, 4, 6, 7), "100": (2, 3, 5, 8, 9)}.items():
        u, v = ds[f"u{h}"].values.astype(np.float32), ds[f"v{h}"].values.astype(np.float32)
        out[:, iu], out[:, iv], out[:, iws] = up(u), up(v), up(np.hypot(u, v))
        s = np.hypot(out[:, iu], out[:, iv])
        calm = s < CALM
        safe = np.where(calm, 1.0, s)
        out[:, isin] = np.where(calm, 0.0, -out[:, iu] / safe)
        out[:, icos] = np.where(calm, 0.0, -out[:, iv] / safe)
    out[:, len(ERA5_CHANNELS):] = clock_and_position(period.hours)
    return out


def icon(data: Path, period: Period) -> np.ndarray:
    """(hours, 2, 41, 41) ICON speed at the HEIGHTS, m/s."""
    ds = open_store(data, "icon").sel(time=period.hours)
    assert np.allclose(ds.latitude.values, GRID_LAT) and np.allclose(ds.longitude.values, GRID_LON)
    return np.stack([ds[f"wind_speed_{h}"].values for h in HEIGHTS], 1).astype(np.float32)


def era5_speed(raw: np.ndarray) -> np.ndarray:
    """(hours, 2, 41, 41) ERA5 speed at the HEIGHTS on the output window, m/s."""
    return crop(raw[:, SPEED])


@dataclass(frozen=True)
class Norm:
    """z-score of the first ZSCORED channels, and per height the scale shared by ERA5 and ICON speed
    (ws_mean, ws_std in HEIGHTS order). Fit hours only."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    ws_mean: tuple[float, ...]
    ws_std: tuple[float, ...]

    @staticmethod
    def fit(raw: np.ndarray, icon_fit: np.ndarray) -> "Norm":
        r = raw[:, :ZSCORED].astype(np.float64)
        ws = [np.concatenate([e.ravel(), i.ravel()]).astype(np.float64)
              for e, i in zip(era5_speed(raw).swapaxes(0, 1), icon_fit.swapaxes(0, 1))]
        return Norm(tuple(r.mean((0, 2, 3))), tuple(r.std((0, 2, 3))),
                    tuple(float(w.mean()) for w in ws), tuple(float(w.std()) for w in ws))

    def inputs(self, raw: np.ndarray) -> np.ndarray:
        x = raw.copy()
        m, s = np.array(self.mean, np.float32)[:, None, None], np.array(self.std, np.float32)[:, None, None]
        x[:, :ZSCORED] = (x[:, :ZSCORED] - m) / s
        return x

    def _ws(self) -> tuple[np.ndarray, np.ndarray]:
        return tuple(np.array(v, np.float32)[:, None, None] for v in (self.ws_mean, self.ws_std))

    def to_scale(self, ws: np.ndarray) -> np.ndarray:
        """(hours, 2, 41, 41) m/s to the shared scale of each height."""
        m, s = self._ws()
        return ((ws - m) / s).astype(np.float32)

    def to_ms(self, z: np.ndarray) -> np.ndarray:
        m, s = self._ws()
        return z * s + m

    def save(self, path: str) -> None:
        json.dump(asdict(self), open(path, "w"), indent=1)

    @staticmethod
    def load(path: str) -> "Norm":
        d = json.load(open(path))
        return Norm(tuple(d["mean"]), tuple(d["std"]), tuple(d["ws_mean"]), tuple(d["ws_std"]))


@dataclass(frozen=True)
class Fields:
    """One period ready for the network: inputs (hours, 16, 57, 57), and ERA5 and ICON speed on the output window
    (hours, 2, 41, 41), all in the normalised units; `times` are the hours."""

    times: np.ndarray
    x: np.ndarray
    base: np.ndarray
    target: Optional[np.ndarray]


def _fields(period: Period, norm: Norm, raw: np.ndarray, target: Optional[np.ndarray]) -> Fields:
    return Fields(period.hours, norm.inputs(raw), norm.to_scale(era5_speed(raw)),
                  None if target is None else norm.to_scale(target))


def fields(data: Path, period: Period, norm: Norm) -> Fields:
    """The network input of `period`, without a target."""
    return _fields(period, norm, raw_channels(data, period), None)


def fit_fields(data: Path, period: Period) -> tuple[Norm, Fields]:
    """The normalisation and the training fields, both from the hours of `period`."""
    raw, target = raw_channels(data, period), icon(data, period)
    norm = Norm.fit(raw, target)
    return norm, _fields(period, norm, raw, target)
