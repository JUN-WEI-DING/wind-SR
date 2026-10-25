"""Station truth.

Data shape: a `Series` is a (time, station) float64 array over one period, with its hour axis and station ids.
Truth and every product share that shape, so scoring never needs to know where a series came from.

Truth is the hourly CODIS `wind_speed_raw` of the 345 stations in UTC. Negative values (missing codes), the hours
of `stuck_mask` (a run of 12 or more hours at one value above 0, or 48 or more at exactly 0) and values below
0.1 m/s are NaN.
"""
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np

from ..data.stations import stations
from ..data.store import open_store
from ..periods import Period, on_hours

CALM_THRESHOLD = 0.1
StationSet = Literal["main", "sensitivity"]
LEVEL_SHIFT_STATIONS = (
    "C0D590 C0E780 C0F9O0 C0F9P0 C0F9U0 C0G620 C0G780 C0G800 C0K510 C0M650 C0O960 C0O980 "
    "C0O990 C0R130 C0R620 C0T9D0 C0V770 C0X190"
).split()
"""The 18 stations with level shifts that the sensitivity station set leaves out."""


@dataclass(frozen=True)
class Series:
    values: np.ndarray
    times: np.ndarray
    station_ids: np.ndarray

    def __post_init__(self):
        if self.values.shape != (len(self.times), len(self.station_ids)):
            raise ValueError(f"values {self.values.shape} do not match {len(self.times)} hours x {len(self.station_ids)} stations")

    def subset(self, keep: np.ndarray) -> "Series":
        return Series(self.values[:, keep], self.times, self.station_ids[keep])


def station_set_mask(station_set: StationSet, ids: np.ndarray) -> np.ndarray:
    """main: all stations; sensitivity: all but the LEVEL_SHIFT_STATIONS."""
    if station_set == "main":
        return np.ones(len(ids), bool)
    return ~np.isin(ids, LEVEL_SHIFT_STATIONS)


@cache
def truth(data: Path, period: Period) -> Series:
    """Station truth on the hours of `period`. Cached; callers must not write into the returned arrays."""
    ds = open_store(data, "stations")
    y = on_hours(ds.wind_speed_raw, period).transpose("time", "station_id").values.astype(np.float64)
    stuck = on_hours(ds.stuck_mask, period).transpose("time", "station_id").values
    y[stuck.astype(bool)] = np.nan
    y[y < 0] = np.nan
    y[y < CALM_THRESHOLD] = np.nan
    return Series(y, period.hours, stations(data).ids)
