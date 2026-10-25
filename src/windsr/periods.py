"""The fixed hours of WindSR, all UTC.

The training period, 2023-01-01T00 to 2024-06-30T23, is cut into 28-day cycles counted from its start: days 0 to
18 of each cycle are fit hours, the only hours the network is trained on; days 20 to 26 are validation weeks, where
the calibrations V1 to V3 are fitted; days 19 and 27 are buffers, used for nothing. The test year,
2024-07-01T00 to 2025-06-30T23, is scored.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

CYCLE_DAYS = 28
FIT_DAYS = tuple(range(0, 19))
VALID_DAYS = tuple(range(20, 27))


@dataclass(frozen=True)
class Period:
    """Hours from `start` to `end`, optionally only some days of the 28-day cycle counted from `start`
    (`cycle_days`)."""

    name: str
    start: np.datetime64
    end: np.datetime64
    cycle_days: Optional[tuple[int, ...]] = None

    def _keep(self, t: np.ndarray) -> np.ndarray:
        keep = (t >= self.start) & (t <= self.end)
        if self.cycle_days is not None:
            day = (t - self.start) // np.timedelta64(1, "D")
            keep &= np.isin(day % CYCLE_DAYS, self.cycle_days)
        return keep

    @property
    def hours(self) -> np.ndarray:
        t = np.arange(self.start, self.end + np.timedelta64(1, "h"), np.timedelta64(1, "h")).astype("datetime64[ns]")
        return t[self._keep(t)]

    def contains(self, times: np.ndarray) -> np.ndarray:
        return self._keep(np.asarray(times).astype("datetime64[ns]"))


def on_hours(da, period: Period):
    """A DataArray with a time axis, on the hours of `period` (NaN where it has none)."""
    return da.reindex(time=period.hours)


_T0, _T1 = np.datetime64("2023-01-01T00", "ns"), np.datetime64("2024-06-30T23", "ns")
TRAIN = Period("train", _T0, _T1)
FIT = Period("fit", _T0, _T1, FIT_DAYS)
VALID = Period("valid", _T0, _T1, VALID_DAYS)
TEST = Period("test", np.datetime64("2024-07-01T00", "ns"), np.datetime64("2025-06-30T23", "ns"))
