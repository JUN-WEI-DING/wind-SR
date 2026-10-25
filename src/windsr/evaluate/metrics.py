"""Per-station metrics on the test year, raw (V0) and calibrated on the validation weeks (V1 to V3).

Pure numpy on (time, station) arrays, then the scoring of one product: its station series on the test year and on
the validation weeks against station truth.
"""
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.stations import stations
from ..periods import TEST, VALID
from .products import Loader, Sampling
from .truth import Series, StationSet, station_set_mask, truth

VARIANTS = ("V0", "V1", "V2", "V3")
MIN_STATION_HOURS = 720


def per_station(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """rmse, bias, corr, crmse and n per station over the hours where model x and truth y are both finite."""
    v = np.isfinite(x) & np.isfinite(y)
    n = v.sum(0)
    xs, ys = np.where(v, x, np.nan), np.where(v, y, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = xs - ys
        rmse = np.sqrt(np.nanmean(d ** 2, 0))
        bias = np.nanmean(d, 0)
        xa, ya = xs - np.nanmean(xs, 0), ys - np.nanmean(ys, 0)
        den = np.sqrt(np.nanmean(xa ** 2, 0)) * np.sqrt(np.nanmean(ya ** 2, 0))
        corr = np.where(den > 0, np.nanmean(xa * ya, 0) / den, np.nan)
    crmse = np.sqrt(np.maximum(rmse ** 2 - bias ** 2, 0))
    return {"rmse": rmse, "bias": bias, "corr": corr, "crmse": crmse, "n": n}


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xm, ym = x.mean(), y.mean()
    b = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
    return float(ym - b * xm), float(b)


@dataclass(frozen=True)
class Calibration:
    """V1 y = k x, V2 y = a + b x (pooled), V3 y = a_s + b_s x per station (global V2 below 720 hours)."""

    k: float
    a: float
    b: float
    a_station: np.ndarray
    b_station: np.ndarray
    n_fallback: int

    def apply(self, variant: str, x: np.ndarray) -> np.ndarray:
        if variant == "V0":
            return x
        if variant == "V1":
            return x * self.k
        if variant == "V2":
            return np.clip(self.a + self.b * x, 0, None)
        if variant == "V3":
            return np.clip(self.a_station + self.b_station * x, 0, None)
        raise ValueError(f"unknown variant {variant!r}; known: {VARIANTS}")


def fit_calibration(model: Series, truth: Series) -> Calibration:
    """Fit on validation-week station hours only. Any other hour is an error."""
    for name, s in (("model", model), ("truth", truth)):
        outside = ~VALID.contains(s.times)
        if outside.any():
            raise ValueError(f"calibration {name} series has {int(outside.sum())} hours outside the validation weeks, "
                             f"first {s.times[outside][0]}")
    if not (np.array_equal(model.times, truth.times) and np.array_equal(model.station_ids, truth.station_ids)):
        raise ValueError("model and truth series are not aligned")
    x, y = model.values, truth.values
    v = np.isfinite(x) & np.isfinite(y)
    k = float((x[v] * y[v]).sum() / (x[v] ** 2).sum())
    a, b = _ols(x[v], y[v])
    a_s, b_s, nf = np.full(x.shape[1], a), np.full(x.shape[1], b), 0
    for j in range(x.shape[1]):
        vj = v[:, j]
        if vj.sum() < MIN_STATION_HOURS or np.ptp(x[vj, j]) == 0:
            nf += 1
            continue
        a_s[j], b_s[j] = _ols(x[vj, j], y[vj, j])
    return Calibration(k, a, b, a_s, b_s, nf)


@dataclass(frozen=True)
class ProductScore:
    name: str
    station_ids: np.ndarray
    by_variant: dict[str, dict[str, np.ndarray]]


def score_product(data: Path, name: str, loader: Loader, sampling: Sampling = "nearest",
                  station_set: StationSet = "main") -> ProductScore:
    """Per-station metrics of one product on the test year, V1 to V3 fitted on the validation weeks."""
    st = stations(data)
    keep = station_set_mask(station_set, st.ids)
    y = truth(data, TEST).subset(keep)
    x = Series(loader(st, TEST, sampling), TEST.hours, st.ids).subset(keep)
    by_variant = {"V0": per_station(x.values, y.values)}
    x_valid = Series(loader(st, VALID, sampling), VALID.hours, st.ids).subset(keep)
    cal = fit_calibration(x_valid, truth(data, VALID).subset(keep))
    for v in VARIANTS[1:]:
        by_variant[v] = per_station(cal.apply(v, x.values), y.values)
    return ProductScore(name, y.station_ids, by_variant)


def summary(scores: list[ProductScore]) -> pd.DataFrame:
    """One row per product and variant: the station means of rmse, bias, corr and crmse, and the number of stations
    with a finite RMSE."""
    rows = []
    for s in scores:
        for variant, m in s.by_variant.items():
            rows.append({"product": s.name, "variant": variant,
                         **{k: float(np.nanmean(m[k])) for k in ("rmse", "bias", "corr", "crmse")},
                         "n_stations": int(np.isfinite(m["rmse"]).sum())})
    return pd.DataFrame(rows)


def per_station_table(scores: list[ProductScore]) -> pd.DataFrame:
    frames = [pd.DataFrame({"product": s.name, "variant": v, "station_id": s.station_ids, **m})
              for s in scores for v, m in s.by_variant.items()]
    return pd.concat(frames, ignore_index=True)
