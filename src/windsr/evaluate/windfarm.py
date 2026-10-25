"""Wind-farm power evaluation of the paper (Ding & Hsieh 2026, Methods "Wind power generation" and "Wind power
estimation modeling", Eqs. 1, 2, 15, 16, Fig. 4). Reported only. It never decides a win; `decision.py` does.

Data shapes: observed and predicted capacity factors (CF) are (time, farm) arrays on the test hours, with farms in
the order of `data.taipower.FARMS`. A `Farm` (`data.taipower`) and a `Turbine` are frozen registry entries.
`Observations` holds the hourly observed CF with its coverage and the files it was read from.

Rules taken from the paper:
- Six Taipower farms. The power at HH:00 is the mean of the 10-minute observations from HH-0:30 to HH+0:30.
- CF = hourly MWh / (1 h x installed MW) (Eq. 2).
- Hub-height wind is the log interpolation between 10 m and 100 m (Eq. 1).
- Losses: WScorr = 0.93 WS below RS - 0.5, a linear ramp of the 7% loss to 0 from RS - 0.5 to RS + 2, and WS above
  (Eq. 15), with RS the rated wind speed. Power is the turbine curve at WScorr times (1 - 0.016)^Y (Eq. 16).
- RMSE of the hourly CF (Eq. 9). Paired comparisons by the Wilcoxon signed-rank test. RMSE by bins of observed CF,
  and RMSE against the temporal aggregation window with a logarithmic fit.

Choices the paper leaves open, fixed here:
- The hourly window holds the seven 10-minute stamps HH-0:30, HH-0:20, ..., HH+0:30 with equal weights. An hour is
  observed only when all seven values are valid.
- A 10-minute value is valid when the dataset marks it `qa_exclude` False and `eligible_reported_item_analysis` True
  and it lies outside every run the dataset flags as `intrinsic-constant-nonzero-run-24h-v1` (a frozen value) or
  `intrinsic-zero-run-7d-v1` (a week or more at zero). Negative net generation, flagged as
  `intrinsic-negative-generation-v1`, is kept, and the hourly CF is clipped at 0.
- Installed MW is the row's `capacity_mw`, or the farm's registry capacity where the row has none.
- Each 10-minute value is converted from Asia/Taipei to UTC. `lag_scan` checks the alignment against product wind.
- RS is the first curve speed at which the curve reaches the turbine's nameplate power. The curve is interpolated
  linearly, is 0 below its first point and stays at its last value above its last point. The paper names no
  cut-out speed.
- A farm's predicted CF is one turbine's power over its nameplate power, so every turbine of a farm is the farm's
  model in the registry.
- Y is the UTC calendar year of the hour minus the commissioning year, at least 0.
- Product wind is the nearest grid cell to the farm, clipped at 0. ERA5-Land has no 100 m wind and is left out.
- Metrics use the hours where the observation and every product are finite. A farm is scored when at least
  `MIN_HOURS_FRACTION` of the test hours are observed, because CF is strongly seasonal.
- Wilcoxon: `scipy.stats.wilcoxon` on the paired hourly squared errors, two-sided, zero differences dropped.
- CF bins have the edges of published FIG4 (0, 0.1, ..., 1.0), closed on the left. The last bin also holds every
  observed CF above 1.
- Aggregation: consecutive blocks of `window` hours from the first scored hour. A block is kept when all its hours are
  in the scored sample.
- Blocks and lags count clock hours. Where the scored hours have gaps, the hours between them are unobserved, so no
  block and no lag pair joins two separate stretches. A block's observed and predicted CF are block means. The fit
  is least squares of RMSE = a + b ln(window).
"""
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ..data.stations import Stations
from ..data.store import open_store
from ..data.taipower import FARMS, RUNS, Farm, read
from ..grid import sample_grid
from ..periods import Period, on_hours
from .products import open_prediction, prediction_grid

LOSS = 0.07
LOSS_RAMP_START = -0.5
LOSS_RAMP_END = 2.0
DEGRADATION = 0.016
HUB_LOW, HUB_HIGH = 10.0, 100.0
WINDOW_OFFSETS = np.arange(-30, 31, 10).astype("timedelta64[m]")
CF_EDGES = np.round(np.arange(0, 1.01, 0.1), 1)
WINDOWS = (1, 3, 6, 12, 24, 72, 168)
LAGS = tuple(range(-12, 13))
MIN_HOURS_FRACTION = 0.5


@dataclass(frozen=True)
class Turbine:
    """A power curve in kW at wind speeds in m/s, and the nameplate power in kW."""

    name: str
    speed: tuple[float, ...]
    power_kw: tuple[float, ...]
    nameplate_kw: float

    @property
    def rated_speed(self) -> float:
        return float(np.asarray(self.speed)[np.asarray(self.power_kw) >= self.nameplate_kw][0])

    def cf(self, ws: np.ndarray) -> np.ndarray:
        """Power over nameplate power at wind speed `ws`, without losses."""
        return np.interp(ws, self.speed, self.power_kw, left=0.0, right=self.power_kw[-1]) / self.nameplate_kw


def _turbine(name: str, nameplate_kw: float, curve: Mapping[float, float]) -> Turbine:
    speed = tuple(sorted(curve))
    return Turbine(name, tuple(float(s) for s in speed), tuple(float(curve[s]) for s in speed), nameplate_kw)


TURBINES = {t.name: t for t in (
    _turbine("Vestas V80", 2000, {
        1: 0, 2: 0, 3: 0, 3.5: 35, 4: 70, 4.5: 117, 5: 165, 5.5: 225, 6: 285, 6.5: 372, 7: 459, 7.5: 580, 8: 701,
        8.5: 832, 9: 964, 9.5: 1127, 10: 1289, 10.5: 1428, 11: 1567, 11.5: 1678, 12: 1788, 12.5: 1865, 13: 1941,
        13.5: 1966, 14: 1990, 14.5: 2000, 15: 2000, 30: 2000}),
    _turbine("Enercon E70", 2300, {
        1: 0, 2: 0, 2.5: 10, 3: 18, 3.5: 37, 4: 56, 4.5: 91, 5: 127, 5.5: 183, 6: 240, 6.5: 320, 7: 400, 7.5: 513,
        8: 626, 8.5: 759, 9: 892, 9.5: 1058, 10: 1223, 10.5: 1407, 11: 1590, 11.5: 1745, 12: 1900, 12.5: 1990,
        13: 2080, 13.5: 2155, 14: 2230, 14.5: 2265, 15: 2300, 16: 2310, 30: 2310}),
    _turbine("HTW5.2-127", 5200, {
        1: 0, 4: 0, 4.5: 150, 5: 300, 5.5: 440, 6: 599, 6.5: 830, 7: 1073, 7.5: 1400, 8: 1735, 8.5: 2130,
        9: 2524, 9.5: 2885, 10: 3218, 10.5: 3610, 11: 3975, 11.5: 4330, 12: 4621, 12.5: 4850, 13: 5020,
        13.5: 5140, 14: 5200, 30: 5200}),
    _turbine("SWT-6.0-154", 6000, {
        1: 0, 3: 0, 3.5: 100, 4: 220, 4.5: 320, 5: 440, 5.5: 575, 6: 721, 6.5: 945, 7: 1173, 7.5: 1485,
        8: 1796, 8.5: 2157, 9: 2517, 9.5: 2940, 10: 3360, 10.5: 3930, 11: 4485, 11.5: 5160, 12: 5792,
        12.5: 5960, 13: 6000, 25: 6000}),
)}


def hub_wind(ws10: np.ndarray, ws100: np.ndarray, height: float) -> np.ndarray:
    """Eq. 1 between 10 m and 100 m."""
    w = (np.log(height) - np.log(HUB_LOW)) / (np.log(HUB_HIGH) - np.log(HUB_LOW))
    return ws10 + w * (ws100 - ws10)


def loss_corrected(ws: np.ndarray, rated_speed: float) -> np.ndarray:
    """Eq. 15."""
    lo, hi = rated_speed + LOSS_RAMP_START, rated_speed + LOSS_RAMP_END
    loss = np.where(ws < lo, LOSS, np.where(ws <= hi, LOSS * (1 - (ws - lo) / (hi - lo)), 0.0))
    return (1 - loss) * ws


def degradation(years: np.ndarray) -> np.ndarray:
    """The factor (1 - T_degrad)^Y of Eq. 16."""
    return (1 - DEGRADATION) ** np.asarray(years, float)


def operational_years(times: np.ndarray, commissioned: int) -> np.ndarray:
    return np.maximum(pd.DatetimeIndex(times).year.to_numpy() - commissioned, 0)


def predicted_cf(ws10: np.ndarray, ws100: np.ndarray, times: np.ndarray, farm: Farm) -> np.ndarray:
    """Eqs. 1, 15 and 16: the CF predicted from (time,) 10 m and 100 m wind at one farm."""
    turbine = TURBINES[farm.turbine]
    ws = loss_corrected(hub_wind(ws10, ws100, farm.hub_m), turbine.rated_speed)
    return turbine.cf(ws) * degradation(operational_years(times, farm.commissioned))


def hourly_cf(times: np.ndarray, cf: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """Eq. 2 on the symmetric window: the mean 10-minute CF from HH-0:30 to HH+0:30 at each of `hours`, NaN unless
    all seven values are finite, clipped at 0. `times` are unique 10-minute stamps with CF `cf` (NaN when invalid)."""
    s = pd.Series(cf, index=pd.DatetimeIndex(times))
    if not s.index.is_unique:
        raise ValueError("10-minute stamps repeat")
    stamps = (np.asarray(hours, "datetime64[ns]")[:, None] + WINDOW_OFFSETS[None]).ravel()
    window = s.reindex(stamps).to_numpy(float).reshape(len(hours), len(WINDOW_OFFSETS))
    return np.clip(window.mean(1), 0, None)


@dataclass(frozen=True)
class Observations:
    """Hourly observed CF, (time, farm) on `hours`, with per-farm coverage and the sha256 of every file read."""

    hours: np.ndarray
    cf: np.ndarray
    coverage: pd.DataFrame
    release_id: str
    release_path: str
    files: dict[str, str]


def observations(root: str, period: Period, farms: Sequence[Farm] = FARMS) -> Observations:
    """Hourly observed CF of `farms` on the hours of `period`, from the Taipower release at `root` as `read` reads
    it."""
    release = read(root, period, [f.unit_name for f in farms])
    rows, runs = release.rows, release.runs
    hours = period.hours
    cf = np.full((len(hours), len(farms)), np.nan)
    coverage = []
    for j, farm in enumerate(farms):
        r = rows[rows.unit_name == farm.unit_name].sort_values("time")
        t = r.time.to_numpy()
        in_run = np.zeros(len(r), bool)
        excluded = {}
        for rule in RUNS:
            hit = np.zeros(len(r), bool)
            for run in runs[(runs.unit_name == farm.unit_name) & (runs.rule_id == rule)].itertuples():
                hit |= (t >= np.datetime64(run.run_first_timestamp)) & (t <= np.datetime64(run.run_last_timestamp))
            excluded[rule] = int(hit.sum())
            in_run |= hit
        mw = r.net_generation_mw.astype(float).to_numpy()
        row_capacity = r.capacity_mw.astype(float).to_numpy()
        capacity = np.where(np.isfinite(row_capacity), row_capacity, farm.capacity_mw)
        flagged = r.qa_exclude.to_numpy(bool) | ~r.eligible_reported_item_analysis.to_numpy(bool)
        valid = ~flagged & ~in_run & np.isfinite(mw)
        cf[:, j] = hourly_cf(t, np.where(valid, mw / capacity, np.nan), hours)
        observed = np.isfinite(cf[:, j])
        coverage.append({
            "farm": farm.unit_name, "name": farm.name, "records": len(r), "records_valid": int(valid.sum()),
            "records_qa_exclude_or_ineligible": int(flagged.sum()),
            **{f"records_in_{rule}": n for rule, n in excluded.items()},
            "records_negative": int((mw < 0).sum()), "records_row_capacity": int(np.isfinite(row_capacity).sum()),
            "records_registry_capacity": int((~np.isfinite(row_capacity)).sum()),
            "capacity_values_mw": " ".join(f"{c:g}" for c in np.unique(capacity)),
            "test_hours": len(hours), "hours_observed": int(observed.sum()),
            "hours_fraction": float(observed.mean()),
            "first_hour": str(hours[observed][0]) if observed.any() else "",
            "last_hour": str(hours[observed][-1]) if observed.any() else "",
            "months_with_hours": int(pd.DatetimeIndex(hours[observed]).to_period("M").nunique()),
            "hours_cf_above_1": int((cf[observed, j] > 1).sum()),
            "max_cf": float(np.nanmax(cf[:, j])) if observed.any() else np.nan,
            "scored": bool(observed.mean() >= MIN_HOURS_FRACTION),
        })
    return Observations(hours, cf, pd.DataFrame(coverage), release.release_id, release.path, release.files)


class NoHubWind(ValueError):
    """The store has no wind_speed_100m, so hub-height wind (Eq. 1) cannot be formed."""


def _stations(farms: Sequence[Farm]) -> Stations:
    return Stations(np.array([f.unit_name for f in farms]), np.array([f.lat for f in farms]),
                    np.array([f.lon for f in farms]))


def rival_wind(data: Path, name: str, period: Period, farms: Sequence[Farm] = FARMS) -> tuple[np.ndarray, np.ndarray]:
    """(time, farm) 10 m and 100 m wind of DATA's `era5` or `icon`."""
    ds = open_store(data, name)
    st = _stations(farms)
    return tuple(np.clip(sample_grid(on_hours(ds[v], period).load(), st, "nearest"), 0, None)
                 for v in ("wind_speed_10m", "wind_speed_100m"))


def store_wind(path: str, period: Period, farms: Sequence[Farm] = FARMS) -> tuple[np.ndarray, np.ndarray]:
    """(time, farm) 10 m and 100 m wind of a prediction store."""
    ds = open_prediction(path)
    if "wind_speed_100m" not in ds:
        raise NoHubWind(f"{path} has {sorted(ds.data_vars)} but no wind_speed_100m")
    st = _stations(farms)
    return tuple(np.clip(sample_grid(prediction_grid(path, period, v), st, "nearest"), 0, None)
                 for v in ("wind_speed_10m", "wind_speed_100m"))


def product_cf(ws10: np.ndarray, ws100: np.ndarray, hours: np.ndarray, farms: Sequence[Farm] = FARMS) -> np.ndarray:
    return np.stack([predicted_cf(ws10[:, j], ws100[:, j], hours, f) for j, f in enumerate(farms)], 1)


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def wilcoxon_p(err_a: np.ndarray, err_b: np.ndarray) -> float:
    d = err_a - err_b
    return float(wilcoxon(d).pvalue) if np.count_nonzero(d) > 0 else np.nan


def _pair_rows(obs: np.ndarray, preds: Mapping[str, np.ndarray], **key) -> list[dict]:
    rows = []
    for a, b in combinations(preds, 2):
        ea, eb = (preds[a] - obs) ** 2, (preds[b] - obs) ** 2
        ra, rb = rmse(preds[a], obs), rmse(preds[b], obs)
        rows.append({**key, "a": a, "b": b, "n": len(obs), "rmse_a": ra, "rmse_b": rb,
                     "pct_b_vs_a": 100 * (rb / ra - 1) if ra > 0 else np.nan, "wilcoxon_p": wilcoxon_p(ea, eb)})
    return rows


def bin_labels(edges: np.ndarray = CF_EDGES) -> list[str]:
    return [f"{a:.1f}-{b:.1f}" for a, b in zip(edges[:-1], edges[1:])]


def cf_bin(obs: np.ndarray, edges: np.ndarray = CF_EDGES) -> np.ndarray:
    """Bin index of each observed CF: [edge_i, edge_i+1), with every value from the last inner edge up in the last bin."""
    return np.clip(np.searchsorted(edges, obs, side="right") - 1, 0, len(edges) - 2)


def block_means(x: np.ndarray, window: int) -> np.ndarray:
    """(time, ...) -> (block, ...) means of consecutive `window`-hour blocks. A trailing partial block is dropped,
    and a block with any NaN is NaN."""
    n = len(x) // window
    return x[:n * window].reshape(n, window, *x.shape[1:]).mean(1)


@dataclass(frozen=True)
class LogFit:
    """RMSE = a + b ln(window), least squares, with R^2."""

    a: float
    b: float
    r2: float


def log_fit(windows: np.ndarray, values: np.ndarray) -> LogFit:
    """Fitted over the windows with a finite RMSE."""
    keep = np.isfinite(values)
    x, values = np.log(np.asarray(windows, float))[keep], np.asarray(values, float)[keep]
    b, a = np.polyfit(x, values, 1)
    ss_res = float(((values - (a + b * x)) ** 2).sum())
    ss_tot = float(((values - values.mean()) ** 2).sum())
    return LogFit(float(a), float(b), 1 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def lag_scan(obs: np.ndarray, pred: np.ndarray, lags: Sequence[int] = LAGS) -> pd.Series:
    """Correlation of obs(t) with pred(t + lag) over the hours where both are finite, per lag in hours."""
    out = {}
    for k in lags:
        o, p = (obs[:len(obs) - k], pred[k:]) if k >= 0 else (obs[-k:], pred[:k])
        m = np.isfinite(o) & np.isfinite(p)
        out[k] = float(np.corrcoef(o[m], p[m])[0, 1]) if m.sum() > 2 else np.nan
    return pd.Series(out, name="corr").rename_axis("lag_h")


@dataclass(frozen=True)
class FarmResults:
    metrics: pd.DataFrame
    pairs: pd.DataFrame
    bins: pd.DataFrame
    aggregation: pd.DataFrame
    lags: pd.DataFrame


def evaluate(obs: Observations, preds: Mapping[str, np.ndarray], farms: Sequence[Farm] = FARMS,
             windows: Sequence[int] = WINDOWS, lag_product: str = "era5") -> FarmResults:
    """Every reported table for the scored farms: per-farm RMSE, paired Wilcoxon tests overall and per CF bin,
    RMSE per aggregation window with its log fit, and the lag scan of `lag_product`."""
    clock = np.arange(obs.hours[0], obs.hours[-1] + np.timedelta64(1, "h"), np.timedelta64(1, "h"))
    at = np.searchsorted(clock, obs.hours)

    def on_clock(x: np.ndarray) -> np.ndarray:
        out = np.full((len(clock), *x.shape[1:]), np.nan)
        out[at] = x
        return out

    obs = replace(obs, hours=clock, cf=on_clock(obs.cf))
    preds = {n: on_clock(p) for n, p in preds.items()}
    metrics, pairs, bins, agg, lags = [], [], [], [], []
    labels = bin_labels()
    for j, farm in enumerate(farms):
        lag = lag_scan(obs.cf[:, j], preds[lag_product][:, j]).reset_index()
        lags.append(lag.assign(farm=farm.unit_name, product=lag_product))
        if not obs.coverage.scored.iloc[j]:
            continue
        o = obs.cf[:, j]
        sample = np.isfinite(o) & np.all([np.isfinite(p[:, j]) for p in preds.values()], 0)
        oj, pj = o[sample], {n: p[sample, j] for n, p in preds.items()}
        for n, p in pj.items():
            metrics.append({"farm": farm.unit_name, "product": n, "n": int(sample.sum()), "rmse": rmse(p, oj),
                            "bias": float((p - oj).mean()), "corr": float(np.corrcoef(p, oj)[0, 1]),
                            "mean_obs": float(oj.mean()), "mean_pred": float(p.mean())})
        pairs += _pair_rows(oj, pj, farm=farm.unit_name, bin="all")
        b = cf_bin(oj)
        for i, label in enumerate(labels):
            k = b == i
            for n, p in pj.items():
                bins.append({"farm": farm.unit_name, "bin": label, "product": n, "n": int(k.sum()),
                             "rmse": rmse(p[k], oj[k]) if k.any() else np.nan})
            if k.sum() > 1:
                pairs += _pair_rows(oj[k], {n: p[k] for n, p in pj.items()}, farm=farm.unit_name, bin=label)
        masked = {n: np.where(sample, p[:, j], np.nan) for n, p in preds.items()}
        o_masked = np.where(sample, o, np.nan)
        rows = []
        for w in windows:
            ob = block_means(o_masked, w)
            keep = np.isfinite(ob)
            for n, p in masked.items():
                rows.append({"farm": farm.unit_name, "product": n, "window_h": w, "blocks": int(keep.sum()),
                             "rmse": rmse(block_means(p, w)[keep], ob[keep])})
        rows = pd.DataFrame(rows)
        for n, g in rows.groupby("product", sort=False):
            f = log_fit(g.window_h.to_numpy(), g.rmse.to_numpy())
            rows.loc[g.index, ["fit_a", "fit_b", "fit_r2"]] = (f.a, f.b, f.r2)
        agg.append(rows)
    return FarmResults(pd.DataFrame(metrics), pd.DataFrame(pairs), pd.DataFrame(bins),
                       pd.concat(agg, ignore_index=True) if agg else pd.DataFrame(), pd.concat(lags, ignore_index=True))
