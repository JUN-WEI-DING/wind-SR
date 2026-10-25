"""Monte Carlo cross-scale evaluation of the paper (Ding & Hsieh 2026, Methods, Eqs. 10-14, Fig. 8).
Reported only. It never decides a win; `decision.py` does.

Data shapes: truth and every product are (time, station) arrays on the same hours and stations. A `UnitTable`
holds one product's unit value for every 0.125 deg cell and every nonempty subset of its stations, indexed by
the subset's bitmask, so a draw of stations becomes one bitmask per cell and a table lookup.

Rules taken from the paper:
- d_i is the mean length of the Delaunay edges from station i to its neighbours N(i) (Eq. 10).
- d'_i = d_i / sum_{j in N(i)} d_j (Eq. 11), and the sampling probability is P_i = d'_i / sum_j d'_j (Eq. 12).
- A draw takes n stations by these probabilities without replacement. Sampled stations in one grid cell are
  averaged into one representative value. The draw value is the mean RMSE over these units (Eq. 13).
- Convergence is a relative change of the running mean of at most 0.001 between successive draws (Eq. 14).
- A quadratic trend of the mean RMSE against n is fitted per product, and crossovers are where two fits meet.

Choices the paper leaves open, fixed here:
- Delaunay and edge lengths use km on an equirectangular projection at the mean station latitude.
- Cells are `grid.nearest_cell`. A unit's hourly truth and product value are the means over its sampled
  stations' finite values, and its RMSE is over the hours where both are finite.
- Each sample size n = 1..N gets a fixed number of draws from `numpy.random.default_rng(seed)`, with
  `rng.choice(N, n, replace=False, p=P)`. Every product is scored on the same draws. The Eq. 14 change at the
  last draw is reported, not used to stop.
- The trend is `np.polyfit(n, mean, 2)` over n = 1..N, with R^2 and the F-test of the quadratic against a
  constant (2 and len - 3 degrees of freedom). Crossovers are the real roots of two fits' difference in
  [0, 1800], flagged `extrapolated` above N.
"""
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.stats import f as f_dist

from ..grid import nearest_cell
from .groups import KM_PER_DEG
from .metrics import per_station

CONVERGENCE = 1e-3
CROSSOVER_RANGE = (0.0, 1800.0)
VARIANT = "paper"
"""The `variant` column of the tables: weighted sampling and the RMSE of each unit's averaged series, as in the
paper."""


@dataclass(frozen=True)
class Weights:
    """Delaunay spacing of each station: `d` in km (Eq. 10), `d_prime` (Eq. 11) and probability `p` (Eq. 12)."""

    d: np.ndarray
    d_prime: np.ndarray
    p: np.ndarray


def sampling_weights(lat: np.ndarray, lon: np.ndarray) -> Weights:
    x = lon * np.cos(np.radians(lat.mean())) * KM_PER_DEG
    y = lat * KM_PER_DEG
    tri = Delaunay(np.c_[x, y])
    edges = np.unique(np.sort(np.concatenate([tri.simplices[:, [i, j]] for i, j in ((0, 1), (1, 2), (0, 2))]), 1), axis=0)
    a, b = edges.T
    length = np.hypot(x[a] - x[b], y[a] - y[b])
    n = len(lat)
    d = (np.bincount(a, length, n) + np.bincount(b, length, n)) / (np.bincount(a, minlength=n) + np.bincount(b, minlength=n))
    d_prime = d / (np.bincount(a, d[b], n) + np.bincount(b, d[a], n))
    return Weights(d, d_prime, d_prime / d_prime.sum())


@dataclass(frozen=True)
class Cells:
    """The cell of each station as `index` (0..C-1) and its `slot`, the bit it sets in its cell's subset mask."""

    index: np.ndarray
    slot: np.ndarray

    @property
    def n_cells(self) -> int:
        return int(self.index.max()) + 1

    @property
    def max_slots(self) -> int:
        return int(self.slot.max()) + 1

    def masks(self, draws: np.ndarray) -> np.ndarray:
        """(draw, n) station indices -> (draw, cell) bitmask of the sampled stations in each cell."""
        flat = np.arange(len(draws))[:, None] * self.n_cells + self.index[draws]
        bits = np.bincount(flat.ravel(), (1 << self.slot[draws]).ravel(), len(draws) * self.n_cells)
        return bits.astype(np.int64).reshape(len(draws), self.n_cells)


def cells_of(lat: np.ndarray, lon: np.ndarray) -> Cells:
    _, index = np.unique(nearest_cell(lat, lon), return_inverse=True)
    slot = np.zeros(len(index), int)
    for c in range(index.max() + 1):
        slot[index == c] = np.arange((index == c).sum())
    return Cells(index, slot)


@dataclass(frozen=True)
class UnitTable:
    """`value[c, m]` is the unit RMSE of the stations of cell c in bitmask m. m = 0 and unused masks are NaN."""

    cells: Cells
    value: np.ndarray

    def draw_values(self, masks: np.ndarray) -> np.ndarray:
        """(draw, cell) bitmasks -> (draw,) mean unit RMSE over the cells each draw sampled (Eq. 13)."""
        v = self.value[np.arange(self.cells.n_cells)[:, None], masks.T]
        finite = np.isfinite(v)
        with np.errstate(invalid="ignore"):
            return np.where(finite, v, 0.0).sum(0) / finite.sum(0)


def _subset_means(a: np.ndarray, members: np.ndarray) -> np.ndarray:
    """Mean of the finite values over each subset's members: a (..., k), members (k, M) of 0/1 -> (..., M)."""
    finite = np.isfinite(a)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.where(finite, a, 0.0) @ members) / (finite @ members)


def unit_table(cells: Cells, x: np.ndarray, y: np.ndarray) -> UnitTable:
    """Unit RMSE of product x against truth y, both (time, station), for every cell and subset of its stations: the
    RMSE of the subset's averaged series."""
    k = cells.max_slots
    value = np.full((cells.n_cells, 1 << k), np.nan)
    for c in range(cells.n_cells):
        idx = np.flatnonzero(cells.index == c)
        idx = idx[np.argsort(cells.slot[idx])]
        members = ((np.arange(1 << len(idx))[None] >> np.arange(len(idx))[:, None]) & 1).astype(float)
        row = per_station(_subset_means(x[:, idx], members), _subset_means(y[:, idx], members))["rmse"]
        value[c, 1:len(row)] = row[1:]
    return UnitTable(cells, value)


def draw_summary(values: np.ndarray) -> dict[str, float]:
    """Mean, sd and the Eq. 14 relative change of the running mean between the last two draws. Computed on the
    values minus the first draw, so equal draws give an sd and a change of exactly 0."""
    first, shifted = values[0], values - values[0]
    if len(values) < 2:
        return {"mean": float(first), "sd": np.nan, "eq14_change": np.nan}
    before, after = first + shifted[:-1].mean(), first + shifted.mean()
    return {"mean": float(after), "sd": float(shifted.std(ddof=1)), "eq14_change": float(abs(after - before) / before)}


def curves(lat: np.ndarray, lon: np.ndarray, truth: np.ndarray, products: Mapping[str, np.ndarray], *,
           draws: int = 5000, seed: int = 0, sizes: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Mean and sd over `draws` draws of the draw value, per sample size n and product.

    The draws come from `default_rng(seed)`, the same station sets for every product. Every product needs a value
    at every station: a unit averages truth over all its stations, so a station without product values would pair
    the product with truth from other stations."""
    for name, x in products.items():
        empty = ~np.isfinite(x).any(0)
        if empty.any():
            raise ValueError(f"{name} has no value at {int(empty.sum())} stations; "
                             "run the curves on the stations where it has values")
    n_stations = len(lat)
    cells = cells_of(lat, lon)
    p = sampling_weights(lat, lon).p
    rows = []
    tables = {name: unit_table(cells, x, truth) for name, x in products.items()}
    rng = np.random.default_rng(seed)
    for n in sizes or range(1, n_stations + 1):
        masks = cells.masks(np.array([rng.choice(n_stations, n, replace=False, p=p) for _ in range(draws)]))
        for name, table in tables.items():
            rows.append({"variant": VARIANT, "n": n, "product": name, "draws": draws,
                         **draw_summary(table.draw_values(masks))})
    return pd.DataFrame(rows, columns=["variant", "n", "product", "mean", "sd", "draws", "eq14_change"])


@dataclass(frozen=True)
class Fit:
    """mean RMSE = coef[0] n^2 + coef[1] n + coef[2], with R^2 and the F-test of the quadratic against a constant."""

    coef: tuple[float, float, float]
    r2: float
    f: float
    p: float


def fit(n: np.ndarray, mean: np.ndarray) -> Fit:
    coef = np.polyfit(n, mean, 2)
    ss_res = float(((mean - np.polyval(coef, n)) ** 2).sum())
    ss_tot = float(((mean - mean.mean()) ** 2).sum())
    dof = len(n) - 3
    f = (ss_tot - ss_res) / 2 / (ss_res / dof) if ss_res > 0 else np.inf
    return Fit(tuple(float(c) for c in coef), 1 - ss_res / ss_tot, float(f), float(f_dist.sf(f, 2, dof)))


def crossovers(fits: Mapping[str, Fit], n_max: int, span: tuple[float, float] = CROSSOVER_RANGE) -> pd.DataFrame:
    """Where each pair of fitted curves meets in `span`, which product is lower on each side, and whether the
    crossover lies beyond the n_max stations observed."""
    rows = []
    lo, hi = span
    for a, b in combinations(fits, 2):
        diff = np.subtract(fits[a].coef, fits[b].coef)
        roots = np.roots(diff) if np.any(diff[:2]) else np.array([])
        real = np.sort(roots.real[np.abs(roots.imag) <= 1e-9 * np.maximum(1.0, np.abs(roots))])
        real = real[(real >= lo) & (real <= hi)]
        bounds = np.r_[lo, real, hi]
        for i, r in enumerate(real):
            below, above = np.polyval(diff, [(bounds[i] + r) / 2, (r + bounds[i + 2]) / 2])
            rows.append({"a": a, "b": b, "n": float(r), "extrapolated": bool(r > n_max),
                         "lower_below": a if below < 0 else b, "lower_above": a if above < 0 else b})
    return pd.DataFrame(rows, columns=["a", "b", "n", "extrapolated", "lower_below", "lower_above"])
