"""The decision rule of the paper's clustering-based validation (`groups.py`).

Product A beats product B when, over the six station groups, (1) the mean of A's group RMSE minus B's
is below 0, (2) A's group RMSE is lower in more than half the groups, and (3) both still hold after
removing any single station. Group RMSE is the mean raw station RMSE. Correlation, the per-group
Wilcoxon p-values, the inter-group variance and the Friedman test are reported, not decided on."""
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

from .metrics import ProductScore


@dataclass(frozen=True)
class StationScores:
    """One product: raw RMSE and correlation per station."""

    station_ids: np.ndarray
    rmse: np.ndarray
    corr: np.ndarray

    def __post_init__(self):
        if not len(self.station_ids) == len(self.rmse) == len(self.corr):
            raise ValueError(f"{len(self.station_ids)} station ids, {len(self.rmse)} RMSE values, {len(self.corr)} correlations")


def station_scores(score: ProductScore) -> StationScores:
    """The decision inputs of one scored product: raw (V0) RMSE and correlation per station."""
    return StationScores(score.station_ids, score.by_variant["V0"]["rmse"], score.by_variant["V0"]["corr"])


@dataclass(frozen=True)
class GroupComparison:
    group: int
    n_stations: int
    rmse_a: float
    rmse_b: float
    wilcoxon_p: float
    corr_a: float
    corr_b: float

    @property
    def rmse_diff(self) -> float:
        return self.rmse_a - self.rmse_b


@dataclass(frozen=True)
class Decision:
    verdict: Literal["better", "not better"]
    reasons: tuple[str, ...]
    groups: tuple[GroupComparison, ...]
    n_stations: int
    rmse_diff: float
    corr_diff: float
    groups_won: int
    var_a: float
    var_b: float
    loo_breakers: tuple[str, ...]


def _wilcoxon_p(x: np.ndarray, y: np.ndarray) -> float:
    return 1.0 if np.all(x == y) else float(wilcoxon(x, y).pvalue)


def _nanmean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def _finite_by_group(inputs: Sequence[StationScores], groups: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """The group of each station, and which stations have a finite RMSE in every input."""
    ids = inputs[0].station_ids
    if any(not np.array_equal(s.station_ids, ids) for s in inputs[1:]):
        raise ValueError("inputs cover different stations or a different station order")
    missing = [s for s in ids if s not in groups]
    if missing:
        raise ValueError(f"{len(missing)} stations have no group, e.g. {missing[:3]}")
    ok = np.logical_and.reduce([np.isfinite(s.rmse) for s in inputs])
    if not ok.any():
        raise ValueError("no station is finite in every input")
    return np.array([groups[s] for s in ids]), ok


def _conditions(rmse_a: np.ndarray, rmse_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Conditions 1 (mean) and 2 (majority) on group RMSEs along the last axis. A NaN group (emptied) is left out."""
    present = np.isfinite(rmse_a)
    n_groups = present.sum(-1)
    diff_sum = np.where(present, rmse_a - rmse_b, 0.0).sum(-1)
    return (n_groups > 0) & (diff_sum < 0), (rmse_a < rmse_b).sum(-1) > n_groups / 2


def decide(a: StationScores, b: StationScores, groups: Mapping[str, int]) -> Decision:
    """The decision rule: does product A beat product B?

    Both sides must cover the same stations in the same order, and `groups` must name the group of each.
    Only stations with a finite RMSE in both are compared.
    """
    g, ok = _finite_by_group([a, b], groups)
    labels = np.unique(g[ok])
    masks = [ok & (g == j) for j in labels]
    comparisons = tuple(GroupComparison(int(j), int(m.sum()), float(a.rmse[m].mean()), float(b.rmse[m].mean()),
                                        _wilcoxon_p(a.rmse[m], b.rmse[m]), _nanmean(a.corr[m]), _nanmean(b.corr[m]))
                        for j, m in zip(labels, masks))
    ra = np.array([c.rmse_a for c in comparisons])
    rb = np.array([c.rmse_b for c in comparisons])
    n_groups, wins, diff = len(comparisons), int((ra < rb).sum()), float(np.mean(ra - rb))

    mean_ok, majority_ok = _conditions(ra, rb)
    breakers: tuple[str, ...] = ()
    if mean_ok and majority_ok:
        idx = np.flatnonzero(ok)
        col = np.searchsorted(labels, g[idx])
        n = np.array([c.n_stations for c in comparisons])[col]
        rows = np.arange(len(idx))
        loo_a, loo_b = np.tile(ra, (len(idx), 1)), np.tile(rb, (len(idx), 1))
        for loo, x in ((loo_a, a.rmse), (loo_b, b.rmse)):
            sums = np.array([x[m].sum() for m in masks])[col]
            loo[rows, col] = np.divide(sums - x[idx], n - 1, out=np.full(len(idx), np.nan), where=n > 1)
        breakers = tuple(str(s) for s in a.station_ids[idx[~np.logical_and(*_conditions(loo_a, loo_b))]])

    reasons = []
    if not majority_ok:
        lost = ", ".join(str(c.group) for c in comparisons if not c.rmse_a < c.rmse_b)
        reasons.append(f"lower in {wins} of {n_groups} groups, needs {n_groups // 2 + 1} (not lower in groups {lost})")
    if not mean_ok:
        reasons.append(f"mean RMSE difference {diff:+.4f} is not below 0")
    if len(breakers) == 1:
        reasons.append(f"removing station {breakers[0]} breaks the rule")
    elif breakers:
        reasons.append(f"removing any one of {len(breakers)} stations breaks the rule: {', '.join(breakers)}")
    verdict = "not better" if reasons else "better"
    if not reasons:
        reasons.append(f"lower in {wins} of {n_groups} groups, mean RMSE difference {diff:+.4f}, "
                       "and no single station's removal changes this")

    return Decision(verdict, tuple(reasons), comparisons, int(ok.sum()), diff,
                    _nanmean(np.array([c.corr_a - c.corr_b for c in comparisons])), wins,
                    float(np.var(ra, ddof=1)), float(np.var(rb, ddof=1)), breakers)


def record(d: Decision) -> dict:
    """The decision as written to test_decision.json."""
    return {"verdict": d.verdict, "reasons": list(d.reasons), "groups_won": d.groups_won,
            "n_groups": len(d.groups), "rmse_diff": d.rmse_diff, "var_a": d.var_a, "var_b": d.var_b,
            "loo_breakers": list(d.loo_breakers), "n_stations": d.n_stations, "corr_diff": d.corr_diff,
            "groups": [vars(c) | {"rmse_diff": c.rmse_diff} for c in d.groups]}


def friedman(scores: Mapping[str, StationScores], groups: Mapping[str, int]) -> tuple[float, float]:
    """Friedman test of group-mean raw RMSE, with the station groups as blocks and the products as treatments.

    Uses the stations with a finite RMSE in every product. Reported, not decided on. Returns (statistic, p).
    """
    if len(scores) < 3:
        raise ValueError(f"the Friedman test needs at least 3 products, got {len(scores)}")
    inputs = list(scores.values())
    g, ok = _finite_by_group(inputs, groups)
    labels = np.unique(g[ok])
    result = friedmanchisquare(*[[s.rmse[ok & (g == j)].mean() for j in labels] for s in inputs])
    return float(result.statistic), float(result.pvalue)
