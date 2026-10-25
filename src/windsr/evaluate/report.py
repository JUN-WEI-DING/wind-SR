"""The evaluation of a model against ERA5, ICON and ERA5-Land on the test year, every facet written to OUT.

- main/: nearest-cell sampling at all 345 stations. The decision rule (`decision.py`) against each rival decides;
  the Friedman test and the calibrated variants V1 to V3 are reported.
- sensitivity/: the same without the 18 level-shift stations, reported only.
- bilinear/: the same under bilinear sampling (ERA5-Land stays cell-based), reported only.
  Each of these three holds test_summary.csv, test_per_station.csv and test_decision.json.
- montecarlo/: the Monte Carlo cross-scale evaluation (`montecarlo.py`) of every product at all stations, raw
  hourly values: curves.csv, weights.csv, crossovers.csv, fits.json.
- windfarm/: the wind-farm evaluation (`windfarm.py`) of ERA5, ICON and the model on DATA/taipower:
  farm_coverage.csv, farm_metrics.csv, farm_pairs.csv, farm_bins.csv, farm_aggregation.csv, farm_lagscan.csv,
  provenance.json.
"""
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import pandas as pd

from ..data.stations import stations
from ..data.store import store_path, taipower
from ..data.taipower import FARMS, RUNS
from ..grid import nearest_cell
from ..periods import TEST
from . import windfarm
from .decision import decide, friedman, record, station_scores
from .groups import station_groups
from .metrics import per_station_table, score_product, summary
from .montecarlo import CONVERGENCE, VARIANT, crossovers, curves, fit, sampling_weights
from .products import RIVALS, Loader, model_product, rivals
from .truth import truth

SCORED = {"main": {}, "sensitivity": {"station_set": "sensitivity"}, "bilinear": {"sampling": "bilinear"}}
DRAWS, SEED = 5000, 0
SHOWN = ["product", "variant", "rmse", "bias", "corr", "crmse", "n_stations"]


def score_facet(data: Path, out: Path, loaders: Mapping[str, Loader], model: str, **config) -> None:
    """Scores every product under one configuration of SCORED and writes its three files to OUT: the summary, the
    per-station table, and the decisions of `model` against each rival with the Friedman test over all products."""
    out.mkdir(parents=True, exist_ok=True)
    scores = [score_product(data, n, loader, **config) for n, loader in loaders.items()]
    table = summary(scores)
    table.to_csv(out / "test_summary.csv", index=False)
    per_station_table(scores).to_csv(out / "test_per_station.csv", index=False)
    groups = station_groups(data)
    inputs = {s.name: station_scores(s) for s in scores}
    stat, p = friedman(inputs, groups)
    result = {"decisions": {f"vs_{b}": record(decide(inputs[model], inputs[b], groups)) for b in RIVALS},
              "friedman": {"products": list(inputs), "statistic": stat, "p": p}}
    json.dump(result, open(out / "test_decision.json", "w"), indent=1)
    print(f"== {out.name}: {'decides' if not config else 'reported only'}")
    print(table[SHOWN].round(4).to_string(index=False))
    print({k: (v["verdict"], v["groups_won"], round(v["rmse_diff"], 4)) for k, v in result["decisions"].items()},
          f"Friedman p {p:.3g}")


def monte_carlo(data: Path, out: Path, loaders: Mapping[str, Loader]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    st = stations(data)
    products = {n: loader(st, TEST, "nearest") for n, loader in loaders.items()}
    table = curves(st.lat, st.lon, truth(data, TEST).values, products, draws=DRAWS, seed=SEED)
    w = sampling_weights(st.lat, st.lon)
    weights = pd.DataFrame({"variant": VARIANT, "station": st.ids, "lat": st.lat, "lon": st.lon,
                            "cell": nearest_cell(st.lat, st.lon), "d": w.d, "d_prime": w.d_prime, "p": w.p})
    fits = {n: fit(g.n.to_numpy(float), g["mean"].to_numpy()) for n, g in table.groupby("product", sort=False)}
    cross = crossovers(fits, len(st.ids))
    table.to_csv(out / "curves.csv", index=False)
    weights.to_csv(out / "weights.csv", index=False)
    cross.to_csv(out / "crossovers.csv", index=False)
    json.dump({n: asdict(f) for n, f in fits.items()}, open(out / "fits.json", "w"), indent=2)
    print(f"== {out.name}: {len(st.ids)} stations, {DRAWS} draws per sample size, reported only")
    picks = [1, 50, 100, 200, len(st.ids)]
    print(table[table.n.isin(picks)].pivot_table(index="n", columns="product", values="mean", sort=False).round(4))
    change = table.eq14_change.max()
    print(f"largest Eq. 14 change at the last draw {change:.2g} (limit {CONVERGENCE})")
    print(cross.round(1).to_string(index=False))


def wind_farms(data: Path, out: Path, model: str, store: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    obs = windfarm.observations(taipower(data), TEST)
    sources = {n: str(store_path(data, n)) for n in ("era5", "icon")}
    winds = {n: windfarm.rival_wind(data, n, TEST) for n in sources}
    not_scored = {"era5_land": "ERA5-Land has no 100 m wind, so hub-height wind (Eq. 1) cannot be formed"}
    try:
        winds[model] = windfarm.store_wind(store, TEST)
        sources[model] = store
    except windfarm.NoHubWind as e:
        not_scored[model] = f"cannot be scored on farms: {e}"
    preds = {n: windfarm.product_cf(w10, w100, obs.hours) for n, (w10, w100) in winds.items()}
    res = windfarm.evaluate(obs, preds)
    obs.coverage.to_csv(out / "farm_coverage.csv", index=False)
    res.metrics.to_csv(out / "farm_metrics.csv", index=False)
    res.pairs.to_csv(out / "farm_pairs.csv", index=False)
    res.bins.to_csv(out / "farm_bins.csv", index=False)
    res.aggregation.to_csv(out / "farm_aggregation.csv", index=False)
    res.lags.to_csv(out / "farm_lagscan.csv", index=False)
    best = res.lags.loc[res.lags.groupby("farm", sort=False)["corr"].idxmax()].set_index("farm")
    json.dump({"observations": {"release_id": obs.release_id, "release_path": obs.release_path,
                                "files_sha256": obs.files, "excluded_runs": list(RUNS),
                                "min_hours_fraction": windfarm.MIN_HOURS_FRACTION},
               "period": [str(obs.hours[0]), str(obs.hours[-1])], "windows_h": list(windfarm.WINDOWS),
               "farms": [asdict(f) | {"rated_speed": windfarm.TURBINES[f.turbine].rated_speed,
                                      "nameplate_kw": windfarm.TURBINES[f.turbine].nameplate_kw} for f in FARMS],
               "products": sources, "not_scored": not_scored,
               "lag_scan_best": {f: {"lag_h": int(r.lag_h), "corr": r["corr"]} for f, r in best.iterrows()}},
              open(out / "provenance.json", "w"), indent=2, ensure_ascii=False)
    print(f"== {out.name}: release {obs.release_id}, reported only")
    print(obs.coverage[["farm", "name", "hours_observed", "hours_fraction", "scored"]].round(3).to_string(index=False))
    print("best lag of obs(t) vs era5(t + lag):", {f: int(r.lag_h) for f, r in best.iterrows()})
    for n, why in not_scored.items():
        print(f"not scored: {n}: {why}")
    if len(res.metrics):
        print(res.metrics.pivot(index="farm", columns="product", values="rmse").round(4))


def evaluate(data: Path, out: Path, model: str, test_store: str, train_store: str) -> None:
    """Every facet into OUT."""
    if model in RIVALS:
        raise ValueError(f"the model cannot be named after a rival ({', '.join(RIVALS)})")
    pd.set_option("display.width", 200)
    loaders = rivals(data) | {model: model_product(test_store, train_store)}
    for label, config in SCORED.items():
        score_facet(data, out / label, loaders, model, **config)
    monte_carlo(data, out / "montecarlo", loaders)
    wind_farms(data, out / "windfarm", model, test_store)
