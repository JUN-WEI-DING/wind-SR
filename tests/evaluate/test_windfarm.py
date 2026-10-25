import json
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xarray as xr

from windsr.data.store import taipower
from windsr.data.taipower import FARMS
from windsr.evaluate.windfarm import (TURBINES, NoHubWind, Observations, block_means, cf_bin, evaluate, hourly_cf,
                                      hub_wind, lag_scan, log_fit, loss_corrected, observations, predicted_cf,
                                      product_cf, rival_wind, rmse, store_wind)
from windsr.grid import GRID_LAT, GRID_LON
from windsr.periods import TEST, Period

H = np.timedelta64(1, "h")
M10 = np.timedelta64(10, "m")
T0 = np.datetime64("2024-07-10T00", "ns")


def _ten_minute(hours=6):
    t = T0 + np.arange(hours * 6 + 7) * M10 - 3 * M10
    return t, T0 + np.arange(hours) * H


def test_hourly_value_is_the_symmetric_mean_from_minus_to_plus_30_minutes():
    t, hours = _ten_minute()
    cf = np.zeros(len(t))
    cf[t == T0 + 2 * H + 30 * np.timedelta64(1, "m")] = 0.7
    out = hourly_cf(t, cf, hours)
    np.testing.assert_allclose(out, [0, 0, 0.1, 0.1, 0, 0], err_msg="HH+0:30 belongs to hours HH and HH+1")
    ramp = (t - T0) / M10 * 0.01
    np.testing.assert_allclose(hourly_cf(t, ramp, hours), np.arange(6) * 0.06, err_msg="symmetric about HH:00")


def test_hour_needs_all_seven_values_and_is_clipped_at_zero():
    t, hours = _ten_minute()
    cf = np.full(len(t), 0.5)
    cf[t == T0 + H + 30 * np.timedelta64(1, "m")] = np.nan
    cf[(t >= T0 + 4 * H - 30 * np.timedelta64(1, "m")) & (t <= T0 + 4 * H + 20 * np.timedelta64(1, "m"))] = -0.1
    out = hourly_cf(t, cf, hours)
    assert np.isnan(out[1]) and np.isnan(out[2])
    assert out[4] == 0.0
    assert out[0] == pytest.approx(0.5)


def test_eq_1_log_interpolation_hits_both_levels():
    ws10, ws100 = np.array([3.0, 5.0]), np.array([6.0, 9.0])
    np.testing.assert_allclose(hub_wind(ws10, ws100, 10), ws10)
    np.testing.assert_allclose(hub_wind(ws10, ws100, 100), ws100)
    np.testing.assert_allclose(hub_wind(ws10, ws100, np.sqrt(1000)), (ws10 + ws100) / 2)


def test_eq_15_regimes_and_continuity():
    rs = 14.0
    assert loss_corrected(np.array([8.0]), rs)[0] == pytest.approx(0.93 * 8.0)
    ws = rs - 0.5 + 1.0
    assert loss_corrected(np.array([ws]), rs)[0] == pytest.approx((1 - 0.07 * (1 - 1.0 / 2.5)) * ws)
    assert loss_corrected(np.array([rs + 3]), rs)[0] == rs + 3
    eps = 1e-9
    for edge in (rs - 0.5, rs + 2.0):
        below, above = loss_corrected(np.array([edge - eps, edge + eps]), rs)
        assert above - below == pytest.approx(0, abs=1e-7)


def test_rated_speed_is_where_the_curve_reaches_nameplate_power():
    assert {n: t.rated_speed for n, t in TURBINES.items()} == {
        "Vestas V80": 14.5, "Enercon E70": 15.0, "HTW5.2-127": 14.0, "SWT-6.0-154": 13.0}
    assert {f.turbine for f in FARMS} <= set(TURBINES)


def test_eq_16_degrades_by_1_6_percent_per_operational_year():
    farm = replace(FARMS[3], commissioned=2020)
    times = np.array(["2020-07-01T00", "2025-07-01T00"], "datetime64[ns]")
    ws10, ws100 = np.full(2, 7.0), np.full(2, 7.0)
    new, old = predicted_cf(ws10, ws100, times, farm)
    assert old / new == pytest.approx(0.984 ** 5)
    turbine = TURBINES[farm.turbine]
    assert new == pytest.approx(np.interp(0.93 * 7.0, turbine.speed, turbine.power_kw) / 2000)


def test_rmse_is_a_root_mean_square_not_a_squared_error():
    assert rmse(np.array([0.0, 0.0]), np.array([0.3, 0.4])) == pytest.approx(np.sqrt((0.09 + 0.16) / 2))


def test_aggregation_windows_are_block_means():
    x = np.arange(10.0)
    np.testing.assert_allclose(block_means(x, 3), [1, 4, 7])
    x[4] = np.nan
    out = block_means(x, 3)
    assert np.isnan(out[1]) and out[0] == 1 and out[2] == 7


def test_cf_bins_put_values_above_one_in_the_last_bin():
    np.testing.assert_array_equal(cf_bin(np.array([0.0, 0.05, 0.1, 0.55, 0.95, 1.0, 1.2])), [0, 0, 1, 5, 9, 9, 9])


def test_log_fit_recovers_a_logarithmic_decay():
    w = np.array([1, 3, 6, 12, 24, 72, 168])
    f = log_fit(w, 0.2 - 0.015 * np.log(w))
    assert (f.a, f.b, f.r2) == pytest.approx((0.2, -0.015, 1.0))


def test_lag_scan_finds_a_known_shift():
    r = np.random.default_rng(0)
    x = r.random(500)
    obs, pred = x[:-3], x[3:]
    corr = lag_scan(obs, pred, range(-5, 6))
    assert corr.idxmax() == -3
    assert lag_scan(x, x, range(-2, 3)).idxmax() == 0


def _obs(cf, scored=True):
    hours = T0 + np.arange(len(cf)) * H
    cov = pd.DataFrame({"farm": [FARMS[0].unit_name], "scored": [scored]})
    return Observations(hours, cf[:, None], cov, "r", "/", {})


def test_evaluate_reports_rmse_pairs_bins_and_aggregation():
    r = np.random.default_rng(1)
    obs = r.random(48)
    good, bad = obs + r.normal(0, 0.05, 48), obs + r.normal(0, 0.3, 48)
    res = evaluate(_obs(obs), {"era5": bad[:, None], "x": good[:, None]}, FARMS[:1], windows=(1, 24))
    m = res.metrics.set_index("product").rmse
    assert m["era5"] == pytest.approx(rmse(bad, obs)) and m["x"] == pytest.approx(rmse(good, obs))
    pair = res.pairs[res.pairs.bin == "all"].iloc[0]
    assert (pair.a, pair.b) == ("era5", "x") and pair.wilcoxon_p < 0.05
    assert res.bins.groupby("product").n.sum().tolist() == [48, 48]
    day = res.aggregation[(res.aggregation["product"] == "x") & (res.aggregation.window_h == 24)].iloc[0]
    assert day.blocks == 2
    assert day.rmse == pytest.approx(rmse(good.reshape(2, 24).mean(1), obs.reshape(2, 24).mean(1)))


def test_evaluate_skips_metrics_of_an_unscored_farm_but_scans_its_lag():
    obs = np.random.default_rng(2).random(48)
    res = evaluate(_obs(obs, scored=False), {"era5": obs[:, None]}, FARMS[:1])
    assert res.metrics.empty and len(res.lags) == 25


def _release(tmp_path, rows, runs):
    part = tmp_path / "partitions" / "2024-07"
    part.mkdir(parents=True)
    (tmp_path / "metadata").mkdir()
    (tmp_path / "release.json").write_text(json.dumps({"release_id": "release-test"}))
    table = pa.table({
        "timestamp": pa.array(rows.timestamp, pa.timestamp("ms", tz="Asia/Taipei")),
        "unit_name": rows.unit_name, "net_generation_mw": pa.array([Decimal(str(v)) for v in rows.mw], pa.decimal128(14, 3)),
        "capacity_mw": pa.array([None if np.isnan(c) else Decimal(str(c)) for c in rows.capacity], pa.decimal128(14, 3)),
        "qa_exclude": rows.qa_exclude, "eligible_reported_item_analysis": rows.eligible})
    pq.write_table(table, part / "data.parquet")
    runs.to_csv(tmp_path / "metadata" / "observation_quality_flags.csv", index=False)


def test_observations_read_utc_cf_with_capacity_fallback_and_dataset_flags(tmp_path):
    farm = replace(FARMS[3], capacity_mw=40.0)
    local = pd.date_range("2024-07-10 07:00", "2024-07-10 12:00", freq="10min", tz="Asia/Taipei")
    n = len(local)
    rows = pd.DataFrame({"timestamp": local, "unit_name": farm.unit_name, "mw": 20.0, "capacity": np.nan,
                         "qa_exclude": False, "eligible": True})
    rows.loc[local >= pd.Timestamp("2024-07-10 09:50", tz="Asia/Taipei"), "capacity"] = 80.0
    rows.loc[local == pd.Timestamp("2024-07-10 10:50", tz="Asia/Taipei"), "qa_exclude"] = True
    other = rows.iloc[:n].assign(unit_name="其他", mw=99.0)
    runs = pd.DataFrame({"unit_name": [farm.unit_name, farm.unit_name],
                         "rule_id": ["intrinsic-zero-run-7d-v1", "intrinsic-negative-generation-v1"],
                         "run_first_timestamp": ["2024-07-10T07:50:00+08:00", "2024-07-10T09:50:00+08:00"],
                         "run_last_timestamp": ["2024-07-10T07:50:00+08:00", "2024-07-10T10:00:00+08:00"]})
    _release(tmp_path, pd.concat([rows, other]), runs)
    period = Period("p", np.datetime64("2024-07-09T23", "ns"), np.datetime64("2024-07-10T03", "ns"))
    obs = observations(str(tmp_path), period, [farm])
    assert obs.release_id == "release-test"
    assert set(obs.files) == {"partitions/2024-07/data.parquet", "metadata/observation_quality_flags.csv"}
    # UTC hours 23..03 are local 07..11; 07:50 local is flagged, so UTC 23 and 00 lack a value.
    np.testing.assert_allclose(obs.cf[:, 0], [np.nan, np.nan, 0.5, (2 * 0.5 + 5 * 0.25) / 7, np.nan], equal_nan=True)
    cov = obs.coverage.iloc[0]
    assert cov.records == n and cov.records_valid == n - 2 and cov.hours_observed == 2
    assert cov["records_in_intrinsic-zero-run-7d-v1"] == 1 and cov.records_registry_capacity == 17


def test_store_without_100m_wind_cannot_be_scored(tmp_path):
    hours = T0 + np.arange(3) * H
    da = xr.DataArray(np.zeros((3, 41, 41), np.float32), dims=("time", "latitude", "longitude"),
                      coords={"time": hours, "latitude": GRID_LAT, "longitude": GRID_LON})
    path = str(tmp_path / "pred.zarr")
    da.to_dataset(name="wind_speed_10m").to_zarr(path)
    period = Period("p", hours[0], hours[-1])
    with pytest.raises(NoHubWind):
        store_wind(path, period)
    xr.Dataset({"wind_speed_10m": da, "wind_speed_100m": da + 2}).to_zarr(path, mode="w")
    ws10, ws100 = store_wind(path, period)
    assert ws10.shape == (3, len(FARMS)) and (ws100 == 2).all()


def test_blocks_and_lags_never_join_two_separate_weeks():
    r = np.random.default_rng(3)
    week = T0 + np.arange(48) * H
    hours = np.r_[week, week + 20 * 24 * H]
    obs = r.random(96)
    cov = pd.DataFrame({"farm": [FARMS[0].unit_name], "scored": [True]})
    res = evaluate(Observations(hours, obs[:, None], cov, "r", "/", {}), {"era5": obs[:, None]}, FARMS[:1],
                   windows=(24, 72))
    assert res.aggregation.set_index("window_h").blocks.to_dict() == {24: 4, 72: 0}
    assert res.metrics.n.item() == 96
    half_day = r.random(12)
    twice = np.r_[half_day, half_day]
    hours = np.r_[week[:12], week[:12] + 20 * 24 * H]
    lags = evaluate(Observations(hours, twice[:, None], cov, "r", "/", {}), {"era5": twice[:, None]}, FARMS[:1]).lags
    corr = lags.set_index("lag_h")["corr"]
    assert corr[0] == pytest.approx(1.0) and np.isnan(corr[12]), "lag 12 would pair the two weeks' hours"


@pytest.mark.data
def test_test_year_scores_four_farms_aligned_in_utc(data):
    obs = observations(taipower(data), TEST)
    assert obs.release_id.startswith("release-taipower-unit-generation")
    assert obs.coverage.set_index("farm").scored.to_dict() == {
        "海洋竹南": False, "離岸一期": True, "鹿威彰濱": False, "彰工": True, "王功": True, "雲麥": True}
    res = evaluate(obs, {"era5": product_cf(*rival_wind(data, "era5", TEST), obs.hours)})
    best = res.lags.loc[res.lags.groupby("farm")["corr"].idxmax()]
    assert (best.lag_h == 0).all(), "observed CF peaks with ERA5 power at lag 0"
