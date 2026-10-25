import numpy as np
import pytest

from windsr.evaluate.decision import StationScores, decide, friedman, record

N = 60
rng = np.random.default_rng(42)
IDS = np.array([f"S{i:03d}" for i in range(N)])
GROUPS = {s: i % 6 for i, s in enumerate(IDS)}
GROUP = np.arange(N) % 6
B_RMSE = rng.uniform(1.5, 2.5, N)
B_CORR = rng.uniform(0.3, 0.8, N)
BASE = StationScores(IDS, B_RMSE, B_CORR)


def shifted(d_rmse, d_corr=0.0):
    d = np.broadcast_to(d_rmse, N).astype(float)
    return StationScores(IDS, B_RMSE + d, B_CORR + d_corr)


def by_group(*deltas):
    return np.array(deltas)[GROUP]


def test_lower_rmse_in_every_group_is_better():
    d = decide(shifted(-0.05), BASE, GROUPS)
    assert d.verdict == "better" and d.groups_won == 6 and d.loo_breakers == ()
    assert len(d.groups) == 6 and all(c.rmse_diff < 0 for c in d.groups)
    assert d.rmse_diff == pytest.approx(-0.05)
    assert d.var_a == pytest.approx(np.var([c.rmse_a for c in d.groups], ddof=1))


def test_lower_in_four_of_six_groups_with_negative_mean_is_better():
    d = decide(shifted(by_group(-0.1, -0.1, -0.1, -0.1, 0.05, 0.05)), BASE, GROUPS)
    assert d.verdict == "better" and d.groups_won == 4 and d.rmse_diff < 0
    assert len(d.reasons) == 1 and d.reasons[0].startswith("lower in 4 of 6 groups")


def test_lower_in_three_of_six_groups_is_not_better():
    d = decide(shifted(by_group(-0.1, -0.1, -0.1, 0.01, 0.01, 0.01)), BASE, GROUPS)
    assert d.verdict == "not better" and d.groups_won == 3 and d.rmse_diff < 0
    assert d.reasons == ("lower in 3 of 6 groups, needs 4 (not lower in groups 3, 4, 5)",)


def test_majority_with_positive_mean_is_not_better():
    d = decide(shifted(by_group(-0.01, -0.01, -0.01, -0.01, -0.01, 0.2)), BASE, GROUPS)
    assert d.verdict == "not better" and d.groups_won == 5
    assert d.reasons == ("mean RMSE difference +0.0250 is not below 0",)


def test_station_whose_removal_breaks_the_majority_is_named():
    delta = by_group(-0.05, -0.05, -0.05, 0.01, 0.05, 0.05)
    delta[3] = -0.5
    d = decide(shifted(delta), BASE, GROUPS)
    assert d.groups_won == 4 and d.rmse_diff < 0
    assert d.verdict == "not better" and d.loo_breakers == ("S003",)
    assert d.reasons == ("removing station S003 breaks the rule",)


def test_significance_is_reported_but_not_required():
    delta = np.where(np.arange(N) // 6 % 2 == 0, -0.3, 0.2)
    d = decide(shifted(delta), BASE, GROUPS)
    assert d.verdict == "better"
    assert all(c.wilcoxon_p > 0.05 for c in d.groups)


def test_correlation_does_not_decide():
    assert decide(shifted(-0.05, d_corr=-0.2), BASE, GROUPS).verdict == "better"
    assert decide(shifted(0.05, d_corr=0.2), BASE, GROUPS).verdict == "not better"


def test_equal_rmse_is_not_better_and_names_both_failures():
    d = decide(BASE, BASE, GROUPS)
    assert d.verdict == "not better" and d.groups_won == 0 and all(c.wilcoxon_p == 1.0 for c in d.groups)
    assert len(d.reasons) == 2 and d.reasons[0].startswith("lower in 0 of 6 groups, needs 4")
    assert d.reasons[1] == "mean RMSE difference +0.0000 is not below 0"


def test_swapping_sides_negates_every_group():
    a = shifted(-0.05)
    ab, ba = decide(a, BASE, GROUPS), decide(BASE, a, GROUPS)
    assert [c.rmse_diff for c in ab.groups] == pytest.approx([-c.rmse_diff for c in ba.groups])
    assert ba.verdict == "not better"


def test_shuffled_order_different_stations_and_missing_groups_rejected():
    perm = np.random.default_rng(3).permutation(N)
    with pytest.raises(ValueError, match="station order"):
        decide(StationScores(IDS[perm], B_RMSE[perm], B_CORR[perm]), BASE, GROUPS)
    with pytest.raises(ValueError, match="different stations"):
        decide(StationScores(IDS[:-6], B_RMSE[:-6], B_CORR[:-6]), BASE, GROUPS)
    with pytest.raises(ValueError, match="no group"):
        decide(BASE, BASE, {s: 0 for s in IDS[:-1]})


def test_nan_rmse_stations_dropped_from_both_sides():
    a = shifted(-0.05)
    a.rmse[:6] = np.nan
    d = decide(a, BASE, GROUPS)
    assert d.n_stations == N - 6 and sum(c.n_stations for c in d.groups) == N - 6
    with pytest.raises(ValueError, match="finite"):
        decide(StationScores(IDS, np.full(N, np.nan), B_CORR), BASE, GROUPS)


def test_station_with_undefined_correlation_is_still_compared_on_rmse():
    a = shifted(-0.05)
    a.corr[GROUP == 0] = np.nan
    d = decide(a, BASE, GROUPS)
    assert d.n_stations == N and d.groups[0].n_stations == 10
    assert np.isnan(d.groups[0].corr_a) and np.isfinite(d.groups[1].corr_a)
    assert d.corr_diff == pytest.approx(0.0, abs=1e-12)


def test_friedman_needs_three_products_and_reports_a_p_value():
    with pytest.raises(ValueError, match="at least 3"):
        friedman({"a": shifted(-0.05), "b": BASE}, GROUPS)
    stat, p = friedman({"a": shifted(-0.1), "b": BASE, "c": shifted(0.1)}, GROUPS)
    assert stat > 0 and 0 <= p <= 1
    with pytest.raises(ValueError, match="station order"):
        friedman({"a": shifted(-0.1), "b": BASE, "c": StationScores(IDS[::-1], B_RMSE, B_CORR)}, GROUPS)


def test_record_holds_the_verdict_and_every_group():
    r = record(decide(shifted(-0.05), BASE, GROUPS))
    assert "label" not in r and r["verdict"] == "better" and r["n_groups"] == len(r["groups"]) == 6
    assert set(r["groups"][0]) == {"group", "n_stations", "rmse_a", "rmse_b", "wilcoxon_p", "corr_a", "corr_b",
                                   "rmse_diff"}
    assert r["groups"][0]["rmse_diff"] == pytest.approx(-0.05)
