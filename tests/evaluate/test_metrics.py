import numpy as np
import pytest

from windsr.evaluate.metrics import fit_calibration, per_station, score_product
from windsr.periods import FIT, TEST, TRAIN, VALID
from windsr.evaluate.truth import Series, truth


def series(times, values):
    return Series(values, times, np.array([f"S{j}" for j in range(values.shape[1])]))


def linear_pair(times, a=0.5, b=0.6, stations=3, seed=0):
    x = np.random.default_rng(seed).uniform(0, 10, (len(times), stations))
    return series(times, x), series(times, a + b * x)


def test_fit_recovers_a_linear_relation():
    x, y = linear_pair(VALID.hours)
    cal = fit_calibration(x, y)
    assert (cal.a, cal.b) == pytest.approx((0.5, 0.6))
    assert np.allclose(cal.apply("V2", x.values), y.values)
    assert np.allclose(cal.apply("V3", x.values), y.values)


@pytest.mark.parametrize("other", [TEST, FIT], ids=["test", "fit"])
@pytest.mark.parametrize("leak", [1, 24, 200])
def test_fit_rejects_hours_outside_the_validation_weeks(other, leak):
    times = np.concatenate([VALID.hours[leak:], other.hours[:leak]])
    x, y = linear_pair(times)
    with pytest.raises(ValueError, match="outside the validation weeks"):
        fit_calibration(x, y)


def test_fit_rejects_test_hours_in_truth_only():
    x, y = linear_pair(VALID.hours)
    y = series(np.concatenate([VALID.hours[5:], TEST.hours[:5]]), y.values)
    with pytest.raises(ValueError, match="truth"):
        fit_calibration(x, y)


def test_split_is_disjoint_buffered_and_covers_every_month():
    fit, valid = FIT.hours, VALID.hours
    assert len(fit) == 9024 and len(valid) == 3192
    assert len(np.intersect1d(fit, valid)) == 0
    i = np.clip(np.searchsorted(fit, valid), 1, len(fit) - 1)
    nearest = np.minimum(np.abs(valid - fit[i - 1]), np.abs(fit[i] - valid))
    assert nearest.min() == np.timedelta64(25, "h"), "one buffer day between fit and validation hours"
    months = lambda t: set(t.astype("datetime64[M]"))
    assert months(valid) == months(TRAIN.hours)
    assert TRAIN.contains(valid).all() and not TEST.contains(valid).any()


def test_per_station_metrics_on_known_errors():
    y = np.tile(np.arange(10.0), (2, 1)).T
    x = y + np.array([1.0, -2.0])
    x[0, 0] = np.nan
    m = per_station(x, y)
    assert m["rmse"] == pytest.approx([1.0, 2.0])
    assert m["bias"] == pytest.approx([1.0, -2.0])
    assert m["crmse"] == pytest.approx([0.0, 0.0], abs=1e-6)
    assert m["corr"] == pytest.approx([1.0, 1.0])
    assert list(m["n"]) == [9, 10]


def test_constant_series_has_no_correlation():
    y = np.random.default_rng(0).uniform(0, 5, (100, 2))
    x = np.column_stack([np.full(100, 3.0), y[:, 1]])
    c = per_station(x, y)["corr"]
    assert np.isnan(c[0]) and c[1] == pytest.approx(1.0)


@pytest.mark.data
def test_scores_the_test_year_with_calibration_fitted_on_the_validation_weeks(data):
    periods = []

    def load(st, period, sampling):
        periods.append(period.name)
        return (truth(data, period).values - 0.3) / 0.7

    s = score_product(data, "m", load)
    assert periods == [TEST.name, VALID.name]
    assert np.nanmax(s.by_variant["V2"]["rmse"]) < 1e-9, "V2 recovers y = 0.3 + 0.7 x"
    assert np.nanmean(s.by_variant["V0"]["rmse"]) > 0.1
    assert len(s.station_ids) == 345
    assert len(score_product(data, "m", load, station_set="sensitivity").station_ids) == 345 - 18
