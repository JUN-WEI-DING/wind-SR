import numpy as np
import pytest

from windsr.grid import GRID_LAT, GRID_LON
from windsr.evaluate.montecarlo import KM_PER_DEG, crossovers, curves, fit, sampling_weights


def _layout(seed=0, n=30):
    r = np.random.default_rng(seed)
    lat = GRID_LAT[r.integers(5, 35, n)] + r.uniform(-0.05, 0.05, n)
    lon = GRID_LON[r.integers(5, 35, n)] + r.uniform(-0.05, 0.05, n)
    lat[1], lon[1] = lat[0] + 0.01, lon[0] + 0.01
    lat[2], lon[2] = lat[0] - 0.01, lon[0] + 0.01
    return lat, lon


def _series(n, seed=1, hours=200):
    r = np.random.default_rng(seed)
    y = r.gamma(2, 1.5, (hours, n))
    y[r.random(y.shape) < 0.1] = np.nan
    return y, {"a": y + r.normal(0, 1, y.shape), "b": 0.8 * y + r.normal(0.3, 1.2, y.shape)}


def test_weights_sum_to_one_and_favour_sparse_stations():
    lat, lon = _layout()
    w = sampling_weights(lat, lon)
    assert w.p.sum() == pytest.approx(1.0)
    assert (w.p > 0).all()
    grid = np.arange(5) * 0.5
    lat = np.r_[np.repeat(grid, 5), 1.0 + 0.01, 1.0 - 0.01, 1.0 + 0.01]
    lon = np.r_[np.tile(grid, 5), 1.0 + 0.01, 1.0 + 0.01, 1.0 - 0.01]
    p = sampling_weights(lat, lon).p
    isolated = 6
    cluster = [12, 25, 26, 27]
    assert lat[12] == lon[12] == 1.0
    assert p[isolated] > p[cluster].max()


def test_eq_10_to_12_on_a_square_with_its_centre():
    lat = np.array([-0.1, -0.1, 0.1, 0.1, 0.0])
    lon = np.array([-0.1, 0.1, -0.1, 0.1, 0.0])
    w = sampling_weights(lat, lon)
    side, half_diagonal = 0.2 * KM_PER_DEG, np.sqrt(2) * 0.1 * KM_PER_DEG
    d_corner, d_centre = (2 * side + half_diagonal) / 3, half_diagonal
    expected_prime = np.r_[[d_corner / (2 * d_corner + d_centre)] * 4, d_centre / (4 * d_corner)]
    np.testing.assert_allclose(w.d, np.r_[[d_corner] * 4, d_centre])
    np.testing.assert_allclose(w.d_prime, expected_prime)
    np.testing.assert_allclose(w.p, expected_prime / expected_prime.sum())


def test_two_stations_in_one_cell_form_one_unit():
    lat = np.array([GRID_LAT[10] + 0.01, GRID_LAT[10] - 0.01, GRID_LAT[30]])
    lon = np.array([GRID_LON[10], GRID_LON[10] + 0.02, GRID_LON[30]])
    y = np.array([[1.0, 3.0, 2.0], [2.0, np.nan, 2.0], [4.0, 0.0, 2.0]])
    x = np.array([[2.0, 2.0, 3.0], [2.0, 2.0, 3.0], [1.0, 1.0, 3.0]])
    out = curves(lat, lon, y, {"x": x}, draws=3, sizes=[3])
    unit = np.sqrt(((2 - 2) ** 2 + (2 - 2) ** 2 + (1 - 2) ** 2) / 3)
    assert out["mean"].item() == pytest.approx((unit + 1.0) / 2), "the RMSE of the averaged series, not mean RMSE"
    assert out.variant.item() == "paper"


def test_all_stations_give_the_same_value_on_every_draw():
    lat, lon = _layout()
    y, products = _series(len(lat))
    out = curves(lat, lon, y, products, draws=20, sizes=[len(lat)])
    assert len(out) == 2
    assert (out.sd == 0).all()
    assert (out.eq14_change == 0).all()


def test_a_fixed_seed_reproduces_and_another_seed_differs():
    lat, lon = _layout()
    y, products = _series(len(lat))
    a = curves(lat, lon, y, products, draws=30, sizes=[1, 5, 12], seed=3)
    b = curves(lat, lon, y, products, draws=30, sizes=[1, 5, 12], seed=3)
    c = curves(lat, lon, y, products, draws=30, sizes=[1, 5, 12], seed=4)
    assert a.equals(b)
    assert not np.allclose(a["mean"], c["mean"])


def test_fit_recovers_a_quadratic_and_crossovers_are_flagged_beyond_the_stations():
    n = np.arange(1, 346, dtype=float)
    fits = {"flat": fit(n, np.full_like(n, 2.0) + 1e-9 * np.sin(n)), "bowl": fit(n, 1e-6 * n ** 2 - 1e-3 * n + 2.2)}
    np.testing.assert_allclose(fits["bowl"].coef, [1e-6, -1e-3, 2.2], atol=1e-9)
    assert fits["bowl"].r2 == pytest.approx(1.0)
    roots = (1000 + np.array([-1, 1]) * np.sqrt(1000 ** 2 - 4 * 200000)) / 2
    out = crossovers(fits, n_max=345)
    np.testing.assert_allclose(out.n, roots, rtol=1e-4)
    assert out.extrapolated.tolist() == [False, True]
    assert out.lower_below.tolist() == ["flat", "bowl"]
    assert out.lower_above.tolist() == ["bowl", "flat"]


def test_a_product_without_values_at_a_station_is_refused():
    lat, lon = _layout()
    y, products = _series(len(lat))
    products["a"][:, 4] = np.nan
    with pytest.raises(ValueError, match="a has no value at 1 stations"):
        curves(lat, lon, y, products, draws=3, sizes=[5])
