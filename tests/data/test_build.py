import shutil
import subprocess

import numpy as np
import pytest
import xarray as xr

from windsr.data.compare import compare
from windsr.data.era5 import bilinear
from windsr.data.era5_land import land_mask, sea_to_nan
from windsr.data.icon import undo_shift
from windsr.data.stations import stuck_mask
from windsr.data.store import STORES, store_path, write_sha256sums

SHA256_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SHA256_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_twelve_hours_at_one_positive_value_are_stuck_and_eleven_are_not():
    x = np.r_[np.full(12, 3.0), 1.0, np.full(11, 2.0), 1.5]
    m = stuck_mask(x)
    assert m[:12].all() and not m[12:].any()


def test_calm_is_stuck_only_after_48_hours():
    x = np.r_[np.zeros(48), 1.0, np.zeros(47), 1.0]
    m = stuck_mask(x)
    assert m[:48].all() and not m[48:].any()


def test_a_missing_hour_ends_a_run():
    x = np.r_[np.full(6, 2.0), np.nan, np.full(6, 2.0), np.full(30, np.nan)]
    assert not stuck_mask(x).any()


def test_undo_shift_moves_each_column_back_west_from_the_switch_hour():
    truth = np.random.default_rng(0).random((5, 3, 6))
    published = truth.copy()
    published[2:, :, 1:] = truth[2:, :, :-1]
    undo_shift(published, 2)
    assert np.array_equal(published[:2], truth[:2])
    assert np.array_equal(published[2:, :, :-1], truth[2:, :, :-1])
    assert np.isnan(published[2:, :, -1]).all()


def bm_like(values):
    t, ny, nx = values.shape
    return xr.DataArray(values.astype(np.float32), dims=("time", "latitude", "longitude"),
                        coords={"latitude": np.arange(ny) * 0.1, "longitude": np.arange(nx) * 0.1})


def test_sea_cells_become_nan_and_land_values_stay():
    u10 = np.ones((4, 2, 3))
    u10[:, 0, 0] = np.nan
    ws = np.random.default_rng(1).uniform(1, 5, (4, 2, 3))
    ws[:, 0, 0] = 0
    out = sea_to_nan(bm_like(ws), land_mask(u10))
    assert out.dtype == np.float32
    assert np.isnan(out.values[:, 0, 0]).all()
    land = np.ones((2, 3), bool)
    land[0, 0] = False
    assert np.array_equal(out.values[:, land], ws.astype(np.float32)[:, land])


def test_a_cell_missing_in_some_hours_only_is_refused():
    u10 = np.ones((4, 2, 3))
    u10[1, 0, 0] = np.nan
    with pytest.raises(AssertionError):
        land_mask(u10)


def test_a_sea_cell_with_wind_is_refused():
    u10 = np.ones((4, 2, 3))
    u10[:, 0, 0] = np.nan
    ws = np.ones((4, 2, 3))
    with pytest.raises(AssertionError):
        sea_to_nan(bm_like(ws), land_mask(u10))


def test_bilinear_keeps_grid_points_and_reproduces_a_plane():
    lat, lon = 20.5 + 0.25 * np.arange(5), 117.5 + 0.25 * np.arange(6)
    plane = lambda y, x: 1 + 2 * y[:, None] - 0.5 * x[None]
    a = np.stack([plane(lat, lon), -plane(lat, lon)]).astype(np.float32)
    lat_new = np.array([20.5, 20.625, 21.375, 21.5], np.float32)
    lon_new = np.array([117.625, 118.0, 118.75], np.float32)
    out = bilinear(a, lat, lon, lat_new, lon_new)
    assert out.dtype == np.float32 and out.shape == (2, 4, 3)
    np.testing.assert_allclose(out[0], plane(lat_new.astype(float), lon_new.astype(float)), rtol=1e-6)
    assert out[1, 0, 1] == a[1, 0, 2]


def test_bilinear_refuses_a_target_outside_the_source_grid():
    lat, lon = np.arange(3.0), np.arange(3.0)
    with pytest.raises(AssertionError):
        bilinear(np.zeros((1, 3, 3), np.float32), lat, lon, np.array([2.5]), np.array([1.0]))


def test_sha256sums_lists_every_store_file_relative_and_sorted(tmp_path):
    (tmp_path / "b.zarr" / "u10").mkdir(parents=True)
    (tmp_path / "a.zarr").mkdir()
    (tmp_path / "b.zarr" / "u10" / "0.0.0").write_bytes(b"abc")
    (tmp_path / "a.zarr" / ".zattrs").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("not a store")
    write_sha256sums(tmp_path, ("b.zarr", "a.zarr"))
    assert (tmp_path / "SHA256SUMS").read_text() == f"{SHA256_EMPTY}  a.zarr/.zattrs\n{SHA256_ABC}  b.zarr/u10/0.0.0\n"
    if shutil.which("sha256sum"):
        subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=tmp_path, check=True)


def write_stores(root, change=None):
    for name in STORES:
        v = np.array([[1.0, np.nan], [3.0, 4.0]], np.float32)
        if name == change:
            v[1, 1] = 5.0
        ds = xr.Dataset({"x": (("time", "c"), v)}, coords={"time": np.arange(2), "c": [0.5, 1.5]})
        ds.to_zarr(store_path(root, name), mode="w", consolidated=True, zarr_format=2)


def test_compare_passes_equal_stores_with_nan(tmp_path):
    write_stores(tmp_path / "built")
    write_stores(tmp_path / "ref")
    assert compare(tmp_path / "built", tmp_path / "ref") == []


def test_compare_names_the_store_and_variable_that_differ(tmp_path):
    write_stores(tmp_path / "built", change="icon")
    write_stores(tmp_path / "ref")
    bad = compare(tmp_path / "built", tmp_path / "ref")
    assert len(bad) == 1 and bad[0].startswith("icon.zarr:x values differ")
