import hashlib

import numpy as np
import pytest
import xarray as xr

from windsr.data import store
from windsr.data.store import STORES, DataError, open_store, verify
from windsr.grid import GRID_LAT, GRID_LON


def _write_manifest(root):
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}" for p in files]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    store.manifest.cache_clear()
    store.open_store.cache_clear()


@pytest.fixture
def fake(tmp_path):
    for name in STORES:
        xr.Dataset({"x": (("time", "latitude"), np.arange(12.0).reshape(4, 3))},
                   coords={"time": np.arange(4), "latitude": [1.0, 2.0, 3.0]}).to_zarr(tmp_path / f"{name}.zarr",
                                                                                     zarr_format=2)
    (tmp_path / "taipower").mkdir()
    (tmp_path / "taipower" / "release.json").write_text("{}")
    _write_manifest(tmp_path)
    return tmp_path


def test_verify_checks_every_listed_file(fake):
    n = len((fake / "SHA256SUMS").read_text().splitlines())
    assert verify(fake) == n and n > 5 * 4
    assert float(open_store(fake, "era5").x.sum()) == 66.0


def test_a_changed_chunk_fails_verify_and_a_changed_coordinate_fails_open(fake):
    (fake / "era5.zarr/x/0.0").write_bytes(b"changed")
    with pytest.raises(DataError, match="era5.zarr/x/0.0 does not match"):
        verify(fake)
    (fake / "icon.zarr/latitude/0").write_bytes(b"changed")
    with pytest.raises(DataError, match="icon.zarr/latitude/0 does not match"):
        open_store(fake, "icon")


def test_an_unlisted_or_missing_file_fails(fake):
    (fake / "stations.zarr/extra").write_text("x")
    with pytest.raises(DataError, match=r"stations.zarr: files differ.*extra \['stations.zarr/extra'\]"):
        verify(fake)
    (fake / "stations.zarr/extra").unlink()
    (fake / "era5_land.zarr/x/0.0").unlink()
    with pytest.raises(DataError, match="missing"):
        open_store(fake, "era5_land")


def test_a_directory_without_manifest_is_refused(tmp_path):
    with pytest.raises(DataError, match="SHA256SUMS does not exist"):
        verify(tmp_path)


@pytest.mark.data
def test_data_directory_matches_its_manifest(data):
    assert verify(data) == len((data / "SHA256SUMS").read_text().splitlines())


@pytest.mark.data
@pytest.mark.parametrize("name", ["era5", "icon"])
def test_grid_constants_equal_store_coordinates(data, name):
    ds = open_store(data, name)
    assert np.allclose(ds.latitude.values, GRID_LAT, atol=1e-9) and np.allclose(ds.longitude.values, GRID_LON, atol=1e-9)
