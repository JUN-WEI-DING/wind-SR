"""The data directory DATA: five Zarr stores and the Taipower extract under fixed names, and DATA/SHA256SUMS listing
every file.

- era5_native.zarr: u10, v10, u100, v100 of native 0.25 deg ERA5 on 29x29 cells, the network input.
- era5.zarr: ERA5 on the 41x41 output grid, the ERA5 product.
- icon.zarr: ICON 10 m and 100 m speed on the 41x41 output grid, the training target and the ICON product.
- era5_land.zarr: ERA5-Land 10 m speed on 52x52 cells, NaN over the sea, the ERA5-Land product.
- stations.zarr: hourly CODIS wind_speed_raw and stuck_mask by station_id, with station latitude and longitude.
- taipower/: 10-minute net generation of the six wind farms, for the wind-farm evaluation.

SHA256SUMS is in `sha256sum` format with paths relative to DATA. `verify` checks every file. Opening a store
checks that its files are the ones listed, and the hashes of its metadata, its coordinates and a sample of its
data chunks, so a rebuilt, edited or foreign store fails fast.
"""
import hashlib
from functools import cache
from pathlib import Path

import xarray as xr

STORES = ("era5_native", "era5", "icon", "era5_land", "stations")
TAIPOWER = "taipower"
PARTS = tuple(f"{name}.zarr" for name in STORES) + (TAIPOWER,)
MANIFEST = "SHA256SUMS"
COORDS = ("time", "latitude", "longitude", "station_id")
DATA_CHUNK_SAMPLE = 3


class DataError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_sha256sums(root: Path, parts=PARTS) -> None:
    """root/SHA256SUMS in sha256sum format: `<hex>  <path>` per file of the parts, paths relative to root, sorted."""
    rel = sorted(p.relative_to(root).as_posix() for s in parts for p in (root / s).rglob("*") if p.is_file())
    (root / MANIFEST).write_text("".join(f"{sha256(root / r)}  {r}\n" for r in rel))


@cache
def manifest(data: Path) -> dict[str, str]:
    """Relative path -> sha256 of every file listed in DATA/SHA256SUMS."""
    path = data / MANIFEST
    if not path.is_file():
        raise DataError(f"{path} does not exist; DATA must be a WindSR data directory")
    entries = {}
    for line in path.read_text().splitlines():
        digest, _, rel = line.partition("  ")
        entries[rel] = digest
    return entries


def _files(data: Path, rel: str) -> list[str]:
    """The files under DATA/`rel`, relative to DATA; they must be exactly the ones the manifest lists."""
    listed = [f for f in manifest(data) if f.startswith(rel + "/")]
    if not listed:
        raise DataError(f"{rel} is not in {data / MANIFEST}")
    on_disk = {q.relative_to(data).as_posix() for q in (data / rel).rglob("*") if q.is_file()}
    if on_disk != set(listed):
        extra, missing = sorted(on_disk - set(listed))[:3], sorted(set(listed) - on_disk)[:3]
        raise DataError(f"{rel}: files differ from {MANIFEST} (extra {extra}, missing {missing})")
    return listed


def _sample(listed: list[str]) -> list[str]:
    meta = [f for f in listed if Path(f).name.startswith(".") or Path(f).name == "zarr.json"]
    coords = [f for f in listed if Path(f).parent.name in COORDS and f not in meta]
    chunks = sorted(set(listed) - set(meta) - set(coords))
    step = max(1, len(chunks) // DATA_CHUNK_SAMPLE)
    return meta + coords + chunks[::step][:DATA_CHUNK_SAMPLE]


def _check(data: Path, files: list[str]) -> None:
    entries = manifest(data)
    for f in files:
        if sha256(data / f) != entries[f]:
            raise DataError(f"{f} does not match {data / MANIFEST}")


def store_path(data: Path, name: str) -> Path:
    """The path of store `name` of STORES in DATA."""
    return data / f"{name}.zarr"


@cache
def open_store(data: Path, name: str) -> xr.Dataset:
    """Store `name` of STORES, after the quick check against the manifest."""
    path = store_path(data, name)
    _check(data, _sample(_files(data, path.name)))
    return xr.open_zarr(path)


def taipower(data: Path) -> Path:
    """DATA/taipower, after checking every file of it against the manifest."""
    _check(data, _files(data, TAIPOWER))
    return data / TAIPOWER


def verify(data: Path) -> int:
    """Check every file listed in SHA256SUMS, and that each store and the Taipower extract hold no file the manifest
    lacks. Returns the number of files checked."""
    for rel in PARTS:
        _files(data, rel)
    files = sorted(manifest(data))
    if missing := [f for f in files if not (data / f).is_file()]:
        raise DataError(f"{len(missing)} files listed in {MANIFEST} are missing, e.g. {missing[:3]}")
    _check(data, files)
    return len(files)
