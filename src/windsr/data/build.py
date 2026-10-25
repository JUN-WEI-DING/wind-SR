"""Build the data directory from raw sources (DATA.md).

usage: windsr build-data OUT --era5 LIST --zenodo DIR --era5-land DIR --codis DIR --taipower DIR

OUT gets era5_native.zarr, era5.zarr, icon.zarr, era5_land.zarr, stations.zarr, the Taipower extract taipower/ and
SHA256SUMS over every file of them. Rebuilding into the same OUT replaces them.
"""
import argparse
from pathlib import Path

from . import era5, era5_land, icon, stations, taipower
from .store import MANIFEST, TAIPOWER, store_path, write_sha256sums


def cli(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="windsr build-data", description="Build the WindSR data directory from raw sources.")
    p.add_argument("out", type=Path, help="data directory to write")
    p.add_argument("--era5", type=Path, required=True,
                   help="text file, one daily ERA5 Zarr store (ARCO-ERA5 variable names) per line, in time order, 2022-12-01 to 2025-08-29")
    p.add_argument("--zenodo", type=Path, required=True,
                   help="unpacked Zenodo record doi:10.5281/zenodo.17647355 holding lr.zarr, hr.zarr, bm.zarr and gs.zarr")
    p.add_argument("--era5-land", type=Path, required=True,
                   help="directory of the ERA5-Land u10 NetCDF subsets from NSF NCAR GDEX d633008")
    p.add_argument("--codis", type=Path, required=True,
                   help="CODIS directory with data/stations/<ID>.parquet, metadata/official/station_list.json and metadata/station_profiles.csv")
    p.add_argument("--taipower", type=Path, required=True,
                   help="Taipower unit-generation release with release.json, partitions/<YYYY-MM>/data.parquet and metadata/observation_quality_flags.csv")
    a = p.parse_args(argv)
    paths = [line.strip() for line in a.era5.read_text().splitlines() if line.strip()]
    a.out.mkdir(parents=True, exist_ok=True)
    era5.build(paths, str(a.zenodo / "lr.zarr"), str(store_path(a.out, "era5_native")), str(store_path(a.out, "era5")))
    icon.build(str(a.zenodo / "hr.zarr"), str(store_path(a.out, "icon")))
    era5_land.build(str(a.zenodo / "bm.zarr"), str(a.era5_land), str(store_path(a.out, "era5_land")))
    stations.build(str(a.zenodo / "gs.zarr"), str(a.codis), str(store_path(a.out, "stations")))
    taipower.build(str(a.taipower), str(a.out / TAIPOWER))
    write_sha256sums(a.out)
    print(f"{MANIFEST} written", len((a.out / MANIFEST).read_text().splitlines()), "files")
