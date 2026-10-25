"""stations.zarr: hourly CODIS WindSpeed.Mean of the 345 Zenodo gs stations in UTC.

CODIS DataTime is naive local time (UTC+08:00), so the UTC value at t is the CODIS value labelled t + 8 h.
Negative missing codes become NaN. Stuck runs are found on each station's full hourly record, where a missing hour
ends a run.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .store import open_store

UTC = pd.date_range("2022-12-01 00:00", "2025-08-29 23:00", freq="h")
LST_OFFSET = pd.Timedelta(hours=8)
STUCK_POS_H, STUCK_ZERO_H = 12, 48
CODIS = "CWA CODIS hourly station reports (https://codis.cwa.gov.tw/StationData), column WindSpeed.Mean, one file per station"
STATION_LIST = "CODIS station list (https://codis.cwa.gov.tw/api/station_list)"
PROFILE = "legacy CODIS station table, column {} (4 decimals)"


def stuck_mask(x: np.ndarray) -> np.ndarray:
    """True on hours inside a run of identical finite values: >= 12 h above 0, or >= 48 h at exactly 0."""
    n = len(x)
    fin = np.isfinite(x)
    brk = np.ones(n, bool)
    brk[1:] = ~(fin[1:] & fin[:-1] & (x[1:] == x[:-1]))
    start = np.flatnonzero(brk)
    length = np.diff(np.append(start, n))
    val = x[start]
    bad = np.isfinite(val) & (((val > 0) & (length >= STUCK_POS_H)) | ((val == 0) & (length >= STUCK_ZERO_H)))
    return np.repeat(bad, length)


def build(zenodo_gs: str, codis: str, out: str) -> None:
    c = Path(codis)
    gs = xr.open_zarr(zenodo_gs).load()
    ids = gs.station_id.values.astype(str)
    assert len(ids) == 345

    raw = np.full((len(UTC), len(ids)), np.nan, np.float64)
    stuck = np.zeros((len(UTC), len(ids)), bool)
    for j, s in enumerate(ids):
        d = pd.read_parquet(c / "data" / "stations" / f"{s}.parquet", columns=["WindSpeed.Mean"])["WindSpeed.Mean"]
        d.index = pd.DatetimeIndex(d.index)
        assert d.index.tz is None and not d.index.duplicated().any(), s
        d = d.sort_index()
        full = pd.date_range(d.index.min(), d.index.max(), freq="h")
        x = d.reindex(full).to_numpy(dtype="float64", copy=True)
        x[x < 0] = np.nan
        win = full.get_indexer(UTC + LST_OFFSET)
        m = stuck_mask(x)
        raw[:, j] = np.where(win >= 0, x[np.clip(win, 0, None)], np.nan)
        stuck[:, j] = np.where(win >= 0, m[np.clip(win, 0, None)], False)
    raw = raw.astype(np.float32)
    ws = np.where(stuck, np.nan, raw).astype(np.float32)

    meta = json.loads((c / "metadata" / "official" / "station_list.json").read_text(encoding="utf-8"))
    official = {}
    for blk in meta["data"]:
        for it in blk["item"]:
            official.setdefault(it["stationID"], it)
    prof = pd.read_csv(c / "metadata" / "station_profiles.csv").set_index("station_id")
    olat = np.array([float(official[s]["latitude"]) if s in official else np.nan for s in ids])
    olon = np.array([float(official[s]["longitude"]) if s in official else np.nan for s in ids])
    remark = np.array([(official.get(s, {}).get("webRemark") or "").replace("\n", " ") for s in ids])
    plat = np.array([prof.loc[s, "latitude_wgs84"] if s in prof.index else np.nan for s in ids], float)
    plon = np.array([prof.loc[s, "longitude_wgs84"] if s in prof.index else np.nan for s in ids], float)
    lat, lon = gs.latitude.values, gs.longitude.values
    off_km = np.hypot((lat - olat) * 111.2, (lon - olon) * 111.2 * np.cos(np.radians(olat)))

    attrs_common = {"source": CODIS,
                    "time_convention": "UTC. CODIS DataTime is local time (fixed UTC+08:00); value at UTC t = CODIS value labelled t + 8 h.",
                    "units": "m s-1"}
    ds = xr.Dataset(
        {"wind_speed_raw": (("time", "station_id"), raw, attrs_common | {"long_name": "10 min mean wind speed, hourly, CODIS missing codes (negative values, e.g. -999) set to NaN; calms kept"}),
         "stuck_mask": (("time", "station_id"), stuck, {"long_name": "stuck sensor: run of >= 12 consecutive hours with the same value > 0, or >= 48 consecutive hours at exactly 0; runs found on the full CODIS record"}),
         "wind_speed": (("time", "station_id"), ws, attrs_common | {"long_name": "wind_speed_raw with stuck_mask hours set to NaN; calms kept"})},
        coords={"time": UTC.values, "station_id": ids,
                "latitude": ("station_id", lat, {"source": "Zenodo gs"}), "longitude": ("station_id", lon, {"source": "Zenodo gs"}),
                "elevation": ("station_id", gs.elevation.values, {"source": "Zenodo gs"}), "group": ("station_id", gs.group.values, {"source": "Zenodo gs"}),
                "grid_id": ("station_id", gs.grid_id.values.astype(str), {"source": "Zenodo gs"}),
                "official_latitude": ("station_id", olat, {"source": STATION_LIST}),
                "official_longitude": ("station_id", olon, {"source": STATION_LIST}),
                "official_offset_km": ("station_id", off_km, {"long_name": "distance between Zenodo gs and official coordinates"}),
                "official_offset_gt_0p2km": ("station_id", off_km > 0.2, {"long_name": "official coordinates differ from Zenodo gs by more than 0.2 km"}),
                "profile_latitude": ("station_id", plat, {"source": PROFILE.format("latitude_wgs84")}),
                "profile_longitude": ("station_id", plon, {"source": PROFILE.format("longitude_wgs84")}),
                "official_web_remark": ("station_id", remark)},
        attrs={"title": "345 Zenodo gs stations, hourly CODIS WindSpeed.Mean in UTC", "built_by": "windsr build-data"})
    enc = {v: {"chunks": (720, 345)} for v in ["wind_speed_raw", "stuck_mask", "wind_speed"]}
    ds.to_zarr(out, mode="w", consolidated=True, encoding=enc, zarr_format=2)
    print("stations built", dict(ds.sizes), "stuck hours", int(stuck.sum()))


@dataclass(frozen=True)
class Stations:
    ids: np.ndarray
    lat: np.ndarray
    lon: np.ndarray


def stations(data: Path) -> Stations:
    ds = open_store(data, "stations")
    return Stations(ds.station_id.values.astype(str), ds.latitude.values.astype(float), ds.longitude.values.astype(float))
