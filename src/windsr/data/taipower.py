"""The Taipower extract: 10-minute net generation of the six wind farms of `FARMS`, cut from a Taipower
unit-generation release to the columns and quality runs that `read` reads.

The release keeps Taipower's layout: `partitions/<YYYY-MM>/data.parquet` by local month and
`metadata/observation_quality_flags.csv`. The extract holds the local months that cover the hours of `stations.UTC`
plus one day either side.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..periods import Period
from .stations import UTC
from .store import sha256

LOCAL_TZ = "Asia/Taipei"
RUNS = ("intrinsic-constant-nonzero-run-24h-v1", "intrinsic-zero-run-7d-v1")
"""The rule ids of the quality runs the extract keeps."""
COLUMNS = ["timestamp", "unit_name", "net_generation_mw", "capacity_mw", "qa_exclude", "eligible_reported_item_analysis"]
RUN_COLUMNS = ["unit_name", "rule_id", "run_first_timestamp", "run_last_timestamp"]
SOURCE = ("Taiwan Power Company (台灣電力股份有限公司), 台灣電力公司各機組過去發電量 (data.gov.tw dataset 37331) and "
          "台灣電力公司各機組發電量即時資訊 (data.gov.tw dataset 8931), Open Government Data License version 1.0 "
          "(https://data.gov.tw/license)")


@dataclass(frozen=True)
class Farm:
    """One farm: its Taipower `unit_name`, the paper's turbine model, hub height and location, the installed MW used
    where a row has no capacity, and the commissioning year. Each `*_source` says where a value comes from."""

    unit_name: str
    name: str
    turbine: str
    hub_m: float
    lat: float
    lon: float
    capacity_mw: float
    capacity_source: str
    commissioned: int
    commissioned_source: str
    note: str = ""


EQUIPMENT = "Taipower dataset metadata/reference_equipment.csv"
REPORTED = "Taipower dataset metadata/reported_items.csv"

FARMS = (
    Farm("海洋竹南", "Formosa 1 Offshore", "SWT-6.0-154", 90, 24.70, 120.82, 128.0,
         f"{EQUIPMENT} energy_admin_private_renewable table 4 rows 55-56: 8 + 120 MW; equals the rows' capacity_mw",
         2019, f"{EQUIPMENT} energy_admin_private_renewable table 4 row 56: commercial year ROC 108 for the 120 MW "
               "second stage (the 8 MW first stage: ROC 106)"),
    Farm("離岸一期", "Offshore Phase 1", "HTW5.2-127", 90, 23.96, 120.27, 109.2,
         f"{EQUIPMENT} taipower_renewable_site_data row 27: 21 x 5.2 MW",
         2021, f"inferred, no commercial date in the metadata: {REPORTED} first_timestamp 2021-04-18; the footnoted "
               "item 離岸一期(註10) reports from 2020-12-29",
         "site data gives hub height 95-99 m; the paper's table gives 90 m"),
    Farm("鹿威彰濱", "Lu-Wei Changbin", "Enercon E70", 64, 24.09, 120.38, 55.2,
         f"{REPORTED} capacity_max_mw; equals the rows' capacity_mw",
         2007, f"{EQUIPMENT} energy_admin_private_renewable table 4 row 21: 彰濱風力發電廠 commercial years ROC 96-98 "
               "(2007-2009); which of 鹿威's units form this 55.2 MW item is not recorded"),
    Farm("彰工", "Changgong", "Vestas V80", 67, 24.14, 120.42, 86.2,
         f"{EQUIPMENT} taipower_renewable_site_data rows 13-16: 46 + 16 + 9.2 + 15 MW",
         2011, f"assumed, not in the metadata; {REPORTED} shows it reporting since the dataset start "
               "2013-08-17, so 2013 is an upper bound",
         "site data: 31 Vestas V80 at 67 m, 4 Enercon E70 at 76 m, 5 Enercon E82 at 78 m (grid-connected "
         "2023-01-18); the paper models all as V80"),
    Farm("王功", "Wanggong", "Enercon E70", 75, 23.99, 120.33, 23.0,
         f"{EQUIPMENT} taipower_renewable_site_data row 17: 10 x 2.3 MW",
         2010, f"assumed, not in the metadata; {REPORTED} shows it reporting since the dataset start "
               "2013-08-17, so 2013 is an upper bound",
         "site data gives hub height 76 m; the paper's table gives 75 m"),
    Farm("雲麥", "YunMai", "Vestas V80", 67, 23.81, 120.26, 46.0,
         f"{EQUIPMENT} taipower_renewable_site_data rows 19-20: 30 + 16 MW",
         2011, f"assumed, not in the metadata; {REPORTED} shows it reporting since the dataset start "
               "2013-08-17 (as 麥寮 until 2016-02-19), so 2013 is an upper bound"),
)


def window(start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The bounds of the UTC hours from `start` to `end` plus one day either side."""
    return pd.Timestamp(start) - pd.Timedelta("1D"), pd.Timestamp(end) + pd.Timedelta("1D")


def months(start, end) -> pd.PeriodIndex:
    """The local months that cover `window(start, end)`."""
    local = [t.tz_localize("UTC").tz_convert(LOCAL_TZ).tz_localize(None) for t in window(start, end)]
    return pd.period_range(*local, freq="M")


def build(src: str, out: str) -> None:
    src, out = Path(src), Path(out)
    names = [f.unit_name for f in FARMS]
    extract = months(UTC[0], UTC[-1])
    for m in extract:
        t = pq.read_table(src / "partitions" / str(m) / "data.parquet", columns=COLUMNS,
                          filters=pc.field("unit_name").isin(names))
        t = t.sort_by([("unit_name", "ascending"), ("timestamp", "ascending")]).replace_schema_metadata(None)
        (out / "partitions" / str(m)).mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out / "partitions" / str(m) / "data.parquet")
    runs = pd.read_csv(src / "metadata" / "observation_quality_flags.csv", usecols=RUN_COLUMNS, dtype=str)
    runs = runs[runs.unit_name.isin(names) & runs.rule_id.isin(RUNS)][RUN_COLUMNS]
    (out / "metadata").mkdir(parents=True, exist_ok=True)
    runs.drop_duplicates().sort_values(RUN_COLUMNS).to_csv(out / "metadata" / "observation_quality_flags.csv", index=False)
    release = json.loads((src / "release.json").read_text())["release_id"]
    (out / "release.json").write_text(json.dumps({"release_id": release, "source": SOURCE,
                                                  "farms": names, "columns": COLUMNS}, ensure_ascii=False, indent=1) + "\n")
    print("taipower extract built", len(extract), "months", len(runs), "excluded runs")


@dataclass(frozen=True)
class Release:
    """What `read` takes from a release: its id and resolved path, the sha256 of every file read, the 10-minute rows
    of the units with their UTC `time`, and the units' runs of RUNS with UTC bounds."""

    release_id: str
    path: str
    files: dict[str, str]
    rows: pd.DataFrame
    runs: pd.DataFrame


def read(root: str, period: Period, units: Sequence[str]) -> Release:
    """Read the Taipower unit-generation release at `root`: only the monthly partitions of `period` plus one day
    either side, and only the rows of `units`. 10-minute stamps are converted to UTC."""
    base = Path(root).resolve()
    release = json.loads((base / "release.json").read_text())["release_id"]
    lo, hi = window(period.start, period.end)
    files, tables = {}, []
    for m in months(period.start, period.end):
        path = base / "partitions" / str(m) / "data.parquet"
        files[path.relative_to(base).as_posix()] = sha256(path)
        tables.append(pq.read_table(path, columns=COLUMNS, filters=pc.field("unit_name").isin(units)).to_pandas())
    flags_path = base / "metadata" / "observation_quality_flags.csv"
    files[flags_path.relative_to(base).as_posix()] = sha256(flags_path)
    runs = pd.read_csv(flags_path, usecols=RUN_COLUMNS)
    runs = runs[runs.unit_name.isin(units) & runs.rule_id.isin(RUNS)].drop_duplicates()
    for c in ("run_first_timestamp", "run_last_timestamp"):
        runs[c] = pd.to_datetime(runs[c], utc=True).dt.tz_localize(None)
    rows = pd.concat(tables, ignore_index=True)
    rows["time"] = rows.timestamp.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
    rows = rows[(rows.time >= lo) & (rows.time <= hi)]
    return Release(release, base.as_posix(), files, rows, runs)
