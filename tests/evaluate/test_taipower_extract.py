import json
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from windsr.data import taipower
from windsr.data.taipower import FARMS
from windsr.evaluate.windfarm import observations
from windsr.periods import Period


def test_taipower_extract_keeps_what_the_farm_evaluation_reads(tmp_path, monkeypatch):
    src, out = tmp_path / "src", tmp_path / "out"
    (src / "partitions" / "2024-07").mkdir(parents=True)
    (src / "metadata").mkdir()
    (src / "release.json").write_text(json.dumps({"release_id": "release-source", "other": 1}))
    local = pd.date_range("2024-07-10 07:00", "2024-07-10 12:00", freq="10min", tz="Asia/Taipei")
    units = [FARMS[3].unit_name] * len(local) + ["其他"] * len(local)
    mw = [Decimal("20.000")] * len(local) + [Decimal("99.000")] * len(local)
    pq.write_table(pa.table({
        "timestamp": pa.array(list(local) * 2, pa.timestamp("ms", tz="Asia/Taipei")), "unit_name": units,
        "net_generation_mw": pa.array(mw, pa.decimal128(14, 3)),
        "capacity_mw": pa.array([Decimal("80.000")] * len(units), pa.decimal128(14, 3)),
        "qa_exclude": [False] * len(units), "eligible_reported_item_analysis": [True] * len(units),
        "note": ["x"] * len(units)}), src / "partitions" / "2024-07" / "data.parquet")
    pd.DataFrame({"unit_name": [FARMS[3].unit_name, FARMS[3].unit_name, "其他"],
                  "rule_id": ["intrinsic-zero-run-7d-v1", "intrinsic-negative-generation-v1", "intrinsic-zero-run-7d-v1"],
                  "run_first_timestamp": ["2024-07-10T07:50:00+08:00"] * 3, "run_last_timestamp": ["2024-07-10T07:50:00+08:00"] * 3,
                  "severity": ["descriptive"] * 3}).to_csv(src / "metadata" / "observation_quality_flags.csv", index=False)
    monkeypatch.setattr(taipower, "UTC", pd.date_range("2024-07-10 00:00", "2024-07-10 23:00", freq="h"))
    taipower.build(str(src), str(out))
    table = pq.read_table(out / "partitions" / "2024-07" / "data.parquet")
    assert table.column_names == taipower.COLUMNS and set(table.column("unit_name").to_pylist()) == {FARMS[3].unit_name}
    assert len(pd.read_csv(out / "metadata" / "observation_quality_flags.csv")) == 1
    period = Period("p", np.datetime64("2024-07-09T23", "ns"), np.datetime64("2024-07-10T03", "ns"))
    a, b = observations(str(src), period), observations(str(out), period)
    np.testing.assert_array_equal(a.cf, b.cf)
    assert b.release_id == "release-source" and a.coverage.equals(b.coverage)
