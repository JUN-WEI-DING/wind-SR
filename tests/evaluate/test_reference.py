from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from windsr.evaluate.metrics import score_product, summary
from windsr.evaluate.products import RIVALS, rivals

pytestmark = pytest.mark.data
REFERENCE = Path(__file__).parent / "reference_rivals.csv"


def test_rivals_reproduce_the_frozen_main_table(data):
    loaders = rivals(data)
    got = summary([score_product(data, n, loaders[n]) for n in RIVALS])
    ref = pd.read_csv(REFERENCE)
    assert got[["product", "variant"]].equals(ref[["product", "variant"]])
    for c in ("rmse", "bias", "corr", "crmse", "n_stations"):
        assert np.array_equal(got[c].to_numpy(float), ref[c].to_numpy(float)), c
