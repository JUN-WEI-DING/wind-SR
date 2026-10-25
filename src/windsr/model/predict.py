"""Predictions of the trained network and their stores.

RUN/test.zarr holds the test year and RUN/train.zarr the training period (whose validation weeks serve the
calibration), each with wind_speed_10m and wind_speed_100m on time/latitude/longitude of the output grid, from the
weight-averaged generator of RUN/epoch050.pt and the normalisation RUN/norm.json.
"""
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from ..grid import GRID_LAT, GRID_LON
from ..periods import TEST, TRAIN
from .inputs import HEIGHTS, Fields, Norm, fields
from .network import Generator

CHECKPOINT = "epoch050.pt"
STORES = {"test": TEST, "train": TRAIN}


def load_generator(path: Path, device: str = "cuda") -> Generator:
    """The weight-averaged generator of a checkpoint, the one that is scored."""
    g = Generator().to(device)
    g.load_state_dict(torch.load(path, map_location=device)["generator_ema"])
    return g.eval()


@torch.no_grad()
def predict(g: Generator, data: Fields, norm: Norm, batch: int = 256, device: str = "cuda") -> np.ndarray:
    """Speed at the HEIGHTS in m/s, (hours, 2, 41, 41)."""
    out = []
    for s in range(0, len(data.x), batch):
        xb = torch.tensor(data.x[s:s + batch], device=device)
        bb = torch.tensor(data.base[s:s + batch], device=device)
        out.append(norm.to_ms(g(xb, bb).cpu().numpy()))
    return np.concatenate(out).astype(np.float32)


def write_store(path: Path, times: np.ndarray, ws: np.ndarray) -> str:
    """A new .zarr with wind_speed_10m and wind_speed_100m on time/latitude/longitude from (hours, 2, 41, 41) m/s;
    refuses to overwrite."""
    coords = {"time": times, "latitude": GRID_LAT, "longitude": GRID_LON}
    ds = xr.Dataset({f"wind_speed_{h}": (("time", "latitude", "longitude"), ws[:, j]) for j, h in enumerate(HEIGHTS)},
                    coords=coords)
    ds.chunk({"time": 720}).to_zarr(path, mode="w-", zarr_format=2)
    return str(path)


def predict_run(data: Path, run: Path, device: str = "cuda") -> dict[str, str]:
    """Writes RUN/test.zarr and RUN/train.zarr; returns their paths by name."""
    norm = Norm.load(run / "norm.json")
    g = load_generator(run / CHECKPOINT, device)
    stores = {}
    for name, period in STORES.items():
        ws = predict(g, fields(data, period, norm), norm, device=device)
        stores[name] = write_store(run / f"{name}.zarr", period.hours, ws)
    return stores
