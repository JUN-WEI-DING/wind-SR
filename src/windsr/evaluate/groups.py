"""Station groups of the paper's clustering-based validation (Ding & Hsieh 2026, Methods, Algorithm 1).

Rules taken from the paper:
- k is the largest number of stations in one 0.125 deg grid cell.
- The k stations of the densest cell start the k groups, one each.
- Groups take turns. On its turn a group takes one unassigned station from the cell with the fewest
  stations assigned so far, choosing the station farthest from the group's nearest member (Eq. 8).
- A group never holds two stations of the same cell.

Choices the paper leaves open, fixed here:
- Ties between cells with the same count are broken by Eq. 8 over all their unassigned stations.
- Eq. 8 (the largest minimum distance) is used, not the "total distance" of Algorithm 1 line 13.
- Distances are in km on an equirectangular projection at each station pair's mean latitude.
- The densest cell with the lowest index starts; its stations seed groups in station-id order.
- A group with no allowed cell left skips its turn.
"""
from functools import cache
from pathlib import Path

import numpy as np

from ..data.stations import stations
from ..grid import nearest_cell

KM_PER_DEG = 111.195


def distance_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    mlat = np.radians((lat[:, None] + lat[None]) / 2)
    dy = lat[:, None] - lat[None]
    dx = (lon[:, None] - lon[None]) * np.cos(mlat)
    return KM_PER_DEG * np.hypot(dx, dy)


def paper_groups(ids: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Group number (0..k-1) of each station, in the order of `ids`."""
    cell = nearest_cell(lat, lon)
    cells, counts = np.unique(cell, return_counts=True)
    k = int(counts.max())
    dist = distance_km(lat, lon)
    group = np.full(len(ids), -1)
    assigned_per_cell = dict.fromkeys(cells.tolist(), 0)

    seed_cell = cells[counts == k].min()
    for g, s in enumerate(sorted(np.flatnonzero(cell == seed_cell), key=lambda i: ids[i])):
        group[s] = g
        assigned_per_cell[seed_cell] += 1

    while (group < 0).any():
        progressed = False
        for g in range(k):
            members = np.flatnonzero(group == g)
            taken = set(cell[members].tolist())
            free = np.flatnonzero((group < 0) & ~np.isin(cell, list(taken)))
            if len(free) == 0:
                continue
            fewest = min(assigned_per_cell[c] for c in cell[free])
            cand = free[[assigned_per_cell[c] == fewest for c in cell[free]]]
            s = cand[dist[np.ix_(cand, members)].min(1).argmax()]
            group[s] = g
            assigned_per_cell[cell[s]] += 1
            progressed = True
        if not progressed:
            raise RuntimeError("stations left that no group may take")
    return group


@cache
def station_groups(data: Path) -> dict[str, int]:
    """Group of every station of DATA, built on all of its stations."""
    st = stations(data)
    return dict(zip(st.ids.tolist(), paper_groups(st.ids, st.lat, st.lon).tolist()))
