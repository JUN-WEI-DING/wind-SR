import numpy as np
import pytest

from windsr.data.stations import stations
from windsr.evaluate.groups import paper_groups, station_groups
from windsr.grid import GRID_LAT, GRID_LON, nearest_cell


def _toy():
    r = np.random.default_rng(0)
    cells = r.integers(0, 30, 80)
    cells[:4] = 7
    lat = GRID_LAT[cells % 10 + 10] + r.uniform(-0.05, 0.05, 80)
    lon = GRID_LON[cells // 10 + 10] + r.uniform(-0.05, 0.05, 80)
    return np.array([f"S{i:02d}" for i in range(80)]), lat, lon


def test_one_station_per_cell_in_each_group_and_every_station_assigned():
    ids, lat, lon = _toy()
    g = paper_groups(ids, lat, lon)
    cell = nearest_cell(lat, lon)
    k = np.unique(cell, return_counts=True)[1].max()
    assert set(g) == set(range(k))
    for j in range(k):
        assert len(set(cell[g == j])) == (g == j).sum()


def test_groups_are_deterministic_and_balanced():
    ids, lat, lon = _toy()
    g = paper_groups(ids, lat, lon)
    assert np.array_equal(g, paper_groups(ids, lat, lon))
    sizes = np.bincount(g)
    assert sizes.max() - sizes.min() <= 1


@pytest.mark.data
def test_the_345_stations_form_six_groups_without_shared_cells(data):
    st = stations(data)
    g = paper_groups(st.ids, st.lat, st.lon)
    cell = nearest_cell(st.lat, st.lon)
    assert g.max() + 1 == 6
    assert np.bincount(g).sum() == 345 and np.ptp(np.bincount(g)) <= 3
    for j in range(6):
        assert len(set(cell[g == j])) == (g == j).sum()
    assert sorted(np.bincount(g)) == [56, 57, 57, 58, 58, 59]
    assert station_groups(data) == dict(zip(st.ids, g))
