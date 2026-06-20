from types import SimpleNamespace

import geopandas as gpd
import numpy as np

from hierachical import HClusters
from merge import plot_merges


class SampleStub:
    def __init__(self, linkage_path):
        self.paths = SimpleNamespace(linkage=linkage_path)
        self.population = np.ones(4, dtype=np.int32)
        self.maximum_distance = 8
        self.num_precincts = 4
        self._districts = [
            np.array([0]),
            np.array([0, 1]),
            np.array([2]),
            np.array([2, 3]),
        ]

    def get_all_districts(self):
        return self._districts

    def compute_distance(self, left, right):
        return len(np.setxor1d(left, right))


def test_merge_and_plot(tmp_path):
    linkage_path = tmp_path / "linkage.npy"
    np.save(
        linkage_path,
        np.array(
            [
                [0, 1, 1, 2],
                [2, 3, 1, 2],
                [4, 5, 2, 4],
            ]
        ),
    )
    sp = SampleStub(linkage_path)
    hc = HClusters(sp)

    left, right, merged = hc.merge_from_k(2)
    assert left == [0, 1]
    assert right == [2, 3]
    assert merged == [0, 1, 2, 3]

    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(range(4), np.zeros(4)))
    fig, axes = plot_merges(sp, hc, gdf, k_values=(2, 3))
    assert axes.shape == (2, 3)
    fig.clear()
