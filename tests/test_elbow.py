from types import SimpleNamespace

import numpy as np

from elbow import compute_elbow, plot_elbow


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


def test_compute_and_plot_elbow(tmp_path):
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

    df = compute_elbow(SampleStub(linkage_path), range(2, 4))

    assert df["k"].tolist() == [2, 3]
    assert df["wcss"].tolist() == [2.0, 1.0]
    assert df["wcss_normalized"].tolist() == [1.0, 0.5]
    assert np.isnan(df["improvement"].iloc[0])
    assert df["improvement"].iloc[1:].tolist() == [0.5]

    fig, axes = plot_elbow(df, selected_k=3)
    assert len(axes) == 2
    fig.clear()
