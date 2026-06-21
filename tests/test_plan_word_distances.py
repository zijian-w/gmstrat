import numpy as np
import pandas as pd

from word import plan_word_distances


class Sample:
    population = np.ones(4, dtype=np.int32)
    num_precincts = 4
    maximum_distance = 4


class Clusters:
    centroids = [np.array([0, 1]), np.array([2, 3])]


def test_plan_word_distances():
    df_words = pd.DataFrame(
        {
            "word_uid": [0, 1],
            "word_str": ["0.0", "0.1"],
        }
    )
    table = plan_word_distances(
        Sample(),
        Clusters(),
        np.array([0, 0, 1, 1]),
        df_words,
    )
    assert table.word_uid.tolist() == [1, 0]
    assert table.distance.tolist() == [0, 4]
