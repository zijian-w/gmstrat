from types import SimpleNamespace

import numpy as np
import pandas as pd

from word import (
    PlanWordBuilder,
    PlanWordResult,
    WordStat,
    compute_fixed_word_stats,
)


class Sample:
    def __init__(self):
        self.population = np.ones(4, dtype=np.int64)
        self.num_precincts = 4
        self.maximum_distance = 4
        self.df_distributions = pd.DataFrame(
            {
                "freq": [0.6, 0.4],
                "plan_vector": [
                    np.array([0, 1], dtype=np.int32),
                    np.array([2, 3], dtype=np.int32),
                ],
            }
        )
        self.districts = [
            np.array([0, 1], dtype=np.int32),
            np.array([2, 3], dtype=np.int32),
            np.array([0, 2], dtype=np.int32),
            np.array([1, 3], dtype=np.int32),
        ]

    def get_all_districts(self):
        return self.districts


def test_fixed_word_stats_matches_materialized_word_stat():
    sp = Sample()
    clusters = SimpleNamespace(centroids=sp.districts)
    plan_words = PlanWordBuilder(
        sp,
        clusters,
        word_degree=16,
        verbose=False,
    ).build()
    fixed_words = (
        plan_words.df_words[["word_uid", "word_str"]]
        .drop_duplicates("word_uid")
        .sort_values("word_uid")
        .head(3)
    )
    filtered = PlanWordResult(
        df_plans=plan_words.df_plans,
        df_words=plan_words.df_words[
            plan_words.df_words.word_uid.isin(fixed_words.word_uid)
        ].copy(),
        district_cluster_distances=[],
    )

    streamed = compute_fixed_word_stats(
        sp,
        clusters,
        fixed_words,
        word_degree=16,
        temperatures=[0, 2],
        total_population=4,
        verbose=False,
    )

    for temperature in [0.0, 2.0]:
        materialized = WordStat(
            filtered,
            temp=temperature,
            total_population=4,
            verbose=False,
        )
        assert np.allclose(
            streamed.stationary[temperature],
            materialized.stationary_distribution(),
        )
        assert np.allclose(
            streamed.flux[temperature],
            materialized.flux_matrix(),
        )

    assert np.isclose(streamed.covered_mass, 1.0)
    assert streamed.covered_plans == 2
