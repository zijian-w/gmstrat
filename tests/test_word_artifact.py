import numpy as np
import pandas as pd

from word import PlanWordResult, WordStat


def test_plan_word_artifact_round_trip(tmp_path):
    df_plans = pd.DataFrame(
        {
            "plan_uid": [0, 1],
            "freq": [0.6, 0.4],
            "plan_vector": [
                np.array([0, 1], dtype=np.int32),
                np.array([1, 2], dtype=np.int32),
            ],
        }
    )
    df_words = pd.DataFrame(
        {
            "plan_uid": [0, 0, 1],
            "plan": [
                np.array([0, 1], dtype=np.int32),
                np.array([0, 1], dtype=np.int32),
                np.array([1, 2], dtype=np.int32),
            ],
            "word": [
                np.array([0, 1], dtype=np.int32),
                np.array([0, 2], dtype=np.int32),
                np.array([0, 1], dtype=np.int32),
            ],
            "word_str": ["0.1", "0.2", "0.1"],
            "word_uid": [0, 1, 0],
            "distance": [1.0, 2.0, 1.0],
            "min_distance": [1.0, 1.0, 1.0],
        }
    )
    district_cluster_distances = [
        [(1.0, 0), (2.0, 1)],
        [(0.5, 1), (1.5, 0)],
    ]
    result = PlanWordResult(
        df_plans=df_plans,
        df_words=df_words,
        district_cluster_distances=district_cluster_distances,
    )

    result.save_artifact(tmp_path)
    loaded = PlanWordResult.load_artifact(tmp_path)

    assert (tmp_path / "plans.npy").is_file()
    assert (tmp_path / "df_plans.feather").is_file()
    assert (tmp_path / "df_words.feather").is_file()
    assert (tmp_path / "district_letter_distances.npy").is_file()
    assert (tmp_path / "district_letter_ids.npy").is_file()
    assert np.array_equal(loaded.df_plans.plan_vector.iloc[0], np.array([0, 1]))
    assert loaded.df_words.columns.tolist() == [
        "plan_uid",
        "word_uid",
        "word_str",
        "distance",
        "min_distance",
    ]
    assert loaded.district_cluster_distances == []

    table = WordStat(loaded, temp=0.0, verbose=False).stationary_table()
    assert table.word.map(tuple).tolist() == [(0, 1), (0, 2)]
