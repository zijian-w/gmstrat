import numpy as np
from scipy import sparse

from cover import greedy_partial_cover, greedy_two_stage_cover


def make_covermat(rows, cols, shape):
    return sparse.coo_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)),
        shape=shape,
    )


def test_greedy_partial_cover():
    covermat = make_covermat(
        rows=[0, 1, 2],
        cols=[0, 0, 1],
        shape=(4, 3),
    )
    result = greedy_partial_cover(covermat)
    assert result.selected_sets.tolist() == [0, 1]
    assert result.uncovered_nodes.tolist() == [3]


def test_greedy_partial_cover_without_candidates():
    result = greedy_partial_cover(sparse.csc_matrix((3, 0), dtype=bool))
    assert result.selected_sets.tolist() == []
    assert result.uncovered_nodes.tolist() == [0, 1, 2]


def test_greedy_two_stage_cover():
    core = make_covermat(
        rows=[0, 1],
        cols=[0, 0],
        shape=(4, 5),
    )
    broad = make_covermat(
        rows=[0, 1, 2, 2, 3],
        cols=[0, 0, 0, 1, 4],
        shape=(4, 5),
    )
    result = greedy_two_stage_cover(core, broad)
    assert result.core_sets.tolist() == [0]
    assert result.core_uncovered_nodes.tolist() == [2, 3]
    assert result.tail_sets.tolist() == [4]
    assert result.tail_uncovered_nodes.tolist() == []
    assert result.selected_sets.tolist() == [0, 4]
