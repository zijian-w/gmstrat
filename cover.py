from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class GreedyCoverResult:
    selected_sets: np.ndarray
    uncovered_nodes: np.ndarray


@dataclass(frozen=True)
class TwoStageCoverResult:
    core_sets: np.ndarray
    tail_sets: np.ndarray
    core_uncovered_nodes: np.ndarray
    tail_uncovered_nodes: np.ndarray

    @property
    def selected_sets(self) -> np.ndarray:
        return np.concatenate([self.core_sets, self.tail_sets])


def _as_csc(covermat):
    covermat = sparse.csc_matrix(covermat, dtype=bool)
    covermat.sum_duplicates()
    covermat.eliminate_zeros()
    return covermat


def greedy_partial_cover(covermat) -> GreedyCoverResult:
    covermat = _as_csc(covermat)
    num_nodes, num_sets = covermat.shape
    if num_sets == 0:
        return GreedyCoverResult(
            selected_sets=np.asarray([], dtype=int),
            uncovered_nodes=np.arange(num_nodes),
        )
    remaining = np.ones(num_sets, dtype=bool)
    uncovered = np.ones(num_nodes, dtype=np.int32)
    selected_sets = []

    while uncovered.any():
        gains = np.asarray(covermat.T @ uncovered).ravel()
        gains[~remaining] = -1
        best_set = int(np.argmax(gains))
        if gains[best_set] <= 0:
            break
        selected_sets.append(best_set)
        start = covermat.indptr[best_set]
        stop = covermat.indptr[best_set + 1]
        uncovered[covermat.indices[start:stop]] = 0
        remaining[best_set] = False

    return GreedyCoverResult(
        selected_sets=np.asarray(selected_sets, dtype=int),
        uncovered_nodes=np.flatnonzero(uncovered),
    )


def greedy_two_stage_cover(core_covermat, broad_covermat) -> TwoStageCoverResult:
    core_covermat = _as_csc(core_covermat)
    broad_covermat = _as_csc(broad_covermat)
    core = greedy_partial_cover(core_covermat)

    broad_core_coverage = np.asarray(
        broad_covermat[core.uncovered_nodes, :][:, core.selected_sets].getnnz(axis=1)
    ).ravel()
    repair_nodes = core.uncovered_nodes[broad_core_coverage == 0]

    candidate_sets = np.setdiff1d(
        np.arange(core_covermat.shape[1]),
        core.selected_sets,
    )
    repair_covermat = broad_covermat[repair_nodes, :][:, candidate_sets]
    repair = greedy_partial_cover(repair_covermat)

    return TwoStageCoverResult(
        core_sets=core.selected_sets,
        tail_sets=candidate_sets[repair.selected_sets],
        core_uncovered_nodes=core.uncovered_nodes,
        tail_uncovered_nodes=repair_nodes[repair.uncovered_nodes],
    )
