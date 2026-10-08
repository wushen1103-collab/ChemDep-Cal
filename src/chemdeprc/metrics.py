from __future__ import annotations

import numpy as np


def evaluate_selection(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    blocks: np.ndarray,
    selected: np.ndarray,
    budget: int,
    q: float,
    within_similarity: float,
    between_similarity: float,
) -> dict[str, float]:
    selected = np.asarray(selected, dtype=int)
    n_selected = int(len(selected))
    active_rate = float(np.mean(labels)) if len(labels) else 0.0
    total_hits = int(np.sum(labels))

    if n_selected == 0:
        true_hits = 0
        false_hits = 0
        fdp = 0.0
        precision = 0.0
        unique_blocks = 0
        max_block_share = 0.0
        mean_pairwise_tanimoto = 0.0
    else:
        selected_labels = labels[selected]
        true_hits = int(np.sum(selected_labels))
        false_hits = int(n_selected - true_hits)
        fdp = float(false_hits / n_selected)
        precision = float(true_hits / n_selected)
        selected_blocks = blocks[selected]
        _, counts = np.unique(selected_blocks, return_counts=True)
        unique_blocks = int(len(counts))
        max_block_share = float(counts.max() / n_selected)
        mean_pairwise_tanimoto = _mean_pairwise_similarity_from_blocks(
            counts, n_selected, within_similarity, between_similarity
        )

    power = float(true_hits / total_hits) if total_hits > 0 else 0.0
    ef = float(precision / active_rate) if active_rate > 0 else 0.0
    top_budget = np.argsort(-scores, kind="mergesort")[:budget]
    top_budget_hits = int(np.sum(labels[top_budget])) if budget > 0 else 0
    raw_top_budget_precision = float(top_budget_hits / max(len(top_budget), 1))

    return {
        "selected_count": float(n_selected),
        "true_hits": float(true_hits),
        "false_hits": float(false_hits),
        "fdp": fdp,
        "fdp_exceeds_q": float(fdp > q),
        "precision": precision,
        "power": power,
        "enrichment_factor": ef,
        "unique_blocks": float(unique_blocks),
        "max_block_share": max_block_share,
        "mean_pairwise_tanimoto": mean_pairwise_tanimoto,
        "empty": float(n_selected == 0),
        "pool_active_rate": active_rate,
        "raw_top_budget_precision": raw_top_budget_precision,
    }


def _mean_pairwise_similarity_from_blocks(
    block_counts: np.ndarray,
    n_selected: int,
    within_similarity: float,
    between_similarity: float,
) -> float:
    if n_selected < 2:
        return 0.0
    total_pairs = n_selected * (n_selected - 1) / 2
    same_pairs = float(np.sum(block_counts * (block_counts - 1) / 2))
    same_fraction = same_pairs / total_pairs
    return float(same_fraction * within_similarity + (1.0 - same_fraction) * between_similarity)

