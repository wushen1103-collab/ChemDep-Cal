from __future__ import annotations

import hashlib
import math

import numpy as np

from chemdeprc.selection import SelectionResult


DIVERSITY_METHODS = {
    "random_b",
    "score_leader70",
    "score_maxmin",
    "score_mmr25",
    "score_mmr50",
    "score_dpp",
}


def is_diversity_method(method: str) -> bool:
    return method.lower() in DIVERSITY_METHODS


def stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def select_diversity(
    method: str,
    *,
    scores: np.ndarray,
    bits: np.ndarray,
    budget: int,
    seed: int,
) -> SelectionResult:
    method = method.lower()
    scores = np.asarray(scores, dtype=float)
    bits = np.asarray(bits, dtype=np.bool_)
    budget = max(0, min(int(budget), len(scores)))
    if budget == 0 or len(scores) == 0:
        return SelectionResult(method, np.array([], dtype=int), np.nan, "empty_budget_or_pool")

    if method == "random_b":
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(scores), size=budget, replace=False)
        return SelectionResult(method, np.sort(selected.astype(int)), np.nan, "random_budget_no_scores")
    if method == "score_leader70":
        selected = _leader_by_score(scores, bits, budget, threshold=0.70)
        return SelectionResult(method, selected, np.nan, "score_ranked_leader_tanimoto70")
    if method == "score_maxmin":
        selected = _maxmin(scores, bits, budget)
        return SelectionResult(method, selected, np.nan, "score_seeded_maxmin_tanimoto")
    if method == "score_mmr25":
        selected = _mmr(scores, bits, budget, diversity_weight=0.25)
        return SelectionResult(method, selected, np.nan, "score_mmr_diversity25_tanimoto")
    if method == "score_mmr50":
        selected = _mmr(scores, bits, budget, diversity_weight=0.50)
        return SelectionResult(method, selected, np.nan, "score_mmr_diversity50_tanimoto")
    if method == "score_dpp":
        selected = _dpp_greedy(scores, bits, budget)
        return SelectionResult(method, selected, np.nan, "score_quality_tanimoto_dpp_greedy")
    raise ValueError(f"Unknown diversity method {method!r}")


def _candidate_pool(scores: np.ndarray, budget: int, *, min_pool: int = 512, max_pool: int = 2048) -> np.ndarray:
    pool_size = min(len(scores), max(min_pool, 20 * max(1, budget)))
    pool_size = min(pool_size, max_pool)
    return np.argsort(-scores, kind="mergesort")[:pool_size].astype(int)


def _normalise_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return np.ones_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


def _tanimoto_to_all(bits_uint: np.ndarray, bit_counts: np.ndarray, idx: int) -> np.ndarray:
    intersections = bits_uint @ bits_uint[idx]
    unions = bit_counts + bit_counts[idx] - intersections
    return intersections / np.maximum(unions, 1)


def _leader_by_score(scores: np.ndarray, bits: np.ndarray, budget: int, *, threshold: float) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    bits_uint = bits.astype(np.uint16, copy=False)
    bit_counts = bits_uint.sum(axis=1, dtype=np.uint16)
    selected: list[int] = []
    selected_mask = np.zeros(len(scores), dtype=bool)
    for idx in order:
        if len(selected) >= budget:
            break
        if not selected:
            selected.append(int(idx))
            selected_mask[int(idx)] = True
            continue
        candidate = bits_uint[int(idx)]
        intersections = bits_uint[selected] @ candidate
        unions = bit_counts[selected] + bit_counts[int(idx)] - intersections
        max_sim = float(np.max(intersections / np.maximum(unions, 1)))
        if max_sim < threshold:
            selected.append(int(idx))
            selected_mask[int(idx)] = True
    if len(selected) < budget:
        for idx in order:
            if not selected_mask[int(idx)]:
                selected.append(int(idx))
                selected_mask[int(idx)] = True
                if len(selected) >= budget:
                    break
    return np.sort(np.asarray(selected, dtype=int))


def _maxmin(scores: np.ndarray, bits: np.ndarray, budget: int) -> np.ndarray:
    pool = _candidate_pool(scores, budget)
    bits_pool = bits[pool].astype(np.uint16, copy=False)
    bit_counts = bits_pool.sum(axis=1, dtype=np.uint16)
    selected_pool = [0]
    available = np.ones(len(pool), dtype=bool)
    available[0] = False
    max_sim = _tanimoto_to_all(bits_pool, bit_counts, 0)
    while len(selected_pool) < budget and available.any():
        candidate_scores = scores[pool]
        order = np.lexsort((-candidate_scores, max_sim))
        next_idx = next(int(i) for i in order if available[int(i)])
        selected_pool.append(next_idx)
        available[next_idx] = False
        max_sim = np.maximum(max_sim, _tanimoto_to_all(bits_pool, bit_counts, next_idx))
    return np.sort(pool[np.asarray(selected_pool, dtype=int)])


def _mmr(scores: np.ndarray, bits: np.ndarray, budget: int, *, diversity_weight: float) -> np.ndarray:
    pool = _candidate_pool(scores, budget)
    bits_pool = bits[pool].astype(np.uint16, copy=False)
    bit_counts = bits_pool.sum(axis=1, dtype=np.uint16)
    score_norm = _normalise_scores(scores[pool])
    selected_pool = [int(np.argmax(score_norm))]
    available = np.ones(len(pool), dtype=bool)
    available[selected_pool[0]] = False
    max_sim = _tanimoto_to_all(bits_pool, bit_counts, selected_pool[0])
    relevance_weight = 1.0 - diversity_weight
    while len(selected_pool) < budget and available.any():
        objective = relevance_weight * score_norm - diversity_weight * max_sim
        objective[~available] = -np.inf
        next_idx = int(np.argmax(objective))
        selected_pool.append(next_idx)
        available[next_idx] = False
        max_sim = np.maximum(max_sim, _tanimoto_to_all(bits_pool, bit_counts, next_idx))
    return np.sort(pool[np.asarray(selected_pool, dtype=int)])


def _dpp_greedy(scores: np.ndarray, bits: np.ndarray, budget: int) -> np.ndarray:
    pool = _candidate_pool(scores, budget, min_pool=256, max_pool=1024)
    bits_pool = bits[pool].astype(np.uint16, copy=False)
    bit_counts = bits_pool.sum(axis=1, dtype=np.uint16)
    quality = 0.05 + 0.95 * _normalise_scores(scores[pool])
    n_pool = len(pool)
    max_k = min(budget, n_pool)
    cis = np.zeros((max_k, n_pool), dtype=np.float64)
    residual = quality * quality
    available = np.ones(n_pool, dtype=bool)
    selected_pool: list[int] = []
    for k in range(max_k):
        masked = np.where(available, residual, -np.inf)
        next_idx = int(np.argmax(masked))
        if not np.isfinite(masked[next_idx]) or masked[next_idx] <= 1e-12:
            break
        selected_pool.append(next_idx)
        available[next_idx] = False
        sim_row = _tanimoto_to_all(bits_pool, bit_counts, next_idx)
        kernel_row = quality[next_idx] * quality * sim_row
        if k > 0:
            projection = cis[:k, next_idx] @ cis[:k]
        else:
            projection = 0.0
        update = (kernel_row - projection) / math.sqrt(max(residual[next_idx], 1e-12))
        cis[k] = update
        residual = np.maximum(residual - update * update, 0.0)
    if len(selected_pool) < budget:
        selected_set = set(selected_pool)
        for idx in np.argsort(-scores[pool], kind="mergesort"):
            idx = int(idx)
            if idx not in selected_set:
                selected_pool.append(idx)
                selected_set.add(idx)
                if len(selected_pool) >= budget:
                    break
    return np.sort(pool[np.asarray(selected_pool, dtype=int)])
