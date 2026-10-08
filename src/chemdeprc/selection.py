from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SelectionResult:
    method: str
    selected: np.ndarray
    estimated_fdp: float
    certificate: str


def select(
    method: str,
    *,
    scores: np.ndarray,
    pvalues: np.ndarray,
    blocks: np.ndarray,
    q: float,
    budget: int,
    block_size_cap: int = 2,
) -> SelectionResult:
    method = method.lower()
    if method == "raw_top_b":
        selected = _top_by_score(scores, budget)
        return SelectionResult(method, selected, np.nan, "no_risk_control")
    if method in {"bh", "no_dependency_block"}:
        selected = _bh(pvalues, q, budget, scores=scores)
        return SelectionResult(method, selected, q, "individual_bh")
    if method == "by":
        harmonic = np.sum(1.0 / np.arange(1, len(pvalues) + 1))
        selected = _bh(pvalues, q / harmonic, budget, scores=scores)
        return SelectionResult(method, selected, q, "individual_by")
    if method == "weighted_bh":
        weights = _inverse_block_size_weights(blocks)
        selected = _weighted_bh(pvalues, weights, q, budget, scores=scores)
        return SelectionResult(method, selected, q, "inverse_block_size_weighted_bh")
    if method == "e_bh":
        evalues = np.minimum(1.0 / np.maximum(pvalues, 1e-12), len(pvalues))
        selected = _e_bh(evalues, q, budget, scores=scores)
        return SelectionResult(method, selected, q, "individual_e_bh_from_conformal_p")
    if method == "block_bh":
        selected = _block_bh(pvalues, blocks, q, budget, cap=None, round_robin=False, scores=scores)
        return SelectionResult(method, selected, q, "simes_block_bh")
    if method.startswith("block_by_cap"):
        method_cap = int(method.replace("block_by_cap", ""))
        selected = _block_bh(
            pvalues,
            blocks,
            q,
            budget,
            cap=method_cap,
            round_robin=True,
            scores=scores,
            block_correction="by",
        )
        return SelectionResult(method, selected, q, f"simes_block_by_cap_{method_cap}")
    if method.startswith("minp_block_bh_cap"):
        method_cap = int(method.replace("minp_block_bh_cap", ""))
        selected = _block_bh(
            pvalues,
            blocks,
            q,
            budget,
            cap=method_cap,
            round_robin=True,
            scores=scores,
            block_p_method="bonferroni_min",
        )
        return SelectionResult(method, selected, q, f"bonferroni_minp_block_bh_cap_{method_cap}")
    if method.startswith("hier_bh_cap"):
        method_cap = int(method.replace("hier_bh_cap", ""))
        selected = _hierarchical_bh(pvalues, blocks, q, budget, cap=method_cap, scores=scores)
        return SelectionResult(method, selected, q, f"simes_block_bh_within_block_bh_cap_{method_cap}")
    if method.startswith("score_cap"):
        method_cap = int(method.replace("score_cap", ""))
        selected = _round_robin_candidates(
            np.arange(len(scores)),
            pvalues=pvalues,
            scores=scores,
            blocks=blocks,
            budget=budget,
            cap=method_cap,
            max_blocks=None,
            rank_by="score",
        )
        return SelectionResult(method, selected, np.nan, f"score_only_posthoc_block_cap_{method_cap}")
    if method.startswith("bh_cap"):
        method_cap = int(method.replace("bh_cap", ""))
        bh_candidates = _bh(pvalues, q, budget=None, scores=scores)
        selected = _round_robin_candidates(
            bh_candidates,
            pvalues=pvalues,
            scores=scores,
            blocks=blocks,
            budget=budget,
            cap=method_cap,
            max_blocks=None,
            rank_by="pvalue",
        )
        return SelectionResult(method, selected, q, f"individual_bh_posthoc_block_cap_{method_cap}")
    if method.startswith("bh_soft"):
        suffix = method.replace("bh_soft", "")
        block_fraction = int(suffix) / 100.0
        max_blocks = max(1, int(math.ceil(budget * block_fraction)))
        bh_candidates = _bh(pvalues, q, budget=None, scores=scores)
        selected = _round_robin_candidates(
            bh_candidates,
            pvalues=pvalues,
            scores=scores,
            blocks=blocks,
            budget=budget,
            cap=None,
            max_blocks=max_blocks,
            rank_by="pvalue",
        )
        return SelectionResult(method, selected, q, f"individual_bh_posthoc_soft_{suffix}pct_blocks")
    if method.startswith("chemdeprc_scoresoft"):
        suffix = method.replace("chemdeprc_scoresoft", "")
        block_fraction = int(suffix) / 100.0
        max_blocks = max(1, int(math.ceil(budget * block_fraction)))
        selected = _block_bh(
            pvalues,
            blocks,
            q,
            budget,
            cap=None,
            round_robin=True,
            max_blocks=max_blocks,
            scores=scores,
            block_rank_by="score",
            member_rank_by="score",
        )
        return SelectionResult(
            method, selected, q, f"chemdeprc_simes_block_bh_score_ranked_soft_{suffix}pct_blocks"
        )
    if method.startswith("chemdeprc_scorecap"):
        method_cap = int(method.replace("chemdeprc_scorecap", ""))
        selected = _block_bh(
            pvalues,
            blocks,
            q,
            budget,
            cap=method_cap,
            round_robin=True,
            scores=scores,
            block_rank_by="score",
            member_rank_by="score",
        )
        return SelectionResult(method, selected, q, f"chemdeprc_simes_block_bh_score_ranked_cap_{method_cap}")
    if method.startswith("chemdeprc_soft"):
        suffix = method.replace("chemdeprc_soft", "")
        block_fraction = int(suffix) / 100.0
        max_blocks = max(1, int(math.ceil(budget * block_fraction)))
        selected = _block_bh(
            pvalues, blocks, q, budget, cap=None, round_robin=True, max_blocks=max_blocks, scores=scores
        )
        return SelectionResult(method, selected, q, f"chemdeprc_simes_block_bh_soft_{suffix}pct_blocks")
    if method.startswith("chemdeprc_cap"):
        method_cap = int(method.replace("chemdeprc_cap", ""))
        selected = _block_bh(pvalues, blocks, q, budget, cap=method_cap, round_robin=True, scores=scores)
        return SelectionResult(method, selected, q, f"chemdeprc_simes_block_bh_cap_{method_cap}")
    if method == "chemdeprc":
        selected = _block_bh(pvalues, blocks, q, budget, cap=block_size_cap, round_robin=True, scores=scores)
        return SelectionResult(method, selected, q, f"chemdeprc_simes_block_bh_cap_{block_size_cap}")
    raise ValueError(f"Unknown method {method!r}")


def _top_by_score(scores: np.ndarray, budget: int) -> np.ndarray:
    if budget <= 0:
        return np.array([], dtype=int)
    return np.argsort(-scores, kind="mergesort")[:budget].astype(int)


def _bh(pvalues: np.ndarray, q: float, budget: int | None = None, scores: np.ndarray | None = None) -> np.ndarray:
    m = len(pvalues)
    order = _rank_pvalues(pvalues, scores)
    ranked = pvalues[order]
    thresholds = q * np.arange(1, m + 1) / m
    ok = np.flatnonzero(ranked <= thresholds)
    if len(ok) == 0:
        return np.array([], dtype=int)
    k = int(ok[-1]) + 1
    selected = order[:k]
    if budget is not None and len(selected) > budget:
        selected = selected[_rank_pvalues(pvalues[selected], None if scores is None else scores[selected])[:budget]]
    return np.sort(selected).astype(int)


def _weighted_bh(
    pvalues: np.ndarray,
    weights: np.ndarray,
    q: float,
    budget: int | None = None,
    scores: np.ndarray | None = None,
) -> np.ndarray:
    adjusted = pvalues / np.maximum(weights, 1e-12)
    selected = _bh(adjusted, q, budget=None, scores=scores)
    if budget is not None and len(selected) > budget:
        selected = selected[_rank_pvalues(adjusted[selected], None if scores is None else scores[selected])[:budget]]
    return np.sort(selected).astype(int)


def _e_bh(evalues: np.ndarray, q: float, budget: int | None = None, scores: np.ndarray | None = None) -> np.ndarray:
    m = len(evalues)
    order = _rank_evalues(evalues, scores)
    ranked = evalues[order]
    thresholds = m / (q * np.arange(1, m + 1))
    ok = np.flatnonzero(ranked >= thresholds)
    if len(ok) == 0:
        return np.array([], dtype=int)
    k = int(ok[-1]) + 1
    selected = order[:k]
    if budget is not None and len(selected) > budget:
        selected = selected[_rank_evalues(evalues[selected], None if scores is None else scores[selected])[:budget]]
    return np.sort(selected).astype(int)


def _block_bh(
    pvalues: np.ndarray,
    blocks: np.ndarray,
    q: float,
    budget: int,
    *,
    cap: int | None,
    round_robin: bool,
    max_blocks: int | None = None,
    scores: np.ndarray | None = None,
    block_rank_by: str = "pvalue",
    member_rank_by: str = "pvalue",
    block_p_method: str = "simes",
    block_correction: str = "bh",
) -> np.ndarray:
    unique_blocks, members_by_block = _members_by_block(blocks)
    block_p = np.empty(len(unique_blocks), dtype=float)
    block_score = np.empty(len(unique_blocks), dtype=float) if scores is not None else None
    for pos, members in enumerate(members_by_block):
        block_p[pos] = _block_pvalue(pvalues[members], method=block_p_method)
        if block_score is not None:
            block_score[pos] = float(np.max(scores[members]))

    if block_correction == "bh":
        selected_block_pos = _bh(block_p, q, budget=None, scores=block_score)
    elif block_correction == "by":
        harmonic = np.sum(1.0 / np.arange(1, len(block_p) + 1))
        selected_block_pos = _bh(block_p, q / harmonic, budget=None, scores=block_score)
    else:
        raise ValueError(f"Unknown block_correction {block_correction!r}")
    if len(selected_block_pos) == 0:
        return np.array([], dtype=int)

    selected_members: list[int] = []
    ordered_block_pos = _rank_blocks(
        selected_block_pos,
        block_p=block_p,
        block_score=block_score,
        rank_by=block_rank_by,
    )
    if max_blocks is not None:
        ordered_block_pos = ordered_block_pos[: max(1, max_blocks)]
    if round_robin:
        ranked_members = []
        for pos in ordered_block_pos:
            members = members_by_block[int(pos)]
            ranked = _rank_members(members, pvalues=pvalues, scores=scores, rank_by=member_rank_by)
            if cap is not None:
                ranked = ranked[:cap]
            ranked_members.append(ranked)
        depth = 0
        while len(selected_members) < budget:
            progressed = False
            for ranked in ranked_members:
                if depth < len(ranked):
                    selected_members.append(int(ranked[depth]))
                    progressed = True
                    if len(selected_members) >= budget:
                        break
            if not progressed:
                break
            depth += 1
    else:
        candidates = np.concatenate([members_by_block[int(pos)] for pos in ordered_block_pos])
        candidates = _rank_members(candidates, pvalues=pvalues, scores=scores, rank_by=member_rank_by)
        selected_members = [int(i) for i in candidates[:budget]]

    return np.sort(np.asarray(selected_members, dtype=int))


def _hierarchical_bh(
    pvalues: np.ndarray,
    blocks: np.ndarray,
    q: float,
    budget: int,
    *,
    cap: int | None,
    scores: np.ndarray | None,
) -> np.ndarray:
    unique_blocks, members_by_block = _members_by_block(blocks)
    block_p = np.empty(len(unique_blocks), dtype=float)
    block_score = np.empty(len(unique_blocks), dtype=float) if scores is not None else None
    for pos, members in enumerate(members_by_block):
        block_p[pos] = _simes_pvalue(pvalues[members])
        if block_score is not None:
            block_score[pos] = float(np.max(scores[members]))

    selected_block_pos = _bh(block_p, q, budget=None, scores=block_score)
    if len(selected_block_pos) == 0:
        return np.array([], dtype=int)

    ordered_block_pos = _rank_blocks(
        selected_block_pos,
        block_p=block_p,
        block_score=block_score,
        rank_by="pvalue",
    )
    ranked_members: list[np.ndarray] = []
    for pos in ordered_block_pos:
        members = members_by_block[int(pos)]
        member_scores = None if scores is None else scores[members]
        rel_selected = _bh(pvalues[members], q, budget=None, scores=member_scores)
        if len(rel_selected) == 0:
            continue
        selected_members = members[rel_selected]
        ranked = _rank_members(selected_members, pvalues=pvalues, scores=scores, rank_by="pvalue")
        if cap is not None:
            ranked = ranked[:cap]
        ranked_members.append(ranked)

    selected: list[int] = []
    depth = 0
    while len(selected) < budget:
        progressed = False
        for ranked in ranked_members:
            if depth < len(ranked):
                selected.append(int(ranked[depth]))
                progressed = True
                if len(selected) >= budget:
                    break
        if not progressed:
            break
        depth += 1
    return np.sort(np.asarray(selected, dtype=int))


def _rank_blocks(
    block_positions: np.ndarray,
    *,
    block_p: np.ndarray,
    block_score: np.ndarray | None,
    rank_by: str,
) -> np.ndarray:
    if rank_by == "pvalue":
        selected_block_scores = None if block_score is None else block_score[block_positions]
        order = _rank_pvalues(block_p[block_positions], selected_block_scores)
    elif rank_by == "score":
        if block_score is None:
            raise ValueError("score-ranked block selection requires scores")
        order = np.argsort(-block_score[block_positions], kind="mergesort")
    else:
        raise ValueError(f"Unknown block rank_by {rank_by!r}")
    return block_positions[order]


def _rank_members(
    members: np.ndarray,
    *,
    pvalues: np.ndarray,
    scores: np.ndarray | None,
    rank_by: str,
) -> np.ndarray:
    if rank_by == "pvalue":
        member_scores = None if scores is None else scores[members]
        order = _rank_pvalues(pvalues[members], member_scores)
    elif rank_by == "score":
        if scores is None:
            raise ValueError("score-ranked member selection requires scores")
        order = np.argsort(-scores[members], kind="mergesort")
    else:
        raise ValueError(f"Unknown member rank_by {rank_by!r}")
    return members[order]


def _round_robin_candidates(
    candidates: np.ndarray,
    *,
    pvalues: np.ndarray,
    scores: np.ndarray,
    blocks: np.ndarray,
    budget: int,
    cap: int | None,
    max_blocks: int | None,
    rank_by: str,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=int)
    if budget <= 0 or len(candidates) == 0:
        return np.array([], dtype=int)

    candidate_blocks = blocks[candidates]
    unique_blocks, inverse = np.unique(candidate_blocks, return_inverse=True)
    order = np.argsort(inverse, kind="mergesort")
    counts = np.bincount(inverse, minlength=len(unique_blocks))
    boundaries = np.concatenate(([0], np.cumsum(counts)))
    ranked_members: list[np.ndarray] = []
    block_order_values = np.empty(len(unique_blocks), dtype=float)
    block_order_scores = np.empty(len(unique_blocks), dtype=float)
    for pos in range(len(unique_blocks)):
        members = candidates[order[boundaries[pos] : boundaries[pos + 1]]]
        block_order_scores[pos] = float(np.max(scores[members]))
        if rank_by == "score":
            ranked = members[np.argsort(-scores[members], kind="mergesort")]
            block_order_values[pos] = -block_order_scores[pos]
        elif rank_by == "pvalue":
            ranked = members[_rank_pvalues(pvalues[members], scores[members])]
            block_order_values[pos] = float(np.min(pvalues[members]))
        else:
            raise ValueError(f"Unknown rank_by {rank_by!r}")
        if cap is not None:
            ranked = ranked[:cap]
        ranked_members.append(ranked)

    block_order = np.lexsort((-block_order_scores, block_order_values))
    if max_blocks is not None:
        block_order = block_order[: max(1, max_blocks)]
    ordered_ranked_members = [ranked_members[int(pos)] for pos in block_order]

    selected: list[int] = []
    depth = 0
    while len(selected) < budget:
        progressed = False
        for ranked in ordered_ranked_members:
            if depth < len(ranked):
                selected.append(int(ranked[depth]))
                progressed = True
                if len(selected) >= budget:
                    break
        if not progressed:
            break
        depth += 1
    return np.sort(np.asarray(selected, dtype=int))


def _rank_pvalues(pvalues: np.ndarray, scores: np.ndarray | None = None) -> np.ndarray:
    if scores is None:
        return np.argsort(pvalues, kind="mergesort")
    return np.lexsort((-scores, pvalues))


def _rank_evalues(evalues: np.ndarray, scores: np.ndarray | None = None) -> np.ndarray:
    if scores is None:
        return np.argsort(-evalues, kind="mergesort")
    return np.lexsort((-scores, -evalues))


def _simes_pvalue(block_pvalues: np.ndarray) -> float:
    ordered = np.sort(block_pvalues)
    n = len(ordered)
    if n == 0:
        return 1.0
    return float(min(1.0, np.min(n * ordered / np.arange(1, n + 1))))


def _block_pvalue(block_pvalues: np.ndarray, *, method: str) -> float:
    if method == "simes":
        return _simes_pvalue(block_pvalues)
    if method == "bonferroni_min":
        if len(block_pvalues) == 0:
            return 1.0
        return float(min(1.0, len(block_pvalues) * np.min(block_pvalues)))
    raise ValueError(f"Unknown block_p_method {method!r}")


def _members_by_block(blocks: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    unique_blocks, inverse = np.unique(blocks, return_inverse=True)
    order = np.argsort(inverse, kind="mergesort")
    counts = np.bincount(inverse, minlength=len(unique_blocks))
    boundaries = np.concatenate(([0], np.cumsum(counts)))
    members = [order[boundaries[pos] : boundaries[pos + 1]] for pos in range(len(unique_blocks))]
    return unique_blocks, members


def _inverse_block_size_weights(blocks: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(blocks, return_inverse=True, return_counts=True)
    raw = 1.0 / counts[inverse]
    return raw / raw.mean()
