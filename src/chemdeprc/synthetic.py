from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Campaign:
    scenario: str
    scores: np.ndarray
    labels: np.ndarray
    blocks: np.ndarray
    calib_scores: np.ndarray
    calib_labels: np.ndarray
    true_blocks: np.ndarray
    active_rate: float
    within_similarity: float
    between_similarity: float


SCENARIOS = ("independent", "weak_corr", "strong_corr", "block_label_noise")


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_campaign(
    *,
    scenario: str,
    seed: int,
    n_test: int = 3000,
    n_calib: int = 1500,
    active_rate: float = 0.04,
    signal_strength: float = 2.1,
) -> Campaign:
    """Generate one target-level screening campaign.

    The generator intentionally separates label correlation from score-error
    correlation. The latter models analog/scaffold groups sharing model mistakes,
    which is the failure mode ChemDep-RC is designed to expose.
    """

    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; expected one of {SCENARIOS}")

    rng = np.random.default_rng(seed)
    if scenario == "independent":
        block_sizes = np.ones(n_test, dtype=int)
        block_activity_sd = 0.05
        score_bias_sd = 0.05
        within_similarity = 0.2
    elif scenario == "weak_corr":
        block_sizes = _sample_block_sizes(rng, n_test, mean_size=6, max_size=20)
        block_activity_sd = 0.65
        score_bias_sd = 0.55
        within_similarity = 0.66
    elif scenario == "strong_corr":
        block_sizes = _sample_block_sizes(rng, n_test, mean_size=10, max_size=40)
        block_activity_sd = 1.05
        score_bias_sd = 1.05
        within_similarity = 0.78
    else:
        block_sizes = _sample_block_sizes(rng, n_test, mean_size=10, max_size=45)
        block_activity_sd = 0.9
        score_bias_sd = 0.8
        within_similarity = 0.8

    true_blocks = np.repeat(np.arange(len(block_sizes)), block_sizes)[:n_test]
    n_blocks = int(true_blocks.max()) + 1
    block_latent = rng.normal(0.0, block_activity_sd, size=n_blocks)
    block_prob = _expit(_logit(active_rate) + block_latent)
    labels = rng.binomial(1, block_prob[true_blocks]).astype(int)

    block_score_bias = rng.normal(0.0, score_bias_sd, size=n_blocks)
    if scenario == "block_label_noise":
        hot_count = max(1, n_blocks // 25)
        largest_blocks = np.argsort(block_sizes)[-max(hot_count * 3, hot_count):]
        hot_blocks = rng.choice(largest_blocks, size=hot_count, replace=False)
        block_score_bias[hot_blocks] += rng.normal(1.8, 0.2, size=len(hot_blocks))
        # These blocks mimic analog series that look attractive to the scorer
        # while not necessarily being enriched in true actives.
        labels[np.isin(true_blocks, hot_blocks)] = rng.binomial(
            1, active_rate * 0.4, size=np.isin(true_blocks, hot_blocks).sum()
        )

    scores = (
        signal_strength * labels
        + block_score_bias[true_blocks]
        + rng.normal(0.0, 1.0, size=n_test)
    )

    calib_blocks = rng.integers(0, max(n_blocks, 2), size=n_calib)
    calib_block_latent = rng.normal(0.0, block_activity_sd, size=max(n_blocks, 2))
    calib_prob = _expit(_logit(active_rate) + calib_block_latent[calib_blocks])
    calib_labels = rng.binomial(1, calib_prob).astype(int)
    calib_scores = (
        signal_strength * calib_labels
        + rng.normal(0.0, score_bias_sd, size=max(n_blocks, 2))[calib_blocks]
        + rng.normal(0.0, 1.0, size=n_calib)
    )

    return Campaign(
        scenario=scenario,
        scores=scores.astype(float),
        labels=labels.astype(int),
        blocks=true_blocks.astype(int),
        calib_scores=calib_scores.astype(float),
        calib_labels=calib_labels.astype(int),
        true_blocks=true_blocks.astype(int),
        active_rate=float(labels.mean()),
        within_similarity=within_similarity,
        between_similarity=0.12,
    )


def _sample_block_sizes(
    rng: np.random.Generator,
    n_items: int,
    *,
    mean_size: int,
    max_size: int,
) -> np.ndarray:
    sizes = []
    total = 0
    while total < n_items:
        size = int(rng.negative_binomial(n=2, p=2 / (2 + mean_size))) + 1
        size = max(1, min(size, max_size))
        sizes.append(size)
        total += size
    sizes[-1] -= total - n_items
    return np.asarray([s for s in sizes if s > 0], dtype=int)


def null_conformal_pvalues(scores: np.ndarray, calib_scores: np.ndarray, calib_labels: np.ndarray) -> np.ndarray:
    """One-sided split conformal p-values for the inactive/null class."""

    null_scores = np.asarray(calib_scores)[np.asarray(calib_labels) == 0]
    if len(null_scores) == 0:
        null_scores = np.asarray(calib_scores)
    sorted_null = np.sort(null_scores)
    greater_equal = len(sorted_null) - np.searchsorted(sorted_null, scores, side="left")
    return (1.0 + greater_equal) / (len(sorted_null) + 1.0)


def random_blocks_like(blocks: np.ndarray, seed: int) -> np.ndarray:
    """Build a label-free placebo block assignment with the same block sizes."""

    rng = np.random.default_rng(seed)
    unique, counts = np.unique(blocks, return_counts=True)
    shuffled = np.repeat(np.arange(len(unique)), counts)
    rng.shuffle(shuffled)
    return shuffled.astype(int)
