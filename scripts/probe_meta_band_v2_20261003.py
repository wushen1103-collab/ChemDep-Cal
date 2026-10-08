#!/usr/bin/env python3
"""V2 calibration ranker with explicit block-evidence features."""

from __future__ import annotations

import numpy as np

import probe_meta_band_20261003 as base
from train_artifact_meta_ranker_20261003 import features as original_features


def evidence_features(scores: np.ndarray, blocks: np.ndarray, null_scores: np.ndarray) -> np.ndarray:
    from chemdeprc.selection import _members_by_block, _simes_pvalue
    from run_chembl_selection import conformal_right_tail_pvalues

    primary = original_features(scores, blocks, null_scores)
    pvalues = conformal_right_tail_pvalues(scores, null_scores)
    _, groups = _members_by_block(blocks)
    bonf = np.empty(len(scores), dtype=np.float32)
    simes = np.empty(len(scores), dtype=np.float32)
    score_std = np.empty(len(scores), dtype=np.float32)
    score_gap = np.empty(len(scores), dtype=np.float32)
    null_hi = float(np.quantile(null_scores, .95))
    high_fraction = np.empty(len(scores), dtype=np.float32)
    for members in groups:
        values = scores[members]
        ordered = np.sort(values)
        bonf[members] = min(1., len(members) * float(np.min(pvalues[members])))
        simes[members] = _simes_pvalue(pvalues[members])
        score_std[members] = float(np.std(values))
        score_gap[members] = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 0.
        high_fraction[members] = float(np.mean(values >= null_hi))
    extra = np.column_stack([
        -np.log10(np.maximum(bonf, 1e-6)),
        -np.log10(np.maximum(simes, 1e-6)),
        np.log(np.maximum(bonf, 1e-6)) - np.log(np.maximum(simes, 1e-6)),
        score_std,
        score_gap,
        high_fraction,
    ]).astype(np.float32)
    return np.column_stack([primary, extra])


base.features = evidence_features


if __name__ == "__main__":
    base.main()
