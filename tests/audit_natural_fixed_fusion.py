#!/usr/bin/env python3
"""Check natural-fusion cells, frozen replays, and reported summaries."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "source_tables"
CURRENT = ROOT / "pr_natural_fixed_fusion_20261007"
REFERENCE = ROOT / "pr_natural_modern_selectors_20261004" / "pooled20.csv"
METHODS = {"score_cap1", "chemdep_cal", "multi_fp_rf", "fixed_75_25_fusion"}


def main() -> None:
    cells = pd.read_csv(CURRENT / "cells.csv")
    old = pd.read_csv(REFERENCE)
    summary = pd.read_csv(CURRENT / "dataset_mean_seed_sd.csv")
    key = ["dataset", "scorer_seed", "method"]
    assert len(cells) == 80 and not cells.duplicated(key).any()
    assert cells.dataset.nunique() == 4 and set(cells.method) == METHODS
    assert cells.groupby(["dataset", "method"]).size().eq(5).all()
    assert cells.selected.eq(50).all() and cells.scaffolds.eq(50).all()
    assert cells.testpool_size.ge(50).all()
    archived = old.loc[old.method.isin({"score_cap1", "chemdep_cal"}), key + ["hits"]]
    matched = cells.merge(archived, on=key, how="inner", validate="one_to_one",
                          suffixes=("", "_old"))
    assert len(matched) == 40 and matched.hits.eq(matched.hits_old).all()
    actual = cells.groupby(["dataset", "method"]).hits.agg(["mean", "std"]).reset_index()
    checked = summary.merge(actual, on=["dataset", "method"],
                            how="inner", validate="one_to_one", suffixes=("_saved", ""))
    assert len(checked) == 16
    assert np.allclose(checked.mean_saved, checked["mean"])
    assert np.allclose(checked.std_saved, checked["std"])
    forest = actual.loc[actual.method.eq("multi_fp_rf"), ["dataset", "mean"]]
    fusion = actual.loc[actual.method.eq("fixed_75_25_fusion"), ["dataset", "mean"]]
    paired = fusion.merge(forest, on="dataset", suffixes=("_fusion", "_forest"))
    assert (paired.mean_fusion > paired.mean_forest).sum() == 2
    assert (paired.mean_fusion < paired.mean_forest).sum() == 2
    print("PASS: 80 capacity-matched cells, 40 archived replay identities, 16 summaries")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
