#!/usr/bin/env python3
"""Target-level summaries for post-confirmation PR revision experiments."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


def exact_signflip(values: np.ndarray) -> float:
    observed = abs(float(np.mean(values)))
    flips = np.array(list(itertools.product((-1., 1.), repeat=len(values))))
    return float(np.mean(np.abs((flips * values).mean(axis=1)) >= observed - 1e-12))


def contrast(frame: pd.DataFrame, group: str, condition, a: str, b: str) -> dict:
    part = frame[condition]
    paired = part.pivot_table(index="target", columns="method", values="hits", aggfunc="mean")
    if len(paired) != 10:
        raise AssertionError((group, len(paired)))
    diff = (paired[a] - paired[b]).to_numpy(float)
    return dict(group=group, contrast=f"{a} - {b}", mean_gain=float(diff.mean()),
                target_sd=float(diff.std(ddof=1)), positive_targets=int((diff > 0).sum()),
                negative_targets=int((diff < 0).sum()), exact_p_descriptive=exact_signflip(diff),
                a_mean=float(paired[a].mean()), b_mean=float(paired[b].mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    paths = sorted(args.raw_dir.glob("seed3550*.csv"))
    if len(paths) != 5:
        raise AssertionError(f"Expected five scorer seeds: {paths}")
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if len(raw) != 29500 or raw["scorer_seed"].nunique() != 5:
        raise AssertionError(raw.shape)
    key = ["target", "scorer_seed", "regime", "rep", "mechanism", "budget", "method"]
    if raw.duplicated(key).any():
        # Original cells also vary by artifact fraction and boost.
        key.extend(["fraction", "boost"])
        if raw.duplicated(key).any():
            raise AssertionError("Duplicate evaluation cells")

    rows = []
    original = raw["regime"].eq("original")
    unseen = raw["regime"].eq("unseen")
    for budget in (20, 40, 50, 75):
        mask = original & raw["budget"].eq(budget)
        rows.append(contrast(raw, "budget", mask, "full", "score_cap1") |
                    {"setting": str(budget)})
    for method in ("score_only", "plus_group_stats", "plus_group_evidence"):
        mask = original & raw["budget"].eq(50)
        rows.append(contrast(raw, "feature_family", mask, "full", method) |
                    {"setting": method})
    for mechanism in ("logit_b5", "logit_b7", "additive_inactive",
                      "blockwide_logit", "heterogeneous_inactive"):
        mask = unseen & raw["mechanism"].eq(mechanism)
        for control in ("score_cap1", "score_only"):
            rows.append(contrast(raw, "unseen_artifact", mask, "full", control) |
                        {"setting": mechanism})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "target_level_contrasts.csv", index=False)
    grouped = raw.groupby(["regime", "mechanism", "budget", "method"], as_index=False).agg(
        hits=("hits", "mean"), selected=("selected", "mean"),
        scaffolds=("scaffolds", "mean"), n_cells=("hits", "size"),
    )
    grouped.to_csv(args.out_dir / "method_means.csv", index=False)
    target = raw.groupby(["target", "regime", "mechanism", "budget", "method"],
                         as_index=False).agg(hits=("hits", "mean"))
    target.to_csv(args.out_dir / "target_means.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
