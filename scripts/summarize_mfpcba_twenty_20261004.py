#!/usr/bin/env python3
"""Assay-level inference for the frozen 20-task MF-PCBA selector audit."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("chemdep_cal", "score_cap1", "score_only_ranker", "xgb_pairwise_cap1",
           "ecfp_score_xgb_cap1")


def exact_signflip(difference: np.ndarray) -> float:
    observed = abs(float(np.mean(difference)))
    signs = np.array(list(itertools.product((-1., 1.), repeat=len(difference))))
    permuted = abs((signs * difference[None, :]).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-12))


def bootstrap_interval(difference: np.ndarray, seed: int = 20261004) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(50000, len(difference)))
    replicates = difference[indices].mean(axis=1)
    low, high = np.quantile(replicates, [.025, .975])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--campaign", choices=("twenty", "twelve"), default="twenty")
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest = json.loads((directory / "cohort_manifest.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(directory / "cohort_audit.csv")
    eligible = manifest["eligible_tasks"]
    expected_n = 20 if args.campaign == "twenty" else 12
    if len(audit) != expected_n * 5 or len(eligible) < 1:
        raise AssertionError("Eligibility ledger incomplete")
    cells = pd.concat([pd.read_csv(directory / "selectors" / f"seed{seed}_cells.csv")
                       for seed in range(1, 6)], ignore_index=True)
    if (len(cells) != 5 * len(eligible) * len(METHODS)
            or cells.duplicated(["task", "seed", "method"]).any()
            or set(cells.method) != set(METHODS)
            or not cells.selected.eq(50).all()
            or not cells.scaffolds.eq(50).all()):
        raise AssertionError("Incomplete or invalid matched grid")
    task = cells.groupby(["task", "method"]).hits.agg(["mean", "std"]).reset_index()
    wide = task.pivot(index="task", columns="method", values="mean").sort_index()
    if list(wide.index) != sorted(eligible):
        raise AssertionError("Unexpected assay set")
    macro = task.groupby("method").agg(hits=("mean", "mean"), assay_sd=("mean", "std"))
    comparisons = []
    for method in METHODS[1:]:
        difference = wide.chemdep_cal.to_numpy() - wide[method].to_numpy()
        low, high = bootstrap_interval(difference)
        comparisons.append({"comparator": method, "mean_gain": float(np.mean(difference)),
                            "ci_low": low, "ci_high": high,
                            "p_exact": exact_signflip(difference),
                            "wins": int(np.sum(difference > 1e-12)),
                            "losses": int(np.sum(difference < -1e-12)),
                            "ties": int(np.sum(abs(difference) <= 1e-12))})
    comparison = pd.DataFrame(comparisons).sort_values("p_exact")
    comparison["holm_limit"] = .05 / np.arange(len(comparison), 0, -1)
    passed = True
    holm_decisions = []
    for row in comparison.itertuples():
        passed = passed and row.mean_gain > 0 and row.p_exact < row.holm_limit
        holm_decisions.append(passed)
    comparison["holm_pass"] = holm_decisions
    best_competitor = macro.drop(index="chemdep_cal").hits.idxmax()
    best = comparison.set_index("comparator").loc[best_competitor]
    min_assays = 10 if args.campaign == "twenty" else 8
    ci_gate = best.ci_low > 0 if args.campaign == "twenty" else comparison.ci_low.gt(0).all()
    gate = (len(eligible) >= min_assays and best.mean_gain > 0 and ci_gate
            and comparison.holm_pass.all() and best.wins >= int(np.ceil(.70 * len(eligible))))
    task.to_csv(directory / "assay_method_mean_sd.csv", index=False)
    macro.to_csv(directory / "macro_method_mean_sd.csv")
    comparison.to_csv(directory / "paired_assay_inference.csv", index=False)
    verdict = {"eligible_assays": len(eligible), "minimum_assays": min_assays,
               "best_matched_competitor": best_competitor,
               "conditional_selector_sota_gate_pass": bool(gate),
               "note": "This audit does not replicate full published systems or their different endpoint."}
    (directory / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print("ELIGIBILITY")
    print(audit.groupby("task").eligible.all().to_string())
    print("MACRO")
    print(macro.to_string())
    print("PAIRED ASSAY INFERENCE")
    print(comparison.to_string(index=False))
    print("VERDICT", verdict)


if __name__ == "__main__":
    main()
