#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


KEY_METHODS = [
    "raw_top_b",
    "bh",
    "block_bh",
    "weighted_bh",
    "chemdeprc_soft75",
    "chemdeprc_cap1",
    "by",
]


def write_pair(df: pd.DataFrame, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(df.to_markdown(index=False, floatfmt=".4f") + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="data/chembl_block_artifact_potent_a7_i6/latest_summary.csv")
    parser.add_argument("--results", default="data/chembl_block_artifact_potent_a7_i6/latest_results.csv")
    parser.add_argument("--outdir", default="tables")
    parser.add_argument("--q", type=float, default=0.7)
    parser.add_argument("--budget-label", default="100")
    parser.add_argument("--artifact-fraction", type=float, default=0.05)
    parser.add_argument("--logit-boost", type=float, default=4.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    summary = pd.read_csv(root / args.summary)
    results = pd.read_csv(root / args.results)
    mask = (
        (summary["q"] == args.q)
        & (summary["budget_label"].astype(str) == str(args.budget_label))
        & np.isclose(summary["artifact_fraction"], args.artifact_fraction)
        & np.isclose(summary["logit_boost"], args.logit_boost)
        & summary["method"].isin(KEY_METHODS)
    )
    cols = [
        "stress_label",
        "artifact_fraction",
        "logit_boost",
        "q",
        "budget_label",
        "budget",
        "method",
        "selected_count",
        "true_hits",
        "fdp",
        "fdp_exceeds_q",
        "precision",
        "power",
        "unique_blocks",
        "max_block_share",
        "mean_pairwise_tanimoto",
        "artifact_block_count",
        "artifact_inactive_molecules",
        "n_targets",
        "n_target_reps",
    ]
    main = summary[mask][cols].sort_values("method")
    stem = f"chembl_block_artifact_q{args.q:g}_B{args.budget_label}_f{args.artifact_fraction:g}_boost{args.logit_boost:g}"
    write_pair(main, outdir, stem)

    base = summary[summary["method"] == "bh"][
        [
            "stress_label",
            "artifact_fraction",
            "logit_boost",
            "q",
            "budget_label",
            "selected_count",
            "true_hits",
            "fdp",
            "fdp_exceeds_q",
            "precision",
            "unique_blocks",
            "max_block_share",
            "mean_pairwise_tanimoto",
        ]
    ]
    comp = summary[summary["method"].isin(["weighted_bh", "by", "block_bh", "chemdeprc_soft75", "chemdeprc_cap1"])]
    delta = comp.merge(
        base,
        on=["stress_label", "artifact_fraction", "logit_boost", "q", "budget_label"],
        suffixes=("", "_bh"),
    )
    for col in [
        "selected_count",
        "true_hits",
        "fdp",
        "fdp_exceeds_q",
        "precision",
        "unique_blocks",
        "max_block_share",
        "mean_pairwise_tanimoto",
    ]:
        delta[f"delta_{col}"] = delta[col] - delta[f"{col}_bh"]
    delta_cols = [
        "stress_label",
        "q",
        "budget_label",
        "method",
        "delta_selected_count",
        "delta_true_hits",
        "delta_fdp",
        "delta_fdp_exceeds_q",
        "delta_precision",
        "delta_unique_blocks",
        "delta_max_block_share",
        "delta_mean_pairwise_tanimoto",
    ]
    delta_main = delta[
        (delta["q"] == args.q)
        & (delta["budget_label"].astype(str) == str(args.budget_label))
        & np.isclose(delta["artifact_fraction"], args.artifact_fraction)
        & np.isclose(delta["logit_boost"], args.logit_boost)
    ][delta_cols].sort_values("method")
    write_pair(delta_main, outdir, f"{stem}_delta_vs_bh")

    sweep = delta[
        (delta["q"] == args.q)
        & (delta["budget_label"].astype(str) == str(args.budget_label))
        & (delta["method"].isin(["weighted_bh", "chemdeprc_soft75", "chemdeprc_cap1"]))
    ][
        [
            "stress_label",
            "artifact_fraction",
            "logit_boost",
            "method",
            "delta_true_hits",
            "delta_fdp",
            "delta_fdp_exceeds_q",
            "delta_unique_blocks",
            "delta_max_block_share",
            "delta_mean_pairwise_tanimoto",
        ]
    ].sort_values(["artifact_fraction", "logit_boost", "method"])
    write_pair(sweep, outdir, f"chembl_block_artifact_sweep_q{args.q:g}_B{args.budget_label}")

    raw_mask = (
        (results["q"] == args.q)
        & (results["budget_label"].astype(str) == str(args.budget_label))
        & np.isclose(results["artifact_fraction"], args.artifact_fraction)
        & np.isclose(results["logit_boost"], args.logit_boost)
    )
    raw = results[raw_mask]
    tests = []
    for method in ["weighted_bh", "block_bh", "chemdeprc_soft75", "chemdeprc_cap1", "by"]:
        paired = raw[raw["method"] == method].merge(
            raw[raw["method"] == "bh"],
            on=["target_chembl_id", "rep", "stress_label", "q", "budget_label"],
            suffixes=("", "_bh"),
        )
        for metric in [
            "true_hits",
            "fdp",
            "fdp_exceeds_q",
            "unique_blocks",
            "max_block_share",
            "mean_pairwise_tanimoto",
        ]:
            diff = paired[metric] - paired[f"{metric}_bh"]
            if len(diff) == 0 or np.allclose(diff, 0):
                stat = np.nan
                pvalue = np.nan
            else:
                stat, pvalue = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            tests.append(
                {
                    "stress_label": paired["stress_label"].iloc[0] if len(paired) else "",
                    "q": args.q,
                    "budget_label": args.budget_label,
                    "method": method,
                    "metric": metric,
                    "mean_delta_vs_bh": float(diff.mean()) if len(diff) else np.nan,
                    "median_delta_vs_bh": float(diff.median()) if len(diff) else np.nan,
                    "wilcoxon_stat": stat,
                    "wilcoxon_p": pvalue,
                    "n_pairs": int(len(diff)),
                }
            )
    tests_df = pd.DataFrame(tests)
    write_pair(tests_df, outdir, f"{stem}_paired_tests")
    print(f"Wrote block-artifact tables to {outdir}")


if __name__ == "__main__":
    main()
