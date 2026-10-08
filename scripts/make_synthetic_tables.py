#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_METHODS = [
    "raw_top_b",
    "bh",
    "weighted_bh",
    "by",
    "chemdeprc_cap1",
    "chemdeprc_cap2",
    "chemdeprc_cap5",
]


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".md":
        path.write_text(df.to_markdown(index=False, floatfmt=".4f") + "\n", encoding="utf-8")
    else:
        raise ValueError(f"Unsupported table suffix: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv")
    parser.add_argument("--outdir", default="tables")
    parser.add_argument("--q", type=float, default=0.7)
    parser.add_argument("--budget-label", default="100")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary_csv)
    outdir = Path(args.outdir)
    metric_cols = [
        "scenario",
        "q",
        "budget_label",
        "budget",
        "method",
        "fdp",
        "fdp_exceeds_q",
        "true_hits",
        "selected_count",
        "precision",
        "power",
        "unique_blocks",
        "max_block_share",
        "mean_pairwise_tanimoto",
        "empty",
    ]

    main = summary[
        (summary["q"] == args.q)
        & (summary["budget_label"].astype(str) == str(args.budget_label))
        & (summary["method"].isin(KEY_METHODS))
    ][metric_cols].sort_values(["scenario", "method"])
    write_table(main, outdir / f"synthetic_main_q{args.q:g}_B{args.budget_label}.csv")
    write_table(main, outdir / f"synthetic_main_q{args.q:g}_B{args.budget_label}.md")

    base = summary[summary["method"] == "bh"][
        [
            "scenario",
            "q",
            "budget_label",
            "budget",
            "fdp",
            "fdp_exceeds_q",
            "true_hits",
            "precision",
            "unique_blocks",
            "max_block_share",
            "mean_pairwise_tanimoto",
        ]
    ]
    comp = summary[summary["method"].isin(["chemdeprc_cap1", "chemdeprc_cap2", "weighted_bh", "by"])]
    delta = comp.merge(base, on=["scenario", "q", "budget_label", "budget"], suffixes=("", "_bh"))
    for col in [
        "fdp",
        "fdp_exceeds_q",
        "true_hits",
        "precision",
        "unique_blocks",
        "max_block_share",
        "mean_pairwise_tanimoto",
    ]:
        delta[f"delta_{col}"] = delta[col] - delta[f"{col}_bh"]

    delta_cols = [
        "scenario",
        "q",
        "budget_label",
        "budget",
        "method",
        "delta_fdp",
        "delta_fdp_exceeds_q",
        "delta_true_hits",
        "delta_precision",
        "delta_unique_blocks",
        "delta_max_block_share",
        "delta_mean_pairwise_tanimoto",
    ]
    block_noise_delta = delta[
        (delta["scenario"] == "block_label_noise")
        & (delta["q"] == args.q)
        & (delta["budget_label"].astype(str).isin(["50", "100", "1pct"]))
    ][delta_cols].sort_values(["budget", "method"])
    write_table(block_noise_delta, outdir / f"synthetic_block_noise_delta_q{args.q:g}.csv")
    write_table(block_noise_delta, outdir / f"synthetic_block_noise_delta_q{args.q:g}.md")

    independent_delta = delta[
        (delta["scenario"] == "independent")
        & (delta["q"] == 0.8)
        & (delta["budget_label"].astype(str) == "100")
        & (delta["method"].isin(["chemdeprc_cap1", "chemdeprc_cap2", "chemdeprc_cap5", "weighted_bh"]))
    ][delta_cols].sort_values(["method"])
    write_table(independent_delta, outdir / "synthetic_independent_degeneracy_q0.8_B100.csv")
    write_table(independent_delta, outdir / "synthetic_independent_degeneracy_q0.8_B100.md")

    generic_tradeoff = delta[
        (delta["scenario"].isin(["weak_corr", "strong_corr"]))
        & (delta["q"] == 0.8)
        & (delta["budget_label"].astype(str) == "100")
        & (delta["method"].isin(["chemdeprc_cap1", "weighted_bh"]))
    ][delta_cols].sort_values(["scenario", "method"])
    write_table(generic_tradeoff, outdir / "synthetic_generic_corr_tradeoff_q0.8_B100.csv")
    write_table(generic_tradeoff, outdir / "synthetic_generic_corr_tradeoff_q0.8_B100.md")

    print(f"Wrote synthetic tables to {outdir}")


if __name__ == "__main__":
    main()

