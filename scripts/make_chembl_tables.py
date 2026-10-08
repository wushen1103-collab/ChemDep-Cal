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
    "block_bh",
    "chemdeprc_soft25",
    "chemdeprc_soft50",
    "chemdeprc_soft75",
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


def write_pair(df: pd.DataFrame, outdir: Path, stem: str) -> None:
    write_table(df, outdir / f"{stem}.csv")
    write_table(df, outdir / f"{stem}.md")


def manifest_table(path: Path, outdir: Path, stem: str) -> None:
    manifest = pd.read_csv(path)
    cols = [
        "target_chembl_id",
        "pref_name",
        "n_panel",
        "n_active",
        "n_inactive",
        "active_rate",
        "unique_scaffolds",
        "calibration_inactive",
        "testpool_active",
        "testpool_inactive",
        "passes_smoke_threshold",
    ]
    table = manifest[cols].sort_values("target_chembl_id")
    write_pair(table, outdir, stem)


def scorer_table(path: Path, outdir: Path, stem: str) -> None:
    metrics = pd.read_csv(path)
    cols = [
        "target_chembl_id",
        "pref_name",
        "n_train",
        "n_calibration",
        "n_testpool",
        "calibration_inactive",
        "testpool_inactive",
        "auroc_testpool",
        "auprc_testpool",
    ]
    table = metrics[cols].sort_values("target_chembl_id")
    write_pair(table, outdir, stem)


def selection_tables(path: Path, outdir: Path, q: float, budget_label: str, stem_prefix: str) -> None:
    summary = pd.read_csv(path)
    metric_cols = [
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
        "empty",
        "n_calib_null",
        "pvalue_floor",
        "n_targets",
    ]
    main = summary[
        (summary["q"] == q)
        & (summary["budget_label"].astype(str) == str(budget_label))
        & (summary["method"].isin(KEY_METHODS))
    ][metric_cols].sort_values("method")
    write_pair(main, outdir, f"{stem_prefix}_q{q:g}_B{budget_label}")

    base = summary[summary["method"] == "bh"][
        [
            "q",
            "budget_label",
            "budget",
            "selected_count",
            "true_hits",
            "fdp",
            "precision",
            "unique_blocks",
            "max_block_share",
            "mean_pairwise_tanimoto",
            "empty",
        ]
    ]
    comp = summary[
        summary["method"].isin(
            [
                "weighted_bh",
                "by",
                "block_bh",
                "chemdeprc_soft25",
                "chemdeprc_soft50",
                "chemdeprc_soft75",
                "chemdeprc_cap1",
                "chemdeprc_cap2",
            ]
        )
    ]
    delta = comp.merge(base, on=["q", "budget_label"], suffixes=("", "_bh"))
    for col in [
        "selected_count",
        "true_hits",
        "fdp",
        "precision",
        "unique_blocks",
        "max_block_share",
        "mean_pairwise_tanimoto",
        "empty",
    ]:
        delta[f"delta_{col}"] = delta[col] - delta[f"{col}_bh"]
    delta_cols = [
        "q",
        "budget_label",
        "budget",
        "method",
        "delta_selected_count",
        "delta_true_hits",
        "delta_fdp",
        "delta_precision",
        "delta_unique_blocks",
        "delta_max_block_share",
        "delta_mean_pairwise_tanimoto",
        "delta_empty",
    ]
    delta = delta[(delta["q"] == q) & (delta["budget_label"].astype(str) == str(budget_label))]
    write_pair(delta[delta_cols].sort_values("method"), outdir, f"{stem_prefix}_delta_vs_bh_q{q:g}_B{budget_label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/chembl_smoke_10x4000_min50/screening_manifest.csv")
    parser.add_argument("--scorer-metrics", default="")
    parser.add_argument("--selection-summary", default="")
    parser.add_argument("--outdir", default="tables")
    parser.add_argument("--manifest-stem", default="chembl_smoke_manifest_min50")
    parser.add_argument("--scorer-stem", default="chembl_ecfp_xgb_scorer_metrics")
    parser.add_argument("--selection-stem-prefix", default="chembl_selection")
    parser.add_argument("--q", type=float, default=0.7)
    parser.add_argument("--budget-label", default="50")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    manifest_table(root / args.manifest, outdir, args.manifest_stem)
    if args.scorer_metrics:
        scorer_table(root / args.scorer_metrics, outdir, args.scorer_stem)
    if args.selection_summary:
        selection_tables(root / args.selection_summary, outdir, args.q, args.budget_label, args.selection_stem_prefix)
    print(f"Wrote ChEMBL tables to {outdir}")


if __name__ == "__main__":
    main()
