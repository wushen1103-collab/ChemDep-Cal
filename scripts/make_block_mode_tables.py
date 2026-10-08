#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODE_SUMMARIES = {
    "murcko": "data/chembl_block_artifact_potent_a7_i6/latest_summary.csv",
    "tanimoto80": "data/chembl_block_artifact_potent_a7_i6_tanimoto80/latest_summary.csv",
    "hybrid75": "data/chembl_block_artifact_potent_a7_i6_hybrid75/latest_summary.csv",
}

KEY_METHODS = ["bh", "weighted_bh", "chemdeprc_soft75", "chemdeprc_cap1", "by"]


def write_pair(df: pd.DataFrame, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(df.to_markdown(index=False, floatfmt=".4f") + "\n", encoding="utf-8")


def load_mode_summaries(root: Path) -> pd.DataFrame:
    frames = []
    for mode, rel_path in MODE_SUMMARIES.items():
        df = pd.read_csv(root / rel_path)
        df.insert(0, "block_mode", mode)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def murcko_testpool_stats(root: Path, scores_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted((root / scores_dir).glob("*_scores.csv")):
        df = pd.read_csv(path, usecols=["target_chembl_id", "pref_name", "split", "murcko_scaffold", "molecule_chembl_id"])
        testpool = df[df["split"] == "testpool"].copy()
        blocks = testpool["murcko_scaffold"].fillna(testpool["molecule_chembl_id"]).astype(str)
        counts = blocks.value_counts()
        rows.append(
            {
                "block_mode": "murcko",
                "target_chembl_id": str(testpool["target_chembl_id"].iloc[0]),
                "pref_name": str(testpool["pref_name"].iloc[0]),
                "n_molecules": int(len(testpool)),
                "n_blocks": int(len(counts)),
                "max_block_size": int(counts.max()),
                "median_block_size": float(counts.median()),
                "singleton_rate": float(np.mean(counts.to_numpy() == 1)),
            }
        )
    return pd.DataFrame(rows)


def alt_testpool_stats(root: Path, mode: str, rel_path: str) -> pd.DataFrame:
    df = pd.read_csv(root / rel_path)
    df = df[df["split"] == "testpool"].copy()
    df.insert(0, "block_mode", mode)
    return df[
        [
            "block_mode",
            "target_chembl_id",
            "pref_name",
            "n_molecules",
            "n_blocks",
            "max_block_size",
            "median_block_size",
            "singleton_rate",
            "tanimoto_edges",
            "murcko_edges",
        ]
    ]


def write_block_stats(root: Path, outdir: Path, scores_dir: str) -> None:
    frames = [
        murcko_testpool_stats(root, scores_dir),
        alt_testpool_stats(root, "tanimoto80", "data/chembl_ecfp_xgb_potent_a7_i6_tanimoto80/block_stats.csv"),
        alt_testpool_stats(root, "hybrid75", "data/chembl_ecfp_xgb_potent_a7_i6_hybrid75/block_stats.csv"),
    ]
    stats = pd.concat(frames, ignore_index=True)
    summary = (
        stats.groupby("block_mode", as_index=False)
        .agg(
            n_targets=("target_chembl_id", "nunique"),
            n_blocks_mean=("n_blocks", "mean"),
            n_blocks_min=("n_blocks", "min"),
            n_blocks_max=("n_blocks", "max"),
            max_block_size_mean=("max_block_size", "mean"),
            max_block_size_max=("max_block_size", "max"),
            singleton_rate_mean=("singleton_rate", "mean"),
        )
        .sort_values("block_mode")
    )
    write_pair(stats.sort_values(["block_mode", "target_chembl_id"]), outdir, "chembl_block_mode_stats_by_target")
    write_pair(summary, outdir, "chembl_block_mode_stats_summary")


def comparison_table(summary: pd.DataFrame, *, q: float, budget: str, stress_label: str) -> pd.DataFrame:
    mask = (
        (summary["q"] == q)
        & (summary["budget_label"].astype(str) == str(budget))
        & (summary["stress_label"] == stress_label)
        & summary["method"].isin(KEY_METHODS)
    )
    cols = [
        "block_mode",
        "stress_label",
        "artifact_fraction",
        "logit_boost",
        "q",
        "budget_label",
        "method",
        "selected_count",
        "true_hits",
        "fdp",
        "fdp_exceeds_q",
        "unique_blocks",
        "max_block_share",
        "mean_pairwise_tanimoto",
    ]
    return summary[mask][cols].sort_values(["block_mode", "method"])


def delta_vs_bh(table: pd.DataFrame) -> pd.DataFrame:
    base = table[table["method"] == "bh"][
        [
            "block_mode",
            "stress_label",
            "q",
            "budget_label",
            "selected_count",
            "true_hits",
            "fdp",
            "fdp_exceeds_q",
            "unique_blocks",
            "max_block_share",
            "mean_pairwise_tanimoto",
        ]
    ]
    comp = table[table["method"] != "bh"].merge(
        base,
        on=["block_mode", "stress_label", "q", "budget_label"],
        suffixes=("", "_bh"),
    )
    for col in [
        "selected_count",
        "true_hits",
        "fdp",
        "fdp_exceeds_q",
        "unique_blocks",
        "max_block_share",
        "mean_pairwise_tanimoto",
    ]:
        comp[f"delta_{col}"] = comp[col] - comp[f"{col}_bh"]
    return comp[
        [
            "block_mode",
            "stress_label",
            "q",
            "budget_label",
            "method",
            "delta_selected_count",
            "delta_true_hits",
            "delta_fdp",
            "delta_fdp_exceeds_q",
            "delta_unique_blocks",
            "delta_max_block_share",
            "delta_mean_pairwise_tanimoto",
        ]
    ].sort_values(["stress_label", "budget_label", "method", "block_mode"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="tables")
    parser.add_argument("--scores-dir", default="data/chembl_ecfp_xgb_potent_a7_i6/scores")
    parser.add_argument("--q", type=float, default=0.7)
    parser.add_argument("--budgets", nargs="+", default=["50", "100"])
    parser.add_argument("--stress-labels", nargs="+", default=["no_artifact", "frac0.1_boost4"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    write_block_stats(root, outdir, args.scores_dir)
    summary = load_mode_summaries(root)

    all_tables = []
    all_deltas = []
    for stress_label in args.stress_labels:
        for budget in args.budgets:
            table = comparison_table(summary, q=args.q, budget=budget, stress_label=stress_label)
            safe_stress = stress_label.replace(".", "p")
            stem = f"chembl_block_mode_{safe_stress}_q{args.q:g}_B{budget}"
            write_pair(table, outdir, stem)
            delta = delta_vs_bh(table)
            write_pair(delta, outdir, f"{stem}_delta_vs_bh")
            all_tables.append(table)
            all_deltas.append(delta)

    write_pair(pd.concat(all_tables, ignore_index=True), outdir, f"chembl_block_mode_all_q{args.q:g}")
    write_pair(pd.concat(all_deltas, ignore_index=True), outdir, f"chembl_block_mode_all_q{args.q:g}_delta_vs_bh")
    print(f"Wrote block-mode tables to {outdir}")


if __name__ == "__main__":
    main()
