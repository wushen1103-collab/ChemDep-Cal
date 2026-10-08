#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROUTES = {
    "bh": "classic_marginal_fdr",
    "by": "classic_dependence_robust_fdr",
    "weighted_bh": "weighted_or_block_size_fdr",
    "block_bh": "block_level_fdr",
    "score_cap1": "score_ranked_diversity_cap",
    "bh_cap1": "posthoc_fdr_plus_diversity_cap",
    "block_by_cap1": "dependence_robust_block_fdr",
    "minp_block_bh_cap1": "familywise_block_screen_plus_cap",
    "hier_bh_cap1": "formal_cap1_special_case_diagnostic",
    "chemdeprc_cap1": "proposed_grouped_risk_aware_cap",
    "chemdeprc_scorecap1": "proposed_score_ranked_grouped_cap",
    "chemdeprc_soft75": "proposed_soft_budget_grouped_cap",
}


PROVENANCE = {
    method: "rerun_same_scores_same_splits_this_work" for method in ROUTES
}


def write_pair(df: pd.DataFrame, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stem}.csv", index=False)
    try:
        text = df.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        text = df.to_string(index=False)
    (outdir / f"{stem}.md").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="data/jctc_revision_mechanism_direct/latest_summary.csv")
    parser.add_argument("--outdir", default="tables")
    parser.add_argument("--artifact-fraction", type=float, default=0.1)
    parser.add_argument("--logit-boost", type=float, default=4.0)
    parser.add_argument("--q", type=float, default=0.7)
    parser.add_argument("--budget-label", default="50")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    summary = pd.read_csv(root / args.summary)
    summary["budget_label"] = summary["budget_label"].astype(str)
    frame = summary[
        (summary["artifact_fraction"] == args.artifact_fraction)
        & (summary["logit_boost"] == args.logit_boost)
        & (summary["q"] == args.q)
        & (summary["budget_label"] == str(args.budget_label))
        & (summary["method"].isin(ROUTES))
    ].copy()
    if frame.empty:
        raise RuntimeError("No grouped baseline rows matched the requested scenario.")
    frame["technical_route"] = frame["method"].map(ROUTES)
    frame["result_source"] = frame["method"].map(PROVENANCE)
    keep = [
        "technical_route",
        "method",
        "true_hits",
        "true_hits_std",
        "fdp",
        "fdp_std",
        "fdp_exceeds_q",
        "unique_blocks",
        "max_block_share",
        "mean_pairwise_tanimoto",
        "n_target_reps",
        "n_targets",
        "result_source",
    ]
    frame = frame[keep].sort_values(["technical_route", "method"])
    stem = (
        f"jctc_grouped_baseline_mainstress_q{args.q:g}_B{args.budget_label}"
        f"_f{args.artifact_fraction:g}_boost{args.logit_boost:g}"
    )
    stem = stem.replace(".", "p")
    write_pair(frame, root / args.outdir, stem)


if __name__ == "__main__":
    main()
