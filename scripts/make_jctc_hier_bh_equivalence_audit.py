#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chemdeprc.selection import select


METHOD_A = "chemdeprc_cap1"
METHOD_B = "hier_bh_cap1"
KEY_COLS = [
    "target_chembl_id",
    "rep",
    "artifact_fraction",
    "logit_boost",
    "block_perturbation",
    "q",
    "budget_label",
]
CELL_COLS = ["artifact_fraction", "logit_boost", "block_perturbation", "q", "budget_label"]
METRICS = ["selected_count", "true_hits", "false_hits", "fdp", "fdp_exceeds_q", "unique_blocks", "max_block_share"]


def write_pair(df: pd.DataFrame, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stem}.csv", index=False)
    try:
        text = df.to_markdown(index=False, floatfmt=".6f")
    except ImportError:
        text = df.to_string(index=False)
    (outdir / f"{stem}.md").write_text(text + "\n", encoding="utf-8")


def compare_raw_metrics(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    use = results[results["method"].isin([METHOD_A, METHOD_B])].copy()
    use["budget_label"] = use["budget_label"].astype(str)
    a = use[use["method"] == METHOD_A][KEY_COLS + METRICS]
    b = use[use["method"] == METHOD_B][KEY_COLS + METRICS]
    paired = a.merge(b, on=KEY_COLS, suffixes=("_chemdeprc", "_hier_bh"))
    if paired.empty:
        raise RuntimeError("No paired ChemDep-RC/hier_bh rows were found")
    for metric in METRICS:
        paired[f"{metric}_delta"] = paired[f"{metric}_chemdeprc"] - paired[f"{metric}_hier_bh"]
        paired[f"{metric}_abs_delta"] = paired[f"{metric}_delta"].abs()
    paired["all_metric_equal"] = (paired[[f"{metric}_abs_delta" for metric in METRICS]] <= 1e-12).all(axis=1)

    metric_summary = []
    for metric in METRICS:
        metric_summary.append(
            {
                "metric": metric,
                "mean_delta": float(paired[f"{metric}_delta"].mean()),
                "max_abs_delta": float(paired[f"{metric}_abs_delta"].max()),
                "nonzero_delta_pairs": int((paired[f"{metric}_abs_delta"] > 1e-12).sum()),
                "n_pairs": int(len(paired)),
            }
        )
    metric_summary_df = pd.DataFrame(metric_summary)

    cell = (
        paired.groupby(CELL_COLS, as_index=False)
        .agg(
            n_target_reps=("all_metric_equal", "size"),
            all_metric_equal_rate=("all_metric_equal", "mean"),
            max_abs_true_hits_delta=("true_hits_abs_delta", "max"),
            max_abs_fdp_delta=("fdp_abs_delta", "max"),
            max_abs_fdp_exceeds_q_delta=("fdp_exceeds_q_abs_delta", "max"),
            max_abs_unique_blocks_delta=("unique_blocks_abs_delta", "max"),
            max_abs_max_block_share_delta=("max_block_share_abs_delta", "max"),
            mean_true_hits_delta=("true_hits_delta", "mean"),
            mean_fdp_delta=("fdp_delta", "mean"),
            mean_fdp_exceeds_q_delta=("fdp_exceeds_q_delta", "mean"),
            mean_unique_blocks_delta=("unique_blocks_delta", "mean"),
            mean_max_block_share_delta=("max_block_share_delta", "mean"),
        )
        .sort_values(CELL_COLS)
    )
    cell["pareto_relation"] = np.where(cell["all_metric_equal_rate"] == 1.0, "exact_metric_tie", "divergent")
    return paired, metric_summary_df, cell


def property_test_equivalence(rng_seed: int, trials: int) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    rows: list[dict] = []
    for trial in range(trials):
        n = int(rng.integers(10, 350))
        n_blocks = int(rng.integers(1, min(n, 80) + 1))
        blocks = rng.integers(0, n_blocks, size=n).astype(str)
        pvalues = np.clip(rng.beta(0.7, 5.0, size=n), 1e-12, 1.0)
        scores = rng.random(n)
        q = float(rng.choice([0.01, 0.025, 0.05, 0.1, 0.2, 0.7, 0.9]))
        budget = int(rng.integers(1, min(n, 120) + 1))
        selected_a = select(METHOD_A, scores=scores, pvalues=pvalues, blocks=blocks, q=q, budget=budget).selected
        selected_b = select(METHOD_B, scores=scores, pvalues=pvalues, blocks=blocks, q=q, budget=budget).selected
        same = bool(np.array_equal(selected_a, selected_b))
        rows.append(
            {
                "trial": trial,
                "n": n,
                "n_blocks": n_blocks,
                "q": q,
                "budget": budget,
                "selected_count_chemdeprc": int(len(selected_a)),
                "selected_count_hier_bh": int(len(selected_b)),
                "exact_same_selected_set": same,
            }
        )
    out = pd.DataFrame(rows)
    if not out["exact_same_selected_set"].all():
        bad = out[~out["exact_same_selected_set"]].head(5)
        raise AssertionError(f"Unexpected ChemDep/hier_bh divergence in property test:\n{bad}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/jctc_revision_mechanism_direct/latest_results.csv")
    parser.add_argument("--outdir", default="tables")
    parser.add_argument("--stem", default="jctc_hier_bh_equivalence")
    parser.add_argument("--property-trials", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=353537)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    results = pd.read_csv(root / args.results)
    paired, metric_summary, cell = compare_raw_metrics(results)
    prop = property_test_equivalence(args.seed, args.property_trials)

    divergent_cells = cell[cell["pareto_relation"] != "exact_metric_tie"].copy()
    if divergent_cells.empty:
        divergent_cells = pd.DataFrame(
            [
                {
                    "status": "no_metric_divergent_cells_detected",
                    "n_cells_checked": int(len(cell)),
                    "n_paired_target_reps": int(len(paired)),
                    "interpretation": (
                        "The audited grid shows metric-level equivalence for ChemDep-RC cap1 and "
                        "hier_bh_cap1 in every q x budget x artifact cell."
                    ),
                }
            ]
        )

    overall = pd.DataFrame(
        [
            {
                "method_a": METHOD_A,
                "method_b": METHOD_B,
                "n_paired_target_reps": int(len(paired)),
                "n_cells": int(len(cell)),
                "all_metric_equal_pairs": int(paired["all_metric_equal"].sum()),
                "all_metric_equal_rate": float(paired["all_metric_equal"].mean()),
                "metric_divergent_pairs": int((~paired["all_metric_equal"]).sum()),
                "metric_divergent_cells": int((cell["pareto_relation"] != "exact_metric_tie").sum()),
                "property_test_trials": int(len(prop)),
                "property_test_exact_same_rate": float(prop["exact_same_selected_set"].mean()),
                "formal_exact_selected_set_rate": 1.0,
                "formal_different_selected_set_rate": 0.0,
                "formal_selection_jaccard": 1.0,
                "exact_set_evidence": (
                    "Formal code-level equivalence plus randomized property tests. The archived raw revision "
                    "grid stores metrics rather than selected indices, so the raw-grid audit verifies metric "
                    "equality across all paired target-replicates."
                ),
                "diagnostic_conclusion": (
                    "ChemDep-RC cap1 is selection-equivalent to hierarchical Simes-BH with one retained "
                    "candidate per selected block under the current p-value-ranked hard-cap implementation."
                ),
                "positioning_action": (
                    "Do not present hier_bh_cap1 as an independent competing baseline for ChemDep-RC cap1. "
                    "Present it as the formal cap1 limiting/special-case diagnostic; claim novelty through "
                    "the scoring-function-agnostic dependency-aware decision layer, component attribution, "
                    "soft/score-ranked variants, budget policy, and empirical frontier behavior."
                ),
            }
        ]
    )

    proof = pd.DataFrame(
        [
            {
                "step": 1,
                "statement": "Both methods compute the same Simes p-value for each dependency block and apply BH to the same block-level p-values.",
            },
            {
                "step": 2,
                "statement": "For any selected block, the block-level BH threshold is at most q, so its Simes p-value is <= q; therefore the within-block BH(q) pass is nonempty and includes the minimum-p candidate.",
            },
            {
                "step": 3,
                "statement": "With cap=1 and p-value member ranking, both methods retain exactly the within-block candidate with the smallest p-value, using the same score tie-breaker.",
            },
            {
                "step": 4,
                "statement": "Round-robin budget allocation over the same ordered selected blocks is therefore identical for chemdeprc_cap1 and hier_bh_cap1.",
            },
            {
                "step": 5,
                "statement": "Equivalence need not hold for soft caps, score-ranked variants, cap>1 variants, alternative block p-values, or explicit budget policies.",
            },
        ]
    )

    write_pair(overall, outdir, f"{args.stem}_overall")
    write_pair(metric_summary, outdir, f"{args.stem}_metric_summary")
    write_pair(cell, outdir, f"{args.stem}_cellwise")
    write_pair(divergent_cells, outdir, f"{args.stem}_divergent_cells")
    write_pair(proof, outdir, f"{args.stem}_formal_relation")

    by_q_budget = (
        paired.groupby(["q", "budget_label"], as_index=False)
        .agg(
            n_paired_target_reps=("all_metric_equal", "size"),
            all_metric_equal_rate=("all_metric_equal", "mean"),
            max_abs_true_hits_delta=("true_hits_abs_delta", "max"),
            max_abs_fdp_delta=("fdp_abs_delta", "max"),
            max_abs_fdp_exceeds_q_delta=("fdp_exceeds_q_abs_delta", "max"),
        )
        .sort_values(["q", "budget_label"])
    )
    write_pair(by_q_budget, outdir, f"{args.stem}_by_q_budget")

    metadata = {
        "source_results": args.results,
        "n_input_rows": int(len(results)),
        "n_paired_target_reps": int(len(paired)),
        "n_cells": int(len(cell)),
        "property_trials": int(len(prop)),
        "seed": args.seed,
        "method_a": METHOD_A,
        "method_b": METHOD_B,
        "selected_indices_in_source_results": False,
        "exact_set_evidence": "formal_relation_plus_randomized_property_tests",
        "generated_tables": [
            f"{args.stem}_overall",
            f"{args.stem}_metric_summary",
            f"{args.stem}_cellwise",
            f"{args.stem}_divergent_cells",
            f"{args.stem}_formal_relation",
            f"{args.stem}_by_q_budget",
        ],
    }
    (outdir / f"{args.stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
