#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def fmt(x: object) -> object:
    if isinstance(x, float):
        return f"{x:.4f}"
    return x


def table(df: pd.DataFrame, columns: list[str] | None = None, n: int | None = None) -> str:
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]].copy()
    if n is not None:
        df = df.head(n).copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda v: f"{v:.4f}")
    return df.to_markdown(index=False)


def read(root: Path, rel: str) -> pd.DataFrame:
    return pd.read_csv(root / rel)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    adaptive = read(root, "tables/pr_adaptive_policy_overall_q0p7.csv")
    adaptive = adaptive[
        adaptive["policy"].isin(
            [
                "oracle_label_aware",
                "label_free_adaptive_tree",
                "fixed_bh",
                "fixed_weighted_bh",
                "fixed_score_cap1",
                "fixed_chemdeprc_cap1",
                "fixed_chemdeprc_soft75",
            ]
        )
    ].sort_values("policy")
    adaptive_regime = read(root, "tables/pr_adaptive_policy_by_regime_q0p7.csv")
    adaptive_freq = read(root, "tables/pr_adaptive_policy_choice_frequency_q0p7.csv")
    adaptive_feat = read(root, "tables/pr_adaptive_policy_features_q0p7.csv")
    phase = read(root, "tables/pr_phase_regime_table_q0p7_nonzero.csv")
    native_overall = read(root, "tables/pr_native_retrieval_overall_q0p7.csv")
    native_verdict = read(root, "tables/pr_native_retrieval_verdict_q0p7.csv")
    native_tests = read(root, "tables/pr_native_retrieval_paired_tests_vs_scorecap_q0p7.csv")
    audit = read(root, "tables/jctc_revision_sota_completion_audit.csv")
    scorecap = read(root, "tables/jctc_direct_mechanism_vs_scorecap_bootstrap.csv")
    frozen = read(root, "tables/jctc_frozen_policy_heldout_q0.7.csv")
    public_verdict = read(root, "tables/public_moleculenet_sota_verdict_q0.7.csv")

    main_scorecap = scorecap[
        (scorecap["artifact_fraction"].round(4) == 0.1)
        & (scorecap["logit_boost"].round(4) == 4.0)
        & (scorecap["budget_label"].astype(str) == "50")
        & (scorecap["method"] == "chemdeprc_cap1")
        & (scorecap["metric"].isin(["true_hits", "fdp", "unique_blocks", "max_block_share"]))
    ].copy()

    native_key_tests = native_tests[
        (native_tests["method"].isin(["chemdeprc_cap1", "chemdeprc_soft75", "weighted_bh", "bh"]))
        & (native_tests["metric"].isin(["true_hits", "fdp", "unique_blocks", "max_block_share"]))
    ].copy()

    completion_rows = pd.DataFrame(
        [
            {
                "PR requirement": "Label-free dependency-pressure + adaptive policy",
                "status": "newly completed",
                "evidence": "94,500 held-out target/cell evaluations; selector uses raw_top_b structure features only.",
                "main files": "tables/pr_adaptive_policy_*.csv",
            },
            {
                "PR requirement": "When to use / not use method",
                "status": "newly completed",
                "evidence": "policy choice frequency plus dependency-pressure x budget-ratio regime table.",
                "main files": "tables/pr_phase_regime_table_q0p7_nonzero.csv",
            },
            {
                "PR requirement": "Matched cap-vs-risk mechanism check",
                "status": "already completed",
                "evidence": "ChemDep cap1 vs score_cap1 gives small but stable risk-layer gain in the main stress cell.",
                "main files": "tables/jctc_direct_mechanism_vs_scorecap_bootstrap.csv",
            },
            {
                "PR requirement": "Scorer/backbone agnosticism",
                "status": "completed as score-backbone diagnostic",
                "evidence": "ECFP-XGB ChEMBL plus LIT-PCBA/DUD-E full and ligand-only external score panels; not claimed as full GNN/pretrained-molecule retraining.",
                "main files": "tables/jctc_external_targetwise_score_backbone_q_all.csv",
            },
            {
                "PR requirement": "PR-native non-chemical benchmark",
                "status": "newly completed",
                "evidence": "sklearn digits and Olivetti faces group retrieval; 5 seeds, natural and visual-cluster-confounded scenarios.",
                "main files": "tables/pr_native_retrieval_*.csv",
            },
            {
                "PR requirement": "Cross-dataset statistical completeness",
                "status": "completed",
                "evidence": "Strict public MoleculeNet suite plus new PR-native Wilcoxon tests; no original-paper numbers mixed.",
                "main files": "tables/public_moleculenet_*.csv; tables/pr_native_retrieval_paired_tests_vs_scorecap_q0p7.csv",
            },
        ]
    )

    md = f"""# ChemDep-RC PR Supplement: Experiments, Data, Storyline

Generated: {created}

## One-line Verdict

For a Pattern Recognition submission, the strongest honest claim is now:

**ChemDep-RC is a label-free, dependency-aware finite-budget selection framework that learns when to use hard caps, soft caps, or ordinary ranking-like policies from score/block structure, and it reaches a conditional frontier rather than an unconditional hit-count SOTA.**

Do **not** claim universal FDP control, unconditional hit-count SOTA, or full docking/GNN/pretrained-backbone SOTA. The claim-safe phrase is:

> conditional frontier-SOTA under dependency/artifact-prone finite-budget selection, with explicit failure-boundary and regime diagnostics.

## PR Reviewer Questions

1. Is this a real method, not just a cap?
   Yes, after this supplement: the new adaptive selector uses label-free dependency-pressure features and is evaluated by leave-target-out policy selection against oracle and fixed policies.

2. When should it be used, and when should it not be used?
   The new phase/regime table shows that no single policy dominates. Small-budget/high-dependency cells favor risk-aware block policies; larger budget-ratio cells increasingly move toward soft allocation or raw/weighted ranking. This is the main PR story.

3. Is it independent of chemistry-specific scoring?
   Partially yes. The selection layer is tested on ECFP-XGB ChEMBL, LIT-PCBA/DUD-E external full/ligand-only score backbones, and a newly added non-chemical image retrieval benchmark. We should still avoid saying we have a full GNN plus molecular-pretraining panel.

## Completion Audit

{table(completion_rows)}

## New Experiment 1: Label-free Adaptive ChemDep-RC

Protocol: train a shallow decision-tree selector on development targets and evaluate by leave-target-out. Features are computed only from score/block structure visible before held-out labels: top-B block concentration, top-B unique ratio, budget-to-pool ratio, p-value extremeness, and top-B similarity. The selector never uses held-out labels.

Key result:

{table(adaptive, ["policy", "true_hits", "fdp", "unique_blocks", "max_block_share", "hit_regret_vs_oracle", "fdp_regret_vs_oracle", "matches_oracle", "pareto_member", "n_cells"])}

Interpretation:

- `label_free_adaptive_tree` has `true_hits=81.5088`, `fdp=0.1936`, and Pareto membership `0.7535` over 94,500 cells.
- It is close to the label-aware oracle (`true_hits=83.5714`, `fdp=0.1660`) without seeing held-out labels.
- It avoids the weakness of fixed `chemdeprc_cap1` (very low concentration but large hit regret) and fixed `score_cap1` (high diversity but high FDP).
- This directly addresses the PR concern that the author may be choosing cap1/soft75 after seeing the test labels.

Hard vs mild regimes:

{table(adaptive_regime[adaptive_regime["policy"].isin(["oracle_label_aware", "label_free_adaptive_tree", "fixed_bh", "fixed_weighted_bh", "fixed_score_cap1", "fixed_chemdeprc_cap1", "fixed_chemdeprc_soft75"])], ["stress_regime", "policy", "true_hits", "fdp", "unique_blocks", "max_block_share", "hit_regret_vs_oracle", "fdp_regret_vs_oracle", "pareto_member", "n_cells"])}

Policy choices by budget:

{table(adaptive_freq, ["budget_label", "chosen_method", "n_cells", "cell_share"])}

Most important label-free policy features:

{table(adaptive_feat, ["feature", "leave_target_out_tree_importance"])}

## New Experiment 2: Applicability Phase / Regime Table

The phase table converts many positive and negative results into a method applicability map. It should be described as a table now; figures can be drawn later.

Top non-empty regimes:

{table(phase, ["pressure_bin", "budget_ratio_bin", "oracle_family", "oracle_method", "n_cells", "cell_share"], n=36)}

Writing interpretation:

- Low pressure plus mid budget-ratio often falls back to `raw_top_b`, meaning strict dependency control is unnecessary when the candidate pool is not concentration-prone.
- High pressure plus small/mid budget-ratio often selects `weighted_bh` or `chemdeprc_cap1`, meaning risk-aware allocation matters most when a small budget is exposed to concentrated dependency.
- High pressure plus large budget-ratio spreads across weighted, raw, score-cap, and soft ChemDep, supporting the transition from hard capacity to soft allocation.

## New Experiment 3: PR-native Non-chemical Benchmark

Protocol: image-style finite-budget retrieval on `sklearn_digits` and `olivetti_faces_group10`, 5 seeds, two scenarios (`natural`, `visual_cluster_confounder`), budgets 25/50/100. Scores come from classifier confidence; dependency blocks are unsupervised visual embedding clusters; labels are used only for evaluation and explicit confounder stress construction.

Overall result:

{table(native_overall, ["method", "true_hits", "fdp", "unique_blocks", "max_block_share", "mean_pairwise_similarity", "pareto_member"])}

Dataset-scenario verdict:

{table(native_verdict, ["dataset", "scenario", "budget_label", "best_true_hits_method", "best_true_hits", "lowest_fdp_method", "lowest_fdp", "best_chemdeprc_method", "best_chemdeprc_true_hits", "best_chemdeprc_fdp", "chemdeprc_non_dominated_rate"])}

Paired tests against `score_cap1`:

{table(native_key_tests, ["baseline", "method", "metric", "mean_delta", "median_delta", "wilcoxon_pvalue", "n_pairs"])}

Interpretation:

- This benchmark should not be sold as ChemDep winning all image retrieval metrics.
- Its value is stronger and more PR-relevant: the same finite-budget dependency tradeoff appears outside chemistry.
- `score_cap1` is an extreme diversity frontier point; `chemdeprc_cap1` is a low-FDP/risk-conservative point; `chemdeprc_soft75` recovers many more hits but can pay FDP.
- This supports the general pattern-recognition formulation: dependency-aware finite-budget selection is not merely a Murcko/scaffold heuristic.

## Existing Mechanism Evidence That Should Move To Main Results

ChemDep cap1 vs score_cap1 in the main artifact stress cell (`q=0.7`, `B=50`, artifact fraction `0.1`, boost `4`):

{table(main_scorecap, ["method", "metric", "mean_delta", "ci95_low", "ci95_high", "n_pairs"])}

This is the clean answer to "is it just a cap?": under identical cap1 diversity, ChemDep adds a small but stable grouped risk layer. The effect is not large, so the manuscript should say "risk refinement beyond score cap" rather than "large improvement beyond cap".

Frozen dev-to-heldout policy evidence:

{table(frozen, ["budget_label", "policy", "heldout_true_hits", "heldout_fdp", "heldout_unique_blocks", "heldout_max_block_share", "n_heldout_scenarios"])}

## Public SOTA / Benchmark Status

JCTC/SOTA audit:

{table(audit, ["criterion", "status", "evidence"])}

Public MoleculeNet verdict snapshot:

{table(public_verdict, n=20)}

Use this wording:

- "ChemDep-RC is non-dominated across a large fraction of public dependency-prone screening cells."
- "It is best/tied-best on selected risk-power-diversity fronts."
- "It is not an unconditional hit-count winner, and this is expected because hard dependency control trades power for reduced concentration."

## Revised PR Storyline

**Problem.** Pointwise prediction/ranking can fail under finite experimental budgets because top-ranked candidates may be dependent variants of the same error pattern or visual/chemical block.

**Failure pattern.** This is not only a molecule problem. The new image retrieval benchmark shows the same structure: high-confidence candidates can cluster into repeated visual modes, making raw top-B or score-only diversity insufficient depending on the regime.

**Method.** ChemDep-RC should be presented as a structured selection layer:

1. diagnose dependency pressure from score/block structure;
2. filter or admit candidates with grouped risk evidence;
3. allocate budget by hard cap, soft cap, or ranking-like policy;
4. choose the policy adaptively without held-out labels.

**Evidence.** The adaptive selector is close to the oracle over 94,500 ChEMBL stress cells; mechanism tables show risk gain beyond score_cap1; external score-backbone and image retrieval panels show the formulation is not locked to ECFP-XGB or Murcko scaffolds.

**Boundary.** The method is strongest when dependency pressure is high and budget is limited. When pressure is low or budget is large, raw/weighted ranking or soft policies can be better. This boundary should be framed as a contribution, not a weakness.

## Contributions For PR-style Manuscript

1. We identify dependent finite-budget set selection as a failure mode of pointwise recognition/ranking systems.
2. We propose ChemDep-RC, a label-free dependency-aware selection framework with adaptive capacity allocation.
3. We provide mechanism-level evidence that the grouped risk layer adds value beyond a score-only cap.
4. We demonstrate domain-level generalization on chemistry and image-style retrieval while explicitly mapping failure regimes.

## Draft Abstract

Finite-budget selection is common in molecular screening, retrieval, and other pattern-recognition pipelines, yet pointwise confidence scores can concentrate on dependent candidates that share the same underlying error mode. We study this problem as dependency-aware finite-budget set selection. ChemDep-RC augments score-based ranking with label-free dependency blocks, grouped risk admission, and adaptive capacity allocation. A new label-free policy selector diagnoses dependency pressure from score/block structure and chooses among ranking, hard-cap, and soft-cap selection without accessing held-out labels. Across 94,500 ChEMBL stress cells, the adaptive policy approaches the label-aware oracle while improving the risk-power-diversity frontier over fixed BH, score-cap, and ChemDep variants. Mechanism-controlled comparisons show that the grouped risk layer provides a small but stable gain beyond score-only cap1. Additional public molecular benchmarks, external score-backbone panels, and image-style retrieval tasks demonstrate that the formulation is not specific to Murcko scaffolds or one chemical scorer. The method is best interpreted as a conditional frontier approach: it is most useful under high dependency pressure and limited budgets, and it naturally relaxes toward softer policies when dependency pressure is weak or budget is large.

## Data Index

- Adaptive policy: `tables/pr_adaptive_policy_overall_q0p7.csv`, `tables/pr_adaptive_policy_by_regime_q0p7.csv`, `tables/pr_adaptive_policy_choice_frequency_q0p7.csv`, `tables/pr_adaptive_policy_features_q0p7.csv`
- Phase/regime: `tables/pr_phase_regime_table_q0p7_nonzero.csv`
- PR-native image retrieval: `data/pr_native_retrieval/latest_results.csv`, `data/pr_native_retrieval/latest_summary.csv`, `tables/pr_native_retrieval_*.csv`
- Mechanism beyond cap: `tables/jctc_direct_mechanism_vs_scorecap_bootstrap.csv`
- Frozen policy: `tables/jctc_frozen_policy_heldout_q0.7.csv`
- External score-backbone: `tables/jctc_external_targetwise_score_backbone_q_all.csv`
- Public benchmark/SOTA audit: `tables/jctc_revision_sota_completion_audit.csv`, `tables/public_moleculenet_*.csv`

## Recommended Claim Boundary

Use:

> ChemDep-RC reaches a conditional frontier-SOTA for dependency-prone finite-budget selection and supplies a deployable label-free policy for choosing the appropriate capacity allocation regime.

Avoid:

> ChemDep-RC is universally SOTA, universally controls FDP/FDR, or outperforms all score backbones in hit count.
"""

    out = docs / "pr_supplement_storyline.md"
    out.write_text(md, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
