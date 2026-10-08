#!/usr/bin/env python3
"""Explain matched ChemDep/score-cap hit differences by exact selection swaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def load_ranker(path: Path) -> XGBClassifier:
    model = XGBClassifier(n_jobs=4)
    model.load_model(path)
    return model


def compare(panel: dict, scores: np.ndarray, model: XGBClassifier,
            expected_score: int, expected_chemdep: int) -> dict:
    from probe_meta_band_v2_20261003 import evidence_features
    from train_artifact_meta_ranker_20261003 import cap_one

    labels = panel["test"].label.to_numpy(int)
    blocks = panel["test_blocks"]
    score_idx = cap_one(scores, blocks, 50)
    chem_scores = model.predict_proba(evidence_features(scores, blocks, panel["null"]))[:, 1]
    chem_idx = cap_one(chem_scores, blocks, 50)
    assert len(score_idx) == len(chem_idx) == 50
    assert len(set(blocks[score_idx])) == len(set(blocks[chem_idx])) == 50
    score_hits = int(labels[score_idx].sum())
    chem_hits = int(labels[chem_idx].sum())
    assert (score_hits, chem_hits) == (expected_score, expected_chemdep), (
        panel["target"], score_hits, chem_hits, expected_score, expected_chemdep)

    score_set, chem_set = set(score_idx), set(chem_idx)
    added = np.asarray(sorted(chem_set - score_set), dtype=int)
    removed = np.asarray(sorted(score_set - chem_set), dtype=int)
    assert len(added) == len(removed)
    score_by_block = {blocks[index]: index for index in score_idx}
    chem_by_block = {blocks[index]: index for index in chem_idx}
    shared_blocks = score_by_block.keys() & chem_by_block.keys()
    return {
        "scorecap_hits": score_hits,
        "chemdep_hits": chem_hits,
        "delta_hits": chem_hits - score_hits,
        "changed_compounds": len(added),
        "changed_scaffolds": len(chem_by_block.keys() - score_by_block.keys()),
        "within_scaffold_replacements": sum(
            score_by_block[block] != chem_by_block[block] for block in shared_blocks),
        "added_true_hits": int(labels[added].sum()),
        "removed_true_hits": int(labels[removed].sum()),
        "added_original_score": float(scores[added].mean()) if len(added) else None,
        "removed_original_score": float(scores[removed].mean()) if len(removed) else None,
    }


def natural(root: Path) -> list[dict]:
    from train_artifact_meta_ranker_20261003 import load_panel

    archive = pd.read_csv(root / "data/pr_natural_modern_selectors_20261004/pooled20.csv")
    expected = {}
    for row in archive.itertuples(index=False):
        if row.method in {"score_cap1", "chemdep_cal"}:
            expected.setdefault((row.dataset, int(row.scorer_seed)), {})[
                row.method] = int(row.hits)
    assert len(expected) == 20
    model = load_ranker(root / "data/pr_natural_modern_selectors_20261004"
                        / "pooled20.chemdep.json")
    rows = []
    for (dataset, seed), hits in sorted(expected.items()):
        assert set(hits) == {"score_cap1", "chemdep_cal"}
        panel = load_panel(root / "data/public_moleculenet_ecfp_xgb/scores"
                           / f"{dataset}_scaffold_ood_seed{seed}_scores.csv")
        rows.append({"source": "moleculenet_natural", "target": dataset,
                     "scorer_seed": seed, "rep": -1, "fraction": 0., "boost": 0.,
                     **compare(panel, panel["test"].score.to_numpy(float), model,
                               hits["score_cap1"], hits["chemdep_cal"])})
    return rows


def synthetic(root: Path, source: str) -> list[dict]:
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from train_artifact_meta_ranker_20261003 import load_panel

    rows = []
    seeds = range(35501, 35506) if source == "fifth_synthetic" else range(35801, 35806)
    for seed in seeds:
        if source == "fifth_synthetic":
            paths = sorted((root / "data/chembl_meta_ungated_fifth_xgb_20261003"
                            / f"seed{seed}" / "scores").glob("*_scores.csv"))
            assert len(paths) == 10
            archive = pd.read_csv(root / "data/pr_fifth_multifp_rf_posthoc_20261005"
                                  / f"seed{seed}_cells.csv")
            expected = {(r.target, int(r.rep), float(r.fraction), float(r.boost)):
                        (int(r.scorecap_hits), int(r.chemdep_hits))
                        for r in archive.itertuples(index=False)}
            model_path = (root / "data/conditional_sota_meta_ungated_fifth_20261003"
                          / f"meta_seed{seed}.model.json")
            base_seed = 355700
        else:
            cohort = root / "data/chembl_modern_eighth_20261005"
            ledger = pd.read_csv(cohort / "technical_eligibility.csv")
            ids = ledger.loc[ledger.eligible, "target_chembl_id"].tolist()
            assert len(ids) == 30
            paths = [cohort / f"eligible_seed{seed}" / "scores"
                     / f"{target}_scores.csv" for target in ids]
            archive = pd.read_csv(cohort / "fixed_selector_results" / f"seed{seed}.csv")
            expected = {}
            for row in archive.itertuples(index=False):
                if row.method in {"score_cap1", "chemdep_cal"}:
                    key = (row.target, int(row.rep), float(row.fraction), float(row.boost))
                    expected.setdefault(key, {})[row.method] = int(row.hits)
            model_path = cohort / "fixed_selector_results" / f"seed{seed}.chemdep.json"
            base_seed = 358700
        assert len(expected) == len(paths) * 40
        model = load_ranker(model_path)
        for index, path in enumerate(paths):
            panel = load_panel(path)
            target, test = panel["target"], panel["test"]
            base = test.score.to_numpy(float)
            target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
            target_seed = base_seed + index * 100_000 + target_hash
            for rep in range(10):
                for fraction in (.2, .3):
                    rng = np.random.default_rng(target_seed + rep * 10_003
                                                + int(fraction * 10_000))
                    artifacts = choose_artifact_blocks(
                        test, artifact_block_col="murcko_scaffold", rng=rng,
                        artifact_fraction=fraction, max_artifact_blocks=25,
                        min_inactive_per_block=2)
                    for boost in (4., 6.):
                        scores = inject_artifact(
                            test, base, artifacts, artifact_block_col="murcko_scaffold",
                            logit_boost=boost)[0]
                        archived = expected[(target, rep, fraction, boost)]
                        if source == "fifth_synthetic":
                            score_hits, chemdep_hits = archived
                        else:
                            assert set(archived) == {"score_cap1", "chemdep_cal"}
                            score_hits, chemdep_hits = (archived["score_cap1"],
                                                        archived["chemdep_cal"])
                        rows.append({"source": source, "target": target,
                                     "scorer_seed": seed, "rep": rep,
                                     "fraction": fraction, "boost": boost,
                                     **compare(panel, scores, model,
                                               score_hits, chemdep_hits)})
        print(source, seed, "verified", len(paths) * 40, "cells", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", choices=("natural", "fifth", "eighth"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    records = natural(root) if args.source == "natural" else synthetic(
        root, "fifth_synthetic" if args.source == "fifth" else "eighth_synthetic")
    frame = pd.DataFrame(records)
    assert (frame.added_true_hits - frame.removed_true_hits == frame.delta_hits).all()
    output = root / "data/pr_selector_swap_audit_20261005"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f"{args.source}_cells.csv", index=False)
    grouped = frame.groupby("target").agg({
        "scorecap_hits": "mean", "chemdep_hits": "mean", "delta_hits": "mean",
        "changed_compounds": "mean", "changed_scaffolds": "mean",
        "within_scaffold_replacements": "mean", "added_true_hits": "mean",
        "removed_true_hits": "mean", "added_original_score": "mean",
        "removed_original_score": "mean",
    })
    grouped.to_csv(output / f"{args.source}_target_means.csv")
    summary = {"source": args.source, "targets": int(frame.target.nunique()),
               "scorer_seeds": int(frame.scorer_seed.nunique()), "cells": len(frame),
               "target_macro": {key: float(value) for key, value in grouped.mean().items()},
               "status": "own exact-selection replay; descriptive, not independent confirmation"}
    (output / f"{args.source}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
