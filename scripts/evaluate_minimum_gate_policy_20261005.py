#!/usr/bin/env python3
"""Replay a fixed, label-blind score-shift gate on archived selector cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def shift(calibration: np.ndarray, test: np.ndarray, quantile: float) -> float:
    return float(np.quantile(logit(test), quantile)
                 - np.quantile(logit(calibration), quantile))


def make_row(source: str, target: str, seed: int, rep: int, fraction: float,
             boost: float, shift_value: float, threshold: float,
             scorecap_hits: int, chemdep_hits: int, rf_hits: int | None) -> dict:
    use_chemdep = shift_value >= threshold
    return {
        "source": source, "target": target, "scorer_seed": seed,
        "rep": rep, "fraction": fraction, "boost": boost,
        "score_shift": shift_value, "gate_chemdep": int(use_chemdep),
        "scorecap_hits": scorecap_hits, "chemdep_hits": chemdep_hits,
        "gate_hits": chemdep_hits if use_chemdep else scorecap_hits,
        "rf_hits": rf_hits,
    }


def synthetic_rows(root: Path, source: str, threshold: float, quantile: float,
                   seed_range: range) -> list[dict]:
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from train_artifact_meta_ranker_20261003 import cap_one, load_panel

    rows = []
    for seed in seed_range:
        if source == "fifth_synthetic":
            score_dir = (root / "data/chembl_meta_ungated_fifth_xgb_20261003"
                         / f"seed{seed}" / "scores")
            paths = sorted(score_dir.glob("*_scores.csv"))
            assert len(paths) == 10
            archived = pd.read_csv(root / "data/pr_fifth_multifp_rf_posthoc_20261005"
                                   / f"seed{seed}_cells.csv")
            start_seed = 355700
        else:
            cohort = root / "data/chembl_modern_eighth_20261005"
            eligible = pd.read_csv(cohort / "technical_eligibility.csv")
            ids = eligible.loc[eligible.eligible, "target_chembl_id"].tolist()
            assert len(ids) == 30
            score_dir = cohort / f"eligible_seed{seed}" / "scores"
            paths = [score_dir / f"{target}_scores.csv" for target in ids]
            assert all(path.is_file() for path in paths)
            archived = pd.read_csv(cohort / "fixed_selector_results" / f"seed{seed}.csv")
            start_seed = 358700
        expected = {}
        for row in archived.itertuples(index=False):
            key = (row.target, int(row.rep), float(row.fraction), float(row.boost))
            if source == "fifth_synthetic":
                assert key not in expected
                assert row.selected == row.scaffolds == 50
                expected[key] = (int(row.scorecap_hits), int(row.chemdep_hits),
                                 int(row.multifp_rf_hits))
            else:
                assert row.selected == row.scaffolds == 50
                expected.setdefault(key, {})[row.method] = int(row.hits)
        assert len(expected) == len(paths) * 40

        for index, path in enumerate(paths):
            panel = load_panel(path)
            target, cal, test = panel["target"], panel["cal"], panel["test"]
            cal_scores = cal.score.to_numpy(float)
            base = test.score.to_numpy(float)
            labels = test.label.to_numpy(int)
            blocks = panel["test_blocks"]
            target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
            target_seed = start_seed + index * 100_000 + target_hash
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
                        key = (target, rep, fraction, boost)
                        if source == "fifth_synthetic":
                            score_hits, chemdep_hits, rf_hits = expected[key]
                        else:
                            hits = expected[key]
                            assert set(hits) == {"score_cap1", "chemdep_cal", "multi_fp_rf",
                                                 "augmented_13view_rf"}
                            score_hits = hits["score_cap1"]
                            chemdep_hits = hits["chemdep_cal"]
                            rf_hits = hits["multi_fp_rf"]
                        replay = int(labels[cap_one(scores, blocks, 50)].sum())
                        assert replay == score_hits, (source, seed, key, replay, score_hits)
                        rows.append(make_row(
                            source, target, seed, rep, fraction, boost,
                            shift(cal_scores, scores, quantile), threshold,
                            score_hits, chemdep_hits, rf_hits))
        print(source, seed, "replayed", len(paths) * 40, "cells", flush=True)
    return rows


def natural_rows(root: Path, threshold: float, quantile: float) -> list[dict]:
    from train_artifact_meta_ranker_20261003 import cap_one, load_panel

    archive = pd.read_csv(root / "data/pr_natural_modern_selectors_20261004"
                          / "pooled20.csv")
    outcomes = {}
    for row in archive.itertuples(index=False):
        if row.method in {"score_cap1", "chemdep_cal"}:
            assert row.selected == row.scaffolds == 50
            outcomes.setdefault((row.dataset, int(row.scorer_seed)), {})[
                row.method] = int(row.hits)
    assert len(outcomes) == 20
    rows = []
    score_dir = root / "data/public_moleculenet_ecfp_xgb/scores"
    for (dataset, seed), hits in sorted(outcomes.items()):
        path = score_dir / f"{dataset}_scaffold_ood_seed{seed}_scores.csv"
        panel = load_panel(path)
        cal, test = panel["cal"], panel["test"]
        scores = test.score.to_numpy(float)
        replay = int(test.label.to_numpy(int)[cap_one(scores, panel["test_blocks"], 50)].sum())
        assert replay == hits["score_cap1"], (dataset, seed, replay, hits)
        rows.append(make_row(
            "moleculenet_natural", dataset, seed, -1, 0., 0.,
            shift(cal.score.to_numpy(float), scores, quantile), threshold,
            hits["score_cap1"], hits["chemdep_cal"], None))
    return rows


def summarize(frame: pd.DataFrame) -> list[dict]:
    result = []
    for source, group in frame.groupby("source"):
        targets = group.groupby("target")
        target_macro = targets[["scorecap_hits", "chemdep_hits", "gate_hits",
                                "rf_hits"]].mean().mean()
        result.append({
            "source": source, "targets": int(group.target.nunique()),
            "scorer_seeds": int(group.scorer_seed.nunique()),
            "cells": len(group), "gate_trigger_rate": float(group.gate_chemdep.mean()),
            "gate_triggered_cells": int(group.gate_chemdep.sum()),
            "target_macro_hits_at_50": {key: (None if pd.isna(value) else float(value))
                                        for key, value in target_macro.items()},
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metric", choices=("q95_logit_shift", "q99_logit_shift"),
                        default="q99_logit_shift")
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    threshold_path = root / "data/label_blind_artifact_gate_probe_20261004/threshold_audit.csv"
    audit = pd.read_csv(threshold_path)
    threshold = float(audit.set_index("metric").loc[args.metric, "threshold"])
    quantile = .95 if args.metric == "q95_logit_shift" else .99
    rows = natural_rows(root, threshold, quantile)
    rows += synthetic_rows(root, "fifth_synthetic", threshold, quantile,
                           range(35501, 35506))
    rows += synthetic_rows(root, "eighth_synthetic", threshold, quantile,
                           range(35801, 35806))
    frame = pd.DataFrame(rows)
    assert len(frame) == 20 + 2000 + 6000
    output = root / f"data/pr_minimum_gate_policy_20261005_{args.metric}"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "cells.csv", index=False)
    report = {
        "status": "retrospective development, not independent confirmation or SOTA",
        "metric": args.metric,
        "threshold": threshold,
        "threshold_file_sha256": hashlib.sha256(threshold_path.read_bytes()).hexdigest(),
        "test_label_use": "synthetic artifact generation, score-cap replay QA, hit counting; never gate",
        "summary": summarize(frame),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
