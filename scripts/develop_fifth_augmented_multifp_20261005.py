#!/usr/bin/env python3
"""Matched fifth-cohort development test of calibration-score augmentation."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

from develop_mfpcba_multifp_rf_20261004 import fingerprints, fit_rf


def augmented_views(panel: dict, evidence_features, choose_artifact_blocks, inject_artifact):
    cal = panel["cal"]
    base = cal.score.to_numpy(float)
    labels = cal.label.to_numpy(int)
    blocks, null = panel["cal_blocks"], panel["null"]
    views = [evidence_features(base, blocks, null)]
    target_hash = int(hashlib.sha256(panel["target"].encode()).hexdigest()[:12], 16) % 1_000_000
    for rep in range(3):
        for fraction in (.2, .3):
            rng = np.random.default_rng(852026 + target_hash + rep * 1009 + int(fraction * 10000))
            artifacts = choose_artifact_blocks(
                cal, artifact_block_col="murcko_scaffold", rng=rng,
                artifact_fraction=fraction, max_artifact_blocks=25,
                min_inactive_per_block=2)
            for boost in (4., 6.):
                changed = inject_artifact(
                    cal, base, artifacts, artifact_block_col="murcko_scaffold",
                    logit_boost=boost)[0]
                views.append(evidence_features(changed, blocks, null))
    matrix = np.vstack(views)
    y = np.tile(labels, len(views))
    if matrix.shape != (13 * len(cal), 16) or len(y) != len(matrix):
        raise AssertionError("Augmented calibration views misaligned")
    if not np.allclose(matrix[:len(cal), 0], base):
        raise AssertionError("First evidence feature must be original score")
    return matrix, y


def run_seed(root_s: str, seed: int, limit: int | None) -> int:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from train_artifact_meta_ranker_20261003 import cap_one, load_panel

    RDLogger.DisableLog("rdApp.*")
    paths = sorted((root / "data/chembl_meta_ungated_fifth_xgb_20261003"
                    / f"seed{seed}" / "scores").glob("*_scores.csv"))
    if len(paths) != 10:
        raise AssertionError("Expected ten fifth-cohort targets")
    panels = [load_panel(path) for path in paths[:limit]]
    reference = pd.read_csv(root / "data/conditional_sota_meta_ungated_fifth_20261003"
                            / f"meta_seed{seed}.csv")
    reference = reference[reference.budget.eq(50) & reference.method.eq("score_cap1")]
    expected = {(r.target, int(r.rep), float(r.fraction), float(r.boost)): int(r.hits)
                for r in reference.itertuples(index=False)}
    rf_reference = pd.read_csv(root / "data/pr_fifth_multifp_rf_posthoc_20261005"
                               / f"seed{seed}_cells.csv")
    rf_expected = {(r.target, int(r.rep), float(r.fraction), float(r.boost)):
                   int(r.multifp_rf_hits) for r in rf_reference.itertuples(index=False)}
    output = root / "data/pr_fifth_augmented_multifp_development_20261005"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for panel_index, panel in enumerate(panels):
        target, cal, test = panel["target"], panel["cal"], panel["test"]
        fp_cal = fingerprints(cal.canonical_smiles)
        fp_test = fingerprints(test.canonical_smiles)
        context, y = augmented_views(
            panel, evidence_features, choose_artifact_blocks, inject_artifact)
        fp_aug = np.tile(fp_cal, (13, 1))
        score_model = fit_rf(np.column_stack([fp_aug, context[:, 0]]), y, seed)
        context_model = fit_rf(np.column_stack([fp_aug, context]), y, seed)
        del fp_aug, fp_cal
        base, labels = test.score.to_numpy(float), test.label.to_numpy(int)
        blocks, null = panel["test_blocks"], panel["null"]
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = 355700 + panel_index * 100_000 + target_hash
        for rep in range(10):
            for fraction in (.2, .3):
                rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
                artifacts = choose_artifact_blocks(
                    test, artifact_block_col="murcko_scaffold", rng=rng,
                    artifact_fraction=fraction, max_artifact_blocks=25,
                    min_inactive_per_block=2)
                for boost in (4., 6.):
                    scores = inject_artifact(
                        test, base, artifacts, artifact_block_col="murcko_scaffold",
                        logit_boost=boost)[0]
                    key = (target, rep, fraction, boost)
                    cap_hits = int(labels[cap_one(scores, blocks, 50)].sum())
                    if cap_hits != expected[key]:
                        raise AssertionError(f"Frozen score-cap replay mismatch: {key}")
                    test_context = evidence_features(scores, blocks, null)
                    p_score = score_model.predict_proba(
                        np.column_stack([fp_test, scores]))[:, 1]
                    p_context = context_model.predict_proba(
                        np.column_stack([fp_test, test_context]))[:, 1]
                    rows.append({
                        "target": target, "seed": seed, "rep": rep,
                        "fraction": fraction, "boost": boost,
                        "score_cap1_hits": cap_hits,
                        "multifp_rf_hits": rf_expected[key],
                        "augmented_score_rf_hits": int(labels[cap_one(p_score, blocks, 50)].sum()),
                        "augmented_context_rf_hits": int(labels[cap_one(p_context, blocks, 50)].sum()),
                    })
        print("seed", seed, "target", target, "completed", flush=True)
    cells = pd.DataFrame(rows)
    if len(cells) != len(panels) * 40 or not cells.notna().all().all():
        raise AssertionError("Incomplete evaluation")
    suffix = "" if limit is None else f"_first{limit}"
    cells.to_csv(output / f"seed{seed}{suffix}_cells.csv", index=False)
    (output / f"seed{seed}{suffix}_manifest.json").write_text(json.dumps({
        "status": "exposed fifth-cohort development only; not independent SOTA evidence",
        "seed": seed, "targets": [p["target"] for p in panels],
        "calibration_unique_labels_only": True,
        "augmentation_views_per_calibration_molecule": 13,
        "models": ["augmented_score_rf", "augmented_context_rf"],
        "baseline": "unchanged original multiFP RF hits reloaded by exact cell key",
    }, indent=2) + "\n")
    return len(cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(35501, 35506)))
    parser.add_argument("--limit-targets", type=int)
    args = parser.parse_args()
    with futures.ProcessPoolExecutor(max_workers=min(5, len(args.seeds))) as pool:
        print(list(pool.map(run_seed, [str(args.root.resolve())] * len(args.seeds),
                            args.seeds, [args.limit_targets] * len(args.seeds))), flush=True)


if __name__ == "__main__":
    main()
