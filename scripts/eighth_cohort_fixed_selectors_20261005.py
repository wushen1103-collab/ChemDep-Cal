#!/usr/bin/env python3
"""Frozen matched eighth-cohort clean/augmented RF and ChemDep comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

from develop_fifth_augmented_multifp_20261005 import augmented_views
from develop_mfpcba_multifp_rf_20261004 import fingerprints, fit_rf


METHODS = ("score_cap1", "chemdep_cal", "multi_fp_rf", "augmented_13view_rf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.seed not in range(35801, 35806):
        raise ValueError("Scorer seed not in frozen range")
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    import probe_meta_band_v2_20261003  # noqa: F401; activate 16-feature training map
    from probe_meta_band_20261003 import training_matrix
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from train_artifact_meta_ranker_20261003 import cap_one, fit_ranker, load_panel

    RDLogger.DisableLog("rdApp.*")
    cohort = root / "data/chembl_modern_eighth_20261005"
    ledger = pd.read_csv(cohort / "technical_eligibility.csv")
    if len(ledger) != 100:
        raise AssertionError("All 100 metadata-fixed targets must appear in eligibility ledger")
    eligible = ledger.loc[ledger.eligible, "target_chembl_id"].tolist()
    paths = [cohort / f"eligible_seed{args.seed}" / "scores"
             / f"{target}_scores.csv" for target in eligible]
    if not eligible or not all(path.is_file() for path in paths):
        raise AssertionError("Missing fixed eligible scorer panel")
    if set(paths[0].parent.glob("*_scores.csv")) != set(paths):
        raise AssertionError("Unexpected scorer panel")
    panels = [load_panel(path) for path in paths]
    x, y = training_matrix(panels, repeats=3)
    chemdep = fit_ranker(x, y, 862026)
    out = cohort / "fixed_selector_results"
    out.mkdir(parents=True, exist_ok=True)
    chemdep_path = out / f"seed{args.seed}.chemdep.json"
    chemdep.save_model(chemdep_path)
    rows = []
    for index, panel in enumerate(panels):
        target, cal, test = panel["target"], panel["cal"], panel["test"]
        fp_cal, fp_test = fingerprints(cal.canonical_smiles), fingerprints(test.canonical_smiles)
        rf = fit_rf(np.column_stack([fp_cal, cal.score.to_numpy(float)]),
                    cal.label.to_numpy(int), args.seed)
        views, labels = augmented_views(
            panel, evidence_features, choose_artifact_blocks, inject_artifact)
        aug_rf = fit_rf(np.column_stack([np.tile(fp_cal, (13, 1)), views[:, 0]]),
                        labels, args.seed)
        base, truth = test.score.to_numpy(float), test.label.to_numpy(int)
        blocks, null = panel["test_blocks"], panel["null"]
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = 358700 + index * 100_000 + target_hash
        for rep in range(10):
            for fraction in (.2, .3):
                rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
                artifacts = choose_artifact_blocks(
                    test, artifact_block_col="murcko_scaffold", rng=rng,
                    artifact_fraction=fraction, max_artifact_blocks=25,
                    min_inactive_per_block=2)
                for boost in (4., 6.):
                    scores = inject_artifact(
                        test, base, artifacts,
                        artifact_block_col="murcko_scaffold", logit_boost=boost)[0]
                    features = np.column_stack([fp_test, scores])
                    predictions = {
                        "score_cap1": scores,
                        "chemdep_cal": chemdep.predict_proba(
                            evidence_features(scores, blocks, null))[:, 1],
                        "multi_fp_rf": rf.predict_proba(features)[:, 1],
                        "augmented_13view_rf": aug_rf.predict_proba(features)[:, 1],
                    }
                    for method, values in predictions.items():
                        chosen = cap_one(values, blocks, 50)
                        if len(chosen) != 50 or len(set(blocks[chosen])) != 50:
                            raise AssertionError(f"Budget/scaffold violation: {target}/{method}")
                        rows.append({"target": target, "scorer_seed": args.seed,
                                     "rep": rep, "fraction": fraction, "boost": boost,
                                     "method": method, "hits": int(truth[chosen].sum()),
                                     "selected": len(chosen), "scaffolds": len(set(blocks[chosen]))})
        print("seed", args.seed, "target", target, "completed", flush=True)
    cells = pd.DataFrame(rows)
    if len(cells) != len(eligible) * 40 * len(METHODS):
        raise AssertionError("Incomplete eighth selector grid")
    cells.to_csv(out / f"seed{args.seed}.csv", index=False)
    (out / f"seed{args.seed}.manifest.json").write_text(json.dumps({
        "status": "methods fixed before eighth-cohort activity retrieval; label-aware synthetic stress",
        "scorer_seed": args.seed, "eligible_targets": eligible,
        "methods": METHODS, "calibration_views_for_augmented_rf": 13,
        "score_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in paths},
        "chemdep_model_sha256": hashlib.sha256(chemdep_path.read_bytes()).hexdigest(),
        "test_label_use": "artifact generation and final hit counting only",
    }, indent=2) + "\n")
    print(cells.groupby("method").hits.mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
