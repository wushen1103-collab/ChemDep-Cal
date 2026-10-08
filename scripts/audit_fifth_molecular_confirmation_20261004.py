#!/usr/bin/env python3
"""Post hoc strong molecular-confirmation comparator on the frozen fifth cohort."""

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
from rdkit.Chem import rdFingerprintGenerator
from xgboost import XGBClassifier

from evaluate_mfpcba_twenty_20261004 import molecular_features


def run_seed(root_s: str, seed: int) -> int:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from train_artifact_meta_ranker_20261003 import cap_one, load_panel

    RDLogger.DisableLog("rdApp.*")
    paths = sorted((root / "data/chembl_meta_ungated_fifth_xgb_20261003"
                    / f"seed{seed}" / "scores").glob("*_scores.csv"))
    if len(paths) != 10:
        raise AssertionError("Expected the original ten frozen targets")
    panels = [load_panel(path) for path in paths]
    reference = pd.read_csv(root / "data/conditional_sota_meta_ungated_fifth_20261003"
                            / f"meta_seed{seed}.csv")
    reference = reference[reference.budget.eq(50) & reference.method.isin(
        ("meta_calibration", "score_cap1"))]
    expected = {(r.target, int(r.rep), float(r.fraction), float(r.boost), r.method): int(r.hits)
                for r in reference.itertuples(index=False)}
    if len(expected) != 800:
        raise AssertionError("Frozen reference incomplete")
    output = root / "data/pr_fifth_molecular_confirmation_posthoc_20261004"
    output.mkdir(parents=True, exist_ok=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows = []
    for panel_index, panel in enumerate(panels):
        target = panel["target"]
        cal, test = panel["cal"], panel["test"]
        ecfp_cal = molecular_features(cal.canonical_smiles, generator)
        ecfp_test = molecular_features(test.canonical_smiles, generator)
        x_cal = np.column_stack([ecfp_cal, cal.score.to_numpy(float)])
        labels_cal = cal.label.to_numpy(int)
        positives = int(labels_cal.sum())
        if positives < 2 or positives == len(labels_cal):
            raise AssertionError(f"Unfit target calibration: {target}")
        model = XGBClassifier(
            n_estimators=160, max_depth=3, learning_rate=.05,
            subsample=.9, colsample_bytree=.9, min_child_weight=8,
            reg_lambda=4., tree_method="hist", eval_metric="logloss",
            n_jobs=4, random_state=862026,
            scale_pos_weight=min(20., max(1., (len(labels_cal) - positives) / positives)),
        ).fit(x_cal, labels_cal)
        model.save_model(output / f"seed{seed}.{target}.ecfp_score.json")
        base = test.score.to_numpy(float)
        labels = test.label.to_numpy(int)
        blocks = panel["test_blocks"]
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
                        test, base, artifacts,
                        artifact_block_col="murcko_scaffold", logit_boost=boost)[0]
                    score_hits = int(labels[cap_one(scores, blocks, 50)].sum())
                    key = (target, rep, fraction, boost)
                    if score_hits != expected[(*key, "score_cap1")]:
                        raise AssertionError(f"Frozen score-cap replay mismatch: {key}")
                    x_test = np.column_stack([ecfp_test, scores])
                    molecular = model.predict_proba(x_test)[:, 1]
                    chosen = cap_one(molecular, blocks, 50)
                    if len(chosen) != 50 or len(set(blocks[chosen])) != 50:
                        raise AssertionError("Budget or scaffold capacity violation")
                    rows.append({"target": target, "seed": seed, "rep": rep,
                                 "fraction": fraction, "boost": boost,
                                 "chemdep_hits": expected[(*key, "meta_calibration")],
                                 "scorecap_hits": score_hits,
                                 "ecfp_score_hits": int(labels[chosen].sum()),
                                 "selected": len(chosen), "scaffolds": len(set(blocks[chosen]))})
        print("seed", seed, "target", target, "completed", flush=True)
    result = pd.DataFrame(rows)
    if len(result) != 400:
        raise AssertionError("Incomplete fifth-cohort molecular grid")
    result.to_csv(output / f"seed{seed}_cells.csv", index=False)
    (output / f"seed{seed}_manifest.json").write_text(json.dumps({
        "status": "post hoc strong comparator on inspected fifth cohort; not independent confirmation",
        "seed": seed, "score_file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in paths},
        "frozen_scorecap_cells_replayed": 400,
        "frozen_chemdep_hits_loaded": 400,
        "test_label_use": "original label-aware artifact construction and final hit evaluation",
        "classifier": "per-target ECFP4 2048 bits plus perturbed score; identical calibration labels",
    }, indent=2) + "\n", encoding="utf-8")
    return len(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        print(list(pool.map(run_seed, [str(args.root.resolve())] * 5,
                            range(35501, 35506))), flush=True)


if __name__ == "__main__":
    main()
