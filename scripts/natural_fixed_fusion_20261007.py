#!/usr/bin/env python3
"""Evaluate the already fixed 75:25 fusion on natural MoleculeNet panels."""

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
from scipy.stats import rankdata
from xgboost import XGBClassifier

from develop_mfpcba_multifp_rf_20261004 import fingerprints, fit_rf


def evaluate_panel(root_s: str, path_s: str) -> list[dict]:
    root, path = Path(root_s), Path(path_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    import probe_meta_band_v2_20261003  # Register the frozen feature map.
    from probe_meta_band_v2_20261003 import evidence_features
    from train_artifact_meta_ranker_20261003 import cap_one, load_panel

    RDLogger.DisableLog("rdApp.*")
    panel = load_panel(path)
    cal, test = panel["cal"], panel["test"]
    seed = int(path.stem.rsplit("_seed", 1)[1].split("_")[0])
    model_path = root / "data/pr_natural_modern_selectors_20261004/pooled20.chemdep.json"
    chemdep = XGBClassifier()
    chemdep.load_model(model_path)
    rf = fit_rf(
        np.column_stack([fingerprints(cal.canonical_smiles), cal.score.to_numpy(float)]),
        cal.label.to_numpy(int), seed,
    )
    scores = test.score.to_numpy(float)
    blocks = panel["test_blocks"]
    labels = test.label.to_numpy(int)
    cp = chemdep.predict_proba(evidence_features(scores, blocks, panel["null"]))[:, 1]
    rp = rf.predict_proba(np.column_stack([fingerprints(test.canonical_smiles), scores]))[:, 1]
    fusion = .75 * rankdata(rp, method="average") / len(rp)
    fusion += .25 * rankdata(cp, method="average") / len(cp)
    dataset = str(pd.read_csv(path, nrows=1).dataset.iloc[0])
    rows = []
    for method, values in (
        ("score_cap1", scores), ("chemdep_cal", cp),
        ("multi_fp_rf", rp), ("fixed_75_25_fusion", fusion),
    ):
        chosen = cap_one(values, blocks, 50)
        if len(chosen) != 50 or len(set(blocks[chosen])) != 50:
            raise AssertionError(f"Budget or cap violation: {dataset}/{seed}/{method}")
        rows.append({"dataset": dataset, "target": panel["target"],
                     "scorer_seed": seed, "method": method,
                     "hits": int(labels[chosen].sum()), "selected": 50,
                     "scaffolds": 50, "testpool_size": len(test)})
    reference = pd.read_csv(root / "data/pr_natural_modern_selectors_20261004/pooled20.csv")
    for row in rows[:2]:
        expected = reference.loc[
            (reference.target == row["target"])
            & (reference.method == row["method"]), "hits"
        ]
        if len(expected) != 1 or row["hits"] != int(expected.iloc[0]):
            raise AssertionError(f"Natural reference replay mismatch: {row}")
    print(f"completed {dataset} seed{seed}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    paths = sorted((root / "data/public_moleculenet_ecfp_xgb/scores").glob(
        "*scaffold_ood_seed*_scores.csv"))
    if len(paths) != 20:
        raise AssertionError(f"Expected 20 natural panels, found {len(paths)}")
    with futures.ProcessPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(evaluate_panel, [str(root)] * 20, map(str, paths)))
    cells = pd.DataFrame([row for group in groups for row in group])
    if len(cells) != 80 or cells.groupby(["dataset", "method"]).size().ne(5).any():
        raise AssertionError("Incomplete natural dataset/seed grid")
    output = root / "data/pr_natural_fixed_fusion_20261007"
    output.mkdir(parents=True, exist_ok=True)
    cells.to_csv(output / "cells.csv", index=False)
    summary = cells.groupby(["dataset", "method"]).hits.agg(["mean", "std"]).reset_index()
    summary.to_csv(output / "dataset_mean_seed_sd.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "status": "post hoc fixed-weight natural-score transfer on inspected panels",
        "rf_weight": .75, "chemdep_weight": .25,
        "calibration": "RF target- and scorer-seed-specific; ChemDep pooled over 20 natural calibration panels",
        "test_label_use": "replay QA and final hit count only",
        "panels_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
