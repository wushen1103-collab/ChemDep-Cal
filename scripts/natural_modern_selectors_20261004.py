#!/usr/bin/env python3
"""Same-panel modern selector controls on four natural scaffold-OOD datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


METHODS = ("chemdep_cal", "xgb_pairwise_cap1", "score_cap1", "mmr_cap1_l095")
FP_COL = "ecfp4_radius2_nbits2048"


def fingerprints(frame: pd.DataFrame) -> list:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    result = []
    for smiles in frame.canonical_smiles.astype(str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise AssertionError("Invalid canonical SMILES in frozen panel")
        result.append(generator.GetFingerprint(mol))
    stored = [DataStructs.CreateFromBinaryText(bytes.fromhex(value))
              for value in frame[FP_COL].astype(str).iloc[:20]]
    for index in range(min(20, len(frame))):
        expected = DataStructs.BulkTanimotoSimilarity(stored[index], stored)
        actual = DataStructs.BulkTanimotoSimilarity(result[index], result[:len(stored)])
        if not np.allclose(expected, actual, atol=1e-7):
            raise AssertionError("Stored and regenerated ECFP4 similarities differ")
    return result


def mmr_cap_one(scores: np.ndarray, blocks: np.ndarray, fps: list,
                budget: int = 50, lam: float = .95) -> np.ndarray:
    available = np.ones(len(scores), dtype=bool)
    sigmoid = 1.0 / (1.0 + np.exp(-scores))
    max_sim = np.zeros(len(scores), dtype=np.float32)
    chosen = []
    while len(chosen) < budget and available.any():
        values = scores if not chosen else lam * sigmoid - (1.0 - lam) * max_sim
        index = int(np.argmax(np.where(available, values, -np.inf)))
        chosen.append(index)
        available[blocks == blocks[index]] = False
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[index], fps),
                                  dtype=np.float32)
        np.maximum(max_sim, similarities, out=max_sim)
    return np.asarray(chosen, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]

    import probe_meta_band_v2_20261003  # Registers the original 16-feature map.
    from modern_selector_benchmark_20261004 import fit_pairwise
    from probe_meta_band_20261003 import training_matrix
    from probe_meta_band_v2_20261003 import evidence_features
    from train_artifact_meta_ranker_20261003 import cap_one, fit_ranker, load_panel

    score_dir = root / "data/public_moleculenet_ecfp_xgb/scores"
    paths = sorted(score_dir.glob("*scaffold_ood_seed*_scores.csv"))
    if len(paths) != 20:
        raise AssertionError(f"Expected 20 frozen natural panels, found {len(paths)}")
    reference_path = root / "data/conditional_sota_meta_ungated_fifth_20261003/moleculenet_natural_ood.csv"
    reference = pd.read_csv(reference_path)
    reference = reference[reference.method.isin(("meta_calibration", "score_cap1"))]
    expected = {(row.target, row.method): int(row.hits)
                for row in reference.itertuples(index=False)}
    if len(expected) != 40:
        raise AssertionError("Original natural-OOD reference is incomplete")

    panels = [load_panel(path) for path in paths]
    x, y = training_matrix(panels, repeats=3)
    full_model = fit_ranker(x, y, 862026)
    pairwise_model = fit_pairwise(panels)
    out_dir = root / "data/pr_natural_modern_selectors_20261004"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_model.save_model(out_dir / "pooled20.chemdep.json")
    pairwise_model.save_model(out_dir / "pooled20.pairwise.json")

    rows = []
    for path, panel in zip(paths, panels):
        test = panel["test"]
        labels = test.label.to_numpy(int)
        blocks = panel["test_blocks"]
        scores = test.score.to_numpy(float)
        features = evidence_features(scores, blocks, panel["null"])
        fps = fingerprints(test)
        selections = {
            "chemdep_cal": cap_one(full_model.predict_proba(features)[:, 1], blocks, 50),
            "xgb_pairwise_cap1": cap_one(pairwise_model.predict(features), blocks, 50),
            "score_cap1": cap_one(scores, blocks, 50),
            "mmr_cap1_l095": mmr_cap_one(scores, blocks, fps),
        }
        dataset = str(pd.read_csv(path, nrows=1).dataset.iloc[0])
        target = panel["target"]
        scorer_seed = int(path.stem.rsplit("_seed", 1)[1].split("_")[0])
        for method, chosen in selections.items():
            if len(chosen) != 50 or len(set(blocks[chosen])) != 50:
                raise AssertionError(f"Capacity mismatch: {target} {method}")
            hits = int(labels[chosen].sum())
            frozen_method = {"chemdep_cal": "meta_calibration", "score_cap1": "score_cap1"}.get(method)
            if frozen_method and hits != expected[(target, frozen_method)]:
                raise AssertionError(f"Natural frozen replay mismatch: {target} {method}")
            rows.append({"dataset": dataset, "target": target, "scorer_seed": scorer_seed,
                         "method": method, "hits": hits, "selected": 50, "scaffolds": 50,
                         "fdp": (50 - hits) / 50, "testpool_size": len(test),
                         "calibration_active": int(panel["cal"].label.sum())})
        print(f"completed {dataset} seed{scorer_seed}", flush=True)

    result = pd.DataFrame(rows)
    if len(result) != 20 * len(METHODS):
        raise AssertionError("Incomplete method-by-dataset grid")
    result.to_csv(out_dir / "pooled20.csv", index=False)
    (out_dir / "pooled20.manifest.json").write_text(json.dumps({
        "status": "post hoc on previously inspected natural scaffold-OOD panels",
        "methods": METHODS,
        "ranker_training": "one model per method pooled over all 20 labelled calibration panels, matching original natural reference",
        "scorer_seeds": [3500, 3501, 3502, 3503, 3504],
        "frozen_replay_rows_checked": 40,
        "score_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")
    print(result.pivot(index=["dataset", "scorer_seed"], columns="method", values="hits").to_string(), flush=True)


if __name__ == "__main__":
    main()
