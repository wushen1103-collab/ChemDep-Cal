#!/usr/bin/env python3
"""Post hoc same-cell MMR and supervised LTR controls for ChemDep-Cal."""

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
from xgboost import XGBClassifier, XGBRanker


METHODS = ("chemdep_cal", "score_cap1", "mmr_cap1_l05", "mmr_cap1_l08",
           "mmr_cap1_l095", "xgb_pairwise_cap1")


def similarity_matrix(smiles: list[str]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    for item in smiles:
        mol = Chem.MolFromSmiles(item)
        if mol is None:
            raise ValueError(f"Invalid frozen test SMILES: {item}")
        fingerprints.append(generator.GetFingerprint(mol))
    matrix = np.eye(len(fingerprints), dtype=np.float32)
    for index in range(1, len(fingerprints)):
        sims = DataStructs.BulkTanimotoSimilarity(fingerprints[index], fingerprints[:index])
        matrix[index, :index] = sims
        matrix[:index, index] = sims
    return matrix


def mmr_cap_one(scores: np.ndarray, blocks: np.ndarray, similarity: np.ndarray,
                budget: int, lam: float) -> np.ndarray:
    if not 0 < lam <= 1:
        raise ValueError(lam)
    sigmoid_scores = 1.0 / (1.0 + np.exp(-scores))
    available = np.ones(len(scores), dtype=bool)
    max_similarity = np.zeros(len(scores), dtype=np.float32)
    selected = []
    while len(selected) < budget and available.any():
        if not selected:
            index = int(np.argmax(np.where(available, scores, -np.inf)))
        else:
            mmr = lam * sigmoid_scores - (1 - lam) * max_similarity
            index = int(np.argmax(np.where(available, mmr, -np.inf)))
        selected.append(index)
        available[blocks == blocks[index]] = False
        np.maximum(max_similarity, similarity[:, index], out=max_similarity)
    return np.asarray(selected, dtype=int)


def fit_pairwise(panels: list[dict]) -> XGBRanker:
    from probe_meta_band_20261003 import training_matrix

    features, labels, query_ids = [], [], []
    for target_index, panel in enumerate(panels):
        x, y = training_matrix([panel], repeats=3)
        view_size = len(panel["cal"])
        if len(x) != view_size * 13 or x.shape[1] != 16:
            raise AssertionError("Unexpected calibration view geometry")
        features.append(x)
        labels.append(y)
        query_ids.append(np.repeat(np.arange(13) + target_index * 13, view_size))
    model = XGBRanker(
        objective="rank:pairwise", n_estimators=160, max_depth=3,
        learning_rate=.05, subsample=.9, colsample_bytree=.9,
        min_child_weight=8, reg_lambda=4.0, tree_method="hist",
        n_jobs=4, random_state=862026,
    )
    model.fit(np.concatenate(features), np.concatenate(labels), qid=np.concatenate(query_ids))
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]

    import probe_meta_band_v2_20261003  # Registers the frozen feature map.
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from train_artifact_meta_ranker_20261003 import cap_one, load_panel

    score_dir = root / "data/chembl_meta_ungated_fifth_xgb_20261003" / f"seed{args.seed}" / "scores"
    paths = sorted(score_dir.glob("*_scores.csv"))
    if len(paths) != 10:
        raise AssertionError(f"Expected ten targets, found {len(paths)}")
    panels = [load_panel(path) for path in paths]
    frozen_dir = root / "data/conditional_sota_meta_ungated_fifth_20261003"
    reference = pd.read_csv(frozen_dir / f"meta_seed{args.seed}.csv")
    reference = reference[reference.budget.eq(50) & reference.method.isin(("meta_calibration", "score_cap1"))]
    expected = {(r.target, int(r.rep), float(r.fraction), float(r.boost), r.method): int(r.hits)
                for r in reference.itertuples(index=False)}
    if len(expected) != 800:
        raise AssertionError("Frozen replay reference is incomplete")

    pooled_model = XGBClassifier()
    pooled_model.load_model(frozen_dir / f"meta_seed{args.seed}.model.json")
    pairwise_model = fit_pairwise(panels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pairwise_model.save_model(args.out.with_suffix(".pairwise.json"))

    rows = []
    for panel_index, panel in enumerate(panels):
        test = panel["test"]
        target = panel["target"]
        blocks = panel["test_blocks"]
        labels = test.label.to_numpy(int)
        base = test.score.to_numpy(float)
        similarity = similarity_matrix(test.canonical_smiles.astype(str).tolist())
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = 355700 + panel_index * 100_000 + target_hash
        for rep in range(10):
            for fraction in (.2, .3):
                rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
                artifacts = choose_artifact_blocks(
                    test, artifact_block_col="murcko_scaffold", rng=rng,
                    artifact_fraction=fraction, max_artifact_blocks=25,
                    min_inactive_per_block=2,
                )
                for boost in (4., 6.):
                    scores = inject_artifact(
                        test, base, artifacts, artifact_block_col="murcko_scaffold",
                        logit_boost=boost,
                    )[0]
                    x_test = evidence_features(scores, blocks, panel["null"])
                    pooled_scores = pooled_model.predict_proba(x_test)[:, 1]
                    pairwise_scores = pairwise_model.predict(x_test)
                    selections = {
                        "chemdep_cal": cap_one(pooled_scores, blocks, 50),
                        "score_cap1": cap_one(scores, blocks, 50),
                        "mmr_cap1_l05": mmr_cap_one(scores, blocks, similarity, 50, .5),
                        "mmr_cap1_l08": mmr_cap_one(scores, blocks, similarity, 50, .8),
                        "mmr_cap1_l095": mmr_cap_one(scores, blocks, similarity, 50, .95),
                        "xgb_pairwise_cap1": cap_one(pairwise_scores, blocks, 50),
                    }
                    for method, chosen in selections.items():
                        if len(chosen) != 50 or len(set(blocks[chosen])) != 50:
                            raise AssertionError(f"Invalid budget/capacity: {target} {method}")
                        hits = int(labels[chosen].sum())
                        frozen_name = {"chemdep_cal": "meta_calibration", "score_cap1": "score_cap1"}.get(method)
                        if frozen_name and hits != expected[(target, rep, fraction, boost, frozen_name)]:
                            raise AssertionError(f"Frozen replay mismatch: {target} {method}")
                        rows.append({"target": target, "scorer_seed": args.seed, "rep": rep,
                                     "fraction": fraction, "boost": boost, "method": method,
                                     "hits": hits, "selected": 50, "scaffolds": 50,
                                     "fdp": (50 - hits) / 50})
        print(f"completed {target}", flush=True)

    result = pd.DataFrame(rows)
    if len(result) != 10 * 10 * 4 * len(METHODS):
        raise AssertionError("Incomplete method grid")
    result.to_csv(args.out, index=False)
    args.out.with_suffix(".manifest.json").write_text(json.dumps({
        "status": "post hoc on previously inspected fifth cohort",
        "scorer_seed": args.seed,
        "methods": list(METHODS),
        "mmr_source": "ScaffAug Algorithm 2, arXiv:2510.16306v2; reranking module only",
        "mmr_adaptation": "same frozen scores; full test pool (all fewer than 500); ECFP4 Tanimoto; added Murcko cap-1",
        "mmr_lambdas": [0.5, 0.8, 0.95],
        "pairwise": "XGBRanker rank:pairwise, pooled labelled calibration, 13 views per target, same 16 features and cap-1",
        "frozen_replay_rows_checked": 800,
        "score_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }, indent=2) + "\n", encoding="utf-8")
    print(result.groupby("method").hits.mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
