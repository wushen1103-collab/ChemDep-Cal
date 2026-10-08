#!/usr/bin/env python3
"""Post hoc fixed-weight rank fusion of multi-FP RF and frozen MoLFormer."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from xgboost import XGBClassifier

from develop_mfpcba_multifp_rf_20261004 import fingerprints, fit_rf
from develop_mfpcba_readout_context_20261004 import quality_frame
from evaluate_mfpcba_twenty_20261004 import choose


def run_task(root_s: str, campaign: str, name: str) -> list[dict]:
    root = Path(root_s)
    directory = root / f"data/mfpcba_{campaign}_20261004"
    cohort_file = (directory / f"{name}_cohort.parquet" if campaign == "twenty"
                   else directory / "cohorts" / f"{name}_cohort.parquet")
    cohort = pd.read_parquet(cohort_file)
    fp = fingerprints(cohort.canonical_smiles)
    with np.load(root / "data/mfpcba_molformer_exposed_development_20261005/embeddings.npz") as source:
        lookup = pd.Series(np.arange(len(source["smiles"])), index=source["smiles"])
        positions = lookup.reindex(cohort.canonical_smiles).to_numpy()
        if not np.isfinite(positions).all():
            raise AssertionError("Missing MoLFormer embedding")
        latent = source["embedding"][positions.astype(int)]
    quality, _ = ((np.empty((len(cohort), 0)), []) if campaign == "twenty"
                  else quality_frame(directory, name, cohort))
    reference_file = (root / "data" /
                      ("mfpcba_multifp_rf_twenty_development_20261004" if campaign == "twenty"
                       else "mfpcba_multifp_rf_development_20261004") / "cells.csv")
    reference = pd.read_csv(reference_file)
    rf_method = "multifp_score_rf" if campaign == "twenty" else "multifp_score_readout_rf"
    mol_reference = pd.read_csv(root / f"data/mfpcba_molformer_{campaign}_development_20261005/cells.csv")
    rows = []
    for seed in range(1, 6):
        split_file = (directory / f"{name}_seed{seed}_indices.npz" if campaign == "twenty"
                      else directory / "cohorts" / f"{name}_seed{seed}_indices.npz")
        with np.load(split_file) as split:
            cal_idx, test_idx = split["calibration"], split["test"]
        cal, test = cohort.iloc[cal_idx], cohort.iloc[test_idx]
        center = float(np.median(cal.score))
        scale = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
        score = expit((cohort.score.to_numpy(float) - center) / scale)
        y_cal, y_test = cal.label.to_numpy(int), test.label.to_numpy(int)
        rf_x = np.column_stack([fp, score, quality])
        mol_x = np.column_stack([latent, score, quality])
        rf = fit_rf(rf_x[cal_idx], y_cal, seed).predict_proba(rf_x[test_idx])[:, 1]
        positives = int(y_cal.sum())
        mol = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=.035, subsample=.9,
            colsample_bytree=.9, min_child_weight=8, reg_lambda=4.,
            tree_method="hist", eval_metric="logloss", n_jobs=8,
            random_state=862026 + seed,
            scale_pos_weight=min(20., max(1., (len(y_cal) - positives) / positives)),
        ).fit(mol_x[cal_idx], y_cal).predict_proba(mol_x[test_idx])[:, 1]
        rf_rank = rankdata(rf, method="average") / len(rf)
        mol_rank = rankdata(mol, method="average") / len(mol)
        ties = np.asarray([int.from_bytes(hashlib.sha256(
            f"{name}/{seed}/{smiles}".encode()).digest()[:8], "big")
            for smiles in test.canonical_smiles], dtype=np.uint64)
        blocks = test.murcko_scaffold.where(
            test.murcko_scaffold.ne(""), "singleton:" + test.canonical_smiles).to_numpy(str)
        predictions = {"multi_fp_rf": rf, "frozen_molformer_xgb": mol}
        for weight in (.2, .5, .8):
            predictions[f"rank_fusion_rf_{int(weight * 100)}"] = (
                weight * rf_rank + (1. - weight) * mol_rank)
        for method, values in predictions.items():
            selected = choose(values, blocks, ties)
            if len(selected) != 50 or len(set(blocks[selected])) != 50:
                raise AssertionError("Budget or scaffold-cap violation")
            hits = int(y_test[selected].sum())
            if method == "multi_fp_rf":
                expected = reference.loc[reference.task.eq(name)
                                         & reference.seed.eq(seed)
                                         & reference.method.eq(rf_method), "hits"]
                if len(expected) != 1 or hits != int(expected.iloc[0]):
                    raise AssertionError(f"RF replay mismatch: {name}/{seed}")
            if method == "frozen_molformer_xgb":
                expected = mol_reference.loc[mol_reference.task.eq(name)
                                             & mol_reference.seed.eq(seed), "hits"]
                if len(expected) != 1 or hits != int(expected.iloc[0]):
                    raise AssertionError(f"MoLFormer replay mismatch: {name}/{seed}")
            rows.append({"task": name, "seed": seed, "method": method, "hits": hits})
        print(campaign, name, seed, "completed", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", choices=("twelve_new", "twenty"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / f"data/mfpcba_{args.campaign}_20261004"
                           / "cohort_manifest.json").read_text())
    names = manifest["eligible_tasks"]
    with futures.ProcessPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(run_task, [str(root)] * len(names),
                               [args.campaign] * len(names), names))
    output = root / f"data/mfpcba_molformer_fusion_{args.campaign}_development_20261005"
    output.mkdir(parents=True, exist_ok=True)
    cells = pd.DataFrame([row for group in groups for row in group])
    cells.to_csv(output / "cells.csv", index=False)
    assay = cells.groupby(["task", "method"]).hits.agg(["mean", "std"]).reset_index()
    assay.to_csv(output / "assay_mean_sd.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "status": "post hoc development on exposed cohorts, not independent SOTA evidence",
        "protocol_sha256": manifest["protocol_sha256"],
        "same_labels_and_inputs": True,
        "fusion": "test-pool percentile ranks; fixed RF weights 0.2, 0.5, 0.8",
        "comparators_replayed_exactly": True,
    }, indent=2) + "\n")
    print(assay.pivot(index="task", columns="method", values="mean").to_string(), flush=True)
    print(assay.groupby("method")["mean"].mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
