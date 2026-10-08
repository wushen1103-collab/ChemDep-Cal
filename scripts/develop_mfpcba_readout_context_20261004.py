#!/usr/bin/env python3
"""Post hoc assay-readout development on the five exposed twelve-chain tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.special import expit
from xgboost import XGBClassifier

from evaluate_mfpcba_twenty_20261004 import choose, molecular_features
from prepare_pubchem_four_untouched_20261004 import verified_chunks


def classifier(labels: np.ndarray) -> XGBClassifier:
    positives = int(labels.sum())
    return XGBClassifier(
        n_estimators=160, max_depth=3, learning_rate=.05, subsample=.9,
        colsample_bytree=.9, min_child_weight=8, reg_lambda=4.,
        tree_method="hist", eval_metric="logloss", n_jobs=4,
        random_state=862026,
        scale_pos_weight=min(20., max(1., (len(labels) - positives) / positives)),
    )


def quality_frame(root: Path, name: str, cohort: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    raw = verified_chunks(root / f"{name}_active_full_chunks")
    raw["CID"] = pd.to_numeric(raw.PUBCHEM_CID, errors="coerce")
    raw = raw.dropna(subset=["CID"]).copy()
    raw.CID = raw.CID.astype(int)
    excluded = {"PUBCHEM_RESULT_TAG", "PUBCHEM_SID", "PUBCHEM_CID",
                "PUBCHEM_EXT_DATASOURCE_SMILES", "PUBCHEM_ACTIVITY_OUTCOME",
                "PUBCHEM_ACTIVITY_SCORE", "PUBCHEM_ACTIVITY_URL",
                "PUBCHEM_ASSAYDATA_COMMENT", "CID"}
    columns = []
    for column in raw.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(raw[column], errors="coerce")
        if values.notna().mean() >= .6 and values.nunique() >= 5:
            raw[column] = values
            columns.append(column)
    if not columns:
        return np.empty((len(cohort), 0)), []
    per_cid = raw.groupby("CID", sort=False)[columns].median()
    aligned = per_cid.reindex(cohort.CID.to_numpy(int))
    columns = [column for column in columns if aligned[column].notna().mean() >= .8
               and aligned[column].nunique() >= 5]
    return aligned[columns].to_numpy(float), columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.root.resolve()
    sys.path[:0] = [str(repo / "src"), str(repo / "scripts")]
    import probe_meta_band_v2_20261003  # noqa: F401; install 16-feature map
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_selection import block_array

    RDLogger.DisableLog("rdApp.*")
    directory = repo / "data/mfpcba_twelve_new_20261004"
    manifest = json.loads((directory / "cohort_manifest.json").read_text(encoding="utf-8"))
    out = repo / "data/mfpcba_readout_development_20261004"
    out.mkdir(parents=True, exist_ok=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows, provenance = [], {}
    for name in manifest["eligible_tasks"]:
        cohort = pd.read_parquet(directory / "cohorts" / f"{name}_cohort.parquet")
        fingerprints = molecular_features(cohort.canonical_smiles, generator)
        quality, columns = quality_frame(directory, name, cohort)
        provenance[name] = {"quality_columns": columns, "quality_dimensions": len(columns),
                            "cohort_n": len(cohort)}
        print(name, "quality columns", columns, flush=True)
        for seed in range(1, 6):
            with np.load(directory / "cohorts" / f"{name}_seed{seed}_indices.npz") as split:
                cal_idx, test_idx = split["calibration"], split["test"]
            cal = cohort.iloc[cal_idx].reset_index(drop=True).copy()
            test = cohort.iloc[test_idx].reset_index(drop=True).copy()
            center = float(np.median(cal.score))
            scale = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
            for frame in (cal, test):
                frame["score"] = expit((frame.score.to_numpy(float) - center) / scale)
                frame["molecule_chembl_id"] = frame.canonical_smiles
                frame["target_chembl_id"] = name
            cal_blocks = block_array(cal, "murcko_scaffold")
            test_blocks = block_array(test, "murcko_scaffold")
            null = cal.loc[cal.label.eq(0), "score"].to_numpy(float)
            ecfp_cal = np.column_stack([fingerprints[cal_idx], cal.score.to_numpy(float)])
            ecfp_test = np.column_stack([fingerprints[test_idx], test.score.to_numpy(float)])
            readout_cal = np.column_stack([ecfp_cal, quality[cal_idx]])
            readout_test = np.column_stack([ecfp_test, quality[test_idx]])
            context_cal = evidence_features(cal.score.to_numpy(float), cal_blocks, null)
            context_test = evidence_features(test.score.to_numpy(float), test_blocks, null)
            combined_cal = np.column_stack([readout_cal, context_cal])
            combined_test = np.column_stack([readout_test, context_test])
            keys = np.asarray([int.from_bytes(hashlib.sha256(
                f"{name}/{seed}/{smiles}".encode()).digest()[:8], "big")
                for smiles in test.canonical_smiles], dtype=np.uint64)
            labels_cal = cal.label.to_numpy(int)
            labels_test = test.label.to_numpy(int)
            frozen = pd.read_csv(directory / "selectors" / f"seed{seed}_cells.csv")
            expected = int(frozen.loc[frozen.task.eq(name)
                                      & frozen.method.eq("ecfp_score_xgb_cap1"), "hits"].iloc[0])
            for method, train_x, test_x in (
                    ("ecfp_score", ecfp_cal, ecfp_test),
                    ("ecfp_score_readout", readout_cal, readout_test),
                    ("ecfp_score_readout_context", combined_cal, combined_test)):
                model = classifier(labels_cal).fit(train_x, labels_cal)
                prediction = model.predict_proba(test_x)[:, 1]
                selected = choose(prediction, test_blocks, keys)
                hits = int(labels_test[selected].sum())
                if len(selected) != 50 or len(set(test_blocks[selected])) != 50:
                    raise AssertionError("Budget or scaffold-cap violation")
                if method == "ecfp_score" and hits != expected:
                    raise AssertionError(f"Frozen molecular baseline mismatch: {name}/{seed}")
                if method != "ecfp_score":
                    model.save_model(out / f"{name}.seed{seed}.{method}.json")
                rows.append({"task": name, "seed": seed, "method": method,
                             "hits": hits, "test_n": len(test),
                             "test_pos": int(labels_test.sum()), "quality_dimensions": len(columns)})
            print(name, seed, "completed", flush=True)
    cells = pd.DataFrame(rows)
    cells.to_csv(out / "cells.csv", index=False)
    assay = cells.groupby(["task", "method"]).hits.agg(["mean", "std"]).reset_index()
    assay.to_csv(out / "assay_mean_sd.csv", index=False)
    summary = assay.groupby("method").agg(macro_hits=("mean", "mean"),
                                           assay_sd=("mean", "std"))
    summary.to_csv(out / "summary.csv")
    (out / "manifest.json").write_text(json.dumps({
        "status": "post hoc development on five previously exposed chains",
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "tasks": provenance,
        "test_label_use": "hit counting only; no readout selection or hyperparameter tuning",
    }, indent=2) + "\n", encoding="utf-8")
    print(assay.to_string(index=False), flush=True)
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
