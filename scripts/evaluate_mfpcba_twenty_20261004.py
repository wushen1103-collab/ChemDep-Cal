#!/usr/bin/env python3
"""Frozen matched cap-1 selectors on independent MF-PCBA assay families."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.special import expit
from xgboost import XGBClassifier


PROTOCOL_SHA256 = "b49197be81c4f7557eaecfabeafa74a0027c4c7fae63f8b7950089f1f0f3af39"
METHODS = ("chemdep_cal", "score_cap1", "score_only_ranker", "xgb_pairwise_cap1",
           "ecfp_score_xgb_cap1")


def choose(values: np.ndarray, blocks: np.ndarray, keys: np.ndarray) -> np.ndarray:
    order = np.lexsort((keys, -values))
    selected, seen = [], set()
    for index in order:
        block = blocks[index]
        if block not in seen:
            selected.append(int(index))
            seen.add(block)
            if len(selected) == 50:
                break
    return np.asarray(selected, dtype=int)


def molecular_features(smiles: pd.Series, generator) -> np.ndarray:
    result = np.empty((len(smiles), 2048), dtype=np.uint8)
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise AssertionError("Invalid frozen molecular structure")
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), result[index])
    return result


def run_seed(root_s: str, seed: int, campaign: str = "twenty") -> int:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    import probe_meta_band_v2_20261003  # noqa: F401; install 16-feature map
    from modern_selector_benchmark_20261004 import fit_pairwise
    from probe_meta_band_20261003 import training_matrix
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_selection import block_array
    from train_artifact_meta_ranker_20261003 import fit_ranker

    RDLogger.DisableLog("rdApp.*")
    if campaign not in {"twenty", "twelve"}:
        raise ValueError(f"Unknown campaign: {campaign}")
    directory = root / f"data/mfpcba_{campaign}_20261004" if campaign == "twenty" else root / "data/mfpcba_twelve_new_20261004"
    protocol_sha256 = (PROTOCOL_SHA256 if campaign == "twenty" else
                       "f2542d1232d14437bea793780fbdf18fa2603ced5ca7fc04ca4642bd7780c2d8")
    expected_n = 20 if campaign == "twenty" else 12
    manifest = json.loads((directory / "cohort_manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != protocol_sha256 or len(manifest["independent_tasks"]) != expected_n:
        raise AssertionError("Frozen protocol or source inventory changed")
    ledger = pd.read_csv(directory / "cohort_audit.csv")
    if len(ledger) != expected_n * 5 or not ledger.groupby("task").size().eq(5).all():
        raise AssertionError("Incomplete eligibility ledger")
    eligible = manifest["eligible_tasks"]
    if sorted(eligible) != [name for name in sorted(manifest["independent_tasks"])
                            if ledger.loc[ledger.task.eq(name), "eligible"].all()]:
        raise AssertionError("Eligibility list inconsistent with ledger")
    panels, provenance = [], {}
    for name in eligible:
        cohort_file = directory / "cohorts" / f"{name}_cohort.parquet" if campaign == "twelve" else directory / f"{name}_cohort.parquet"
        split_file = directory / "cohorts" / f"{name}_seed{seed}_indices.npz" if campaign == "twelve" else directory / f"{name}_seed{seed}_indices.npz"
        cohort = pd.read_parquet(cohort_file)
        with np.load(split_file) as split:
            cal = cohort.iloc[split["calibration"]].reset_index(drop=True).copy()
            test = cohort.iloc[split["test"]].reset_index(drop=True).copy()
        center = float(np.median(cal.score))
        scale = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
        if scale <= 0:
            raise AssertionError("Changed eligible score IQR")
        for frame in (cal, test):
            frame["score"] = expit((frame.score.to_numpy(float) - center) / scale)
            frame["molecule_chembl_id"] = frame.canonical_smiles
            frame["target_chembl_id"] = name
        panels.append({"target": name, "cal": cal, "test": test,
                       "null": cal.loc[cal.label.eq(0), "score"].to_numpy(float),
                       "cal_blocks": block_array(cal, "murcko_scaffold"),
                       "test_blocks": block_array(test, "murcko_scaffold")})
        provenance[name] = {
            "cohort_sha256": hashlib.sha256(cohort_file.read_bytes()).hexdigest(),
            "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
            "cal_n": len(cal), "test_n": len(test), "test_pos": int(test.label.sum()),
        }
    output = directory / "selectors"
    output.mkdir(parents=True, exist_ok=True)
    x, y = training_matrix(panels, repeats=3)
    if x.shape[1] != 16:
        raise AssertionError("Changed ChemDep feature geometry")
    full = fit_ranker(x, y, 862026)
    score_only = fit_ranker(x[:, :4], y, 862026)
    pairwise = fit_pairwise(panels)
    full.save_model(output / f"seed{seed}.chemdep.json")
    score_only.save_model(output / f"seed{seed}.score_only.json")
    pairwise.save_model(output / f"seed{seed}.pairwise.json")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows = []
    for panel in panels:
        name, cal, test = panel["target"], panel["cal"], panel["test"]
        train_features = np.column_stack([
            molecular_features(cal.canonical_smiles, generator), cal.score.to_numpy(float)])
        test_features = np.column_stack([
            molecular_features(test.canonical_smiles, generator), test.score.to_numpy(float)])
        labels_cal = cal.label.to_numpy(int)
        positives = max(int(labels_cal.sum()), 1)
        ecfp = XGBClassifier(
            n_estimators=160, max_depth=3, learning_rate=.05, subsample=.9,
            colsample_bytree=.9, min_child_weight=8, reg_lambda=4.,
            tree_method="hist", eval_metric="logloss", n_jobs=4,
            random_state=862026,
            scale_pos_weight=min(20., max(1., (len(labels_cal) - positives) / positives)),
        )
        ecfp.fit(train_features, labels_cal)
        ecfp.save_model(output / f"seed{seed}.{name}.ecfp_score.json")
        scores = test.score.to_numpy(float)
        labels = test.label.to_numpy(int)
        blocks = panel["test_blocks"]
        features = evidence_features(scores, blocks, panel["null"])
        values = {
            "chemdep_cal": full.predict_proba(features)[:, 1],
            "score_cap1": scores,
            "score_only_ranker": score_only.predict_proba(features[:, :4])[:, 1],
            "xgb_pairwise_cap1": pairwise.predict(features),
            "ecfp_score_xgb_cap1": ecfp.predict_proba(test_features)[:, 1],
        }
        keys = np.asarray([int.from_bytes(hashlib.sha256(
            f"{name}/{seed}/{smiles}".encode()).digest()[:8], "big")
            for smiles in test.canonical_smiles], dtype=np.uint64)
        tie_rng = np.random.default_rng(20261004 + seed + int.from_bytes(
            hashlib.sha256(name.encode()).digest()[:4], "big"))
        tie_hits = [int(labels[choose(scores, blocks, tie_rng.random(len(scores)))].sum())
                    for _ in range(1000)]
        for method in METHODS:
            selected = choose(values[method], blocks, keys)
            if len(selected) != 50 or len(set(blocks[selected])) != 50:
                raise AssertionError("Capacity or budget violation")
            rows.append({"task": name, "seed": seed, "method": method,
                         "hits": int(labels[selected].sum()), "selected": len(selected),
                         "scaffolds": len(set(blocks[selected])), "test_n": len(test),
                         "test_pos": int(labels.sum()),
                         "scorecap_random_tie_mean": float(np.mean(tie_hits)),
                         "scorecap_random_tie_sd": float(np.std(tie_hits, ddof=1))})
        print("seed", seed, "task", name, "completed", flush=True)
    result = pd.DataFrame(rows)
    if len(result) != len(eligible) * len(METHODS) or result.duplicated(
            ["task", "seed", "method"]).any():
        raise AssertionError("Incomplete frozen selector grid")
    result.to_csv(output / f"seed{seed}_cells.csv", index=False)
    (output / f"seed{seed}_manifest.json").write_text(json.dumps({
        "protocol_sha256": protocol_sha256, "seed": seed, "eligible_tasks": eligible,
        "methods": METHODS, "source_commit": manifest["source_commit"],
        "test_label_use": "selection endpoint and diagnostic random-tie audit only",
        "provenance": provenance,
    }, indent=2) + "\n", encoding="utf-8")
    return len(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", choices=("twenty", "twelve"), default="twenty")
    args = parser.parse_args()
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        print(list(pool.map(run_seed, [str(args.root.resolve())] * 5, range(1, 6),
                            [args.campaign] * 5)), flush=True)


if __name__ == "__main__":
    main()
