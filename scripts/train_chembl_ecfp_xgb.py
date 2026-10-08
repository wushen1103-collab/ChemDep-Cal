#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
from xgboost import XGBClassifier


FP_COL = "ecfp4_radius2_nbits2048"


def fingerprint_matrix(values: pd.Series) -> np.ndarray:
    hex_values = values.astype(str).to_list()
    if not hex_values:
        return np.zeros((0, 2048), dtype=np.float32)
    packed = b"".join(bytes.fromhex(item) for item in hex_values)
    byte_width = len(bytes.fromhex(hex_values[0]))
    byte_matrix = np.frombuffer(packed, dtype=np.uint8).reshape(len(hex_values), byte_width)
    bits = np.unpackbits(byte_matrix, axis=1, bitorder="big")[:, :2048]
    return bits.astype(np.float32, copy=False)


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def train_one(task: tuple) -> dict:
    panel_path, outdir, seed, threads, n_estimators, max_depth, learning_rate = task
    panel_path = Path(panel_path)
    outdir = Path(outdir)
    target_id = panel_path.stem.replace("_panel", "")
    panel = pd.read_csv(panel_path)
    if FP_COL not in panel.columns:
        raise ValueError(f"{panel_path} is missing {FP_COL}")

    train = panel[panel["split"] == "train"].copy()
    calibration = panel[panel["split"] == "calibration"].copy()
    testpool = panel[panel["split"] == "testpool"].copy()
    if train["label"].nunique() < 2:
        raise ValueError(f"{target_id} train split has one class; cannot train binary scorer")

    x_train = fingerprint_matrix(train[FP_COL])
    y_train = train["label"].to_numpy(dtype=np.int8)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=1.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=threads,
        random_state=seed,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight, verbose=False)

    scored_parts = []
    metrics: dict[str, float | int | str] = {
        "target_chembl_id": target_id,
        "pref_name": str(panel["pref_name"].iloc[0]),
        "model": "ecfp4_xgboost_hist",
        "seed": seed,
        "threads": threads,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
    }
    for split_name, split_df in [("train", train), ("calibration", calibration), ("testpool", testpool)]:
        x = fingerprint_matrix(split_df[FP_COL])
        y = split_df["label"].to_numpy(dtype=np.int8)
        score = model.predict_proba(x)[:, 1] if len(split_df) else np.array([], dtype=float)
        out = split_df.copy()
        out["score"] = score
        out["score_rank_in_split"] = pd.Series(score).rank(method="first", ascending=False).astype(int).to_numpy()
        out["scorer"] = "ecfp4_xgboost_hist"
        scored_parts.append(out)
        metrics[f"n_{split_name}"] = int(len(split_df))
        metrics[f"{split_name}_active"] = int(np.sum(y))
        metrics[f"{split_name}_inactive"] = int(len(y) - np.sum(y))
        metrics[f"{split_name}_active_rate"] = float(np.mean(y)) if len(y) else float("nan")
        metrics[f"auroc_{split_name}"] = safe_auc(y, score)
        metrics[f"auprc_{split_name}"] = safe_ap(y, score)

    scores = pd.concat(scored_parts, ignore_index=True)
    score_dir = outdir / "scores"
    model_dir = outdir / "models"
    score_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    scores_path = score_dir / f"{target_id}_scores.csv"
    model_path = model_dir / f"{target_id}_xgb.json"
    scores.to_csv(scores_path, index=False)
    model.save_model(model_path)
    metrics["scores_path"] = str(scores_path)
    metrics["model_path"] = str(model_path)
    return metrics


def append_log(root: Path, metadata: dict, metrics_path: Path) -> None:
    log_path = root / "EXPERIMENT_LOG.md"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now} ChEMBL ECFP-XGBoost scoring\n\n")
        f.write("Metadata:\n\n")
        f.write("```json\n")
        f.write(json.dumps(metadata, indent=2, sort_keys=True))
        f.write("\n```\n\n")
        f.write(f"- Scorer metrics: `{metrics_path}`\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/chembl_smoke_10x4000_min50/screening_manifest.csv")
    parser.add_argument("--outdir", default="data/chembl_ecfp_xgb_min50")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads-per-model", type=int, default=0)
    parser.add_argument("--leave-cpus-free", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3501)
    parser.add_argument("--n-estimators", type=int, default=350)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest = pd.read_csv(root / args.manifest)
    manifest = manifest[manifest["passes_smoke_threshold"].astype(bool)].copy()
    panel_paths = [root / item for item in manifest["panel_path"].astype(str)]
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    usable_cpus = max(1, (os.cpu_count() or 4) - args.leave_cpus_free)
    workers = args.workers or min(len(panel_paths), max(1, usable_cpus // 24))
    workers = max(1, min(workers, len(panel_paths), usable_cpus))
    threads = args.threads_per_model or max(1, usable_cpus // workers)
    threads = max(1, min(threads, usable_cpus))

    tasks = [
        (path, outdir, args.seed + i, threads, args.n_estimators, args.max_depth, args.learning_rate)
        for i, path in enumerate(panel_paths)
    ]
    rows = []
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(train_one, task) for task in tasks]
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="xgb panels"):
            rows.append(fut.result())

    metrics = pd.DataFrame(rows).sort_values("target_chembl_id")
    metrics_path = outdir / "scorer_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    metadata = {
        "manifest": args.manifest,
        "outdir": args.outdir,
        "targets": len(panel_paths),
        "workers": workers,
        "threads_per_model": threads,
        "leave_cpus_free": args.leave_cpus_free,
        "model": "ECFP4 radius 2, 2048 bits + XGBoost hist",
        "train_split_only": True,
        "calibration_labels_not_used_for_training": True,
        "seed": args.seed,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_log(root, metadata, metrics_path.relative_to(root))
    print(metrics.to_markdown(index=False, floatfmt=".4f"))
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
