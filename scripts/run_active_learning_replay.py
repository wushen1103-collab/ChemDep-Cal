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
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from run_chembl_selection import FP_COL, block_array, evaluate_real_selection, fingerprint_matrix, select_real_method


def seed_training_indices(labels: np.ndarray, rng: np.random.Generator, n_pos: int, n_neg: int) -> np.ndarray:
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    chosen_pos = rng.choice(pos, size=min(n_pos, len(pos)), replace=False) if len(pos) else np.array([], dtype=int)
    chosen_neg = rng.choice(neg, size=min(n_neg, len(neg)), replace=False) if len(neg) else np.array([], dtype=int)
    return np.unique(np.concatenate([chosen_pos, chosen_neg])).astype(int)


def fit_scores(bits: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_labels = labels[train_idx]
    if len(np.unique(train_labels)) < 2:
        prior = float(np.mean(train_labels)) if len(train_labels) else 0.01
        scores = np.full(len(labels), prior, dtype=float)
        uncertainty = np.full(len(labels), 0.5, dtype=float)
        return scores, uncertainty
    clf = RandomForestClassifier(
        n_estimators=160,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=seed,
    )
    clf.fit(bits[train_idx].astype(np.uint8), train_labels)
    scores = clf.predict_proba(bits.astype(np.uint8))[:, 1]
    uncertainty = 1.0 - np.abs(scores - 0.5) * 2.0
    return scores, uncertainty


def select_acquisition(
    *,
    strategy: str,
    scores: np.ndarray,
    uncertainty: np.ndarray,
    blocks: np.ndarray,
    bits: np.ndarray,
    queried: np.ndarray,
    batch_size: int,
    seed_parts: tuple[object, ...],
) -> np.ndarray:
    available = ~queried
    avail_idx = np.flatnonzero(available)
    if len(avail_idx) == 0:
        return np.array([], dtype=int)
    batch_size = min(batch_size, len(avail_idx))
    strategy = strategy.lower()
    if strategy == "score":
        order = np.argsort(-scores[avail_idx], kind="mergesort")[:batch_size]
        return avail_idx[order]
    if strategy == "uncertainty":
        order = np.argsort(-uncertainty[avail_idx], kind="mergesort")[:batch_size]
        return avail_idx[order]
    if strategy == "score_mmr25":
        local = select_real_method(
            "score_mmr25",
            scores=scores[avail_idx],
            pvalues=np.ones(len(avail_idx)),
            blocks=blocks[avail_idx],
            bits=bits[avail_idx],
            q=0.7,
            budget=batch_size,
            seed_parts=seed_parts,
        ).selected
        return avail_idx[local]
    if strategy == "chemdeprc_scorecap1":
        local_scores = scores[avail_idx]
        pvalues = 1.0 - np.clip(local_scores, 1e-6, 1.0 - 1e-6)
        local = select_real_method(
            "chemdeprc_scorecap1",
            scores=local_scores,
            pvalues=pvalues,
            blocks=blocks[avail_idx],
            bits=bits[avail_idx],
            q=0.7,
            budget=batch_size,
            seed_parts=seed_parts,
        ).selected
        if len(local) < batch_size:
            chosen = set(local.tolist())
            fill_order = np.argsort(-local_scores, kind="mergesort")
            fill = [idx for idx in fill_order if int(idx) not in chosen]
            local = np.asarray(local.tolist() + fill[: batch_size - len(local)], dtype=int)
        return avail_idx[local[:batch_size]]
    raise ValueError(f"Unknown active-learning strategy {strategy!r}")


def run_panel(
    scores_path: Path,
    *,
    strategies: list[str],
    batch_size: int,
    rounds: int,
    seed: int,
) -> list[dict]:
    frame = pd.read_csv(scores_path)
    target_id = str(frame["target_chembl_id"].iloc[0])
    pref_name = str(frame["pref_name"].iloc[0])
    dataset = str(frame["dataset"].iloc[0]) if "dataset" in frame.columns else "ChEMBL"
    scenario = str(frame["scenario"].iloc[0]) if "scenario" in frame.columns else "active_learning_replay"
    testpool = frame[frame["split"] == "testpool"].copy().reset_index(drop=True)
    labels = testpool["label"].to_numpy(dtype=np.int8)
    blocks = block_array(testpool, "murcko_scaffold")
    bits = fingerprint_matrix(testpool[FP_COL])
    base_scores = testpool["score"].to_numpy(dtype=float)
    rows = []
    for s_pos, strategy in enumerate(strategies):
        rng = np.random.default_rng(seed + s_pos * 100_003)
        queried = np.zeros(len(testpool), dtype=bool)
        init = seed_training_indices(labels, rng, n_pos=3, n_neg=30)
        queried[init] = True
        train_idx = np.flatnonzero(queried)
        for round_id in range(rounds + 1):
            # Evaluate the cumulative queried set against true labels and blocks.
            selected = np.flatnonzero(queried)
            metrics = evaluate_real_selection(
                labels=labels,
                scores=base_scores,
                blocks=blocks,
                bits=bits,
                selected=selected,
                budget=len(selected),
                q=0.7,
            )
            rows.append(
                {
                    "target_chembl_id": target_id,
                    "pref_name": pref_name,
                    "dataset": dataset,
                    "scenario": scenario,
                    "strategy": strategy,
                    "round": round_id,
                    "batch_size": batch_size,
                    "cumulative_budget": int(len(selected)),
                    "seed": seed,
                    **metrics,
                }
            )
            if round_id >= rounds:
                break
            scores, uncertainty = fit_scores(bits, labels, train_idx, seed + round_id * 997 + s_pos)
            batch = select_acquisition(
                strategy=strategy,
                scores=scores,
                uncertainty=uncertainty,
                blocks=blocks,
                bits=bits,
                queried=queried,
                batch_size=batch_size,
                seed_parts=(target_id, seed, strategy, round_id),
            )
            queried[batch] = True
            train_idx = np.flatnonzero(queried)
    return rows


def run_panel_task(task: tuple) -> list[dict]:
    scores_path, strategies, batch_size, rounds, seed = task
    return run_panel(
        Path(scores_path),
        strategies=strategies,
        batch_size=batch_size,
        rounds=rounds,
        seed=seed,
    )


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "selected_count",
        "true_hits",
        "false_hits",
        "fdp",
        "precision",
        "power",
        "unique_blocks",
        "max_block_share",
        "mean_pairwise_tanimoto",
    ]
    grouped = results.groupby(["strategy", "round", "batch_size", "cumulative_budget"], as_index=False)[metrics]
    mean = grouped.mean()
    std = grouped.std().rename(columns={m: f"{m}_std" for m in metrics})
    counts = results.groupby(["strategy", "round", "batch_size", "cumulative_budget"], as_index=False)["target_chembl_id"].nunique().rename(columns={"target_chembl_id": "n_targets"})
    return mean.merge(std, on=["strategy", "round", "batch_size", "cumulative_budget"]).merge(counts, on=["strategy", "round", "batch_size", "cumulative_budget"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", default="data/chembl_ecfp_xgb_potent_a7_i6/scores")
    parser.add_argument("--outdir", default="data/jctc_active_learning_replay")
    parser.add_argument("--strategies", nargs="+", default=["score", "uncertainty", "score_mmr25", "chemdeprc_scorecap1"])
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3611)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--leave-cpus-free", type=int, default=30)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    score_paths = sorted((root / args.scores_dir).glob("*_scores.csv"))
    usable_cpus = max(1, (os.cpu_count() or 4) - args.leave_cpus_free)
    workers = args.workers or min(len(score_paths), usable_cpus)
    workers = max(1, min(workers, len(score_paths), usable_cpus))
    tasks = [
        (path, args.strategies, args.batch_size, args.rounds, args.seed + i * 1009)
        for i, path in enumerate(score_paths)
    ]
    rows = []
    if workers == 1:
        for task in tqdm(tasks, desc="active-learning replay"):
            rows.extend(run_panel_task(task))
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_panel_task, task) for task in tasks]
            for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="active-learning replay"):
                rows.extend(fut.result())
    results = pd.DataFrame(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results.to_csv(outdir / f"active_learning_replay_results_{stamp}.csv", index=False)
    results.to_csv(outdir / "latest_results.csv", index=False)
    summary = summarize(results)
    summary.to_csv(outdir / f"active_learning_replay_summary_{stamp}.csv", index=False)
    summary.to_csv(outdir / "latest_summary.csv", index=False)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scores_dir": args.scores_dir,
        "strategies": args.strategies,
        "batch_size": args.batch_size,
        "rounds": args.rounds,
        "seed": args.seed,
        "workers": workers,
        "note": "Sequential replay baseline. This is not a faithful reproduction of the 2025 ALBF paper; it tests whether a ChemDep allocation layer remains useful in iterative feedback.",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary.to_markdown(index=False, floatfmt=".4f"))


if __name__ == "__main__":
    main()
