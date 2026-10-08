#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from run_chembl_block_artifact_stress import (
    choose_artifact_blocks,
    conformal_right_tail_pvalues,
    fingerprint_matrix,
    inject_artifact,
    parse_budget,
    select_real_method,
    summarize_stress,
)
from run_chembl_selection import FP_COL, block_array


DEFAULT_JCTC_METHODS = (
    "raw_top_b",
    "bh",
    "by",
    "weighted_bh",
    "block_bh",
    "score_cap1",
    "score_mmr25",
    "score_dpp",
    "chemdeprc_cap1",
    "chemdeprc_soft50",
    "chemdeprc_soft75",
    "chemdeprc_scoresoft50",
)

DEFAULT_PERTURBATIONS = ("none",)


def stable_fraction(*parts: object) -> float:
    payload = "::".join(str(part) for part in parts)
    value = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
    return value / float(16**16 - 1)


def stable_int(modulus: int, *parts: object) -> int:
    if modulus <= 1:
        return 0
    payload = "::".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % modulus


def parse_level(name: str, prefix: str) -> float:
    suffix = name.replace(prefix, "")
    return float(suffix) / 100.0


def perturb_blocks(
    base_blocks: np.ndarray,
    molecule_ids: np.ndarray,
    *,
    perturbation: str,
    seed: int,
    rep: int,
) -> np.ndarray:
    perturbation = perturbation.lower()
    base_blocks = base_blocks.astype(str)
    molecule_ids = molecule_ids.astype(str)
    if perturbation in {"none", "murcko"}:
        return base_blocks.copy()
    if perturbation == "singleton":
        return np.asarray([f"singleton::{mid}" for mid in molecule_ids], dtype=object)
    if perturbation == "all_merged":
        return np.asarray(["all_merged"] * len(base_blocks), dtype=object)

    if perturbation.startswith("split"):
        n_parts = int(perturbation.replace("split", ""))
        return np.asarray(
            [
                f"{block}::split{stable_int(n_parts, seed, rep, block, mid)}"
                for block, mid in zip(base_blocks, molecule_ids)
            ],
            dtype=object,
        )

    unique_blocks = np.asarray(sorted(set(base_blocks.tolist())), dtype=object)
    if perturbation.startswith("merge"):
        group_size = int(perturbation.replace("merge", ""))
        ordered = sorted(unique_blocks.tolist(), key=lambda b: stable_fraction(seed, rep, "merge", b))
        mapping = {block: f"merge{group_size}::{i // max(1, group_size)}" for i, block in enumerate(ordered)}
        return np.asarray([mapping[block] for block in base_blocks], dtype=object)

    if perturbation.startswith("missing"):
        fraction = parse_level(perturbation, "missing")
        out: list[str] = []
        for block, mid in zip(base_blocks, molecule_ids):
            if stable_fraction(seed, rep, "missing", block, mid) < fraction:
                out.append(f"missing_singleton::{mid}")
            else:
                out.append(block)
        return np.asarray(out, dtype=object)

    if perturbation.startswith("false"):
        fraction = parse_level(perturbation, "false")
        chosen = [
            block for block in unique_blocks.tolist() if stable_fraction(seed, rep, "false_chosen", block) < fraction
        ]
        if not chosen:
            return base_blocks.copy()
        n_groups = max(1, int(math.ceil(len(chosen) / 5.0)))
        chosen_set = set(chosen)
        mapping = {
            block: f"false{int(round(fraction * 100))}::{stable_int(n_groups, seed, rep, 'false_group', block)}"
            for block in chosen
        }
        return np.asarray([mapping[block] if block in chosen_set else block for block in base_blocks], dtype=object)

    raise ValueError(f"Unknown block perturbation {perturbation!r}")


def pairwise_tanimoto_matrix(bits: np.ndarray) -> np.ndarray:
    if len(bits) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    x = bits.astype(np.uint16, copy=False)
    intersections = x @ x.T
    bit_counts = x.sum(axis=1, dtype=np.uint16)
    unions = bit_counts[:, None] + bit_counts[None, :] - intersections
    return (intersections / np.maximum(unions, 1)).astype(np.float32, copy=False)


def mean_pairwise_from_matrix(
    sim: np.ndarray,
    selected: np.ndarray,
    *,
    max_pairs: int,
    seed_parts: tuple[object, ...],
) -> tuple[float, int, str]:
    selected = np.asarray(selected, dtype=int)
    n = len(selected)
    if n < 2:
        return 0.0, 0, "empty_or_singleton"
    n_pairs = n * (n - 1) // 2
    if n_pairs <= max_pairs:
        tri = np.triu_indices(n, k=1)
        return float(np.mean(sim[np.ix_(selected, selected)][tri])), int(n_pairs), "exact"
    seed = int(hashlib.sha256("::".join(str(p) for p in seed_parts).encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    left = rng.integers(0, n, size=max_pairs)
    right = rng.integers(0, n - 1, size=max_pairs)
    right = right + (right >= left)
    return float(np.mean(sim[selected[left], selected[right]])), int(max_pairs), "sampled"


def evaluate_selection_fast(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    eval_blocks: np.ndarray,
    selection_blocks: np.ndarray,
    sim: np.ndarray,
    selected: np.ndarray,
    budget: int,
    q: float,
    max_tanimoto_pairs: int,
    seed_parts: tuple[object, ...],
) -> dict[str, float | str]:
    selected = np.asarray(selected, dtype=int)
    n_selected = int(len(selected))
    active_rate = float(np.mean(labels)) if len(labels) else 0.0
    total_hits = int(np.sum(labels))
    if n_selected == 0:
        true_hits = 0
        false_hits = 0
        fdp = 0.0
        precision = 0.0
        unique_blocks = 0
        max_block_share = 0.0
        selection_unique_blocks = 0
        selection_max_block_share = 0.0
        mean_tanimoto = 0.0
        tanimoto_pairs = 0
        tanimoto_mode = "empty"
    else:
        selected_labels = labels[selected]
        true_hits = int(np.sum(selected_labels))
        false_hits = int(n_selected - true_hits)
        fdp = float(false_hits / n_selected)
        precision = float(true_hits / n_selected)

        _, eval_counts = np.unique(eval_blocks[selected], return_counts=True)
        unique_blocks = int(len(eval_counts))
        max_block_share = float(eval_counts.max() / n_selected)

        _, selection_counts = np.unique(selection_blocks[selected], return_counts=True)
        selection_unique_blocks = int(len(selection_counts))
        selection_max_block_share = float(selection_counts.max() / n_selected)

        mean_tanimoto, tanimoto_pairs, tanimoto_mode = mean_pairwise_from_matrix(
            sim, selected, max_pairs=max_tanimoto_pairs, seed_parts=seed_parts
        )

    power = float(true_hits / total_hits) if total_hits > 0 else 0.0
    enrichment = float(precision / active_rate) if active_rate > 0 else 0.0
    top_budget = np.argsort(-scores, kind="mergesort")[:budget]
    top_budget_hits = int(np.sum(labels[top_budget])) if budget > 0 else 0
    raw_top_budget_precision = float(top_budget_hits / max(len(top_budget), 1))
    return {
        "selected_count": float(n_selected),
        "true_hits": float(true_hits),
        "false_hits": float(false_hits),
        "fdp": fdp,
        "fdp_exceeds_q": float(fdp > q),
        "precision": precision,
        "power": power,
        "enrichment_factor": enrichment,
        "unique_blocks": float(unique_blocks),
        "max_block_share": max_block_share,
        "selection_unique_blocks": float(selection_unique_blocks),
        "selection_max_block_share": selection_max_block_share,
        "mean_pairwise_tanimoto": mean_tanimoto,
        "mean_pairwise_tanimoto_pairs": float(tanimoto_pairs),
        "mean_pairwise_tanimoto_mode": tanimoto_mode,
        "empty": float(n_selected == 0),
        "pool_active_rate": active_rate,
        "raw_top_budget_precision": raw_top_budget_precision,
    }


def run_target(task: tuple) -> list[dict]:
    (
        scores_path,
        q_values,
        budget_labels,
        methods,
        reps,
        artifact_fractions,
        logit_boosts,
        perturbations,
        max_artifact_blocks,
        min_inactive_per_block,
        selection_block_col,
        artifact_block_col,
        max_tanimoto_pairs,
        seed,
    ) = task
    scores_path = Path(scores_path)
    scores = pd.read_csv(scores_path)
    target_id = str(scores["target_chembl_id"].iloc[0])
    pref_name = str(scores["pref_name"].iloc[0])
    dataset = str(scores["dataset"].iloc[0]) if "dataset" in scores.columns else "ChEMBL"
    scenario = str(scores["scenario"].iloc[0]) if "scenario" in scores.columns else "jctc_stress_grid"
    panel_seed = scores["panel_seed"].iloc[0] if "panel_seed" in scores.columns else ""
    calibration = scores[scores["split"] == "calibration"].copy()
    testpool = scores[scores["split"] == "testpool"].copy().reset_index(drop=True)
    null_scores = calibration.loc[calibration["label"] == 0, "score"].to_numpy(dtype=float)
    if len(null_scores) == 0 or len(testpool) == 0:
        return []

    labels = testpool["label"].to_numpy(dtype=np.int8)
    base_scores = testpool["score"].to_numpy(dtype=float)
    eval_blocks = block_array(testpool, artifact_block_col)
    base_selection_blocks = block_array(testpool, selection_block_col)
    molecule_ids = testpool["molecule_chembl_id"].astype(str).to_numpy()
    bits = fingerprint_matrix(testpool[FP_COL])
    sim = pairwise_tanimoto_matrix(bits)
    rows: list[dict] = []
    target_seed = seed + int(hashlib.sha256(target_id.encode("utf-8")).hexdigest()[:12], 16) % 1_000_000

    for rep in range(reps):
        for artifact_fraction in artifact_fractions:
            for logit_boost in logit_boosts:
                if artifact_fraction == 0.0 and logit_boost != 0.0:
                    continue
                if artifact_fraction > 0.0 and logit_boost == 0.0:
                    continue
                rng = np.random.default_rng(target_seed + rep * 10_003 + int(artifact_fraction * 10_000))
                artifact_blocks = choose_artifact_blocks(
                    testpool,
                    artifact_block_col=artifact_block_col,
                    rng=rng,
                    artifact_fraction=artifact_fraction,
                    max_artifact_blocks=max_artifact_blocks,
                    min_inactive_per_block=min_inactive_per_block,
                )
                perturbed_scores, artifact_molecules, artifact_inactive_molecules = inject_artifact(
                    testpool,
                    base_scores,
                    artifact_blocks,
                    artifact_block_col=artifact_block_col,
                    logit_boost=logit_boost,
                )
                pvalues = conformal_right_tail_pvalues(perturbed_scores, null_scores)
                stress_label = (
                    "no_artifact"
                    if artifact_fraction == 0.0
                    else f"frac{artifact_fraction:g}_boost{logit_boost:g}"
                )
                for block_perturbation in perturbations:
                    selection_blocks = perturb_blocks(
                        base_selection_blocks,
                        molecule_ids,
                        perturbation=block_perturbation,
                        seed=target_seed,
                        rep=rep,
                    )
                    for q in q_values:
                        for budget_label in budget_labels:
                            budget = parse_budget(budget_label, len(testpool))
                            for method in methods:
                                result = select_real_method(
                                    method,
                                    scores=perturbed_scores,
                                    pvalues=pvalues,
                                    blocks=selection_blocks,
                                    bits=bits,
                                    q=q,
                                    budget=budget,
                                    seed_parts=(
                                        target_id,
                                        dataset,
                                        scenario,
                                        panel_seed,
                                        rep,
                                        stress_label,
                                        block_perturbation,
                                        q,
                                        budget_label,
                                    ),
                                )
                                metrics = evaluate_selection_fast(
                                    labels=labels,
                                    scores=perturbed_scores,
                                    eval_blocks=eval_blocks,
                                    selection_blocks=selection_blocks,
                                    sim=sim,
                                    selected=result.selected,
                                    budget=budget,
                                    q=q,
                                    max_tanimoto_pairs=max_tanimoto_pairs,
                                    seed_parts=(
                                        target_id,
                                        rep,
                                        stress_label,
                                        block_perturbation,
                                        q,
                                        budget_label,
                                        method,
                                    ),
                                )
                                rows.append(
                                    {
                                        "target_chembl_id": target_id,
                                        "pref_name": pref_name,
                                        "dataset": dataset,
                                        "scenario": scenario,
                                        "panel_seed": panel_seed,
                                        "rep": rep,
                                        "stress_label": stress_label,
                                        "artifact_fraction": artifact_fraction,
                                        "logit_boost": logit_boost,
                                        "block_perturbation": block_perturbation,
                                        "artifact_block_count": len(artifact_blocks),
                                        "artifact_molecules": artifact_molecules,
                                        "artifact_inactive_molecules": artifact_inactive_molecules,
                                        "q": q,
                                        "budget_label": budget_label,
                                        "budget": budget,
                                        "method": method,
                                        "selection_block_col": selection_block_col,
                                        "artifact_block_col": artifact_block_col,
                                        "certificate": result.certificate,
                                        "n_testpool": int(len(testpool)),
                                        "n_testpool_active": int(np.sum(labels)),
                                        "n_testpool_inactive": int(len(labels) - np.sum(labels)),
                                        "n_calibration": int(len(calibration)),
                                        "n_calib_null": int(len(null_scores)),
                                        "pvalue_floor": float(1.0 / (len(null_scores) + 1.0)),
                                        "pvalue_min": float(np.min(pvalues)),
                                        "pvalue_median": float(np.median(pvalues)),
                                        **metrics,
                                    }
                                )
    return rows


def summarize_jctc(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "budget",
        "artifact_block_count",
        "artifact_molecules",
        "artifact_inactive_molecules",
        "selected_count",
        "true_hits",
        "false_hits",
        "fdp",
        "fdp_exceeds_q",
        "precision",
        "power",
        "enrichment_factor",
        "unique_blocks",
        "max_block_share",
        "selection_unique_blocks",
        "selection_max_block_share",
        "mean_pairwise_tanimoto",
        "mean_pairwise_tanimoto_pairs",
        "empty",
        "pool_active_rate",
        "n_calib_null",
        "pvalue_floor",
        "pvalue_min",
    ]
    keys = [
        "stress_label",
        "artifact_fraction",
        "logit_boost",
        "block_perturbation",
        "q",
        "budget_label",
        "method",
    ]
    grouped = results.groupby(keys, as_index=False)[metrics]
    mean = grouped.mean()
    std = grouped.std().rename(columns={m: f"{m}_std" for m in metrics})
    target_counts = results.groupby(keys, as_index=False)["target_chembl_id"].nunique().rename(
        columns={"target_chembl_id": "n_targets"}
    )
    unit_counts = results.groupby(keys, as_index=False).size().rename(columns={"size": "n_target_reps"})
    return mean.merge(std, on=keys, how="left").merge(target_counts, on=keys, how="left").merge(
        unit_counts, on=keys, how="left"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", default="data/chembl_ecfp_xgb_potent_a7_i6/scores")
    parser.add_argument("--outdir", default="data/jctc_stress_grid")
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--artifact-fractions", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--logit-boosts", nargs="+", type=float, default=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0])
    parser.add_argument("--block-perturbations", nargs="+", default=list(DEFAULT_PERTURBATIONS))
    parser.add_argument("--max-artifact-blocks", type=int, default=25)
    parser.add_argument("--min-inactive-per-block", type=int, default=2)
    parser.add_argument("--q-values", nargs="+", type=float, default=[0.7])
    parser.add_argument("--budgets", nargs="+", default=["10", "20", "30", "50", "75", "100", "150", "200", "500"])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_JCTC_METHODS))
    parser.add_argument("--selection-block-col", default="murcko_scaffold")
    parser.add_argument("--artifact-block-col", default="murcko_scaffold")
    parser.add_argument("--max-tanimoto-pairs", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--leave-cpus-free", type=int, default=30)
    parser.add_argument("--seed", type=int, default=353500)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    scores_dir = root / args.scores_dir
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    score_paths = sorted(scores_dir.glob("*_scores.csv"))
    if not score_paths:
        raise RuntimeError(f"No score files found under {scores_dir}")
    usable_cpus = max(1, (os.cpu_count() or 4) - args.leave_cpus_free)
    workers = args.workers or min(len(score_paths), usable_cpus)
    workers = max(1, min(workers, len(score_paths), usable_cpus))
    artifact_fractions = sorted(set(args.artifact_fractions))
    logit_boosts = sorted(set(args.logit_boosts))
    if 0.0 not in artifact_fractions:
        artifact_fractions = [0.0] + artifact_fractions
    if 0.0 not in logit_boosts:
        logit_boosts = [0.0] + logit_boosts

    tasks = [
        (
            path,
            args.q_values,
            args.budgets,
            args.methods,
            args.reps,
            artifact_fractions,
            logit_boosts,
            args.block_perturbations,
            args.max_artifact_blocks,
            args.min_inactive_per_block,
            args.selection_block_col,
            args.artifact_block_col,
            args.max_tanimoto_pairs,
            args.seed + i * 100_000,
        )
        for i, path in enumerate(score_paths)
    ]
    rows: list[dict] = []
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_target, task) for task in tasks]
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="jctc stress grid"):
            rows.extend(fut.result())
    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No JCTC stress-grid rows were produced")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = outdir / f"jctc_stress_grid_results_{stamp}.csv"
    summary_path = outdir / f"jctc_stress_grid_summary_{stamp}.csv"
    results.to_csv(results_path, index=False)
    results.to_csv(outdir / "latest_results.csv", index=False)
    summary_df = summarize_jctc(results)
    summary_df.to_csv(summary_path, index=False)
    summary_df.to_csv(outdir / "latest_summary.csv", index=False)
    metadata = {
        "scores_dir": args.scores_dir,
        "outdir": args.outdir,
        "reps": args.reps,
        "targets": len(score_paths),
        "artifact_fractions": artifact_fractions,
        "logit_boosts": logit_boosts,
        "block_perturbations": args.block_perturbations,
        "q_values": args.q_values,
        "budgets": args.budgets,
        "methods": args.methods,
        "selection_block_col": args.selection_block_col,
        "artifact_block_col": args.artifact_block_col,
        "workers": workers,
        "max_tanimoto_pairs": args.max_tanimoto_pairs,
        "metric_note": "hits, FDP, true Murcko unique blocks, and max block share are exact; mean pairwise Tanimoto is exact up to max_tanimoto_pairs pairs and sampled above that threshold",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_df.head(50).to_markdown(index=False, floatfmt=".4f"))
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
