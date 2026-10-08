#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from chemdeprc.diversity_selection import is_diversity_method, select_diversity, stable_seed
from chemdeprc.selection import select


FP_COL = "ecfp4_radius2_nbits2048"
DEFAULT_METHODS = (
    "raw_top_b",
    "bh",
    "weighted_bh",
    "by",
    "e_bh",
    "block_bh",
    "random_b",
    "score_leader70",
    "score_maxmin",
    "score_mmr25",
    "score_mmr50",
    "score_dpp",
    "score_cap1",
    "score_cap2",
    "score_cap5",
    "bh_cap1",
    "bh_cap2",
    "bh_cap5",
    "bh_soft75",
    "chemdeprc_scoresoft50",
    "chemdeprc_scoresoft75",
    "chemdeprc_scorecap1",
    "chemdeprc_scorecap2",
    "chemdeprc_scorecap5",
    "chemdeprc_soft25",
    "chemdeprc_soft50",
    "chemdeprc_soft75",
    "chemdeprc_cap1",
    "chemdeprc_cap2",
    "chemdeprc_cap5",
)


def parse_budget(value: str, n_test: int) -> int:
    value = value.strip().lower()
    if value.endswith("pct"):
        pct = float(value[:-3]) / 100.0
        return max(1, int(round(n_test * pct)))
    return max(1, min(int(value), n_test))


def conformal_right_tail_pvalues(scores: np.ndarray, null_scores: np.ndarray) -> np.ndarray:
    sorted_null = np.sort(null_scores)
    counts_ge = len(sorted_null) - np.searchsorted(sorted_null, scores, side="left")
    return (1.0 + counts_ge) / (len(sorted_null) + 1.0)


def fingerprint_matrix(values: pd.Series) -> np.ndarray:
    hex_values = values.astype(str).to_list()
    if not hex_values:
        return np.zeros((0, 2048), dtype=np.bool_)
    packed = b"".join(bytes.fromhex(item) for item in hex_values)
    byte_width = len(bytes.fromhex(hex_values[0]))
    byte_matrix = np.frombuffer(packed, dtype=np.uint8).reshape(len(hex_values), byte_width)
    bits = np.unpackbits(byte_matrix, axis=1, bitorder="big")[:, :2048]
    return bits.astype(np.bool_, copy=False)


def block_array(frame: pd.DataFrame, block_col: str) -> np.ndarray:
    if block_col not in frame.columns:
        raise ValueError(f"Block column {block_col!r} is missing from scores file")
    fallback = frame["molecule_chembl_id"].astype(str)
    values = frame[block_col].where(frame[block_col].notna(), fallback)
    return values.astype(str).to_numpy()


def mean_pairwise_tanimoto(bits: np.ndarray, selected: np.ndarray) -> float:
    selected = np.asarray(selected, dtype=int)
    if len(selected) < 2:
        return 0.0
    x = bits[selected].astype(np.uint16)
    intersections = x @ x.T
    bit_counts = x.sum(axis=1, dtype=np.uint16)
    unions = bit_counts[:, None] + bit_counts[None, :] - intersections
    tri = np.triu_indices(len(selected), k=1)
    values = intersections[tri] / np.maximum(unions[tri], 1)
    return float(np.mean(values))


def evaluate_real_selection(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    blocks: np.ndarray,
    bits: np.ndarray,
    selected: np.ndarray,
    budget: int,
    q: float,
) -> dict[str, float]:
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
        mean_tanimoto = 0.0
    else:
        selected_labels = labels[selected]
        true_hits = int(np.sum(selected_labels))
        false_hits = int(n_selected - true_hits)
        fdp = float(false_hits / n_selected)
        precision = float(true_hits / n_selected)
        selected_blocks = blocks[selected]
        _, counts = np.unique(selected_blocks, return_counts=True)
        unique_blocks = int(len(counts))
        max_block_share = float(counts.max() / n_selected)
        mean_tanimoto = mean_pairwise_tanimoto(bits, selected)

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
        "mean_pairwise_tanimoto": mean_tanimoto,
        "empty": float(n_selected == 0),
        "pool_active_rate": active_rate,
        "raw_top_budget_precision": raw_top_budget_precision,
    }


def first_value(frame: pd.DataFrame, column: str, default: object = "") -> object:
    if column not in frame.columns or frame.empty:
        return default
    value = frame[column].iloc[0]
    if pd.isna(value):
        return default
    return value


def select_real_method(
    method: str,
    *,
    scores: np.ndarray,
    pvalues: np.ndarray,
    blocks: np.ndarray,
    bits: np.ndarray,
    q: float,
    budget: int,
    seed_parts: tuple[object, ...],
):
    if is_diversity_method(method):
        return select_diversity(
            method,
            scores=scores,
            bits=bits,
            budget=budget,
            seed=stable_seed(*seed_parts, method),
        )
    return select(
        method,
        scores=scores,
        pvalues=pvalues,
        blocks=blocks,
        q=q,
        budget=budget,
        block_size_cap=2,
    )


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "budget",
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
        "mean_pairwise_tanimoto",
        "empty",
        "pool_active_rate",
        "n_calib_null",
        "pvalue_floor",
        "pvalue_min",
    ]
    keys = ["q", "budget_label", "method"]
    grouped = results.groupby(keys, as_index=False)[metrics]
    mean = grouped.mean()
    std = grouped.std().rename(columns={m: f"{m}_std" for m in metrics})
    counts = results.groupby(keys, as_index=False)["target_chembl_id"].nunique().rename(
        columns={"target_chembl_id": "n_targets"}
    )
    return mean.merge(std, on=keys, how="left").merge(counts, on=keys, how="left")


def run_target(
    scores_path: Path,
    q_values: list[float],
    budget_labels: list[str],
    methods: list[str],
    block_col: str,
) -> list[dict]:
    scores = pd.read_csv(scores_path)
    target_id = str(scores["target_chembl_id"].iloc[0])
    pref_name = str(scores["pref_name"].iloc[0])
    dataset = str(first_value(scores, "dataset", "ChEMBL"))
    scenario = str(first_value(scores, "scenario", "routine_scaffold_ood"))
    panel_seed = first_value(scores, "panel_seed", "")
    calibration = scores[scores["split"] == "calibration"].copy()
    testpool = scores[scores["split"] == "testpool"].copy().reset_index(drop=True)
    null_scores = calibration.loc[calibration["label"] == 0, "score"].to_numpy(dtype=float)
    if len(null_scores) == 0 or len(testpool) == 0:
        return []

    labels = testpool["label"].to_numpy(dtype=np.int8)
    test_scores = testpool["score"].to_numpy(dtype=float)
    pvalues = conformal_right_tail_pvalues(test_scores, null_scores)
    blocks = block_array(testpool, block_col)
    bits = fingerprint_matrix(testpool[FP_COL])
    rows = []
    for q in q_values:
        for budget_label in budget_labels:
            budget = parse_budget(budget_label, len(testpool))
            for method in methods:
                result = select_real_method(
                    method,
                    scores=test_scores,
                    pvalues=pvalues,
                    blocks=blocks,
                    bits=bits,
                    q=q,
                    budget=budget,
                    seed_parts=(target_id, dataset, scenario, panel_seed, q, budget_label),
                )
                metrics = evaluate_real_selection(
                    labels=labels,
                    scores=test_scores,
                    blocks=blocks,
                    bits=bits,
                    selected=result.selected,
                    budget=budget,
                    q=q,
                )
                rows.append(
                    {
                        "target_chembl_id": target_id,
                        "pref_name": pref_name,
                        "dataset": dataset,
                        "scenario": scenario,
                        "panel_seed": panel_seed,
                        "q": q,
                        "budget_label": budget_label,
                        "budget": budget,
                        "method": method,
                        "block_col": block_col,
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


def append_log(root: Path, metadata: dict, summary_path: Path, results_path: Path) -> None:
    log_path = root / "EXPERIMENT_LOG.md"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now} ChEMBL real-panel selection\n\n")
        f.write("Metadata:\n\n")
        f.write("```json\n")
        f.write(json.dumps(metadata, indent=2, sort_keys=True))
        f.write("\n```\n\n")
        f.write(f"- Raw target-level results: `{results_path}`\n")
        f.write(f"- Aggregate summary: `{summary_path}`\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", default="data/chembl_ecfp_xgb_min50/scores")
    parser.add_argument("--outdir", default="data/chembl_selection_xgb_min50")
    parser.add_argument("--q-values", nargs="+", type=float, default=[0.5, 0.7, 0.8, 0.9])
    parser.add_argument("--budgets", nargs="+", default=["25", "50", "100", "10pct"])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--block-col", default="murcko_scaffold")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    scores_dir = root / args.scores_dir
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    score_paths = sorted(scores_dir.glob("*_scores.csv"))
    rows: list[dict] = []
    for path in tqdm(score_paths, desc="chembl selection"):
        rows.extend(run_target(path, args.q_values, args.budgets, args.methods, args.block_col))

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No ChEMBL selection rows were produced")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = outdir / f"chembl_selection_results_{stamp}.csv"
    summary_path = outdir / f"chembl_selection_summary_{stamp}.csv"
    latest_results = outdir / "latest_results.csv"
    latest_summary = outdir / "latest_summary.csv"
    results.to_csv(results_path, index=False)
    results.to_csv(latest_results, index=False)
    summary = summarize(results)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(latest_summary, index=False)

    metadata = {
        "scores_dir": args.scores_dir,
        "outdir": args.outdir,
        "targets": int(results["target_chembl_id"].nunique()),
        "q_values": args.q_values,
        "budgets": args.budgets,
        "methods": args.methods,
        "block_col": args.block_col,
        "calibration_null": "calibration split molecules with retrospective inactive label",
        "candidate_pool": "full scaffold-held-out testpool",
        "formal_warning": "small calibration-null counts make p-value floors coarse; empty selections are expected when floor > q / candidate_count",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_log(root, metadata, summary_path.relative_to(root), results_path.relative_to(root))
    print(summary.to_markdown(index=False, floatfmt=".4f"))
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
