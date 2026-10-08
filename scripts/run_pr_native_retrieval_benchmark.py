#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_olivetti_faces, load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from chemdeprc.diversity_selection import is_diversity_method, select_diversity, stable_seed
from chemdeprc.selection import select


METHODS = [
    "raw_top_b",
    "bh",
    "weighted_bh",
    "score_mmr25",
    "score_dpp",
    "score_cap1",
    "chemdeprc_cap1",
    "chemdeprc_soft75",
]

METRICS = ["selected_count", "true_hits", "fdp", "precision", "unique_blocks", "max_block_share", "mean_pairwise_similarity"]
HIGHER_IS_BETTER = ["true_hits", "unique_blocks"]
LOWER_IS_BETTER = ["fdp", "max_block_share", "mean_pairwise_similarity"]


def write_pair(df: pd.DataFrame, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(df.to_markdown(index=False, floatfmt=".4f") + "\n", encoding="utf-8")


def conformal_right_tail_pvalues(scores: np.ndarray, null_scores: np.ndarray) -> np.ndarray:
    sorted_null = np.sort(null_scores)
    counts_ge = len(sorted_null) - np.searchsorted(sorted_null, scores, side="left")
    return (1.0 + counts_ge) / (len(sorted_null) + 1.0)


def parse_budget(value: str, n: int) -> int:
    value = str(value).strip().lower()
    if value.endswith("pct"):
        return max(1, min(n, int(round(float(value[:-3]) / 100.0 * n))))
    return max(1, min(n, int(value)))


def dataset_panels(data_home: Path) -> list[dict[str, object]]:
    digits = load_digits()
    panels: list[dict[str, object]] = [
        {
            "dataset": "sklearn_digits",
            "features": digits.data.astype(np.float32) / 16.0,
            "labels": digits.target.astype(int),
            "target_names": [f"digit_{i}" for i in range(10)],
            "source": "sklearn.datasets.load_digits",
        }
    ]
    try:
        faces = fetch_olivetti_faces(data_home=str(data_home), download_if_missing=True, shuffle=False)
        # Aggregate 40 identities into 10 super-classes to make finite-budget hit counts meaningful.
        panels.append(
            {
                "dataset": "olivetti_faces_group10",
                "features": faces.data.astype(np.float32),
                "labels": (faces.target.astype(int) // 4).astype(int),
                "target_names": [f"face_group_{i}" for i in range(10)],
                "source": "sklearn.datasets.fetch_olivetti_faces; identity groups target//4",
            }
        )
    except Exception as exc:  # pragma: no cover - depends on remote dataset availability.
        panels.append(
            {
                "dataset": "sklearn_digits_lowres_shifted",
                "features": np.roll(digits.data.astype(np.float32) / 16.0, shift=1, axis=1),
                "labels": digits.target.astype(int),
                "target_names": [f"digit_{i}" for i in range(10)],
                "source": f"fallback derived from sklearn.datasets.load_digits after Olivetti fetch failed: {type(exc).__name__}",
            }
        )
    return panels


def split_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outer = StratifiedShuffleSplit(n_splits=1, train_size=0.50, random_state=seed)
    train_idx, temp_idx = next(outer.split(np.zeros_like(labels), labels))
    inner = StratifiedShuffleSplit(n_splits=1, train_size=0.50, random_state=seed + 17)
    calib_rel, test_rel = next(inner.split(np.zeros_like(labels[temp_idx]), labels[temp_idx]))
    return train_idx, temp_idx[calib_rel], temp_idx[test_rel]


def classifier_scores(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    calib_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_components = max(2, min(48, features.shape[1], len(train_idx) - 1))
    clf = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=seed),
        LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto", random_state=seed),
    )
    clf.fit(features[train_idx], labels[train_idx])
    calib_proba = clf.predict_proba(features[calib_idx])
    test_proba = clf.predict_proba(features[test_idx])
    embedder = make_pipeline(StandardScaler(), PCA(n_components=n_components, random_state=seed + 1))
    embedder.fit(features[train_idx])
    calib_embed = embedder.transform(features[calib_idx])
    test_embed = embedder.transform(features[test_idx])
    classes = clf.named_steps["logisticregression"].classes_.astype(int)
    return calib_proba, test_proba, calib_embed, test_embed, classes


def visual_blocks(test_embed: np.ndarray, seed: int) -> np.ndarray:
    n = len(test_embed)
    n_clusters = max(4, min(30, n // 12))
    n_clusters = min(n_clusters, max(1, n))
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(test_embed)
    return np.asarray([f"visual_cluster_{x:03d}" for x in labels], dtype=object)


def bits_from_embedding(embed: np.ndarray) -> np.ndarray:
    med = np.median(embed, axis=0, keepdims=True)
    bits = embed > med
    if bits.shape[1] < 8:
        extra = embed >= np.mean(embed, axis=0, keepdims=True)
        bits = np.concatenate([bits, extra], axis=1)
    return bits.astype(np.bool_, copy=False)


def boost_negative_visual_clusters(
    scores: np.ndarray,
    labels: np.ndarray,
    blocks: np.ndarray,
    *,
    boost: float,
    max_blocks: int,
) -> tuple[np.ndarray, int, int]:
    candidates = []
    for block in sorted(set(blocks.tolist())):
        idx = np.flatnonzero(blocks == block)
        negatives = idx[labels[idx] == 0]
        positives = idx[labels[idx] == 1]
        if len(negatives) < 2:
            continue
        purity_penalty = len(positives) / max(1, len(idx))
        candidates.append((float(np.max(scores[negatives])) - purity_penalty, block, negatives))
    candidates.sort(reverse=True, key=lambda item: item[0])
    chosen = candidates[:max_blocks]
    out = scores.copy()
    touched = 0
    for _, _, negatives in chosen:
        clipped = np.clip(out[negatives], 1e-5, 1 - 1e-5)
        out[negatives] = expit(logit(clipped) + boost)
        touched += len(negatives)
    return out, len(chosen), touched


def mean_pairwise_similarity(embed: np.ndarray, selected: np.ndarray) -> float:
    selected = np.asarray(selected, dtype=int)
    if len(selected) < 2:
        return 0.0
    sim = cosine_similarity(embed[selected])
    tri = np.triu_indices(len(selected), k=1)
    return float(np.mean(sim[tri]))


def evaluate(
    labels: np.ndarray,
    scores: np.ndarray,
    blocks: np.ndarray,
    embed: np.ndarray,
    selected: np.ndarray,
    q: float,
) -> dict[str, float]:
    selected = np.asarray(selected, dtype=int)
    n_selected = int(len(selected))
    if n_selected == 0:
        return {
            "selected_count": 0.0,
            "true_hits": 0.0,
            "false_hits": 0.0,
            "fdp": 0.0,
            "fdp_exceeds_q": 0.0,
            "precision": 0.0,
            "unique_blocks": 0.0,
            "max_block_share": 0.0,
            "mean_pairwise_similarity": 0.0,
        }
    hits = int(np.sum(labels[selected]))
    false = int(n_selected - hits)
    _, counts = np.unique(blocks[selected], return_counts=True)
    fdp = false / n_selected
    return {
        "selected_count": float(n_selected),
        "true_hits": float(hits),
        "false_hits": float(false),
        "fdp": float(fdp),
        "fdp_exceeds_q": float(fdp > q),
        "precision": float(hits / n_selected),
        "unique_blocks": float(len(counts)),
        "max_block_share": float(counts.max() / n_selected),
        "mean_pairwise_similarity": mean_pairwise_similarity(embed, selected),
    }


def select_method(
    method: str,
    *,
    scores: np.ndarray,
    pvalues: np.ndarray,
    blocks: np.ndarray,
    bits: np.ndarray,
    q: float,
    budget: int,
    seed_parts: tuple[object, ...],
) -> np.ndarray:
    if is_diversity_method(method):
        return select_diversity(
            method,
            scores=scores,
            bits=bits,
            budget=budget,
            seed=stable_seed(*seed_parts, method),
        ).selected
    return select(method, scores=scores, pvalues=pvalues, blocks=blocks, q=q, budget=budget, block_size_cap=2).selected


def run_panel(panel: dict[str, object], seeds: list[int], budgets: list[str], q_values: list[float], methods: list[str]) -> list[dict]:
    features = np.asarray(panel["features"], dtype=np.float32)
    labels = np.asarray(panel["labels"], dtype=int)
    target_names = list(panel["target_names"])
    rows = []
    for seed in seeds:
        train_idx, calib_idx, test_idx = split_indices(labels, seed)
        calib_proba, test_proba, _, test_embed, classes = classifier_scores(features, labels, train_idx, calib_idx, test_idx, seed)
        blocks = visual_blocks(test_embed, seed)
        bits = bits_from_embedding(test_embed)
        for class_pos, target_label in enumerate(classes):
            target_name = target_names[int(target_label)] if int(target_label) < len(target_names) else str(target_label)
            calib_binary = (labels[calib_idx] == target_label).astype(np.int8)
            test_binary = (labels[test_idx] == target_label).astype(np.int8)
            calib_scores = calib_proba[:, class_pos]
            base_scores = test_proba[:, class_pos]
            null_scores = calib_scores[calib_binary == 0]
            if len(null_scores) == 0 or np.sum(test_binary) == 0:
                continue
            scenarios = [("natural", base_scores, 0, 0)]
            boosted, n_blocks, n_touched = boost_negative_visual_clusters(
                base_scores,
                test_binary,
                blocks,
                boost=2.5,
                max_blocks=max(1, min(5, len(set(blocks.tolist())) // 5)),
            )
            scenarios.append(("visual_cluster_confounder", boosted, n_blocks, n_touched))
            for scenario, scores, confounder_blocks, confounder_negatives in scenarios:
                pvalues = conformal_right_tail_pvalues(scores, null_scores)
                for q in q_values:
                    for budget_label in budgets:
                        budget = parse_budget(budget_label, len(scores))
                        for method in methods:
                            selected = select_method(
                                method,
                                scores=scores,
                                pvalues=pvalues,
                                blocks=blocks,
                                bits=bits,
                                q=q,
                                budget=budget,
                                seed_parts=(panel["dataset"], seed, target_label, scenario, budget_label, q),
                            )
                            metrics = evaluate(test_binary, scores, blocks, test_embed, selected, q)
                            rows.append(
                                {
                                    "dataset": panel["dataset"],
                                    "source": panel["source"],
                                    "scenario": scenario,
                                    "seed": seed,
                                    "target_label": int(target_label),
                                    "target_name": target_name,
                                    "q": q,
                                    "budget_label": str(budget_label),
                                    "budget": budget,
                                    "method": method,
                                    "n_train": int(len(train_idx)),
                                    "n_calibration": int(len(calib_idx)),
                                    "n_testpool": int(len(test_idx)),
                                    "n_testpool_positive": int(np.sum(test_binary)),
                                    "n_visual_blocks": int(len(set(blocks.tolist()))),
                                    "confounder_blocks": int(confounder_blocks),
                                    "confounder_negative_images": int(confounder_negatives),
                                    "pvalue_floor": float(1.0 / (len(null_scores) + 1.0)),
                                    "pvalue_min": float(np.min(pvalues)),
                                    **metrics,
                                }
                            )
    return rows


def dominates(a: pd.Series, b: pd.Series) -> bool:
    strict = False
    for metric in HIGHER_IS_BETTER:
        if a[metric] < b[metric] - 1e-12:
            return False
        strict = strict or a[metric] > b[metric] + 1e-12
    for metric in LOWER_IS_BETTER:
        if a[metric] > b[metric] + 1e-12:
            return False
        strict = strict or a[metric] < b[metric] - 1e-12
    return strict


def add_frontier(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "scenario", "seed", "target_label", "q", "budget_label"]
    rows = []
    for _, group in results.groupby(keys, sort=False):
        group = group.copy().reset_index(drop=True)
        flags = []
        for i, row in group.iterrows():
            flags.append(not any(dominates(other, row) for j, other in group.iterrows() if j != i))
        group["pareto_member"] = flags
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "scenario", "q", "budget_label", "method"]
    metrics = METRICS + ["fdp_exceeds_q", "pareto_member", "confounder_blocks", "confounder_negative_images"]
    mean = results.groupby(keys, as_index=False)[metrics].mean()
    std = results.groupby(keys, as_index=False)[metrics].std().rename(columns={m: f"{m}_std" for m in metrics})
    n = results.groupby(keys, as_index=False).size().rename(columns={"size": "n_tasks"})
    return mean.merge(std, on=keys, how="left").merge(n, on=keys, how="left")


def paired_tests(results: pd.DataFrame, baseline: str = "score_cap1") -> pd.DataFrame:
    keys = ["dataset", "scenario", "seed", "target_label", "q", "budget_label"]
    rows = []
    base = results[results["method"] == baseline][keys + METRICS]
    for method in sorted(set(results["method"]) - {baseline}):
        cur = results[results["method"] == method][keys + METRICS]
        merged = cur.merge(base, on=keys, suffixes=("", "_baseline"))
        for metric in ["true_hits", "fdp", "unique_blocks", "max_block_share", "mean_pairwise_similarity"]:
            diff = merged[metric] - merged[f"{metric}_baseline"]
            try:
                stat, pvalue = wilcoxon(diff)
            except ValueError:
                stat, pvalue = np.nan, np.nan
            rows.append(
                {
                    "baseline": baseline,
                    "method": method,
                    "metric": metric,
                    "mean_delta": float(diff.mean()),
                    "median_delta": float(diff.median()),
                    "wilcoxon_statistic": float(stat) if math.isfinite(stat) else np.nan,
                    "wilcoxon_pvalue": float(pvalue) if math.isfinite(pvalue) else np.nan,
                    "n_pairs": int(len(diff)),
                }
            )
    return pd.DataFrame(rows)


def verdict(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in summary.groupby(["dataset", "scenario", "q", "budget_label"], sort=False):
        chem = group[group["method"].str.startswith("chemdeprc")]
        best_hits = group.sort_values(["true_hits", "fdp"], ascending=[False, True]).iloc[0]
        lowest_fdp = group.sort_values(["fdp", "true_hits"], ascending=[True, False]).iloc[0]
        best_chem = chem.sort_values(["true_hits", "fdp"], ascending=[False, True]).iloc[0] if len(chem) else None
        rows.append(
            {
                "dataset": key[0],
                "scenario": key[1],
                "q": key[2],
                "budget_label": key[3],
                "best_true_hits_method": best_hits["method"],
                "best_true_hits": best_hits["true_hits"],
                "lowest_fdp_method": lowest_fdp["method"],
                "lowest_fdp": lowest_fdp["fdp"],
                "best_chemdeprc_method": best_chem["method"] if best_chem is not None else "",
                "best_chemdeprc_true_hits": best_chem["true_hits"] if best_chem is not None else np.nan,
                "best_chemdeprc_fdp": best_chem["fdp"] if best_chem is not None else np.nan,
                "chemdeprc_non_dominated_rate": float(chem["pareto_member"].max()) if len(chem) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data/pr_native_retrieval")
    parser.add_argument("--table-outdir", default="tables")
    parser.add_argument("--data-home", default="data/pr_native_raw")
    parser.add_argument("--seeds", nargs="+", type=int, default=[3500, 3501, 3502, 3503, 3504])
    parser.add_argument("--budgets", nargs="+", default=["25", "50", "100"])
    parser.add_argument("--q-values", nargs="+", type=float, default=[0.7])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    table_outdir = root / args.table_outdir
    panels = dataset_panels(root / args.data_home)
    rows = []
    for panel in tqdm(panels, desc="PR-native image retrieval panels"):
        rows.extend(run_panel(panel, args.seeds, args.budgets, args.q_values, args.methods))
    results = add_frontier(pd.DataFrame(rows))
    summary = summarize(results)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = outdir / f"pr_native_retrieval_results_{timestamp}.csv"
    summary_path = outdir / f"pr_native_retrieval_summary_{timestamp}.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    results.to_csv(outdir / "latest_results.csv", index=False)
    summary.to_csv(outdir / "latest_summary.csv", index=False)

    q_label = str(args.q_values[0]).replace(".", "p") if len(args.q_values) == 1 else "all"
    write_pair(summary, table_outdir, f"pr_native_retrieval_summary_q{q_label}")
    overall = (
        summary.groupby(["method"], as_index=False)[
            ["true_hits", "fdp", "unique_blocks", "max_block_share", "mean_pairwise_similarity", "pareto_member"]
        ]
        .mean()
        .sort_values(["pareto_member", "true_hits", "fdp"], ascending=[False, False, True])
    )
    write_pair(overall, table_outdir, f"pr_native_retrieval_overall_q{q_label}")
    write_pair(paired_tests(results), table_outdir, f"pr_native_retrieval_paired_tests_vs_scorecap_q{q_label}")
    write_pair(verdict(summary), table_outdir, f"pr_native_retrieval_verdict_q{q_label}")

    metadata = {
        "created_utc": timestamp,
        "datasets": [{"dataset": p["dataset"], "source": p["source"]} for p in panels],
        "seeds": args.seeds,
        "budgets": args.budgets,
        "q_values": args.q_values,
        "methods": args.methods,
        "protocol": "Classifier confidence gives candidate scores; unsupervised visual embedding clusters define dependency blocks; target labels are used only for evaluation and for the explicit confounder stress construction.",
        "results": str(results_path.relative_to(root)),
        "summary": str(summary_path.relative_to(root)),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
