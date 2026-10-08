#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from chemdeprc.metrics import evaluate_selection
from chemdeprc.selection import select
from chemdeprc.synthetic import SCENARIOS, make_campaign, null_conformal_pvalues, random_blocks_like


METHODS = (
    "raw_top_b",
    "bh",
    "by",
    "weighted_bh",
    "e_bh",
    "block_bh",
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
    "chemdeprc_cap1",
    "chemdeprc_cap2",
    "chemdeprc_cap5",
    "no_dependency_block",
    "random_block",
)


def parse_budget(value: str, n_test: int) -> int:
    value = value.strip().lower()
    if value.endswith("pct"):
        pct = float(value[:-3]) / 100.0
        return max(1, int(round(n_test * pct)))
    return int(value)


def run_campaign(args: tuple) -> list[dict]:
    scenario, rep, seed, n_test, n_calib, q_values, budgets, methods, cap = args
    campaign = make_campaign(scenario=scenario, seed=seed, n_test=n_test, n_calib=n_calib)
    pvalues = null_conformal_pvalues(campaign.scores, campaign.calib_scores, campaign.calib_labels)
    rows = []
    for q in q_values:
        for budget_label, budget in budgets:
            for method in methods:
                blocks = campaign.blocks
                actual_method = method
                if method == "random_block":
                    blocks = random_blocks_like(campaign.blocks, seed + 17_003)
                    actual_method = "chemdeprc_cap2"
                result = select(
                    actual_method,
                    scores=campaign.scores,
                    pvalues=pvalues,
                    blocks=blocks,
                    q=q,
                    budget=budget,
                    block_size_cap=cap,
                )
                metrics = evaluate_selection(
                    labels=campaign.labels,
                    scores=campaign.scores,
                    blocks=blocks,
                    selected=result.selected,
                    budget=budget,
                    q=q,
                    within_similarity=campaign.within_similarity,
                    between_similarity=campaign.between_similarity,
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "rep": rep,
                        "seed": seed,
                        "method": method,
                        "q": q,
                        "budget_label": budget_label,
                        "budget": budget,
                        "n_test": n_test,
                        "n_calib": n_calib,
                        "block_size_cap": _method_cap(method, cap),
                        "certificate": result.certificate,
                        **metrics,
                    }
                )
    return rows


def _method_cap(method: str, default_cap: int) -> int | None:
    if method.startswith("chemdeprc_cap"):
        return int(method.replace("chemdeprc_cap", ""))
    if method in {"chemdeprc", "random_block"}:
        return default_cap
    return None


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "fdp",
        "fdp_exceeds_q",
        "true_hits",
        "selected_count",
        "precision",
        "power",
        "enrichment_factor",
        "unique_blocks",
        "max_block_share",
        "mean_pairwise_tanimoto",
        "empty",
    ]
    grouped = df.groupby(["scenario", "q", "budget_label", "budget", "method"], as_index=False)[metrics]
    mean = grouped.mean()
    std = grouped.std().rename(columns={m: f"{m}_std" for m in metrics})
    keys = ["scenario", "q", "budget_label", "budget", "method"]
    out = mean.merge(std, on=keys, how="left")
    counts = df.groupby(keys, as_index=False).size()
    return out.merge(counts, on=keys, how="left")


def append_log(root: Path, metadata: dict, summary_path: Path, results_path: Path) -> None:
    log_path = root / "EXPERIMENT_LOG.md"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now} Synthetic dependency run\n\n")
        f.write("Metadata:\n\n")
        f.write("```json\n")
        f.write(json.dumps(metadata, indent=2, sort_keys=True))
        f.write("\n```\n\n")
        f.write(f"- Raw results: `{results_path}`\n")
        f.write(f"- Summary table: `{summary_path}`\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=3000)
    parser.add_argument("--n-calib", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 30))
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--q-values", nargs="+", type=float, default=[0.5, 0.7, 0.8, 0.9])
    parser.add_argument("--budgets", nargs="+", default=["50", "100", "1pct"])
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--block-size-cap", type=int, default=2)
    parser.add_argument("--outdir", default="data/synthetic")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    budgets = [(b, parse_budget(b, args.n_test)) for b in args.budgets]
    max_workers = max(1, min(args.workers, max(1, (os.cpu_count() or 4) - 30)))

    tasks = []
    for scenario in args.scenarios:
        for rep in range(args.reps):
            seed = args.seed + rep + 10_000 * list(SCENARIOS).index(scenario)
            tasks.append(
                (
                    scenario,
                    rep,
                    seed,
                    args.n_test,
                    args.n_calib,
                    tuple(args.q_values),
                    tuple(budgets),
                    tuple(args.methods),
                    args.block_size_cap,
                )
            )

    all_rows: list[dict] = []
    with cf.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_campaign, task) for task in tasks]
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="synthetic"):
            all_rows.extend(fut.result())

    df = pd.DataFrame(all_rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = outdir / f"synthetic_results_{stamp}.csv"
    summary_path = outdir / f"synthetic_summary_{stamp}.csv"
    latest_results = outdir / "latest_results.csv"
    latest_summary = outdir / "latest_summary.csv"
    df.to_csv(results_path, index=False)
    df.to_csv(latest_results, index=False)
    summary = summarize(df)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(latest_summary, index=False)

    metadata = {
        "reps": args.reps,
        "n_test": args.n_test,
        "n_calib": args.n_calib,
        "seed": args.seed,
        "workers_requested": args.workers,
        "workers_used": max_workers,
        "scenarios": args.scenarios,
        "q_values": args.q_values,
        "budgets": [{"label": label, "value": value} for label, value in budgets],
        "methods": args.methods,
        "block_size_cap": args.block_size_cap,
    }
    append_log(root, metadata, summary_path.relative_to(root), results_path.relative_to(root))
    print(summary.to_markdown(index=False, floatfmt=".4f"))
    print(f"\nWrote {results_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
