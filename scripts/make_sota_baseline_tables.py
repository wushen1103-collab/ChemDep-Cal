#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


KEY_METHODS = [
    "raw_top_b",
    "bh",
    "weighted_bh",
    "by",
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
]

REPORT_METRICS = [
    "selected_count",
    "true_hits",
    "fdp",
    "fdp_exceeds_q",
    "precision",
    "unique_blocks",
    "max_block_share",
    "mean_pairwise_tanimoto",
]

HIGHER_IS_BETTER = ["true_hits", "unique_blocks"]
LOWER_IS_BETTER = ["fdp", "fdp_exceeds_q", "max_block_share", "mean_pairwise_tanimoto"]
PAIRED_BASELINES = ["bh", "weighted_bh", "block_bh", "score_cap1", "bh_cap1", "bh_soft75"]
PAIRED_METRICS = ["true_hits", "fdp", "fdp_exceeds_q", "unique_blocks", "max_block_share", "mean_pairwise_tanimoto"]


def write_pair(df: pd.DataFrame, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(df.to_markdown(index=False, floatfmt=".4f") + "\n", encoding="utf-8")


def method_family(method: str) -> str:
    if method.startswith("chemdeprc"):
        return "ChemDep-RC"
    if method.startswith("bh_cap") or method.startswith("bh_soft"):
        return "BH-posthoc"
    if method.startswith("score_cap") or method == "raw_top_b":
        return "score-only"
    if method in {"random_b"}:
        return "classic-control"
    if method.startswith("score_leader") or method.startswith("score_maxmin") or method.startswith("score_mmr") or method == "score_dpp":
        return "external-diversity"
    if method in {"bh", "by", "weighted_bh", "e_bh"}:
        return "individual-FDR"
    if method == "block_bh":
        return "block-FDR"
    return "other"


def attach_bh_deltas(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    base_cols = keys + REPORT_METRICS
    bh = df[df["method"] == "bh"][base_cols].copy()
    merged = df.merge(bh, on=keys, suffixes=("", "_bh"), how="left")
    for metric in REPORT_METRICS:
        merged[f"delta_{metric}_vs_bh"] = merged[metric] - merged[f"{metric}_bh"]
    drop_cols = [f"{metric}_bh" for metric in REPORT_METRICS]
    return merged.drop(columns=drop_cols)


def dominates(row: pd.Series, other: pd.Series) -> bool:
    no_worse = True
    strict = False
    for metric in HIGHER_IS_BETTER:
        if pd.isna(row[metric]) or pd.isna(other[metric]):
            return False
        if row[metric] < other[metric] - 1e-12:
            no_worse = False
        if row[metric] > other[metric] + 1e-12:
            strict = True
    for metric in LOWER_IS_BETTER:
        if pd.isna(row[metric]) or pd.isna(other[metric]):
            return False
        if row[metric] > other[metric] + 1e-12:
            no_worse = False
        if row[metric] < other[metric] - 1e-12:
            strict = True
    return no_worse and strict


def add_pareto_flags(df: pd.DataFrame, unit_col: str = "evaluation_unit") -> pd.DataFrame:
    flagged = []
    for _, group in df.groupby(unit_col, sort=False):
        group = group.copy().reset_index(drop=True)
        non_dominated = []
        dominated_by_cap1 = []
        cap1 = group[group["method"] == "chemdeprc_cap1"]
        cap1_row = cap1.iloc[0] if len(cap1) else None
        for i, row in group.iterrows():
            is_dominated = any(dominates(other, row) for j, other in group.iterrows() if j != i)
            non_dominated.append(not is_dominated)
            dominated_by_cap1.append(bool(cap1_row is not None and row["method"] != "chemdeprc_cap1" and dominates(row, cap1_row)))
        group["non_dominated"] = non_dominated
        group["dominates_chemdeprc_cap1"] = dominated_by_cap1
        flagged.append(group)
    return pd.concat(flagged, ignore_index=True) if flagged else df


def compact_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "evaluation_unit",
        "method",
        "method_family",
        "non_dominated",
        "dominates_chemdeprc_cap1",
        *REPORT_METRICS,
        "delta_true_hits_vs_bh",
        "delta_fdp_vs_bh",
        "delta_fdp_exceeds_q_vs_bh",
        "delta_unique_blocks_vs_bh",
        "delta_max_block_share_vs_bh",
        "delta_mean_pairwise_tanimoto_vs_bh",
    ]
    return df[[c for c in cols if c in df.columns]].sort_values(["evaluation_unit", "method_family", "method"])


def prepare_synthetic(path: Path, q: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["scenario"] == "block_label_noise") & (np.isclose(df["q"], q)) & df["method"].isin(KEY_METHODS)].copy()
    df["evaluation_unit"] = "synthetic:block_label_noise:B" + df["budget_label"].astype(str)
    df["method_family"] = df["method"].map(method_family)
    df = attach_bh_deltas(df, ["scenario", "q", "budget_label", "budget"])
    return compact_columns(add_pareto_flags(df))


def prepare_chembl(path: Path, q: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(np.isclose(df["q"], q)) & df["method"].isin(KEY_METHODS)].copy()
    df["evaluation_unit"] = "chembl_routine:B" + df["budget_label"].astype(str)
    df["method_family"] = df["method"].map(method_family)
    df = attach_bh_deltas(df, ["q", "budget_label", "budget"])
    return compact_columns(add_pareto_flags(df))


def prepare_artifact(path: Path, q: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(np.isclose(df["q"], q)) & df["method"].isin(KEY_METHODS)].copy()
    df["evaluation_unit"] = (
        "chembl_artifact:f"
        + df["artifact_fraction"].map(lambda x: f"{x:g}")
        + "_boost"
        + df["logit_boost"].map(lambda x: f"{x:g}")
        + ":B"
        + df["budget_label"].astype(str)
    )
    df["method_family"] = df["method"].map(method_family)
    df = attach_bh_deltas(df, ["stress_label", "artifact_fraction", "logit_boost", "q", "budget_label"])
    return compact_columns(add_pareto_flags(df))


def verdict_table(frontier: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for unit, group in frontier.groupby("evaluation_unit", sort=False):
        chem = group[group["method_family"] == "ChemDep-RC"].copy()
        best_chem_hits = chem["true_hits"].max() if len(chem) else np.nan
        best_chem = chem.sort_values(
            ["true_hits", "fdp", "unique_blocks"],
            ascending=[False, True, False],
        ).iloc[0] if len(chem) else None
        best_hits = group.sort_values(["true_hits", "fdp"], ascending=[False, True]).iloc[0]
        lowest_fdp = group.sort_values(["fdp", "true_hits"], ascending=[True, False]).iloc[0]
        max_unique = group.sort_values(["unique_blocks", "true_hits"], ascending=[False, False]).iloc[0]
        chem_non_dominated = bool(chem["non_dominated"].any()) if len(chem) else False
        rows.append(
            {
                "evaluation_unit": unit,
                "best_true_hits_method": best_hits["method"],
                "best_true_hits": best_hits["true_hits"],
                "best_chemdeprc_method": best_chem["method"] if best_chem is not None else "",
                "best_chemdeprc_true_hits": best_chem_hits,
                "hit_gap_best_minus_chemdeprc": best_hits["true_hits"] - best_chem_hits,
                "lowest_fdp_method": lowest_fdp["method"],
                "lowest_fdp": lowest_fdp["fdp"],
                "max_unique_blocks_method": max_unique["method"],
                "max_unique_blocks": max_unique["unique_blocks"],
                "chemdeprc_family_non_dominated": chem_non_dominated,
                "n_methods_dominating_chemdeprc_cap1": int(group["dominates_chemdeprc_cap1"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _best_chemdeprc_by_unit(frontier: pd.DataFrame) -> dict[str, str]:
    best: dict[str, str] = {}
    for unit, group in frontier[frontier["method_family"] == "ChemDep-RC"].groupby("evaluation_unit", sort=False):
        row = group.sort_values(["true_hits", "fdp", "unique_blocks"], ascending=[False, True, False]).iloc[0]
        best[str(unit)] = str(row["method"])
    return best


def _paired_rows(
    df: pd.DataFrame,
    *,
    pair_keys: list[str],
    best_methods: dict[str, str],
    source: str,
) -> list[dict]:
    rows: list[dict] = []
    for unit, method in best_methods.items():
        unit_df = df[df["evaluation_unit"] == unit]
        method_df = unit_df[unit_df["method"] == method]
        if method_df.empty:
            continue
        baselines = [m for m in PAIRED_BASELINES if m != method and m in set(unit_df["method"])]
        for baseline in baselines:
            paired = method_df.merge(
                unit_df[unit_df["method"] == baseline],
                on=pair_keys,
                suffixes=("", "_baseline"),
                how="inner",
            )
            for metric in PAIRED_METRICS:
                diff = paired[metric] - paired[f"{metric}_baseline"]
                if len(diff) == 0 or np.allclose(diff, 0):
                    stat = np.nan
                    pvalue = np.nan
                else:
                    stat, pvalue = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                rows.append(
                    {
                        "source": source,
                        "evaluation_unit": unit,
                        "method": method,
                        "baseline": baseline,
                        "metric": metric,
                        "mean_delta": float(diff.mean()) if len(diff) else np.nan,
                        "median_delta": float(diff.median()) if len(diff) else np.nan,
                        "wilcoxon_stat": stat,
                        "wilcoxon_p": pvalue,
                        "n_pairs": int(len(diff)),
                    }
                )
    return rows


def paired_tests(
    *,
    synthetic_results: Path,
    chembl_results: Path,
    artifact_results: Path,
    frontier: pd.DataFrame,
    q: float,
) -> pd.DataFrame:
    best_methods = _best_chemdeprc_by_unit(frontier)
    rows: list[dict] = []
    if synthetic_results.exists():
        df = pd.read_csv(synthetic_results)
        df = df[(df["scenario"] == "block_label_noise") & np.isclose(df["q"], q) & df["method"].isin(KEY_METHODS)].copy()
        df["evaluation_unit"] = "synthetic:block_label_noise:B" + df["budget_label"].astype(str)
        rows.extend(
            _paired_rows(
                df,
                pair_keys=["scenario", "rep", "q", "budget_label", "budget"],
                best_methods=best_methods,
                source="synthetic",
            )
        )
    if chembl_results.exists():
        df = pd.read_csv(chembl_results)
        df = df[np.isclose(df["q"], q) & df["method"].isin(KEY_METHODS)].copy()
        df["evaluation_unit"] = "chembl_routine:B" + df["budget_label"].astype(str)
        rows.extend(
            _paired_rows(
                df,
                pair_keys=["target_chembl_id", "q", "budget_label", "budget"],
                best_methods=best_methods,
                source="chembl_routine",
            )
        )
    if artifact_results.exists():
        df = pd.read_csv(artifact_results)
        df = df[np.isclose(df["q"], q) & df["method"].isin(KEY_METHODS)].copy()
        df["evaluation_unit"] = (
            "chembl_artifact:f"
            + df["artifact_fraction"].map(lambda x: f"{x:g}")
            + "_boost"
            + df["logit_boost"].map(lambda x: f"{x:g}")
            + ":B"
            + df["budget_label"].astype(str)
        )
        rows.extend(
            _paired_rows(
                df,
                pair_keys=[
                    "target_chembl_id",
                    "rep",
                    "stress_label",
                    "artifact_fraction",
                    "logit_boost",
                    "q",
                    "budget_label",
                ],
                best_methods=best_methods,
                source="chembl_artifact",
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-summary", default="data/synthetic_sota_baselines_block_noise_r200_n20000/latest_summary.csv")
    parser.add_argument("--synthetic-results", default="data/synthetic_sota_baselines_block_noise_r200_n20000/latest_results.csv")
    parser.add_argument("--chembl-summary", default="data/chembl_sota_baselines_potent_a7_i6/latest_summary.csv")
    parser.add_argument("--chembl-results", default="data/chembl_sota_baselines_potent_a7_i6/latest_results.csv")
    parser.add_argument(
        "--artifact-summary",
        default="data/chembl_block_artifact_sota_baselines_potent_a7_i6/latest_summary.csv",
    )
    parser.add_argument(
        "--artifact-results",
        default="data/chembl_block_artifact_sota_baselines_potent_a7_i6/latest_results.csv",
    )
    parser.add_argument("--outdir", default="tables")
    parser.add_argument("--q", type=float, default=0.7)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    tables = []
    source_specs = [
        ("sota_synthetic_baselines", root / args.synthetic_summary, prepare_synthetic),
        ("sota_chembl_routine_baselines", root / args.chembl_summary, prepare_chembl),
        ("sota_chembl_artifact_baselines", root / args.artifact_summary, prepare_artifact),
    ]
    for stem, path, prep in source_specs:
        if not path.exists():
            print(f"Skipping missing summary: {path}")
            continue
        table = prep(path, args.q)
        write_pair(table, outdir, f"{stem}_q{args.q:g}")
        tables.append(table)
    if tables:
        frontier = pd.concat(tables, ignore_index=True)
        write_pair(frontier, outdir, f"sota_pareto_frontier_q{args.q:g}")
        write_pair(verdict_table(frontier), outdir, f"sota_verdict_q{args.q:g}")
        tests = paired_tests(
            synthetic_results=root / args.synthetic_results,
            chembl_results=root / args.chembl_results,
            artifact_results=root / args.artifact_results,
            frontier=frontier,
            q=args.q,
        )
        if not tests.empty:
            write_pair(tests, outdir, f"sota_paired_tests_q{args.q:g}")
    print(f"Wrote SOTA baseline tables to {outdir}")


if __name__ == "__main__":
    main()
