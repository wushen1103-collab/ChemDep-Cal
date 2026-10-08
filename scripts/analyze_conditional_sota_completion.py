#!/usr/bin/env python3
"""Target-cluster comparisons for the frozen conditional-SOTA audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["true_hits", "fdp", "unique_blocks", "max_block_share", "selected_count"]


def target_comparisons(frame: pd.DataFrame, dataset: str, reference: str, *, seed: int) -> pd.DataFrame:
    target_mean = frame.groupby(["target_chembl_id", "method"], as_index=False)[METRICS].mean()
    targets = sorted(target_mean["target_chembl_id"].unique())
    methods = sorted(target_mean["method"].unique())
    draws = np.random.default_rng(seed).integers(0, len(targets), size=(20000, len(targets)))
    records = []
    for metric in METRICS:
        wide = target_mean.pivot(index="target_chembl_id", columns="method", values=metric).reindex(targets)
        for comparator in methods:
            if comparator == reference:
                continue
            delta = (wide[reference] - wide[comparator]).to_numpy(dtype=float)
            if np.isnan(delta).any():
                raise AssertionError(f"Missing target-method pair: {dataset} {metric} {comparator}")
            samples = delta[draws].mean(axis=1)
            lo, hi = np.quantile(samples, [0.025, 0.975])
            records.append({
                "dataset": dataset,
                "reference": reference,
                "comparator": comparator,
                "metric": metric,
                "n_targets": len(targets),
                "mean_delta": delta.mean(),
                "target_sd_delta": delta.std(ddof=1),
                "ci95_low": lo,
                "ci95_high": hi,
                "target_wins": int((delta > 0).sum()),
                "target_losses": int((delta < 0).sum()),
            })
    return pd.DataFrame(records)


def publication_target_summary(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    target_mean = frame.groupby(["target_chembl_id", "method"], as_index=False)[METRICS].mean()
    out = target_mean.groupby("method", as_index=False)[METRICS].agg(["mean", "std"])
    out.columns = ["method"] + [
        f"{metric}_target_{stat}" for metric, stat in out.columns.to_flat_index()[1:]
    ]
    out.insert(0, "dataset", dataset)
    out.insert(2, "n_targets", target_mean["target_chembl_id"].nunique())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cols = [
        "target_chembl_id", "rep", "artifact_fraction", "logit_boost", "q",
        "block_perturbation", "budget_label", "method", *METRICS,
    ]
    original_path = root / "data/jctc_budget_phase/latest_results.csv"
    original = pd.read_csv(original_path, usecols=cols)
    original = original[
        np.isclose(original["q"], 0.7)
        & (original["budget_label"].astype(str) == "50")
        & (original["block_perturbation"] == "none")
    ].copy()
    main = original[
        np.isclose(original["artifact_fraction"], 0.1)
        & np.isclose(original["logit_boost"], 4.0)
    ].copy()
    opdiv_path = root / "data/conditional_sota_opdiv_main_20261003/opdiv_raw.csv"
    opdiv = pd.read_csv(opdiv_path)
    assert len(opdiv) == 1500 and opdiv["full_selection"].all()
    assert (opdiv["status"] == "optimal").all()
    key = ["target_chembl_id", "rep"]
    raw_sanity = main[main["method"] == "raw_top_b"][key + ["true_hits"]]
    joined = opdiv.merge(raw_sanity, on=key, suffixes=("_opdiv", "_original"), validate="many_to_one")
    assert len(joined) == 1500
    assert (joined["raw_sanity_hits"] == joined["true_hits_original"]).all()
    adaptive_path = root / "tables/conditional_sota_adaptive_cell_20261003/deployed_adaptive_B50_hard_cells.csv"
    adaptive = pd.read_csv(adaptive_path)
    assert len(adaptive) == 3000
    adaptive["method"] = "adaptive_tree"
    adaptive_main = adaptive[
        np.isclose(adaptive["artifact_fraction"], 0.1)
        & np.isclose(adaptive["logit_boost"], 4.0)
    ]
    assert len(adaptive_main) == 500
    main_with_opdiv = pd.concat([
        main, opdiv.reindex(columns=main.columns),
        adaptive_main.reindex(columns=main.columns),
    ], ignore_index=True)

    hard_original = original[
        (original["artifact_fraction"] >= 0.1)
        & (original["logit_boost"] >= 4.0)
    ].copy()
    hard = pd.concat([hard_original, adaptive.reindex(columns=hard_original.columns)], ignore_index=True)
    count = hard.groupby("method").size()
    assert count.nunique() == 1 and count.iloc[0] == 3000
    for dataset, frame in [
        ("chembl_main_B50_f0p1_b4", main_with_opdiv),
        ("chembl_hard_B50_6scenarios", hard),
    ]:
        summary = frame.groupby("method", as_index=False)[METRICS].agg(["mean", "std"])
        summary.columns = ["method"] + [f"{metric}_{stat}" for metric, stat in summary.columns.to_flat_index()[1:]]
        summary.to_csv(outdir / f"{dataset}_summary.csv", index=False)
        comp = target_comparisons(frame, dataset, "chemdeprc_cap1", seed=353501)
        comp.to_csv(outdir / f"{dataset}_paired_target.csv", index=False)
        target_comparisons(frame, dataset, "adaptive_tree", seed=353503).to_csv(
            outdir / f"{dataset}_adaptive_paired_target.csv", index=False,
        )

    publication = [
        publication_target_summary(main_with_opdiv, "ChEMBL_main_B50"),
        publication_target_summary(hard, "ChEMBL_hard_B50"),
    ]
    revision_path = root / "data/pr_figure_reproduction/jctc_revision_mechanism_results_20261001T153530Z.csv"
    revision = pd.read_csv(revision_path, usecols=cols)
    revision = revision[
        np.isclose(revision["q"], 0.7)
        & (revision["budget_label"].astype(str) == "50")
        & (revision["block_perturbation"] == "none")
        & np.isclose(revision["artifact_fraction"], 0.1)
        & np.isclose(revision["logit_boost"], 4.0)
    ].copy()
    revision_opdiv_path = root / "data/conditional_sota_opdiv_revision_main_20261003/opdiv_raw.csv"
    revision_opdiv = pd.read_csv(revision_opdiv_path)
    assert len(revision_opdiv) == 1500
    assert revision_opdiv["full_selection"].all() and (revision_opdiv["status"] == "optimal").all()
    revision_scorecap = revision[revision["method"] == "score_cap1"][key + ["true_hits"]]
    revision_check = revision_opdiv.merge(
        revision_scorecap, on=key, suffixes=("_opdiv", "_original"), validate="many_to_one",
    )
    assert len(revision_check) == 1500
    assert (revision_check["raw_sanity_hits"] == revision_check["true_hits_original"]).all()
    revision = pd.concat([revision, revision_opdiv.reindex(columns=revision.columns)], ignore_index=True)
    revision.to_csv(outdir / "revision_main_B50_method_cells.csv", index=False)
    publication.append(publication_target_summary(revision, "ChEMBL_revision_main_B50"))
    target_comparisons(
        revision, "ChEMBL_revision_main_B50", "chemdeprc_cap1", seed=353504,
    ).to_csv(outdir / "ChEMBL_revision_main_B50_paired_target.csv", index=False)
    revision.groupby(["target_chembl_id", "method"], as_index=False)[METRICS].mean().to_csv(
        outdir / "ChEMBL_revision_main_B50_target_means.csv", index=False,
    )
    external = []
    completeness = []
    for name, path in [
        ("LIT_PCBA_full", root / "data/conditional_sota_litpcba_full_20261003/raw.csv"),
        ("DUDE_full", root / "data/conditional_sota_dude_full_20261003/raw.csv"),
        ("LIT_PCBA_ligand", root / "data/conditional_sota_litpcba_ligand_20261003/raw.csv"),
        ("DUDE_ligand", root / "data/conditional_sota_dude_ligand_20261003/raw.csv"),
    ]:
        frame = pd.read_csv(path)
        assert frame["target_chembl_id"].nunique() == 8
        assert frame["rep"].nunique() == 5
        summary = frame.groupby("method", as_index=False)[METRICS].agg(["mean", "std"])
        summary.columns = ["method"] + [f"{metric}_{stat}" for metric, stat in summary.columns.to_flat_index()[1:]]
        summary.insert(0, "dataset", name)
        external.append(summary)
        publication.append(publication_target_summary(frame, name))
        coverage = frame.groupby("method", as_index=False).agg(
            n_rows=("rep", "size"),
            n_empty=("selected_count", lambda s: int((s == 0).sum())),
            n_full=("selected_count", lambda s: int((s == 50).sum())),
            mean_selected=("selected_count", "mean"),
            mean_false_hits=("false_hits", "mean"),
        )
        nonempty = frame[frame["selected_count"] > 0].groupby("method")["fdp"].mean()
        coverage["mean_fdp_nonempty"] = coverage["method"].map(nonempty)
        coverage.insert(0, "dataset", name)
        completeness.append(coverage)
        target_comparisons(frame, name, "chemdeprc_cap1", seed=353502).to_csv(
            outdir / f"{name}_paired_target.csv", index=False,
        )
        frame.groupby(["target_chembl_id", "method"], as_index=False)[METRICS].mean().to_csv(
            outdir / f"{name}_target_means.csv", index=False,
        )
    pd.concat(external, ignore_index=True).to_csv(outdir / "external_summary.csv", index=False)
    pd.concat(completeness, ignore_index=True).to_csv(
        outdir / "external_selection_completeness.csv", index=False,
    )
    pd.concat(publication, ignore_index=True).to_csv(outdir / "publication_target_mean_std.csv", index=False)
    manifest = {
        "original_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
        "opdiv_sha256": hashlib.sha256(opdiv_path.read_bytes()).hexdigest(),
        "revision_source_sha256": hashlib.sha256(revision_path.read_bytes()).hexdigest(),
        "revision_opdiv_sha256": hashlib.sha256(revision_opdiv_path.read_bytes()).hexdigest(),
        "unit_of_inference": "target; 50 ChEMBL repeats or 5 external repeats are within-target measurements",
        "main_cell": "q=0.7, B=50, artifact fraction=0.1, boost=4",
        "hard_aggregate": "q=0.7, B=50, artifact fraction >=0.1, boost >=4 (six scenarios, post-hoc sensitivity)",
        "resampling": "20,000 paired target-cluster bootstrap draws, percentile 95% CI",
        "warning": "All labels are retrospective; artifact injection is simulated and depends on inactive labels; no prospective FDR or wet-lab claim.",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Wrote target-cluster comparisons to", outdir)


if __name__ == "__main__":
    main()
