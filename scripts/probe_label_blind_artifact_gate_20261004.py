#!/usr/bin/env python3
"""Exploratory score-shift gate; scenario detection is not activity prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SOURCES = {
    "development_third": "data/chembl_meta_gate_third_cohort_xgb_20261003/scores",
    "development_fourth": "data/chembl_meta_ungated_fourth_xgb_20261003/seed35401/scores",
    "inspected_fifth": "data/chembl_meta_ungated_fifth_xgb_20261003/seed35501/scores",
}


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def statistics(cal_scores: np.ndarray, test_scores: np.ndarray,
               cal_blocks: np.ndarray, test_blocks: np.ndarray) -> dict[str, float]:
    cal_logit, test_logit = logit(cal_scores), logit(test_scores)
    q95, q99 = np.quantile(cal_logit, [.95, .99])

    def coherent_tail(values: np.ndarray, blocks: np.ndarray, threshold: float) -> float:
        frame = pd.DataFrame({"block": blocks, "logit": values})
        grouped = frame.groupby("block").logit.agg(["count", "median"])
        group = grouped[grouped["count"].ge(2)]
        return float(group["median"].gt(threshold).mean()) if len(group) else 0.0

    return {
        "q95_logit_shift": float(np.quantile(test_logit, .95) - q95),
        "q99_logit_shift": float(np.quantile(test_logit, .99) - q99),
        "q99_exceedance": float(np.mean(test_logit > q99)),
        "coherent_q95_shift": coherent_tail(test_logit, test_blocks, q95)
                               - coherent_tail(cal_logit, cal_blocks, q95),
    }


def run_panel(panel: dict, source: str, reps: int) -> list[dict]:
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact

    cal = panel["cal"]
    test = panel["test"]
    base = test.score.to_numpy(float)
    cal_scores = cal.score.to_numpy(float)
    blocks = panel["test_blocks"]
    common = {"source": source, "target": panel["target"]}
    rows = [{**common, "scenario": "natural", "rep": -1, "fraction": 0.,
             "boost": 0., **statistics(cal_scores, base, panel["cal_blocks"], blocks)}]
    target_hash = int(hashlib.sha256(panel["target"].encode()).hexdigest()[:12], 16) % 1_000_000
    for rep in range(reps):
        for fraction in (.2, .3):
            rng = np.random.default_rng(20261004 + target_hash + rep * 1009
                                        + int(fraction * 10000))
            artifacts = choose_artifact_blocks(
                test, artifact_block_col="murcko_scaffold", rng=rng,
                artifact_fraction=fraction, max_artifact_blocks=25,
                min_inactive_per_block=2)
            for boost in (4., 6.):
                changed, _, _ = inject_artifact(
                    test, base, artifacts, artifact_block_col="murcko_scaffold",
                    logit_boost=boost)
                rows.append({**common, "scenario": "injected", "rep": rep,
                             "fraction": fraction, "boost": boost,
                             **statistics(cal_scores, changed, panel["cal_blocks"], blocks)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from train_artifact_meta_ranker_20261003 import load_panel

    rows = []
    for source, relative in SOURCES.items():
        files = sorted((root / relative).glob("*_scores.csv"))
        if len(files) < 8:
            raise AssertionError(f"Unexpected ChEMBL panel count: {source}")
        for file in files:
            rows.extend(run_panel(load_panel(file), source, args.reps))
        print(source, len(files), "panels completed", flush=True)
    natural_files = sorted((root / "data/public_moleculenet_ecfp_xgb/scores").glob(
        "*_scaffold_ood_seed3500_scores.csv"))
    for file in natural_files:
        panel = load_panel(file)
        rows.append({"source": "external_natural", "target": panel["target"],
                     "scenario": "natural", "rep": -1, "fraction": 0., "boost": 0.,
                     **statistics(panel["cal"].score.to_numpy(float),
                                  panel["test"].score.to_numpy(float),
                                  panel["cal_blocks"], panel["test_blocks"])})
    frame = pd.DataFrame(rows)
    output = root / "data/label_blind_artifact_gate_probe_20261004"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "scenario_statistics.csv", index=False)
    dev = frame[frame.source.str.startswith("development_")]
    stage = frame[frame.source.isin(["inspected_fifth", "external_natural"])]
    thresholds = []
    for metric in ("q95_logit_shift", "q99_logit_shift", "q99_exceedance",
                   "coherent_q95_shift"):
        natural_max = float(dev.loc[dev.scenario.eq("natural"), metric].max())
        threshold = np.nextafter(natural_max, np.inf)
        detection = float(dev.loc[dev.scenario.eq("injected"), metric].ge(threshold).mean())
        thresholds.append({"metric": metric, "threshold": threshold,
                           "development_injected_detection": detection,
                           "fifth_injected_detection": float(stage.loc[
                               (stage.source == "inspected_fifth") &
                               stage.scenario.eq("injected"), metric].ge(threshold).mean()),
                           "fifth_natural_false_alarms": int(stage.loc[
                               (stage.source == "inspected_fifth") &
                               stage.scenario.eq("natural"), metric].ge(threshold).sum()),
                           "external_natural_false_alarms": int(stage.loc[
                               stage.source.eq("external_natural"), metric].ge(threshold).sum())})
    summary = pd.DataFrame(thresholds)
    summary.to_csv(output / "threshold_audit.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "status": "post hoc scenario detector development; no SOTA claim",
        "development_sources": ["development_third", "development_fourth"],
        "inspected_validation_sources": ["inspected_fifth", "external_natural"],
        "threshold_rule": "above maximum development-natural value; no activity-label selection",
        "reps": args.reps,
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
