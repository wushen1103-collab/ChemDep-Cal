#!/usr/bin/env python3
"""Compare calibration-only scorer signal with augmented-RF test gains."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from train_artifact_meta_ranker_20261003 import load_panel

    rows = []
    for cohort in ("fifth", "seventh"):
        result_dir = root / f"data/pr_{cohort}_augmented_multifp_development_20261005"
        seeds = range(35501, 35506) if cohort == "fifth" else range(35701, 35706)
        for seed in seeds:
            if cohort == "fifth":
                paths = sorted((root / "data/chembl_meta_ungated_fifth_xgb_20261003"
                                / f"seed{seed}" / "scores").glob("*_scores.csv"))
            else:
                directory = root / "data/chembl_modern_seventh_20261005"
                targets = pd.read_csv(directory / "technical_eligibility.csv").query(
                    "eligible").target_chembl_id.tolist()
                paths = [directory / f"eligible_seed{seed}" / "scores"
                         / f"{target}_scores.csv" for target in targets]
            hits = pd.read_csv(result_dir / f"seed{seed}_cells.csv")
            for path in paths:
                panel = load_panel(path)
                cal = panel["cal"]
                labels = cal.label.to_numpy(int)
                scores = cal.score.to_numpy(float)
                if len(set(labels)) != 2:
                    raise AssertionError("Calibration panel must contain both classes")
                sub = hits[hits.target.eq(panel["target"])]
                rows.append({
                    "cohort": cohort, "target": panel["target"], "seed": seed,
                    "cal_n": len(cal), "cal_pos": int(labels.sum()),
                    "cal_score_auc": roc_auc_score(labels, scores),
                    "cal_score_gap": float(scores[labels == 1].mean()
                                           - scores[labels == 0].mean()),
                    "rf_hits": sub.multifp_rf_hits.mean(),
                    "augmented_rf_hits": sub.augmented_score_rf_hits.mean(),
                    "augmented_gain": (sub.augmented_score_rf_hits
                                       - sub.multifp_rf_hits).mean(),
                })
    result = pd.DataFrame(rows)
    out = root / "data/pr_calibration_score_signal_development_20261005"
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "per_seed.csv", index=False)
    grouped = result.groupby(["cohort", "target"], as_index=False).agg(
        cal_n=("cal_n", "mean"), cal_pos=("cal_pos", "mean"),
        cal_score_auc=("cal_score_auc", "mean"),
        cal_score_gap=("cal_score_gap", "mean"),
        rf_hits=("rf_hits", "mean"), augmented_gain=("augmented_gain", "mean"))
    grouped.to_csv(out / "per_target.csv", index=False)
    print(grouped.to_string(index=False), flush=True)
    print("AUC/gain correlation", grouped.cal_score_auc.corr(grouped.augmented_gain), flush=True)
    print("Gap/gain correlation", grouped.cal_score_gap.corr(grouped.augmented_gain), flush=True)


if __name__ == "__main__":
    main()
