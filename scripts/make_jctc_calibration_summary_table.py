#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def write_pair(df: pd.DataFrame, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stem}.csv", index=False)
    try:
        text = df.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        text = df.to_string(index=False)
    (outdir / f"{stem}.md").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-table", default="tables/jctc_nominal_alpha_calibration.csv")
    parser.add_argument("--outdir", default="tables")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / args.calibration_table)
    out = (
        df.groupby(["q", "method"], as_index=False)
        .agg(
            n_cells=("fdp", "size"),
            pass_cells=("risk_wording", lambda s: int((s == "empirical_mean_fdp_le_alpha").sum())),
            mean_selected=("selected_count", "mean"),
            mean_true_hits=("true_hits", "mean"),
            mean_fdp=("fdp", "mean"),
            max_fdp=("fdp", "max"),
            mean_fdp_minus_alpha=("empirical_fdp_minus_nominal_alpha", "mean"),
            max_fdp_minus_alpha=("empirical_fdp_minus_nominal_alpha", "max"),
            mean_exceed_prob=("fdp_exceeds_q", "mean"),
        )
        .sort_values(["q", "method"])
    )
    out["pass_rate"] = out["pass_cells"] / out["n_cells"]
    write_pair(out, root / args.outdir, "jctc_nominal_alpha_calibration_summary")


if __name__ == "__main__":
    main()
