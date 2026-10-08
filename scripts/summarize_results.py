#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    path = Path(args.results_csv)
    df = pd.read_csv(path)
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
    keys = ["scenario", "q", "budget", "method"]
    if "budget_label" in df.columns:
        keys = ["scenario", "q", "budget_label", "budget", "method"]
    summary = (
        df.groupby(keys, as_index=False)[metrics]
        .mean()
        .sort_values(["scenario", "q", "budget", "fdp", "true_hits"], ascending=[True, True, True, True, False])
    )
    print(summary.to_markdown(index=False, floatfmt=".4f"))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
