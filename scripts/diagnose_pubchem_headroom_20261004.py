#!/usr/bin/env python3
"""Post hoc attainable-hit ceiling for the explicit-follow-up cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


AIDS = ("AID1798", "AID435034", "AID463087")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    base = args.root.resolve() / "data/pubchem_primary_welqrate_20261004/confirmatory_failures"
    rows = []
    for aid in AIDS:
        cohort = pd.read_parquet(base / f"{aid}_cohort.parquet")
        for seed in range(1, 6):
            with np.load(base / f"{aid}_seed{seed}_indices.npz") as split:
                test = cohort.iloc[split["test"]]
            blocks = test.murcko_scaffold.where(
                test.murcko_scaffold.ne(""), "singleton:" + test.CID.astype(str))
            positive_scaffolds = int(test.groupby(blocks).label.max().sum())
            oracle_hits = min(50, positive_scaffolds)
            cells = pd.read_csv(base / "selectors" / f"seed{seed}_cells.csv")
            cells = cells[cells.aid.eq(aid)].set_index("method")
            if not cells.selected.eq(50).all() or not cells.scaffolds.eq(50).all():
                raise AssertionError("Inconsistent capacity")
            if int(cells.hits.max()) > oracle_hits:
                raise AssertionError("Observed method exceeds oracle cap")
            rows.append({"aid": aid, "seed": seed, "test_size": len(test),
                         "test_actives": int(test.label.sum()),
                         "empty_scaffold_rows": int(test.murcko_scaffold.eq("").sum()),
                         "positive_scaffolds": positive_scaffolds,
                         "oracle_cap1_hits": oracle_hits,
                         "score_cap1_hits": int(cells.loc["score_cap1", "hits"]),
                         "chemdep_hits": int(cells.loc["chemdep_cal", "hits"]),
                         "oracle_minus_score_cap1": oracle_hits - int(cells.loc["score_cap1", "hits"]),
                         "oracle_minus_chemdep": oracle_hits - int(cells.loc["chemdep_cal", "hits"])})
    frame = pd.DataFrame(rows)
    frame.to_csv(base / "headroom_posthoc.csv", index=False)
    print(frame.to_string(index=False))
    print(frame.groupby("aid")[["oracle_cap1_hits", "score_cap1_hits", "chemdep_hits",
                                 "oracle_minus_score_cap1"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
