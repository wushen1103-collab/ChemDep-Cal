#!/usr/bin/env python3
"""Extract fifth-cohort partition counts from frozen scored panels."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    rows = []
    for seed in range(35501, 35506):
        folder = root / "data/chembl_meta_ungated_fifth_xgb_20261003" / f"seed{seed}" / "scores"
        paths = sorted(folder.glob("*_scores.csv"))
        if len(paths) != 10:
            raise AssertionError((seed, len(paths)))
        for path in paths:
            frame = pd.read_csv(path)
            target = str(frame["target_chembl_id"].iloc[0])
            for split, part in frame.groupby("split"):
                rows.append(dict(target=target, seed=seed, split=split,
                                 molecules=len(part), active=int(part["label"].sum()),
                                 inactive=int((part["label"] == 0).sum()),
                                 scaffolds=int(part["murcko_scaffold"].nunique())))
    result = pd.DataFrame(rows)
    for (target, split), part in result.groupby(["target", "split"]):
        if part[["molecules", "active", "inactive", "scaffolds"]].nunique().max() != 1:
            raise AssertionError(f"Split counts vary by scorer seed: {target}/{split}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result[result.seed.eq(35501)].drop(columns="seed").to_csv(args.out, index=False)
    print(result[result.seed.eq(35501)].drop(columns="seed").to_string(index=False))


if __name__ == "__main__":
    main()
