#!/usr/bin/env python3
"""Cluster-paired uncertainty for the calibration-trained ranker pilot."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--method", default="meta_calibration")
    p.add_argument("--expected-methods", type=int, default=5)
    args = p.parse_args()
    raw = pd.read_csv(args.input)
    keys = ["target", "rep", "fraction", "boost", "budget"]
    cells = raw.groupby(keys)["method"].nunique()
    if not cells.eq(args.expected_methods).all():
        raise AssertionError("Missing method cells")
    target = raw.groupby(["target", "budget", "method"], as_index=False).agg(
        hits=("hits", "mean"), fdp=("fdp", "mean"),
        selected=("selected", "mean"), blocks=("blocks", "mean"),
    )
    records = []
    rng = np.random.default_rng(353600)
    for budget in (10, 50, 75):
        panel = target[target["budget"].eq(budget)]
        left = panel[panel["method"].eq(args.method)].set_index("target")
        for reference in ("minp_block_bh_cap1", "chemdeprc_cap1", "weighted_bh", "score_cap1"):
            right = panel[panel["method"].eq(reference)].set_index("target")
            if not left.index.equals(right.index):
                raise AssertionError("Target mismatch")
            delta = (left["hits"] - right["hits"]).to_numpy(float)
            samples = rng.integers(0, len(delta), size=(50000, len(delta)))
            ci = np.quantile(delta[samples].mean(axis=1), [.025, .975])
            observed = abs(float(delta.mean()))
            exact = np.mean([
                abs(np.mean(np.asarray(signs) * delta)) >= observed - 1e-12
                for signs in itertools.product((-1., 1.), repeat=len(delta))
            ])
            records.append({
                "budget": budget, "reference": reference,
                "n_targets": len(delta), "hits_delta": delta.mean(),
                "hits_ci_low": ci[0], "hits_ci_high": ci[1],
                "positive_targets": int((delta > 0).sum()),
                "negative_targets": int((delta < 0).sum()),
                "exact_signflip_p_two_sided": exact,
                "fdp_delta": float((left["fdp"] - right["fdp"]).mean()),
                "selected_delta": float((left["selected"] - right["selected"]).mean()),
                "blocks_delta": float((left["blocks"] - right["blocks"]).mean()),
            })
    out = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
