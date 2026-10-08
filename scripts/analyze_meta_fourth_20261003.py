#!/usr/bin/env python3
"""Audit the frozen fourth cohort at the target, not injection, level."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = tuple(range(35401, 35406))
KEYS = ["target", "scorer_seed", "rep", "fraction", "boost"]
PRIMARY = "meta_calibration"
COMPARATORS = (
    "minp_score_fill", "minp_block_bh_cap1", "weighted_bh",
    "score_cap1", "chemdeprc_cap1", "opdiv_tanimoto0.7",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_signflip(diff: np.ndarray) -> float:
    observed = abs(float(np.mean(diff)))
    return float(np.mean([
        abs(float(np.mean(diff * signs))) >= observed - 1e-12
        for signs in itertools.product((-1., 1.), repeat=len(diff))
    ]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--reps", type=int, default=10)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    chunks = []
    hashes = {}
    target_set = None
    for seed in SEEDS:
        files = {
            "meta": args.input_dir / f"meta_seed{seed}.csv",
            "fill": args.input_dir / f"fill_seed{seed}.csv",
            "opdiv": args.input_dir / f"opdiv_seed{seed}.csv",
        }
        for kind, path in files.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            hashes[path.name] = sha256(path)
        meta = pd.read_csv(files["meta"])
        meta = meta[meta["budget"].eq(50)].copy()
        meta["scorer_seed"] = seed
        fill = pd.read_csv(files["fill"])
        fill["scorer_seed"] = seed
        opdiv = pd.read_csv(files["opdiv"])
        opdiv["scorer_seed"] = seed

        targets = set(meta["target"].unique())
        if target_set is None:
            target_set = targets
        elif targets != target_set:
            raise AssertionError(f"Target mismatch in scorer seed {seed}")
        cells = len(targets) * args.reps * 4
        expected_methods = set(COMPARATORS) - {"minp_score_fill", "opdiv_tanimoto0.7"}
        expected_methods.add(PRIMARY)
        if set(meta["method"].unique()) != expected_methods:
            raise AssertionError(f"Unexpected methods in seed {seed}: {set(meta['method'])}")
        for frame, kind, multiplier in ((meta, "meta", len(expected_methods)), (fill, "fill", 1), (opdiv, "opdiv", 1)):
            if len(frame) != cells * multiplier or frame.duplicated(KEYS + ["method"]).any():
                raise AssertionError(f"Incomplete/duplicate {kind} cells in seed {seed}")
        if not opdiv["selected"].eq(50).all() or not opdiv["status"].eq("optimal").all():
            raise AssertionError(f"OPDiv non-full/non-optimal in seed {seed}")
        if not fill["selected"].eq(50).all() or not fill["blocks"].eq(50).all():
            raise AssertionError(f"Min-p fill incomplete in seed {seed}")

        ref = meta[meta["method"].eq("score_cap1")].set_index(KEYS)["hits"].sort_index()
        opdiv_sanity = opdiv.set_index(KEYS)["score_cap1_sanity_hits"].sort_index()
        if not ref.equals(opdiv_sanity):
            raise AssertionError(f"OPDiv scores/injections do not match meta grid in seed {seed}")
        chunks.extend((meta, fill, opdiv))

    raw = pd.concat(chunks, ignore_index=True)
    n_targets = len(target_set)
    raw.to_csv(args.outdir / "combined_primary_raw.csv", index=False)
    seed_means = raw.groupby(["scorer_seed", "method"], as_index=False).agg(
        hits=("hits", "mean"), fdp=("fdp", "mean"),
        selected=("selected", "mean"), blocks=("blocks", "mean"),
    )
    seed_means.to_csv(args.outdir / "seed_method_means.csv", index=False)
    target_means = raw.groupby(["target", "method"], as_index=False).agg(
        hits=("hits", "mean"), fdp=("fdp", "mean"),
        selected=("selected", "mean"), blocks=("blocks", "mean"),
    )
    target_means.to_csv(args.outdir / "target_method_means.csv", index=False)
    summary = target_means.groupby("method", as_index=False).agg(
        hits=("hits", "mean"), hits_target_sd=("hits", "std"),
        fdp=("fdp", "mean"), fdp_target_sd=("fdp", "std"),
        selected=("selected", "mean"), blocks=("blocks", "mean"),
    ).sort_values("hits", ascending=False)
    summary.to_csv(args.outdir / "publication_mean_target_sd.csv", index=False)

    pivot = target_means.pivot(index="target", columns="method", values="hits")
    rng = np.random.default_rng(354700)
    paired = []
    for comparator in COMPARATORS:
        diff = (pivot[PRIMARY] - pivot[comparator]).to_numpy(float)
        draws = rng.integers(0, n_targets, size=(100000, n_targets))
        ci = np.quantile(diff[draws].mean(axis=1), [.025, .975])
        paired.append({
            "comparator": comparator,
            "mean_hits_delta": float(diff.mean()),
            "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "positive_targets": int((diff > 1e-10).sum()),
            "negative_targets": int((diff < -1e-10).sum()),
            "tied_targets": int((abs(diff) <= 1e-10).sum()),
            "exact_signflip_p_two_sided": exact_signflip(diff),
        })
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(args.outdir / "paired_target_bootstrap.csv", index=False)
    primary = paired_frame.set_index("comparator").loc["minp_score_fill"]
    passed = bool(
        primary["ci_low"] > 0
        and primary["exact_signflip_p_two_sided"] < .05
        and paired_frame["mean_hits_delta"].gt(0).all()
    )
    audit = {
        "n_targets": n_targets, "targets": sorted(target_set),
        "scorer_seeds": list(SEEDS), "injection_reps_per_seed_condition": args.reps,
        "n_cells_per_method": n_targets * len(SEEDS) * args.reps * 4,
        "input_sha256": hashes,
        "preregistered_primary_passed": passed,
        "interpretation": "Conditional within-protocol only; label-aware artifact simulation, not external or wet-lab SOTA",
    }
    (args.outdir / "integrity_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(paired_frame.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
