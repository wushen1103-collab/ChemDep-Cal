#!/usr/bin/env python3
"""Fifth-cohort matched-cap-one confirmation and familywise audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_meta_fourth_20261003 import exact_signflip, sha256


SEEDS = tuple(range(35501, 35506))
KEYS = ["target", "scorer_seed", "rep", "fraction", "boost"]
PRIMARY = "meta_calibration"
MAIN = ("score_cap1", "minp_score_fill", "weighted_bh_cap1_fill")
SECONDARY = ("chemdeprc_cap1", "minp_block_bh_cap1", "weighted_bh", "opdiv_tanimoto0.7")
META_METHODS = {PRIMARY, "score_cap1", "minp_block_bh_cap1", "chemdeprc_cap1", "weighted_bh"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--reps", type=int, default=10)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    frames = []
    hashes = {}
    target_set = None
    for seed in SEEDS:
        files = {
            "meta": args.input_dir / f"meta_seed{seed}.csv",
            "fill": args.input_dir / f"fill_seed{seed}.csv",
            "weighted": args.input_dir / f"weighted_cap1_seed{seed}.csv",
            "opdiv": args.input_dir / f"opdiv_seed{seed}.csv",
        }
        parts = {}
        for kind, path in files.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            hashes[path.name] = sha256(path)
            parts[kind] = pd.read_csv(path)
            parts[kind]["scorer_seed"] = seed
        meta = parts["meta"]
        meta = meta[meta["budget"].eq(50)].copy()
        parts["meta"] = meta
        targets = set(meta["target"].unique())
        if target_set is None:
            target_set = targets
        elif targets != target_set:
            raise AssertionError(f"Target-set mismatch in seed {seed}")
        cells = len(targets) * args.reps * 4
        if set(meta["method"].unique()) != META_METHODS:
            raise AssertionError(f"Unexpected meta methods in seed {seed}")
        for kind, frame in parts.items():
            multiplier = len(META_METHODS) if kind == "meta" else 1
            if len(frame) != cells * multiplier or frame.duplicated(KEYS + ["method"]).any():
                raise AssertionError(f"Incomplete/duplicate {kind} in seed {seed}")
        for kind in ("fill", "weighted", "opdiv"):
            frame = parts[kind]
            if not frame["selected"].eq(50).all() or not frame["blocks"].eq(50).all() and kind != "opdiv":
                raise AssertionError(f"Non-full cap-one {kind} in seed {seed}")
        if not parts["opdiv"]["status"].eq("optimal").all():
            raise AssertionError(f"Non-optimal OPDiv in seed {seed}")
        sanity = meta[meta["method"].eq("score_cap1")].set_index(KEYS)["hits"].sort_index()
        replay = parts["opdiv"].set_index(KEYS)["score_cap1_sanity_hits"].sort_index()
        if not sanity.equals(replay):
            raise AssertionError(f"OPDiv score replay mismatch in seed {seed}")
        frames.extend(parts.values())

    raw = pd.concat(frames, ignore_index=True)
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

    n_targets = len(target_set)
    pivot = target_means.pivot(index="target", columns="method", values="hits")
    rng = np.random.default_rng(355700)
    paired = []
    for comparator in MAIN + SECONDARY:
        diff = (pivot[PRIMARY] - pivot[comparator]).to_numpy(float)
        draws = rng.integers(0, n_targets, size=(100000, n_targets))
        ci = np.quantile(diff[draws].mean(axis=1), [.025, .975])
        paired.append({
            "comparator": comparator, "mean_hits_delta": float(diff.mean()),
            "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "positive_targets": int((diff > 1e-10).sum()),
            "negative_targets": int((diff < -1e-10).sum()),
            "tied_targets": int((np.abs(diff) <= 1e-10).sum()),
            "exact_signflip_p_two_sided": exact_signflip(diff),
        })
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(args.outdir / "paired_target_bootstrap.csv", index=False)
    main = paired_frame[paired_frame["comparator"].isin(MAIN)].copy()
    ordered_p = sorted(main["exact_signflip_p_two_sided"].to_numpy(float))
    holm_thresholds = [.05 / (len(MAIN) - i) for i in range(len(MAIN))]
    holm_all = all(pv < threshold for pv, threshold in zip(ordered_p, holm_thresholds))
    passed = bool(
        main["mean_hits_delta"].gt(0).all()
        and main["ci_low"].gt(0).all()
        and holm_all
        and paired_frame.set_index("comparator").loc["chemdeprc_cap1", "mean_hits_delta"] > 0
    )
    audit = {
        "n_targets": n_targets, "targets": sorted(target_set),
        "scorer_seeds": list(SEEDS), "injection_reps_per_seed_condition": args.reps,
        "n_cells_per_method": n_targets * len(SEEDS) * args.reps * 4,
        "main_comparators": MAIN, "holm_sorted_p": ordered_p,
        "holm_thresholds": holm_thresholds, "holm_all_passed": holm_all,
        "preregistered_matched_cap1_claim_passed": passed,
        "input_sha256": hashes,
        "scope": "Within-protocol ChEMBL simulated high-artifact B50 cap-one selection only",
    }
    (args.outdir / "integrity_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(paired_frame.to_string(index=False))
    print(json.dumps({k: v for k, v in audit.items() if k != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
