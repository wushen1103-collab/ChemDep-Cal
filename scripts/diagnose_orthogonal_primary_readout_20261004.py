#!/usr/bin/env python3
"""Development-only test of primary fluorescence-intensity information."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TASKS = {"AID1843": (1672, "B score Inhibitor Ratio", "B score Inhibitor Intensity"),
         "AID2258": (2239, "B score Potentiator Ratio", "B score Potentiator Intensity")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from evaluate_pubchem_four_untouched_20261004 import cap_one_tiebreak
    from run_chembl_selection import block_array

    base = root / "data/pubchem_four_untouched_20261004"
    rows = []
    for aid, (primary_aid, ratio_name, intensity_name) in TASKS.items():
        source = base / f"AID{primary_aid}_active_full_chunks"
        manifest = json.loads((source / "download_manifest.json").read_text(encoding="utf-8"))
        raw = pd.concat([pd.read_csv(source / row["part"], skiprows=[1, 2, 3], low_memory=False)
                         for row in manifest["chunks"]], ignore_index=True)
        raw["CID"] = pd.to_numeric(raw.PUBCHEM_CID, errors="coerce")
        raw["ratio"] = pd.to_numeric(raw[ratio_name], errors="coerce")
        raw["intensity"] = pd.to_numeric(raw[intensity_name], errors="coerce")
        raw = raw.dropna(subset=["CID", "ratio", "intensity"])
        raw = raw[np.isfinite(raw.ratio) & np.isfinite(raw.intensity)].copy()
        raw.CID = raw.CID.astype(int)
        per_cid = raw.groupby("CID", as_index=False)[["ratio", "intensity"]].median()
        cohort = pd.read_parquet(base / "cohorts" / f"{aid}_cohort.parquet")
        merged = cohort.merge(per_cid, on="CID", how="left", validate="one_to_one")
        if merged[["ratio", "intensity"]].isna().any().any():
            raise AssertionError(f"{aid}: missing orthogonal readout")
        print(aid, "n", len(merged), "median |intensity| by label",
              merged.groupby("label").intensity.apply(lambda x: float(x.abs().median())).to_dict(),
              flush=True)
        for seed in range(1, 6):
            with np.load(base / "cohorts" / f"{aid}_seed{seed}_indices.npz") as split:
                dev_idx = np.concatenate([split["train"], split["calibration"]])
                test_idx = split["test"]
            dev = merged.iloc[dev_idx]
            test = merged.iloc[test_idx].reset_index(drop=True)
            features = pd.DataFrame({"score": merged.score,
                                     "abs_intensity": merged.intensity.abs()})
            test_blocks = block_array(test.assign(molecule_chembl_id=test.CID.astype(str)),
                                      "murcko_scaffold")
            keys = np.asarray([int.from_bytes(hashlib.sha256(
                f"{aid}/{seed}/{cid}".encode()).digest()[:8], "big")
                for cid in test.CID], dtype=np.uint64)
            variants = {"score_only": ["score"],
                        "score_plus_abs_intensity": ["score", "abs_intensity"]}
            for name, columns in variants.items():
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    class_weight="balanced", C=1., max_iter=1000, random_state=20261004 + seed))
                model.fit(features.iloc[dev_idx][columns], dev.label)
                scores = model.predict_proba(features.iloc[test_idx][columns])[:, 1]
                selected = cap_one_tiebreak(scores, test_blocks, 50, keys)
                rows.append({"aid": aid, "seed": seed, "method": name,
                             "hits": int(test.label.iloc[selected].sum()),
                             "selected": len(selected), "scaffolds": len(set(test_blocks[selected])),
                             "n_train": len(dev), "n_test": len(test)})
            selected = cap_one_tiebreak(test.score.to_numpy(float), test_blocks, 50, keys)
            rows.append({"aid": aid, "seed": seed, "method": "raw_score_cap1",
                         "hits": int(test.label.iloc[selected].sum()),
                         "selected": len(selected), "scaffolds": len(set(test_blocks[selected])),
                         "n_train": len(dev), "n_test": len(test)})
    output = base / "orthogonal_readout_posthoc"
    output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    if len(result) != 30 or not result.selected.eq(50).all() or not result.scaffolds.eq(50).all():
        raise AssertionError("Incomplete same-budget grid")
    result.to_csv(output / "cells.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "status": "post hoc development after observed selector outcomes; not SOTA validation",
        "input": "primary PubChem activity score and absolute initial fluorescence intensity B-score",
        "train": "same 60% train plus 20% calibration labels for both logistic variants",
        "test": "unchanged five scaffold-split test pools, B=50 cap1",
    }, indent=2) + "\n", encoding="utf-8")
    print(result.groupby(["aid", "method"]).hits.agg(["mean", "std"]).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
