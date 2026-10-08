#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


def analog_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return f"invalid::{smiles}"
    try:
        generic = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))
        generic_smiles = Chem.MolToSmiles(generic, isomericSmiles=False)
        if generic_smiles:
            return generic_smiles
    except Exception:
        pass
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:
        return f"fallback::{smiles}"


def add_block_column(df: pd.DataFrame, block_col: str) -> tuple[pd.DataFrame, list[dict]]:
    out = df.copy()
    out[block_col] = ""
    stats = []
    smiles_col = "canonical_smiles" if "canonical_smiles" in out.columns else "smiles"
    for split, split_df in out.groupby("split", sort=False):
        values = [f"analog_{split}_{analog_key(smiles)}" for smiles in tqdm(split_df[smiles_col], desc=split, leave=False)]
        out.loc[split_df.index, block_col] = values
        counts = pd.Series(values).value_counts()
        stats.append(
            {
                "split": split,
                "n_molecules": int(len(split_df)),
                "block_col": block_col,
                "mode": "analog_series_generic_murcko",
                "threshold": "",
                "n_blocks": int(len(counts)),
                "max_block_size": int(counts.max()) if len(counts) else 0,
                "median_block_size": float(counts.median()) if len(counts) else 0.0,
                "singleton_rate": float((counts == 1).mean()) if len(counts) else 0.0,
            }
        )
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--block-col", default="analog_series_block")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    scores_dir = root / args.scores_dir
    outdir = root / args.outdir
    out_scores = outdir / "scores"
    out_scores.mkdir(parents=True, exist_ok=True)
    score_paths = sorted(scores_dir.glob("*_scores.csv"))
    if not score_paths:
        raise RuntimeError(f"No score files found under {scores_dir}")

    all_stats = []
    for path in tqdm(score_paths, desc="analog-series blocks"):
        df = pd.read_csv(path)
        out_df, stats = add_block_column(df, args.block_col)
        out_df.to_csv(out_scores / path.name, index=False)
        target_id = str(df["target_chembl_id"].iloc[0])
        pref_name = str(df["pref_name"].iloc[0])
        for row in stats:
            row["target_chembl_id"] = target_id
            row["pref_name"] = pref_name
            all_stats.append(row)

    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(outdir / "block_stats.csv", index=False)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scores_dir": args.scores_dir,
        "outdir": args.outdir,
        "block_col": args.block_col,
        "targets": len(score_paths),
        "block_definition": "RDKit generic Bemis-Murcko scaffold used as an analog-series proxy",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(stats_df[stats_df["split"] == "testpool"].to_markdown(index=False, floatfmt=".4f"))
    print(f"Wrote {outdir}")


if __name__ == "__main__":
    main()
