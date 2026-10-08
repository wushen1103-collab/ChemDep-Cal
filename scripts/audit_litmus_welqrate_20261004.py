#!/usr/bin/env python3
"""Inspect the third-party WelQrate mirror before designing any selector experiment."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    data_path = directory / "WELQRATE_AID1798.parquet"
    train_path = directory / "seed1_WELQRATE_AID1798.train_idx.csv"
    test_path = directory / "seed1_WELQRATE_AID1798.test_idx.csv"
    for path in (data_path, train_path, test_path):
        print(path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
    frame = pd.read_parquet(data_path)
    print("data shape", frame.shape, "columns", frame.columns.tolist())
    print(frame.head(2).to_string(index=False))
    print("dtypes", frame.dtypes.to_dict())
    train = pd.read_csv(train_path, header=None)
    test = pd.read_csv(test_path, header=None)
    print("train index file", train.shape, train.columns.tolist(), train.head(4).to_string(index=False))
    print("test index file", test.shape, test.columns.tolist(), test.head(4).to_string(index=False))
    for name, table in (("train", train), ("test", test)):
        for column in table.columns:
            values = table[column].to_numpy()
            if np.issubdtype(values.dtype, np.integer):
                print(name, column, "min", int(values.min()), "max", int(values.max()),
                      "unique", len(np.unique(values)))
    train_ids = train.iloc[:, 0].to_numpy(int)
    test_ids = test.iloc[:, 0].to_numpy(int)
    if len(np.intersect1d(train_ids, test_ids)) or len(np.union1d(train_ids, test_ids)) != len(frame):
        raise AssertionError("Litmus split does not partition all molecule indices")
    print("actives full/train/test", int(frame.AID1798.sum()),
          int(frame.iloc[train_ids].AID1798.sum()), int(frame.iloc[test_ids].AID1798.sum()))
    scaffolds = []
    for smiles in frame.SMILES:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise AssertionError("Invalid SMILES")
        scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False))
    train_scaffolds = set(np.asarray(scaffolds, dtype=object)[train_ids])
    test_scaffolds = set(np.asarray(scaffolds, dtype=object)[test_ids])
    print("scaffold overlap", len(train_scaffolds & test_scaffolds),
          "train unique", len(train_scaffolds), "test unique", len(test_scaffolds))


if __name__ == "__main__":
    main()
