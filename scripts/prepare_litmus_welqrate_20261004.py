#!/usr/bin/env python3
"""Build fixed, scaffold-disjoint train/calibration/test panels for three AIDs."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


EXPECTED = {"AID435034": (60359, 78), "AID1798": (60706, 164),
            "AID463087": (95650, 652)}


def prepare_one(task: tuple[str, str, str]) -> list[dict]:
    source_s, output_s, aid = task
    source, output = Path(source_s), Path(output_s)
    RDLogger.DisableLog("rdApp.*")
    frame = pd.read_parquet(source / f"WELQRATE_{aid}.parquet")
    if frame.shape != (EXPECTED[aid][0], 2) or int(frame[aid].sum()) != EXPECTED[aid][1]:
        raise AssertionError(f"{aid}: row or active count differs from the published table")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    scaffolds, canonical = [], []
    packed = np.empty((len(frame), 256), dtype=np.uint8)
    bit_buffer = np.zeros(2048, dtype=np.uint8)
    for i, smiles in enumerate(frame.SMILES):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise AssertionError(f"{aid}: invalid molecule row {i}")
        canonical.append(Chem.MolToSmiles(molecule, isomericSmiles=True))
        scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False))
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), bit_buffer)
        packed[i] = np.packbits(bit_buffer, bitorder="little")
    metadata = pd.DataFrame({"SMILES": frame.SMILES, "canonical_smiles": canonical,
                             "label": frame[aid].astype(np.int8), "murcko_scaffold": scaffolds})
    output.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(output / f"{aid}_molecules.parquet", index=False)
    np.save(output / f"{aid}_ecfp4_packed.npy", packed)
    rows = []
    all_scaffolds = np.asarray(scaffolds, dtype=object)
    labels = metadata.label.to_numpy(np.int8)
    for seed in range(1, 6):
        train_file = source / f"seed{seed}_WELQRATE_{aid}.train_idx.csv"
        test_file = source / f"seed{seed}_WELQRATE_{aid}.test_idx.csv"
        source_train = pd.read_csv(train_file, header=None).iloc[:, 0].to_numpy(int)
        test = pd.read_csv(test_file, header=None).iloc[:, 0].to_numpy(int)
        if len(np.unique(source_train)) != len(source_train) or len(np.unique(test)) != len(test):
            raise AssertionError(f"{aid}/seed{seed}: duplicate split indices")
        if len(source_train) + len(test) != len(frame) or len(np.intersect1d(source_train, test)):
            raise AssertionError(f"{aid}/seed{seed}: original split is not a partition")
        if set(all_scaffolds[source_train]) & set(all_scaffolds[test]):
            raise AssertionError(f"{aid}/seed{seed}: original split overlaps scaffolds")
        group_frame = pd.DataFrame({"scaffold": all_scaffolds[source_train],
                                    "label": labels[source_train]})
        group_positive = group_frame.groupby("scaffold", sort=True).label.max()
        positive_groups = group_positive.index[group_positive.eq(1)].to_numpy(str)
        negative_groups = group_positive.index[group_positive.eq(0)].to_numpy(str)
        digest = hashlib.sha256(f"{aid}/{seed}".encode()).digest()
        rng = np.random.default_rng(20261004 + int.from_bytes(digest[:4], "big"))
        selected = []
        for groups in (positive_groups, negative_groups):
            count = max(1, int(round(0.2 * len(groups))))
            selected.extend(rng.permutation(groups)[:count])
        mask = np.isin(all_scaffolds[source_train], np.asarray(selected, dtype=object))
        calibration, train = source_train[mask], source_train[~mask]
        if len(set(all_scaffolds[train]) & set(all_scaffolds[calibration])):
            raise AssertionError("Scaffold leakage between scorer train and calibration")
        if len(set(all_scaffolds[calibration]) & set(all_scaffolds[test])):
            raise AssertionError("Scaffold leakage between calibration and test")
        np.savez_compressed(output / f"{aid}_seed{seed}_indices.npz",
                            train=train, calibration=calibration, test=test)
        row = {"aid": aid, "seed": seed, "n_total": len(frame),
               "n_train": len(train), "n_calibration": len(calibration), "n_test": len(test),
               "active_train": int(labels[train].sum()),
               "active_calibration": int(labels[calibration].sum()),
               "active_test": int(labels[test].sum()),
               "scaffolds_train": len(set(all_scaffolds[train])),
               "scaffolds_calibration": len(set(all_scaffolds[calibration])),
               "scaffolds_test": len(set(all_scaffolds[test])),
               "eligible": int(labels[calibration].sum()) >= 5
               and int(labels[test].sum()) >= 5
               and len(set(all_scaffolds[test])) >= 50}
        rows.append(row)
    print(aid, "processed", len(frame), "molecules", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    with futures.ProcessPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(prepare_one,
                                [(str(source), str(output), aid) for aid in EXPECTED]))
    ledger = pd.DataFrame([row for part in results for row in part]).sort_values(["aid", "seed"])
    ledger.to_csv(output / "split_audit.csv", index=False)
    (output / "split_manifest.json").write_text(json.dumps({
        "status": "post hoc, frozen before new selector results",
        "source": "Litmus mirror commit 10c6bd4d8d6bd0116929239864d6bbddb4e46ab0",
        "not_official_welqrate_split": True,
        "calibration_rule": "20 percent of positive-bearing and 20 percent of negative-only train scaffolds, deterministic shuffle",
        "ecfp4": "RDKit Morgan radius 2, 2048 bits, little-endian packed",
        "source_manifest_sha256": hashlib.sha256((source / "download_manifest.json").read_bytes()).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")
    print(ledger.to_string(index=False))
    if not ledger.eligible.all():
        print("WARNING: one or more splits failed the frozen eligibility rule", flush=True)


if __name__ == "__main__":
    main()
