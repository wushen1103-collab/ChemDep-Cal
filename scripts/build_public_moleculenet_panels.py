#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.*")
FP_COL = "ecfp4_radius2_nbits2048"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    url: str
    smiles_col: str
    label_col: str
    positive_label: str
    task_family: str


DATASETS = {
    "hiv": DatasetSpec(
        key="hiv",
        display_name="MoleculeNet HIV",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        smiles_col="smiles",
        label_col="HIV_active",
        positive_label="HIV_active=1",
        task_family="long_tail_bioactivity",
    ),
    "bace": DatasetSpec(
        key="bace",
        display_name="MoleculeNet BACE",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        smiles_col="mol",
        label_col="Class",
        positive_label="Class=1",
        task_family="bioactivity",
    ),
    "bbbp": DatasetSpec(
        key="bbbp",
        display_name="MoleculeNet BBBP",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        smiles_col="smiles",
        label_col="p_np",
        positive_label="p_np=1",
        task_family="adme",
    ),
    "clintox_ct_tox": DatasetSpec(
        key="clintox_ct_tox",
        display_name="MoleculeNet ClinTox CT_TOX",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        smiles_col="smiles",
        label_col="CT_TOX",
        positive_label="CT_TOX=1",
        task_family="toxicity",
    ),
}


SCENARIOS = {
    "scaffold_ood": "label-free scaffold split, 60/20/20 train/calibration/testpool by Murcko scaffold",
    "cold_start20": "same scaffold split with only 20 percent of train molecules retained",
    "missing_fp25": "same scaffold split with 25 percent of ECFP bits zeroed before scoring",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data/public_moleculenet")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[3500, 3501, 3502, 3503, 3504])
    parser.add_argument("--min-panel", type=int, default=300)
    parser.add_argument("--min-active", type=int, default=20)
    parser.add_argument("--min-inactive", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    raw_dir = outdir / "raw"
    panel_dir = outdir / "panels"
    raw_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    for dataset_key in args.datasets:
        if dataset_key not in DATASETS:
            raise ValueError(f"Unknown dataset {dataset_key!r}; expected one of {sorted(DATASETS)}")
        spec = DATASETS[dataset_key]
        raw_path = download_dataset(spec, raw_dir)
        base_panel, base_stats = curate_dataset(raw_path, spec)
        for scenario in args.scenarios:
            if scenario not in SCENARIOS:
                raise ValueError(f"Unknown scenario {scenario!r}; expected one of {sorted(SCENARIOS)}")
            for seed in args.seeds:
                panel = base_panel.copy()
                panel = add_balanced_scaffold_split(panel, seed=seed)
                if scenario == "cold_start20":
                    panel = keep_cold_start_train_subset(panel, seed=seed, train_fraction=0.20)
                elif scenario == "missing_fp25":
                    panel[FP_COL] = mask_fingerprint_bits(panel[FP_COL], seed=seed, drop_fraction=0.25)
                panel_id = f"MLNET_{spec.key.upper()}_{scenario}_s{seed}"
                panel["target_chembl_id"] = panel_id
                panel["pref_name"] = f"{spec.display_name} {scenario}"
                panel["dataset"] = spec.key
                panel["task_name"] = spec.display_name
                panel["scenario"] = scenario
                panel["panel_seed"] = seed
                panel["positive_label"] = spec.positive_label
                panel["task_family"] = spec.task_family
                panel_path = panel_dir / f"{spec.key}_{scenario}_seed{seed}_panel.csv"
                panel.to_csv(panel_path, index=False)
                stats = panel_stats(panel)
                passes = (
                    stats["n_panel"] >= args.min_panel
                    and stats["n_active"] >= args.min_active
                    and stats["n_inactive"] >= args.min_inactive
                    and stats["train_active"] > 0
                    and stats["train_inactive"] > 0
                    and stats["calibration_inactive"] > 0
                    and stats["testpool_active"] > 0
                    and stats["testpool_inactive"] > 0
                )
                manifest_rows.append(
                    {
                        "target_chembl_id": panel_id,
                        "pref_name": f"{spec.display_name} {scenario}",
                        "dataset": spec.key,
                        "task_name": spec.display_name,
                        "task_family": spec.task_family,
                        "scenario": scenario,
                        "panel_seed": seed,
                        "panel_path": panel_path.relative_to(root).as_posix(),
                        "panel_sha256": sha256_file(panel_path),
                        "source": "MoleculeNet raw CSV from DeepChem data bucket",
                        "source_url": spec.url,
                        "source_raw_sha256": sha256_file(raw_path),
                        "source_columns": f"smiles={spec.smiles_col}; label={spec.label_col}",
                        "positive_label": spec.positive_label,
                        "split_policy": SCENARIOS[scenario],
                        "block_source": "structure_only_murcko_scaffold",
                        "label_leakage_check": "splits, blocks, and missing-bit masks do not use test labels",
                        "result_source": "rerun_by_us",
                        "passes_smoke_threshold": passes,
                        **base_stats,
                        **stats,
                    }
                )
                time.sleep(args.sleep)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = outdir / "screening_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    write_license_card(outdir / "license_card.md")
    append_log(root, args, manifest_path.relative_to(root), manifest)
    print(manifest.to_markdown(index=False, floatfmt=".4f"))
    print(f"Wrote {manifest_path}")


def download_dataset(spec: DatasetSpec, raw_dir: Path) -> Path:
    suffix = ".csv.gz" if spec.url.endswith(".gz") else ".csv"
    path = raw_dir / f"{spec.key}{suffix}"
    if path.exists():
        return path
    headers = {"User-Agent": "ChemDep-Cal/1.0 public MoleculeNet benchmark"}
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(spec.url, headers=headers, timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
            return path
        except Exception as exc:  # pragma: no cover - network defensive path
            last_error = exc
            time.sleep(3.0 * (attempt + 1))
    raise RuntimeError(f"Could not download {spec.key} from {spec.url}") from last_error


def curate_dataset(raw_path: Path, spec: DatasetSpec) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(raw_path)
    required = [spec.smiles_col, spec.label_col]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"{spec.key} is missing columns {missing}")
    raw = raw[required].dropna().copy()
    raw[spec.label_col] = raw[spec.label_col].astype(int)
    rows = []
    invalid_smiles = 0
    for i, row in raw.iterrows():
        smiles = str(row[spec.smiles_col]).strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles += 1
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or canonical
        rows.append(
            {
                "molecule_chembl_id": f"MLNET_{spec.key.upper()}_{int(i):07d}",
                "canonical_smiles": canonical,
                "murcko_scaffold": scaffold,
                "label": int(row[spec.label_col]),
            }
        )
    curated = pd.DataFrame(rows)
    if curated.empty:
        raise RuntimeError(f"{spec.key} produced no valid molecules")
    duplicates_before = int(curated.duplicated("canonical_smiles").sum())
    conflicts = int(curated.groupby("canonical_smiles")["label"].nunique().gt(1).sum())
    curated = (
        curated.groupby("canonical_smiles", as_index=False)
        .agg(
            molecule_chembl_id=("molecule_chembl_id", "first"),
            murcko_scaffold=("murcko_scaffold", "first"),
            label=("label", lambda x: int(np.mean(x) >= 0.5)),
        )
        .copy()
    )
    curated[FP_COL] = curated["canonical_smiles"].map(fingerprint_hex)
    stats = {
        "n_raw_rows": int(len(raw)),
        "n_valid_smiles_before_dedup": int(len(rows)),
        "invalid_smiles": invalid_smiles,
        "duplicate_canonical_smiles": duplicates_before,
        "conflicting_duplicate_smiles": conflicts,
    }
    return curated, stats


def add_balanced_scaffold_split(panel: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    grouped = (
        panel.groupby("murcko_scaffold", as_index=False)
        .agg(block_size=("label", "size"), positives=("label", "sum"))
        .copy()
    )
    total_n = float(len(panel))
    total_pos = float(panel["label"].sum())
    total_neg = total_n - total_pos
    ratios = {"train": 0.6, "calibration": 0.2, "testpool": 0.2}
    target_n = {split: total_n * ratio for split, ratio in ratios.items()}
    target_pos = {split: max(1.0, total_pos * ratio) for split, ratio in ratios.items()}
    target_neg = {split: max(1.0, total_neg * ratio) for split, ratio in ratios.items()}
    split_names = ["train", "calibration", "testpool"]

    best_mapping: dict[str, str] | None = None
    best_score = float("inf")
    for attempt in range(200):
        rng = np.random.default_rng(seed + attempt * 1009)
        shuffled = grouped.assign(_tie=rng.random(len(grouped))).sort_values(
            ["block_size", "positives", "_tie"], ascending=[False, False, True]
        )
        counts = {split: {"n": 0.0, "pos": 0.0, "neg": 0.0} for split in split_names}
        mapping: dict[str, str] = {}
        for row in shuffled.itertuples(index=False):
            block = str(row.murcko_scaffold)
            size = float(row.block_size)
            pos = float(row.positives)
            neg = size - pos
            scores = []
            for split in split_names:
                n_load = counts[split]["n"] / max(target_n[split], 1.0)
                pos_load = counts[split]["pos"] / max(target_pos[split], 1.0)
                neg_load = counts[split]["neg"] / max(target_neg[split], 1.0)
                overflow = max(0.0, (counts[split]["n"] + size - target_n[split]) / max(target_n[split], 1.0))
                score = (
                    n_load
                    + 0.75 * (pos_load if pos > 0 else neg_load)
                    + 2.0 * overflow
                )
                scores.append(score)
            split = split_names[int(np.argmin(scores))]
            mapping[block] = split
            counts[split]["n"] += size
            counts[split]["pos"] += pos
            counts[split]["neg"] += neg
        quality = sum(abs(counts[s]["n"] - target_n[s]) for s in split_names)
        if quality < best_score:
            best_mapping = dict(mapping)
            best_score = quality
        if all(counts[s]["pos"] > 0 and counts[s]["neg"] > 0 for s in split_names):
            best_mapping = mapping
            break
    if best_mapping is None:
        raise RuntimeError("Failed to construct scaffold split")
    out = panel.copy()
    out["split"] = out["murcko_scaffold"].map(best_mapping)
    return out


def keep_cold_start_train_subset(panel: pd.DataFrame, *, seed: int, train_fraction: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 17_171)
    train = panel[panel["split"] == "train"].copy()
    keep_indices: list[int] = []
    for label, group in train.groupby("label"):
        size = max(1, int(round(len(group) * train_fraction)))
        chosen = rng.choice(group.index.to_numpy(), size=min(size, len(group)), replace=False)
        keep_indices.extend(int(idx) for idx in chosen)
    keep_mask = (panel["split"] != "train") | panel.index.isin(keep_indices)
    return panel.loc[keep_mask].copy().reset_index(drop=True)


def mask_fingerprint_bits(values: pd.Series, *, seed: int, drop_fraction: float) -> pd.Series:
    rng = np.random.default_rng(seed + 24_025)
    n_bits = 2048
    n_drop = int(round(n_bits * drop_fraction))
    drop_idx = np.sort(rng.choice(n_bits, size=n_drop, replace=False))

    def mask_one(hex_value: object) -> str:
        byte_arr = np.frombuffer(bytes.fromhex(str(hex_value)), dtype=np.uint8)
        bits = np.unpackbits(byte_arr, bitorder="big")[:n_bits].copy()
        bits[drop_idx] = 0
        return np.packbits(bits, bitorder="big").tobytes().hex()

    return values.map(mask_one)


def panel_stats(panel: pd.DataFrame) -> dict:
    split_counts = panel.groupby(["split", "label"]).size().unstack(fill_value=0)
    scaffold_counts = panel.groupby("murcko_scaffold").size()
    out = {
        "n_panel": int(len(panel)),
        "n_active": int((panel["label"] == 1).sum()),
        "n_inactive": int((panel["label"] == 0).sum()),
        "active_rate": float((panel["label"] == 1).mean()) if len(panel) else float("nan"),
        "unique_scaffolds": int(panel["murcko_scaffold"].nunique()),
        "min_scaffold_block_size": int(scaffold_counts.min()) if len(scaffold_counts) else 0,
        "median_scaffold_block_size": float(scaffold_counts.median()) if len(scaffold_counts) else 0.0,
        "max_scaffold_block_size": int(scaffold_counts.max()) if len(scaffold_counts) else 0,
    }
    for split in ["train", "calibration", "testpool"]:
        out[f"{split}_active"] = int(split_counts.get(1, pd.Series(dtype=int)).get(split, 0))
        out[f"{split}_inactive"] = int(split_counts.get(0, pd.Series(dtype=int)).get(split, 0))
    return out


def fingerprint_hex(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((2048,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return np.packbits(arr, bitorder="big").tobytes().hex()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_license_card(path: Path) -> None:
    path.write_text(
        """# Public MoleculeNet Benchmark License Card

Source: MoleculeNet raw CSV files served from the DeepChem data bucket:
https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/

The generated panels are derived benchmark artifacts for reproducible
experiments. Before redistribution, re-check the current licenses of the
underlying datasets and cite MoleculeNet plus the original task sources.
""",
        encoding="utf-8",
    )


def append_log(root: Path, args: argparse.Namespace, manifest_path: Path, manifest: pd.DataFrame) -> None:
    log_path = root / "EXPERIMENT_LOG.md"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = {
        "outdir": args.outdir,
        "datasets": args.datasets,
        "scenarios": args.scenarios,
        "seeds": args.seeds,
        "panels": int(len(manifest)),
        "passed_panels": int(manifest["passes_smoke_threshold"].sum()),
        "download_policy": "direct public CSV URLs; heavy Python packages were not installed",
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now} Public MoleculeNet panel construction\n\n")
        f.write("Metadata:\n\n")
        f.write("```json\n")
        f.write(json.dumps(metadata, indent=2, sort_keys=True))
        f.write("\n```\n\n")
        f.write(f"- Manifest: `{manifest_path}`\n")


if __name__ == "__main__":
    main()
