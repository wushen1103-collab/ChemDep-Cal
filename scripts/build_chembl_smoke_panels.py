#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

DEFAULT_TARGET_IDS = [
    "CHEMBL203",   # EGFR
    "CHEMBL279",   # KDR/VEGFR2
    "CHEMBL240",   # KCNH2/hERG
    "CHEMBL1862",  # ABL1
    "CHEMBL1827",  # PDE5A
    "CHEMBL217",   # DRD2
    "CHEMBL235",   # PPARG
    "CHEMBL267",   # SRC
    "CHEMBL301",   # CDK2
    "CHEMBL2971",  # JAK2
    "CHEMBL2842",  # MTOR
    "CHEMBL214",   # HTR1A
    "CHEMBL228",   # SERT
    "CHEMBL220",   # ACHE
    "CHEMBL205",   # CA2
]

ENDPOINT_TYPES = {"IC50", "EC50", "KI", "KD"}
RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class TargetInfo:
    target_chembl_id: str
    pref_name: str
    organism: str
    target_type: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data/chembl_smoke")
    parser.add_argument("--target-ids", nargs="+", default=DEFAULT_TARGET_IDS)
    parser.add_argument("--max-targets", type=int, default=10)
    parser.add_argument("--max-activities-per-target", type=int, default=8000)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--min-active", type=int, default=50)
    parser.add_argument("--min-inactive", type=int, default=200)
    parser.add_argument("--min-scaffolds", type=int, default=50)
    parser.add_argument("--active-threshold", type=float, default=6.0)
    parser.add_argument("--inactive-threshold", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()
    if args.inactive_threshold >= args.active_threshold:
        raise ValueError("--inactive-threshold must be smaller than --active-threshold")

    outdir = Path(args.outdir)
    raw_dir = outdir / "raw"
    panel_dir = outdir / "panels"
    raw_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    targets = resolve_targets(args.target_ids)
    manifest_rows = []
    accepted = 0
    for target in targets:
        if accepted >= args.max_targets:
            break
        print(f"Fetching {target.target_chembl_id} {target.pref_name}", flush=True)
        raw_records = fetch_activities(
            target,
            raw_dir=raw_dir,
            max_records=args.max_activities_per_target,
            page_size=args.page_size,
            sleep=args.sleep,
        )
        panel, stats = curate_panel(
            raw_records,
            target,
            seed=args.seed + accepted,
            active_threshold=args.active_threshold,
            inactive_threshold=args.inactive_threshold,
        )
        if len(panel) == 0:
            continue

        panel_path = panel_dir / f"{target.target_chembl_id}_panel.csv"
        panel.to_csv(panel_path, index=False)
        digest = sha256_file(panel_path)
        passes = (
            stats["n_active"] >= args.min_active
            and stats["n_inactive"] >= args.min_inactive
            and stats["unique_scaffolds"] >= args.min_scaffolds
        )
        manifest_rows.append(
            {
                "target_chembl_id": target.target_chembl_id,
                "pref_name": target.pref_name,
                "organism": target.organism,
                "target_type": target.target_type,
                "panel_path": str(panel_path),
                "panel_sha256": digest,
                "source": "ChEMBL webresource API",
                "source_mode": f"smoke_limited_{args.max_activities_per_target}_activities",
                "license": "ChEMBL data; see license_card.md",
                "active_threshold": f"median_pchembl >= {args.active_threshold:g}",
                "inactive_threshold": f"median_pchembl <= {args.inactive_threshold:g}",
                "gray_zone": f"{args.inactive_threshold:g} < median_pchembl < {args.active_threshold:g}",
                "split_policy": "label-free scaffold split, 60/20/20 train/calibration/testpool by scaffold",
                "block_source": "structure_only_murcko_scaffold",
                "label_leakage_check": "blocks_and_splits_do_not_use_test_labels",
                "passes_smoke_threshold": passes,
                **stats,
            }
        )
        accepted += int(passes)
        time.sleep(args.sleep)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = outdir / "screening_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    write_license_card(outdir / "license_card.md")
    print(manifest.to_markdown(index=False, floatfmt=".4f"))
    print(f"Wrote {manifest_path}")


def resolve_targets(target_ids: list[str]) -> list[TargetInfo]:
    targets: list[TargetInfo] = []
    seen: set[str] = set()
    for tid in target_ids:
        if tid in seen:
            continue
        seen.add(tid)
        hit = chembl_get_json(f"{CHEMBL_API}/target/{tid}.json")
        organism = hit.get("organism") or ""
        target_type = hit.get("target_type") or ""
        if organism != "Homo sapiens" or target_type != "SINGLE PROTEIN":
            print(f"Skipping {tid}: organism={organism!r}, target_type={target_type!r}", flush=True)
            continue
        targets.append(
            TargetInfo(
                target_chembl_id=tid,
                pref_name=hit.get("pref_name") or tid,
                organism=organism,
                target_type=target_type,
            )
        )
    return targets


def fetch_activities(
    target: TargetInfo,
    *,
    raw_dir: Path,
    max_records: int,
    page_size: int,
    sleep: float,
) -> list[dict]:
    cache_path = raw_dir / f"{target.target_chembl_id}_activities.jsonl"
    if cache_path.exists():
        return [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    records: list[dict] = []
    offset = 0
    page_size = max(1, min(page_size, 1000))
    while len(records) < max_records:
        params = {
            "target_chembl_id": target.target_chembl_id,
            "pchembl_value__isnull": "false",
            "limit": page_size,
            "offset": offset,
        }
        payload = chembl_get_json(f"{CHEMBL_API}/activity.json", params=params)
        page_records = payload.get("activities", [])
        if not page_records:
            break
        remaining = max_records - len(records)
        records.extend(dict(item) for item in page_records[:remaining])
        meta = payload.get("page_meta", {})
        print(
            f"  {target.target_chembl_id}: fetched {len(records)}/{min(max_records, meta.get('total_count', max_records))}",
            flush=True,
        )
        if not meta.get("next") or len(page_records) < page_size:
            break
        offset += page_size
        time.sleep(sleep)
    with cache_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return records


def chembl_get_json(url: str, params: dict | None = None, retries: int = 3) -> dict:
    headers = {"User-Agent": "ChemDep-Cal/1.0 (research benchmark)"}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - network defensive path
            last_error = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"ChEMBL request failed after {retries} attempts: {url}") from last_error


def curate_panel(
    records: list[dict],
    target: TargetInfo,
    *,
    seed: int,
    active_threshold: float,
    inactive_threshold: float,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    invalid_smiles = 0
    for record in records:
        endpoint = normalize_endpoint(record.get("standard_type"))
        if endpoint not in ENDPOINT_TYPES:
            continue
        smiles = record.get("canonical_smiles")
        pchembl = parse_float(record.get("pchembl_value"))
        mol_id = record.get("molecule_chembl_id")
        if not smiles or not mol_id or pchembl is None:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles += 1
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or smiles
        rows.append(
            {
                "target_chembl_id": target.target_chembl_id,
                "pref_name": target.pref_name,
                "molecule_chembl_id": mol_id,
                "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
                "murcko_scaffold": scaffold,
                "endpoint": endpoint,
                "pchembl_value": pchembl,
                "assay_chembl_id": record.get("assay_chembl_id"),
                "document_chembl_id": record.get("document_chembl_id"),
            }
        )
    if not rows:
        return pd.DataFrame(), empty_stats(records, invalid_smiles)

    raw = pd.DataFrame(rows)
    grouped = raw.groupby("molecule_chembl_id")
    panel = grouped.agg(
        target_chembl_id=("target_chembl_id", "first"),
        pref_name=("pref_name", "first"),
        canonical_smiles=("canonical_smiles", "first"),
        murcko_scaffold=("murcko_scaffold", "first"),
        median_pchembl=("pchembl_value", "median"),
        n_measurements=("pchembl_value", "size"),
        endpoints=("endpoint", lambda x: ";".join(sorted(set(x)))),
        assay_count=("assay_chembl_id", pd.Series.nunique),
        document_count=("document_chembl_id", pd.Series.nunique),
    ).reset_index()
    panel["label"] = np.select(
        [panel["median_pchembl"] >= active_threshold, panel["median_pchembl"] <= inactive_threshold],
        [1, 0],
        default=-1,
    )
    gray_count = int((panel["label"] == -1).sum())
    panel = panel[panel["label"] != -1].copy()
    if panel.empty:
        return panel, empty_stats(records, invalid_smiles, gray_count=gray_count)
    panel = add_scaffold_split(panel, seed=seed)
    panel["ecfp4_radius2_nbits2048"] = panel["canonical_smiles"].map(fingerprint_hex)

    scaffold_counts = panel.groupby("murcko_scaffold").size()
    split_counts = panel.groupby(["split", "label"]).size().unstack(fill_value=0)
    stats = {
        "n_raw_records": len(records),
        "n_endpoint_records": len(raw),
        "n_unique_molecules_before_gray": int(len(grouped)),
        "n_panel": int(len(panel)),
        "n_active": int((panel["label"] == 1).sum()),
        "n_inactive": int((panel["label"] == 0).sum()),
        "active_rate": float((panel["label"] == 1).mean()),
        "gray_molecules_removed": gray_count,
        "invalid_smiles": invalid_smiles,
        "unique_scaffolds": int(panel["murcko_scaffold"].nunique()),
        "min_scaffold_block_size": int(scaffold_counts.min()),
        "median_scaffold_block_size": float(scaffold_counts.median()),
        "max_scaffold_block_size": int(scaffold_counts.max()),
        "train_active": int(split_counts.get(1, pd.Series()).get("train", 0)),
        "train_inactive": int(split_counts.get(0, pd.Series()).get("train", 0)),
        "calibration_active": int(split_counts.get(1, pd.Series()).get("calibration", 0)),
        "calibration_inactive": int(split_counts.get(0, pd.Series()).get("calibration", 0)),
        "testpool_active": int(split_counts.get(1, pd.Series()).get("testpool", 0)),
        "testpool_inactive": int(split_counts.get(0, pd.Series()).get("testpool", 0)),
    }
    return panel, stats


def add_scaffold_split(panel: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scaffolds = panel["murcko_scaffold"].drop_duplicates().to_numpy()
    rng.shuffle(scaffolds)
    scaffold_sizes = panel.groupby("murcko_scaffold").size().to_dict()
    total = len(panel)
    targets = {"train": 0.6 * total, "calibration": 0.2 * total, "testpool": 0.2 * total}
    split_by_scaffold: dict[str, str] = {}
    counts = {"train": 0, "calibration": 0, "testpool": 0}
    for scaffold in sorted(scaffolds, key=lambda s: scaffold_sizes[s], reverse=True):
        split = min(counts, key=lambda key: counts[key] / targets[key])
        split_by_scaffold[scaffold] = split
        counts[split] += scaffold_sizes[scaffold]
    panel = panel.copy()
    panel["split"] = panel["murcko_scaffold"].map(split_by_scaffold)
    return panel


def normalize_endpoint(value: object) -> str:
    return str(value or "").upper().strip()


def parse_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def fingerprint_hex(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((2048,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return np.packbits(arr).tobytes().hex()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def empty_stats(records: list[dict], invalid_smiles: int, gray_count: int = 0) -> dict:
    return {
        "n_raw_records": len(records),
        "n_endpoint_records": 0,
        "n_unique_molecules_before_gray": 0,
        "n_panel": 0,
        "n_active": 0,
        "n_inactive": 0,
        "active_rate": 0.0,
        "gray_molecules_removed": gray_count,
        "invalid_smiles": invalid_smiles,
        "unique_scaffolds": 0,
        "min_scaffold_block_size": 0,
        "median_scaffold_block_size": 0.0,
        "max_scaffold_block_size": 0,
        "train_active": 0,
        "train_inactive": 0,
        "calibration_active": 0,
        "calibration_inactive": 0,
        "testpool_active": 0,
        "testpool_inactive": 0,
    }


def write_license_card(path: Path) -> None:
    path.write_text(
        """# ChEMBL Smoke Data License Card

Source: ChEMBL webresource API, https://www.ebi.ac.uk/chembl/

This smoke dataset stores derived target panels and metadata for experiment
development. Do not redistribute raw ChEMBL dumps through this repository.
Before public release, re-check the current ChEMBL license page and cite the
specific ChEMBL release used.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
