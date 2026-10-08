#!/usr/bin/env python3
"""Retrieve standardized PubChem isomeric SMILES for primary-screen CIDs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PRIMARY = (626, 628, 449739)
CHUNK = 1000
URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/IsomericSMILES/CSV"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--limit-chunks", type=int)
    args = parser.parse_args()
    directory = args.directory.resolve()
    cids = set()
    source_hashes = {}
    for aid in PRIMARY:
        source = directory / f"AID{aid}_full_chunks"
        manifest = json.loads((source / "download_manifest.json").read_text(encoding="utf-8"))
        if not manifest["complete"]:
            raise AssertionError(f"AID{aid}: full assay incomplete")
        for record in manifest["chunks"]:
            file = source / record["part"]
            if hashlib.sha256(file.read_bytes()).hexdigest() != record["sha256"]:
                raise AssertionError(f"AID{aid}: source hash changed")
            frame = pd.read_csv(file, skiprows=[1, 2, 3], low_memory=False)
            cids.update(pd.to_numeric(frame.PUBCHEM_CID, errors="coerce").dropna().astype(int))
        source_hashes[str(aid)] = hashlib.sha256(
            (source / "download_manifest.json").read_bytes()).hexdigest()
    ordered = sorted(cids)
    pieces = [ordered[i:i + CHUNK] for i in range(0, len(ordered), CHUNK)]
    output = directory / "cid_isomeric_smiles_chunks"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, subset in enumerate(pieces):
        if args.limit_chunks is not None and index >= args.limit_chunks:
            break
        file = output / f"part_{index:03d}.csv"
        if not file.exists():
            payload = urllib.parse.urlencode({"cid": ",".join(map(str, subset))}).encode("ascii")
            error = None
            for attempt in range(5):
                try:
                    request = urllib.request.Request(
                        URL, data=payload,
                        headers={"User-Agent": "ChemDep-Cal-repro/1.0",
                                 "Content-Type": "application/x-www-form-urlencoded"})
                    with urllib.request.urlopen(request, timeout=120) as response:
                        data = response.read()
                    if not data.startswith(b'"CID","SMILES"'):
                        raise ValueError("Unexpected PubChem CID property header")
                    temporary = file.with_suffix(".csv.part")
                    temporary.write_bytes(data)
                    temporary.replace(file)
                    error = None
                    break
                except Exception as exc:
                    error = exc
                    time.sleep(2 ** attempt)
            if error is not None:
                raise RuntimeError(f"CID chunk {index} failed") from error
        data = file.read_bytes()
        with io.StringIO(data.decode("utf-8-sig")) as handle:
            rows = list(csv.DictReader(handle))
        received = [int(row["CID"]) for row in rows]
        if len(received) != len(set(received)) or not set(received).issubset(subset):
            raise AssertionError(f"CID chunk {index}: malformed response")
        records.append({"part": file.name, "requested": len(subset),
                        "returned": len(rows), "sha256": hashlib.sha256(data).hexdigest()})
        print(index + 1, "/", len(pieces), "CID properties", len(rows), flush=True)
        time.sleep(.25)
    manifest = {"url": URL, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "source_manifest_sha256": source_hashes,
                "unique_cids": len(ordered), "chunk_size": CHUNK,
                "chunks": records, "complete": len(records) == len(pieces)}
    (output / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("total unique CIDs", len(ordered), flush=True)


if __name__ == "__main__":
    main()
