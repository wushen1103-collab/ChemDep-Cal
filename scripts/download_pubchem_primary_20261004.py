#!/usr/bin/env python3
"""Fetch PubChem full primary-screen records in checked SID chunks."""

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


PRIMARY = {"AID1798": 626, "AID435034": 628, "AID463087": 449739}
CHUNK = 2000


def one_assay(directory: Path, name: str, aid: int, limit_chunks: int | None) -> None:
    concise_file = directory / f"AID{aid}_concise.csv"
    with concise_file.open(newline="", encoding="utf-8") as handle:
        concise = list(csv.DictReader(handle))
    sids = [str(row["SID"]) for row in concise]
    if len(sids) != len(set(sids)):
        raise AssertionError(f"AID{aid}: duplicate SIDs in concise source")
    output = directory / f"AID{aid}_full_chunks"
    output.mkdir(parents=True, exist_ok=True)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV"
    chunk_rows = []
    pieces = [sids[i:i + CHUNK] for i in range(0, len(sids), CHUNK)]
    for index, subset in enumerate(pieces):
        if limit_chunks is not None and index >= limit_chunks:
            break
        destination = output / f"part_{index:03d}.csv"
        if not destination.exists():
            payload = urllib.parse.urlencode({"sid": ",".join(subset)}).encode("ascii")
            error = None
            for attempt in range(5):
                try:
                    request = urllib.request.Request(
                        url, data=payload,
                        headers={"User-Agent": "ChemDep-Cal-repro/1.0",
                                 "Content-Type": "application/x-www-form-urlencoded"})
                    with urllib.request.urlopen(request, timeout=120) as response:
                        data = response.read()
                    if not data.startswith(b"PUBCHEM_RESULT_TAG,"):
                        raise ValueError("Unexpected full-assay CSV header")
                    temporary = destination.with_suffix(".csv.part")
                    temporary.write_bytes(data)
                    temporary.replace(destination)
                    error = None
                    break
                except Exception as exc:
                    error = exc
                    time.sleep(2 ** attempt)
            if error is not None:
                raise RuntimeError(f"AID{aid} chunk {index} failed") from error
        data = destination.read_bytes()
        with io.StringIO(data.decode("utf-8-sig")) as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader if row["PUBCHEM_SID"] not in
                    ("", "RESULT_TYPE", "RESULT_DESCR", "RESULT_UNIT")]
        received = [row["PUBCHEM_SID"] for row in rows]
        if len(received) != len(subset) or set(received) != set(subset):
            raise AssertionError(f"AID{aid} chunk {index}: SID partition mismatch")
        if "PUBCHEM_ACTIVITY_SCORE" not in rows[0] or "PUBCHEM_EXT_DATASOURCE_SMILES" not in rows[0]:
            raise AssertionError(f"AID{aid} chunk {index}: expected score/SMILES fields absent")
        chunk_rows.append({"part": destination.name, "rows": len(rows),
                           "sha256": hashlib.sha256(data).hexdigest(),
                           "bytes": len(data)})
        print(name, aid, index + 1, "/", len(pieces), len(rows), flush=True)
        time.sleep(.3)
    manifest = {"welqrate_aid": name, "primary_aid": aid, "source_url": url,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "concise_sha256": hashlib.sha256(concise_file.read_bytes()).hexdigest(),
                "expected_sids": len(sids), "chunk_size": CHUNK,
                "chunks": chunk_rows,
                "complete": len(chunk_rows) == len(pieces)}
    (output / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--aid", choices=list(PRIMARY))
    parser.add_argument("--limit-chunks", type=int)
    args = parser.parse_args()
    directory = args.directory.resolve()
    for name, aid in PRIMARY.items():
        if args.aid is None or args.aid == name:
            one_assay(directory, name, aid, args.limit_chunks)


if __name__ == "__main__":
    main()
