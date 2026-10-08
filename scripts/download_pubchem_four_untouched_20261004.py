#!/usr/bin/env python3
"""Download four fixed primary-hit/follow-up campaigns from PubChem PUG-REST."""

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


CAMPAIGNS = {
    "AID1843": (1672, (2032, 463252)),
    "AID2258": (2239, (2287,)),
    "AID2689": (2661, (2821,)),
    "AID488997": (488975, (504840, 493221, 588401)),
}
BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
SID_CHUNK = 1000
CID_CHUNK = 1000


def request_bytes(url: str, payload: bytes | None = None) -> bytes:
    error = None
    for attempt in range(6):
        try:
            headers = {"User-Agent": "ChemDep-Cal-independent-replication/1.0"}
            if payload is not None:
                headers["Content-Type"] = "application/x-www-form-urlencoded"
            request = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except Exception as exc:
            error = exc
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Could not fetch {url}") from error


def checked_file(path: Path, url: str, *, payload: bytes | None = None,
                 prefix: bytes | None = None) -> bytes:
    if not path.exists():
        data = request_bytes(url, payload)
        if prefix is not None and not data.startswith(prefix):
            raise AssertionError(f"Unexpected response header: {url}")
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(path)
    data = path.read_bytes()
    if prefix is not None and not data.startswith(prefix):
        raise AssertionError(f"Unexpected cached header: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source_files = []
    all_cids = set()
    for name, (primary_aid, followups) in CAMPAIGNS.items():
        sid_url = f"{BASE}/assay/aid/{primary_aid}/sids/TXT?sids_type=active"
        sid_path = directory / f"AID{primary_aid}_active_sids.txt"
        sid_data = checked_file(sid_path, sid_url)
        sid_text = sid_data.decode("utf-8-sig").split()
        if not sid_text or not all(item.isdigit() for item in sid_text):
            raise AssertionError(f"AID{primary_aid}: malformed active SID response")
        sids = [int(item) for item in sid_text]
        if len(sids) != len(set(sids)):
            raise AssertionError(f"AID{primary_aid}: duplicate active SID")
        source_files.append({"file": sid_path.name, "url": sid_url,
                             "rows": len(sids), "sha256": hashlib.sha256(sid_data).hexdigest()})
        output = directory / f"AID{primary_aid}_active_full_chunks"
        output.mkdir(parents=True, exist_ok=True)
        pieces = [sids[i:i + SID_CHUNK] for i in range(0, len(sids), SID_CHUNK)]
        chunks = []
        for index, subset in enumerate(pieces):
            path = output / f"part_{index:03d}.csv"
            url = f"{BASE}/assay/aid/{primary_aid}/CSV"
            payload = urllib.parse.urlencode({"sid": ",".join(map(str, subset))}).encode("ascii")
            data = checked_file(path, url, payload=payload, prefix=b"PUBCHEM_RESULT_TAG,")
            with io.StringIO(data.decode("utf-8-sig")) as handle:
                table = list(csv.DictReader(handle))
            rows = [row for row in table if row.get("PUBCHEM_SID", "") not in
                    ("", "RESULT_TYPE", "RESULT_DESCR", "RESULT_UNIT")]
            received = [int(row["PUBCHEM_SID"]) for row in rows]
            if len(received) != len(subset) or set(received) != set(subset):
                raise AssertionError(f"AID{primary_aid} chunk {index}: SID mismatch")
            if any(row["PUBCHEM_ACTIVITY_OUTCOME"] != "Active" for row in rows):
                raise AssertionError(f"AID{primary_aid} chunk {index}: non-active source row")
            if "PUBCHEM_ACTIVITY_SCORE" not in rows[0]:
                raise AssertionError(f"AID{primary_aid} chunk {index}: missing score")
            all_cids.update(int(row["PUBCHEM_CID"]) for row in rows if row["PUBCHEM_CID"].isdigit())
            chunks.append({"part": path.name, "rows": len(rows),
                           "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
            print(name, "primary chunk", index + 1, "/", len(pieces), flush=True)
            time.sleep(.25)
        manifest = {"campaign": name, "primary_aid": primary_aid, "active_sid_count": len(sids),
                    "sid_source_sha256": hashlib.sha256(sid_data).hexdigest(), "chunks": chunks,
                    "complete": len(chunks) == len(pieces)}
        manifest_path = output / "download_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for followup_aid in followups:
            path = directory / f"AID{followup_aid}_concise.csv"
            url = f"{BASE}/assay/aid/{followup_aid}/concise/CSV"
            data = checked_file(path, url, prefix=b'"AID","SID","CID",')
            with io.StringIO(data.decode("utf-8-sig")) as handle:
                rows = list(csv.DictReader(handle))
            required = {"SID", "CID", "Activity Outcome"}
            if not rows or not required.issubset(rows[0]):
                raise AssertionError(f"AID{followup_aid}: unexpected follow-up schema")
            source_files.append({"file": path.name, "url": url, "rows": len(rows),
                                 "sha256": hashlib.sha256(data).hexdigest()})
            print(name, "follow-up", followup_aid, len(rows), flush=True)
            time.sleep(.25)

    ordered = sorted(all_cids)
    output = directory / "cid_isomeric_smiles_chunks"
    output.mkdir(parents=True, exist_ok=True)
    pieces = [ordered[i:i + CID_CHUNK] for i in range(0, len(ordered), CID_CHUNK)]
    cid_chunks = []
    url = f"{BASE}/compound/cid/property/IsomericSMILES/CSV"
    for index, subset in enumerate(pieces):
        path = output / f"part_{index:03d}.csv"
        payload = urllib.parse.urlencode({"cid": ",".join(map(str, subset))}).encode("ascii")
        data = checked_file(path, url, payload=payload, prefix=b'"CID","SMILES"')
        with io.StringIO(data.decode("utf-8-sig")) as handle:
            rows = list(csv.DictReader(handle))
        received = [int(row["CID"]) for row in rows]
        if len(received) != len(set(received)) or not set(received).issubset(subset):
            raise AssertionError(f"CID chunk {index}: malformed property response")
        cid_chunks.append({"part": path.name, "requested": len(subset), "returned": len(rows),
                           "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        print("CID properties", index + 1, "/", len(pieces), flush=True)
        time.sleep(.25)
    (output / "download_manifest.json").write_text(json.dumps({
        "unique_cids": len(ordered), "chunks": cid_chunks,
        "complete": len(cid_chunks) == len(pieces), "url": url,
    }, indent=2) + "\n", encoding="utf-8")
    (directory / "source_manifest.json").write_text(json.dumps({
        "campaigns": CAMPAIGNS, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "sources": source_files, "primary_candidate_cids": len(ordered),
    }, indent=2) + "\n", encoding="utf-8")
    print("All four campaigns downloaded; unique primary-hit CIDs", len(ordered), flush=True)


if __name__ == "__main__":
    main()
