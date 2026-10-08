#!/usr/bin/env python3
"""Label-free MoLFormer embeddings for already exposed MF-PCBA cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_ID = "ibm-research/MoLFormer-XL-both-10pct"
REVISION = "compat-v4"
MAX_LENGTH = 202


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "data/mfpcba_molformer_exposed_development_20261005"
    output.mkdir(parents=True, exist_ok=True)
    sources, smiles = [], set()
    for campaign in ("twenty", "twelve_new"):
        directory = root / f"data/mfpcba_{campaign}_20261004"
        manifest = json.loads((directory / "cohort_manifest.json").read_text())
        for name in manifest["eligible_tasks"]:
            source = (directory / f"{name}_cohort.parquet" if campaign == "twenty"
                      else directory / "cohorts" / f"{name}_cohort.parquet")
            frame = pd.read_parquet(source, columns=["canonical_smiles"])
            smiles.update(frame.canonical_smiles.astype(str))
            sources.append({"campaign": campaign, "task": name, "source": str(source),
                            "molecules": len(frame)})
    unique = sorted(smiles)
    if args.smoke:
        unique = unique[:8]
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
                                              trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, revision=REVISION,
                                      trust_remote_code=True,
                                      deterministic_eval=True).cuda().eval()
    revision = getattr(model.config, "_commit_hash", None)
    outputs, truncated = [], 0
    for start in range(0, len(unique), args.batch_size):
        batch = unique[start:start + args.batch_size]
        lengths = tokenizer(batch, add_special_tokens=True, truncation=False)["input_ids"]
        truncated += sum(len(ids) > MAX_LENGTH for ids in lengths)
        tokens = tokenizer(batch, padding=True, truncation=True,
                           max_length=MAX_LENGTH, return_tensors="pt")
        tokens = {key: value.cuda(non_blocking=True) for key, value in tokens.items()}
        with torch.inference_mode():
            pooled = model(**tokens).pooler_output
            outputs.append(pooled.float().cpu().numpy())
        print(f"embedded {min(start + len(batch), len(unique))}/{len(unique)}", flush=True)
    embeddings = np.concatenate(outputs).astype(np.float32)
    if not np.isfinite(embeddings).all():
        raise AssertionError("Nonfinite embedding")
    if args.smoke:
        print("smoke", embeddings.shape, "truncated", truncated, flush=True)
        return
    np.savez_compressed(output / "embeddings.npz", smiles=np.asarray(unique),
                        embedding=embeddings)
    metadata = {
        "status": "post hoc exposed-assay development; no new follow-up outcomes accessed",
        "model": MODEL_ID, "requested_revision": REVISION, "commit": revision,
        "pooling": "author's attention-mask mean pooler",
        "max_tokens": MAX_LENGTH, "truncated_molecules": truncated,
        "molecules": len(unique), "dimensions": int(embeddings.shape[1]),
        "smiles_sha256": hashlib.sha256("\n".join(unique).encode()).hexdigest(),
        "sources": sources,
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: metadata[key] for key in
                      ("commit", "molecules", "dimensions", "truncated_molecules")}))


if __name__ == "__main__":
    main()
