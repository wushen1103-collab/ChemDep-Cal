# ChemDep-Cal

Code and evaluation cells for calibration-supervised, one-per-scaffold
finite-budget virtual-screening selection. The primary experiments use a
50-compound budget, labelled calibration scores, and Murcko scaffold groups.
The upstream molecular score, test pool, and capacity are shared within each
matched comparison.

## Contents

- `src/chemdeprc/`: selection and diversity-control primitives. The module
  name is retained for compatibility with the original experiment scripts.
- `scripts/`: the complete small Python experiment-script set for panel
  construction, scoring, calibration ranking, baselines, supplementary
  analyses, fusion, and evaluation. The one unrelated cross-project script
  with a hard-coded private path is excluded.
- `source_tables/`: five-seed cells, summary tables, and supplementary analysis
  CSVs. Two source tables have machine-specific paths converted to repository-
  relative paths. These are
  our reruns, not values transcribed from other papers; see
  [RESULT_INDEX.md](RESULT_INDEX.md) for current-paper entry points.
- `data/archives/chemdep-cal-chembl-scores-clean.tar.gz`: compressed ChEMBL
  scorer inputs, model files, and fixed-selector references (31.7 MB). It
  excludes raw API responses and machine-specific paths.
- `data/conditional_sota_meta_ungated_fifth_20261003/`: frozen primary ranker
  models and outcome references; `data/pr_natural_modern_selectors_20261004/`
  contains the pooled natural-score ranker model.

See [data provenance and licensing](DATA_SOURCES.md) before redistributing
derived molecular records.

## Environment

Python 3.10+ is recommended. RDKit is most reliably installed with conda-forge:

```bash
conda create -n chemdep-cal -c conda-forge python=3.10 rdkit
conda activate chemdep-cal
python -m pip install -r requirements.txt -r requirements-realdata.txt
```

Run commands from the repository root. On Linux/macOS, set `PYTHONPATH=src`;
on PowerShell use `$env:PYTHONPATH="src"`. The numerical result audits do not
require RDKit or XGBoost.

## Verify the archived paper numbers

```bash
python tests/verify_fifth.py
python tests/audit_eighth_fixed_fusion.py
python tests/audit_natural_fixed_fusion.py
```

The first audit reconstructs the fifth-cohort 44.445 versus 42.123 hit means
from 2,000 matched cells per selector. The second joins 6,000 eighth-cohort
cells against all seven matched selectors and checks the 40.401 versus 40.064
target-macro comparison. The third checks 80 natural-score cells across four
datasets and five scorer seeds. The conditional lead pertains to the defined
synthetic score-stress benchmark; natural-score transfer is mixed.

## Rerun ChEMBL selectors

```bash
python scripts/unpack_scores.py
python scripts/eighth_cohort_fixed_selectors_20261005.py --root . --seed 35801
python scripts/eighth_fixed_fusion_20261005.py --root .
```

The archive contains the exact fifth-cohort scored panels for seeds
35501--35505 and the later 30-target scored panels for seeds 35801--35805.
Run the fixed-selector command for each listed seed; the fusion command
processes all five seeds. The fifth-cohort feature and
perturbation code is in `probe_meta_band_v2_20261003.py`,
`train_artifact_meta_ranker_20261003.py`, and
`run_chembl_block_artifact_stress.py`. The source CSVs retain exact cell keys
for paired recalculation even when model-library versions change.

To rebuild the later ChEMBL cohort from the public API, start with the frozen
`data/chembl_modern_eighth_20261005/catalog_metadata.json` and run the cohort
builder and scorer scripts. API releases can change returned activities; the
included score archive is the fixed input for exact replication.

## Natural-score experiments

The four natural tasks are BACE, BBBP, ClinTox CT_TOX, and HIV. Download and
construct all 60 original scenario/seed panels, then fit their scorers:

```bash
python scripts/build_public_moleculenet_panels.py --outdir data/public_moleculenet
python scripts/train_chembl_ecfp_xgb.py --manifest data/public_moleculenet/screening_manifest.csv --outdir data/public_moleculenet_ecfp_xgb --seed 3600 --n-estimators 300
python scripts/natural_fixed_fusion_20261007.py --root .
```

Only scaffold-OOD panels enter the 20-panel natural fusion, while the complete
60-panel build preserves the original scorer-seed assignment. Exact source
version and package-version agreement is needed for byte-identical score
reconstruction. Molecular records are downloaded from their upstream hosts,
not redistributed here; the archived 80 evaluation cells are independently
checkable with the audit above.

## Evaluation protocol

Training labels fit upstream scorers. Calibration labels train the selector;
test labels are hidden from the selector and used by the retrospective
simulator for score perturbation and hit accounting. The 75:25 RF/ChemDep
percentile-rank fusion was selected on development data, then evaluated on a
later 30-target ChEMBL cohort. The included cell tables state target, scorer
seed, perturbation, method, selected count, scaffold count, and hits.

The code is MIT-licensed. ChEMBL-derived records in the archive retain their
upstream CC BY-SA 3.0 terms; see [DATA_SOURCES.md](DATA_SOURCES.md).

Some supplementary scripts require optional `lightgbm`, `catboost`, `torch`,
or `transformers` packages. Author-implementation controls such as OPDiv and
MVS-A additionally require their original public source trees; these third-party
repositories and checkpoints are not mirrored here. Archived result cells allow
their reported comparisons to be checked without those installations.
