# Data provenance and redistribution

The ChEMBL scorer-input archive in `data/archives/` contains derived molecular
records and scores from the public ChEMBL API. ChEMBL data are available under
[CC BY-SA 3.0](https://chembl.gitbook.io/chembl-interface-documentation/about).
Attribute ChEMBL when using these records and observe its share-alike terms for
redistributed data. The MIT license in this repository applies to the code, not
to upstream ChEMBL records.

The four natural-score tasks (BACE, BBBP, ClinTox CT_TOX, HIV) are downloaded
by `scripts/build_public_moleculenet_panels.py` from the public DeepChem data
bucket. Their raw molecular records and scored panels are not redistributed
here because rights may differ across the underlying tasks. The five-seed
result cells and summaries are included. Consult [MoleculeNet](https://moleculenet.org/)
and each original dataset before redistributing the downloaded records.

All numeric `source_tables/` cells are our own reruns or summaries, not numbers
copied from external papers. Synthetic score perturbations use retrospective
test labels only inside the simulator. Selection code receives perturbed scores,
calibration evidence, and scaffold IDs, not hidden test labels.
