# Public data contract

Every dataset build produces an immutable JSON manifest containing source URL,
upstream release, license, retrieval time, file size, SHA-256, record count,
residue/token count, filtering decisions, and split identity. A directory name
or successful download is not a dataset receipt.

## Planned mixtures

- sequence: UniRef plus public metagenomic sequence sources where licensing and
  redistribution are documented;
- sequence-text: reviewed UniProtKB/Swiss-Prot records, GO annotations, EC
  labels, Pfam descriptions, and publication-derived text with provenance;
- structure: PDB experimental structures and AlphaFold DB predictions with
  source confidence retained;
- evaluation-only: ProteinGym and task-specific frozen benchmark snapshots.

Experimental and predicted structures are separate fields and separate metrics.
Synthetic descriptions are also tagged and cannot be reported as curated text.

Structure provenance is never collapsed: wwPDB experimental coordinates are
CC0 and AlphaFold DB v6 predictions are CC BY 4.0. Predicted structures retain
per-residue pLDDT and are evaluated separately from experimental structures.

Swiss-Prot annotations are indexed by stable entry name and accession from the
official release flat file. Text is joined only when a Swiss-Prot canonical
sequence exactly matches the representative sequence of a UniRef50 cluster;
the pair inherits that cluster's existing split. This intentionally sacrifices
coverage to prevent a reviewed member from crossing a sequence-cluster split.

## Split and contamination rules

All modalities inherit the protein-cluster split. Text or structure records for
a held-out protein may not leak into training through another table. Exact
sequence hashes and configurable identity-cluster IDs are checked before shard
publication.

Validation is used for model selection. Frozen test manifests are sealed until
the candidate checkpoint, inference code revision, and evaluation command are
fixed. Official test data is never used for checkpoint selection.

## Current remote asset

The official UniProt 2026_02 UniRef50 FASTA is verified at
`/ssd/scxi253/pretraining2/data/raw/uniref/2026_02/uniref50.fasta.gz`. Its byte
count and upstream MD5 match the release metadata, an independent SHA-256 is in
`data_sources/uniref50-2026_02.json`, and `gzip -t` passes. This is a verified
raw input. `openprotein-build-uniref` converts
it in one streaming pass into deterministic 98/1/1 hash splits and 96/1/1 CSV
shards, hashes every output shard, and fails closed if the source byte count,
MD5, SHA-256, or record count differs from the pinned release receipt.

Job 1460853 completed that conversion. The immutable manifest at
`/ssd/scxi253/pretraining2/data/processed/uniref50-2026_02/manifest.json` covers
98 shards and 38,794,121 records: 38,018,492 train, 387,876 validation, and
387,753 sealed test. The split contains 12,171,016,896 residues in total.

The older `/ssd/scxi253/datasets/uniref50/data` tree is a separate roughly
4.54M-row third-party snapshot with no upstream byte receipt. It remains
pilot-only and is never conflated with the official 38.79M-record corpus.

The official Swiss-Prot 2026_02 flat file is also verified: 699,031,150 bytes,
upstream MD5 `868b301a6ec93955f4e4355d579d8683`, and independent SHA-256
`741bcb144f98b8d10f0369b145d562b6751bfd17c285e936553aeb9cb54ab592`.
Its verified SQLite annotation index contains 575,503 records. The repaired
exact UniRef-representative join contains 192,401 byte-identical sequence-text
pairs with zero sequence mismatches and a sealed test split.

AlphaFold DB Swiss-Prot v6 is verified as an opaque 28,580,050,944-byte tar with
SHA-256 `30873553b5f9fb5499be24ade2ec7f0fdbae570dda15237c2dbe1096ecda1eee`;
a full `tar -tf` passed. Structure indexing streamed the tar without extracting
files and produced a content-addressed SQLite database with 550,122 predicted
structures, C-alpha coordinates, and pLDDT values.

The ProteinGym v1.3 substitution assay ZIP is held unopened at 43,021,128 bytes
with SHA-256 `3a83766254ac9ac9984ec25cb73c6e010ea4418f5e35f143933e6b6e6473b921`.
Neither its archive listing nor assay rows have been accessed. It remains
evaluation-only until checkpoint and evaluator sealing. The tagged reference
CSV and official baseline-score ZIP are also held by exact byte count and
SHA-256; no ZIP member or benchmark row has been opened.
