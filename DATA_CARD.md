# OpenProtein public-data card

Status: verified build receipts through 2026-09-02. Training, validation, and
sealed test evidence are kept separate.

## Sequence corpus

UniRef50 release 2026_02 is the official sequence source. The verified gzip is
8,770,260,598 bytes with SHA-256
`e44b7e22dd719346997a6ead032a44bb214cea3dd1793b843b4b7a51836b95f7`.
Its deterministic processed corpus contains 38,794,121 records and
12,171,016,896 residues:

| Split | Records | Residues | Use |
| --- | ---: | ---: | --- |
| Train | 38,018,492 | 11,927,459,383 | optimization |
| Validation | 387,876 | 121,538,068 | checkpoint selection |
| Test | 387,753 | 122,019,445 | sealed |

Processed-manifest SHA-256:
`be8cbb50fe6b8811f486b5cb9a2ad72f1018c9105d239734d70ca7deeef97530`.
An older third-party 4.54M-row tree is pilot-only and never counts as official
training evidence.

## Scratch-pretraining MGnify90 corpus

The scratch 7B data path uses MGnify Proteins release 2024_04, whose official
README reports 717,738,164 90%-identity cluster representatives and
121,126,699,551 residues. A deterministic 50% accession-hash selection with a
98/1/1 train/validation/test split is designed to leave about 59.35B training
residues; the builder rejects the result if train residues are below 50B.

The first transfer candidate was 717 FASTA+zstd shards from `LiteFold/Mgnify`,
pinned to repository commit
`42712c93c6995a75f6f1d89931f95ddd78567b30`. The pinned tree contains
75,096,691,452 compressed bytes. Each shard must match its Hugging Face LFS
SHA-256, after which a complete streaming pass must reproduce the official EBI
record and residue totals. This validates the pinned mirror objects and semantic
release totals, not byte identity with EBI's 79,301,981,114-byte gzip whose MD5
is `c7e812f9542d48cf425c5329b8571e68`.

All 717 pinned shards have been retrieved and SHA-256 verified. The retrieval
receipt is `VERIFIED_SHARDED_INPUT`, and its tree-manifest SHA-256 is
`02d3617228119cd7c79080f8e56acc81b18c1744dc8472cdc54e3b4a24af06f4`.
However, the required streaming pass found 716,738,164 records and
120,943,014,900 residues, short of EBI by exactly 1,000,000 records and
183,684,651 residues. The packed builder rejected the source and atomically
removed its temporary output. Thus `VERIFIED_SHARDED_INPUT` means only that the
pinned mirror bytes were retrieved intact; it is not a complete-corpus receipt.
The mirror is not admissible for formal training unless a separately verified
official prefix closes both exact gaps with a continuous accession boundary.

The official gzip prefix does close those gaps exactly: its first 1,000,000
records contain 183,684,651 residues, and official record 1,000,001 matches the
mirror's first record (`MGYP004618177781`) in accession, length, and sequence
SHA-256. This diagnoses the mirror omission but does not promote the mirror to
formal training data. The accepted fallback requires the complete official gzip
to pass the published byte count and MD5, gzip integrity, and the full record
and residue totals before packed output is published.

The complete official gzip was subsequently retrieved and verified at exactly
79,301,981,114 bytes with upstream MD5
`c7e812f9542d48cf425c5329b8571e68`, SHA-256
`e1597a12a2f10b0574c172cb820576916d86ebd24e680801506834a2191cdb08`,
and a passed gzip integrity check. A full semantic pass reproduced all
717,738,164 records and 121,126,699,551 residues. The deterministic 50% packed
selection contains 60,566,268,552 residues, including 59,354,481,535 training
residues in 128 content-addressed shards. Retrieval receipt SHA-256 is
`40d7ee0e02ee0f20330c982acf20fb67eef7e518c179be6ded1dfc07b267f530`.

The authoritative EBI data is CC0 1.0; the mirror distribution card says CC BY
4.0, so both notices are retained. This section currently describes an intact
but semantically incomplete raw mirror that was rejected, not completed training
data.

## Curated sequence-text pairs

UniProtKB/Swiss-Prot release 2026_02 provides reviewed annotations. The raw file
contains 575,503 records and has SHA-256
`741bcb144f98b8d10f0369b145d562b6751bfd17c285e936553aeb9cb54ab592`.
Text is joined only when a Swiss-Prot canonical sequence exactly equals the
representative sequence of an existing UniRef50 cluster. This yields 192,401
pairs with zero sequence mismatches:

| Split | Pairs | Residues | Use |
| --- | ---: | ---: | --- |
| Train | 188,473 | 76,448,555 | optimization/tokenizer training |
| Validation | 1,942 | 787,236 | model selection |
| Test | 1,986 | 847,932 | sealed |

Join-manifest SHA-256:
`eebf7babe67352fba6b3c061233d8318a3a7e0d10b0e2e4f8ce1b022d1a0d4e4`.
The 32,768-token unigram tokenizer was trained only on 188,473 training
annotations; its SHA-256 is
`920d8d50f98c3d1aea1285b4126c09d283f771302b0129c38258be3df457cd98`.

## Predicted structures

AlphaFold DB Swiss-Prot v6 supplies predicted coordinates under CC BY 4.0. The
verified tar is 28,580,050,944 bytes with SHA-256
`30873553b5f9fb5499be24ade2ec7f0fdbae570dda15237c2dbe1096ecda1eee`.
The streamed index contains 550,122 structures and 195,055,067 residues. Joins
require accession and exact full-length sequence agreement. Training retains
pLDDT and samples C-alpha pairs only above the configured confidence threshold.
These records are always labelled predicted, not experimental.

Structure-manifest SHA-256:
`dac92d404564340b1fea06316f63b4737a4cec1d616332909bed0e850fdf6acb`.

## Frozen evaluation artifacts

ProteinGym v1.3 is evaluation-only. The assay ZIP, tagged reference CSV, and
official baseline-score ZIP are held by exact byte count and SHA-256 in archive
manifest
`3e518d343fe1ef9d9d0b5383bfaa990c643dfbb52486ad90326d47a6048fe3b7`.
No ZIP listing, assay row, or official score row is opened until the final
checkpoint and evaluator are sealed. Frozen test results are never used for
model selection.

## Processing and leakage controls

- Modalities inherit the protein-cluster split; held-out proteins cannot return
  through text or structure tables.
- Exact sequence matching is fail-closed, and zero-pair joins are rejected.
- Validation masks and file lists are fixed across sequence checkpoint
  selection.
- Raw files, processed databases, checkpoints, and evaluation seals are bound
  by independent SHA-256 receipts.
- Experimental and predicted structures, official and pilot corpora, and
  validation and frozen-test results may not be merged into one evidence label.

## Licensing and known biases

UniProt and AlphaFold DB data are CC BY 4.0 and require attribution. ProteinGym
project code is MIT. Swiss-Prot exact-match filtering favors reviewed proteins
with curated annotations. UniRef clustering, taxonomic sampling, annotation
availability, and AlphaFold confidence introduce coverage and representation
biases. Absence of an annotation is not evidence that a function is absent.
