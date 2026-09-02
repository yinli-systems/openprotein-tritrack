# OpenProtein scratch 7B model card

Status: scale gates complete and formal pretraining running, 2026-08-24. No
scratch-pretraining completion, validation selection, frozen-test, or quality
claim is currently approved.

## Model

`openprotein-scratch-7b` is a randomly initialized bidirectional protein masked
language model. It has 35 pre-normalized Transformer layers, width 4,096, 32
attention heads, feed-forward width 16,384, rotary positions, GELU activations,
a tied 33-token input/output vocabulary, and a 1,024-token context. Its exact
parameter count is 7,065,227,297, and every parameter is trainable.

No ESMC, ESM-2, adapter, or other model checkpoint is loaded. Only a checkpoint
from the same hash-bound random-initialization lineage may be used to resume a
run.

## Optimization and evidence contract

The objective is 15% residue MLM with ESM-style 80/10/10 corruption. Loss and
gradients are normalized by the exact global number of masked targets across
all ranks and accumulation micro-batches. The implementation uses BF16 compute,
FP32 sharded parameters and AdamW states, FSDP2, activation checkpointing,
restartable packed-data cursors, and atomic SHA-256 checkpoint manifests.

Existing 125M, 1B, and bounded 7B receipts from the former per-rank-mean
objective remain valid only for throughput, memory, and checkpoint mechanics.
The corrected 125M and 1B 1,000-step and recovery gates have passed. The 1B
run reached 18,653,687 cumulative tokens at step 1,100, and its final nine-file,
12,143,440,244-byte checkpoint was independently verified against its manifest.
A corrected large-batch eight-RTX-4090 7B checkpoint gate completed 1,000 steps
and 100,559,889 tokens, then loaded the full step-1,000 checkpoint and continued
to step 1,100 and 110,480,043 tokens under identical provenance. The final
169.6GB retained step-zero/step-1,100 artifact set was independently rehashed
after archival. On the formal MGnify corpus, the selected eight-RTX-5090
throughput configuration completed 50 finite steps and 10,936,363 tokens at
23,678 average and 24,461 median tokens/s, with 26.29 GiB peak allocation. All
of these receipts remain `SCALE_TEST` evidence.

Formal job 1464257 was stopped and superseded at step 1,770 after an audit found
that its immutable trainer would have tied LR decay to each operational segment
boundary. Because it had not yet passed the 2,000-step warmup, it executed zero
affected decay steps, but its provenance cannot be resumed. Replacement job
1465137 starts from a new random initialization in read-only source snapshot
`efb2df4391504bed59822a36972a2644d4c03e04800bbad4483a3549fa84aa91`.
Its 5,000-step operational boundary is independent of the global 260,000-step
LR horizon. The 84,802,445,816 manifest-declared bytes in its step-zero
checkpoint passed an independent full SHA-256 audit; manifest SHA-256 is
`cabecc75bbc7e9a3e7aa6778ad11effb08a43e7279531464c64177e7271379a2`.
The run also crossed the first post-warmup boundary with 202 finite continuous
rows through step 2,020 and 430,598,467 tokens. The observed step-2,010 and
step-2,020 learning rates exactly match the global step-260,000 decay and differ
from the rejected segment-dependent schedule; post-warmup health-receipt
SHA-256 is
`8cf00abfb3529c88b798d49e8a06bf022e74dfc6b5cdb77f12ebfd449e4c7e42`.
The fully verified step-2,500 checkpoint then became the sole active restart
point under a latest-one sliding retention policy. Raw step-zero checkpoint
files are no longer retained; their prior integrity/provenance receipt remains
the evidence boundary.
The sole active checkpoint root was subsequently migrated behind its unchanged
logical path to a separate HDD quota after every destination file was rehashed.
This is storage placement only; model, config, source, scheduler dependency, and
evidence level are unchanged.
This remains operational running-state evidence only, not a completed segment
or `SCRATCH_PRETRAIN` receipt.

## Data

Verified UniRef50 2026_02 packed data is used for engineering tests. It contains
12,171,016,896 residues and is not accepted as a sufficient 7B pretraining
corpus. The formal minimum is 50 billion hash-bound training tokens from a
complete, verified MGnify90 2024_04 source and deterministic train/validation/
test split. The pinned mirror failed the official source-total check. The
replacement complete official EBI gzip matches 79,301,981,114 bytes and MD5
`c7e812f9542d48cf425c5329b8571e68`, passes gzip integrity, and is retained in a
verified read-only archive. The deterministic packed selection contains
59,354,481,535 train residues with zero unknown residues; training shards are
bound by train-manifest SHA-256
`7640aa5a28defb5a371910dcd1e43720597a27c4c5fbc283a0bbac0721530db8`.

## Evaluation boundary

Validation-only checkpoint selection must be fixed before any frozen test is
opened. ProteinGym v1.3 assay and official baseline rows remain sealed. No claim
of superiority over ESM-2 15B, ESMC-6B, or any task-specific model is allowed
without the one-time frozen paired-evaluation receipt.

## Intended use and limitations

The intended use is research on protein sequence representations and controlled
downstream evaluation. The model is not a medical device or a source of claims
about safety, efficacy, function, synthesizability, or release suitability.
There is currently no completed pretrained scratch checkpoint, no demonstrated
downstream quality, and no frozen benchmark result. The hash-frozen 50B-token
production job has started, but its step-zero and early finite metrics are not
a completed segment or pretraining-completion claim.

Detailed architecture, job IDs, hashes, failures, and claim boundaries are in
`docs/SCRATCH_7B.md`, `docs/COMPUTE.md`, and `docs/EVIDENCE.md`.
