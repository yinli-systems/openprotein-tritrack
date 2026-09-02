# OpenProtein scratch 7B contract

## Claim boundary

`openprotein-scratch-7b` is a new randomly initialized, fully trainable protein
masked-language model. It does not load ESMC, ESM-2, an adapter, or any other
model checkpoint. Resuming one of its own hash-bound distributed checkpoints is
continuation of the same scratch run, not pretrained initialization.

The older `openprotein-7b` configuration is a separate 7.08B-parameter composed
system with a frozen ESMC 6B backbone and a 724.9M-parameter trainable adapter.
Its results are controls only and are never evidence that this scratch model was
pretrained.

## Frozen architecture

| Field | Contract |
| --- | ---: |
| Objective | bidirectional 15% residue MLM, 80/10/10 corruption |
| Layers | 35 |
| Hidden width | 4,096 |
| Attention heads | 32 (128 dimensions per head) |
| Feed-forward width | 16,384 |
| Positions | rotary |
| Normalization / activation | pre-LayerNorm / GELU |
| Dropout | 0 |
| Context | 1,024 tokens including CLS and EOS |
| Vocabulary | ESM-2-compatible 33 tokens |
| Input/output embeddings | tied |
| Initialization | random normal, residual outputs scaled by `1/sqrt(2L)` |
| Exact parameters | 7,065,227,297, all trainable |

The implementation uses PyTorch scaled-dot-product attention with a broadcast
key-padding mask; it does not allocate a quadratic boolean attention mask.
The architecture follows the public ESM-2 encoder shape: BERT-style masked
encoding, RoPE, GELU feed-forward layers, and tied MLM output. The exact 7B
shape is project-specific rather than a renamed ESM checkpoint.

## Optimizer and parallelism contract

- AdamW: beta1 0.9, beta2 0.98, epsilon 1e-8, weight decay 0.1; bias and
  normalization vectors are excluded from decay.
- MLM loss and gradients are normalized by the exact global count of masked
  targets across every rank and accumulation micro-batch, rather than averaging
  unequal per-rank means.
- BF16 forward/backward with FP32 sharded parameters, gradient reduction, and
  Adam moments.
- Full activation checkpointing and PyTorch FSDP2 per encoder layer.
- One node: 8-way full sharding. Multiple nodes: 8-way intra-node sharding and
  inter-node replication (2D HSDP), avoiding layer all-gathers over the slower
  inter-node link.
- Distributed Checkpoint stores model and optimizer DTensors plus one exact
  data/RNG cursor per rank. Every file is byte-counted and SHA-256 bound before
  an atomic checkpoint publish.
- `checkpoint_keep_last` may retain step zero plus only the newest N restart
  checkpoints. Pruning happens only after the replacement checkpoint is fully
  published, and all ranks wait for pruning to finish before training resumes.
- A formal run may stop cleanly only at a predeclared periodic checkpoint using
  `segment_end_step`. Such a receipt is labelled
  `SCRATCH_PRETRAIN_SEGMENT`, records `target_reached=false`, and is not a
  completion claim. `SCRATCH_PRETRAIN` remains exclusive to the 50B-token
  target. Resume fails closed if active metrics extend beyond the selected
  checkpoint, requiring archival and an atomic rewind first.
- A 7B run fails closed below eight CUDA ranks. Eight RTX 5090 GPUs are only the
  minimum fit target; 16-32 GPUs are the elapsed-time target.

With FP32 parameter, gradient, and Adam states, the ideal sharded state floor is
about 13.16 GiB/GPU at eight-way sharding before activations, communication
buffers, allocator overhead, and temporary layer all-gathers. Four 32 GiB GPUs
would start near 26.32 GiB/GPU before those costs and are therefore rejected as
the primary 7B plan.

## Data contract

The existing verified UniRef50 2026_02 corpus has 12,171,016,896 residues. It is
valid for pipeline and scaling tests, but only 1.72 residues per model parameter
and is not accepted as a complete 7B pretraining corpus.

The first formal gate is at least 50B hash-bound, diverse train tokens without
blind repetition. The executable source is a deterministic 50% selection from
the MGnify Proteins 2024_04 MGnify90 representatives. EBI reports 717,738,164
cluster representatives containing 121,126,699,551 residues for that release;
the expected 98% training share of the selected corpus is about 59.35B residues.
The builder fails closed unless the complete source reproduces the official
record and residue totals and the selected training split exceeds 50B residues.

The first transport candidate was the sharded `LiteFold/Mgnify` mirror, pinned
to commit `42712c93c6995a75f6f1d89931f95ddd78567b30`. All 717 compressed
objects and their Hugging Face LFS SHA-256 values were verified. A complete
streaming audit nevertheless found only 716,738,164 records and
120,943,014,900 residues: exactly 1,000,000 records and 183,684,651 residues
short of the authoritative EBI release. The pinned mirror is therefore rejected
as a complete formal source. It remains useful only as a hash-bound transport
candidate if an independently verified official prefix exactly closes both
gaps and its accession boundary is continuous. No packed formal corpus has been
accepted from it. The mirror's distribution card says CC BY 4.0 while the
authoritative EBI release is CC0, so both notices are retained.

An official-gzip prefix audit confirms the diagnosis: its first 1,000,000
records contain exactly 183,684,651 residues, and official record 1,000,001 is
`MGYP004618177781`, the mirror's first record. Its length (134) and sequence
SHA-256 (`b1c417bad44d6ffb895f846497e5c3dc3e9bc6e6d89b557997da6ba73d662813`)
also match. The formal fallback nevertheless downloads the complete EBI gzip
and requires its official byte count, MD5, gzip integrity, and a full semantic
pass; the boundary observation alone is not a formal-corpus receipt.

The first resumed complete-size candidate was also rejected: although it had
exactly 79,301,981,114 bytes, its independently measured MD5 was
`c2bbba0a9737dce7ac5df375651a98b5` rather than EBI's
`c7e812f9542d48cf425c5329b8571e68`. Those bytes are retained read-only in the
HDD quarantine. The active recovery uses full-checksum Aspera comparison and
successfully replaced the corrupt ranges. The accepted source has exactly
79,301,981,114 bytes, MD5 `c7e812f9542d48cf425c5329b8571e68`, SHA-256
`e1597a12a2f10b0574c172cb820576916d86ebd24e680801506834a2191cdb08`,
and a passed `gzip -t`; retrieval receipt SHA-256 is
`40d7ee0e02ee0f20330c982acf20fb67eef7e518c179be6ded1dfc07b267f530`.
The complete semantic pass reproduced 717,738,164 records and
121,126,699,551 residues. The deterministic 50% accession-hash selection then
produced 358,895,949 records and 60,566,268,552 residues with zero invalid or
unknown residues. Its 98% train split contains 351,716,200 records and
59,354,481,535 residues across 128 packed shards; train-manifest SHA-256 is
`7640aa5a28defb5a371910dcd1e43720597a27c4c5fbc283a0bbac0721530db8`.

MGnify 2026_07 remains a separately defined newer source with 1,661,142,385
representatives and 250,711,458,083 residues. It is not used by the 2024_04
mirror path and must never be named in that path's receipts.

Before formal training, the mixture must have receipts for:

1. official release statistics and license, every accepted transport object's
   size and SHA-256, and an exact full-stream match to the official totals;
2. accession-hash selection and disjoint train/validation/test separation;
3. tokenizer counts, invalid/unknown residue counts, lengths, records, and
   residues by source and split;
4. packed uint8 token/index shard hashes;
5. an untouched frozen-test policy.

The packed format stores one uint8 per residue and fixed-width offset/length
indices. Its training cursor is directly restartable and does not replay the
corpus to recover a checkpoint position.

## Scale ladder and launch gates

| Gate | Exact parameters | Minimum evidence |
| --- | ---: | --- |
| 125M | 127,049,761 | 1,000 finite steps, throughput, loss trend, checkpoint/resume |
| 1B | 1,011,437,601 | same contract on four GPUs, plus scaling efficiency |
| 7B | 7,065,227,297 | eight-GPU 1,000-step run, peak memory margin, DCP restart equivalence |
| Formal 7B | 7,065,227,297 | at least 50B tokens, immutable source/config, periodic recovery drills |

The 7B production job may start only after all three scale gates pass and the
measured throughput, storage use, queue availability, and projected cost are
recorded. The rough dense-training estimate is `6 * parameters * tokens`: 50B
tokens is about 2.12e21 FLOP. This formula is a planning estimate, not a measured
RTX 5090 benchmark.

## Evaluation and evidence levels

- `UNIT`: analytical/instantiated counts and CPU tiny-model tests only.
- `SCALE_TEST`: real GPU memory, throughput, loss, and recovery mechanics; not
  pretraining completion and not a quality claim.
- `SCRATCH_PRETRAIN`: emitted only by the random-init trainer after the declared
  token budget and checkpoint contract complete on hash-bound train data.
- `VALIDATION`: fixed validation-only checkpoint selection.
- `FROZEN_TEST`: one final evaluation after checkpoint and evaluator sealing;
  never used for model selection.

Current status: the historical 125M and 1B 1,000-step scale runs and their
recovery drills passed throughput/checkpoint mechanics under the former
per-rank-mean loss. Corrected global-masked-token jobs 1462979/1462986 have now
passed the 125M 1,000-step and recovery gates. Corrected 1B jobs
1462978/1462987 also completed 1,000 steps and a step-1,000-to-1,100 recovery,
reaching 18,653,687 cumulative tokens. All 110 logged rows are finite and
continuous; recovery-segment median throughput is 3,906.30 tokens/s and final
loss is 2.58577. The final nine-file, 12,143,440,244-byte checkpoint was
independently verified, with manifest SHA-256
`667ebefaf179ebed24591c2e3d8a44c771b48ffa8e1a419a66079727418a54b6`;
initial/recovery receipt SHA-256 values are
`621579c3e40ce8e71e62459665b01176bcb419cbc9df92c657506736db88a575`
and `5699f1e7970701c6d361118d95ad533a0159f1fa8c85f5e3a3c4ec00f9973359`.
These are scale-test and recovery evidence, not pretraining completion. Jobs
1462004/1462431 completed a bounded two-node, eight-rank 7B
engineering run through step 100 and a same-topology restart. At step 100 it had
processed 1,733,529 tokens; loss moved from 4.39990 at step 10 to 2.70312, median
throughput was 254.83 tokens/s, and peak allocated memory was 15.89 GiB per GPU.
Its 17-file, 84,802,452,521-byte DCP manifest SHA-256 is
`1aeef3124f102ab2156953ee8202d93397485bd6c979ad409b47314e3cfdf6a3`;
the passed recovery receipt SHA-256 is
`0dc7a3da617c4cefb0e5610627f5febf984f8b0442558802d9395eca39a41cda`.
This cross-node full-shard result is communication-bound and is neither a
single-node throughput claim nor the required 1,000-step 7B gate; its loss
trajectory also predates global masked-token normalization.

Jobs 1462516 and 1462061 independently completed 100 finite steps on one node
with eight RTX 4090 and eight RTX 5090 GPUs, respectively. Both processed
1,733,529 tokens and peaked at 15,887.34 MiB allocated per GPU. The 4090 node's
median was 859.38 tokens/s and 20.24 seconds/step; the 5090 node's median was
3,266.16 tokens/s and 5.47 seconds/step. Their receipt SHA-256 values are
`b9b0e51cb693e339008639aef44935957c612d331b8b68d3b716d650a6870bfd`
and `e97e746f2d9392d08f3d09253bf2e8d2682d55454e6eba6e3f6568efa2417a35`.
Checkpoint-free 20-step throughput probes then increased the 4090 result to
1,473.33 tokens/s at a 2,048-token micro-batch with two accumulation steps and
2,468.58 tokens/s at a 4,096-token micro-batch with one accumulation step. They
are throughput/memory observations only; formal pretraining is forbidden from
disabling the step-zero or final checkpoint.

Equivalent 5090 probes reached 4,688.58 tokens/s at a 4,096-token micro-batch
and 5,392.73 tokens/s at an 8,192-token micro-batch, with 15,792.04 and
16,560.68 MiB peak allocated memory. A first 256-record lookahead experiment is
rejected because replenishing the pool after every batch favored newly arriving
long records and could starve shorter records. The corrected implementation
sorts a fixed window and consumes it completely before reading another window;
its tests bind that window into restart state and prohibit source-cursor advance
until the window is drained. Job 1462921 then completed 100 finite steps with
the corrected policy on eight RTX 4090 GPUs. It processed 2,890,097 non-padding
tokens at 5,522.08 average and 5,692.99 median tokens/s, reached 94.342%
aggregate packing efficiency, and observed 15,804.13 MiB peak allocated memory
across the step metrics. Partial tail batches at steps 21, 42, 63, and 84 prove
that four complete fixed windows were drained before refill. Its receipt
SHA-256 is `d3e222069e9652c874fd25c777824454c1bf2feffa6b7f87ef73ad07d74a36ea`.
This receipt remains valid as throughput/packing evidence, but a subsequent
trainer audit found that its optimization loss averaged unequal per-rank means.
Its loss trajectory is therefore not promoted to the formal optimization gate.
The corrected global masked-token weighting has a gradient-equivalence
regression test. Job 1462981 then completed a corrected 100-step run on eight
RTX 4090 GPUs with an 8,192-token budget per rank. It processed 5,469,844 tokens
at 9,418.29 average and 9,872.46 median tokens/s, with 89.070% aggregate packing
efficiency and 16,561.68 MiB maximum observed allocation. Partial window tails
occurred every 11 steps through step 99, proving nine complete fixed-window
drains. Its metrics and receipt SHA-256 values are
`574e1723f48bd9d36cb253328110e71ba5af5cb77e78584a7dcb469cbd8d4704`
and `528aa8c9bfe8948f1011a6e5cf345a02dfcbf7df689b9ab4dcb64edc7892516a`;
the receipt explicitly declares `loss_normalization=global_masked_tokens`.
Job 1463028 then doubled the per-rank budget to 16,384 tokens and enlarged the
fixed window to 1,024 records. It completed 100 finite steps and 10,126,641
tokens at 14,181.35 aggregate and 15,257.60 median tokens/s, 94.777% aggregate
packing efficiency, and 19,121.21 MiB peak allocation. Its metrics/receipt
SHA-256 values are
`467fd8aff48c2b416033c4b15e42d018394960a46e952996030971b182133e59`
and `7d0f6b13df120407359e8c511e5e63300c67d69405e7660fb5d18f1ec4e39d72`.
The separate 1,000-step job 1463046 completed 100,559,889 tokens and published
its full step-1,000 DCP. Predeclared recovery job 1463047 loaded that checkpoint
and continued to step 1,100 and 110,480,043 tokens. Its retained step-zero and
step-1,100 artifacts contain 40 files and 169,605,443,825 bytes; every file was
independently rehashed during read-only HDD archival. Initial/recovery receipt
SHA-256 values are
`9fcae3b344e8aa14f7f7f56073a6646329c03d10155823a48d8269760a1fb07e`
and `e4f10e3dd8121101c99840db982fee5f4e99e7133b243abd7e1ef5d2b4740fa3`.

The original multi-node verifier and a later link-local NCCL route both failed
before valid HSDP evidence. A route-derived `bond0` probe then passed, after
which true two-node x eight-GPU jobs 1464069/1464070 reported
`FSDP2_HYBRID_SHARD` and completed 100 finite steps. Their 4,096/8,192-token
per-rank configurations averaged only 3,266.99/5,627.55 tokens/s. Both are
slower than the older single-node B8192 result despite consuming twice the
GPUs, so HSDP is rejected. The selection receipt SHA-256 is
`121eb05dd354ffc021c8d8c0ffa3c6cd64aeafd8d8ba1f4a5e91f5eb9de73fb4`.

On the accepted MGnify train corpus, selected job 1464107 completed 50 finite
steps and 10,936,363 tokens at 23,677.92 average and 24,460.82 median tokens/s,
93.967% aggregate packing efficiency, and 26,292.41 MiB peak allocation. The
corrected formal 50B-token config is frozen in source snapshot
`efb2df4391504bed59822a36972a2644d4c03e04800bbad4483a3549fa84aa91`;
it uses 2,500-step checkpoints, 5,000-step clean operational segments, and one
global step-260,000 LR horizon that does not restart or shorten at a segment
boundary. Job 1464257 is superseded because its older immutable trainer
conflated the segment and schedule endpoints; it was stopped at step 1,770
before any affected post-warmup step. Replacement job 1465137 started from a
new random initialization. Its 18-file step-zero checkpoint contains
84,802,445,816 manifest-declared bytes and has manifest SHA-256
`cabecc75bbc7e9a3e7aa6778ad11effb08a43e7279531464c64177e7271379a2`;
the exact file set, every declared size/SHA-256, random initialization, and new
provenance passed an independent audit. The job subsequently completed the
first operational segment at step 5,000 and 1,064,746,261 tokens. Repair audit
job 1467456 completed 0:0 and rehashed the exact 17-file, 84,814,663,928-byte
step-5,000 checkpoint, verified all 500 metrics rows and the global LR formula,
then submitted exactly one step-10,000 successor (job 1467474) and its
dependency audit (job 1467475). The successor is pending for an eight-GPU
single-node placement; this is `SCRATCH_PRETRAIN_SEGMENT` evidence only, not
50B-token completion or model-quality evidence.

## Primary references

- [Meta ESM repository and model table](https://github.com/facebookresearch/esm)
- [ESM-2 architecture implementation](https://github.com/facebookresearch/esm/blob/main/esm/model/esm2.py)
- [Protein language-model scaling study](https://arxiv.org/abs/2411.02142)
- [PyTorch FSDP2 documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [TorchTitan FSDP2 design and meta initialization](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)
- [MGnify Proteins 2024_04 README](https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2024_04/README.txt)
- [MGnify Proteins 2024_04 checksums](https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2024_04/md5sum.txt)
- [Pinned transport repository](https://huggingface.co/datasets/LiteFold/Mgnify)
