# Compute and storage plan

All throughput below is tied to a receipt. Extrapolations are labeled and are
not allocation evidence.

## Measured scratch-7B path

The randomly initialized, fully trainable 7,065,227,297-parameter model has
completed bounded 100-step runs on eight RTX 4090 and eight RTX 5090 GPUs. With
a 1,024-token micro-batch budget per rank and four accumulation steps, measured
median throughput was 859.38 and 3,266.16 non-padding tokens/s, respectively.
Both peaked at 15,887.34 MiB allocated per GPU and wrote atomic 84.80GB DCP
checkpoints. The same eight ranks split over two 5090 nodes reached only 254.83
tokens/s, so that topology is rejected for elapsed-time planning.

Checkpoint-free, scale-test-only probes on eight 4090 GPUs measured 1,473.33
average tokens/s at 2,048 tokens/rank with two accumulation steps and 2,468.58
tokens/s at 4,096 tokens/rank with one accumulation step. Eight 5090 GPUs
measured 4,688.58 tokens/s at 4,096 tokens/rank and 5,392.73 tokens/s at 8,192
tokens/rank. Formal pretraining cannot use the checkpoint suppression flag. At
the corrected fixed-window 4,096-token/rank probe, eight RTX 4090 GPUs completed
100 steps and four full window drains at 5,522.08 average tokens/s, 94.342%
aggregate packing efficiency, and 15,804.13 MiB observed peak allocation. That
historical result projected to about 105 days for 50B tokens before checkpoint
overhead and scheduler gaps.

The 5,522.08-token/s receipt predates the global masked-token normalization
fix. A corrected eight-RTX-4090 run with 8,192 tokens/rank completed 100 finite
steps, nine complete fixed-window drains, and 5,469,844 tokens at 9,418.29
average tokens/s. A subsequent corrected run doubled the per-rank budget to
16,384 tokens and increased the fixed window to 1,024 records. Job 1463028
completed 100 finite steps and 10,126,641 tokens at 14,181.35 aggregate and
15,257.60 median tokens/s, 94.777% aggregate packing efficiency, and 19,121.21
MiB peak allocation. The corresponding 1,000-step job 1463046 and recovery job
1463047 then passed the full DCP gate through step 1,100 and 110,480,043 tokens.

A first rolling length-lookahead result is excluded because its refill policy
could starve short records. The fixed-window drain-before-refill replacement is
restart-bound. On the accepted MGnify train corpus, eight-RTX-5090 job 1464107
used 32,768 tokens/rank, at most 256 sequences/rank, and a 4,096-record fixed
window. It completed 50 finite steps and 10,936,363 tokens at 23,677.92 average
and 24,460.82 median tokens/s, 93.967% aggregate packing efficiency, and
26,292.41 MiB peak allocation. The same startup path rehashed every declared
packed-train file before step 1.

At that measured average, 50B tokens project to 24.44 days of uninterrupted
step time before checkpoint, startup, queue, or failure overhead. The measured
218,727 tokens/step imply about 228,595 optimizer steps. The frozen production
config therefore uses a 260,000-step fail-closed ceiling, 2,500-step periodic
checkpoints, and 5,000-step clean segments. One measured segment projects to
12.83 hours and one checkpoint interval to 6.42 hours before checkpoint I/O;
both fit inside the 36-hour job limit with margin. These are planning
extrapolations, not training-completion evidence or a scheduler guarantee.
The initial formal job 1464257 was stopped at step 1,770 after a schedule audit
found that the trainer used the operational segment endpoint as the LR-decay
horizon. The run had executed only warmup steps, but its immutable provenance is
superseded and cannot be resumed. Replacement job 1465137 uses a corrected
global step-260,000 LR horizon that is invariant to the step-5,000 operational
stop. It starts from a new random initialization; its exact-file-set step-zero
audit independently rehashed 84,802,445,816 declared bytes. This is an
operational health check, not a replacement for the completed 50-step capacity
receipt or a full-segment timing result.

To keep training fail-closed after interactive monitoring stops, one 1xRTX-4090
audit job is submitted with `afterok` dependency on each 8xRTX-5090 training
segment. It independently checks the segment receipt, exact global LR at every
logged step, finite/continuous metrics, token/provenance agreement, and every
checkpoint file hash before submitting one successor segment and its next
audit. Any mismatch stops the chain. The audit GPU is a scheduler requirement;
hashing itself is CPU and shared-storage I/O.

True two-node, 16-rank HSDP was enabled only after a route-derived `bond0`
collective probe rejected link-local interfaces. The 4,096-token/rank HSDP job
1464069 completed 100 finite steps at only 3,266.99 tokens/s despite using twice
the GPUs. The paired 8,192-token/rank job 1464070 completed at 5,627.55 average
and 5,946.19 median tokens/s with 94.129% aggregate packing efficiency. Both are
slower in aggregate than the older single-node B8192 gate at 11,879.49 tokens/s
despite twice the GPUs. HSDP is therefore rejected for production elapsed-time
planning; topology-selection receipt SHA-256 is
`121eb05dd354ffc021c8d8c0ffa3c6cd64aeafd8d8ba1f4a5e91f5eb9de73fb4`.

## Measured single-GPU stage-1 path

Job 1460879 ran the 7,076,898,432-parameter model on one RTX 5090 with the
6,352,005,184-parameter ESMC backbone frozen and 724,893,248 bridge/head
parameters trainable. With protein length 256, micro-batch 1, and eight-way
gradient accumulation, post-warmup throughput was:

- p10: 5,605.4 non-padding tokens/s;
- median: 5,621.0 tokens/s;
- p90: 5,624.8 tokens/s;
- peak allocated GPU memory: 24,547 MiB.

The 100-step run processed 204,135 tokens with no non-finite loss or gradient.
Its 75-second Slurm elapsed time includes loading the 25.4 GB backbone and
writing one 4,323,145,735-byte checkpoint. Step-level throughput excludes that
startup/checkpoint overhead.

## Measured four-GPU stage-1 path

Job 1460917 completed 1,000 optimizer steps on four RTX 5090 GPUs with the
official UniRef50 2026_02 shards. It processed 8,182,499 non-padding tokens with
no non-finite loss, gradient, or throughput row. After the first 50 warmup
steps, globally reduced throughput was:

- p10: 15,555.6 tokens/s;
- median: 15,920.5 tokens/s;
- p90: 16,012.5 tokens/s;
- peak allocated memory on the busiest rank: 24,547 MiB.

The median is 70.8% of ideal four-way scaling from the measured single-GPU
median. The receipt's 12,686.9 tokens/s includes ten 4.323 GB checkpoint writes
but excludes model loading and artifact verification. Slurm elapsed time was
695 seconds, giving 11,773.4 tokens/s when startup and final shutdown are also
included.

## Measured-throughput extrapolation

The table is an extrapolation from the completed benchmark, not a completed
pretraining run or scheduler guarantee:

| Stage-1 budget | Steady median | Checkpoint-inclusive receipt | Full Slurm elapsed |
| --- | ---: | ---: | ---: |
| 1 billion tokens | 17.4 hours | 21.9 hours | 23.6 hours |
| 10 billion tokens | 7.3 days | 9.1 days | 9.8 days |
| one 12.171B-residue corpus pass | 8.8 days | 11.1 days | 12.0 days |

Capacity planning should use the checkpoint-inclusive or full-Slurm columns,
not the optimistic steady-step median. Multi-job scheduler gaps are still
unmeasured and would extend calendar time.

The original fixed-record run was configured for 122,250 optimizer steps using
an early-run token-rate estimate. That assumption failed as the stream reached
shorter records. It was cancelled at step 62,380 with 402,649,214 globally
reduced non-padding tokens after the packed replacement passed its recovery and
throughput gate. Even if every remaining fixed-record step had reached the
8,192-token theoretical maximum, the final count could not have exceeded
893,104,254. It therefore cannot establish a 1B-token run or `PRETRAIN` evidence;
its checkpoints and logs remain preserved.

The replacement sequence path uses deterministic dynamic micro-batches bounded
to 256 padded protein tokens per rank and at most 32 records per micro-batch.
It declares `max_tokens=1,000,000,000`; globally reduced non-padding tokens stop
the run, while 130,000 optimizer steps are only a fail-closed safety limit. If
the token target is not reached by that limit, the trainer writes an
`INCOMPLETE` receipt and exits nonzero. A `PRETRAIN` receipt is possible only
after the token target is reached. The production path still runs from a
read-only, content-addressed source snapshot and checkpoints every 10,000 steps.

The trainer has a single-node DDP path with file-level rank sharding,
gradient-accumulation `no_sync`, globally reduced token/loss metrics, and
checkpoint schema v4 containing each rank's data and RNG state plus the original
cross-stage initialization lineage. Dynamic-batch checkpoints additionally
retain a prefetched overflow record so restart does not drop or duplicate it.
Jobs 1461476 and 1461664 completed the packed-path step-100 checkpoint and
step-100-to-step-1000 recovery. The combined metrics are exactly continuous from
step 1 through 1,000 and sum to 8,182,499 tokens, or 99.884% of the theoretical
token capacity. Post-resume p10/median/p90 throughput was
14,875.0/15,248.9/15,345.9 tokens/s with 24,547 MiB peak memory. This passed the
replacement gate. Original production job 1461683 used the exact-token
contract until its storage-only failure. Resume job 1464156 continued from the
independently verified step-120,000 boundary under identical training
provenance and completed at step 137,148 with 1,000,004,951 tokens. Its final
checkpoint and passed `PRETRAIN` receipt were independently rehashed; this
remains frozen-backbone adapter-control evidence only.

The packed 1B validation-only policy was fixed before completion: steps 10,000
through 130,000 are the only selection candidates, all use 1,000 deterministic
validation batches and one immutable evaluator snapshot, and the selector uses
minimum bridge MLM loss with an earliest-step exact-tie break. Its archived,
read-only policy SHA-256 is
`435565a843238b1882ac2d3d5a1c42a50b9e1c9b6b8730200ada52261878ebcc`.
All 13 policy candidates completed validation with exactly 1,000 batches and
38,449 masked tokens. Step 100,000 had the minimum bridge loss, 1.8089416642,
and was selected without opening ProteinGym.

## Staged allocation

1. Complete sequence, sequence-text, and geometry pilots independently and
   verify validation metrics and restart receipts.
2. Measure single-node four-GPU scaling on the exact production mixture.
3. Run a 1B-token shakeout with validation-only checkpoint selection.
4. Continue toward a declared 10B-token or one-corpus-pass stage-1 budget only
   if validation improves and sequence-only controls do not regress.
5. Unfreeze selected ESMC blocks only after a separate FSDP memory/throughput
   benchmark. No time estimate is assigned before that measurement.

The sequence-text continuation uses the packed-policy-selected step-100,000
checkpoint as its initialization and runs 10,000 steps (about 19M multimodal
tokens at the measured rate), checkpointing every 1,000 steps. Its immutable
source snapshot is
`74b0ef631002c5526332e8ca1c61819d9b28724459ff0505aed3c99af0783ca7`.
Initializer validation job 1464573 gates training job 1464574 through an
`afterok` dependency. Training completed at step 10,000 with 19,268,943 tokens;
validation jobs 1464848--1464853 evaluated only the predeclared 1k, 5k, and 10k
milestones. The fail-closed selector chose step 10,000. Protected-geometry jobs
1464882--1464889 form a dependency chain from three initializer validations,
through geometry-only training, to final sequence/text/structure validation and
a checkpoint-delta audit. All eight jobs completed 0:0 and the fail-closed
finalizer passed. A current-source four-rank synthetic opaque-archive smoke then
completed before the final checkpoint/evaluator seal was created. The one
permitted four-RTX-5090 official ProteinGym job, 1464976, then failed 1:0 after
58 seconds: every rank found zero official ESM-2 15B score-archive members for
its first assigned assay. No output receipt was created. Official assay rows had
already been opened, so no corrected rerun is allowed under the frozen one-time
policy.

## Storage

- ESMC local snapshot: 25.41 GB content-addressed;
- UniRef50 raw FASTA: 8.77 GB;
- UniRef50 processed CSV shards: 16.72 GB;
- Swiss-Prot flat file: 0.70 GB, plus its annotation index;
- AlphaFold DB Swiss-Prot v6 PDB tar: 28.58 GB verified;
- stage-1 adapter checkpoint: 4.323 GB each; ten retained checkpoints need about
  43 GB;
- scratch 7B full-shard checkpoint: about 84.80 GB at world size eight;
- MGnify 2024_04 official gzip: 79.30 GB before deterministic packed output.

The earlier 200 GB estimate applies only to the adapter track. Scratch 7B uses
the account-scoped `/ssd/scxi253` NVMe quota for active files and
`/data/run01/scxi253/pretraining2-archive` for inactive, hash-verified history.
The rejected 75.10 GB MGnify mirror; three completed historical 7B runs;
historical and corrected 1B scratch gates; and the completed packed adapter
recovery gate were moved to that archive and replaced by exact-path symlinks
only after every receipt/manifest-bound byte was reverified at the destination.
The earlier archival batches independently rehashed 169,604,897,810 and
103,949,124,208 checkpoint-content bytes before releasing their SSD duplicates.
A final capacity pass also rehashed all ten state files and every non-state file
in each of two completed adapter pilots, covering 43,232,302,758 and
43,231,750,713 file bytes, before replacing their SSD paths with read-only HDD
symlinks. Active training directories were not moved.
A retention policy must preserve milestone/recovery checkpoints and may remove
only explicitly superseded intermediates after their hashes and receipts are
archived.

During corrected formal job 1465137, account-scoped SSD availability fell from
about 385 GB to 269 GB while the active run itself had not created a new
checkpoint. To restore checkpoint headroom without touching the active chain or
unrelated runs, the permanently superseded job-1464257 run was copied to the
separate HDD archive. The archive contains 20 run files totaling
84,802,517,076 bytes plus both job logs; all 17 checkpoint-manifest files and
84,802,445,816 declared checkpoint bytes passed destination size/SHA-256
verification before the HDD tree was write-locked and the exact obsolete SSD
directory was removed. Archive and SSD-relocation receipt SHA-256 values are
`980dddc5794b4988ca3ef715442acb71667ab593dc248d48c9d302304edd76eb` and
`6f3ac31c59e9800d66b389c196291aef1cbdbab990934aadb6c7785ee2a2ca35`.
The observed account-scoped SSD availability was about 346 GB at the 21:15
Asia/Shanghai post-migration snapshot. The full 1 GB login-home quota is kept
out of the training data, checkpoint, log, and receipt paths.

At 2026-08-25 01:31 Asia/Shanghai, after the active step-2,500 checkpoint and
other account-scoped writes, SSD availability was about 202 GB. The active
checkpoint itself was independently verified as 17 manifest files and
84,816,914,936 declared bytes; its receipt SHA-256 is
`1daf2a01c1432748e55234ccbff8b83ca61da321dc97bf7ccab321e5599ee8c1`.
The remaining space still exceeds one full checkpoint write, but storage stays
an explicit monitored risk until retention removes the intermediate checkpoint
or the account-scoped quota gains headroom.

The active formal run now uses an exact one-checkpoint sliding window. After
the step-2,500 checkpoint passed full size/SHA-256 verification, the raw
step-zero directory was removed; its integrity and random-initialization
provenance remain recorded by the earlier immutable receipt, but the raw files
are intentionally not retained. Because the immutable trainer already prunes
all older nonzero checkpoints with `checkpoint_keep_last=1`, removing step zero
once means each future successful save leaves only the newest checkpoint.
Retention intent/applied receipt SHA-256 values are
`e527b59c901ad1be0a751d434e60e0e05e5201facda5624cb68cc1499baf3b72` and
`cdb9ba88d89e020cf1db3c939ab953f271248452a8b93bad0d1ee4c7dd4a4629`.
No active source, config, job, or dependency was changed.
After asynchronous quota accounting settled, observed SSD availability was
about 298 GB at 2026-08-25 02:00 Asia/Shanghai, with `step-00002500` as the
only checkpoint directory.

Account-scoped SSD headroom later fell again to about 152 GB even though the
formal run still had only one checkpoint. Before the next save, the verified
checkpoint root was copied to
`/data/run01/scxi253/pretraining2-active/scratch-7b-mgnify90-2024_04-formal-50b-global-lr-v2/checkpoints`.
All 17 manifest files and 84,816,914,936 declared bytes passed destination
SHA-256 verification. The logical `/ssd/.../checkpoints` path is now a writable
symlink to that HDD root, so the immutable trainer and audit chain retain their
exact paths while future saves/pruning use the separate quota. The redundant
SSD copy was released only after a path write probe and manifest recheck passed.
Copy/switch receipt SHA-256 values are
`57cb2998827b40c935613dd916ab2f54649735bb7757a5b71df0be949b74e8d8` and
`5278f5f79222f208b6f7c06e232fbd51dbed6c0286f81b6288a9165987ab44a9`.
At the 04:25 snapshot, the HDD quota had about 942 GB available and the SSD
quota about 231 GB; checkpoint capacity is no longer coupled to the latter.
