# OpenProtein

OpenProtein is an evidence-gated, public-data protein foundation-model project.
It contains a completed warm-started 1B multimodal track, a higher-capacity
TriTrack architecture under validation, and two strictly separate 7B tracks:

1. `openprotein-scratch-7b` is a new 7,065,227,297-parameter protein masked
language model. It is randomly initialized, loads no pretrained model, and
updates every parameter.
2. The older `openprotein-7b` configuration combines a frozen ESMC-6B backbone
   with a 724.9M-parameter multimodal adapter. It is a continued-training
   control and is never evidence for scratch pretraining.

This repository is deliberately evidence-gated. A model configuration, a
successful unit test, a short GPU smoke run, a pretraining run, and a frozen
benchmark result are different evidence levels. Until the corresponding receipt
exists, the README does **not** claim that OpenProtein beats ESMC, ESM-2, or any
other baseline.

## Current snapshot

- The completed 1B multimodal model processed 1,000,045,102 tokens on four RTX
  4090 GPUs and has an optimizer-free, reconstruction-verified export. It beats
  ProTrek-650M on the paired frozen-encoder EC comparison, while ProTrek remains
  clearly stronger on same-pool sequence-text retrieval. See the exact metrics
  and limitations in [1B multimodal results](docs/MULTIMODAL_1B_RESULTS.md).
- The 1.2B TriTrack successor has passed only a 10-step four-GPU
  checkpoint/resume smoke. It has no quality claim yet.
- The random-initialized 7B track reached step 5,000 and 1,064,746,261 tokens;
  it has not completed its pretraining target or established model quality.

## Scratch 7B

The scratch architecture, optimizer, data, scale-ladder, and evidence contract
are specified in [Scratch 7B contract](docs/SCRATCH_7B.md) and the
[scratch model card](SCRATCH_MODEL_CARD.md). Historical 125M and 1B runs passed
1,000-step throughput/checkpoint tests, but their optimization semantics were
superseded when an audit replaced unequal per-rank MLM means with exact global
masked-token normalization. The corrected 125M 1,000-step/recovery gate has
passed; the corrected 1B 1,000-step/recovery gate has also passed, including an
independently hash-verified step-1,100 checkpoint. The corrected single-node 7B
checkpoint gate has now completed 1,000 steps and 100,559,889 tokens, then
loaded its independently verified step-1,000 DCP and continued exactly to step
1,100 and 110,480,043 tokens. This remains a `SCALE_TEST`, not pretraining
completion.

The complete official MGnify Proteins 2024_04 gzip now matches upstream bytes
and MD5, passes gzip integrity, and has a SHA-256 receipt. Its deterministic 50%
selection contains 60,566,268,552 residues, including 59,354,481,535 train
residues in 128 hash-bound packed shards. Two eight-RTX-5090 production-corpus
probes reverified every train-manifest file before training. The selected
32,768-token/rank, 256-sequence configuration completed 50 finite steps and
10,936,363 tokens at 23,678 average and 24,461 median tokens/s, with 26.29 GiB
peak allocation. Formal job 1464257 is now superseded: an audit found that its
immutable trainer would have used the step-5,000 operational boundary as the
LR-decay horizon. It was stopped at step 1,770, before the first affected
post-warmup step, and cannot be resumed. Replacement job 1465137 starts from a
new random initialization under read-only source snapshot
`efb2df4391504bed59822a36972a2644d4c03e04800bbad4483a3549fa84aa91`.
Its operational segments no longer alter the global step-260,000 LR horizon.
All 84,802,445,816 manifest-declared step-zero bytes independently passed
SHA-256 verification, and its first metrics are finite with the exact declared
warmup LR. The run subsequently crossed the first post-warmup boundary: all 202
logged rows through step 2,020 and 430,598,467 tokens remained finite and
continuous, while steps 2,010/2,020 exactly followed the global step-260,000 LR
decay rather than the rejected step-5,000 segment schedule. The immutable
post-warmup health-receipt SHA-256 is
`8cf00abfb3529c88b798d49e8a06bf022e74dfc6b5cdb77f12ebfd449e4c7e42`.
After the first periodic checkpoint at step 2,500 passed an independent full
84,816,914,936-byte verification, active retention was tightened to a true
latest-one sliding window. The raw step-zero and step-2,500 files were
subsequently pruned after the step-5,000 checkpoint was committed; their
immutable integrity/provenance receipts remain preserved.
When account-scoped SSD headroom later fell to 152 GB, the fully verified active
checkpoint root was moved behind the unchanged `/ssd/.../checkpoints` logical
path to a separate writable HDD quota. Every manifest-declared byte was
reverified before the path-preserving switch, and the training/audit provenance
did not change. Job 1465137 then completed its first segment at step 5,000 and
1,064,746,261 tokens. Repair audit job 1467456 independently verified all 17
checkpoint files and 84,814,663,928 declared bytes, the exact file set, all 500
logged metric rows, the global LR schedule, and full provenance. Successor job
1467474 was canceled before execution, so the current formal boundary remains
step 5,000 and 1,064,746,261 tokens. This is a completed
`SCRATCH_PRETRAIN_SEGMENT`, not the 50B-token `SCRATCH_PRETRAIN` completion or a
quality result.

## Frozen-backbone control

The original proposal treated ESM-2 15B as the frontier. The separate control
track preserves the ESMC sequence path and adds roughly 0.72B parameters for
multimodal fusion. That historical design choice does not constrain or describe
the scratch model.

The bridge supports four objectives:

1. protein masked-token prediction, including text-conditioned infilling;
2. prefix-language-model annotation generation;
3. sequence-text contrastive alignment;
4. sampled residue-pair distance classification without allocating a full
   quadratic pair tensor.

The packed frozen-backbone control has now completed its exact
1,000,000,000-token contract: resume job 1464156 stopped at step 137,148 and
1,000,004,951 globally reduced tokens with a passed `PRETRAIN` receipt. A
predeclared validation-only sweep over all 13 checkpoints from step 10,000 to
130,000 selected step 100,000 by minimum bridge MLM loss (1.80894). This is
adapter-control evidence only. A new policy-bound 1k/5k/10k sequence-text gate
completed from that selected checkpoint and selected step 10,000 by minimum
validation text loss after every candidate passed the conditioning and sequence
regression gates. Its protected predicted-geometry continuation is now running
under a separately frozen policy and passed its fail-closed finalizer. The final
checkpoint then entered the one permitted sealed ProteinGym evaluation. That
single attempt failed closed because the frozen evaluator could not resolve the
declared ESM-2 15B score-archive member contract. It produced no result. Because
official assay rows were opened, the one-time boundary is exhausted: no corrected
rerun or frozen score is claimed.

See [Architecture](docs/ARCHITECTURE.md), [Data contract](docs/DATA.md),
[Compute plan](docs/COMPUTE.md), [Evaluation contract](docs/EVALUATION.md), and
[Evidence ledger](docs/EVIDENCE.md). Standard release-facing summaries are in
the [adapter model card](MODEL_CARD.md) and [data card](DATA_CARD.md). The optimizer-free
publication gate is in [Release contract](docs/RELEASE.md).

## Quick local check

```bash
python -m pytest
python -m openprotein.cli.inspect configs/model/openprotein-7b.json
python - <<'PY'
from openprotein.scratch.config import load_scratch_model_config
config = load_scratch_model_config("configs/scratch/model/openprotein-scratch-7b.json")
print(config.parameter_count)
PY
```

Frozen ProteinGym evaluation is a two-command, no-overwrite workflow:

```bash
openprotein-seal-evaluation CHECKPOINT BENCHMARK_MANIFEST MODEL_CONFIG \
  TOKENIZER_JSON MODEL_ARTIFACT_MANIFEST EVALUATION_SEAL
openprotein-evaluate-proteingym EVALUATION_SEAL FROZEN_RECEIPT
```

The first command must complete before any assay or official baseline score row
is opened. The second verifies every sealed hash again before model loading.

The frozen-control backbone is an external, immutable dependency and is not
stored in Git. The scratch model has no external model-weight dependency.

## Current claim boundary

The repository currently contains implementations and scale evidence under
active validation. No completed 50B-token scratch-pretraining run or successful
frozen-benchmark receipt is claimed here. The frozen-ESMC adapter track is a
separate control and cannot satisfy the scratch-model claim.
