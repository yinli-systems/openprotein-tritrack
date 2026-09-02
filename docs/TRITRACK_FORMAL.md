# TriTrack 1.2B formal training contract

Status: predeclared 2026-09-03 before the first formal 3B-token run.

## Objective

TriTrack tests whether a larger contextual text path, multi-positive alignment,
residue-conditioned structure fusion, masked-structure embeddings, and
structure-modality dropout improve the completed OpenProtein 1B model without
regressing its frozen ESM-2 sequence path.

The loaded model contains 1,168,564,995 parameters, of which 517,521,741 are
trainable. The ESM-2 650M backbone remains frozen. This is therefore
warm-started multimodal pretraining, not scratch pretraining.

## Data boundary

All 575,503 Swiss-Prot canonical sequences are clustered with MMseqs2 release
18 commit `8cc5ce367b5638c4306c2d7cfc652dd099a4643f` at 30% identity and 80%
bidirectional coverage. Connected-component, single-step clustering assigns an
entire detected homology component to one deterministic 98/1/1 split. Training
rows require an exact accession and full-sequence match to AlphaFold DB.

The 32,768-piece SentencePiece tokenizer is trained only after this split and
only from training annotations. Validation and test annotations are excluded.
Broad UniRef50 sequence-only replay may contain homologous amino-acid sequences;
it exposes no held-out text or structure target. Consequently, the new split is
multimodal-label-disjoint, not proof that the warm-started sequence backbone has
never encountered a related protein.

## Efficiency gate

One six-RTX-4090 allocation probes per-rank microbatches 12, 16, and 20 for
exactly 20 steps at sequence length 512. The first five startup steps are
discarded. The selected configuration maximizes median aggregate token
throughput subject to all of the following:

- successful exit and exact steps 1 through 20;
- finite loss, gradient norm, throughput, and peak memory on every step;
- at least 15% physical device-memory headroom;
- identical model, data, objectives, replay cadence, and optimizer settings
  between the probe and its bound formal configuration.

The selector emits a no-overwrite, content-addressed receipt. The formal job
refuses to run when the receipt and immutable source snapshot differ.

## Formal run

- Hardware: one node, six RTX 4090 GPUs.
- Token target: 3,000,000,000 aggregate sequence plus text tokens.
- Schedule: 100,000,000-token warmup followed by token-based cosine decay.
- Objectives: protein MLM 1.0, text LM 0.5, multi-positive contrastive 0.2,
  sampled geometry 0.25, and masked structure denoising 0.25.
- Sequence replay: one of every three batches, weight 1.0.
- Checkpoints: every 2,500 optimizer steps plus the exact token boundary.

Reaching the token target establishes only `PRETRAIN` completion. It does not
establish model quality.

## Validation and publication gates

The new test split stays sealed during training and checkpoint selection. The
first post-training job runs validation only and must report:

1. sequence MLM loss against the exact frozen ESM-2 path;
2. paired versus shuffled text conditioning with uncertainty;
3. bidirectional retrieval Recall@1/5/10 on the full validation split;
4. predicted-geometry loss and accuracy, including long-range pairs and fitted
   marginal/separation priors;
5. exact hashes for the source, model, tokenizer, data, checkpoint, and results.

A publication-quality claim additionally requires same-protocol ESM-2,
ProTrek-650M, SaProt-650M, DPLM-2-650M, and ESM3-open baselines; a fresh
ProteinGym evaluation; GO/DeepLoc; PDB-date or CAMEO structure metrics; and
component ablations. Seeds on one split are not counted as independent
datasets, and the final report must disclose any pretraining-overlap uncertainty.
