# OpenProtein 1B multimodal results

Status: completed pretraining, exported model, and evaluation snapshot as of
2026-09-02. This page separates measured results from planned work and does not
promote smoke tests into quality claims.

## Completed model and training

The completed model warm-starts from a frozen ESM-2 650M sequence backbone and
trains a 360,203,340-parameter multimodal bridge. The loaded model contains
1,011,246,594 parameters in total. It processed 1,000,045,102 tokens on four
RTX 4090 GPUs in approximately 8.63 hours. Measured peak device memory was
17,087 MiB.

This is a warm-started multimodal model, not a random-initialized 1B foundation
model. The exported, optimizer-free release reconstructed successfully with the
same parameter counts; large weights and checkpoints are deliberately excluded
from Git.

## Frozen-encoder EC comparison

On the paired-seed TorchProtein EC frozen-probe comparison, the model won all
10 aggregate metrics and all 30 paired seed comparisons against ProTrek-650M.
Selected metrics are:

| Split and metric | OpenProtein 1B | ProTrek-650M | Delta |
| --- | ---: | ---: | ---: |
| EC identity <95%, micro-AUPRC | 0.861025 | 0.847326 | +0.013699 |
| EC identity <95%, F1-max | 0.834037 | 0.819029 | +0.015008 |
| EC identity <30%, micro-AUPRC | 0.700283 | 0.693474 | +0.006809 |

The final paired-comparison audit SHA-256 is
`cdaa3b4b0a6fe4c136fb97d6558570541946d4f6c208a0a7b5da221cfe689116`.
These are consistent gains under one frozen-encoder protocol, generally 0.5 to
1.5 percentage points; they are not a universal model-ranking claim.

## Multimodal validation

| Capability | Result | Interpretation |
| --- | ---: | --- |
| Sequence bridge loss | 1.953530 | Slightly worse than frozen ESM-2 at 1.950045; delta +0.003486 |
| Structure geometry accuracy | 0.360355 | Above uniform 0.015625 |
| Structure geometry loss | 2.688258 | Below uniform 4.158883 |
| Long-range geometry accuracy | 0.427666 | Positive predicted-structure signal |
| Text perplexity | 5.104 | Validation-only checkpoint evidence |
| Paired minus shuffled text loss | -0.263889 | Protein conditioning improves text loss |
| Protein to text R@1 / R@10 | 0.3033 / 0.6560 | Full 1,942-pair validation pool |
| Text to protein R@1 / R@10 | 0.3409 / 0.7044 | Full 1,942-pair validation pool |

On the same retrieval pool, ProTrek-650M remained materially stronger:
protein-to-text R@1/R@10 was 0.6133/0.8548 and text-to-protein R@1/R@10 was
0.5407/0.8064. OpenProtein therefore has credible EC and geometry evidence but
does not lead the retrieval comparison.

## Claim boundaries

- ProteinGym rows were not opened by this 1B run.
- Public-test access occurred during earlier comparison iterations, so the
  latest comparison cohort is not a pristine first-look holdout.
- Pretraining overlap with downstream proteins cannot currently be excluded.
- AlphaFold-derived geometry is predicted-structure validation, not
  experimental folding evidence.
- No medical, therapeutic, biosafety, or wet-lab efficacy claim is supported.

## Next architecture: TriTrack 1.2B

The repository also contains the new `openprotein-tritrack-1.2b` configuration:
1,168,564,995 total parameters, 517,521,741 trainable parameters, 18 bridge
layers, a contextual shared text encoder, multi-positive contrastive learning,
residue-conditioned structure gating, masked-structure embeddings, and
structure-modality dropout.

Its current evidence is only a four-RTX-4090 checkpoint/resume smoke: step 5
was saved and resumed exactly through step 10, for 104,604 tokens, with
7,507.55 MiB peak memory and approximately 15.5k token/s on the last logged
step. The immutable source snapshot SHA-256 is
`f19b081f2504f3f7dd9960cf8c9cabc13d50ec073503a5e9fc2def9be0f2d3d3`.
No TriTrack quality result is claimed yet.

## Scratch 7B status

The separate 7,065,227,297-parameter random-initialized model reached step 5,000
and 1,064,746,261 tokens. Its successor was canceled before execution. This is
a completed training segment and checkpoint-recovery result, not completed 7B
pretraining and not a downstream quality result.
