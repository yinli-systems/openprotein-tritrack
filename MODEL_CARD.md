# OpenProtein-7B frozen-backbone adapter model card

Status: exact-token adapter pretraining and downstream gates complete; the
one-time frozen evaluation failed closed on 2026-08-24 without producing a
score. No superiority claim is approved.

This card describes only the frozen-ESMC adapter/control track. The separately
random-initialized model is documented in `SCRATCH_MODEL_CARD.md`; evidence from
the two tracks is not interchangeable.

## Model

OpenProtein-7B combines an external frozen ESMC-6B sequence backbone with eight
trainable multimodal bridge blocks, a tied text embedding/output matrix,
sequence-text projection heads, and a sampled residue-pair distance head.

| Component | Parameters |
| --- | ---: |
| External ESMC-6B backbone | 6,352,005,184 |
| OpenProtein adapter and heads | 724,893,248 |
| Total loaded model | 7,076,898,432 |

The default release contains only the optimizer-free adapter, exact model
configuration, and trained text tokenizer. It does not redistribute ESMC-6B;
users must obtain the exact external backbone under its upstream terms and pass
content verification during reconstruction.

## Inputs and outputs

- Protein sequence: masked-token logits, bridged residue states, and pooled
  sequence embeddings.
- Curated functional text: prefix-LM token logits and text embeddings.
- Sampled residue pairs: 64-bin distance logits from a low-rank geometry head.

Maximum configured lengths are 2,048 protein tokens and 512 text tokens.
Training and current validation use protein length 256. Performance at the
configured maxima is not established by the current receipts.

## Current evidence

- Exact loaded parameter count and real RTX 5090 forward/backward are verified.
- The original fixed-record four-GPU run was rejected because its step limit
  could not reach 1B tokens. The packed replacement completed its exact-token
  contract in job 1464156 at step 137,148 and 1,000,004,951 tokens. Its passed
  `PRETRAIN` receipt SHA-256 is
  `2c68883a9b27524221eff6c783a5f1fedd8519e940fce8f8af458a0c074a1533`.
  This is frozen-backbone adapter pretraining, never scratch-7B evidence.
- The complete predeclared 13-checkpoint validation sweep selected step 100,000
  with bridge MLM loss 1.80894 and perplexity 6.10398. The policy-bound selection
  receipt SHA-256 is
  `38af3ed7361c121631234195c3f1b809bffc9f97382a5064f83adb1f2e513b32`;
  all validation receipts report no test access and ProteinGym remains sealed.
- The predeclared 1k/5k/10k sequence-text selection chose step 10,000. Its text
  LM loss is 4.21813; the shuffled-protein control is 4.29599, and the
  paired-minus-shuffled batch-mean normal 95% interval is
  [-0.08501, -0.07124]. Sequence bridge loss is 1.81005, only +0.00110 from the
  packed step-100,000 initializer. The policy-bound selection receipt SHA-256 is
  `e8771187a77dab65f688b6b1893f1a3420bf35bcd0c6370946e2059e3cd4fc1f`.
- The protected predicted-geometry continuation passed its fail-closed policy:
  loss improved from 4.13639 to 3.02448 and accuracy from 0.02476 to 0.34384 on
  127,280 AlphaFold DB predicted-structure targets. Only the 1,327,168
  geometry-head parameters changed; 723,566,080 shared parameters and all
  sequence/text validation metrics remained bitwise identical. Gate receipt
  SHA-256 is
  `2c2bc6aaf86cd5478c6aa9cdbabb4555641d57a9897997fc749869a8c09c8561`.
- The final checkpoint and evaluator were sealed before official access. The
  single permitted four-GPU ProteinGym v1.3 attempt, job 1464976, failed because
  the frozen evaluator found no score-archive member matching its declared
  ESM-2 15B path contract. Assay rows had already been opened, so the one-time
  boundary is exhausted and no corrected rerun is permitted. No score or
  superiority result exists.

Exact job IDs, hashes, negative controls, and failed candidates are in
`docs/EVIDENCE.md`.

## Intended use

The model is intended for research on protein-sequence representations,
validation-gated variant scoring, sequence-conditioned annotation experiments,
and sampled structural-geometry analysis. It may be used to generate hypotheses
that are subsequently evaluated by domain experts and appropriate experiments.

## Out-of-scope use and safety

This model is not a medical device, diagnostic, therapeutic recommendation
system, or substitute for expert review. It must not be presented as evidence
that a protein is safe, effective, synthesizable, non-pathogenic, or suitable
for release. Generated sequence or function hypotheses require independent
biosafety review and wet-lab validation. Users are responsible for applicable
laws, institutional policies, screening requirements, and upstream licenses.

## Limitations

- The backbone is frozen; current training adapts bridge and task-head
  parameters rather than learning a new 7B sequence model from scratch.
- The 1B-token `PRETRAIN` receipt applies only to the frozen-backbone adapter
  control and does not establish scratch pretraining or benchmark quality.
- Curated sequence-text coverage is limited by exact UniRef/Swiss-Prot joining.
- The structure stage uses AlphaFold DB predictions and pLDDT filtering, not
  experimental structures or a validated structure-prediction benchmark.
- Current text retrieval is weak, and annotation factuality has not passed a
  frozen evaluation.
- No claim of superiority over ESM-2 15B, ESMC-6B, or task-specialized methods
  is allowed. The only permitted sealed benchmark attempt failed without a
  score and cannot be retried under the fixed policy.

## Licenses and attribution

Repository code is Apache-2.0. The pinned Biohub ESM source snapshot is MIT;
ESMC model use and distribution are governed by its upstream model terms.
UniProt and AlphaFold DB inputs are CC BY 4.0 and require attribution.
ProteinGym project code is MIT. Consult `docs/RELEASE.md` and the source
manifests before redistribution.
