# Architecture contract

## Production model

OpenProtein-7B is a composition, not a renamed checkpoint:

- immutable ESMC-6B masked-language-model backbone: 6,352,005,184 parameters;
- eight 2,560-wide pre-normalized SwiGLU bridge blocks;
- a 32,768-token text embedding/output matrix;
- sequence-text projection heads and a sampled-pair geometry head.

Protein and text tokens are fused in one bridge sequence. Protein embeddings
come from the ESMC encoder; text embeddings are learned by OpenProtein. A
modality embedding distinguishes the two segments. Rotary position embeddings
are applied inside each bridge block.

Two attention policies are part of the model contract:

- `bidirectional`: all non-padding protein and text tokens can attend to one
  another. This is used for geometry and text-conditioned protein infilling,
  but never for contrastive retrieval scoring.
- `prefix_lm`: protein tokens form a bidirectional prefix. Text tokens can attend
  to all protein tokens and prior text tokens; protein tokens cannot read text
  targets. This prevents annotation leakage during text generation and keeps
  the protein-side contrastive embedding independent of text.

The implementation rejects text-LM labels under `bidirectional`; that policy
would expose future target tokens. Contrastive text embeddings are pooled from
isolated text token plus modality embeddings, not fused hidden states that have
already seen the paired protein. Cosine logits use a fixed 0.07 temperature.
These constraints prevent paired-retrieval leakage.

The text embedding uses PyTorch's checkpoint-compatible unit-variance
parameterization, but both text-token inputs and tied text-output logits are
multiplied by `hidden_size ** -0.5`. Without this explicit scale, a tied
32,768-way output begins with logits of order `sqrt(hidden_size)` and produces
an artificial text loss in the hundreds or thousands before learning begins.
The scaling is applied only to text positions, so an existing sequence-only
checkpoint's protein path is unchanged.

The ESMC sequence-only path remains available without the bridge. This is a
non-regression control, not evidence that a trained multimodal model improves on
ESMC.

## Objective mixture

The model exposes separately logged losses. Training must never report only an
opaque weighted sum.

| Loss | Input | Target | Required split rule |
| --- | --- | --- | --- |
| Protein MLM | sequence, optional text | masked amino acids | cluster split |
| Text LM | sequence prefix + annotation | next annotation token | protein-cluster split |
| Contrastive | paired sequence and annotation | matched pair | protein-cluster split |
| Geometry | sequence + sampled residue pairs | distance bins | structure/date/cluster split |

Pairwise geometry is sampled. The head projects residue states to a low-rank
space and classifies selected residue pairs, so memory is `O(L + N_pairs)` rather
than `O(L^2 * bins)`.

AlphaFold coordinates are indexed without extraction, retain pLDDT, and are
joined by accession plus exact full-length sequence. Geometry training samples
only confident C-alpha pairs, shifts residue indices for the protein CLS token,
and pads unused pair slots with `ignore_index` labels.

The first shared-bridge structure pilot is treated as a negative control if it
improves geometry while regressing text. The protected geometry-head stage
therefore detaches bridge states before the geometry head and assigns zero
weight to protein MLM; zero-weight losses are excluded from the autograd graph,
so AdamW cannot alter shared parameters through weight decay. This stage learns
a structure decoder over the validated sequence-text representation without
claiming that predicted structures improved the shared representation.

## Parameter accounting

`openprotein-inspect` computes the parameter contract analytically before the
6B backbone is instantiated. Production is valid only if the calculated count
is within [7B, 15B] and a loaded-model count later agrees exactly.

## Scaling policy

Stage 0 trains tiny randomly initialized configurations to validate numerics,
checkpoint/restart, and data determinism. Stage 1 freezes ESMC and trains bridge
and heads. Sequence, sequence-text, and sequence-structure stages can initialize
from the preceding adapter checkpoint while explicitly resetting optimizer,
data cursor, and RNG state; this operation is distinct from strict same-stage
resume and records both source and destination provenance. Each transition must
rerun sequence validation to detect forgetting. Stage 2 may unfreeze selected
top ESMC blocks only after sequence-only non-regression checks. A full-backbone
run requires an explicit throughput, memory, and recovery receipt first.
