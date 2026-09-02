# OpenProtein 1B multimodal pretraining contract

## Frozen scope

The model is a protein sequence/structure/function-text model, not an
image-language model.  It warm-starts from Meta ESM-2 650M and adds a 12-layer,
1280-wide multimodal bridge, curated-function text embedding, confidence-gated
continuous structure adapter, contrastive heads, and sampled geometry head.  The
analytical total is 1,011,246,594 unique loaded parameters, including 651,043,254
parameters in the instantiated ESM-2 backbone and 360,203,340 trainable
multimodal parameters.  The source safetensors contains 652,358,616 tensor
elements; the loaded unique-parameter count is authoritative because the model
ties or omits compatibility tensors during instantiation.

This is deliberately a warm-start rather than a random 1B run.  DPLM-2 reports
both the value of sequence-pretrained initialization and catastrophic forgetting
when the multimodal structure corpus is much smaller than the original sequence
corpus.  ESM-2 is frozen during the first formal phase, every joint batch retains
protein MLM, and every fourth microbatch is broad UniRef50 sequence replay.

Primary design sources:

- DPLM-2: <https://openreview.net/pdf?id=5z9GjHgerY>
- multimodal protein design-space study: <https://openreview.net/pdf?id=jEcQP3lGlq>
- SaProt: <https://openreview.net/pdf?id=6MRm3G4NiU>
- ProtST: <https://openreview.net/pdf?id=ZOOwHgxfR4>
- ProteinGym: <https://proteingym.org/>

## Architecture and objectives

- Sequence backbone: ESM-2 t33 650M, 33-token vocabulary, hidden size 1280,
  frozen in phase A.
- Fusion: 12 pre-normalized RoPE/SwiGLU blocks, 20 heads, 5120 FFN, residual
  scale 0.2.
- Structure input: 12 rigid-motion-invariant CA-trace channels (pLDDT, eight
  local distances, bend cosine, pseudo-dihedral sine/cosine).  Residues below
  pLDDT 70 are hidden.  A zero-initialized learned gate makes the initial
  sequence/text path exactly equal to the warm start.
- Function text: train-split-only 32,768-piece SentencePiece model and
  UniRef50-to-Swiss-Prot exact-sequence joins.
- Objectives: protein MLM 1.0, causal function-text LM 0.5, cross-GPU
  sequence-text contrastive 0.1, sampled long-range distance classification
  0.25, and masked continuous-structure denoising 0.25.
- Complexity: structural supervision is O(L + sampled pairs).  It does not use
  O(L^2) triangle updates or pair-representation alignment, whose measured cost
  was not justified by the cited design-space ablation.

## Data and leakage contract

- Sequence replay comes only from the inherited UniRef50 train split.
- Text comes only from exact Swiss-Prot representatives and inherits the
  UniRef split.
- AlphaFold structures join by accession and byte-identical full sequence.
- Validation is allowed for checkpoint selection; inherited test, ProteinGym,
  CAMEO/PDB-date, and motif-scaffolding sets are frozen until selection.
- The AlphaFold pLDDT field is a confidence estimate, not experimental truth.

## Selection and metric gates

The smoke gate requires finite loss and gradients for all five objectives,
peak memory below the physical 24 GiB limit, non-zero cross-GPU contrastive
batch, valid checkpoint SHA-256, and an exact resume continuation.

Formal checkpoint selection uses validation only:

1. protein MLM loss delta versus frozen ESM-2 and versus a shuffled-structure
   control;
2. text perplexity plus paired-vs-shuffled conditioning delta with bootstrap
   confidence interval;
3. structure-denoising error stratified by pLDDT and distance-bin accuracy;
4. retrieval Recall@1/5/10 in both sequence-to-text and text-to-sequence
   directions;
5. no regression beyond the predeclared sequence-loss tolerance.

Only after a checkpoint is selected are frozen external metrics run:

- ProteinGym macro Spearman, AUC, MCC, NDCG@10, and recall with per-assay
  bootstrap confidence intervals;
- GO MF/BP/CC and EC Fmax, DeepLoc accuracy, and long-range contact P@L;
- PDB-date/CAMEO folding RMSD and TM-score; inverse-folding AAR and
  self-consistency TM-score;
- generation validity (pLDDT/scTM), diversity, novelty, and motif-scaffolding
  success at motif RMSD < 1 A and pLDDT > 70 or scTM > 0.8.

Public numbers are context, not acceptance thresholds, unless tokenizer,
training data, and evaluation protocol are matched exactly.

## Compute gate

The first allocation is a 10-step, 4x RTX 4090 BF16 DDP smoke at sequence
length 256.  Batch-cap jobs then compare sequence length 512 at microbatch 16,
24, and 32.  The selected formal candidate uses microbatch 24, no gradient
accumulation, a global contrastive batch of 96, fused AdamW, and deterministic
sequence replay.  Its measured peak is 17,087 MiB of 24,564 MiB and its
post-warmup aggregate throughput is about 43.8k token/s, preserving 30.4 percent
physical-memory headroom.  Microbatch 32 is rejected despite fitting because it
is slower and leaves only 12.8 percent headroom.  The first formal stage stops
at 1B multimodal tokens for frozen validation rather than repeating the small
curated triple corpus merely to consume a seven-day allocation.
