# Evaluation contract

The success claim is not “training completed.” It requires a versioned model,
sealed data manifests, exact baseline revisions, and paired evaluation receipts.

## Baselines

At minimum:

- ESM-2 15B, because it is named in the original objective;
- ESMC-6B, because it is the stronger open 2026 sequence baseline;
- smaller compute-matched controls;
- task-specialized methods where a benchmark requires them.

## Gates

1. sequence validation MLM loss and perplexity-like masked-token metrics;
2. zero-shot variant ranking on a frozen ProteinGym snapshot;
3. function/family transfer with fixed probes and cluster-aware splits;
4. contact/distance prediction on a date- and identity-filtered structure set;
5. sequence-text retrieval and annotation generation with factuality checks;
6. design metrics clearly labeled computational, followed by wet-lab evidence
   before any biological efficacy claim.

Results are reported by task and dataset. “Overall better” is allowed only when
the predeclared aggregation rule passes and no safety-critical regression is
hidden by averaging.

Sequence-stage checkpoint selection is validation-only. Every 10,000-step
milestone is evaluated with the same evaluator revision, validation file list,
mask count, frozen-backbone control, and training provenance.
`openprotein-select-sequence-checkpoint` rejects any drift in those fields and
selects minimum bridge MLM loss, breaking an exact tie in favor of the earlier
step. This selection is fixed before any frozen ProteinGym row is opened.
For the packed 1B-token frozen-backbone control, the predeclared candidates are
exactly steps 10,000 through 130,000 at 10,000-step intervals. The non-milestone
token-budget endpoint is retained as completion/recovery evidence but is not
added to the selection set. Evaluation may begin only after a passed 1B-token
`PRETRAIN` receipt exists. The immutable policy is
`configs/eval/uniref50-packed-1b-sequence-selection.json`.
The policy-bound selector additionally requires the complete candidate-step
set, an exact 1B-token `PRETRAIN` receipt, matching checkpoint training
provenance, and the frozen evaluator/source/data/artifact/tokenizer hashes
before it calls the base minimum-loss selector. Any omitted candidate or drift
fails closed without producing a selection receipt.

Sequence-text checkpoint selection is predeclared at steps 1k, 5k, and 10k.
A candidate is eligible only if the normal 95% interval for paired-minus-shuffled
text loss lies entirely below zero and its sequence MLM loss is no more than
0.01 above the selected initializer. Among eligible checkpoints, minimum text
LM loss wins and an exact tie selects the earlier step. The selector binds the
training/initialization receipts and rejects validation-data, evaluator,
checkpoint-provenance, or policy drift.
For the continuation from packed step 100,000, policy schema v2 additionally
binds the exact upstream 13-point selection receipt, initialization checkpoint
manifest/state hashes, immutable source and evaluator package, sequence/text
validation file hashes, tokenizer and model-artifact hashes, and explicit
evaluation settings (lengths, batch sizes, mask rates, and seeds). It also
requires the initializer and every candidate sequence receipt to be exactly
comparable. The archived policy SHA-256 is
`e38d6c3f59b1de447218c9f20a8c2f3112af50423eda4774799ceb1471e84a84`.

## Evidence levels

- `STATIC`: code/config/parameter accounting only;
- `LOCAL_SMOKE`: tiny CPU or local-device execution;
- `GPU_SMOKE`: real allocated GPU forward/backward;
- `PILOT`: bounded training run with checkpoint recovery;
- `PRETRAIN`: declared token budget completed;
- `FROZEN_EVAL`: sealed benchmark on fixed checkpoint;
- `EXPERIMENTAL`: external laboratory validation.

No lower level may be described as a higher one.

Text validation reports both paired annotation loss and a deterministic
within-batch control in which protein prefixes are rotated while text targets
stay fixed. A low language-model loss alone is not evidence of protein
conditioning; the paired loss must beat this shuffled-protein control before a
sequence-conditioned annotation claim is allowed.

Before any frozen benchmark rows are read, `openprotein-seal-evaluation` binds
the checkpoint state and manifest, an archive-level benchmark manifest, model
config, tokenizer, ESMC artifact manifest and bytes, and evaluation-code tree
by SHA-256. The archive-level manifest binds the opaque assay ZIP, official
baseline-score ZIP, and tagged reference CSV by byte count and SHA-256; it does
not require opening a ZIP to construct the seal. Only after the seal verifies
does the evaluator verify all three artifacts, open their rows, require exact
mutant-set equality, and evaluate once. Any subsequent drift aborts evaluation.
The zero-shot variant path uses masked marginal log odds and validates wildtype
residues against each assay sequence before scoring.

The ProteinGym runner scores the frozen ESMC head and OpenProtein bridge from
the same backbone forward pass and the same masked windows. For targets longer
than the declared model context, each mutated position uses a deterministic
centered window, clipped only at sequence ends. Multi-substitution scores are
the sum of per-position mutant-minus-wildtype log odds, matching the standard
masked-marginal rule.

Every assay and official baseline score file is byte- and SHA-256-bound by the
benchmark manifest before any row is opened. Baseline files must contain the
exact same mutant set as the assay. The primary aggregate first averages assays
for each UniProt ID within selection type, then macro-averages selection types.
The receipt reports every assay, UniProt/category aggregates, the predeclared
margin gate, and `test_data_accessed=true`; an empty required-baseline list can
never pass the claim gate.

To match the official ProteinGym performance script, each assay Spearman is
rounded to three decimals before UniProt/category aggregation; the unrounded
value is retained separately for audit. A superiority claim additionally
requires a deterministic 10,000-sample paired bootstrap: UniProt-selection
units are resampled within each functional category, category means are
macro-averaged, and the 95% candidate-minus-baseline lower bound must be at
least zero for both ESM-2 15B and the local ESMC-6B control.

Production ProteinGym evaluation uses four independent GPU ranks. Rank zero
verifies the seal and all three archive hashes, then ranks deterministically own
reference rows by `reference_index % world_size`; a skipped rank never opens the
assay or baseline member owned by another rank. Results are gathered, checked
for duplicate assay IDs, sorted deterministically, and written only by rank
zero. This distributed path must pass an opaque-archive synthetic smoke with
the production checkpoint format before the real frozen seal is created.
Synthetic manifests force `SYNTHETIC_EVAL_SMOKE`,
`test_data_accessed=false`, `synthetic_data_accessed=true`, and a failed claim
gate regardless of scores, so a pipeline smoke cannot be mistaken for a frozen
ProteinGym result.
