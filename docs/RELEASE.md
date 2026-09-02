# Release contract

OpenProtein code is Apache-2.0. The pinned Biohub ESM source snapshot is MIT,
while ESMC model distribution and use must also follow the terms attached to
the upstream model repository. UniProt and AlphaFold DB inputs are CC BY 4.0
and require attribution. ProteinGym's project repository is MIT; benchmark
receipts must retain its version and source metadata.

The default release is an optimizer-free OpenProtein adapter, not a repackaged
ESM-2 or ESMC checkpoint. `openprotein-export-adapter` verifies the training
checkpoint, model config, tokenizer, and every external-backbone artifact byte,
then writes:

- `adapter.safetensors`, containing only trainable bridge and head tensors;
- `model-config.json`, the exact model configuration used by the checkpoint;
- `text-tokenizer.model` for checkpoints whose provenance binds multimodal text;
- `pretrain-receipt.json` when the export is bound to completed `PRETRAIN` evidence;
- `manifest.json`, with hashes, parameter count, upstream dependency receipts,
  checkpoint provenance, and the exact claim boundary;
- `README.md`, explaining how to obtain and verify the external backbone.

`openprotein-reconstruct-adapter` verifies every bundled file, the external
protein tokenizer, every external-backbone artifact, checkpoint binding, tensor
name, shape, dtype, and parameter count before loading. `--backbone-root` permits
the same content-addressed ESMC files to live at a different absolute path; it
does not relax any file hash. Reconstruction writes a no-overwrite JSON receipt.
The GPU smoke wrapper `scripts/smoke_release_reconstruction.sbatch` exercises
the production 7.08B model and adapter format; a pilot checkpoint receipt proves
the release mechanism only, while the final release still requires the selected
pretraining checkpoint and frozen benchmark gates below.

After reconstruction, `openprotein-score-variants` emits paired ESMC and
OpenProtein masked-marginal scores from the same backbone forward pass.
`openprotein-generate-annotation` performs deterministic greedy prefix-LM text
generation and binds its output to the reconstruction receipt. These are
computational inference tools, not biological or clinical evidence.
`openprotein-predict-geometry` returns coarse 64-bin distance predictions for
explicit one-based residue pairs. It is a decoder trained on confidence-filtered
AlphaFold DB predictions, not an experimental structure predictor.

Release is blocked until all of the following exist:

1. a declared pretraining-token receipt and recovery check;
2. validation-only selection records that never touch frozen test rows;
3. a sealed ProteinGym receipt with ESM-2 15B and exact ESMC controls;
4. model and data cards with licenses, intended use, limitations, safety notes,
   compute/runtime details, and all failed as well as passed gates;
5. a clean-environment reconstruction that reproduces parameter count, adapter
   hashes, and benchmark aggregation from the published manifests.

No computational release may claim biological efficacy. Any design candidate
requires an independently documented experimental validation stage.
