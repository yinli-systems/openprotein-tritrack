#!/usr/bin/env bash
#SBATCH --job-name=afdb-index
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/afdb-index-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/afdb-index-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
cd "$root/source"
"$root/runtime/venv/bin/python" -m openprotein.cli.build_structure_index \
  "$root/data/raw/alphafold/v6/swissprot_pdb_v6.tar" \
  "$root/data/processed/alphafold-swissprot-v6/structures.sqlite" \
  "$root/data/processed/alphafold-swissprot-v6/manifest.json" \
  --source-definition "$root/source/data_sources/alphafold-swissprot-v6.json"
