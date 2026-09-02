#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /ssd/USER/PROJECT_ROOT" >&2
  exit 2
fi

project_root=$1
source_root="$project_root/source"
vendor_root="$project_root/vendor/esm-d0207ea3c8cbf072679ece3bb332787d25e06852"
bootstrap_root="$project_root/bootstrap"
runtime_root="$project_root/runtime"
cache_root="$project_root/cache"
uv_bin="$bootstrap_root/local/bin/uv"
python_install="$runtime_root/python"
python_bin="$python_install/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12"
venv_python="$runtime_root/venv/bin/python"

[[ -f "$source_root/pyproject.toml" ]]
[[ -f "$vendor_root/pyproject.toml" ]]
mkdir -p "$bootstrap_root" "$runtime_root/tmp" "$cache_root/pip" "$cache_root/uv"

if [[ ! -x "$uv_bin" ]]; then
  PIP_CACHE_DIR="$cache_root/pip" TMPDIR="$runtime_root/tmp" \
    python3 -m pip install --disable-pip-version-check --prefix "$bootstrap_root" uv==0.11.31
fi

export UV_CACHE_DIR="$cache_root/uv"
export UV_PYTHON_INSTALL_DIR="$python_install"
export TMPDIR="$runtime_root/tmp"

if [[ ! -x "$python_bin" ]]; then
  "$uv_bin" python install 3.12.13 --no-bin
fi
if [[ ! -x "$venv_python" ]]; then
  "$uv_bin" venv "$runtime_root/venv" --python "$python_bin" --seed
fi

"$uv_bin" pip install \
  --python "$venv_python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128
"$uv_bin" pip install \
  --python "$venv_python" \
  numpy==2.3.5 \
  safetensors==0.8.0 \
  accelerate==1.14.0 \
  transformers==5.15.1 \
  attrs==26.1.0 \
  einops==0.8.2 \
  pygtrie==2.5.0 \
  sentencepiece==0.2.2 \
  zstandard==0.25.0 \
  pytest==9.1.1
"$uv_bin" pip install --python "$venv_python" --no-deps -e "$vendor_root"
"$uv_bin" pip install --python "$venv_python" -e "$source_root"

cd "$source_root"
"$venv_python" -m pytest -q
"$uv_bin" pip freeze --python "$venv_python" > "$runtime_root/runtime.freeze.txt"

printf 'runtime_ready_python=%s\n' "$venv_python"
printf 'GPU verification is still required; do not create .prepared on the login node.\n'
