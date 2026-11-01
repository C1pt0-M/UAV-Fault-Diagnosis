#!/usr/bin/env bash
# Run only from the frozen, unpacked minimal bundle. Six sequential source-only jobs.
set -euo pipefail

if [[ $# -ne 0 ]]; then
    echo "No hyperparameter overrides accepted. Configure only PYTHON_BIN, SOURCE_CACHE_DIR, SOURCE_AUDIT_DIR and SOURCE_RUN_ROOT." >&2
    exit 2
fi
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CACHE_DIR="${SOURCE_CACHE_DIR:-/root/autodl-tmp/online_cache_v2}"
AUDIT_DIR="${SOURCE_AUDIT_DIR:-$CODE_ROOT/inputs/source_sampling}"
RUN_ROOT="${SOURCE_RUN_ROOT:-/root/autodl-tmp/experiments/source_normal_sampling_v1/runs/$(date -u +%Y%m%dT%H%M%SZ)}"
PROTOCOL="$CODE_ROOT/configs/experiments/normal_source_sampling_train_v1.json"
TRAINER="$CODE_ROOT/scripts/research/train_source_normal_sampling.py"

if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing existing run root: $RUN_ROOT" >&2
    exit 1
fi
cd -- "$CODE_ROOT"
if [[ ! -f SHA256SUMS ]]; then
    echo "Missing SHA256SUMS: use the frozen training bundle, not an arbitrary working tree." >&2
    exit 1
fi
sha256sum --check --strict SHA256SUMS
export PYTHONPATH="$CODE_ROOT/packages/uav_fault:$CODE_ROOT/scripts/research${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Deliberately do not install anything or replace a CUDA Torch wheel automatically.
"$PYTHON_BIN" - "$PROTOCOL" <<'PY'
import importlib.util
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    protocol = json.load(stream)
missing = [name for name in ("torch", "numpy", "pandas") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing dependencies: " + ", ".join(missing) + ". See README_source_normal_sampling_v1.md; no automatic installation performed.")
import numpy
import pandas
import torch
actual = {"torch": str(torch.__version__), "numpy": numpy.__version__, "pandas": pandas.__version__, "torch_cuda": torch.version.cuda}
for name, expected in protocol["formal_environment"].items():
    if actual[name] != expected:
        raise SystemExit(f"Environment mismatch: {name}={actual[name]!r}; expected {expected!r}. Do not start training until resolved.")
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable. Formal CPU fallback is forbidden.")
print("Frozen environment verified:", actual, torch.cuda.get_device_name(0), flush=True)
PY

mkdir -p -- "$(dirname -- "$RUN_ROOT")"
mkdir -- "$RUN_ROOT"
mkdir -- "$RUN_ROOT/logs"
"$PYTHON_BIN" -u "$TRAINER" preflight --protocol "$PROTOCOL" --cache "$CACHE_DIR" --audit "$AUDIT_DIR" \
    --output "$RUN_ROOT/preflight.json" 2>&1 | tee "$RUN_ROOT/logs/preflight.log"

for seed in 42 43 44; do
    for arm in uniform_replacement balanced_source_normal; do
        "$PYTHON_BIN" -u "$TRAINER" run --protocol "$PROTOCOL" --cache "$CACHE_DIR" --audit "$AUDIT_DIR" \
            --seed "$seed" --arm "$arm" --output "$RUN_ROOT/seed_$seed/$arm" \
            2>&1 | tee "$RUN_ROOT/logs/seed_${seed}_${arm}.log"
    done
done
"$PYTHON_BIN" -u "$TRAINER" summarize --protocol "$PROTOCOL" --runs "$RUN_ROOT" \
    --output "$RUN_ROOT/summary.json" 2>&1 | tee "$RUN_ROOT/logs/summary.log"
echo "All six source-training jobs completed. Summary: $RUN_ROOT/summary.json"
