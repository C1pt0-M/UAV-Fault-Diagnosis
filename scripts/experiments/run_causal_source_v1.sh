#!/usr/bin/env bash
# Two registered source-development jobs only. Does not allocate compute.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'Usage: bash scripts/experiments/run_causal_source_v1.sh DATA_ROOT NEW_RUN_ROOT' >&2
  exit 2
fi
DATA_ROOT="$1"
RUN_ROOT="$2"
PYTHON="${PYTHON:-python}"
PROTOCOL=configs/experiments/causal_source_development_v1.json
if [[ -e "$RUN_ROOT" ]]; then
  echo 'Refusing existing run root; preserve previous/partial jobs.' >&2
  exit 1
fi
# Formal CUDA/environment checks execute before creating any run directory.
"$PYTHON" - <<'PY'
import json, sys
from pathlib import Path
import numpy as np
import torch
p=json.loads(Path('configs/experiments/causal_source_development_v1.json').read_text(encoding='utf-8'))
if not torch.cuda.is_available():
    raise SystemExit('BLOCKED_NO_CUDA: no CPU fallback and no compute allocation')
actual={'torch':str(torch.__version__),'numpy':np.__version__,'torch_cuda':torch.version.cuda}
for key,value in p['formal_environment'].items():
    if actual[key]!=value:
        raise SystemExit('Formal environment mismatch: '+key)
PY
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$RUN_ROOT"
"$PYTHON" scripts/research/train_causal_source_v1.py preflight \
  --protocol "$PROTOCOL" --data "$DATA_ROOT" --output "$RUN_ROOT/preflight"
for ARM in uniform_replacement balanced_source_normal; do
  "$PYTHON" -u scripts/research/train_causal_source_v1.py run \
    --protocol "$PROTOCOL" --data "$DATA_ROOT" \
    --output "$RUN_ROOT/$ARM" --arm "$ARM" --seed 42 \
    2>&1 | tee "$RUN_ROOT/$ARM.log"
done
# Both runs are source-val development results. Do not auto-deploy or score tests.
