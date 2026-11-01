#!/usr/bin/env bash
# Run from the unpacked code bundle; cache and graph directories are its siblings.
set -euo pipefail

TOPOLOGY_CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TOPOLOGY_DATA_ROOT="$(dirname -- "$TOPOLOGY_CODE_ROOT")"
TOPOLOGY_CACHE_DIR="${TOPOLOGY_CACHE_DIR:-$TOPOLOGY_DATA_ROOT/topology_cache_v1}"
TOPOLOGY_GRAPH_DIR="${TOPOLOGY_GRAPH_DIR:-$TOPOLOGY_DATA_ROOT/topology_graph}"
TOPOLOGY_RUN_DIR="${TOPOLOGY_RUN_DIR:-$TOPOLOGY_CODE_ROOT/outputs/topology_gnn_full}"

if [[ -e "$TOPOLOGY_RUN_DIR" ]]; then
    echo "Output already exists: $TOPOLOGY_RUN_DIR. Set TOPOLOGY_RUN_DIR to a new path."
    exit 1
fi
export PYTHONPATH="$TOPOLOGY_CODE_ROOT/packages/uav_fault${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
cd -- "$TOPOLOGY_CODE_ROOT"

python -X utf8 -u scripts/experiments/preflight_topology_training.py \
    --cache-dir "$TOPOLOGY_CACHE_DIR" \
    --graph-dir "$TOPOLOGY_GRAPH_DIR" \
    --report "$TOPOLOGY_CODE_ROOT/preflight.json"

python -c 'import torch; assert torch.cuda.is_available(), "Full training requires a CUDA GPU"; print(torch.__version__, torch.cuda.get_device_name(0))'

python -X utf8 -u -m uav_fault.models.train_topology_gnn \
    --cache-dir "$TOPOLOGY_CACHE_DIR" \
    --graph-dir "$TOPOLOGY_GRAPH_DIR" \
    --output-dir "$TOPOLOGY_RUN_DIR" \
    --epochs 100 --batch-size 512 --num-workers 4 \
    --patience 20 --min-epochs 30 --min-delta 0 \
    --norm batch --seed 42 "$@"
