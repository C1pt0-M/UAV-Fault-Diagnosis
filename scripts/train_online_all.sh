#!/bin/bash
# Rfly realtime binary 在线故障检测模型训练（一键脚本）
# 用法:
#   cd ~/UAV-Fault-Diagnosis
#   bash scripts/train_online_all.sh
#   systemd-inhibit --what=sleep bash scripts/train_online_all.sh
#
# 日志输出到 logs/train_online_<timestamp>.log

set -eo pipefail
cd "$(dirname "$0")/.."

# Python 解释器优先级:
# 1) 外部显式传入 PYTHON
# 2) 项目 .venv
# 3) 系统 python / python3
if [ -z "${PYTHON:-}" ]; then
    if [ -x ".venv/bin/python" ]; then
        PYTHON=".venv/bin/python"
    elif command -v python >/dev/null 2>&1; then
        PYTHON="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
    else
        echo "[ERROR] 未找到可用 Python，请先激活环境或设置 PYTHON=/path/to/python"
        exit 1
    fi
fi

# 训练缓存目录优先级:
# 1) 外部显式传入 CACHE_DIR
# 2) 仓库默认 Rfly 在线缓存 data/processed/online_cache_rfly
if [ -z "${CACHE_DIR:-}" ]; then
    CACHE_DIR="data/processed/online_cache_rfly"
fi

BATCH_SIZE=${BATCH_SIZE:-512}
MAX_SAMPLES=${MAX_SAMPLES:-0}  # 0=全量训练
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-20}
MIN_EPOCHS=${MIN_EPOCHS:-30}

# 按缓存目录自动给输出目录加后缀，避免覆盖历史模型
if [ -z "${OUTPUT_SUFFIX:-}" ]; then
    CACHE_BASENAME="$(basename "$CACHE_DIR")"
    if [ "$CACHE_BASENAME" = "online_cache_rfly" ]; then
        OUTPUT_SUFFIX="_rfly"
    else
        OUTPUT_SUFFIX="_${CACHE_BASENAME}"
    fi
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/train_online_${TIMESTAMP}.log"

echo "=== 训练日志: $LOG ==="
echo "开始时间: $(date)" | tee "$LOG"
echo "任务: Rfly realtime binary 训练" | tee -a "$LOG"
echo "Python: $PYTHON" | tee -a "$LOG"
echo "Cache: $CACHE_DIR" | tee -a "$LOG"
TRAIN_MODELS=${TRAIN_MODELS:-cnn1d}
echo "Models: $TRAIN_MODELS" | tee -a "$LOG"

train_model() {
    local model="$1"
    local step_label=""
    local extra_args=()

    case "$model" in
        bilstm)
            step_label="BiLSTM"
            extra_args=(--hidden-size "${BILSTM_HIDDEN_SIZE:-96}" --num-layers "${BILSTM_NUM_LAYERS:-1}")
            ;;
        cnn1d)
            step_label="1D CNN"
            extra_args=(--hidden-size "${CNN1D_HIDDEN_SIZE:-64}")
            ;;
        transformer)
            step_label="Transformer"
            extra_args=(--hidden-size "${TRANSFORMER_HIDDEN_SIZE:-64}" --num-layers "${TRANSFORMER_NUM_LAYERS:-2}")
            ;;
        mstca)
            step_label="MSTCA"
            extra_args=(--hidden-size "${MSTCA_HIDDEN_SIZE:-64}")
            ;;
        *)
            echo "[ERROR] 不支持的模型: $model (支持: bilstm, cnn1d, transformer, mstca)" | tee -a "$LOG"
            exit 1
            ;;
    esac

    echo "" | tee -a "$LOG"
    echo "====== Step 2: 训练 ${step_label} (${model}) ======" | tee -a "$LOG"
    "$PYTHON" src/models/train_online_detector.py \
        --model "$model" \
        --cache-dir "$CACHE_DIR" \
        --output-dir "models/online_detector_${model}${OUTPUT_SUFFIX}" \
        --epochs "$EPOCHS" --patience "$PATIENCE" --min-epochs "$MIN_EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --max-train-samples "$MAX_SAMPLES" \
        "${extra_args[@]}" \
        2>&1 | tee -a "$LOG"
}

# Step 1: 检查/重建缓存

echo "" | tee -a "$LOG"
echo "====== Step 1: 检查/重建 Rfly 在线缓存 ======" | tee -a "$LOG"

CACHE_OK=0
if [ -f "$CACHE_DIR/norm_stats.json" ]; then
    CACHE_OK=$(CACHE_NORM_PATH="$CACHE_DIR/norm_stats.json" $PYTHON - <<'PY'
import json
import os
from pathlib import Path
p = Path(os.environ["CACHE_NORM_PATH"])
try:
    d = json.loads(p.read_text(encoding="utf-8"))
    cols = d.get("columns") or []
    ok = (
        (float(d.get("target_hz", 0.0)) == 50.0)
        and (int(d.get("window_size", 0)) == 150)
        and (int(d.get("stride", 0)) == 10)
        and (str(d.get("feature_schema", "")) == "rfly_seq_columns_v1")
        and (len(cols) == 31)
    )
    print(1 if ok else 0)
except Exception:
    print(0)
PY
)
fi

if [ "$CACHE_OK" -eq 1 ]; then
    echo "Rfly 缓存已存在且配置匹配(target_hz=50, window_size=150, stride=10, feature_schema=rfly_seq_columns_v1)，跳过重建" | tee -a "$LOG"
else
    if [ "$CACHE_DIR" != "data/processed/online_cache_rfly" ]; then
        echo "[ERROR] $CACHE_DIR 中缺少可用缓存(norm_stats.json 不存在或配置不匹配)。" | tee -a "$LOG"
        echo "        请确认缓存上传完整，或显式设置 CACHE_DIR=data/processed/online_cache_rfly 后再允许重建。" | tee -a "$LOG"
        exit 1
    fi
    echo "Rfly 缓存不存在或配置不匹配，开始重建默认缓存..." | tee -a "$LOG"
    $PYTHON src/data/build_online_cache.py --output "$CACHE_DIR" --target-hz 50 --window-size 150 --stride 10 2>&1 | tee -a "$LOG"
fi

# Step 2: 默认训练单模型（可通过 TRAIN_MODELS 覆盖）

echo "" | tee -a "$LOG"
echo "====== Step 2: 执行模型训练 ======" | tee -a "$LOG"
TRAINED_MODELS=()
for model in $TRAIN_MODELS; do
    train_model "$model"
    TRAINED_MODELS+=("models/online_detector_${model}${OUTPUT_SUFFIX}/")
done

echo "" | tee -a "$LOG"
echo "====== 全部完成 ======" | tee -a "$LOG"
echo "结束时间: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "模型保存位置:" | tee -a "$LOG"
for model_dir in "${TRAINED_MODELS[@]}"; do
    echo "  $model_dir" | tee -a "$LOG"
done
echo "" | tee -a "$LOG"
echo "测试结果请查看各目录下的 metrics_test.json" | tee -a "$LOG"
