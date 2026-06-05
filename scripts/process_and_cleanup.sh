#!/bin/bash
# 处理 RflyMAD 数据并边处理边清理，避免磁盘爆满

set -e
cd /home/jack/Data_processing_tools
source /home/jack/miniconda3/bin/activate UAV-Fault-Diagnosis

ORIGINAL=/home/jack/UAV-Fault-Diagnosis/data
OUTPUT=/home/jack/UAV-Fault-Diagnosis/data/processed
KEEP_PATTERNS="actuator_armed\|actuator_outputs\|estimator_attitude\|estimator_status\|sensor_combined\|vehicle_command\|vehicle_global_position\|vehicle_gps_position\|vehicle_magnetometer"

# HIL 剩余: 1=motor 2=propeller 6-10=sensors
FAULTS="1 2 6 7 8 9 10"
FAULT_NAMES=("" "motor" "propeller" "" "" "" "accelerometer" "gyroscope" "magnetometer" "barometer" "GPS")

for fault in $FAULTS; do
    fname=${FAULT_NAMES[$fault]}
    echo ""
    echo "========================================"
    echo "处理 HIL / $fname (fault=$fault)"
    echo "========================================"

    python Rflytool_main.py \
        --original_path "$ORIGINAL" \
        --restore_path "$OUTPUT" \
        --sub_dataset 2 \
        --fault_type $fault \
        --flight_status 0 \
        --trans_freq 50 2>&1 | tail -3

    # 立即清理该故障类型的 CSV
    echo "清理 HIL/$fname 的 CSV..."
    find "$ORIGINAL/HIL" -path "*/$fname/*/Log/*.csv" -type f | grep -v "$KEEP_PATTERNS" | xargs rm -f 2>/dev/null || true

    echo "当前磁盘:"
    df -h /home/jack --output=avail,pcent | tail -1
done

echo ""
echo "========================================"
echo "HIL 完成！开始处理 Real..."
echo "========================================"

# Real: 11=no_fault 1=motor 6-10=sensors (无 propeller)
FAULTS="11 1 6 7 8 9 10"

for fault in $FAULTS; do
    fname=${FAULT_NAMES[$fault]}
    [ "$fault" -eq 11 ] && fname="no_fault"

    echo ""
    echo "========================================"
    echo "处理 Real / $fname (fault=$fault)"
    echo "========================================"

    python Rflytool_main.py \
        --original_path "$ORIGINAL" \
        --restore_path "$OUTPUT" \
        --sub_dataset 3 \
        --fault_type $fault \
        --flight_status 0 \
        --trans_freq 50 2>&1 | tail -3

    # 立即清理
    echo "清理 Real/$fname 的 CSV..."
    find "$ORIGINAL/Real" -path "*/$fname/*/Log/*.csv" -type f | grep -v "$KEEP_PATTERNS" | xargs rm -f 2>/dev/null || true
    find "$ORIGINAL/Real" -path "*/$fname/*/rfly_real_*/*.csv" -type f | xargs rm -f 2>/dev/null || true

    echo "当前磁盘:"
    df -h /home/jack --output=avail,pcent | tail -1
done

echo ""
echo "========================================"
echo "全部完成！"
csv_count=$(ls "$OUTPUT"/*.csv 2>/dev/null | wc -l)
echo "CSV 文件数: $csv_count / 4984"
echo "最终磁盘:"
df -h /home/jack --output=avail,pcent | tail -1
echo "========================================"
