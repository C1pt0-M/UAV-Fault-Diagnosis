#!/bin/bash
# 处理所有数据：用 --flight_status 0 一次处理全部飞行状态
# 已存在的 CSV 会被覆盖（不影响结果），缺失的会被生成

set -e
cd /home/jack/Data_processing_tools
source /home/jack/miniconda3/bin/activate UAV-Fault-Diagnosis

ORIGINAL=/home/jack/UAV-Fault-Diagnosis/data
OUTPUT=/home/jack/UAV-Fault-Diagnosis/data/processed
KEEP_PATTERNS="actuator_armed\|actuator_outputs\|estimator_attitude\|estimator_status\|sensor_combined\|vehicle_command\|vehicle_global_position\|vehicle_gps_position\|vehicle_magnetometer"

run_one() {
    local sub=$1
    local fault=$2
    local pname="SIL"
    [ "$sub" -eq 2 ] && pname="HIL"
    [ "$sub" -eq 3 ] && pname="Real"

    echo ""
    echo ">>> $pname fault=$fault (flight_status=0, 全部飞行状态)"
    timeout 600s python Rflytool_main.py \
        --original_path "$ORIGINAL" \
        --restore_path "$OUTPUT" \
        --sub_dataset $sub \
        --fault_type $fault \
        --flight_status 0 \
        --trans_freq 50 2>&1 | tail -3 || true
    echo "当前进度: $(ls "$OUTPUT"/*.csv 2>/dev/null | wc -l) / 4984"
}

cleanup() {
    local platform=$1
    echo "清理 $platform 的临时 CSV..."
    find "$ORIGINAL/$platform" -path "*/Log/*.csv" -type f | grep -v "$KEEP_PATTERNS" | xargs rm -f 2>/dev/null || true
    if [ "$platform" = "Real" ]; then
        find "$ORIGINAL/$platform" -path "*/rfly_real_*/*.csv" -type f | xargs rm -f 2>/dev/null || true
    fi
    df -h /home/jack --output=avail,pcent | tail -1
}

# SIL: motor(1) propeller(2) sensors(6-10) no_fault(11)
echo "========== SIL =========="
for fault in 1 2 6 7 8 9 10 11; do
    run_one 1 $fault
done
cleanup SIL

# HIL: motor(1) propeller(2) sensors(6-10) no_fault(11)
echo "========== HIL =========="
for fault in 1 2 6 7 8 9 10 11; do
    run_one 2 $fault
done
cleanup HIL

# Real: motor(1) sensors(6-10) no_fault(11) (无 propeller)
echo "========== Real =========="
for fault in 1 6 7 8 9 10 11; do
    run_one 3 $fault
done
cleanup Real

echo ""
echo "========================================"
echo "全部完成！"
csv_count=$(ls "$OUTPUT"/*.csv 2>/dev/null | wc -l)
echo "CSV 文件数: $csv_count / 4984"
df -h /home/jack --output=avail,pcent | tail -1
echo "========================================"
