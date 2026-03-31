#!/bin/bash
# 只处理缺失的部分：SIL 传感器 + HIL 部分 + Real 部分

set -e
cd /home/jack/Data_processing_tools
source /home/jack/miniconda3/bin/activate UAV-Fault-Diagnosis

ORIGINAL=/home/jack/UAV-Fault-Diagnosis/data
OUTPUT=/home/jack/UAV-Fault-Diagnosis/data/processed
KEEP_PATTERNS="actuator_armed\|actuator_outputs\|estimator_attitude\|estimator_status\|sensor_combined\|vehicle_command\|vehicle_global_position\|vehicle_gps_position\|vehicle_magnetometer"

FAULT_NAMES=(" " "motor" "propeller" " " " " " " "accelerometer" "gyroscope" "magnetometer" "barometer" "GPS")
MANEUVERS="1 2 3 4 5 6"
MANEUVER_NAMES=(" " "hover" "acce" "dece" "circling" "velocity" "waypoint")

run_one() {
    local sub=$1
    local fault=$2
    local flight=$3
    local fname=${FAULT_NAMES[$fault]}
    [ "$sub" -eq 3 ] && [ "$fault" -eq 11 ] && fname="no_fault"
    local mname=${MANEUVER_NAMES[$flight]}
    local pname="SIL"
    [ "$sub" -eq 2 ] && pname="HIL"
    [ "$sub" -eq 3 ] && pname="Real"

    echo ""
    echo "========================================"
    echo "处理 $pname / $fname / $mname"
    echo "  fault=$fault  flight_status=$flight"
    echo "========================================"
    timeout 300s python Rflytool_main.py \
        --original_path "$ORIGINAL" \
        --restore_path "$OUTPUT" \
        --sub_dataset $sub \
        --fault_type $fault \
        --flight_status $flight \
        --trans_freq 50 2>&1 | tail -3 || true
    echo "当前进度: $(ls "$OUTPUT"/*.csv 2>/dev/null | wc -l) / 4984"
}

# ----------------------- SIL 传感器 (6-10) ---------------------
echo "========================================"
echo "开始处理 SIL 传感器 (缺失部分)"
echo "========================================"
FAULTS_SIL_SENSORS="6 7 8 9 10"
for fault in $FAULTS_SIL_SENSORS; do
    for flight in $MANEUVERS; do
        run_one 1 $fault $flight
    done
    fname=${FAULT_NAMES[$fault]}
    echo "清理 SIL/$fname 的 CSV..."
    find "$ORIGINAL/SIL" -path "*/$fname/*/Log/*.csv" -type f | grep -v "$KEEP_PATTERNS" | xargs rm -f 2>/dev/null || true
    df -h /home/jack --output=avail,pcent | tail -1
done

# ----------------------- HIL 缺失部分 -------------------------
echo ""
echo "========================================"
echo "开始处理 HIL 缺失部分"
echo "========================================"
# HIL 已有 2076，应该有 2205，缺 129
# 需要补充处理所有 fault_type
FAULTS_HIL="1 2 6 7 8 9 10 11"
for fault in $FAULTS_HIL; do
    for flight in $MANEUVERS; do
        run_one 2 $fault $flight
    done
    fname=${FAULT_NAMES[$fault]}
    [ "$fault" -eq 11 ] && fname="no_fault"
    echo "清理 HIL/$fname 的 CSV..."
    find "$ORIGINAL/HIL" -path "*/$fname/*/Log/*.csv" -type f | grep -v "$KEEP_PATTERNS" | xargs rm -f 2>/dev/null || true
    df -h /home/jack --output=avail,pcent | tail -1
done

# ----------------------- Real 缺失部分 ------------------------
echo ""
echo "========================================"
echo "开始处理 Real 缺失部分"
echo "========================================"
# Real 已有 376，应该有 493，缺 117
FAULTS_REAL="11 1 6 7 8 9 10"
for fault in $FAULTS_REAL; do
    for flight in $MANEUVERS; do
        run_one 3 $fault $flight
    done
    fname=${FAULT_NAMES[$fault]}
    [ "$fault" -eq 11 ] && fname="no_fault"
    echo "清理 Real/$fname 的 CSV..."
    find "$ORIGINAL/Real" -path "*/$fname/*/Log/*.csv" -type f | grep -v "$KEEP_PATTERNS" | xargs rm -f 2>/dev/null || true
    find "$ORIGINAL/Real" -path "*/$fname/*/rfly_real_*/*.csv" -type f | xargs rm -f 2>/dev/null || true
    df -h /home/jack --output=avail,pcent | tail -1
done

echo ""
echo "========================================"
echo "全部完成！"
csv_count=$(ls "$OUTPUT"/*.csv 2>/dev/null | wc -l)
echo "CSV 文件数: $csv_count / 4984"
df -h /home/jack --output=avail,pcent | tail -1
echo "========================================"
