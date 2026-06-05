#!/bin/bash
# 批量处理 RflyMAD 数据集（完整 48 列，50Hz）
# 故障类型：motor(1) propeller(2) accelerometer(6) gyroscope(7) magnetometer(8) barometer(9) GPS(10) no_fault(11)
# 平台：SIL(1) HIL(2) Real(3)
# 输出到 data/processed/，不污染原始数据

set -e
cd /home/jack/Data_processing_tools
source /home/jack/miniconda3/bin/activate UAV-Fault-Diagnosis

ORIGINAL=/home/jack/UAV-Fault-Diagnosis/data
OUTPUT=/home/jack/UAV-Fault-Diagnosis/data/processed
mkdir -p "$OUTPUT"

# 我们需要的故障类型
FAULTS="1 2 6 7 8 9 10 11"
# motor propeller accelerometer gyroscope magnetometer barometer GPS no_fault

FAULT_NAMES=("" "motor" "propeller" "" "" "" "accelerometer" "gyroscope" "magnetometer" "barometer" "GPS" "no_fault")

total_success=0
total_fail=0

for platform in 1 2 3; do
    case $platform in
        1) pname="SIL" ;;
        2) pname="HIL" ;;
        3) pname="Real" ;;
    esac

    for fault in $FAULTS; do
        fname=${FAULT_NAMES[$fault]}

        # Real 平台没有 propeller 数据
        if [ "$platform" -eq 3 ] && [ "$fault" -eq 2 ]; then
            echo "[跳过] Real/propeller - 无数据"
            continue
        fi

        echo ""
        echo "========================================"
        echo "处理 $pname / $fname (sub=$platform fault=$fault)"
        echo "========================================"

        python Rflytool_main.py \
            --original_path "$ORIGINAL" \
            --restore_path "$OUTPUT" \
            --sub_dataset $platform \
            --fault_type $fault \
            --flight_status 0 \
            --trans_freq 50 2>&1 | tail -3

        if [ $? -eq 0 ]; then
            ((total_success++))
        else
            echo "[错误] $pname/$fname 处理失败"
            ((total_fail++))
        fi
    done
done

echo ""
echo "========================================"
echo "全部完成！成功: $total_success, 失败: $total_fail"
csv_count=$(ls "$OUTPUT"/*.csv 2>/dev/null | wc -l)
echo "CSV 文件数: $csv_count"
if [ "$csv_count" -gt 0 ]; then
    col_count=$(head -1 "$(ls "$OUTPUT"/*.csv | head -1)" | tr ',' '\n' | wc -l)
    echo "每文件列数: $col_count"
fi
echo "========================================"
