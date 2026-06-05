#!/bin/bash 
 
set -euo pipefail 
 
TOOL_ROOT="/home/jack/UAV-Fault-Diagnosis/vendor/Data_processing_tools_linux" 
ORIGINAL="/home/jack/RflyMAD" 
OUTPUT="/home/jack/UAV-Fault-Diagnosis/data/RflyMAD/processed_official_50hz" 
SOURCE_ROOT="${SOURCE_ROOT:-$ORIGINAL}" 
WORK_ROOT="${WORK_ROOT:-$SOURCE_ROOT}" 
LOG_ROOT="${LOG_ROOT:-/home/jack/UAV-Fault-Diagnosis/data/RflyMAD/logs}" 
TRANS_FREQ="50" 
TARGET_PLATFORM="${1:-all}" 
FAULT_FILTER="${2:-}" 
FLIGHT_FILTER="${3:-0}" 
OUTPUT_OVERRIDE="${4:-}" 
 
if [ -n "$OUTPUT_OVERRIDE" ]; then 
    OUTPUT="$OUTPUT_OVERRIDE" 
fi 
 
BUILD_BATCH_ID="${BUILD_BATCH_ID:-$(date +%Y%m%d_%H%M%S)}" 
FAILED_LOG="${FAILED_LOG:-$LOG_ROOT/failed_cases.log}" 
CASE_MAPPING_PATH="${CASE_MAPPING_PATH:-$LOG_ROOT/case_mapping.csv}" 
 
mkdir -p "$OUTPUT" 
mkdir -p "$LOG_ROOT" 
: > "$FAILED_LOG" 
: > "$CASE_MAPPING_PATH" 

run_one() { 
    local sub=$1 
    local fault=$2 
    local pname="SIL" 
    if [ "$sub" -eq 2 ]; then 
        pname="HIL" 
    fi 
    if [ "$sub" -eq 3 ]; then 
        pname="Real" 
    fi 
    echo "" 
    echo "========================================" 
    echo "platform=$pname fault=$fault flight_status=$FLIGHT_FILTER freq=${TRANS_FREQ}Hz" 
    echo "source_raw=$SOURCE_ROOT" 
    echo "working_raw=$WORK_ROOT" 
    echo "processed_output=$OUTPUT" 
    echo "log_root=$LOG_ROOT" 
    echo "build_batch_id=$BUILD_BATCH_ID" 
    echo "========================================" 
    if ! "$PYTHON_BIN" "$TOOL_ROOT/Rflytool_main.py" --original_path "$WORK_ROOT" --restore_path "$OUTPUT" --sub_dataset "$sub" --fault_type "$fault" --flight_status "$FLIGHT_FILTER" --trans_freq "$TRANS_FREQ" --case_mapping_path "$CASE_MAPPING_PATH" --build_batch_id "$BUILD_BATCH_ID"; then 
        echo "platform=$pname fault=$fault flight_status=$FLIGHT_FILTER working_root=$WORK_ROOT case_mapping=$CASE_MAPPING_PATH" >> "$FAILED_LOG" 
        return 0
    fi 
} 
 
run_platform() { 
    local sub=$1 
    local default_faults="" 
    if [ "$sub" -eq 3 ]; then 
        default_faults="1 6 7 8 9 10 11" 
    else 
        default_faults="1 2 3 4 5 6 7 8 9 10 11" 
    fi 
    local FAULTS_TO_RUN="" 
    if [ -n "$FAULT_FILTER" ]; then 
        FAULTS_TO_RUN="$FAULT_FILTER" 
    else 
        FAULTS_TO_RUN="$default_faults" 
    fi 
    for fault in $FAULTS_TO_RUN; do 
        run_one "$sub" "$fault" 
    done 
} 


if [ -x "/home/jack/miniconda3/envs/UAV-Fault-Diagnosis/bin/python" ]; then
    PYTHON_BIN="/home/jack/miniconda3/envs/UAV-Fault-Diagnosis/bin/python"
else
    PYTHON_BIN="python3"
fi

case "$TARGET_PLATFORM" in
    SIL) run_platform 1 ;;
    HIL) run_platform 2 ;;
    Real) run_platform 3 ;;
    all) run_platform 1; run_platform 2; run_platform 3 ;;
    *)
        echo "Usage: bash scripts/process_rflymad_v2.sh [SIL|HIL|Real|all]"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "done: processed output -> $OUTPUT"
echo "done: case mapping -> $CASE_MAPPING_PATH"
echo "done: failure log -> $FAILED_LOG"
echo "========================================"
