from pathlib import Path

import pytest


pytestmark = pytest.mark.external


def test_process_rflymad_v2_script_exists_and_uses_official_ids_with_50hz_override():
    script_path = Path("scripts/process_rflymad_v2.sh")

    assert script_path.exists()

    content = script_path.read_text()
    assert 'ORIGINAL="/home/jack/RflyMAD"' in content
    assert (
        'OUTPUT="/home/jack/UAV-Fault-Diagnosis/data/RflyMAD/processed_official_50hz"'
        in content
    )
    assert (
        'TOOL_ROOT="/home/jack/UAV-Fault-Diagnosis/vendor/Data_processing_tools_linux"'
        in content
    )

    # 官方 sub_dataset: 1=SIL, 2=HIL, 3=Real
    assert "run_platform 1; run_platform 2; run_platform 3" in content

    # SIL/HIL 全 fault，按官方 fault_type 编号
    assert 'default_faults="1 2 3 4 5 6 7 8 9 10 11"' in content

    # Real 只处理当前目录中实际存在的 fault
    assert 'default_faults="1 6 7 8 9 10 11"' in content

    # 默认 flight_status=0，可被定向补跑覆盖
    assert 'FLIGHT_FILTER="${3:-0}"' in content
    assert '--flight_status "$FLIGHT_FILTER"' in content

    # 项目主线覆盖官方默认 20Hz，固定 50Hz
    assert 'TRANS_FREQ="50"' in content
    assert '--trans_freq "$TRANS_FREQ"' in content

    # 跑前创建输出目录
    assert 'mkdir -p "$OUTPUT"' in content


def test_process_rflymad_v2_script_supports_running_single_platform():
    content = Path("scripts/process_rflymad_v2.sh").read_text()

    assert 'TARGET_PLATFORM="${1:-all}"' in content
    assert 'case "$TARGET_PLATFORM" in' in content
    assert "SIL) run_platform 1 ;;" in content
    assert "HIL) run_platform 2 ;;" in content
    assert "Real) run_platform 3 ;;" in content
    assert "all) run_platform 1" in content


def test_process_rflymad_v2_script_logs_failures_and_continues():
    content = Path("scripts/process_rflymad_v2.sh").read_text()

    assert 'FAILED_LOG="$OUTPUT/failed_cases.log"' in content
    assert ': > "$FAILED_LOG"' in content
    assert 'if ! "$PYTHON_BIN" "$TOOL_ROOT/Rflytool_main.py"' in content
    assert (
        'echo "platform=$pname fault=$fault flight_status=$FLIGHT_FILTER" >> "$FAILED_LOG"'
        in content
    )
    assert "return 0" in content


def test_process_rflymad_v2_script_supports_fault_list_resume():
    content = Path("scripts/process_rflymad_v2.sh").read_text()

    assert 'FAULT_FILTER="${2:-}"' in content
    assert 'if [ -n "$FAULT_FILTER" ]; then' in content
    assert 'FAULTS_TO_RUN="$FAULT_FILTER"' in content
    assert 'FAULTS_TO_RUN="$default_faults"' in content


def test_process_rflymad_v2_script_supports_flight_mode_filter_and_output_override():
    content = Path("scripts/process_rflymad_v2.sh").read_text()

    assert 'FLIGHT_FILTER="${3:-0}"' in content
    assert 'OUTPUT_OVERRIDE="${4:-}"' in content
    assert 'if [ -n "$OUTPUT_OVERRIDE" ]; then' in content
    assert 'OUTPUT="$OUTPUT_OVERRIDE"' in content
    assert '--flight_status "$FLIGHT_FILTER"' in content
