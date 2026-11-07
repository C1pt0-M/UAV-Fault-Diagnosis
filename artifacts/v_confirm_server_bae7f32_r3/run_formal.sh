#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/autodl-tmp/UAV-Fault-Diagnosis
OUT=/root/autodl-tmp/v_confirm_upload_bae7f32_r3
CONTROL=$ROOT/artifacts/v_confirm_fixed100_mstca_v1/formal_sequence_r3
PY=$ROOT/artifacts/s_ablation_mstca_v1/runtime_pytest_r1/bin/python
DATA=/root/autodl-tmp/causal_source_v1_20261031/unpacked/data
MAN=$ROOT/artifacts/v_confirm_fixed100_mstca_v1/preflight_r3/implementation_manifest.json
export CUBLAS_WORKSPACE_CONFIG=:4096:8 ATEN_CPU_CAPABILITY=default PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
stop_on_error() {
    local code="$1" line="$2" command="$3"
    trap - ERR
    printf 'FAILED exit=%s line=%s command=%s time=%s\n' "$code" "$line" "$command" "$(date -u +%FT%TZ)" > "$CONTROL/FAILED.txt"
    exit "$code"
}
trap 'stop_on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
MAN_SHA=$(sha256sum "$MAN")
MAN_SHA=${MAN_SHA%% *}
test "$MAN_SHA" = 09fdac3e1cc6e1433af770a0e9ae4281d482dcf91af6305c36f7c301bdaee858
test ! -e "$CONTROL/COMPLETE.txt"
test ! -e "$CONTROL/FAILED.txt"
printf 'START %s head=%s manifest_raw_sha256=%s\n' "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$MAN_SHA"
for S in 42 43 44 45 46; do
    test ! -e "$ROOT/artifacts/v_confirm_fixed100_mstca_v1/val/seed$S"
    test ! -e "$CONTROL/logs/seed$S.log"
    printf 'SEED_START %s %s\n' "$S" "$(date -u +%FT%TZ)"
    "$PY" -I -B scripts/research/train_v_confirm.py --data "$DATA" --seed "$S" \
        --output "$ROOT/artifacts/v_confirm_fixed100_mstca_v1/val/seed$S" \
        --implementation-manifest "$MAN" --implementation-manifest-sha256 "$MAN_SHA" \
        > "$CONTROL/logs/seed$S.log" 2>&1
    "$PY" -I -B "$OUT/control.py" verify-seed "$S" > "$CONTROL/logs/seed${S}_verify.log" 2>&1
    printf 'SEED_COMPLETE %s %s\n' "$S" "$(date -u +%FT%TZ)"
done
test ! -e "$ROOT/artifacts/v_confirm_fixed100_mstca_v1/evaluation_val"
printf 'EVALUATION_START %s\n' "$(date -u +%FT%TZ)"
"$PY" -I -B scripts/research/evaluate_v_confirm.py --data "$DATA" \
    --runs "$ROOT/artifacts/v_confirm_fixed100_mstca_v1/val" \
    --output "$ROOT/artifacts/v_confirm_fixed100_mstca_v1/evaluation_val" \
    > "$CONTROL/logs/evaluation.log" 2>&1
"$PY" -I -B "$OUT/control.py" complete > "$CONTROL/logs/completion_verify.log" 2>&1
printf 'COMPLETE %s\n' "$(date -u +%FT%TZ)" > "$CONTROL/COMPLETE.txt"
printf 'COMPLETE %s\n' "$(date -u +%FT%TZ)"
