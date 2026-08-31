#!/usr/bin/env bash
# Wait for the four frozen physical GPUs, then generate, audit, and score HPSv2.
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
GEN_PYTHON=/home/bycao/miniforge3/envs/diff_attn/bin/python
EVAL_PYTHON=/home/bycao/miniforge3/envs/repldm_eval/bin/python
RUNNER=$ROOT/eval-pipeline/generate_hpsv2_relational_renderer.py
SCORER=$ROOT/eval-pipeline/score_hpsv2_relational_renderer.py
ANALYZER=$ROOT/eval-pipeline/analyze_hpsv2_relational_renderer.py
SCORING_CONFIG=$ROOT/eval-pipeline/configs/hpsv2_full_scoring_v1.yaml
RUN_DIR=$ROOT/outputs/hpsv2_relational_renderer/full_v1
QUEUE_DIR=$ROOT/outputs/hpsv2_relational_renderer/queue_v1
STATE_PATH=$QUEUE_DIR/state.json
LOG_PATH=$QUEUE_DIR/queue.log
QUEUE_LOCK=$QUEUE_DIR/queue.lock
DEVICES=2,5,6,7
MIN_FREE_MIB=22000
MAX_UTILIZATION=5
POLL_SECONDS=30
REQUIRED_STABLE_POLLS=2
GPU_QUERY_TIMEOUT_SECONDS=10
LAUNCH_HEAD=

if [ "${1:-}" = "--status" ]; then
    if [ -f "$STATE_PATH" ]; then
        cat "$STATE_PATH"
    else
        printf '{"status":"not_started"}\n'
    fi
    exit 0
fi
if [ "$#" -ne 0 ]; then
    echo "usage: $0 [--status]" >&2
    exit 2
fi

mkdir -p "$QUEUE_DIR"
exec 9>"$QUEUE_LOCK"
if ! flock -n 9; then
    echo "another HPSv2 queue owns $QUEUE_LOCK" >&2
    exit 75
fi
exec > >(tee -a "$LOG_PATH") 2>&1

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

validate_repository_snapshot() {
    if ! status=$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all); then
        log "cannot read the repository worktree state"
        return 1
    fi
    if [ -n "$status" ]; then
        log "repository worktree changed after review; refusing the next stage"
        return 1
    fi
    if ! branch=$(git -C "$ROOT" symbolic-ref --short -q HEAD); then
        log "repository HEAD is detached"
        return 1
    fi
    if [ "$branch" != "rl-version" ]; then
        log "repository branch changed from rl-version to $branch"
        return 1
    fi
    if ! head=$(git -C "$ROOT" rev-parse HEAD); then
        log "cannot resolve repository HEAD"
        return 1
    fi
    if ! remote=$(git -C "$ROOT" rev-parse refs/remotes/origin/rl-version); then
        log "cannot resolve origin/rl-version"
        return 1
    fi
    if [ "$head" != "$remote" ]; then
        log "repository HEAD is not the pushed origin/rl-version commit"
        return 1
    fi
    if [ -n "$LAUNCH_HEAD" ] && [ "$head" != "$LAUNCH_HEAD" ]; then
        log "repository HEAD changed after this queue started"
        return 1
    fi
    for relative in \
      eval-pipeline/generate_hpsv2_relational_renderer.py \
      eval-pipeline/score_hpsv2_relational_renderer.py \
      eval-pipeline/analyze_hpsv2_relational_renderer.py \
      eval-pipeline/run_hpsv2_relational_renderer_queue.sh \
      eval-pipeline/configs/hpsv2_relational_renderer_full_v1.yaml \
      eval-pipeline/configs/hpsv2_full_scoring_v1.yaml \
      eval-pipeline/prompts/hpsv2_official_3200.csv \
      eval-pipeline/prompts/hpsv2_official_3200_manifest.json; do
        if ! tracked=$(git -C "$ROOT" ls-files --error-unmatch "$relative" 2>/dev/null); then
            log "queue input is not tracked at HEAD: $relative"
            return 1
        fi
        if [ "$tracked" != "$relative" ]; then
            log "queue input resolved ambiguously: $relative"
            return 1
        fi
    done
    if [ -z "$LAUNCH_HEAD" ]; then
        LAUNCH_HEAD=$head
    fi
}

WAITING_SINCE=$(timestamp)
if [ -f "$STATE_PATH" ]; then
    RECORDED_WAIT=$(jq -r '.waiting_since // empty' "$STATE_PATH" 2>/dev/null || true)
    [ -z "$RECORDED_WAIT" ] || WAITING_SINCE=$RECORDED_WAIT
fi

write_state() {
    status=$1
    stage=$2
    snapshot=${3:-}
    stable_polls=${4:-0}
    temporary=$STATE_PATH.tmp.$$
    jq -n \
      --arg status "$status" \
      --arg stage "$stage" \
      --arg updated_at "$(timestamp)" \
      --arg waiting_since "$WAITING_SINCE" \
      --arg devices "$DEVICES" \
      --arg launch_head "$LAUNCH_HEAD" \
      --arg snapshot "$snapshot" \
      --argjson min_free_mib "$MIN_FREE_MIB" \
      --argjson max_utilization "$MAX_UTILIZATION" \
      --argjson stable_polls "$stable_polls" \
      --argjson pid "$$" \
      '{status:$status,stage:$stage,updated_at:$updated_at,
        waiting_since:$waiting_since,devices:$devices,pid:$pid,
        launch_head:$launch_head,
        min_free_mib:$min_free_mib,max_utilization:$max_utilization,
        stable_polls:$stable_polls,gpu_snapshot:$snapshot}' >"$temporary"
    mv "$temporary" "$STATE_PATH"
}

gpu_snapshot() {
    timeout --foreground "$GPU_QUERY_TIMEOUT_SECONDS" nvidia-smi \
      --query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
}

compute_snapshot() {
    timeout --foreground "$GPU_QUERY_TIMEOUT_SECONDS" nvidia-smi \
      --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits
}

devices_ready() {
    snapshot=$1
    processes=$2
    shift 2
    for device in "$@"; do
        line=$(printf '%s\n' "$snapshot" | awk -F',' -v target="$device" '
          { for (i=1; i<=NF; i++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i) }
          $1 == target { print $2 " " $5 " " $6 }
        ')
        [ -n "$line" ] || return 1
        read -r gpu_uuid free_mib utilization <<EOF
$line
EOF
        [ "$free_mib" -ge "$MIN_FREE_MIB" ] || return 1
        [ "$utilization" -le "$MAX_UTILIZATION" ] || return 1
        if printf '%s\n' "$processes" | awk -F',' -v target="$gpu_uuid" '
          { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1) }
          $1 == target { found=1 }
          END { exit(found ? 0 : 1) }
        '; then
            return 1
        fi
    done
}

wait_for_devices() {
    waiting_status=$1
    waiting_stage=$2
    shift 2
    stable=0
    log "waiting for physical GPUs $* with no compute processes"
    while true; do
        if ! snapshot=$(gpu_snapshot 2>&1); then
            stable=0
            log "nvidia-smi GPU query failed; keeping the queue in wait state: $snapshot"
            write_state "$waiting_status" "$waiting_stage" \
              "nvidia_smi_gpu_error: $snapshot" "$stable"
            sleep "$POLL_SECONDS"
            continue
        fi
        if ! processes=$(compute_snapshot 2>&1); then
            stable=0
            log "nvidia-smi process query failed; keeping the queue in wait state: $processes"
            write_state "$waiting_status" "$waiting_stage" \
              "nvidia_smi_process_error: $processes" "$stable"
            sleep "$POLL_SECONDS"
            continue
        fi
        state_snapshot=$(printf 'gpu_inventory:\n%s\ncompute_processes:\n%s' \
          "$snapshot" "$processes")
        if devices_ready "$snapshot" "$processes" "$@"; then
            stable=$((stable + 1))
        else
            stable=0
        fi
        write_state "$waiting_status" "$waiting_stage" "$state_snapshot" "$stable"
        if [ "$stable" -ge "$REQUIRED_STABLE_POLLS" ]; then
            log "physical GPUs $* passed the safety threshold for $stable polls"
            return 0
        fi
        sleep "$POLL_SECONDS"
    done
}

require_devices_ready_now() {
    if ! snapshot=$(gpu_snapshot 2>&1); then
        log "final GPU query failed; returning to wait: $snapshot"
        return 1
    fi
    if ! processes=$(compute_snapshot 2>&1); then
        log "final process query failed; returning to wait: $processes"
        return 1
    fi
    devices_ready "$snapshot" "$processes" "$@"
}

fail_stage() {
    stage=$1
    code=$2
    write_state failed "$stage" "exit_code=$code" 0
    log "$stage failed with exit code $code; downstream stages were not run"
    exit "$code"
}

require_repository_snapshot_or_fail() {
    stage=$1
    if ! validate_repository_snapshot; then
        fail_stage "$stage" 1
    fi
}

unexpected_failure() {
    code=$1
    command=$2
    trap - ERR
    write_state failed unexpected_queue_error "exit_code=$code command=$command" 0 \
      || true
    log "unexpected queue error (exit $code): $command"
    exit "$code"
}
trap 'unexpected_failure "$?" "$BASH_COMMAND"' ERR

require_repository_snapshot_or_fail initial_repository_snapshot
wait_for_devices waiting_for_all_gpus gpu_wait 2 5 6 7
while ! require_devices_ready_now 2 5 6 7; do
    log "GPU state changed before launch; returning to the stable-wait gate"
    wait_for_devices waiting_for_all_gpus gpu_wait 2 5 6 7
done
require_repository_snapshot_or_fail pre_generation_repository_snapshot
write_state running generation "" "$REQUIRED_STABLE_POLLS"
log "starting frozen 12,800-image generation"
env -u CUDA_VISIBLE_DEVICES -u CUDA_DEVICE_ORDER \
  "$GEN_PYTHON" "$RUNNER" --devices "$DEVICES" || fail_stage generation "$?"

require_repository_snapshot_or_fail pre_audit_repository_snapshot
write_state running complete_run_audit "" 0
log "generation returned successfully; auditing every PNG, sidecar, and manifest row"
env -u CUDA_VISIBLE_DEVICES -u CUDA_DEVICE_ORDER \
  "$GEN_PYTHON" "$RUNNER" --devices "$DEVICES" --audit-only \
  || fail_stage complete_run_audit "$?"

require_repository_snapshot_or_fail pre_scoring_wait_repository_snapshot
log "complete-run audit passed; waiting for physical cuda:2 before strict scoring"
wait_for_devices waiting_for_scoring_gpu scoring_gpu_wait 2
while ! require_devices_ready_now 2; do
    log "cuda:2 state changed before scoring; returning to the stable-wait gate"
    wait_for_devices waiting_for_scoring_gpu scoring_gpu_wait 2
done
require_repository_snapshot_or_fail pre_scoring_repository_snapshot
write_state running scoring "" "$REQUIRED_STABLE_POLLS"
log "starting strict HPSv2 scoring on physical cuda:2"
env -u CUDA_VISIBLE_DEVICES -u CUDA_DEVICE_ORDER \
  "$EVAL_PYTHON" "$SCORER" \
    --run_dir "$RUN_DIR" \
    --config "$SCORING_CONFIG" \
    --device cuda:2 \
    --strict \
    --require-scorer-provenance \
    --require-exclusive-gpu \
  || fail_stage scoring "$?"

require_repository_snapshot_or_fail pre_analysis_repository_snapshot
write_state running paired_analysis "" 0
log "strict scoring finished; starting frozen style/group/paired analysis"
"$EVAL_PYTHON" "$ANALYZER" --run-dir "$RUN_DIR" \
  || fail_stage paired_analysis "$?"

write_state complete analysis_complete "" 0
log "generation, complete-run audit, strict scoring, and paired analysis finished"
