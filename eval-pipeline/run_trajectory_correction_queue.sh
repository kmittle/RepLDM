#!/usr/bin/env bash
# Registered S7 development -> selector -> conditional validation queue.
# No renderer/RL stage is present, and no external process is signaled.
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
GEN_PYTHON=$(printenv S7_GEN_PYTHON || true)
EVAL_PYTHON=$(printenv S7_EVAL_PYTHON || true)
QUEUE_DIR=$(printenv S7_QUEUE_DIR || true)
DEV_RUN_DIR=$(printenv S7_DEV_RUN_DIR || true)
VAL_RUN_DIR=$(printenv S7_VAL_RUN_DIR || true)
VAL_ACTIONS=$(printenv S7_VAL_ACTIONS || true)
MIN_FREE_MIB=$(printenv S7_MIN_FREE_MIB || true)
POLL_SECONDS=$(printenv S7_POLL_SECONDS || true)
DEVICES=$(printenv S7_DEVICES || true)
[ -n "$GEN_PYTHON" ] || GEN_PYTHON=/home/bycao/miniforge3/envs/diff_attn/bin/python
[ -n "$EVAL_PYTHON" ] || EVAL_PYTHON=/home/bycao/miniforge3/envs/repldm_eval/bin/python
[ -n "$QUEUE_DIR" ] || QUEUE_DIR=$ROOT/outputs/trajectory_correction/queue_v1
[ -n "$DEV_RUN_DIR" ] || DEV_RUN_DIR=$ROOT/outputs/trajectory_correction/development_v2
[ -n "$VAL_RUN_DIR" ] || VAL_RUN_DIR=$ROOT/outputs/trajectory_correction/validation_v1
[ -n "$VAL_ACTIONS" ] || VAL_ACTIONS=$ROOT/eval-pipeline/configs/trajectory_correction_validation_v1.yaml
[ -n "$MIN_FREE_MIB" ] || MIN_FREE_MIB=22000
[ -n "$POLL_SECONDS" ] || POLL_SECONDS=30

DEV_ACTIONS=$ROOT/eval-pipeline/configs/trajectory_correction_development.yaml
VAL_TEMPLATE=$ROOT/eval-pipeline/configs/trajectory_correction_validation_template.yaml
DEV_PROMPTS=$ROOT/eval-pipeline/prompts/trajectory_correction_heldout_v1.csv
VAL_PROMPTS=$ROOT/eval-pipeline/prompts/trajectory_correction_validation_v1.csv
# 11 development prompt rows x 2 seeds x 7 registered actions.
EXPECTED_DEV_TASKS=154
# 44 validation prompt rows x 3 confirmation seeds x (baseline + native
# reference + one selected correction). The validation loader filters the
# seven-entry template to those three actions after freezing selected_action.
EXPECTED_VAL_TASKS=396
DRY_RUN=0
STATUS_ONLY=0
GPU=
SELECTED_ACTION=
CURRENT_STAGE=initializing
NULL_WRITTEN=0

usage() {
    cat <<'USAGE'
Usage: run_trajectory_correction_queue.sh [options]
  --dry-run              Exercise state/branch logic without GPU work.
                         S7_DRY_RUN_SELECTION=ancestral_mix_050 exercises pass.
  --status               Print current state and exit.
  --queue-dir PATH       Override state/log directory.
  --min-free-mib N       Required free VRAM (default 22000).
  --poll-seconds N       GPU/process poll interval (default 30).
  --devices LIST         Restrict GPU selection to comma-separated indices.
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --status) STATUS_ONLY=1 ;;
        --queue-dir)
            shift; [ "$#" -gt 0 ] || { echo "--queue-dir needs a path" >&2; exit 2; }; QUEUE_DIR=$1 ;;
        --min-free-mib)
            shift; [ "$#" -gt 0 ] || { echo "--min-free-mib needs a number" >&2; exit 2; }; MIN_FREE_MIB=$1 ;;
        --poll-seconds)
            shift; [ "$#" -gt 0 ] || { echo "--poll-seconds needs a number" >&2; exit 2; }; POLL_SECONDS=$1 ;;
        --devices)
            shift; [ "$#" -gt 0 ] || { echo "--devices needs a list" >&2; exit 2; }; DEVICES=$1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if ! [[ "$MIN_FREE_MIB" =~ ^[0-9]+$ ]] || [ "$MIN_FREE_MIB" -le 0 ]; then
    echo "min-free-mib must be a positive integer" >&2; exit 2
fi
if ! [[ "$POLL_SECONDS" =~ ^[0-9]+$ ]] || [ "$POLL_SECONDS" -le 0 ]; then
    echo "poll-seconds must be a positive integer" >&2; exit 2
fi
command -v jq >/dev/null 2>&1 || { echo "jq is required" >&2; exit 2; }
command -v flock >/dev/null 2>&1 || { echo "flock is required" >&2; exit 2; }

STATE_PATH=$QUEUE_DIR/state.json
AUDIT_PATH=$QUEUE_DIR/inputs.json
NULL_ROUTE_PATH=$QUEUE_DIR/null_route.json
LOG_PATH=$QUEUE_DIR/queue.log
LOCK_PATH=$QUEUE_DIR/queue.lock

if [ "$STATUS_ONLY" -eq 1 ]; then
    if [ -f "$STATE_PATH" ]; then cat "$STATE_PATH"; else
        printf '{"status":"not_started","queue_dir":"%s"}\n' "$QUEUE_DIR"
    fi
    exit 0
fi

mkdir -p "$QUEUE_DIR"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    echo "another S7 queue instance owns $LOCK_PATH" >&2; exit 75
fi
exec > >(tee -a "$LOG_PATH") 2>&1

log() { printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
require_file() { [ -f "$1" ] || { log "missing file: $1"; return 1; }; }
state_value() { [ -f "$STATE_PATH" ] && jq -r --arg k "$1" '.[$k] // empty' "$STATE_PATH"; }

registered_task_count() {
    prompts=$1; actions=$2; seed_count=$3
    prompt_count=$(awk 'NR > 1 && NF { n++ } END { print n + 0 }' "$prompts")
    action_count=$(awk '/^[[:space:]]*-[[:space:]]*id:[[:space:]]*/ { n++ } END { print n + 0 }' "$actions")
    printf '%s\n' "$((prompt_count * seed_count * action_count))"
}

validation_action_count() {
    config=$1
    "$EVAL_PYTHON" - "$config" <<'PY'
import sys

import yaml


with open(sys.argv[1]) as handle:
    config = yaml.safe_load(handle) or {}
if config.get("schema") != "trajectory_correction_validation_v1":
    raise SystemExit("validation config has the wrong schema")
actions = config.get("actions")
if not isinstance(actions, list) or not actions:
    raise SystemExit("validation config must contain a non-empty actions list")
if any(not isinstance(action, dict) for action in actions):
    raise SystemExit("validation config actions must be mappings")

ids = [str(action.get("id", "")) for action in actions]
if any(not action_id for action_id in ids) or len(ids) != len(set(ids)):
    raise SystemExit("validation config actions must have unique non-empty ids")
if ids.count("no_correction") != 1:
    raise SystemExit("validation config must register exactly one no_correction action")

# Mirror generate.load_actions(): references remain visible even though they
# are ineligible for selection, while one selected action is retained after
# the frozen file exists. The template has no selected action, so count that
# one authorized slot without guessing which candidate wins.
reference_ids = {
    action_id
    for action, action_id in zip(actions, ids)
    if not bool(action.get("selection_eligible", True))
}
selected = config.get("selected_action")
if selected not in (None, ""):
    selected = str(selected)
    selected_ids = {"no_correction", selected} | reference_ids
    filtered = [action_id for action_id in ids if action_id in selected_ids]
    if len(filtered) < 2 or selected not in filtered:
        raise SystemExit("frozen validation selected_action is not registered")
    print(len(filtered))
else:
    eligible_candidates = [
        action_id
        for action, action_id in zip(actions, ids)
        if action_id != "no_correction"
        and bool(action.get("selection_eligible", True))
    ]
    if not eligible_candidates:
        raise SystemExit("validation template has no selectable correction action")
    print(len({"no_correction", eligible_candidates[0]} | reference_ids))
PY
}

validate_task_counts() {
    dev_count=$(registered_task_count "$DEV_PROMPTS" "$DEV_ACTIONS" 2)
    val_prompt_count=$(awk 'NR > 1 && NF { n++ } END { print n + 0 }' "$VAL_PROMPTS")
    val_actions_per_task=$(validation_action_count "$VAL_TEMPLATE")
    val_count=$((val_prompt_count * 3 * val_actions_per_task))
    if [ "$dev_count" -ne "$EXPECTED_DEV_TASKS" ] || [ "$val_count" -ne "$EXPECTED_VAL_TASKS" ]; then
        log "registered task count mismatch: development=$dev_count/$EXPECTED_DEV_TASKS validation=$val_count/$EXPECTED_VAL_TASKS (actions_per_prompt=$val_actions_per_task)"
        return 1
    fi
}

write_state() {
    stage=$1; status=$2; reason=; terminal=false
    [ "$#" -ge 3 ] && reason=$3
    [ "$#" -ge 4 ] && terminal=$4
    jq -n \
      --arg schema s7_trajectory_correction_queue_state_v1 \
      --arg stage "$stage" --arg status "$status" --arg reason "$reason" \
      --argjson terminal "$terminal" --arg updated "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --arg pid "$$" --arg queue_dir "$QUEUE_DIR" --arg gpu "$GPU" \
      --arg min_free "$MIN_FREE_MIB" --arg selected "$SELECTED_ACTION" \
      --arg dev_run "$DEV_RUN_DIR" --arg val_run "$VAL_RUN_DIR" \
      --arg val_actions "$VAL_ACTIONS" --arg dev_tasks "$EXPECTED_DEV_TASKS" \
      --arg val_tasks "$EXPECTED_VAL_TASKS" \
      '{schema:$schema,stage:$stage,status:$status,reason:$reason,terminal:$terminal,
        updated_at_utc:$updated,queue_pid:($pid|tonumber),queue_dir:$queue_dir,
        gpu:$gpu,min_free_mib:($min_free|tonumber),selected_action:$selected,
        development_run_dir:$dev_run,validation_run_dir:$val_run,
        validation_actions:$val_actions,
        expected_development_tasks:($dev_tasks|tonumber),
        expected_validation_tasks:($val_tasks|tonumber)}' \
      > "$STATE_PATH.tmp"
    mv -f "$STATE_PATH.tmp" "$STATE_PATH"
}

write_null_route() {
    reason=$1; exit_code=0; [ "$#" -ge 2 ] && exit_code=$2
    jq -n \
      --arg schema s7_trajectory_correction_null_route_v1 \
      --arg reason "$reason" --arg stage "$CURRENT_STAGE" \
      --arg code "$exit_code" --arg selected "$SELECTED_ACTION" \
      --arg created "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" --arg state "$STATE_PATH" \
      '{schema:$schema,status:"null_route",reason:$reason,failed_stage:$stage,
        exit_code:($code|tonumber),selected_action:$selected,created_at_utc:$created,
        state_path:$state}' > "$NULL_ROUTE_PATH.tmp"
    mv -f "$NULL_ROUTE_PATH.tmp" "$NULL_ROUTE_PATH"
    NULL_WRITTEN=1
    write_state null_route null_route "$reason" true
    log "null-route: $reason (exit_code=$exit_code)"
}

write_audit() {
    require_file "$DEV_ACTIONS"; require_file "$VAL_TEMPLATE"
    require_file "$DEV_PROMPTS"; require_file "$VAL_PROMPTS"
    validate_task_counts || return 1
    jq -n \
      --arg schema s7_trajectory_correction_queue_inputs_v1 \
      --arg created "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --arg commit "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)" \
      --arg script_sha "$(sha256_file "$0")" \
      --arg dev_actions "$DEV_ACTIONS" --arg dev_actions_sha "$(sha256_file "$DEV_ACTIONS")" \
      --arg dev_prompts "$DEV_PROMPTS" --arg dev_prompts_sha "$(sha256_file "$DEV_PROMPTS")" \
      --arg val_template "$VAL_TEMPLATE" --arg val_template_sha "$(sha256_file "$VAL_TEMPLATE")" \
      --arg val_prompts "$VAL_PROMPTS" --arg val_prompts_sha "$(sha256_file "$VAL_PROMPTS")" \
      --arg dev_tasks "$EXPECTED_DEV_TASKS" --arg val_tasks "$EXPECTED_VAL_TASKS" \
      '{schema:$schema,created_at_utc:$created,git_commit:$commit,
        queue_script_sha256:$script_sha,development_actions:$dev_actions,
        development_actions_sha256:$dev_actions_sha,development_prompts:$dev_prompts,
        development_prompts_sha256:$dev_prompts_sha,validation_template:$val_template,
        validation_template_sha256:$val_template_sha,validation_prompts:$val_prompts,
        validation_prompts_sha256:$val_prompts_sha,
        expected_development_tasks:($dev_tasks|tonumber),
        expected_validation_tasks:($val_tasks|tonumber),
        validation_seeds:[11,29,101],rl_auto_start:false}' > "$AUDIT_PATH.tmp"
    if [ -f "$AUDIT_PATH" ]; then
        [ "$(jq -r .git_commit "$AUDIT_PATH")" = "$(jq -r .git_commit "$AUDIT_PATH.tmp")" ] &&
        [ "$(jq -r .queue_script_sha256 "$AUDIT_PATH")" = "$(jq -r .queue_script_sha256 "$AUDIT_PATH.tmp")" ] &&
        [ "$(jq -r .development_actions_sha256 "$AUDIT_PATH")" = "$(jq -r .development_actions_sha256 "$AUDIT_PATH.tmp")" ] &&
        [ "$(jq -r .development_prompts_sha256 "$AUDIT_PATH")" = "$(jq -r .development_prompts_sha256 "$AUDIT_PATH.tmp")" ] &&
        [ "$(jq -r .validation_template_sha256 "$AUDIT_PATH")" = "$(jq -r .validation_template_sha256 "$AUDIT_PATH.tmp")" ] &&
        [ "$(jq -r .validation_prompts_sha256 "$AUDIT_PATH")" = "$(jq -r .validation_prompts_sha256 "$AUDIT_PATH.tmp")" ] || {
            rm -f "$AUDIT_PATH.tmp"; log "queue input audit mismatch"; return 1;
        }
        rm -f "$AUDIT_PATH.tmp"
    else
        mv -f "$AUDIT_PATH.tmp" "$AUDIT_PATH"
    fi
}

manifest_complete() {
    [ -f "$1" ] || return 1
    [ "$(awk 'NF { n++ } END { print n+0 }' "$1")" = "$2" ]
}

run_complete() {
    kind=$1; run_dir=$2; actions=$3; prompts=$4; seeds=$5; expected=$6
    [ -f "$run_dir/config.json" ] || return 1
    [ -f "$run_dir/manifest.jsonl" ] || return 1
    if [ "$kind" = scores ] && [ ! -f "$run_dir/scores.jsonl" ]; then
        return 1
    fi
    count=$(
        "$EVAL_PYTHON" "$ROOT/eval-pipeline/validate_trajectory_run.py" \
          --run-dir "$run_dir" --actions "$actions" --prompts "$prompts" \
          --seeds "$seeds" --kind "$kind" 2>/dev/null
    ) || return 1
    [ "$count" = "$expected" ]
}

validation_action_matches() {
    "$EVAL_PYTHON" - "$1" "$2" "$DEV_RUN_DIR/trajectory_correction_selection.json" "$DEV_ACTIONS" "$DEV_RUN_DIR" <<'PY'
import hashlib
import json
import os
import sys
import yaml

with open(sys.argv[1]) as handle:
    value = yaml.safe_load(handle) or {}
with open(sys.argv[3], "rb") as handle:
    selection_hash = hashlib.sha256(handle.read()).hexdigest()
with open(sys.argv[3]) as handle:
    selection = json.load(handle)
with open(sys.argv[4], "rb") as handle:
    actions_hash = hashlib.sha256(handle.read()).hexdigest()
with open(sys.argv[4]) as handle:
    source = yaml.safe_load(handle) or {}
selected = str(sys.argv[2])
source_by_id = {str(action.get("id")): action for action in source.get("actions", [])}
selected_spec = source_by_id.get(selected) or {}
run_dir = os.path.abspath(sys.argv[5])
config_path = os.path.join(run_dir, "config.json")
manifest_path = os.path.join(run_dir, "manifest.jsonl")
scores_path = os.path.join(run_dir, "scores.jsonl")
def digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()
run_config = {}
if os.path.isfile(config_path):
    with open(config_path) as handle:
        run_config = json.load(handle)
provenance = value.get("selection_provenance") or {}
selection_provenance = selection.get("provenance") or {}
valid = (
    value.get("schema") == "trajectory_correction_validation_v1"
    and str(value.get("selected_action", "")) == sys.argv[2]
    and provenance.get("selection_sha256") == selection_hash
    and provenance.get("selected_action") == sys.argv[2]
    and provenance.get("development_actions_sha256") == actions_hash
    and selection_provenance.get("actions_sha256") == actions_hash
    and selected_spec.get("type") == "trajectory_correction"
    and bool(selected_spec.get("selection_eligible", True))
    and os.path.abspath(str(selection_provenance.get("run_dir", ""))) == run_dir
    and os.path.abspath(str(provenance.get("development_run_dir", ""))) == run_dir
    and provenance.get("run_config_sha256") == digest(config_path)
    and provenance.get("manifest_sha256") == digest(manifest_path)
    and provenance.get("scores_sha256") == digest(scores_path)
    and provenance.get("run_contract_sha256") == run_config.get("run_contract_sha256")
    and selection_provenance.get("run_config_sha256") == provenance.get("run_config_sha256")
    and selection_provenance.get("manifest_sha256") == provenance.get("manifest_sha256")
    and selection_provenance.get("scores_sha256") == provenance.get("scores_sha256")
    and selection_provenance.get("selector_version")
    and selection_provenance.get("selector_script_sha256")
    and selection_provenance.get("selector_git_commit")
)
raise SystemExit(0 if valid else 1)
PY
}

registered_development_config() {
    run_dir=$1
    actions_sha=$(sha256_file "$DEV_ACTIONS")
    prompts_sha=$(sha256_file "$DEV_PROMPTS")
    [ -f "$run_dir/config.json" ] || return 1
    jq -e --arg actions "$actions_sha" --arg prompts "$prompts_sha" \
      '.actions_sha256 == $actions and .prompts_sha256 == $prompts and
       .split_role == "development" and (.seeds | map(tonumber)) == [0,42] and
       .trajectory_registered == true and
       .action_schema == "trajectory_correction_actions_v1" and
       .registered_sampling.model == "stabilityai/stable-diffusion-xl-base-1.0" and
       .registered_sampling.scheduler == "EulerDiscreteScheduler" and
       .registered_sampling.extra_unet_calls == 0 and
       (.run_contract_sha256 | type) == "string" and (.run_contract_sha256 | length) == 64' \
      "$run_dir/config.json" >/dev/null
}

allowed_gpu() {
    [ -z "$DEVICES" ] && return 0
    compact=$(printf '%s' "$DEVICES" | tr -d '[:space:]')
    case ",$compact," in *,"$1",*) return 0 ;; *) return 1 ;; esac
}

find_gpu() {
    test_free=$(printenv S7_TEST_FREE_MIB || true)
    if [ -n "$test_free" ]; then
        [ "$test_free" -ge "$MIN_FREE_MIB" ] && printf '%s\n' "$(printenv S7_TEST_GPU || echo 0)"
        return 0
    fi
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    while IFS=, read -r index used free total; do
        index=$(printf '%s' "$index" | tr -d '[:space:]')
        free=$(printf '%s' "$free" | tr -d '[:space:]')
        if [[ "$index" =~ ^[0-9]+$ ]] && [[ "$free" =~ ^[0-9]+$ ]] &&
           [ "$free" -ge "$MIN_FREE_MIB" ] && allowed_gpu "$index"; then
            printf '%s\n' "$index"; return 0
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
}

wait_existing() {
    [ "$DRY_RUN" -eq 1 ] && return 0
    while :; do
        # Wait for every independently launched S7 stage that targets either
        # registered run.  Match the active paths rather than a versioned
        # literal so a resumed queue cannot overlap a v3/v4 run.  Keep the
        # queue's own inspection helpers out of the result: their awk source
        # contains the same path expressions.
        pids=$(ps -eo pid=,comm=,args= | awk -v self="$$" \
          -v dev="$DEV_RUN_DIR" -v val="$VAL_RUN_DIR" '
          $1 == self {next}
          index($0, dev) == 0 && index($0, val) == 0 {next}
          $0 !~ /eval-pipeline\/(generate|score|select_trajectory_correction)\.py/ {next}
          $0 ~ /run_trajectory_correction_queue\.sh/ {next}
          $2 ~ /^(awk|ps|rg|grep|sed)$/ {next}
          {print $1}
        ')
        [ -z "$pids" ] && return 0
        CURRENT_STAGE=existing_process_wait
        write_state "$CURRENT_STAGE" waiting "pids=$pids"
        log "existing S7 process(es): $pids; waiting without signaling"
        sleep "$POLL_SECONDS"
    done
}

wait_gpu() {
    while :; do
        found=$(find_gpu)
        if [ -n "$found" ]; then GPU=$found; log "selected GPU $GPU"; return 0; fi
        CURRENT_STAGE=waiting_for_gpu
        write_state "$CURRENT_STAGE" waiting "no GPU with >=$MIN_FREE_MIB MiB"
        log "waiting for >=$MIN_FREE_MIB MiB free"
        if [ "$DRY_RUN" -eq 1 ]; then GPU=$(printenv S7_TEST_GPU || echo 0); return 0; fi
        sleep "$POLL_SECONDS"
    done
}

render_command() {
    rendered=
    for part in "$@"; do
        printf -v quoted '%q' "$part"
        rendered="$rendered $quoted"
    done
    printf '%s' "$rendered"
}

run_stage() {
    stage=$1; shift; CURRENT_STAGE=$stage; write_state "$stage" running
    log "command:$stage$(render_command "$@")"
    if [ "$DRY_RUN" -eq 1 ]; then log "dry-run: command not executed"
    elif "$@"; then :
    else
        code=$?; write_null_route "command_failed:$stage" "$code"; return "$code"
    fi
    write_state "$stage" complete
}

main() {
    if [ "$(state_value terminal || true)" = true ]; then
        log "terminal state already recorded; exiting idempotently"; return 0
    fi

    if [ "$DRY_RUN" -eq 0 ] && [ -e "$DEV_RUN_DIR/manifest.jsonl" ] &&
       ! registered_development_config "$DEV_RUN_DIR"; then
        write_null_route development_config_mismatch 1; return 1
    fi
    if [ "$DRY_RUN" -eq 1 ] || ! run_complete manifest "$DEV_RUN_DIR" "$DEV_ACTIONS" "$DEV_PROMPTS" "0,42" "$EXPECTED_DEV_TASKS"; then
        wait_existing; wait_gpu; mkdir -p "$DEV_RUN_DIR"
        run_stage development_generation "$GEN_PYTHON" "$ROOT/eval-pipeline/generate.py" \
          --devices "$GPU" --prompts "$DEV_PROMPTS" --out_dir "$DEV_RUN_DIR" \
          --actions "$DEV_ACTIONS" --split_role development --seeds 0,42 || return $?
    fi
    if [ "$DRY_RUN" -eq 0 ] && ! run_complete manifest "$DEV_RUN_DIR" "$DEV_ACTIONS" "$DEV_PROMPTS" "0,42" "$EXPECTED_DEV_TASKS"; then
        write_null_route incomplete_development_manifest 1; return 1
    fi
    if [ "$DRY_RUN" -eq 0 ] && ! registered_development_config "$DEV_RUN_DIR"; then
        write_null_route development_config_missing 1; return 1
    fi

    if [ "$DRY_RUN" -eq 1 ] || ! run_complete scores "$DEV_RUN_DIR" "$DEV_ACTIONS" "$DEV_PROMPTS" "0,42" "$EXPECTED_DEV_TASKS"; then
        wait_existing; wait_gpu
        run_stage development_scoring "$EVAL_PYTHON" "$ROOT/eval-pipeline/score.py" \
          --run_dir "$DEV_RUN_DIR" --device "$GPU" --strict || return $?
    fi
    if [ "$DRY_RUN" -eq 1 ] || [ ! -f "$DEV_RUN_DIR/trajectory_correction_selection.json" ]; then
        wait_existing
        run_stage development_selection "$EVAL_PYTHON" "$ROOT/eval-pipeline/select_trajectory_correction.py" \
          --run_dir "$DEV_RUN_DIR" --actions "$DEV_ACTIONS" || return $?
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        DRY_SELECTION=$(printenv S7_DRY_RUN_SELECTION || true)
        [ -n "$DRY_SELECTION" ] || DRY_SELECTION=no_correction
        SELECTED_ACTION=$DRY_SELECTION
    else
        SELECTED_ACTION=$(jq -r '.selected_action // empty' "$DEV_RUN_DIR/trajectory_correction_selection.json")
    fi
    if [ -z "$SELECTED_ACTION" ] || [ "$SELECTED_ACTION" = no_correction ]; then
        CURRENT_STAGE=selector_gate
        write_null_route selector_no_correction 0; return 0
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        write_state validation_queue awaiting_review dry_run_validation_queued true
        log "dry-run pass branch complete; freeze/validation commands not executed"; return 0
    fi

    if [ ! -f "$VAL_ACTIONS" ]; then
        run_stage validation_freeze "$EVAL_PYTHON" "$ROOT/eval-pipeline/freeze_trajectory_correction_validation.py" \
          --selection "$DEV_RUN_DIR/trajectory_correction_selection.json" \
          --template "$VAL_TEMPLATE" --source-actions "$DEV_ACTIONS" --output "$VAL_ACTIONS" || return $?
    else
        if ! validation_action_matches "$VAL_ACTIONS" "$SELECTED_ACTION"; then
            write_null_route validation_action_mismatch 1; return 1
        fi
    fi

    if ! run_complete manifest "$VAL_RUN_DIR" "$VAL_ACTIONS" "$VAL_PROMPTS" "11,29,101" "$EXPECTED_VAL_TASKS"; then
        wait_existing; wait_gpu; mkdir -p "$VAL_RUN_DIR"
        run_stage validation_generation "$GEN_PYTHON" "$ROOT/eval-pipeline/generate.py" \
          --devices "$GPU" --prompts "$VAL_PROMPTS" --out_dir "$VAL_RUN_DIR" \
          --actions "$VAL_ACTIONS" --split_role validation_confirmation --seeds 11,29,101 || return $?
    fi
    if ! run_complete manifest "$VAL_RUN_DIR" "$VAL_ACTIONS" "$VAL_PROMPTS" "11,29,101" "$EXPECTED_VAL_TASKS"; then
        write_null_route incomplete_validation_manifest 1; return 1
    fi
    if ! run_complete scores "$VAL_RUN_DIR" "$VAL_ACTIONS" "$VAL_PROMPTS" "11,29,101" "$EXPECTED_VAL_TASKS"; then
        wait_existing; wait_gpu
        run_stage validation_scoring "$EVAL_PYTHON" "$ROOT/eval-pipeline/score.py" \
          --run_dir "$VAL_RUN_DIR" --device "$GPU" --strict || return $?
    fi
    write_state complete awaiting_review validation_scored_pending_review true
    log "validation scores complete; awaiting review, no RL stage started"
}

on_exit() {
    code=$?
    if [ "$code" -ne 0 ] && [ "$NULL_WRITTEN" -eq 0 ]; then
        write_null_route "shell_failure:$CURRENT_STAGE" "$code" || true
    fi
}
trap on_exit EXIT
write_audit
main
