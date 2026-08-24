#!/usr/bin/env bash
# DORMANT S7 handoff watcher draft.
#
# This script reads one S7 state.json and never starts, stops, or signals a
# process in its default mode. A handoff command is run only when the caller
# supplies both --execute and --ready-file, plus an explicit command after --.
set -Eeuo pipefail

STATE_PATH=
READY_PATH=
POLL_SECONDS=30
EXECUTE=0
STAMP_PATH=
HANDOFF_CMD=()

usage() {
    cat <<'USAGE'
Usage:
  s7_dormant_handoff_watcher.sh --state PATH [--poll-seconds N]
  s7_dormant_handoff_watcher.sh --state PATH --ready-file PATH --execute \
      [--stamp PATH] -- COMMAND [ARG ...]

Default mode only waits for .terminal == true in the selected S7 state.json.
It does not inspect GPUs and does not launch a command.

Handoff mode requires all of:
  --ready-file PATH   a file containing exactly S8_READY_v1
  --execute           explicit authorization for one handoff command
  -- COMMAND ...      command and arguments, passed without eval
  --stamp PATH        optional once-only marker; existing marker is a no-op
USAGE
}

die() { printf 's7 watcher: %s\n' "$*" >&2; exit 2; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --state)
            shift; [ "$#" -gt 0 ] || die "--state needs a path"; STATE_PATH=$1 ;;
        --ready-file)
            shift; [ "$#" -gt 0 ] || die "--ready-file needs a path"; READY_PATH=$1 ;;
        --poll-seconds)
            shift; [ "$#" -gt 0 ] || die "--poll-seconds needs a number"; POLL_SECONDS=$1 ;;
        --stamp)
            shift; [ "$#" -gt 0 ] || die "--stamp needs a path"; STAMP_PATH=$1 ;;
        --execute) EXECUTE=1 ;;
        --)
            shift; HANDOFF_CMD=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (use -- before a handoff command)" ;;
    esac
    shift
done

[ -n "$STATE_PATH" ] || die "--state is required"
[[ "$POLL_SECONDS" =~ ^[0-9]+$ ]] || die "poll-seconds must be a non-negative integer"
[ "$POLL_SECONDS" -gt 0 ] || die "poll-seconds must be positive"
command -v jq >/dev/null 2>&1 || die "jq is required to inspect state.json"

if [ "$EXECUTE" -eq 1 ]; then
    [ -n "$READY_PATH" ] || die "--execute requires --ready-file"
    [ "${#HANDOFF_CMD[@]}" -gt 0 ] || die "--execute requires a command after --"
else
    [ -z "$READY_PATH" ] || die "--ready-file requires --execute"
    [ "${#HANDOFF_CMD[@]}" -eq 0 ] || die "a command requires --execute"
fi

state_is_terminal() {
    # jq's -e makes malformed or partially-written JSON a non-terminal poll.
    [ -r "$STATE_PATH" ] || return 1
    jq -e '
        (.schema == "s7_trajectory_correction_queue_state_v1") and
        (.terminal == true)
    ' "$STATE_PATH" >/dev/null 2>&1
}

wait_for_terminal() {
    local last_signature=
    local signature
    while :; do
        if [ -r "$STATE_PATH" ]; then
            signature=$(sha256sum "$STATE_PATH" 2>/dev/null | awk '{print $1}' || true)
            if [ -n "$signature" ] && [ "$signature" != "$last_signature" ]; then
                last_signature=$signature
                printf 's7 watcher: observed state update %s\n' "$signature"
            fi
        fi
        if state_is_terminal; then
            printf 's7 watcher: terminal=true in %s\n' "$STATE_PATH"
            return 0
        fi
        sleep "$POLL_SECONDS"
    done
}

validate_ready() {
    [ -f "$READY_PATH" ] || die "READY file is missing: $READY_PATH"
    # Permit either the exact token or one conventional trailing newline, but
    # reject embedded newlines, extra bytes, and whitespace.
    if ! cmp -s "$READY_PATH" <(printf 'S8_READY_v1') &&
       ! cmp -s "$READY_PATH" <(printf 'S8_READY_v1\n'); then
        die "READY file must contain exactly S8_READY_v1"
    fi
}

claim_stamp() {
    [ -n "$STAMP_PATH" ] || return 0
    if [ -e "$STAMP_PATH" ]; then
        printf 's7 watcher: handoff stamp already exists; not running command\n'
        exit 0
    fi
    # noclobber makes the once-only marker safe when two dormant watchers wake
    # on the same state transition. The marker contains no executable data.
    ( set -o noclobber; printf '%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$STAMP_PATH" ) \
        2>/dev/null || {
            printf 's7 watcher: another watcher claimed %s; not running command\n' "$STAMP_PATH"
            exit 0
        }
}

wait_for_terminal

if [ "$EXECUTE" -eq 0 ]; then
    printf 's7 watcher: dormant wait completed; no command was launched\n'
    exit 0
fi

validate_ready
claim_stamp
printf 's7 watcher: executing explicit handoff command\n'
exec "${HANDOFF_CMD[@]}"
