# S7 Dormant Handoff Watcher (Draft)

`eval-pipeline/s7_dormant_handoff_watcher.sh` is intentionally dormant. It
only reads one S7 `state.json` until the JSON field `.terminal` is `true`. It
does not call `nvidia-smi`, inspect GPU memory, signal PIDs, or launch a
renderer in its default mode. A malformed or partially-written state is
treated as a non-terminal poll.

The safe observation command is:

```bash
eval-pipeline/s7_dormant_handoff_watcher.sh \
  --state /absolute/path/to/s7/queue/state.json \
  --poll-seconds 30
```

This exits after observing the registered S7 schema with `terminal: true` and
prints no handoff command. It is therefore suitable for a human-reviewed
handoff boundary.

To authorize a handoff, a human must create a READY file containing exactly
`S8_READY_v1`, then pass `--execute` and an explicit command after `--`:

```bash
eval-pipeline/s7_dormant_handoff_watcher.sh \
  --state /absolute/path/to/s7/queue/state.json \
  --ready-file /absolute/path/to/S8_READY \
  --stamp /absolute/path/to/s7/queue/s8_handoff.started \
  --execute -- /absolute/path/to/reviewed-command --arg value
```

The command is passed as an argument vector (there is no `eval`). The stamp is
optional but recommended for an idempotent queue handoff. This draft does not
create the READY file, does not choose an S8 action, and does not authorize GPU
work. Before adoption, review the exact state path, command, provenance
requirements, and whether a terminal `null_route` should be accepted.

The final launch gate belongs to the human/operator, outside this watcher. A
reviewed command should be invoked only after all three predicates are true:

```text
S7 state.json: .terminal == true
CFG-EC worktree: READY file contains exactly S8_READY_v1
selected GPU: memory.free >= 22000 MiB
```

For example, the following is a review template, not a command to run during
the dormant audit. It keeps the GPU check outside the state-only watcher and
passes the reviewed command as an explicit argv:

```bash
STATE=/absolute/path/to/s7/queue/state.json
READY=/absolute/path/to/repldm-cfgec/S8_READY
GPU_INDEX=0
FREE_MIB=$(nvidia-smi -i "$GPU_INDEX" \
  --query-gpu=memory.free --format=csv,noheader,nounits)
test "$(jq -r '.terminal // false' "$STATE")" = true \
  && test -f "$READY" \
  && grep -Fqx 'S8_READY_v1' "$READY" \
  && test "$FREE_MIB" -ge 22000 \
  && eval-pipeline/s7_dormant_handoff_watcher.sh \
       --state "$STATE" --ready-file "$READY" --execute \
       -- /absolute/path/to/reviewed-cfg-ec-command --devices "$GPU_INDEX"
```

The template intentionally has no fallback GPU, no process termination, and no
automatic action selection. If any predicate fails, the short-circuit leaves
the command unstarted.
