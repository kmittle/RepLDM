---
name: check
description: >-
  Verify only RepLDM's invocation-time pending Git change-set through independent Codex review,
  minimal fixes, project-specific smoke tests, and safe checkpoint commits until the requested
  number of fresh rounds are clean. Use only when the user invokes $check or explicitly asks for
  this exact pending-change hardening and commit loop. Freeze the starting baseline and staged,
  unstaged, deleted, and non-ignored untracked scope for the entire run.
---

# Check Pending RepLDM Changes

Harden only the pending change-set that exists when the skill starts. Keep the Git baseline and file
scope fixed so commits made during the loop do not make later reviews appear smaller. Review changed
hunks and their direct blast radius, not the whole repository.

The main agent is the only editor, tester, stager, committer, or branch changer. Invocation
authorizes commits only for the frozen RepLDM change-set and verified fixes directly required by it.
Never push, rewrite history, switch to an existing unrelated branch, or include unrelated work. If
the current branch is the local or remote default/protected branch, create and switch to one new
working branch before the first loop commit; otherwise stay on the current branch.

## Invocation Parameters

The default `n` is `2`. A positive integer after `$check` sets `n`: `$check 1` passes after one
complete independent review round with no confirmed errors. It does not mean one reviewer, one
test, or a weaker audit; every round still runs all applicable lenses, verification, and smoke
tests. The safety cap remains 12 rounds, and the current pending change-set is always the scope.

## Freeze the baseline and scope

Run from the root returned by `git rev-parse --show-toplevel`. Confirm it contains
`AttentionGuidance/`, `InferencePipelines/`, and `eval-pipeline/`, then read applicable
`AGENTS.md`, `CLAUDE.md`, `README.md`, and subsystem documentation.

At startup:

1. Set `BASE` to `git rev-parse HEAD` and never change it during the run. Record the branch and
   symbolic branch ref, using an explicit detached marker when `git symbolic-ref -q HEAD` has no
   result.
2. Collect tracked paths separately from
   `git diff --no-renames --cached --name-only HEAD` and
   `git diff --no-renames --name-only`. Add paths from
   `git ls-files --others --exclude-standard`. Retain deleted paths and both sides of renames, then
   deduplicate and announce the frozen list.
3. Record paths that occur in both tracked lists. A checkpoint would combine or replace their
   distinct pre-existing index and worktree states, so stop and ask whether to preserve the current
   index or commit the complete worktree snapshot before starting the loop.
4. Inspect file types, symlinks, and sizes. Stop and ask if the frozen set contains experiment
   outputs, model weights, checkpoints, datasets, caches, large binaries, or paths that should live
   under ignored `outputs/` or `pretrained_ckpts/`; never silently commit them. Permit an intentional
   paper figure under `fig/` only after confirming its provenance and documentation reference.
5. If the frozen list is empty, report that there is nothing to check and stop.

On every round, inspect these layers separately:

- committed since invocation: `git diff --no-renames BASE HEAD -- <frozen paths>`;
- current index: `git diff --no-renames --cached HEAD -- <frozen paths>`;
- current worktree: `git diff --no-renames -- <frozen paths>`;
- frozen untracked files directly, including structured notebook content when applicable.

Read direct callers, public exports, docs, configs, notebooks, and scripts outside the frozen set
only as evidence. Confirmed out-of-scope defects are report-only. If a correct fix requires an
out-of-scope path, stop for a user decision; never expand the frozen set silently.

## Review RepLDM contracts

Cover these lenses across each panel, with emphasis on behavior changed by the diff:

1. **Tensor and scheduler correctness:** shapes, dtype/device movement, numerical edge cases,
   Attention Guidance schedules and reverse `t_index`, scheduler-step ordering, Stage-2 trigger
   area, restart indices, anchor normalization, tiling, and CPU offload.
2. **Pipeline blast radius:** SDXL, ControlNet, and FreeScale signatures and intentionally different
   parameter names; package exports; callbacks and return values; condition-image propagation;
   runnable `InferCases/` scripts and notebooks.
3. **Evaluation contracts:** prompt/seed/scale pairing, manifest and score schemas, resume behavior,
   scorer registry/config/output keys, metric direction, offline weight checks, and the separate
   `diff_attn` generation and `repldm_eval` scoring environments documented in
   `eval-pipeline/README.md` (the generation lock currently records Python 3.11.10).
4. **Docs and repository policy:** commands, flags, defaults, paths, links, branch claims, local-only
   generation/preprocessing model loading, the explicit scoring-weight download exception in
   `eval-pipeline/prestage_weights.py`, and generated-artifact exclusions. Distinguish objective
   contradictions from examples explicitly presented as typical values.
5. **Leftovers and runnability:** undefined or stale symbols, debug output, dead code caused by the
   change, accidental artifacts, and Python 3.9 or the pinned `diffusers>=0.32.1,<0.33`
   compatibility.

Reject subjective style preferences, speculative output-quality changes, unrelated pre-existing
defects, and failures caused only by unavailable CUDA, weights, datasets, or optional packages.

## Run the loop

Initialize `round = 1`, `consecutive_clean = 0`, `checkpoint_created = false`, and `STREAK_TARGET = 2`
(or the positive integer supplied after `$check`). Require `STREAK_TARGET` consecutive clean rounds
and stop at round 12 if convergence is not reached.

For each round:

1. Require `spawn_agent`; stop if unavailable. Spawn a fresh read-only panel with
   `fork_turns="none"` and available parallel slots. Provide the repository, `BASE`, frozen paths,
   separate diff-layer instructions, current lens, and output schema, but no prior findings or
   expected answers.
2. Forbid edits, file creation/deletion, staging, commits, branch changes, skill calls, and
   training/inference. Require `NO_FINDINGS` or findings with `file`, `line`, `severity`, `problem`,
   `evidence`, and `minimal_fix`.
3. Before launch, fingerprint every existing frozen file, all three diff layers, the non-ignored
   untracked list, status, `HEAD`, and symbolic branch ref. Wait for all reviewers and compare the
   fingerprints. Invalidate the round and stop on unexpected mutation or an out-of-scope status
   entry; never blanket-restore user work.
4. Merge duplicates and verify each claim directly. Count only objective, in-scope defects with
   concrete evidence.
5. When the panel is clean, validate and checkpoint the frozen change-set once if
   `checkpoint_created` is false, set `checkpoint_created = true`, then increment the clean streak.
   When verified defects remain, reset the streak, minimally fix them within the frozen set,
   validate, create one fixing-round commit, and set `checkpoint_created = true`.
   Stop when the clean streak reaches `STREAK_TARGET`.

Stop and ask if the same finding returns after two attempted fixes or correct behavior is ambiguous.

## Validate the frozen changes

Use `apply_patch` and preserve all invocation-time work. For package-level checks, select
`${REPLDM_PYTHON}` only when it is executable and Python `>=3.9`; otherwise use the documented
`diff_attn` Python 3.11.10 interpreter as the fallback. For generation commands use that same
`diff_attn` interpreter; for scoring-only commands use the documented `repldm_eval` interpreter.
Do not assume that a conda environment named `repldm` exists. Report any fallback, and use the
selected interpreter or command prefix for every `python` placeholder below.

- For Python changes, run `python -m compileall AttentionGuidance InferencePipelines eval-pipeline`
  and `python -m py_compile` for changed Python entry points under `InferCases/`.
- For package or pipeline changes, import affected public symbols. When PyTorch is available, run a
  small CPU-only `AttnGuidance` forward check and assert shape, dtype, device, and finite values.
- For eval changes, run affected CLI `--help` paths and focused model-free checks. Use the separate
  scoring interpreter from `eval-pipeline/README.md` only where its dependencies are required.
- Parse changed YAML, CSV, JSON, JSONL, and notebooks with structured parsers; run `bash -n` for
  shell changes; verify changed docs against real code and commands.
- Use `eval-pipeline/prompts/smoke.csv` and an ignored `outputs/` directory for minimal generation
  only when generation behavior changed and suitable GPU/cached local weights are available. Never
  download weights implicitly. Put long runs in a new tmux window and remove loop-created artifacts.

Do not run full training, full-resolution inference, or broad scoring by default. If a loop fix
cannot pass a relevant check, reverse only that fix's hunks with `apply_patch` or a safe pre-fix
snapshot and stop without committing when it cannot be corrected.

## Commit the verified snapshot

After green checks, stage frozen paths explicitly and inspect the exact staged diff. On the first
fixing round or first clean checkpoint, this records the invocation-time RepLDM worktree snapshot
plus verified fixes; later fixing commits contain only additional fixes inside the same frozen set.
Never use `git add .` or `git add -A`, and never stage outside the frozen set.

Do not path-stage a file with unresolved distinct index/worktree state; the startup decision must
state which layer to preserve. Keep `BASE` fixed after every commit and continue reviewing
`git diff BASE`. If correction returns the entire frozen set to `BASE`, skip the empty commit and
continue the clean-round loop.

Use imperative subjects consistent with repository history, for example:

```text
check(iter 2): fix pending pipeline regression
check: checkpoint verified pending changes
```

Do not add a vendor-specific co-author trailer unless requested. Never push.

## Report the result

List the branch, `BASE`, frozen paths, rounds and final streak, verified findings and fixes, checks
run or skipped, and commit hashes/subjects. Confirm the final `STREAK_TARGET` independent rounds were clean,
or explain the round cap, oscillation, reviewer mutation, out-of-scope blocker, unsafe artifact, or
unresolved user decision.
