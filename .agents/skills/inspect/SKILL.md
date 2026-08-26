---
name: inspect
description: >-
  Audit RepLDM code, documentation, configuration, and experiment interfaces for objective
  defects, then minimally fix, validate, and commit until three fresh independent Codex review
  rounds are clean. Use only when the user invokes $inspect or explicitly requests this exact
  audit-fix-commit loop and authorizes its commits. Accept an optional path/glob scope or the
  selectors code, docs, pipelines, attention, or eval; default to the whole repository.
---

# Inspect RepLDM

Drive the requested RepLDM scope to a verified-clean state. Orchestrate the loop as the main agent.
Only the main agent may edit, test, stage, commit, or change branches. Use fresh read-only Codex
reviewers for every inspection round so clean rounds remain independent.

Treat invocation as authorization only for commits containing fixes produced by this loop. Never
push, rewrite history, switch to an existing unrelated branch, or include unrelated user work. If
the current branch is the local or remote default/protected branch, create and switch to one new
working branch immediately before the first loop commit; otherwise remain on the current branch.

## Prepare the repository

1. Resolve the repository with `git rev-parse --show-toplevel` and run from that root. Stop if it is
   not the RepLDM repository containing `AttentionGuidance/`, `InferencePipelines/`, and
   `eval-pipeline/`.
2. Read the applicable `AGENTS.md`, `CLAUDE.md`, `README.md`, and scoped design documentation. Read
   `eval-pipeline/README.md`, `doc/research/EXPERIMENT_PLAN.md`, and
   `doc/research/EXPERIMENT_RESULTS.md` when the scope touches evaluation or experiments. Treat
   current code and configuration as authoritative over stale
   prose, except that explicit experiment requirements remain requirements.
3. Record the branch, `HEAD`, `git status --short`, separate cached and unstaged diffs, and
   non-ignored untracked paths. Record an explicit detached marker when `git symbolic-ref -q HEAD`
   has no result. Snapshot the existing index so later commits cannot absorb staged work that
   predates the loop.
4. Preserve every pre-existing dirty hunk. It may be inspected when in scope, but it remains
   user-owned unless a verified fix must overlap it. If a loop fix cannot be isolated from user
   work for staging, stop and ask instead of committing the combined file.

## Resolve the scope

- With no argument, inspect tracked and non-ignored untracked repository source, notebooks,
  configuration, shell scripts, and documentation. Include deletions and both sides of renames from
  the separate index and worktree layers. Exclude `.git` and ignored, untracked generated artifacts.
  Keep any tracked or non-ignored output, weight, checkpoint, dataset, cache, or generated figure in
  scope for tracking-policy and reference review, while skipping inspection of opaque binary
  contents. Treat intentional paper figures under `fig/` as allowed tracked artifacts when their
  provenance and documentation references are valid.
- For paths or globs, inspect those paths and their direct blast radius through callers, exports,
  docs, configs, notebooks, and runnable scripts.
- For `code` or `docs`, restrict the primary scope to that category. Map `attention` to
  `AttentionGuidance/`, `pipelines` to `InferencePipelines/` plus affected `InferCases/` callers,
  and `eval` to `eval-pipeline/` plus experiment documentation.
- Treat out-of-scope files as read-only evidence. Report confirmed defects outside scope without
  fixing them or resetting the clean streak. Never expand the writable scope silently.

Announce the resolved scope before the first round.

## Apply RepLDM review lenses

Ask every panel for a full review while rotating emphasis across these project-specific contracts:

1. **Attention Guidance:** tensor shapes, dtype/device preservation, schedule and density indexing,
   optional FFT filtering, scaling/decay behavior, and numerical edge cases. Verify from current
   code that pipeline calls obey the reverse `t_index` convention rather than confusing it with a
   scheduler timestep.
2. **RepLDM pipelines:** diffusers 0.21.4 compatibility; Stage-1 scheduler/guidance ordering;
   aspect-ratio and Stage-2 trigger math; `init_rates` versus upsampling stages; VAE encode/decode,
   anchor-statistic normalization, CPU offload, tiling, and latent dtype/device transitions.
   Check that Attention Guidance remains limited to the intended stage.
3. **ControlNet and FreeScale:** condition-image sizes and per-stage propagation, control guidance
   bounds, FreeScale transformer method binding, window/dilation logic, and intentionally different
   Attention Guidance parameter names. Do not assume the three pipelines share identical APIs.
4. **Public contracts and examples:** exports from both installable packages; pipeline signatures,
   callbacks, return values, and `images[-1]`; CSV parsing, multiprocessing/GPU selection, and
   notebook/script call sites under `InferCases/`, which are runnable examples rather than a Python
   package.
5. **Evaluation instrument:** generation/score/aggregate manifest schemas, resume behavior,
   scorer registry/config/output-key consistency, metric direction, prompt/seed/scale pairing,
   offline weight checks, and the deliberate separation between the Python 3.9 `repldm` generation
   environment and the scoring environment documented in `eval-pipeline/README.md`.
6. **Repository policy:** diffusion, ControlNet, and preprocessing model loads must remain
   local-only unless the user explicitly changes that policy. Treat the documented, opt-in
   `eval-pipeline/prestage_weights.py` download as the scoring-weight exception; scoring itself must
   remain offline afterward. Experiment outputs, caches, weights, and smoke artifacts must stay
   untracked, while intentional paper figures belong under `fig/`. Check docs, commands, paths,
   defaults, and `main` versus paper-reproduction `base` claims against reality.

Do not report subjective style, speculative quality improvements, expected infrastructure absence,
or paper-versus-current differences that the repository explicitly documents.

## Run independent rounds

Initialize `round = 1` and `consecutive_clean = 0`. Require three consecutive clean rounds and stop
at round 12 if the loop has not converged.

For each round:

1. Require `spawn_agent`; stop if it is unavailable because the independent-clean guarantee cannot
   be met. Spawn a fresh read-only panel with `fork_turns="none"`, using available parallel slots.
   Give each reviewer only the repository path, resolved scope, current lens, current diff-reading
   instructions, and output schema. Do not reveal earlier findings, fixes, or expected answers.
2. Forbid reviewers from editing, creating, deleting, staging, committing, changing branches,
   invoking skills, or launching training/inference. Require `NO_FINDINGS` or findings containing
   `file`, `line`, `severity`, `problem`, `evidence`, and `minimal_fix`.
3. Before launch, fingerprint in-scope contents, separate cached and unstaged diffs, the untracked
   list, status, `HEAD`, and symbolic branch ref. Wait for the entire panel and compare every
   fingerprint. Invalidate the round and stop on unexpected mutation; never blanket-restore user
   work.
4. Merge duplicates and verify every claim directly in current files and affected callers. Count
   only objective in-scope defects supported by concrete evidence.
5. If triage is clean, increment `consecutive_clean`. If defects remain, reset it to zero, apply the
   minimal fixes, validate them, and make one fixing-round commit before the next fresh round.

Stop and ask the user if the same finding returns after two attempted fixes or correct behavior is
ambiguous.

## Validate fixes

Use `apply_patch`, preserve local style, and avoid wholesale formatting of the large pipeline files.
Select `${REPLDM_PYTHON}` only when it names an executable Python 3.9-compatible interpreter;
otherwise prefer `conda run -n repldm python`. If neither is available, use another documented
Python 3.9 environment when possible and report the exact fallback. Use that selected interpreter
or command prefix for every `python` placeholder below.

Apply the lightest relevant checks, increasing coverage with risk:

- For any Python change, run `python -m compileall AttentionGuidance InferencePipelines
  eval-pipeline`. Also run `python -m py_compile` on changed Python entry points under `InferCases/`.
- For package or pipeline changes, import the affected public symbols. When PyTorch is available,
  run a CPU-only `AttnGuidance` forward check with a small random latent and assert shape, dtype,
  device, and finite values. Do not load model weights for this check.
- For eval changes, run affected CLI `--help` paths and focused model-free checks. Use the scoring
  interpreter documented in `eval-pipeline/README.md` only for scorer changes that require its
  dependencies; do not pretend the generation environment contains scoring-only packages.
- Parse changed YAML, CSV, JSON, JSONL, and notebooks with structured parsers. Run `bash -n` on
  changed shell scripts. Verify every changed doc command, flag, path, link, and symbol against code.
- Run minimal generation with `eval-pipeline/prompts/smoke.csv` and an ignored `outputs/` directory
  only when generation behavior changed, a suitable GPU and cached local weights exist, and the run
  is practical. Never download weights implicitly. Put long/background runs in a new tmux window as
  required by `CLAUDE.md`, and remove only artifacts created by the loop.

Do not launch full training, full-resolution inference, or broad evaluation by default. Treat absent
CUDA, weights, datasets, or optional metric packages as infrastructure limits, not code defects.
If a fix fails a relevant check, reverse only that fix's own hunks with `apply_patch` or a safe
pre-fix snapshot and stop without committing when it cannot be corrected.

## Commit safely

Commit once per fixing round after checks pass. Stage only loop-owned hunks and inspect
`git diff --cached`; path-stage a file only when it was clean at startup and all its changes belong
to the loop. Compare the index with its startup snapshot before committing. If unrelated staged
hunks would be included, stop and ask rather than unstaging, resetting, or committing them. Never
use `git add .` or `git add -A`.

Follow repository history with an imperative, scoped subject, for example:

```text
inspect(iter 4): fix ControlNet stage transition
```

Do not create empty commits or add a vendor-specific co-author trailer unless requested. Never
commit generated outputs, cached weights, datasets, or smoke artifacts. Never push.

## Report the result

Report the resolved scope, branch and starting `HEAD`, rounds run, verified findings fixed, checks
run or skipped with infrastructure reasons, and each commit hash/subject. Confirm the final three
fresh rounds were clean, or state the round cap, oscillation, mutation, out-of-scope blocker, or
user decision that stopped convergence.
