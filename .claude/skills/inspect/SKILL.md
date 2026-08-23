---
name: inspect
description: Run a closed-loop QA pass over the RepLDM repo's code and docs — independent multi-agent inspection → fix → lightweight smoke test → commit — looping until 3 consecutive independent inspections find zero errors. Use when asked to "inspect", audit, sanity-check, or clean up the repository for correctness and doc/code consistency.
---

# /inspect — closed-loop code & doc inspection

Drive the repository to a verified-clean state by repeating

    检查 (inspect) → 改错 (fix) → smoke test → git 提交 (commit)

until **3 consecutive, independent inspections each find 0 confirmed errors**. Honor the conventions in `CLAUDE.md`: background/long-running jobs run in a new tmux window of the current session; clean up smoke-test artifacts afterward so the project tree stays clean.

## Scope

- Default: the whole repo — both **code** and **documentation** (`CLAUDE.md`, `README*`, docstrings, comments).
- If the invocation passes an argument (a path or subsystem, e.g. `/inspect AttentionGuidance`), restrict the inspection to that scope; leave everything else alone.

## Loop state

Track across iterations:
- `clean_streak` — consecutive inspections with 0 confirmed errors. Start at 0.
- `iteration` — 1-based counter; **incremented at the top of Step 1** on every pass.
- `MAX_ITERATIONS = 15` — safety backstop, **checked at the top of Step 1**.

**Convergence: stop when `clean_streak >= 3`. Hard stop when `iteration > MAX_ITERATIONS`** (report as non-converged).

## Step 0 — Preflight (once, before the loop)

1. Confirm the current branch is `rl-version` (`git branch --show-current`). If it is something else, stop and ask the user.
2. Confirm the conda env exists: `conda env list` must list `repldm`. Every Python run below uses `conda run -n repldm ...`.
3. Record the **pre-existing dirty set**: capture the literal output of `git status --short` at runtime and treat *exactly those paths* as off-limits — this runtime snapshot is the single source of truth, not any hard-coded example. They were already modified/untracked before /inspect started (WIP edits, untracked files/dirs). NEVER stage, commit, or delete any path in this set that the inspection did not itself modify — the loop only ever commits files it actually changes. (Gitignored artifacts such as `outputs/` and `__pycache__/` won't appear here; leave them alone regardless.)

## The iteration

### Step 1 — 检查 (independent inspection)

**At the top of every pass:** increment `iteration`. If `iteration > MAX_ITERATIONS`, STOP immediately → Final report (mark non-converged). Otherwise proceed.

Run a **fresh** inspection each iteration using multi-agent orchestration: spawn independent reviewer subagents (via the Agent/Task tool; if a workflow-orchestration tool is available, use it to fan out and adversarially verify). The reviewing agents must NOT be shown findings from any previous iteration; every inspection is independent. Fan out across these four dimensions in parallel, then adversarially verify every candidate before it counts:

- **D1 — Code correctness.** Logic bugs; wrong tensor ops / shapes / dtype / device placement; control-flow and off-by-one errors; misuse of the **reverse** `t_index` convention (pipelines call `attn_guidance(num_timesteps - 1 - i, latents, alpha)` — `t_index` is the reverse step index, not the timestep); scheduler/index errors; diffusers 0.21.4 API misuse.
- **D2 — Doc↔code consistency.** Every checkable claim in `CLAUDE.md` / `README` vs the actual code: parameter names, default values, file paths, function/class names, signatures, return types, exported symbols, branch names. Ground truth to check against:
  - Exports: `from AttentionGuidance import AttnGuidance`; `from InferencePipelines import RepLDMSDXLPipeline, RepLDMSDXLControlNetPipeline, FreeScaleSDXLPipeline`.
  - `AttnGuidance.__init__(dtype, device, num_total_steps, h, w, attn_type='vanilla', guidance_scale=0.001, guidance_density='all', guidance_scale_decay=None, power_calibrate=None, guidance_filter=None, attn_scaling=None)`; call signature `__call__(t_index, latents, alpha_t=None, scale=None)`.
  - Distinguish a real inconsistency (doc states a wrong default/name/path) from an example value the doc explicitly frames as "typical" (e.g. "~0.004–0.005" examples vs a `0.001` default) — only the former is an error.
- **D3 — Dead/broken references.** Broken imports, nonexistent file paths, dead URLs, undefined or unused symbols, references to renamed things.
- **D4 — Style & redundancy** (lower priority). Dead code, duplication, inconsistent naming. Flag ONLY when the fix is safe and clearly beneficial; never flag subjective style or anything that could change behavior.

Then, for **each** candidate finding, spawn independent skeptics (≥3 verifiers, distinct lenses) prompted to REFUTE it, defaulting to "refuted" when uncertain. Keep a finding only if a majority confirm it is a **real, actionable** error. Dedup by file+line. Step 1's output is the list of **confirmed errors** (each with `file:line` and a concrete fix).

To keep the streak meaningful, vary the agent/lens combination between iterations so three "clean" verdicts don't all share one blind spot.

### Step 2 — Converge or continue

- **0 confirmed errors** → `clean_streak += 1`; log `clean_streak/3`. If `clean_streak >= 3`, **STOP** → Final report. Otherwise loop back to Step 1 (subject to its iteration/`MAX_ITERATIONS` check) — no fix / test / commit this iteration, just re-inspect independently.
- **≥1 confirmed error** → `clean_streak = 0`; continue to Step 3.

### Step 3 — 改错 (fix)

Apply the **minimal correct** fix for each confirmed error, matching surrounding style. Do not opportunistically refactor beyond the confirmed findings.

### Step 4 — smoke test

Validate that the packages still import and the core module still runs, with no GPU and no model weights. Write the script to a temp dir **outside the repo** (the session scratchpad), and run it in the `repldm` env:

```python
# smoke_test.py — written to the scratchpad dir, deleted after the run
import torch
from AttentionGuidance import AttnGuidance
from InferencePipelines import (
    RepLDMSDXLPipeline,
    RepLDMSDXLControlNetPipeline,
    FreeScaleSDXLPipeline,
)

ag = AttnGuidance(torch.float32, "cpu", 10, 32, 32)
fake = torch.randn(1, 4, 32, 32)
out = ag(0, fake)                       # one TFSA / guidance step, CPU, no weights
assert out.shape == fake.shape, out.shape
print("SMOKE OK", tuple(out.shape))
```

Run: `conda run -n repldm python <scratchpad>/smoke_test.py`. This is a fast, foreground run — no tmux window needed. (Reserve a new tmux window for genuinely long-running/background jobs, per `CLAUDE.md`.)

If the smoke test **fails**, treat the failure as a new error: fix it and re-run. **Never commit code that fails the smoke test.**

### Step 5 — git 提交 (commit)

Stage **only** the files this iteration changed — list them explicitly; do NOT `git add -A` (that would sweep in the pre-existing dirty set from Step 0). Commit to `rl-version`:

```
inspect(iter N): <short summary of fixes>

- <file:line> — <what was wrong → how it was fixed>
...

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

Before committing, if `git diff --cached --quiet` reports no staged changes (the "fix" produced no net diff — e.g. it was content-identical or two edits cancelled), **skip the commit** for this iteration, note it in the log, and proceed to Step 6. Never create an empty commit.

If a confirmed fix must edit a file that was **already dirty** in the pre-existing set, commit only the inspection's own hunks (`git add -p` / targeted patch staging); if isolating the hunks is impractical, pause and report rather than committing the user's unrelated WIP.

### Step 6 — cleanup

Delete the scratchpad `smoke_test.py` (it lives outside the repo). `__pycache__/` is gitignored, so leave it as-is — no need to distinguish new from pre-existing `.pyc`. Confirm `git status --short` shows nothing beyond the intended commit and the pre-existing dirty set from Step 0; never touch pre-existing artifacts (`outputs/`, `pretrained_ckpts/`, weights, etc.).

Then loop back to **Step 1** (subject to its iteration/`MAX_ITERATIONS` check).

## Final report

When `clean_streak >= 3` (or `MAX_ITERATIONS` is hit), report:
- iterations run; per-iteration confirmed-error counts; the commits made (hashes + summaries);
- the final 3 clean inspections and which dimension/lens combinations confirmed them;
- if stopped at `MAX_ITERATIONS` without converging, say so plainly and list the still-open findings.

## Guardrails

- A finding counts as an "error" only after adversarial confirmation — this keeps the "0 errors" bar meaningful and avoids churn on false positives.
- Minimal diffs; never commit broken code; never touch the pre-existing dirty set or pre-existing artifacts.
- All Python runs through `conda run -n repldm`.
- Honor `CLAUDE.md`: background/long-running processes in a new tmux window; the project tree stays clean after every smoke test.
