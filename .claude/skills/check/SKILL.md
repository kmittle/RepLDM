---
name: check
description: Pre-commit review of ONLY the git uncommitted changes (working tree vs HEAD) — independent multi-agent inspection → fix → lightweight smoke test — looping until 3 consecutive independent inspections find zero errors in the change set, then commit the cleaned changes once. Use before committing to sanity-check pending work. For a whole-repo audit, use /inspect instead.
---

# /check — closed-loop review of uncommitted changes

Like `/inspect`, but **scoped to only the git uncommitted changes**, run as a pre-commit gate. Repeat

    检查 (inspect) → 改错 (fix) → smoke test

until **3 consecutive, independent inspections each find 0 confirmed errors in the change set**, then commit the cleaned changes **once**. Everything else — the four inspection dimensions, adversarial verification, the lightweight smoke test, convergence, and the `CLAUDE.md` conventions — is identical to `/inspect`; only the scope and the commit model differ.

## Scope — the change set under review

Pin the **baseline** = the HEAD commit at the moment /check starts (`git rev-parse HEAD`). The loop makes **no commits until the end**, so HEAD stays fixed throughout. The **change set** is everything uncommitted relative to that baseline:
- tracked modifications, staged **and** unstaged: `git diff --name-only HEAD`
- untracked, non-ignored new files: `git ls-files --others --exclude-standard`

This change set is the **only** thing inspected — do not audit the rest of the repo. As the loop applies fixes, they land in the working tree and are automatically picked up and re-inspected on the next pass. Inspect **source, documentation, and config** files in the change set; skip binaries / weights / large data (nothing to check there).

If the invocation passes a path argument, intersect the change set with that path.

If the change set is empty at Step 0, report "nothing to check" and exit.

## Loop state

- `clean_streak` — consecutive inspections with 0 confirmed errors. Start at 0.
- `iteration` — 1-based counter; **incremented at the top of Step 1** on every pass.
- `MAX_ITERATIONS = 15` — safety backstop, **checked at the top of Step 1**.

**Convergence: stop looping when `clean_streak >= 3`** → Finalize. **Hard stop when `iteration > MAX_ITERATIONS`** → non-converged: do NOT commit, report open findings.

## Step 0 — Preflight (once, before the loop)

1. Record the current branch — the eventual commit target. (No specific branch is required; /check commits your pending work where you are. If the current branch is a protected/default branch — `main` or `master`, or whatever `git symbolic-ref --short refs/remotes/origin/HEAD` reports — pause and confirm before the final commit.)
2. Confirm the conda env exists: `conda env list` must list `repldm`. Every Python run below uses `conda run -n repldm ...`.
3. Pin `BASELINE = git rev-parse HEAD` and capture the **change set** (above). If it is empty, exit with "nothing to check".

## The iteration

### Step 1 — 检查 (independent inspection of the change set)

**At the top of every pass:** increment `iteration`. If `iteration > MAX_ITERATIONS`, STOP immediately → Final report (mark non-converged, no commit). Otherwise proceed.

Run a **fresh** inspection each iteration using multi-agent orchestration: spawn independent reviewer subagents (via the Agent/Task tool; if a workflow-orchestration tool is available, use it to fan out and adversarially verify). The reviewing agents must NOT be shown findings from any previous iteration; every inspection is independent. Inspect **only the change set**, across these four dimensions in parallel, then adversarially verify every candidate before it counts:

- **D1 — Code correctness** (changed code). Logic bugs; wrong tensor ops / shapes / dtype / device placement; control-flow and off-by-one errors; misuse of the **reverse** `t_index` convention (pipelines call `attn_guidance(num_timesteps - 1 - i, latents, alpha)` — `t_index` is the reverse step index, not the timestep); scheduler/index errors; diffusers 0.21.4 API misuse.
- **D2 — Doc↔code consistency, reaching beyond the diff.** If the changed **code** makes any claim in `CLAUDE.md` / `README` stale (parameter names, default values, paths, signatures, return types, exported symbols, branch names), that is an error **even though the doc file is not in the change set** — and vice versa for a changed doc that now contradicts current code. Ground truth: `from AttentionGuidance import AttnGuidance`; `from InferencePipelines import RepLDMSDXLPipeline, RepLDMSDXLControlNetPipeline, FreeScaleSDXLPipeline`; `AttnGuidance.__init__(dtype, device, num_total_steps, h, w, attn_type='vanilla', guidance_scale=0.001, guidance_density='all', guidance_scale_decay=None, power_calibrate=None, guidance_filter=None, attn_scaling=None)`; `__call__(t_index, latents, alpha_t=None, scale=None)`. Distinguish a real inconsistency from a value the doc explicitly frames as a "typical" example.
- **D3 — Dead/broken references introduced by the changes.** Broken imports, nonexistent paths, dead URLs, undefined/unused symbols — **including references elsewhere in the repo that the change breaks** (e.g. a symbol renamed in a changed file but still imported by an unchanged file).
- **D4 — Style & redundancy** (lower priority, changed code). Dead code, duplication, inconsistent naming. Flag ONLY when the fix is safe and clearly beneficial; never flag subjective style or anything that could change behavior.

Then, for **each** candidate finding, spawn independent skeptics (≥3 verifiers, distinct lenses) prompted to REFUTE it, defaulting to "refuted" when uncertain. Keep a finding only if a majority confirm it is a **real, actionable** error. Dedup by file+line. Step 1's output is the list of **confirmed errors** (each with `file:line` and a concrete fix).

To keep the streak meaningful, vary the agent/lens combination between iterations so three "clean" verdicts don't all share one blind spot.

### Step 2 — Converge or continue

- **0 confirmed errors** → `clean_streak += 1`; log `clean_streak/3`. If `clean_streak >= 3`, **exit the loop → Finalize**. Otherwise loop back to Step 1 (subject to its iteration/`MAX_ITERATIONS` check) — no fix / smoke test this iteration, just re-inspect independently.
- **≥1 confirmed error** → `clean_streak = 0`; continue to Step 3.

### Step 3 — 改错 (fix)

Apply the **minimal correct** fix for each confirmed error, matching surrounding style. Do not opportunistically refactor beyond the confirmed findings. (Fixes land in the working tree and extend the change set under review.)

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

Run: `conda run -n repldm python <scratchpad>/smoke_test.py`. Fast, foreground — no tmux window needed (reserve a new tmux window for genuinely long-running/background jobs, per `CLAUDE.md`).

If the smoke test **fails**, treat the failure as a new error: fix it and re-run. **Never proceed on a failing smoke test.**

Then loop back to **Step 1** (subject to its iteration/`MAX_ITERATIONS` check).

## Finalize — single commit (only after `clean_streak >= 3`)

1. **Re-run the smoke test once** to confirm the final to-be-committed state passes. If it fails, do NOT commit — treat as non-converged, report, and leave the working tree as-is.
2. **Stage the reviewed change set explicitly, by path.** Stage the tracked changes plus exactly the untracked files that were inspected as source/doc/config in Step 1 (i.e. those that passed the binaries/weights/symlink/data exclusion). Echo the explicit file list for confirmation. Do NOT `git add -A` blindly, and do NOT stage untracked artifacts that were excluded from inspection.
3. **Safety guard — what gets committed:** before committing, list exactly what is staged. If it includes anything that should not be tracked — model weights / large binaries (`*.safetensors`, `*.ckpt`, `*.pt`, `*.pth`, `*.bin`, `*.onnx`, `*.h5`, …), symlinks (e.g. `pretrained_ckpts/`), datasets, or other non-source artifacts — **exclude it and confirm with the user**. For an **un-gitignored** weights symlink/dir that keeps reappearing in the change set (e.g. `pretrained_ckpts/`), recommend the durable fix: offer to append it to `.gitignore`, then re-capture the change set — a one-time exclude does not converge, since the path returns on every future run. If the current branch is protected/default (see Step 0), also confirm before committing.
4. Commit **once** to the current branch:

```
check: <summary of the reviewed change set + fixes applied>

- <what was reviewed>
- <file:line> — <error found → fix> (for each fix)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

5. **Cleanup:** delete the scratchpad `smoke_test.py`. `__pycache__/` is gitignored — leave it as-is. Confirm `git status --short` is as expected (the change set is now committed; nothing stray remains).

## Final report

- iterations run; per-iteration confirmed-error counts; the fixes applied;
- the commit hash + subject — or, if not committed, the reason ("non-converged at MAX_ITERATIONS" / "awaiting your confirmation on weights/main");
- the final 3 clean inspections and which dimension/lens combinations confirmed them;
- if stopped at `MAX_ITERATIONS` without converging: say so plainly, list the still-open findings, and **leave the working tree as-is** (no commit).

## Guardrails

- A finding counts as an "error" only after adversarial confirmation — this keeps the "0 errors" bar meaningful and avoids churn on false positives.
- Inspect **only the change set** (plus what it directly affects: docs describing changed code, and references the change breaks). Do not audit the whole repo — that's `/inspect`.
- Minimal diffs; **never commit broken or non-converged code**; the single final commit needs explicit staging plus the weights/binary safety guard.
- All Python runs through `conda run -n repldm`.
- Honor `CLAUDE.md`: background/long-running processes in a new tmux window; the project tree stays clean after every smoke test.
