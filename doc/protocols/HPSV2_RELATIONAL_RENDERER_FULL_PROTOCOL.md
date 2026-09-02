# HPSv2 Relational Renderer Full-Benchmark Protocol

## Question

This experiment tests one fixed hypothesis chosen before viewing HPSv2 output:
local relations in the frozen SDXL U-Net may provide a better latent correction
direction than ordinary local smoothing or a matched random direction. It does
not tune a policy and does not claim that the renderer is already suitable for
RL.

## Complete Benchmark

Use every official HPSv2 prompt: `anime`, `concept-art`, `paintings`, and
`photo`, with 800 rows per style and 3,200 rows total. Generate one image per
prompt and experiment setting, following the official benchmark convention.
The official source contains repeated prompt text; preserve every row and pair
results by `(style, style_index)`. A smoke run may test imports or one forward
pass, but its images must never be scored or reported as method evidence.

The prompt source is HPDv2 revision
`b9517430f34b1080d4f741118d2a440155a165cd`. The frozen CSV and source hashes
are recorded in `eval-pipeline/prompts/hpsv2_official_3200_manifest.json`.
Use one deterministic seed per official row: `20260831 + prompt_index`. This
gives all 3,200 rows distinct initial noise while keeping the four experiment
settings for a row exactly paired. Do not reset every row to the same noise.

## Fixed Experiment Settings

All four settings use the same SDXL checkpoint, initial noise, 50-step Euler
schedule, CFG 7.5, negative prompt, VAE, precision, and decoder:

| Experiment setting | Purpose |
|---|---|
| `no_ag` | Unmodified SDXL baseline |
| `feature_axis_r1_pos` | Proposed U-Net feature-affinity direction |
| `uniform_axis_r1_pos` | Tests whether plain local smoothing is enough |
| `random_axis_r1_pos` | Tests whether any matched direction produces the gain |

The three renderer settings use the same `axis-r1` graph, positive sign,
16x16 grid, scheduler-native Euler mapping, and target update ratio `0.02`.
They preserve channel mean and covariance and must use one ordinary U-Net call
per denoising step. No scale, sign, layer, step interval, or reward is changed
after generation starts.

## Execution and Audit

The complete matrix is `3,200 x 4 = 12,800` images. Assign whole prompt blocks
to GPUs 4, 5, 6, and 7 so all four paired settings for one prompt stay on the
same device. Each PNG must have an atomic JSON sidecar containing the prompt
row, seed, setting, image hash, device identity, scheduler values, U-Net call
count, renderer/provider diagnostics, and generation-attempt identity. Workers
write only inside a private attempt directory. The parent publishes files to
the canonical `images/` directory only after every worker, GPU watchdog, and
model post-audit succeeds. An atomic `accepted.json` binds the ordered task IDs
and every PNG/sidecar hash; attempts that fail or never reach this receipt are
poisoned and cannot be resumed. Each attempt contains at most 50 prompt blocks
per GPU (800 images across four GPUs), so a later failure preserves all earlier
accepted batches. The final manifest must contain exactly 12,800 distinct task
IDs and the complete Cartesian product; partial manifests cannot be scored.

The run contract also records the Python and generation-package versions,
CUDA/cuDNN runtime, driver, GPU UUID/PCI inventory, deterministic flags, and
the SHA-256 manifest for all 18 SDXL files selected by the fp16 loader. Each
worker copies exactly those files to its own read-only stage, verifies the tree,
loads through a pinned `/proc/self/fd` root, verifies the same tree before and
after loading, and removes the stage on exit. The parent also hashes the shared
source before generation and after worker success or failure. A resumed process
must reproduce the same environment record, and all four settings for one
prompt must report the same real GPU UUID and PCI address.

Run review, tests, commit, and push before launching. The worktree must remain
clean and the launch commit must exist on `origin/rl-version`. The queue pins
that commit when it starts and rechecks the branch, HEAD, remote-tracking ref,
clean worktree, and tracked entry points before generation, audit, scoring, and
analysis. A later code or protocol change fails the queue instead of mixing two
repository states in one experiment. The queue treats a GPU as free only when
it passes the memory/utilization limits and has no listed compute process for
two consecutive polls. It checks again immediately before launch. During
generation, the parent checks the compute-process list every two seconds and
permits only its four worker PIDs. After generation, the queue independently
waits for GPU 4 before starting the scorer, whose own watchdog permits only the
scorer PID. This is not an atomic cluster reservation, so a late process can
overlap briefly; once detected, the current stage fails instead of continuing
silently on a shared GPU. Abort signaling is lock-free, so an exited worker
cannot leave the parent waiting on a publication lock. The parent poisons every
private attempt as soon as monitoring or a worker fails. A late worker may finish
its private write, but the parent will never accept or publish files from that
attempt. Every system query has a ten-second timeout. A temporary query failure
keeps the waiting queue alive, while a monitoring failure during generation or
scoring fails closed.
Each started worker must emit exactly one readiness record and exactly one
completion record. The completion record is held until the parent has performed
the final GPU identity/process audit; a missing or duplicate record fails the
attempt even when all expected PNG/sidecar files exist.

## Scoring and Decision

Score all 12,800 images in strict mode with HPSv2.1, TOPIQ-NR, ImageReward,
patch ImageReward, CLIP, aesthetic score, and the registered pixel witnesses.
Report the official four style means, their overall average, and the official
population standard deviation across each style's ten contiguous 80-prompt
groups. Also report
paired 10,000-sample bootstrap intervals. Point estimates and official groups
retain all 3,200 rows, including the 28 duplicate rows. Confidence intervals
resample the 3,172 exact prompt-text clusters so repeated text is not treated as
independent evidence.
`analyze_hpsv2_relational_renderer.py` must re-audit the complete run, require
hash-bound strict scores, and write the style, group, paired-comparison, guard,
and plain-language report artifacts before the queue is considered complete.
For hash-bound runs, the scorer hashes the exact bytes it decodes and performs
another complete image audit after scoring, so cached or new scores cannot be
silently attached to replaced PNG files.
Scoring progress is written to a content-addressed private JSONL. One atomic
progress receipt points to the latest complete JSONL and binds the manifest, run
contract, scoring configuration, scorer weights, and CUDA execution record. A
new JSONL becomes resumable only when that receipt is replaced, so interruption
during a checkpoint leaves the previous checkpoint valid. Invalid or changed
progress is discarded, and a GPU-watchdog failure clears the whole private
attempt. The canonical `scores.jsonl` is published only after the watchdog exits
cleanly, followed by an atomic `scoring_success.json` that binds the complete
score-file hash and ordered task IDs. Analysis recomputes every binding; missing
receipts and changed finite score values both fail.
An already valid canonical score/receipt pair remains untouched during a rerun.
Before final publication, the scorer fsyncs content-addressed backups of the old
pair and an atomic transaction journal. Startup reconciles this journal before
loading any scorer: a fully written new pair is committed, otherwise the old
pair is restored byte for byte. Backups and the journal are removed only after a
directory fsync. This also covers `SIGKILL` and host failure between replacing
the score file and replacing its receipt; ordinary startup, model-loading, and
watchdog failures cannot erase an earlier completed evaluation.
The scoring YAML itself enables the exclusive-GPU watchdog even if a caller
omits the matching CLI flag. Every score row records the exact watchdog mode,
physical CUDA device, polling interval, and canonical contract hash; analysis
rejects absent, disabled, changed, or non-`cuda:4` monitoring provenance.
The queue also forces Hugging Face and Transformers offline during formal
scoring. Before any formal scorer plugin is imported, the process installs an
irreversible process-wide Linux seccomp filter, synchronized to every existing
thread, that denies socket and io_uring network syscalls. The process verifies
every `/proc/self/task/*` seccomp mode and rescans open sockets after installation;
cached Python functions, native code, new threads, and child processes inherit
the same denial. The Python socket guard remains only as an early, readable error.
Before model construction, all 31 required model, tokenizer, architecture, and
ImageReward source files are copied into a private read-only directory. This
inventory includes the OpenAI CLIP BPE vocabulary and HPS `ViT-H-14.json`; both
tokenizers use their staged vocabularies explicitly, and `ftfy`/`regex` versions
are bound. Models load only through the pinned directory descriptor; file
identity and SHA-256 are checked after every model load and after scoring. Compact
aggregate Python-source tree hashes bind `hpsv2`, OpenAI `clip`, `pyiqa`, and
`timm`; the private scorer package initializer and provenance builder are bound
as explicit implementation sources. These
formal loaders live in `hpsv2_scorers/`, leaving the shared `scorers/` bytes used
by older registered experiments unchanged. The copied-file manifests are part
of the registered scorer provenance, so scoring and analysis reject changed
implementations, packages, source files, preprocessing, or checkpoint bytes.
The repository scorer stores raw HPS cosine values; multiply them by 100 only
for the official table format and keep raw values in all paired calculations.

The feature direction advances only if its HPSv2 interval is above `no_ag`,
`uniform`, and `random`, while TOPIQ-NR, CLIP, clipping, saturation, and
contrast all pass the frozen guards. Otherwise this fixed direction cannot be
used as an RL, OPD, DPO, or distillation target. A positive result still needs
full compositional benchmarks and a prompt-disjoint human study before a paper
claim.
