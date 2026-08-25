"""Config-driven scoring runner for the RepLDM eval-pipeline (env: `repldm_eval`).

Reads a run's manifest, runs the configured Scorers (self-contained metric modules
under scorers/, each decoupled from Sana), and writes scores.jsonl. Each metric is
declared in a yaml config (configs/eval_common.yaml). Weights are validated up front
so a missing metric is SKIPPED with a warning instead of crashing the run.

Resume is ADDITIVE: a row is recomputed only if it lacks an enabled metric's columns,
and existing columns are preserved — so you can add CLIP/HPSv2/aesthetic to an already-
scored run without redoing ImageReward. Output is rewritten atomically every 50 images.

  /home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/score.py \
      --run_dir outputs/exp_spectral_headroom/pilot --device cuda:0 --strict
"""
import argparse
import json
import os
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

import yaml  # noqa: E402
import scorers  # noqa: E402,F401  (import registers all metrics)
from scorers.base import REGISTRY  # noqa: E402
from s7_provenance import (  # noqa: E402
    PROVENANCE_SCHEMA,
    image_sha256,
    validate_design_rows,
    validate_run_contract,
    validate_scores_against_manifest,
    validate_sidecar,
)


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _unique_rows(rows, label):
    result = {}
    for row in rows:
        row_id = str(row.get("id", ""))
        if not row_id or row_id in result:
            raise ValueError(f"{label} contains duplicate or empty id {row_id!r}")
        result[row_id] = row
    return result


def resolve_device(device, cuda_available):
    """Normalize CLI GPU indices to the Torch ``cuda:N`` device form."""
    value = str(device).strip()
    if value.isdecimal():
        value = f"cuda:{value}"
    if value != "cpu" and not cuda_available:
        return "cpu"
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--config", default=os.path.join(THIS, "configs", "eval_common.yaml"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--metrics", default=None, help="comma-separated override of config 'metrics'")
    ap.add_argument(
        "--strict", action="store_true",
        help="fail if any requested scorer cannot load or score an image",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    import torch
    device = resolve_device(args.device, torch.cuda.is_available())
    try:
        torch.device(device)
    except (RuntimeError, TypeError) as exc:
        ap.error(f"invalid --device {args.device!r}: {exc}")
    metric_names = args.metrics.split(",") if args.metrics else cfg.get("metrics", [])
    params = dict(cfg.get("params", {}))

    manifest = load_jsonl(os.path.join(args.run_dir, "manifest.jsonl"))
    if not manifest:
        raise RuntimeError("manifest.jsonl is empty or missing")
    manifest_by_id = _unique_rows(manifest, "manifest")
    run_config_path = os.path.join(args.run_dir, "config.json")
    run_config = {}
    if os.path.isfile(run_config_path):
        with open(run_config_path) as handle:
            run_config = json.load(handle)
    s7_run = bool(run_config.get("trajectory_registered")) or any(
        row.get("provenance_schema") == PROVENANCE_SCHEMA for row in manifest
    )
    if s7_run:
        contract_hash = validate_run_contract(run_config)
        action_ids = [str(action.get("id")) for action in run_config.get("actions", [])]
        seeds = [int(value) for value in run_config.get("seeds", [])]
        validate_design_rows(
            manifest,
            expected_action_ids=action_ids or None,
            expected_seeds=seeds or None,
        )
        for row in manifest:
            if row.get("provenance_schema") != PROVENANCE_SCHEMA:
                raise RuntimeError(f"{row.get('id')}: missing S7 provenance schema")
            validate_sidecar(
                row,
                args.run_dir,
                expected_contract_sha256=contract_hash,
            )
    scores_path = os.path.join(args.run_dir, "scores.jsonl")
    existing_rows = load_jsonl(scores_path)
    existing = _unique_rows(existing_rows, "scores")
    if s7_run and existing_rows:
        try:
            validate_scores_against_manifest(manifest, existing_rows)
        except ValueError:
            # Old or stale rows are discarded and recomputed below.  Keeping
            # them would silently attach metrics to a replaced image.
            existing = {}

    # instantiate scorers (validate weights first; skip cleanly if missing/broken)
    active = []
    unavailable = []
    for name in metric_names:
        if name not in REGISTRY:
            message = f"unknown metric '{name}'"
            print(f"[warn] {message}, skipping", flush=True)
            unavailable.append(message)
            continue
        cls = REGISTRY[name]
        ready, msg = cls.weights_status(**params)
        if not ready:
            message = f"'{name}' weights missing -> {msg}"
            print(f"[skip] {message}", flush=True)
            unavailable.append(message)
            continue
        try:
            active.append((name, cls(device=device, **params)))
            print(f"[ok] loaded scorer '{name}' on {device}", flush=True)
        except Exception as e:
            message = f"'{name}' failed to init -> {e}"
            print(f"[skip] {message}", flush=True)
            unavailable.append(message)
    if args.strict and unavailable:
        raise RuntimeError("requested scorers unavailable: " + "; ".join(unavailable))
    if not active:
        print("no active scorers; nothing to do", flush=True)
        return

    need_keys = {k for _, sc in active for k, _ in sc.OUTPUT_KEYS}
    def score_is_current(row):
        if row["id"] not in existing or not need_keys.issubset(existing[row["id"]].keys()):
            return False
        if not s7_run:
            return True
        score = existing[row["id"]]
        return (
            score.get("image_sha256") == row.get("image_sha256")
            and score.get("run_contract_sha256") == row.get("run_contract_sha256")
        )

    todo = [r for r in manifest if not score_is_current(r)]
    print(f"{len(manifest)} images; {len(todo)} to (re)score with {[n for n, _ in active]}", flush=True)

    def flush():
        tmp = scores_path + ".tmp"
        with open(tmp, "w") as out:
            for r in manifest:
                if r["id"] in existing:
                    out.write(json.dumps(existing[r["id"]]) + "\n")
        os.replace(tmp, scores_path)

    from PIL import Image
    for i, r in enumerate(todo):
        image_path = os.path.join(args.run_dir, r["image_path"])
        if s7_run and image_sha256(image_path) != r.get("image_sha256"):
            raise RuntimeError(f"{r['id']}: image changed after manifest validation")
        img = Image.open(image_path).convert("RGB")
        metadata_keys = (
            "id", "prompt_index", "bucket", "seed", "scale", "action_id",
            "action_type", "band_scales", "image_path",
        )
        rec = existing.get(
            r["id"], {key: r[key] for key in metadata_keys if key in r}
        )
        if s7_run:
            rec.update(
                {
                    "provenance_schema": PROVENANCE_SCHEMA,
                    "image_sha256": r["image_sha256"],
                    "run_contract_sha256": r["run_contract_sha256"],
                    "action_sha256": r.get("action_sha256"),
                }
            )
        for name, sc in active:
            scorer_keys = {key for key, _ in sc.OUTPUT_KEYS}
            if scorer_keys.issubset(rec):
                continue
            try:
                rec.update(sc.score_image(img, r["prompt"]))
            except Exception as e:
                print(f"[warn] {name} failed on {r['id']}: {e}", flush=True)
                if args.strict:
                    existing[r["id"]] = rec
                    flush()
                    raise RuntimeError(f"{name} failed on {r['id']}") from e
        existing[r["id"]] = rec
        if (i + 1) % 50 == 0 or i == len(todo) - 1:
            flush()
            print(f"  scored {i + 1}/{len(todo)}", flush=True)
    flush()
    if s7_run:
        validate_scores_against_manifest(manifest, [existing[row["id"]] for row in manifest])
    print(f"scores -> {scores_path}", flush=True)


if __name__ == "__main__":
    main()
