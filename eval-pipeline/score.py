"""Config-driven scoring runner for the RepLDM eval-pipeline (env: `repldm_eval`).

Reads a run's manifest, runs the configured Scorers (self-contained metric modules
under scorers/, each decoupled from Sana), and writes scores.jsonl. Each metric is
declared in a yaml config (configs/eval_common.yaml). Weights are validated up front
so a missing metric is SKIPPED with a warning instead of crashing the run.

Resume is additive when enabled metrics and their execution provenance still match.
Strict runs bind source, package/runtime, model asset, and preprocessing metadata;
any drift invalidates the row before old columns can be reused. Output is rewritten
atomically every 50 images.

  /home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/score.py \
      --run_dir outputs/exp_spectral_headroom/pilot --device cuda:0 --strict
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
import sys
import tempfile

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

import yaml  # noqa: E402
import scorers  # noqa: E402,F401  (import registers all metrics)
import scorers.base as scorer_base  # noqa: E402
from scorers.base import REGISTRY  # noqa: E402
from scorer_provenance import (  # noqa: E402
    SCORER_PROVENANCE_SCHEMA,
    build_scorer_provenance,
    registered_scorer_provenance_contract,
    validate_hardened_score_rows,
)
from s7_provenance import (  # noqa: E402
    PROVENANCE_SCHEMA,
    image_sha256,
    json_sha256,
    validate_design_rows,
    validate_run_contract,
    validate_scores_against_manifest,
    validate_sidecar,
)


CFG_SCORING_SCHEMA = "cfg_scoring_contract_v1"
CFG_ACTION_SCHEMA = "cfg_baselines_v1"
CFG_SCORING_METRICS = ("pixel", "clip", "hps", "iqa")
CFG_SCORING_PARAMS = {
    "patch_crops": 5,
    "clip_model": "ViT-B/32",
    "clipscore_w": 2.5,
}
CFG_REQUIRED_SCORE_KEYS = (
    "colorfulness",
    "laplacian_sharpness",
    "clipped_fraction",
    "mean_saturation",
    "contrast_std",
    "clip_cosine",
    "clipscore",
    "hpsv2",
    "topiq_nr",
)
CFG_SCORING_FIELDS = {"metrics", "strict", "params", "required_score_keys"}


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


def finite_numeric(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def required_scorer_provenance_schema(cfg, run_config, cli_required=False):
    """Resolve an optional fail-closed provenance requirement."""
    requested = []
    provenance_config = cfg.get("scorer_provenance")
    if provenance_config is not None:
        if not isinstance(provenance_config, dict) or set(provenance_config) != {
            "required_schema"
        }:
            raise RuntimeError(
                "scorer_provenance config must contain only required_schema"
            )
        requested.append(provenance_config.get("required_schema"))
    run_requirement = run_config.get("required_scorer_provenance_schema")
    if run_requirement is not None:
        requested.append(run_requirement)
    if cli_required:
        requested.append(SCORER_PROVENANCE_SCHEMA)
    if not requested:
        return None
    if any(value != SCORER_PROVENANCE_SCHEMA for value in requested):
        raise RuntimeError(
            f"unsupported scorer provenance requirement; expected "
            f"{SCORER_PROVENANCE_SCHEMA!r}"
        )
    return SCORER_PROVENANCE_SCHEMA


@contextmanager
def scoring_output_lock(run_dir):
    """Exclude generation and concurrent scoring from one run directory."""
    # Generation uses the same lock file, so scoring cannot consume a manifest
    # while a resumed generator may still replace images or sidecars.
    lock_path = os.path.join(os.path.abspath(run_dir), ".generate.lock")
    handle = open(lock_path, "a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "generation or scoring is already running for "
                f"{os.path.abspath(run_dir)}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextmanager
def atomic_text_writer(path):
    """Publish a text file through a unique same-directory temporary file."""
    destination = os.path.abspath(path)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.",
        suffix=".tmp",
        dir=os.path.dirname(destination),
    )
    handle = None
    try:
        handle = os.fdopen(fd, "w", encoding="utf-8")
        with handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if handle is None:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def cfg_scoring_contract(run_config, metric_names, params, strict):
    """Load and enforce the scoring recipe bound by a registered CFG run."""
    if run_config.get("cfg_baseline_registered") is not True:
        return None
    actions_path = run_config.get("actions_yaml")
    actions_hash = run_config.get("actions_sha256")
    if not isinstance(actions_path, str) or not os.path.isfile(actions_path):
        raise RuntimeError("registered CFG run actions YAML is unavailable for scoring")
    with open(actions_path, "rb") as handle:
        actions_bytes = handle.read()
    if hashlib.sha256(actions_bytes).hexdigest() != actions_hash:
        raise RuntimeError("registered CFG run actions YAML changed before scoring")
    actions_config = yaml.safe_load(actions_bytes) or {}
    if not isinstance(actions_config, dict):
        raise RuntimeError("registered CFG actions YAML must contain a mapping")
    if actions_config.get("schema") != CFG_ACTION_SCHEMA:
        raise RuntimeError("registered CFG actions YAML has the wrong schema")
    scoring = actions_config.get("scoring")
    if not isinstance(scoring, dict):
        raise RuntimeError("registered CFG actions lack a scoring contract")
    if set(scoring) != CFG_SCORING_FIELDS:
        raise RuntimeError("registered CFG scoring fields differ from the v1 contract")
    registered_metrics = scoring.get("metrics")
    registered_params = scoring.get("params")
    if registered_metrics != list(CFG_SCORING_METRICS):
        raise RuntimeError("registered CFG YAML metrics differ from the v1 contract")
    if list(metric_names) != list(CFG_SCORING_METRICS):
        raise RuntimeError(
            f"registered CFG scoring requires metrics {list(CFG_SCORING_METRICS)}, "
            f"got {list(metric_names)}"
        )
    if strict is not True or scoring.get("strict") is not True:
        raise RuntimeError("registered CFG scoring requires --strict")
    if json_sha256(registered_params) != json_sha256(CFG_SCORING_PARAMS):
        raise RuntimeError("registered CFG YAML params differ from the v1 contract")
    if json_sha256(params) != json_sha256(CFG_SCORING_PARAMS):
        raise RuntimeError("scoring config params differ from CFG registration")
    if scoring.get("required_score_keys") != list(CFG_REQUIRED_SCORE_KEYS):
        raise RuntimeError(
            "registered CFG required score keys differ from the v1 contract"
        )
    return {
        "schema": CFG_SCORING_SCHEMA,
        "action_schema": CFG_ACTION_SCHEMA,
        "metrics": list(CFG_SCORING_METRICS),
        "strict": True,
        "params": dict(CFG_SCORING_PARAMS),
        "required_score_keys": list(CFG_REQUIRED_SCORE_KEYS),
        "actions_sha256": actions_hash,
    }


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
    ap.add_argument(
        "--require-scorer-provenance",
        action="store_true",
        help="require the hardened scorer provenance schema (implies --strict)",
    )
    args = ap.parse_args()
    if args.require_scorer_provenance:
        args.strict = True

    with scoring_output_lock(args.run_dir):
        return _score_run(args, ap)


def _score_run(args, ap):
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise RuntimeError("scoring config must contain a YAML mapping")
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
    if (
        run_config.get("split_role") == "engineering_smoke"
        or run_config.get("engineering_only") is True
    ):
        raise RuntimeError(
            "structural engineering smoke forbids quality scoring"
        )
    required_provenance_schema = required_scorer_provenance_schema(
        cfg,
        run_config,
        getattr(args, "require_scorer_provenance", False),
    )
    if required_provenance_schema is not None and not args.strict:
        raise RuntimeError("hardened scorer provenance requires --strict")
    s7_run = any(
        run_config.get(flag) is True
        for flag in (
            "trajectory_registered",
            "scheduler_baseline_registered",
            "cfg_baseline_registered",
            "native_renderer_registered",
        )
    ) or any(
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
    registered_scoring = cfg_scoring_contract(
        run_config, metric_names, params, args.strict
    )
    registered_scoring_sha256 = (
        json_sha256(registered_scoring) if registered_scoring is not None else None
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

    scorer_provenance = None
    scorer_provenance_sha256 = None
    if args.strict:
        scorer_provenance, scorer_provenance_sha256 = build_scorer_provenance(
            active,
            params=params,
            device=device,
            runner_path=__file__,
            base_path=scorer_base.__file__,
            source_root=THIS,
        )
    registered_scorer_contract = None
    registered_scorer_hash = None
    for registration in (cfg, run_config):
        candidate_contract, candidate_hash = registered_scorer_provenance_contract(
            registration
        )
        if candidate_contract is not None:
            if (
                registered_scorer_contract is not None
                and candidate_contract != registered_scorer_contract
            ):
                raise RuntimeError("registered scorer provenance contracts disagree")
            registered_scorer_contract = candidate_contract
        if candidate_hash is not None:
            if (
                registered_scorer_hash is not None
                and candidate_hash != registered_scorer_hash
            ):
                raise RuntimeError("registered scorer provenance hashes disagree")
            registered_scorer_hash = candidate_hash
    if registered_scorer_hash is not None:
        if scorer_provenance is None:
            raise RuntimeError(
                "a registered scorer provenance contract requires --strict"
            )
        if scorer_provenance_sha256 != registered_scorer_hash:
            raise RuntimeError(
                "loaded scorer provenance differs from the registered contract"
            )
        if registered_scorer_contract is not None and (
            scorer_provenance != registered_scorer_contract
        ):
            raise RuntimeError(
                "loaded scorer provenance differs from the registered payload"
            )

    output_keys = [key for _, scorer in active for key, _ in scorer.OUTPUT_KEYS]
    if len(output_keys) != len(set(output_keys)):
        raise RuntimeError("active scorers declare duplicate output keys")
    need_keys = set(output_keys)
    if registered_scoring is not None and need_keys != set(CFG_REQUIRED_SCORE_KEYS):
        raise RuntimeError(
            "registered CFG scorer outputs differ from required_score_keys"
        )

    def score_provenance_is_current(row, score):
        if s7_run:
            if (
                score.get("image_sha256") != row.get("image_sha256")
                or score.get("run_contract_sha256")
                != row.get("run_contract_sha256")
            ):
                return False
            if registered_scoring is not None and not (
                score.get("scoring_contract") == registered_scoring
                and score.get("scoring_contract_sha256")
                == registered_scoring_sha256
            ):
                return False
        if scorer_provenance is not None:
            return (
                score.get("scorer_provenance") == scorer_provenance
                and score.get("scorer_provenance_sha256")
                == scorer_provenance_sha256
            )
        return True

    def score_is_current(row):
        if row["id"] not in existing or not need_keys.issubset(existing[row["id"]].keys()):
            return False
        score = existing[row["id"]]
        if not score_provenance_is_current(row, score):
            return False
        return not (args.strict or s7_run) or all(
            finite_numeric(score.get(key)) for key in need_keys
        )

    todo = [r for r in manifest if not score_is_current(r)]
    print(f"{len(manifest)} images; {len(todo)} to (re)score with {[n for n, _ in active]}", flush=True)

    def flush():
        with atomic_text_writer(scores_path) as out:
            for r in manifest:
                if r["id"] in existing:
                    out.write(
                        json.dumps(
                            existing[r["id"]],
                            allow_nan=not args.strict and registered_scoring is None,
                        )
                        + "\n"
                    )

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
        prior = existing.get(r["id"])
        rec = (
            prior
            if prior is not None and score_provenance_is_current(r, prior)
            else {key: r[key] for key in metadata_keys if key in r}
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
            if registered_scoring is not None:
                rec.update(
                    {
                        "scoring_contract": registered_scoring,
                        "scoring_contract_sha256": registered_scoring_sha256,
                    }
                )
        if scorer_provenance is not None:
            rec.update(
                {
                    "scorer_provenance": scorer_provenance,
                    "scorer_provenance_sha256": scorer_provenance_sha256,
                }
            )
        for name, sc in active:
            scorer_keys = {key for key, _ in sc.OUTPUT_KEYS}
            if scorer_keys.issubset(rec) and (
                not (args.strict or s7_run)
                or all(finite_numeric(rec.get(key)) for key in scorer_keys)
            ):
                continue
            for key in scorer_keys:
                rec.pop(key, None)
            try:
                scored = sc.score_image(img, r["prompt"])
                if registered_scoring is not None and (
                    not isinstance(scored, dict) or set(scored) != scorer_keys
                ):
                    raise ValueError(
                        f"{name} returned fields outside its registered output contract"
                    )
                if (args.strict or s7_run) and any(
                    not finite_numeric(scored.get(key)) for key in scorer_keys
                ):
                    raise ValueError(f"{name} returned non-finite strict scores")
                rec.update(scored)
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
    if scorer_provenance is not None:
        validate_hardened_score_rows(
            [existing[row["id"]] for row in manifest],
            required_schema=required_provenance_schema
            or SCORER_PROVENANCE_SCHEMA,
            expected_sha256=registered_scorer_hash,
            expected_contract=registered_scorer_contract,
        )
    print(f"scores -> {scores_path}", flush=True)


if __name__ == "__main__":
    main()
