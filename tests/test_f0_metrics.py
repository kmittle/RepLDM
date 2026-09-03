from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct

import pytest
import torch

from latent_renderer_training.f0_metrics import (
    BOOTSTRAP_METHOD,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    build_f0_metric_action,
    F0_METRIC_ROW_SCHEMA,
    F0_PIXEL_SUMMARY_SCHEMA,
    F0_QUERY_BUDGET,
    build_f0_power_artifact,
    build_f0_screen_registration,
    compute_f0_pixel_summary,
    compute_f0_phase_summary,
    f0_screen_design,
    validate_f0_phase_evidence,
    validate_f0_power_evidence,
    validate_f0_screen_registration_evidence,
    write_f0_phase_evidence,
    write_f0_power_evidence,
    write_f0_screen_registration_evidence,
)
from latent_renderer_training.gates import validate_f0_gate
from latent_renderer_training.operations import tensor_operation_output_sha256
from latent_renderer_training.witnesses import (
    TOPIQ_NR_MODEL_ID,
    TOPIQ_NR_PREPROCESS_SHA256,
    TOPIQ_NR_ROLE,
)


_F0_CONTRACT_HASH = "e" * 64
_REWARD_STATISTICS_HASH = "a" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _binding(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _write_bytes(path: Path, payload: bytes) -> dict[str, object]:
    path.write_bytes(payload)
    return _binding(path)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _reservation_id(label: str) -> str:
    value = _digest(label)[:32]
    return f"{value[:8]}-{value[8:12]}-{value[12:16]}-{value[16:20]}-{value[20:]}"


def _score_output_sha256(value: object) -> str:
    tensor_hash = hashlib.sha256(struct.pack("<f", float(value))).hexdigest()
    return tensor_operation_output_sha256(tensor_hash)


_TEST_INITIAL_RENDERER = "8" * 64
_TEST_OPSD_RENDERER = "c" * 64
_TEST_CROSSFIT_RENDERERS = {
    str(fold): _digest(f"crossfit-renderer:{fold}") for fold in range(4)
}


def _row_provenance(
    row: dict[str, object],
    *,
    initial_renderer_state_sha256: str,
    crossfit_renderer_state_sha256: dict[str, str],
    opsd_teacher_renderer_sha256: str,
) -> dict[str, object]:
    phase = str(row["phase"])
    prompt = str(row["prompt_id"])
    seed = int(row["generation_seed"])
    decision = int(row["decision_index"])
    held_out_fold = int(prompt.rsplit("-", 1)[1]) % 4 if phase == "train" else None
    realization = "out_of_fold_student" if phase == "train" else "final_teacher"
    branches: dict[str, object] = {}
    for branch, value_field in (
        ("anchor", "anchor"),
        ("plus", "plus"),
        ("minus", "minus"),
        ("realized", realization),
    ):
        identity = f"{phase}:{prompt}:{seed}:{branch}"
        if branch != "anchor":
            identity += f":{decision}"
        renderer_state_sha256 = (
            initial_renderer_state_sha256
            if branch != "realized"
            else crossfit_renderer_state_sha256[str(held_out_fold)]
            if phase == "train"
            else opsd_teacher_renderer_sha256
        )
        branches[branch] = {
            "reuse_scope": "shared_prompt_seed" if branch == "anchor" else "forbidden",
            "renderer_state_sha256": renderer_state_sha256,
            "image_sha256": _digest(f"image:{identity}"),
            "image_reward": {
                "reservation_id": _reservation_id(f"reward:{identity}"),
                "output_sha256": _score_output_sha256(
                    row["reward_select"][value_field]
                ),
            },
            "topiq_nr": {
                "reservation_id": _reservation_id(f"witness:{identity}"),
                "output_sha256": _score_output_sha256(row["topiq_nr"][value_field]),
            },
        }
    return {
        "target_record_sha256": _digest(
            f"target:{phase}:{prompt}:{seed}:{decision}"
        ),
        "pixel_summary_schema": F0_PIXEL_SUMMARY_SCHEMA,
        "held_out_fold": held_out_fold,
        "branches": branches,
    }


def _passing_rows(
    phase: str,
    *,
    initial_renderer_state_sha256: str = _TEST_INITIAL_RENDERER,
    crossfit_renderer_state_sha256: dict[str, str] | None = None,
    opsd_teacher_renderer_sha256: str = _TEST_OPSD_RENDERER,
) -> list[dict[str, object]]:
    crossfit_renderer_state_sha256 = (
        dict(_TEST_CROSSFIT_RENDERERS)
        if crossfit_renderer_state_sha256 is None
        else dict(crossfit_renderer_state_sha256)
    )
    if phase == "train":
        prompt_count = 64
        seeds = (2026090101, 2026090102)
        realization_field = "out_of_fold_student"
        realization_pixel_prefix = "out_of_fold"
    elif phase == "validation":
        prompt_count = 32
        seeds = (2026090191, 2026090192)
        realization_field = "final_teacher"
        realization_pixel_prefix = "final_teacher"
    else:
        raise ValueError(phase)
    rows = [
        {
            "schema": F0_METRIC_ROW_SCHEMA,
            "phase": phase,
            "prompt_id": f"{phase}-{prompt_index:02d}",
            "generation_seed": seed,
            "decision_index": decision,
            "valid_nonzero_gradient": True,
            "violations": {
                "no_op_parity": False,
                "scheduler_parity": False,
                "finite_value": False,
                "moment": False,
                "hard_cap": False,
            },
            "reward_select": {
                "anchor": 0.0,
                "plus": 0.20,
                "minus": -0.05,
                realization_field: 0.10,
            },
            "topiq_nr": {
                "anchor": 0.50,
                "plus": 0.51,
                "minus": 0.50,
                realization_field: 0.51,
            },
            "pixel": {
                "anchor_clipped_fraction": 0.01,
                "plus_clipped_fraction": 0.01,
                "minus_clipped_fraction": 0.01,
                f"{realization_pixel_prefix}_clipped_fraction": 0.01,
                "anchor_mean_saturation": 0.20,
                "plus_mean_saturation": 0.20,
                "minus_mean_saturation": 0.20,
                f"{realization_pixel_prefix}_mean_saturation": 0.20,
                "anchor_contrast": 1.0,
                "plus_contrast": 1.0,
                "minus_contrast": 1.0,
                f"{realization_pixel_prefix}_contrast": 1.0,
            },
            "target_cap": {"plus_at_98pct": False, "minus_at_98pct": False},
        }
        for prompt_index in range(prompt_count)
        for seed in seeds
        for decision in (8, 24, 40)
    ]
    for row in rows:
        row["provenance"] = _row_provenance(
            row,
            initial_renderer_state_sha256=initial_renderer_state_sha256,
            crossfit_renderer_state_sha256=crossfit_renderer_state_sha256,
            opsd_teacher_renderer_sha256=opsd_teacher_renderer_sha256,
        )
    return rows


def _sealed_metric_ledger(
    rows: list[dict[str, object]], shared: dict[str, object]
) -> tuple[bytes, bytes]:
    """Build the complete F0 ledger from the registered operation plan.

    The gate intentionally rejects aggregated reservations.  Keep this test
    fixture honest by materializing every physical call, including the U-Net
    and scheduler suffixes and the five deterministic optimizer fits.
    """
    records: list[dict[str, object]] = []
    used_reservation_ids: set[str] = set()
    operation_index = 0

    def append(record: dict[str, object]) -> None:
        value = dict(record)
        value["sequence"] = len(records) + 1
        value["previous_record_hash"] = (
            None if not records else records[-1]["record_hash"]
        )
        value["run_contract"] = _F0_CONTRACT_HASH
        value["record_hash"] = hashlib.sha256(_canonical(value)).hexdigest()
        records.append(value)

    def fresh_id(label: str) -> str:
        nonlocal operation_index
        operation_index += 1
        reservation_id = _reservation_id(f"f0-operation:{operation_index}:{label}")
        if reservation_id in used_reservation_ids:
            raise AssertionError("F0 fixture generated a duplicate reservation id")
        return reservation_id

    def add_pair(
        reservation_id: str,
        kind: str,
        metadata: dict[str, object],
        result: dict[str, object],
    ) -> None:
        if reservation_id in used_reservation_ids:
            raise AssertionError("F0 fixture generated a duplicate reservation id")
        used_reservation_ids.add(reservation_id)
        append(
            {
                "type": "reservation",
                "reservation_id": reservation_id,
                "kind": kind,
                "amount": 1,
                "metadata": metadata,
                "wall_time": 0.0,
            }
        )
        receipt_result = dict(result)
        receipt_result["result_hash"] = hashlib.sha256(
            _canonical(receipt_result)
        ).hexdigest()
        append(
            {
                "type": "receipt",
                "reservation_id": reservation_id,
                "kind": kind,
                "amount": 1,
                "metadata": metadata,
                "success": True,
                "result": receipt_result,
                "wall_time": 0.0,
            }
        )

    code_hash = hashlib.sha256(str(shared["code_commit"]).encode("ascii")).hexdigest()
    common_bound = {
        "run_contract_hash": _F0_CONTRACT_HASH,
        "renderer_frame_contract_hash": shared["renderer_frame_contract_hash"],
        "calibration_hash": shared["calibration_hash"],
        "data_manifest_sha256": shared["selected_payload_manifest_sha256"],
        "reward_config_sha256": shared["reward_config_sha256"],
    }
    zero_history = [
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    ]

    def identity(kind: str, *, witness: bool = False) -> dict[str, object]:
        if witness:
            return {
                "model": TOPIQ_NR_MODEL_ID,
                "role": TOPIQ_NR_ROLE,
                "preprocess_hash": shared["witness_preprocess_sha256"],
                "model_config_sha256": shared["witness_config_sha256"],
                "model_asset_manifest_sha256": shared[
                    "witness_asset_manifest_sha256"
                ],
            }
        if kind in {"reward_forward", "reward_backward"}:
            return {
                "model": "ImageReward-v1.0",
                "role": "training_reward",
                "preprocess_hash": shared["reward_preprocess_sha256"],
                "model_config_sha256": shared["reward_config_sha256"],
                "model_asset_manifest_sha256": shared[
                    "reward_asset_manifest_sha256"
                ],
            }
        return {
            "model": "repldm-runtime",
            "role": kind,
            "preprocess_hash": shared["reward_preprocess_sha256"],
            "model_config_sha256": shared["reward_config_sha256"],
            "model_asset_manifest_sha256": shared["reward_asset_manifest_sha256"],
        }

    def metadata_for(
        kind: str, context: dict[str, object], operation_identity: dict[str, object]
    ) -> dict[str, object]:
        return {
            "code_hash": code_hash,
            "data_hash": shared["selected_payload_manifest_sha256"],
            "checkpoint_hash": context["checkpoint_hash"],
            "method_allocation": {"kind": kind, "amount": 1, "method": "f0"},
            "split": context["split"],
            "prompt": context["prompt"],
            "seed": context["seed"],
            "step": context["step"],
            "prefix": context["prefix"],
            "branch": context["branch"],
            "action": context["action"],
            **operation_identity,
            **common_bound,
        }

    def result_for(
        kind: str,
        operation_identity: dict[str, object],
        *,
        image_hash: str,
        output_hash: str,
        parent_id: str | None = None,
        parent_output_hash: str | None = None,
    ) -> dict[str, object]:
        result = {
            "image_hash": image_hash,
            "output_hash": output_hash,
            "reward_preprocess_hash": operation_identity["preprocess_hash"],
            "model": operation_identity["model"],
            "role": operation_identity["role"],
            "model_config_sha256": operation_identity["model_config_sha256"],
            "model_asset_manifest_sha256": operation_identity[
                "model_asset_manifest_sha256"
            ],
            "scalar_or_gradient": {
                "unet_forward": "tensor",
                "scheduler_step": "tensor",
                "vae_decode": "tensor",
                "reward_forward": "scalar",
                "reward_backward": "gradient",
                "optimizer_step": "optimizer_state",
            }[kind],
            "wall_seconds": 0.0,
            "cached_parent": parent_output_hash,
            "parent_reservation_id": parent_id,
            "failure": None,
            **common_bound,
        }
        return result

    def add_plain(
        kind: str,
        context: dict[str, object],
        *,
        label: str,
        image_hash: str | None = None,
        output_hash: str | None = None,
        operation_identity: dict[str, object] | None = None,
    ) -> str:
        operation_identity = (
            identity(kind) if operation_identity is None else operation_identity
        )
        image_hash = image_hash or _digest(f"f0-image:{label}")
        output_hash = output_hash or image_hash
        metadata = metadata_for(kind, context, operation_identity)
        result = result_for(
            kind,
            operation_identity,
            image_hash=image_hash,
            output_hash=output_hash,
        )
        reservation_id = fresh_id(label)
        add_pair(reservation_id, kind, metadata, result)
        return reservation_id

    def metric_context(
        row: dict[str, object],
        *,
        branch_key: str,
        ledger_branch: str,
        checkpoint_hash: str,
    ) -> dict[str, object]:
        provenance = row["provenance"]
        target = str(provenance["target_record_sha256"])
        action = build_f0_metric_action(
            branch=ledger_branch,
            target_record_sha256=None if ledger_branch == "anchor" else target,
            renderer_action_history=zero_history,
        )
        return {
            "split": str(row["phase"]),
            "prompt": str(row["prompt_id"]),
            "seed": int(row["generation_seed"]),
            "step": 49,
            "prefix": 0,
            "branch": ledger_branch,
            "action": action,
            "checkpoint_hash": checkpoint_hash,
            "branch_key": branch_key,
        }

    seen_metric: set[str] = set()

    def add_metric_branch(
        row: dict[str, object],
        *,
        branch_key: str,
        ledger_branch: str,
        checkpoint_hash: str,
    ) -> dict[str, object]:
        context = metric_context(
            row,
            branch_key=branch_key,
            ledger_branch=ledger_branch,
            checkpoint_hash=checkpoint_hash,
        )
        branch_value = row["provenance"]["branches"][branch_key]
        reward_id = str(branch_value["image_reward"]["reservation_id"])
        if reward_id in seen_metric:
            return context
        seen_metric.add(reward_id)
        image_sha256 = str(branch_value["image_sha256"])
        image_output = tensor_operation_output_sha256(image_sha256)
        vae_identity = identity("vae_decode")
        vae_metadata = metadata_for("vae_decode", context, vae_identity)
        vae_result = result_for(
            "vae_decode",
            vae_identity,
            image_hash=image_output,
            output_hash=image_output,
        )
        parent_id = _reservation_id(f"vae:{reward_id}")
        add_pair(parent_id, "vae_decode", vae_metadata, vae_result)

        for scorer, score_identity, score in (
            (
                "image_reward",
                identity("reward_forward"),
                row["reward_select"][ledger_branch],
            ),
            (
                "topiq_nr",
                identity("reward_forward", witness=True),
                row["topiq_nr"][ledger_branch],
            ),
        ):
            reference = branch_value[scorer]
            reservation_id = str(reference["reservation_id"])
            score_metadata = metadata_for(
                "reward_forward", context, score_identity
            )
            score_result = result_for(
                "reward_forward",
                score_identity,
                image_hash=image_sha256,
                output_hash=_score_output_sha256(score),
                parent_id=parent_id,
                parent_output_hash=image_output,
            )
            add_pair(
                reservation_id,
                "reward_forward",
                score_metadata,
                score_result,
            )
        return context

    def add_target_operations(
        row: dict[str, object], *, decision: int, position: int
    ) -> None:
        phase = str(row["phase"])
        prompt = str(row["prompt_id"])
        seed = int(row["generation_seed"])
        action = [list(item) for item in zero_history[: position + 1]]
        context = {
            "split": phase,
            "prompt": prompt,
            "seed": seed,
            "step": decision,
            "prefix": 0,
            "branch": "anchor",
            "action": action,
            "checkpoint_hash": shared["initial_renderer_state_sha256"],
        }
        base_image = _digest(f"f0-target-image:{phase}:{prompt}:{seed}:{decision}")
        image_output = tensor_operation_output_sha256(base_image)
        vae_identity = identity("vae_decode")
        vae_id = add_plain(
            "vae_decode",
            context,
            label=f"target-vae:{phase}:{prompt}:{seed}:{decision}",
            image_hash=image_output,
            output_hash=image_output,
            operation_identity=vae_identity,
        )
        reward_identity = identity("reward_forward")
        reward_output = _score_output_sha256(row["reward_select"]["anchor"])
        reward_result = result_for(
            "reward_forward",
            reward_identity,
            image_hash=base_image,
            output_hash=reward_output,
            parent_id=vae_id,
            parent_output_hash=image_output,
        )
        target_reward_id = fresh_id(
            f"target-reward:{phase}:{prompt}:{seed}:{decision}"
        )
        add_pair(
            target_reward_id,
            "reward_forward",
            metadata_for("reward_forward", context, reward_identity),
            reward_result,
        )
        backward_identity = identity("reward_backward")
        backward_result = result_for(
            "reward_backward",
            backward_identity,
            image_hash=base_image,
            output_hash=_digest(
                f"f0-target-gradient:{phase}:{prompt}:{seed}:{decision}"
            ),
            parent_id=target_reward_id,
            parent_output_hash=reward_output,
        )
        add_pair(
            fresh_id(f"target-backward:{phase}:{prompt}:{seed}:{decision}"),
            "reward_backward",
            metadata_for("reward_backward", context, backward_identity),
            backward_result,
        )

    # Index the baseline rows by prompt/seed/decision so the operation plan is
    # emitted in the same order as the independent gate oracle.
    row_by_key = {
        (
            str(row["phase"]),
            str(row["prompt_id"]),
            int(row["generation_seed"]),
            int(row["decision_index"]),
        ): row
        for row in rows
    }
    blocks = sorted(
        {
            (phase, prompt, seed)
            for phase, prompt, seed, _decision in row_by_key
        }
    )
    for phase, prompt, seed in blocks:
        anchor_row = row_by_key[(phase, prompt, seed, 8)]
        anchor_context = add_metric_branch(
            anchor_row,
            branch_key="anchor",
            ledger_branch="anchor",
            checkpoint_hash=shared["initial_renderer_state_sha256"],
        )
        anchor_history = anchor_context["action"]["renderer_action_history"]
        for step in range(50):
            prior_count = sum(decision < step for decision in (8, 24, 40))
            add_plain(
                "unet_forward",
                {
                    **anchor_context,
                    "step": step,
                    "prefix": 0,
                    "branch": "anchor",
                    "action": anchor_history[:prior_count],
                },
                label=f"anchor-unet:{phase}:{prompt}:{seed}:{step}",
            )
            add_plain(
                "scheduler_step",
                {
                    **anchor_context,
                    "step": step,
                    "prefix": 0,
                    "branch": "anchor",
                    "action": anchor_history[
                        : prior_count + int(step in (8, 24, 40))
                    ],
                },
                label=f"anchor-scheduler:{phase}:{prompt}:{seed}:{step}",
            )

        for decision in (8, 24, 40):
            position = (8, 24, 40).index(decision)
            row = row_by_key[(phase, prompt, seed, decision)]
            add_target_operations(row, decision=decision, position=position)
            for branch in ("plus", "minus"):
                context = add_metric_branch(
                    row,
                    branch_key=branch,
                    ledger_branch=branch,
                    checkpoint_hash=shared["initial_renderer_state_sha256"],
                )
                history = context["action"]["renderer_action_history"]
                for step in range(decision, 50):
                    prior_count = sum(item < step for item in (8, 24, 40))
                    if step == decision:
                        prior_count = position
                    add_plain(
                        "scheduler_step",
                        {
                            **context,
                            "step": step,
                            "prefix": 0,
                            "branch": branch,
                            "action": history[
                                : prior_count + int(step in (8, 24, 40))
                            ],
                        },
                        label=f"{branch}-scheduler:{phase}:{prompt}:{seed}:{decision}:{step}",
                    )
                for step in range(decision + 1, 50):
                    prior_count = sum(item < step for item in (8, 24, 40))
                    add_plain(
                        "unet_forward",
                        {
                            **context,
                            "step": step,
                            "prefix": 0,
                            "branch": branch,
                            "action": history[:prior_count],
                        },
                        label=f"{branch}-unet:{phase}:{prompt}:{seed}:{decision}:{step}",
                    )

            realization = (
                "out_of_fold_student" if phase == "train" else "final_teacher"
            )
            if phase == "train":
                held_out_fold = int(row["provenance"]["held_out_fold"])
                realization_checkpoint = shared["crossfit_renderer_state_sha256"][
                    str(held_out_fold)
                ]
            else:
                # Non-F0 contract fixtures reuse this ledger builder only to
                # exercise contract rejection paths and do not carry a teacher
                # checkpoint field.  Keep their synthetic realization bound to
                # the immutable C0 hash; F0 fixtures provide the real field.
                realization_checkpoint = shared.get(
                    "opsd_teacher_renderer_sha256",
                    shared["initial_renderer_state_sha256"],
                )
            context = add_metric_branch(
                row,
                branch_key="realized",
                ledger_branch=realization,
                checkpoint_hash=realization_checkpoint,
            )
            history = context["action"]["renderer_action_history"]
            for step in range(decision, 50):
                prior_count = sum(item < step for item in (8, 24, 40))
                if step == decision:
                    prior_count = position
                add_plain(
                    "scheduler_step",
                    {
                        **context,
                        "step": step,
                        "prefix": 0,
                        "branch": realization,
                        "action": history[
                            : prior_count + int(step in (8, 24, 40))
                        ],
                    },
                    label=f"{realization}-scheduler:{phase}:{prompt}:{seed}:{decision}:{step}",
                )
            for step in range(decision + 1, 50):
                prior_count = sum(item < step for item in (8, 24, 40))
                add_plain(
                    "unet_forward",
                    {
                        **context,
                        "step": step,
                        "prefix": 0,
                        "branch": realization,
                        "action": history[:prior_count],
                    },
                    label=f"{realization}-unet:{phase}:{prompt}:{seed}:{decision}:{step}",
                )

    # Rebuild the exact deterministic batches used by the four cross-fit fits
    # and the final T_OPSD fit.
    def stable_id(prompt: str, seed: int, decision: int) -> str:
        digest = hashlib.sha256(
            f"{prompt}\0{seed}\0{decision}".encode("utf-8")
        ).hexdigest()[:20]
        return f"train-{seed}-{decision:02d}-{digest}"

    train_rows = [
        (
            str(row["prompt_id"]),
            int(row["generation_seed"]),
            int(row["decision_index"]),
            int(row["provenance"]["held_out_fold"]),
        )
        for row in rows
        if row["phase"] == "train"
    ]

    def fit_batches(
        fit_rows: list[tuple[str, int, int, int]], optimizer_seed: int
    ) -> list[list[str]]:
        stable_rows = [
            (stable_id(prompt, seed, decision), fold)
            for prompt, seed, decision, fold in fit_rows
        ]
        batches: list[list[str]] = []
        epoch = 0
        while len(batches) < 200:
            ordered = sorted(
                stable_rows,
                key=lambda item: hashlib.sha256(
                    f"sha256_epoch_order_v1\0{optimizer_seed}\0{epoch}\0{item[0]}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
            )
            for offset in range(0, len(ordered), 8):
                batch = [item[0] for item in ordered[offset : offset + 8]]
                if len(batch) != 8:
                    raise AssertionError("F0 optimizer fixture produced a short batch")
                batches.append(batch)
                if len(batches) == 200:
                    break
            epoch += 1
        return batches

    optimizer_roles = [
        (f"crossfit-fold-{fold}", 2026090100 + fold)
        for fold in range(4)
    ] + [("T_OPSD", 2026090104)]
    for role, optimizer_seed in optimizer_roles:
        fit_rows = (
            train_rows
            if role == "T_OPSD"
            else [row for row in train_rows if row[3] != int(role.rsplit("-", 1)[1])]
        )
        for step, target_ids in enumerate(fit_batches(fit_rows, optimizer_seed)):
            context = {
                "split": "train",
                "prompt": f"{role}-update-{step:03d}",
                "seed": optimizer_seed,
                "step": step,
                "prefix": 8,
                "branch": "f0",
                "action": {
                    "fit_role": role,
                    "optimizer_seed": optimizer_seed,
                    "target_stable_ids": target_ids,
                },
                "checkpoint_hash": _digest(f"f0-optimizer-checkpoint:{role}"),
            }
            add_plain(
                "optimizer_step",
                context,
                label=f"optimizer:{role}:{step}",
            )

    reserved = {
        kind: sum(
            1
            for record in records
            if record["type"] == "reservation" and record["kind"] == kind
        )
        for kind in F0_QUERY_BUDGET
    }
    if reserved != F0_QUERY_BUDGET:
        raise AssertionError(
            f"F0 fixture operation counts differ from the budget: {reserved}"
        )

    ledger_payload = b"".join(_canonical(record) + b"\n" for record in records)
    unsigned_seal = {
        "schema": "repldm.query_ledger_seal.v2",
        "run_contract": _F0_CONTRACT_HASH,
        "budget": dict(sorted(F0_QUERY_BUDGET.items())),
        "strict_provenance": True,
        "record_count": len(records),
        "tip_record_hash": records[-1]["record_hash"],
        "root_record_hash": records[0]["record_hash"],
    }
    seal = dict(unsigned_seal)
    seal["seal_hash"] = hashlib.sha256(_canonical(unsigned_seal)).hexdigest()
    return ledger_payload, _canonical(seal) + b"\n"


def _legacy_power_artifact() -> dict[str, object]:
    def phase(prompt_count: int) -> dict[str, object]:
        return {
            "prompt_count": prompt_count,
            "metrics": {
                metric: {"required_power": 0.80, "planned_power": 0.90}
                for metric in (
                    "direction_accuracy",
                    "g_target",
                    "topiq_nr_delta",
                    "realization_ratio",
                )
            },
        }

    return {
        "schema": "repldm.renderer_f0_power.v1",
        "status": "frozen_before_output",
        "f0_run_contract_sha256": _F0_CONTRACT_HASH,
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "unit": "prompt",
        },
        "minimum_effects": {
            "direction_accuracy": 0.55,
            "g_target": 0.10,
            "topiq_nr_delta": 0.005,
            "realization_ratio": 0.25,
        },
        "phases": {"train": phase(64), "validation": phase(32)},
    }


@pytest.mark.parametrize("phase", ("train", "validation"))
def test_f0_phase_evidence_round_trip(tmp_path: Path, phase: str) -> None:
    path = tmp_path / f"{phase}.jsonl"
    binding, stored = write_f0_phase_evidence(
        path,
        _passing_rows(phase),
        phase=phase,
        reward_scale=0.25,
        reward_statistics_sha256=_REWARD_STATISTICS_HASH,
        f0_run_contract_sha256=_F0_CONTRACT_HASH,
        screen_registration_sha256="b" * 64,
    )

    recomputed, normalized_binding = validate_f0_phase_evidence(
        binding,
        phase=phase,
        reward_scale=0.25,
        reward_statistics_sha256=_REWARD_STATISTICS_HASH,
        f0_run_contract_sha256=_F0_CONTRACT_HASH,
        screen_registration_sha256="b" * 64,
    )

    assert recomputed == stored
    assert normalized_binding == binding
    assert recomputed["passed"] is True
    assert recomputed["state_count"] == (384 if phase == "train" else 192)
    assert recomputed["coverage_by_decision"] == {"8": 1.0, "24": 1.0, "40": 1.0}
    assert recomputed["bootstrap"] == {
        "method": BOOTSTRAP_METHOD,
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "unit": "prompt",
        "interval_probabilities": [0.025, 0.975],
        "interval_role": "descriptive_stability_filter",
    }
    header = json.loads(path.read_text(encoding="ascii").splitlines()[0])
    expected_maxima = (
        {
            "unet_forwards": 35_200,
            "vae_decodes": 1_664,
            "reward_forwards": 2_944,
            "reward_backwards": 384,
            "image_reward_forwards": 1_664,
            "topiq_nr_forwards": 1_280,
        }
        if phase == "train"
        else {
            "unet_forwards": 17_600,
            "vae_decodes": 832,
            "reward_forwards": 1_472,
            "reward_backwards": 192,
            "image_reward_forwards": 832,
            "topiq_nr_forwards": 640,
        }
    )
    assert header["protocol_query_maxima"] == expected_maxima
    assert (
        expected_maxima["image_reward_forwards"]
        + expected_maxima["topiq_nr_forwards"]
        == expected_maxima["reward_forwards"]
    )


def test_f0_phase_rejects_stored_summary_numeric_type_tamper(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    binding, _summary = write_f0_phase_evidence(
        path,
        _passing_rows("train"),
        phase="train",
        reward_scale=0.25,
        reward_statistics_sha256=_REWARD_STATISTICS_HASH,
        f0_run_contract_sha256=_F0_CONTRACT_HASH,
        screen_registration_sha256="b" * 64,
    )
    records = [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]
    records[-1]["prompt_count"] = 64.0
    binding = _write_bytes(
        path, b"\n".join(_canonical(record) for record in records) + b"\n"
    )

    with pytest.raises(ValueError, match="stored summary"):
        validate_f0_phase_evidence(
            binding,
            phase="train",
            reward_scale=0.25,
            reward_statistics_sha256=_REWARD_STATISTICS_HASH,
            f0_run_contract_sha256=_F0_CONTRACT_HASH,
            screen_registration_sha256="b" * 64,
        )


def test_f0_phase_writer_refuses_to_replace_existing_evidence(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    binding, _summary = write_f0_phase_evidence(
        path,
        _passing_rows("train"),
        phase="train",
        reward_scale=0.25,
        reward_statistics_sha256=_REWARD_STATISTICS_HASH,
        f0_run_contract_sha256=_F0_CONTRACT_HASH,
        screen_registration_sha256="b" * 64,
    )
    original = path.read_bytes()
    changed = _passing_rows("train")
    for row in changed:
        row["topiq_nr"]["out_of_fold_student"] = 0.512

    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_f0_phase_evidence(
            path,
            changed,
            phase="train",
            reward_scale=0.25,
            reward_statistics_sha256=_REWARD_STATISTICS_HASH,
            f0_run_contract_sha256=_F0_CONTRACT_HASH,
            screen_registration_sha256="b" * 64,
        )

    assert path.read_bytes() == original
    assert _binding(path) == binding


@pytest.mark.parametrize("field", ("sha256", "bytes"))
def test_f0_phase_rejects_binding_mismatch(
    tmp_path: Path, field: str
) -> None:
    binding, _summary = write_f0_phase_evidence(
        tmp_path / "train.jsonl",
        _passing_rows("train"),
        phase="train",
        reward_scale=0.25,
        reward_statistics_sha256=_REWARD_STATISTICS_HASH,
        f0_run_contract_sha256=_F0_CONTRACT_HASH,
        screen_registration_sha256="b" * 64,
    )
    tampered = dict(binding)
    tampered[field] = "c" * 64 if field == "sha256" else int(binding["bytes"]) + 1

    with pytest.raises(ValueError, match="bytes differ"):
        validate_f0_phase_evidence(
            tampered,
            phase="train",
            reward_scale=0.25,
            reward_statistics_sha256=_REWARD_STATISTICS_HASH,
            f0_run_contract_sha256=_F0_CONTRACT_HASH,
            screen_registration_sha256="b" * 64,
        )


def test_f0_phase_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _binding_value, _summary = write_f0_phase_evidence(
        path,
        _passing_rows("train"),
        phase="train",
        reward_scale=0.25,
        reward_statistics_sha256=_REWARD_STATISTICS_HASH,
        f0_run_contract_sha256=_F0_CONTRACT_HASH,
        screen_registration_sha256="b" * 64,
    )
    lines = path.read_bytes().splitlines()
    lines[1] = b'{"phase":"train",' + lines[1][1:]
    binding = _write_bytes(path, b"\n".join(lines) + b"\n")

    with pytest.raises(ValueError, match="invalid JSON"):
        validate_f0_phase_evidence(
            binding,
            phase="train",
            reward_scale=0.25,
            reward_statistics_sha256=_REWARD_STATISTICS_HASH,
            f0_run_contract_sha256=_F0_CONTRACT_HASH,
            screen_registration_sha256="b" * 64,
        )


def test_f0_phase_rejects_inconsistent_shared_anchor() -> None:
    rows = _passing_rows("train")
    rows[1]["reward_select"]["anchor"] = 0.01

    with pytest.raises(ValueError, match="anchor differs across decisions"):
        compute_f0_phase_summary(rows, phase="train", reward_scale=0.25)


def test_f0_independent_witness_is_target_plus_not_realization() -> None:
    rows = _passing_rows("validation")
    for row in rows:
        row["topiq_nr"]["final_teacher"] = 0.40

    summary = compute_f0_phase_summary(
        rows, phase="validation", reward_scale=0.25
    )

    assert summary["topiq_nr_delta"]["mean"] == pytest.approx(0.01)
    assert summary["topiq_nr_realized_delta"]["mean"] == pytest.approx(-0.10)
    assert summary["checks"]["topiq_nr"] is True
    assert summary["passed"] is True


@pytest.mark.parametrize("branch", ("plus", "minus", "final_teacher"))
def test_f0_pixel_guards_cover_every_terminal_branch(branch: str) -> None:
    rows = _passing_rows("validation")
    for row in rows:
        row["pixel"][f"{branch}_clipped_fraction"] = 0.02

    summary = compute_f0_phase_summary(
        rows, phase="validation", reward_scale=0.25
    )

    assert summary["checks"]["clipped_fraction"] is False
    assert summary["passed"] is False


@pytest.mark.parametrize(
    "case, message",
    (
        ("duplicate", "unique"),
        ("seed", "generation seed"),
        ("count", "exactly 384 rows"),
        ("order", "canonical"),
    ),
)
def test_f0_phase_rejects_invalid_logical_state_coverage(
    case: str, message: str
) -> None:
    rows = _passing_rows("train")
    if case == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif case == "seed":
        rows[0]["generation_seed"] = 7
    elif case == "count":
        rows.pop()
    elif case == "order":
        rows[0], rows[1] = rows[1], rows[0]

    with pytest.raises(ValueError, match=message):
        compute_f0_phase_summary(rows, phase="train", reward_scale=0.25)


@pytest.mark.parametrize(
    "case, failed_check",
    (
        ("safety", "safety"),
        ("coverage", "coverage"),
        ("direction", "direction_accuracy"),
        ("target", "g_target"),
        ("topiq", "topiq_nr"),
        ("clipped", "clipped_fraction"),
        ("saturation", "mean_saturation"),
        ("contrast", "contrast"),
        ("cap", "target_cap"),
        ("realization", "g_realized"),
    ),
)
def test_f0_phase_recomputes_each_protocol_gate(
    case: str, failed_check: str
) -> None:
    rows = _passing_rows("validation")
    for row in rows:
        if case == "safety":
            row["violations"]["no_op_parity"] = True
        elif case == "coverage" and row["decision_index"] == 8:
            row["valid_nonzero_gradient"] = False
        elif case == "direction":
            row["reward_select"]["minus"] = 0.30
        elif case == "target":
            row["reward_select"].update(
                {"plus": 0.02, "minus": -0.05, "final_teacher": 0.01}
            )
        elif case == "topiq":
            row["topiq_nr"]["plus"] = 0.504
        elif case == "clipped":
            row["pixel"]["final_teacher_clipped_fraction"] = 0.012
        elif case == "saturation":
            row["pixel"]["final_teacher_mean_saturation"] = 0.206
        elif case == "contrast":
            row["pixel"]["final_teacher_contrast"] = 0.94
        elif case == "cap":
            row["target_cap"]["plus_at_98pct"] = True
        elif case == "realization":
            row["reward_select"]["final_teacher"] = 0.03

    summary = compute_f0_phase_summary(rows, phase="validation", reward_scale=0.25)

    assert summary["checks"][failed_check] is False
    assert summary["passed"] is False


def test_f0_power_validator_rejects_underpowered_plan(tmp_path: Path) -> None:
    artifact = _legacy_power_artifact()
    artifact["phases"]["validation"]["metrics"]["g_target"][
        "planned_power"
    ] = 0.79
    path = tmp_path / "power.json"
    binding = _write_bytes(path, _canonical(artifact) + b"\n")

    with pytest.raises(ValueError, match="underpowered"):
        validate_f0_power_evidence(
            binding, f0_run_contract_sha256=_F0_CONTRACT_HASH
        )


def test_f0_power_v1_rejects_even_high_claims_and_production_write(
    tmp_path: Path,
) -> None:
    artifact = _legacy_power_artifact()
    path = tmp_path / "claimed-power.json"
    binding = _write_bytes(path, _canonical(artifact) + b"\n")

    with pytest.raises(ValueError, match="non-recomputable self-report"):
        validate_f0_power_evidence(
            binding, f0_run_contract_sha256=_F0_CONTRACT_HASH
        )
    with pytest.raises(ValueError, match="non-recomputable self-report"):
        write_f0_power_evidence(tmp_path / "refused.json", artifact)
    with pytest.raises(ValueError, match="non-recomputable self-report"):
        build_f0_power_artifact(
            f0_run_contract_sha256=_F0_CONTRACT_HASH,
            phase_power=artifact["phases"],
        )
    assert not (tmp_path / "refused.json").exists()


def test_f0_screen_registration_round_trip(tmp_path: Path) -> None:
    artifact = build_f0_screen_registration(
        f0_run_contract_sha256=_F0_CONTRACT_HASH
    )
    binding = write_f0_screen_registration_evidence(
        tmp_path / "screen-registration.json", artifact
    )

    validated, normalized_binding = validate_f0_screen_registration_evidence(
        binding, f0_run_contract_sha256=_F0_CONTRACT_HASH
    )

    assert validated == artifact
    assert normalized_binding == binding
    assert validated["evidence_tier"] == "engineering_screen"
    assert validated["inferential_claim_allowed"] is False
    assert validated["design"] == f0_screen_design()
    assert validated["pixel_summary"] == {
        "schema": F0_PIXEL_SUMMARY_SCHEMA,
        "input": {
            "layout": "rgb_nchw",
            "batch_size": 1,
            "source": "finite_floating_tensor",
            "working_device": "cpu",
            "working_dtype": "float32",
        },
        "quantization": {
            "formula": "uint8(round_ties_to_even(clamp(x,0,1)*255))"
        },
        "clipped_fraction": {
            "formula": "mean(any_channel(rgb_uint8<=lo or rgb_uint8>=hi))",
            "lo": 2,
            "hi": 253,
        },
        "mean_saturation": {
            "color_model": "HSV",
            "formula": "mean(where(max>epsilon,(max-min)/(max+epsilon),0))",
            "rgb_scale": "rgb_uint8/255",
            "epsilon": 1e-6,
        },
        "contrast": {
            "formula": "population_std(0.299*R+0.587*G+0.114*B)",
            "rgb_scale": "uint8_0_255",
            "luma": "BT.601",
            "correction": 0,
        },
    }
    assert validated["limitations"] == {
        "no_power_claim": True,
        "no_significance_claim": True,
        "no_benchmark_substitution": True,
        "not_opd_dpo_rl_result": True,
    }
    obligation = validated["benchmark_obligation"]
    assert obligation["checkpoint"] == "T_OPSD"
    assert obligation["hpsv2"]["image_count"] == 3_200
    assert obligation["geneval"] == {
        "complete": True,
        "official": True,
        "prompt_count": 553,
        "images_per_prompt": 4,
        "image_count": 2_212,
    }
    assert obligation["required_before_performance_claim"] is True


@pytest.mark.parametrize(
    "case, message",
    (
        ("inferential", "forbid inferential claims"),
        ("design", "design differs"),
        ("threshold", "thresholds differs"),
        ("pixel", "pixel_summary differs"),
        ("benchmark", "benchmark_obligation differs"),
        ("limitation", "limitations differs"),
        ("self_report", "fields differ"),
    ),
)
def test_f0_screen_registration_rejects_tamper_and_self_report(
    tmp_path: Path, case: str, message: str
) -> None:
    artifact = build_f0_screen_registration(
        f0_run_contract_sha256=_F0_CONTRACT_HASH
    )
    if case == "inferential":
        artifact["inferential_claim_allowed"] = True
    elif case == "design":
        artifact["design"]["validation"]["prompt_count"] = 33
    elif case == "threshold":
        artifact["thresholds"]["direction_accuracy"]["minimum_mean"] = 0.50
    elif case == "pixel":
        artifact["pixel_summary"]["clipped_fraction"]["lo"] = 0
    elif case == "benchmark":
        artifact["benchmark_obligation"]["hpsv2"]["image_count"] = 64
    elif case == "limitation":
        artifact["limitations"]["no_power_claim"] = False
    elif case == "self_report":
        artifact["planned_power"] = 0.99
    path = tmp_path / f"tampered-{case}.json"
    binding = _write_bytes(path, _canonical(artifact) + b"\n")

    with pytest.raises(ValueError, match=message):
        validate_f0_screen_registration_evidence(
            binding, f0_run_contract_sha256=_F0_CONTRACT_HASH
        )
    with pytest.raises(ValueError):
        write_f0_screen_registration_evidence(
            tmp_path / f"refused-{case}.json", artifact
        )


def test_compute_f0_pixel_summary_matches_frozen_uint8_definition() -> None:
    images = torch.tensor(
        [[[[0.0, 0.5]], [[0.5, 0.5]], [[1.0, 0.5]]]],
        dtype=torch.float32,
    )

    summary = compute_f0_pixel_summary(images)

    colorful_luma = 0.587 * 128.0 + 0.114 * 255.0
    gray_luma = 128.0
    assert summary["clipped_fraction"] == 0.5
    assert summary["mean_saturation"] == pytest.approx(0.5 / 1.000001)
    assert summary["contrast"] == pytest.approx(
        abs(gray_luma - colorful_luma) / 2.0
    )


@pytest.mark.parametrize(
    "images",
    (
        torch.zeros(2, 3, 1, 1),
        torch.zeros(1, 1, 1, 1),
        torch.zeros(1, 3, 1, 1, dtype=torch.int64),
        torch.full((1, 3, 1, 1), float("nan")),
    ),
)
def test_compute_f0_pixel_summary_rejects_invalid_tensor(images: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="pixel summary"):
        compute_f0_pixel_summary(images)


def _gate_fixture(
    tmp_path: Path,
    *,
    train_rows: list[dict[str, object]] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    shared = {
        "code_commit": "1" * 40,
        "selected_view_release_id": "selected-view-" + "2" * 20,
        "selected_view_manifest_sha256": "3" * 64,
        "selected_payload_manifest_sha256": "4" * 64,
        "renderer_frame_contract_hash": "5" * 64,
        "calibration_hash": "6" * 64,
        "action_contract_hash": "7" * 64,
        "initial_renderer_state_sha256": _TEST_INITIAL_RENDERER,
        "reward_config_sha256": "9" * 64,
        "reward_preprocess_sha256": "a" * 64,
        "reward_asset_manifest_sha256": "d" * 64,
        "witness_config_sha256": "e" * 64,
        "witness_preprocess_sha256": TOPIQ_NR_PREPROCESS_SHA256,
        "witness_asset_manifest_sha256": "f" * 64,
        "opsd_teacher_state_manifest_sha256": "b" * 64,
        "opsd_teacher_renderer_sha256": _TEST_OPSD_RENDERER,
        "crossfit_renderer_state_sha256": dict(_TEST_CROSSFIT_RENDERERS),
    }
    reward_statistics = {
        "schema": "repldm.renderer_reward_statistics.v1",
        "status": "frozen",
        "anchor_count": 128,
        "estimator": "median_iqr_over_1.349_floor_1e-6",
        "location": 0.5,
        "scale": 0.25,
        "initial_renderer_state_sha256": shared["initial_renderer_state_sha256"],
        "reward_config_sha256": shared["reward_config_sha256"],
        "reward_preprocess_sha256": shared["reward_preprocess_sha256"],
        "selected_view_release_id": shared["selected_view_release_id"],
        "anchors": [
            {
                "prompt_id": f"train-{prompt_index:02d}",
                "generation_seed": seed,
                "image_sha256": hashlib.sha256(
                    f"{prompt_index}:{seed}".encode("ascii")
                ).hexdigest(),
                "reward": float(prompt_index),
            }
            for prompt_index in range(64)
            for seed in (2026090101, 2026090102)
        ],
    }
    reward_path = tmp_path / "reward-statistics.json"
    reward_binding = _write_bytes(reward_path, _canonical(reward_statistics))
    reward_hash = str(reward_binding["sha256"])
    contract = dict(shared)
    contract.update(
        {
            "reward_statistics_sha256": reward_hash,
            "artifacts": {
                "reward_statistics": {
                    "path": str(reward_path.resolve()),
                    "sha256": reward_hash,
                }
            },
            "runtime": {"reward_dtype": "float32"},
        }
    )

    registration_binding = write_f0_screen_registration_evidence(
        tmp_path / "screen-registration.json",
        build_f0_screen_registration(
            f0_run_contract_sha256=_F0_CONTRACT_HASH
        ),
    )
    baseline_train_rows = _passing_rows("train")
    selected_train_rows = (
        train_rows if train_rows is not None else baseline_train_rows
    )
    validation_rows = _passing_rows("validation")
    train_binding, train_summary = write_f0_phase_evidence(
        tmp_path / "train.jsonl",
        selected_train_rows,
        phase="train",
        reward_scale=0.25,
        reward_statistics_sha256=reward_hash,
        f0_run_contract_sha256=_F0_CONTRACT_HASH,
        screen_registration_sha256=str(registration_binding["sha256"]),
    )
    validation_binding, validation_summary = write_f0_phase_evidence(
        tmp_path / "validation.jsonl",
        validation_rows,
        phase="validation",
        reward_scale=0.25,
        reward_statistics_sha256=reward_hash,
        f0_run_contract_sha256=_F0_CONTRACT_HASH,
        screen_registration_sha256=str(registration_binding["sha256"]),
    )
    ledger_payload, seal_payload = _sealed_metric_ledger(
        [*baseline_train_rows, *validation_rows], shared
    )
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_binding = _write_bytes(ledger_path, ledger_payload)
    seal_binding = _write_bytes(
        ledger_path.with_name(ledger_path.name + ".seal"), seal_payload
    )

    gate = {
        "schema": "repldm.renderer_f0_gate.v5",
        "status": "passed",
        "f0_run_contract_sha256": _F0_CONTRACT_HASH,
        "selected_payload_sha256": shared["selected_payload_manifest_sha256"],
        "reward_statistics_sha256": reward_hash,
        "screen_registration": registration_binding,
        "ledger": ledger_binding,
        "ledger_seal": seal_binding,
        "train_gate": {
            "passed": train_summary["passed"],
            "prompt_count": train_summary["prompt_count"],
            "state_count": train_summary["state_count"],
            "safety_violations": train_summary["safety_violations"],
            "metrics": train_binding,
        },
        "validation_gate": {
            "passed": validation_summary["passed"],
            "prompt_count": validation_summary["prompt_count"],
            "state_count": validation_summary["state_count"],
            "safety_violations": validation_summary["safety_violations"],
            "metrics": validation_binding,
        },
    }
    for field, value in shared.items():
        if field != "selected_payload_manifest_sha256":
            gate[field] = value
    return gate, contract


def _rewrite_gate_phase(
    gate: dict[str, object],
    *,
    phase: str,
    mutate,
) -> None:
    phase_gate = gate[f"{phase}_gate"]
    path = Path(phase_gate["metrics"]["path"])
    records = [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]
    rows = records[1:-1]
    mutate(rows)
    summary = compute_f0_phase_summary(rows, phase=phase, reward_scale=0.25)
    payload = b"\n".join(_canonical(value) for value in (*records[:1], *rows, summary)) + b"\n"
    path.write_bytes(payload)
    phase_gate["metrics"] = _binding(path)
    for field in ("passed", "prompt_count", "state_count", "safety_violations"):
        phase_gate[field] = summary[field]


def test_f0_gate_rejects_obsolete_power_schema(tmp_path: Path) -> None:
    gate, contract = _gate_fixture(tmp_path)
    gate["schema"] = "repldm.renderer_f0_gate.v2"
    gate["power"] = gate.pop("screen_registration")

    with pytest.raises(ValueError, match="fields differ"):
        validate_f0_gate(gate, contract=contract)


def test_f0_gate_rejects_passed_claim_when_raw_rows_fail(
    tmp_path: Path,
) -> None:
    rows = _passing_rows("train")
    for row in rows:
        row["topiq_nr"]["plus"] = 0.504
    gate, contract = _gate_fixture(tmp_path, train_rows=rows)
    gate["train_gate"]["passed"] = True

    with pytest.raises(ValueError, match="claims differ"):
        validate_f0_gate(gate, contract=contract)


@pytest.mark.parametrize(
    ("phase", "branch", "replacement"),
    (
        ("train", "anchor", _TEST_CROSSFIT_RENDERERS["0"]),
        ("train", "plus", _TEST_CROSSFIT_RENDERERS["0"]),
        ("train", "minus", _TEST_CROSSFIT_RENDERERS["0"]),
        ("train", "realized", _TEST_CROSSFIT_RENDERERS["1"]),
        ("validation", "realized", _TEST_CROSSFIT_RENDERERS["0"]),
    ),
)
def test_f0_gate_rejects_metric_branch_from_wrong_renderer_checkpoint(
    tmp_path: Path,
    phase: str,
    branch: str,
    replacement: str,
) -> None:
    gate, contract = _gate_fixture(tmp_path)

    def mutate(rows: list[dict[str, object]]) -> None:
        selected = rows[:3] if branch == "anchor" else rows[:1]
        for row in selected:
            row["provenance"]["branches"][branch][
                "renderer_state_sha256"
            ] = replacement

    _rewrite_gate_phase(gate, phase=phase, mutate=mutate)
    with pytest.raises(ValueError, match="wrong renderer checkpoint"):
        validate_f0_gate(gate, contract=contract)


def test_f0_gate_rejects_incomplete_crossfit_renderer_map(tmp_path: Path) -> None:
    gate, contract = _gate_fixture(tmp_path)
    del gate["crossfit_renderer_state_sha256"]["3"]

    with pytest.raises(ValueError, match="cross-fit renderer map"):
        validate_f0_gate(gate, contract=contract)


def test_f0_gate_rejects_forged_score_and_caller_supplied_output_hash(
    tmp_path: Path,
) -> None:
    gate, contract = _gate_fixture(tmp_path)

    def mutate(rows: list[dict[str, object]]) -> None:
        rows[0]["reward_select"]["plus"] = 0.25
        rows[0]["provenance"]["branches"]["plus"]["image_reward"][
            "output_sha256"
        ] = _score_output_sha256(0.25)

    _rewrite_gate_phase(gate, phase="train", mutate=mutate)

    with pytest.raises(ValueError, match="receipt differs"):
        validate_f0_gate(gate, contract=contract)


def test_f0_gate_rejects_state_specific_receipt_reuse(tmp_path: Path) -> None:
    gate, contract = _gate_fixture(tmp_path)

    def mutate(rows: list[dict[str, object]]) -> None:
        rows[1]["provenance"]["branches"]["plus"]["image_reward"] = copy.deepcopy(
            rows[0]["provenance"]["branches"]["plus"]["image_reward"]
        )

    _rewrite_gate_phase(gate, phase="train", mutate=mutate)

    with pytest.raises(ValueError, match="receipt|action|reused"):
        validate_f0_gate(gate, contract=contract)


def test_f0_gate_rejects_image_hash_detached_from_vae_parent(tmp_path: Path) -> None:
    gate, contract = _gate_fixture(tmp_path)

    def mutate(rows: list[dict[str, object]]) -> None:
        rows[0]["provenance"]["branches"]["plus"]["image_sha256"] = _digest(
            "forged-image"
        )

    _rewrite_gate_phase(gate, phase="train", mutate=mutate)

    with pytest.raises(ValueError, match="receipt|image"):
        validate_f0_gate(gate, contract=contract)


@pytest.mark.parametrize(
    "field, value, message",
    (
        ("passed", 1, "must be boolean"),
        ("prompt_count", 64.0, "non-negative integer"),
        ("safety_violations", False, "non-negative integer"),
    ),
)
def test_f0_gate_rejects_type_confused_outer_claims(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    gate, contract = _gate_fixture(tmp_path)
    gate["train_gate"][field] = value

    with pytest.raises(ValueError, match=message):
        validate_f0_gate(gate, contract=contract)


def test_f0_nonconstant_prompt_bootstrap_is_frozen(tmp_path: Path) -> None:
    rows = _passing_rows("train")
    for row in rows:
        prompt_index = int(str(row["prompt_id"]).rsplit("-", 1)[1])
        seed_offset = 0.002 if row["generation_seed"] == 2026090102 else 0.0
        decision_offset = {8: 0.0, 24: 0.001, 40: 0.002}[row["decision_index"]]
        row["reward_select"]["plus"] = (
            0.15 + 0.001 * prompt_index + seed_offset + decision_offset
        )
        row["reward_select"]["minus"] = (
            row["reward_select"]["plus"] - 0.05
            if prompt_index < 48
            else row["reward_select"]["plus"] + 0.05
        )
        row["reward_select"]["out_of_fold_student"] = 0.06 + 0.0005 * prompt_index
        row["topiq_nr"]["plus"] = 0.506 + 0.00005 * prompt_index
        row["topiq_nr"]["out_of_fold_student"] = 0.507 + 0.00005 * prompt_index
        row["pixel"]["out_of_fold_contrast"] = 0.99 + 0.0003 * prompt_index

    binding, stored = write_f0_phase_evidence(
        tmp_path / "nonconstant.jsonl",
        rows,
        phase="train",
        reward_scale=0.25,
        reward_statistics_sha256=_REWARD_STATISTICS_HASH,
        f0_run_contract_sha256=_F0_CONTRACT_HASH,
        screen_registration_sha256="b" * 64,
    )
    recomputed, _normalized = validate_f0_phase_evidence(
        binding,
        phase="train",
        reward_scale=0.25,
        reward_statistics_sha256=_REWARD_STATISTICS_HASH,
        f0_run_contract_sha256=_F0_CONTRACT_HASH,
        screen_registration_sha256="b" * 64,
    )

    assert recomputed == stored
    assert recomputed["passed"] is True
    assert recomputed["direction_accuracy"] == {
        "mean": 0.75,
        "bootstrap_interval_95": [0.640625, 0.859375],
    }
    assert recomputed["g_target"]["bootstrap_interval_95"] == pytest.approx(
        [0.7159375, 0.7525], abs=1e-12
    )
