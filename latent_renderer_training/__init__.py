"""Shared training primitives for OPD, DPO, and RL latent renderers."""

from .contracts import ActionSpaceContract, contract_hash
from .distributions import SquashedGaussian, transformed_gaussian_kl
from .objectives import dpo_loss, normalized_transition_loss, normalized_transition_losses, opd_loss, per_decision_rl_loss, search_distill_loss
from .ledger import QueryLedger, QueryReservation
from .rollout import (
    BranchTrajectory,
    DecisionProposal,
    RolloutCollection,
    Transition,
    collect_antithetic_rollout,
    replay_shared_prefix,
)
from .teachers import RewardTargetPair, TargetStepConfig, construct_reward_targets
from .authorization import (
    AUTHORIZATION_SCHEMA,
    AuthorizationBinding,
    TrainingAuthorization,
    require_authorization_binding,
    write_authorization_receipt,
)
from .run_contract import RUN_CONTRACT_SCHEMA, TrainingRunContract, validate_run_contract_payload
from .launcher import TRAINING_CONFIG_SCHEMA, LaunchResult, launch_training
from .storage import AtomicRolloutStore, CheckpointProvenance
from .preferences import PREFERENCE_LABEL_SCHEMA, PreferenceLabelProvenance
from .renderer import (
    CALIBRATION_DECISION_INDICES,
    CALIBRATION_SCHEMA,
    CALIBRATION_STATE_COUNT,
    CANONICAL_FRAME_SLOTS,
    FRAME_SCHEMA,
    EulerFrameDiagnostics,
    EulerFrameOutput,
    EulerFrameState,
    EulerMappedOutput,
    EulerNativeFrameV1,
    FrameCalibration,
    FrameCalibrationSample,
    calibrate_global_mask,
    haar_tangent_frame,
    mask_sha256,
    phase_matched_frame,
    project_fixed_moment_tangent,
    tensor_sha256,
)

__all__ = [
    "ActionSpaceContract", "BranchTrajectory", "QueryLedger", "QueryReservation",
    "SquashedGaussian", "Transition", "contract_hash", "dpo_loss",
    "normalized_transition_loss", "normalized_transition_losses", "opd_loss", "per_decision_rl_loss",
    "replay_shared_prefix", "search_distill_loss", "transformed_gaussian_kl",
    "RewardTargetPair", "TargetStepConfig", "construct_reward_targets",
    "AUTHORIZATION_SCHEMA", "AuthorizationBinding", "TrainingAuthorization",
    "require_authorization_binding", "write_authorization_receipt",
    "RUN_CONTRACT_SCHEMA", "TrainingRunContract", "validate_run_contract_payload",
    "TRAINING_CONFIG_SCHEMA", "LaunchResult", "launch_training",
    "AtomicRolloutStore", "CheckpointProvenance",
    "PREFERENCE_LABEL_SCHEMA", "PreferenceLabelProvenance",
    "DecisionProposal", "RolloutCollection", "collect_antithetic_rollout",
    "CALIBRATION_DECISION_INDICES", "CALIBRATION_SCHEMA", "CALIBRATION_STATE_COUNT",
    "CANONICAL_FRAME_SLOTS", "FRAME_SCHEMA", "EulerFrameDiagnostics",
    "EulerFrameOutput", "EulerFrameState", "EulerMappedOutput",
    "EulerNativeFrameV1", "FrameCalibration", "FrameCalibrationSample",
    "calibrate_global_mask", "haar_tangent_frame", "mask_sha256",
    "phase_matched_frame", "project_fixed_moment_tangent", "tensor_sha256",
]
