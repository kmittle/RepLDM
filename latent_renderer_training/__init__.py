"""Shared training primitives for OPD, DPO, and RL latent renderers."""

from .contracts import ActionSpaceContract, contract_hash
from .distributions import SquashedGaussian, transformed_gaussian_kl
from .objectives import dpo_loss, normalized_transition_loss, normalized_transition_losses, opd_loss, per_decision_rl_loss, search_distill_loss
from .ledger import QueryLedger, QueryReservation
from .rollout import BranchTrajectory, Transition, replay_shared_prefix
from .teachers import RewardTargetPair, TargetStepConfig, construct_reward_targets
from .authorization import AUTHORIZATION_SCHEMA, TrainingAuthorization, write_authorization_receipt

__all__ = [
    "ActionSpaceContract", "BranchTrajectory", "QueryLedger", "QueryReservation",
    "SquashedGaussian", "Transition", "contract_hash", "dpo_loss",
    "normalized_transition_loss", "normalized_transition_losses", "opd_loss", "per_decision_rl_loss",
    "replay_shared_prefix", "search_distill_loss", "transformed_gaussian_kl",
    "RewardTargetPair", "TargetStepConfig", "construct_reward_targets",
    "AUTHORIZATION_SCHEMA", "TrainingAuthorization", "write_authorization_receipt",
]
