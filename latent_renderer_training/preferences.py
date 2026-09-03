"""Immutable, contract-bound preference labels for pair-based training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping


PREFERENCE_LABEL_SCHEMA = "repldm.renderer_preference_label.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LABELS = frozenset({"plus", "minus", "tie"})


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("preference label must contain finite JSON values") from exc


def _finite(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _hash(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class PreferenceLabelProvenance:
    """One ledgered branch preference under the frozen reward normalization."""

    rollout_id: str
    prompt_id: str
    generation_seed: int
    split: str
    label: str
    chosen_branch: str
    rejected_branch: str
    weight: float
    plus_reward: float
    minus_reward: float
    normalized_plus_reward: float
    normalized_minus_reward: float
    reward_location: float
    reward_scale: float
    tie_epsilon: float
    plus_image_sha256: str
    minus_image_sha256: str
    reward_statistics_sha256: str
    reward_config_sha256: str
    reward_preprocess_sha256: str

    def __post_init__(self) -> None:
        for name in ("rollout_id", "prompt_id", "split"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if (
            isinstance(self.generation_seed, bool)
            or not isinstance(self.generation_seed, int)
            or self.generation_seed < 0
        ):
            raise ValueError("generation_seed must be a non-negative integer")
        if self.label not in _LABELS:
            raise ValueError("preference label must be plus, minus, or tie")
        for name in (
            "plus_reward",
            "minus_reward",
            "normalized_plus_reward",
            "normalized_minus_reward",
            "reward_location",
            "reward_scale",
            "tie_epsilon",
            "weight",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), label=name))
        if self.reward_scale < 1e-6:
            raise ValueError("reward_scale must satisfy the frozen 1e-6 floor")
        if self.tie_epsilon != 1e-6:
            raise ValueError("tie_epsilon differs from the frozen protocol")
        for name in (
            "plus_image_sha256",
            "minus_image_sha256",
            "reward_statistics_sha256",
            "reward_config_sha256",
            "reward_preprocess_sha256",
        ):
            _hash(getattr(self, name), label=name)

        expected_plus = (self.plus_reward - self.reward_location) / self.reward_scale
        expected_minus = (self.minus_reward - self.reward_location) / self.reward_scale
        if not math.isclose(
            self.normalized_plus_reward, expected_plus, rel_tol=1e-12, abs_tol=1e-12
        ) or not math.isclose(
            self.normalized_minus_reward, expected_minus, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("normalized rewards differ from the frozen statistics")

        difference = self.plus_reward - self.minus_reward
        if difference > self.tie_epsilon:
            expected = ("plus", "plus", "minus", 1.0)
        elif difference < -self.tie_epsilon:
            expected = ("minus", "minus", "plus", 1.0)
        else:
            expected = ("tie", "anchor", "plus", 0.0)
        actual = (self.label, self.chosen_branch, self.rejected_branch, self.weight)
        if actual != expected:
            raise ValueError("preference outcome differs from the frozen tie rule")

    @classmethod
    def from_rewards(
        cls,
        *,
        rollout_id: str,
        prompt_id: str,
        generation_seed: int,
        split: str,
        plus_reward: float,
        minus_reward: float,
        reward_location: float,
        reward_scale: float,
        plus_image_sha256: str,
        minus_image_sha256: str,
        reward_statistics_sha256: str,
        reward_config_sha256: str,
        reward_preprocess_sha256: str,
        tie_epsilon: float = 1e-6,
    ) -> "PreferenceLabelProvenance":
        plus = _finite(plus_reward, label="plus_reward")
        minus = _finite(minus_reward, label="minus_reward")
        location = _finite(reward_location, label="reward_location")
        scale = _finite(reward_scale, label="reward_scale")
        epsilon = _finite(tie_epsilon, label="tie_epsilon")
        if plus > minus + epsilon:
            label, chosen, rejected, weight = "plus", "plus", "minus", 1.0
        elif minus > plus + epsilon:
            label, chosen, rejected, weight = "minus", "minus", "plus", 1.0
        else:
            label, chosen, rejected, weight = "tie", "anchor", "plus", 0.0
        return cls(
            rollout_id=rollout_id,
            prompt_id=prompt_id,
            generation_seed=generation_seed,
            split=split,
            label=label,
            chosen_branch=chosen,
            rejected_branch=rejected,
            weight=weight,
            plus_reward=plus,
            minus_reward=minus,
            normalized_plus_reward=(plus - location) / scale,
            normalized_minus_reward=(minus - location) / scale,
            reward_location=location,
            reward_scale=scale,
            tie_epsilon=epsilon,
            plus_image_sha256=plus_image_sha256,
            minus_image_sha256=minus_image_sha256,
            reward_statistics_sha256=reward_statistics_sha256,
            reward_config_sha256=reward_config_sha256,
            reward_preprocess_sha256=reward_preprocess_sha256,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PreferenceLabelProvenance":
        if not isinstance(value, Mapping):
            raise TypeError("preference label must be a mapping")
        payload = json.loads(_canonical(dict(value)).decode("utf-8"))
        required = {
            "schema",
            "rollout_id",
            "prompt_id",
            "generation_seed",
            "split",
            "label",
            "chosen_branch",
            "rejected_branch",
            "weight",
            "plus_reward",
            "minus_reward",
            "normalized_plus_reward",
            "normalized_minus_reward",
            "reward_location",
            "reward_scale",
            "tie_epsilon",
            "plus_image_sha256",
            "minus_image_sha256",
            "reward_statistics_sha256",
            "reward_config_sha256",
            "reward_preprocess_sha256",
        }
        if set(payload) != required or payload.get("schema") != PREFERENCE_LABEL_SCHEMA:
            raise ValueError("preference label fields differ from the registered schema")
        payload.pop("schema")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": PREFERENCE_LABEL_SCHEMA, **self.__dict__}

    @property
    def sha256(self) -> str:
        """Hash used to match this label to its successful ledger receipt."""
        return hashlib.sha256(_canonical(self.to_dict())).hexdigest()

    def preference(self) -> tuple[str, str, bool]:
        return self.chosen_branch, self.rejected_branch, self.label == "tie"

    @property
    def signed_advantage(self) -> float:
        if self.label == "tie":
            return 0.0
        difference = self.normalized_plus_reward - self.normalized_minus_reward
        return difference / max(abs(difference), 1e-6)

    def validate_rewards(self, plus: float, minus: float) -> None:
        if float(plus) != self.plus_reward or float(minus) != self.minus_reward:
            raise ValueError("preference label rewards differ from the scored rollout")


__all__ = ["PREFERENCE_LABEL_SCHEMA", "PreferenceLabelProvenance"]
