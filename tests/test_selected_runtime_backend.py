from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "eval-pipeline"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from data_catalog.selected import DecodedImage, model_binding_sha256  # noqa: E402
from data_catalog.selected_runtime_backend import build_runtime_v1  # noqa: E402


def _binding(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.absolute()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


class _FakeTokenizer:
    roots: ClassVar[list[Path]] = []
    used_roots: ClassVar[list[Path]] = []

    def __init__(self, root: str | Path | None = None):
        self.root = None if root is None else Path(root)

    @classmethod
    def from_pretrained(cls, root: str, *, local_files_only: bool):
        assert local_files_only is True
        cls.roots.append(Path(root))
        return cls(root)

    def __call__(self, texts, **kwargs):
        if self.root is not None:
            type(self).used_roots.append(self.root)
        if isinstance(texts, str):
            texts = [texts]
        count = len(texts)
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": torch.ones((count, 77), dtype=torch.long)}
        ids = [[1, 2, 3] for _ in range(count)]
        mask = [[1, 1, 1] for _ in range(count)]
        return {
            "input_ids": ids[0] if count == 1 else ids,
            "attention_mask": mask[0] if count == 1 else mask,
        }


class _TruncationCheckingTokenizer(_FakeTokenizer):
    """Ensure protected-text encoding requests the fixed CLIP context policy."""

    def __call__(self, texts, **kwargs):
        if kwargs.get("return_tensors") == "pt":
            assert kwargs.get("truncation") is True
            assert kwargs.get("max_length") == 77
        return super().__call__(texts, **kwargs)


class _FakeModel:
    def eval(self):
        return self

    def encode_text(self, ids):
        values = ids.float().sum(dim=1)
        return torch.stack((values, values * 0 + 1, values * 0 + 2), dim=1)

    def encode_image(self, images):
        values = images.float().mean(dim=(1, 2, 3))
        return torch.stack((values * 0 + 1, values * 0 + 2, values * 0 + 3), dim=1)


def _fake_clip_module(calls: list[dict[str, object]]) -> ModuleType:
    module = ModuleType("clip")

    def load(path, *, device, jit, download_root):
        calls.append(
            {
                "path": path,
                "device": device,
                "jit": jit,
                "download_root": download_root,
            }
        )
        return _FakeModel(), lambda image: torch.ones(3, 2, 2)

    module.load = load  # type: ignore[attr-defined]
    return module


def _config(tmp_path: Path, *, count: int = 3) -> tuple[dict[str, object], Path, Path]:
    checkpoint = tmp_path / "ViT-B-32.pt"
    checkpoint.write_bytes(b"local openai checkpoint")
    tokenizer_models = []
    for index in (1, 2):
        root = tmp_path / f"tokenizer-{index}"
        root.mkdir()
        vocab = root / "vocab.json"
        merges = root / "merges.txt"
        vocab.write_text("{}\n", encoding="utf-8")
        merges.write_text("#version: 0.2\n", encoding="utf-8")
        tokenizer_models.append(
            {
                "id": "openai/clip-tokenizer",
                "revision": "b" * 40,
                "files": [_binding(vocab), _binding(merges)],
            }
        )
    model = {
        "id": "openai/ViT-B-32",
        "revision": "a" * 40,
        "files": [_binding(checkpoint)],
    }
    semantic_index = tmp_path / "semantic.npz"
    image_index = tmp_path / "image.npz"
    ids = np.asarray([f"protected-{i}" for i in range(count)])
    embeddings = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (count, 1))
    np.savez(semantic_index, ids=ids, embeddings=embeddings)
    np.savez(image_index, ids=ids, embeddings=embeddings)
    phash_index = tmp_path / "phash.jsonl"
    phash_index.write_bytes(
        b"".join(
            (
                json.dumps({"id": f"protected-{i}", "phash": f"{i + 1:016x}"}) + "\n"
            ).encode()
            for i in range(count)
        )
    )
    protected = {
        "holdout_rows": count,
        "normalized_unique_prompts": count,
        "unique_images": count,
        "semantic_text": _binding(semantic_index),
        "phash": _binding(phash_index),
        "image_embedding": _binding(image_index),
    }
    config: dict[str, object] = {
        "classifier": {"model": model},
        "semantic_text": {"model": model},
        "image_embedding": {"model": model},
        "tokenizers": [
            {"id": "sdxl_tokenizer", "model": tokenizer_models[0]},
            {"id": "sdxl_tokenizer_2", "model": tokenizer_models[1]},
        ],
        "clip_tokenizer_id": "sdxl_tokenizer",
        "protected_index": protected,
    }
    model_hash = model_binding_sha256(model)
    config["runtime_bindings"] = {
        "classifier": model_hash,
        "semantic_text": model_hash,
        "image_embedding": model_hash,
        "tokenizer:sdxl_tokenizer": model_binding_sha256(tokenizer_models[0]),
        "tokenizer:sdxl_tokenizer_2": model_binding_sha256(tokenizer_models[1]),
        "protected:semantic_text": protected["semantic_text"]["sha256"],
        "protected:phash": protected["phash"]["sha256"],
        "protected:image_embedding": protected["image_embedding"]["sha256"],
    }
    config["runtime_index_counts"] = {
        "semantic_text": count,
        "phash": count,
        "image_embedding": count,
    }
    return config, checkpoint, phash_index


def _image() -> DecodedImage:
    image = Image.new("RGB", (2, 2), color=(128, 64, 32))
    pixels = image.tobytes()
    digest = hashlib.sha256(
        (2).to_bytes(8, "big") + (2).to_bytes(8, "big") + pixels
    ).hexdigest()
    return DecodedImage(2, 2, pixels, digest)


@pytest.fixture
def fake_dependencies(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []
    monkeypatch.setitem(sys.modules, "clip", _fake_clip_module(calls))
    transformers = ModuleType("transformers")
    transformers.CLIPTokenizer = _FakeTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    _FakeTokenizer.roots.clear()
    _FakeTokenizer.used_roots.clear()
    return calls


def test_backend_loads_bound_assets_and_implements_all_gates(
    tmp_path: Path, fake_dependencies
):
    config, checkpoint, _ = _config(tmp_path)
    runtime = build_runtime_v1(config, tmp_path, tmp_path)

    assert dict(runtime.bindings) == config["runtime_bindings"]
    assert dict(runtime.index_counts) == {
        "semantic_text": 3,
        "phash": 3,
        "image_embedding": 3,
    }
    assert (
        runtime.tokenize(
            "sdxl_tokenizer",
            "a prompt",
            max_tokens=77,
            add_special_tokens=True,
            truncation=False,
        ).token_count
        == 3
    )
    classifier = runtime.classify(
        "record",
        _image(),
        {"nature": ["a nature image"], "urban": ["an urban image"]},
    )
    assert classifier.stratum == "nature"
    assert runtime.nearest_protected_text("a prompt").nearest_id == "protected-0"
    assert (
        runtime.nearest_protected_phash("0000000000000001").nearest_id == "protected-0"
    )
    assert runtime.nearest_protected_image(_image()).nearest_id == "protected-0"
    assert len(fake_dependencies) == 1, (
        "shared model descriptors should load one checkpoint"
    )
    assert fake_dependencies[0]["path"] == str(checkpoint)
    assert fake_dependencies[0]["device"] == "cpu"
    assert fake_dependencies[0]["jit"] is False
    assert all(path.name.startswith("tokenizer-") for path in _FakeTokenizer.roots)
    runtime.close()


def test_formal_parent_requires_protected_manifest_before_model_loading(
    tmp_path: Path, fake_dependencies
) -> None:
    config, _checkpoint, _ = _config(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "repldm.data_catalog.v1",
                "release_id": "catalog-" + "1" * 20,
                "candidate_catalog_complete": True,
                "development_build": False,
                "verify_paths": True,
                "training_ready": False,
                "complete": False,
                "protected_normalized_unique_prompts": 46619,
                "protected_unique_images": 37160,
                "artifacts": [
                    {"path": "benchmark_holdouts.jsonl", "rows": 49393}
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="requires a protected index manifest"):
        build_runtime_v1(config, tmp_path, tmp_path)
    assert fake_dependencies == []


def test_backend_uses_the_declared_clip_tokenizer_after_list_reordering(
    tmp_path: Path, fake_dependencies
):
    config, _, _ = _config(tmp_path)
    config["clip_tokenizer_id"] = "sdxl_tokenizer_2"
    config["tokenizers"] = list(reversed(config["tokenizers"]))

    runtime = build_runtime_v1(config, tmp_path, tmp_path)
    runtime.nearest_protected_text("a prompt")
    runtime.classify(
        "record",
        _image(),
        {"nature": ["a nature image"], "urban": ["an urban image"]},
    )

    assert _FakeTokenizer.used_roots
    assert all(path.name == "tokenizer-2" for path in _FakeTokenizer.used_roots)
    runtime.close()


def test_protected_text_encoding_uses_fixed_context_truncation(
    tmp_path: Path, fake_dependencies, monkeypatch: pytest.MonkeyPatch
):
    config, _, _ = _config(tmp_path)
    transformers = ModuleType("transformers")
    transformers.CLIPTokenizer = _TruncationCheckingTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    runtime = build_runtime_v1(config, tmp_path, tmp_path)
    assert runtime.nearest_protected_text("a long protected caption").nearest_id == "protected-0"
    runtime.close()


@pytest.mark.parametrize("value", [None, "missing_tokenizer"])
def test_backend_requires_a_declared_clip_tokenizer(
    tmp_path: Path, fake_dependencies, value
):
    config, _, _ = _config(tmp_path)
    if value is None:
        config.pop("clip_tokenizer_id")
    else:
        config["clip_tokenizer_id"] = value
    with pytest.raises(RuntimeError, match="CLIP tokenizer binding"):
        build_runtime_v1(config, tmp_path, tmp_path)


def test_backend_rejects_checkpoint_tampering_before_loading(
    tmp_path: Path, fake_dependencies
):
    config, checkpoint, _ = _config(tmp_path)
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="differs from its config binding"):
        build_runtime_v1(config, tmp_path, tmp_path)
    assert fake_dependencies == []


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("row_count", "row count differs"),
        ("non_normalized", "row-normalized"),
        ("non_finite", "non-finite"),
    ],
)
def test_backend_rejects_invalid_embedding_indexes(
    tmp_path: Path, fake_dependencies, mutation: str, message: str
):
    config, _, _ = _config(tmp_path)
    path = Path(config["protected_index"]["semantic_text"]["path"])
    if mutation == "row_count":
        np.savez(
            path,
            ids=np.asarray(["only-one"]),
            embeddings=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        )
    elif mutation == "non_normalized":
        np.savez(
            path,
            ids=np.asarray([f"protected-{i}" for i in range(3)]),
            embeddings=np.ones((3, 3), dtype=np.float32),
        )
    else:
        values = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (3, 1))
        values[1, 0] = np.nan
        np.savez(
            path,
            ids=np.asarray([f"protected-{i}" for i in range(3)]),
            embeddings=values,
        )
    config["protected_index"]["semantic_text"] = _binding(path)
    with pytest.raises(RuntimeError, match=message):
        build_runtime_v1(config, tmp_path, tmp_path)


def test_backend_rejects_invalid_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config, _, _ = _config(tmp_path)
    monkeypatch.setenv("REPLDM_SELECTED_RUNTIME_DEVICE", "bogus")
    with pytest.raises(RuntimeError, match="must be cpu or cuda"):
        build_runtime_v1(config, tmp_path, tmp_path)


def test_backend_rejects_phash_schema_and_binding_tampering(
    tmp_path: Path, fake_dependencies
):
    config, _, phash_path = _config(tmp_path)
    phash_path.write_text(
        json.dumps({"id": "protected-0", "phash": "bad"}) + "\n",
        encoding="utf-8",
    )
    config["protected_index"]["phash"] = _binding(phash_path)
    with pytest.raises(RuntimeError, match="invalid hash"):
        build_runtime_v1(config, tmp_path, tmp_path)
