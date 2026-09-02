"""Source adapters for the five user-provided external directories."""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from .io import iter_jsonl
from .schema import benchmark_matches, make_record


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def resolve_path(value: str | Path, repository_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def path_within_root(path: str | Path, root: str | Path, *, label: str) -> Path:
    """Resolve a payload path and reject traversal or symlink escape."""
    _reject_symlink_path(path, label=label)
    _reject_symlink_path(root, label=label)
    root_lexical = Path(os.path.abspath(root))
    candidate_lexical = Path(os.path.abspath(path))
    try:
        relative = candidate_lexical.relative_to(root_lexical)
    except ValueError:
        relative = None
    if root_lexical.is_symlink():
        raise ValueError(f"{label} uses a symlinked root: {root}")
    if relative is not None:
        cursor = root_lexical
        for component in relative.parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise ValueError(f"{label} uses a symlinked path: {path}")
    candidate = Path(path).resolve(strict=False)
    root_path = Path(root).resolve(strict=True)
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its declared root: {path}") from exc
    return candidate


def _reject_symlink_path(path: str | Path, *, label: str) -> None:
    """Reject a path or any existing parent component that is a symlink."""
    lexical = Path(os.path.abspath(path))
    cursor = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(f"{label} uses a symlinked path: {path}")


def iter_image_files(root: Path) -> Iterator[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {root}")
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


@dataclass(frozen=True)
class SourceContext:
    repository_root: Path
    protected_prompts: Mapping[str, set[str]]
    allowed_training_license_statuses: frozenset[str]
    physical_roots: Mapping[str, Path] = field(default_factory=dict)

    def declared_root(
        self, root: str | Path, *, source_roots: Iterable[str], label: str
    ) -> Path:
        """Resolve a dataset root and bind it to its declared physical source."""
        names = tuple(str(name) for name in source_roots)
        _reject_symlink_path(root, label=label)
        candidate = Path(root).resolve(strict=True)
        if self.physical_roots:
            unknown = [name for name in names if name not in self.physical_roots]
            if unknown or not names:
                raise ValueError(f"{label} names unknown or empty physical source roots: {names}")
        allowed = [self.physical_roots[name] for name in names if name in self.physical_roots]
        if not allowed:
            # Small adapter fixtures may omit the optional physical-root map;
            # their own configured root still receives traversal checks.
            return candidate
        for physical in allowed:
            try:
                candidate.relative_to(Path(physical).resolve(strict=True))
                return candidate
            except ValueError:
                continue
        raise ValueError(f"{label} is outside its declared source roots: {root}")

    def metadata_path(
        self, path: str | Path, *, source_roots: Iterable[str], label: str
    ) -> Path:
        """Resolve and bind a manifest or auxiliary file to declared roots."""
        names = tuple(str(name) for name in source_roots)
        _reject_symlink_path(path, label=label)
        candidate = Path(path).resolve(strict=True)
        if self.physical_roots:
            unknown = [name for name in names if name not in self.physical_roots]
            if unknown or not names:
                raise ValueError(f"{label} names unknown or empty physical source roots: {names}")
            for name in names:
                physical = self.physical_roots[name]
                try:
                    return path_within_root(candidate, physical, label=label)
                except ValueError:
                    continue
            raise ValueError(f"{label} is outside its declared source roots: {path}")
        return candidate

    def payload_path(
        self, path: str | Path, *, source_roots: Iterable[str], label: str
    ) -> Path:
        """Bind a payload to one of the physical roots named by a record."""
        names = tuple(str(name) for name in source_roots)
        _reject_symlink_path(path, label=label)
        if self.physical_roots:
            unknown = [name for name in names if name not in self.physical_roots]
            if unknown or not names:
                raise ValueError(f"{label} names unknown or empty physical source roots: {names}")
        roots = [self.physical_roots[name] for name in names if name in self.physical_roots]
        # Keep the lexical path intact until ``path_within_root`` performs its
        # symlink-component audit.  Resolving first would erase a link that
        # points back inside the declared root and silently weaken the gate.
        candidate = Path(path)
        if not roots:
            raise ValueError(f"no physical root is registered for {label}")
        for root in roots:
            try:
                return path_within_root(candidate, root, label=label)
            except ValueError:
                continue
        raise ValueError(f"{label} escapes all declared source roots: {path}")

    def record(
        self,
        *,
        prompt: str | None,
        additional_firewall_prompts: Iterable[str | None] = (),
        **kwargs: Any,
    ) -> dict[str, Any]:
        if (
            kwargs.get("training_eligible") is True
            and kwargs.get("license_status") not in self.allowed_training_license_statuses
        ):
            kwargs["training_eligible"] = False
            kwargs["exclusion_reason"] = "license_status_not_allowed_for_training"
        matches = set(benchmark_matches(prompt, self.protected_prompts))
        for candidate in additional_firewall_prompts:
            matches.update(benchmark_matches(candidate, self.protected_prompts))
        return make_record(
            prompt=prompt,
            benchmark_exact_match=sorted(matches),
            **kwargs,
        )


def iter_four_k_lsdb(spec: Mapping[str, Any], context: SourceContext) -> Iterator[dict[str, Any]]:
    manifest = context.metadata_path(
        resolve_path(spec["manifest"], context.repository_root),
        source_roots=spec["source_roots"],
        label="4KLSDB manifest",
    )
    manifest_path = str(manifest)
    for line_number, row in enumerate(iter_jsonl(manifest), 1):
        raw_image_path = Path(row["path"])
        if not raw_image_path.is_absolute():
            raw_image_path = manifest.parent / raw_image_path
        image_path = context.payload_path(
            raw_image_path,
            source_roots=spec["source_roots"],
            label=f"4KLSDB image at line {line_number}",
        )
        yield context.record(
            source=spec["id"],
            stable_key=image_path.name,
            source_roots=spec["source_roots"],
            split="train",
            prompt=row.get("caption"),
            image_path=image_path,
            width=row.get("w"),
            height=row.get("h"),
            license_name="CC-BY-4.0",
            license_status="verified_from_dataset_card",
            modality="image_text",
            intended_use=(
                "opd_teacher_targets",
                "dpo_candidate_generation",
                "rl_prompt_pool",
                "high_resolution_finetuning",
            ),
            training_eligible=True,
            source_record={
                "source_file": manifest_path,
                "line_number": line_number,
                "file_name": image_path.name,
            },
        )


def _image_name_index(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in iter_image_files(root):
        previous = result.setdefault(path.name, path)
        if previous != path:
            raise ValueError(f"duplicate image filename under {root}: {path.name}")
    return result


def iter_pixverve(spec: Mapping[str, Any], context: SourceContext) -> Iterator[dict[str, Any]]:
    manifest = context.metadata_path(
        resolve_path(spec["manifest"], context.repository_root),
        source_roots=spec["source_roots"],
        label="PixVerve manifest",
    )
    manifest_path = str(manifest)
    image_root = context.declared_root(
        resolve_path(spec["image_root"], context.repository_root),
        source_roots=spec["source_roots"],
        label="PixVerve image root",
    )
    image_paths = _image_name_index(image_root)
    for line_number, row in enumerate(iter_jsonl(manifest), 1):
        file_name = str(row["file_name"])
        try:
            image_path = path_within_root(
                image_paths.pop(file_name), image_root, label="PixVerve image"
            )
        except KeyError as exc:
            raise FileNotFoundError(f"PixVerve metadata has no image: {file_name}") from exc
        short_caption = row.get("short_caption")
        long_caption = row.get("long_caption")
        model_prompt = short_caption.strip() if isinstance(short_caption, str) else ""
        raw_prompt = long_caption.strip() if isinstance(long_caption, str) else ""
        prompt = model_prompt or raw_prompt or None
        missing_model_prompt = not bool(model_prompt)
        yield context.record(
            source=spec["id"],
            stable_key=file_name,
            source_roots=spec["source_roots"],
            split="train",
            prompt=prompt,
            additional_firewall_prompts=(long_caption,),
            image_path=image_path,
            license_name="Apache-2.0",
            license_status="verified_from_dataset_card",
            modality="image_text",
            intended_use=(
                "opd_teacher_targets",
                "dpo_candidate_generation",
                "rl_prompt_pool",
                "high_resolution_finetuning",
            ),
            training_eligible=not missing_model_prompt,
            exclusion_reason=("missing_model_prompt" if missing_model_prompt else None),
            source_record={
                "source_file": manifest_path,
                "line_number": line_number,
                "file_name": file_name,
                "data_type": row.get("data_type"),
                "model_prompt": model_prompt or None,
                "raw_prompt": raw_prompt or None,
                "short_caption": short_caption,
                "long_caption": long_caption,
            },
        )
    if image_paths:
        sample = sorted(image_paths)[:3]
        raise ValueError(f"PixVerve has {len(image_paths):,} images without metadata: {sample}")


def _prompt_rows(path: Path, source_format: str) -> Iterator[tuple[int, str, dict[str, Any]]]:
    if source_format == "text":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                prompt = line.strip()
                if prompt:
                    yield line_number, prompt, {}
        return
    if source_format == "jsonl":
        for line_number, row in enumerate(iter_jsonl(path), 1):
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"missing prompt at {path}:{line_number}")
            yield line_number, prompt.strip(), row
        return
    raise ValueError(f"unsupported prompt source format: {source_format!r}")


def iter_sana_prompts(spec: Mapping[str, Any], context: SourceContext) -> Iterator[dict[str, Any]]:
    for split_spec in spec["splits"]:
        split = str(split_spec["name"])
        path = context.metadata_path(
            resolve_path(split_spec["path"], context.repository_root),
            source_roots=spec["source_roots"],
            label=f"Sana prompt source {split}",
        )
        source_file = str(path)
        eligible = split_spec["training_eligible"]
        if not isinstance(eligible, bool):
            raise ValueError(f"Sana prompt split {split!r} training_eligible must be boolean")
        reason = split_spec.get("exclusion_reason")
        for line_number, prompt, metadata in _prompt_rows(path, split_spec["format"]):
            yield context.record(
                source=spec["id"],
                stable_key=f"{split}:{line_number}",
                source_roots=spec["source_roots"],
                split=split,
                prompt=prompt,
                image_path=None,
                license_name=spec["license"],
                license_status=spec["license_status"],
                modality="prompt",
                intended_use=tuple(spec["intended_use"]),
                training_eligible=eligible,
                exclusion_reason=reason,
                source_record={
                    "source_file": source_file,
                    "line_number": line_number,
                    "metadata": metadata,
                },
            )


def iter_aesthetic_4k(spec: Mapping[str, Any], context: SourceContext) -> Iterator[dict[str, Any]]:
    root = context.declared_root(
        resolve_path(spec["root"], context.repository_root),
        source_roots=spec["source_roots"],
        label="Aesthetic-4K root",
    )
    train_root = root / "train/images"
    for image_path in iter_image_files(train_root):
        image_path = path_within_root(
            image_path, root, label="Aesthetic-4K training image"
        )
        relative = image_path.relative_to(root).as_posix()
        yield context.record(
            source=spec["id"],
            stable_key=relative,
            source_roots=spec["source_roots"],
            split="train",
            prompt=None,
            image_path=image_path,
            license_name="MIT",
            license_status="verified_from_dataset_card",
            modality="image_only",
            intended_use=("latent_structure_pretraining",),
            training_eligible=True,
            source_record={"relative_path": relative},
        )
    for sidecar in sorted((root / "eval").glob("size_*/metadata.jsonl")):
        sidecar = context.metadata_path(
            sidecar, source_roots=spec["source_roots"], label="Aesthetic-4K metadata"
        )
        source_file = str(sidecar)
        for line_number, row in enumerate(iter_jsonl(sidecar), 1):
            image_path = path_within_root(
                sidecar.parent / str(row["file_name"]),
                sidecar.parent,
                label="Aesthetic-4K evaluation image",
            )
            yield context.record(
                source=spec["id"],
                stable_key=image_path.relative_to(root).as_posix(),
                source_roots=spec["source_roots"],
                split=sidecar.parent.name,
                prompt=row.get("text"),
                image_path=image_path,
                license_name="MIT",
                license_status="verified_from_dataset_card",
                modality="benchmark_image_text",
                intended_use=("aesthetic4k_evaluation_only",),
                training_eligible=False,
                exclusion_reason="dataset_eval_split",
                source_record={
                    "source_file": source_file,
                    "line_number": line_number,
                    "file_name": row["file_name"],
                },
            )


def _load_style_names(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("style_id2name")
    if not isinstance(values, dict):
        raise ValueError(f"style_id2name is missing from {path}")
    return {str(key): str(value) for key, value in values.items()}


def iter_style30k(spec: Mapping[str, Any], context: SourceContext) -> Iterator[dict[str, Any]]:
    root = context.declared_root(
        resolve_path(spec["root"], context.repository_root),
        source_roots=spec["source_roots"],
        label="Style30K root",
    )
    listing = context.metadata_path(
        root / "style30k/list.txt", source_roots=spec["source_roots"], label="Style30K listing"
    )
    source_file = str(listing)
    style_mapper = context.metadata_path(
        root / "style30k/style_mapper.json",
        source_roots=spec["source_roots"],
        label="Style30K style mapper",
    )
    style_names = _load_style_names(style_mapper)
    with listing.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            relative = line.strip()
            if not relative:
                continue
            image_path = path_within_root(
                root / relative, root, label="Style30K image"
            )
            match = re.match(r"(s\d{4})____", image_path.name)
            if not match:
                raise ValueError(f"Style30K filename lacks style id: {image_path.name}")
            style_id = match.group(1)
            if style_id not in style_names:
                raise ValueError(f"Style30K style id is not mapped: {style_id}")
            yield context.record(
                source=spec["id"],
                stable_key=relative,
                source_roots=spec["source_roots"],
                split="train",
                prompt=None,
                image_path=image_path,
                license_name="CC-BY-4.0",
                license_status="verified_from_dataset_card",
                modality="image_style",
                intended_use=("style_structure_pretraining",),
                training_eligible=True,
                source_record={
                    "source_file": source_file,
                    "line_number": line_number,
                    "style_id": style_id,
                    "style_name": style_names[style_id],
                },
            )


def iter_omnistyle(spec: Mapping[str, Any], context: SourceContext) -> Iterator[dict[str, Any]]:
    root = context.declared_root(
        resolve_path(spec["root"], context.repository_root),
        source_roots=spec["source_roots"],
        label="OmniStyle root",
    )
    style_names = _load_style_names(
        context.metadata_path(
            resolve_path(spec["style_mapper"], context.repository_root),
            source_roots=spec["source_roots"],
            label="OmniStyle style mapper",
        )
    )
    for role in ("content", "style"):
        for image_path in iter_image_files(root / role):
            image_path = path_within_root(
                image_path, root, label="OmniStyle image"
            )
            relative = image_path.relative_to(root).as_posix()
            prompt: str | None = None
            metadata: dict[str, Any] = {"role": role, "relative_path": relative}
            modality = "image_text" if role == "content" else "image_style"
            if role == "content":
                category, separator, description = image_path.stem.partition("_")
                prompt = description.strip() if separator else image_path.stem
                metadata["category"] = category
            else:
                match = re.match(r"(s\d{4})____", image_path.name)
                style_id = match.group(1) if match else ""
                metadata.update({"style_id": style_id, "style_name": style_names.get(style_id)})
            yield context.record(
                source=spec["id"],
                stable_key=relative,
                source_roots=spec["source_roots"],
                split=f"available_{role}",
                prompt=prompt,
                image_path=image_path,
                license_name="not_bundled",
                license_status="review_required",
                modality=modality,
                intended_use=("style_or_content_pretraining_after_license_review",),
                training_eligible=False,
                exclusion_reason="dataset_license_not_bundled",
                source_record=metadata,
            )


PIXEL_RENDER_LICENSES = {
    "data/unsplash-full": ("Unsplash Dataset Terms", "noncommercial_research"),
    "data/LSR": ("mixed_or_unknown_upstream_image_terms", "review_required"),
    "data/laion-4k": ("mixed_or_unknown_upstream_image_terms", "review_required"),
    "data/4kwallpapers": ("not_bundled", "review_required"),
    "data/wallhaven_images": ("not_bundled", "review_required"),
    "data/downloaded_wallpapers": ("not_bundled", "review_required"),
}


def iter_pixel_render_images(spec: Mapping[str, Any], context: SourceContext) -> Iterator[dict[str, Any]]:
    manifest = context.metadata_path(
        resolve_path(spec["manifest"], context.repository_root),
        source_roots=spec["source_roots"],
        label="Pixel-Render manifest",
    )
    manifest_path = str(manifest)
    data_root = context.declared_root(
        resolve_path(spec["data_root"], context.repository_root),
        source_roots=spec["source_roots"],
        label="Pixel-Render data root",
    )
    numbered_rows = enumerate(iter_jsonl(manifest), 1)
    with ThreadPoolExecutor(max_workers=32) as executor:
        while batch := list(islice(numbered_rows, 4096)):
            prepared = []
            for line_number, row in batch:
                relative = str(row["path"])
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(
                        f"Pixel-Render manifest path escapes its data root: {relative}"
                    )
                prefix = "/".join(relative_path.parts[:2])
                try:
                    license_name, license_status = PIXEL_RENDER_LICENSES[prefix]
                except KeyError as exc:
                    raise ValueError(
                        f"unclassified Pixel-Render source prefix: {prefix}"
                    ) from exc
                declared_path = path_within_root(
                    data_root / relative_path,
                    data_root,
                    label="Pixel-Render manifest image",
                )
                prepared.append(
                    (
                        line_number,
                        row,
                        relative,
                        prefix,
                        license_name,
                        license_status,
                        declared_path,
                    )
                )

            exists_flags = executor.map(
                os.path.isfile, (str(item[-1]) for item in prepared)
            )
            for item, image_exists in zip(prepared, exists_flags):
                (
                    line_number,
                    row,
                    relative,
                    prefix,
                    license_name,
                    license_status,
                    declared_path,
                ) = item
                if not image_exists:
                    exclusion_reason = "source_image_missing"
                else:
                    exclusion_reason = "source_not_approved_for_training"
                yield context.record(
                    source=spec["id"],
                    stable_key=relative,
                    source_roots=spec["source_roots"],
                    split="train",
                    prompt=None,
                    image_path=declared_path if image_exists else None,
                    width=row.get("width"),
                    height=row.get("height"),
                    license_name=license_name,
                    license_status=license_status,
                    modality="image_only" if image_exists else "missing_image_reference",
                    payload_integrity=(
                        "unbound_path_no_checksum" if image_exists else "missing_reference"
                    ),
                    intended_use=(
                        "latent_structure_pretraining",
                        "autoencoder_frequency_analysis",
                    )
                    if image_exists
                    else ("source_inventory_only",),
                    training_eligible=False,
                    exclusion_reason=exclusion_reason,
                    source_record={
                        "source_file": manifest_path,
                        "line_number": line_number,
                        "relative_path": relative,
                        "source_prefix": prefix,
                        "image_exists": image_exists,
                        **(
                            {"declared_image_path": str(declared_path)}
                            if not image_exists
                            else {}
                        ),
                    },
                )


def iter_sana_feature_cache(spec: Mapping[str, Any], context: SourceContext) -> Iterator[dict[str, Any]]:
    root = context.declared_root(
        resolve_path(spec["root"], context.repository_root),
        source_roots=spec["source_roots"],
        label="Sana feature-cache root",
    )
    meta_path = context.metadata_path(
        root / "meta.json", source_roots=spec["source_roots"], label="Sana feature-cache metadata"
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    total = int(meta["total"])
    num_shards = int(meta["num_shards"])
    for shard in range(num_shards):
        rows = len(range(shard, total, num_shards))
        feature_path = context.metadata_path(
            root / f"feat_shard{shard}.bin",
            source_roots=spec["source_roots"],
            label="Sana feature-cache feature shard",
        )
        mask_path = context.metadata_path(
            root / f"mask_shard{shard}.bin",
            source_roots=spec["source_roots"],
            label="Sana feature-cache mask shard",
        )
        expected_feature_bytes = rows * int(meta["model_max_length"]) * int(meta["dim"]) * 2
        expected_mask_bytes = rows * int(meta["model_max_length"]) * 2
        if feature_path.stat().st_size != expected_feature_bytes:
            raise ValueError(f"incomplete Sana feature shard: {feature_path}")
        if mask_path.stat().st_size != expected_mask_bytes:
            raise ValueError(f"incomplete Sana mask shard: {mask_path}")
        yield context.record(
            source=spec["id"],
            stable_key=str(shard),
            source_roots=spec["source_roots"],
            split="derived_cache",
            prompt=None,
            image_path=None,
            license_name="derived_from_4klsdb_and_gemma_2_2b_it",
            license_status="model_and_dataset_terms_apply",
            modality="text_feature_cache",
            payload_integrity="unbound_auxiliary_payload",
            intended_use=("sana_backbone_text_embedding_cache",),
            training_eligible=False,
            exclusion_reason="auxiliary_cache_not_a_standalone_training_sample",
            source_record={
                "meta_file": str(meta_path.resolve()),
                "shard": shard,
                "rows": rows,
                "feature_path": str(feature_path.resolve()),
                "feature_bytes": expected_feature_bytes,
                "mask_path": str(mask_path.resolve()),
                "mask_bytes": expected_mask_bytes,
                "ids_sha256": meta["ids_sha256"],
                "text_encoder_name": meta["text_encoder_name"],
                "model_max_length": meta["model_max_length"],
                "dim": meta["dim"],
                "dtype": meta["dtype"],
            },
        )


ADAPTERS: dict[str, Callable[[Mapping[str, Any], SourceContext], Iterable[dict[str, Any]]]] = {
    "four_k_lsdb": iter_four_k_lsdb,
    "pixverve": iter_pixverve,
    "sana_prompts": iter_sana_prompts,
    "aesthetic_4k": iter_aesthetic_4k,
    "style30k": iter_style30k,
    "omnistyle": iter_omnistyle,
    "pixel_render_images": iter_pixel_render_images,
    "sana_feature_cache": iter_sana_feature_cache,
}


def iter_dataset_records(
    spec: Mapping[str, Any], context: SourceContext
) -> Iterable[dict[str, Any]]:
    adapter_name = str(spec["adapter"])
    try:
        adapter = ADAPTERS[adapter_name]
    except KeyError as exc:
        raise ValueError(f"unknown data adapter: {adapter_name!r}") from exc
    return adapter(spec, context)


def four_k_lsdb_ids_sha256(manifest: Path) -> str:
    ids: list[str] = []
    for row in iter_jsonl(manifest):
        ids.append(Path(row["path"]).stem)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
