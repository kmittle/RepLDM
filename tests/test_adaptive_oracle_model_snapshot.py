import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "eval-pipeline" / "adaptive_oracle_model_snapshot.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "adaptive_oracle_model_snapshot_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


snapshot = load_module()


def manifest_files(payloads):
    return {
        name: (len(value), hashlib.sha256(value).hexdigest())
        for name, value in payloads.items()
    }


class SnapshotFixture:
    def __init__(self, directory, payloads):
        self.root = Path(directory)
        self.model_repository = (
            self.root / snapshot.MODEL_CACHE / snapshot.MODEL_REPOSITORY_DIR
        )
        self.model_root = (
            self.model_repository / "snapshots" / snapshot.MODEL_REVISION
        )
        self.staging_parent = self.root / "staging"
        self.staging_parent.mkdir(parents=True)
        self.staging_parent.chmod(0o700)
        self.payloads = dict(payloads)
        self.files = manifest_files(payloads)
        self.manifest_sha256 = snapshot.canonical_sha256(
            snapshot.expected_model_manifest(self.files)
        )

    def write_regular_sources(self):
        for name, value in self.payloads.items():
            path = self.model_root.joinpath(*Path(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)

    def write_hf_blob_source(self, name, value, *, blob_name="a" * 64):
        blob = self.model_repository / "blobs" / blob_name
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(value)
        path = self.model_root.joinpath(*Path(name).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(os.path.relpath(blob, path.parent))
        return path, blob

    def stage(self):
        return snapshot.stage_model_snapshot(
            self.root,
            staging_parent=self.staging_parent,
            files=self.files,
            expected_manifest_sha256=self.manifest_sha256,
        )

    def verify(self, record):
        return snapshot.verify_staged_model_snapshot(
            record,
            files=self.files,
            expected_manifest_sha256=self.manifest_sha256,
        )


class AdaptiveOracleModelSnapshotTest(unittest.TestCase):
    def test_frozen_manifest_is_complete_and_self_consistent(self):
        manifest = snapshot.expected_model_manifest()
        self.assertEqual(manifest["loaded_file_count"], 18)
        self.assertEqual(manifest["model_id"], snapshot.MODEL_ID)
        self.assertEqual(manifest["revision"], snapshot.MODEL_REVISION)
        self.assertEqual(
            snapshot.canonical_sha256(manifest),
            snapshot.MODEL_SNAPSHOT_MANIFEST_SHA256,
        )
        paths = [row["path"] for row in manifest["files"]]
        self.assertEqual(paths, sorted(paths))
        self.assertEqual(len(paths), len(set(paths)))
        self.assertIn("unet/diffusion_pytorch_model.fp16.safetensors", paths)
        self.assertNotIn("unet/model.onnx", paths)

    def test_validate_uses_stable_non_following_source_descriptors(self):
        payloads = {"unet/config.json": b"config", "unet/model.bin": b"weights"}
        with tempfile.TemporaryDirectory() as directory:
            fixture = SnapshotFixture(directory, payloads)
            fixture.write_regular_sources()
            observed = snapshot.validate_model_snapshot(
                fixture.root,
                files=fixture.files,
                expected_manifest_sha256=fixture.manifest_sha256,
            )
            self.assertEqual(observed["manifest_sha256"], fixture.manifest_sha256)
            (fixture.model_root / "unet" / "model.bin").write_bytes(b"drifted")
            with self.assertRaisesRegex(ValueError, "size differs|SHA-256 differs"):
                snapshot.validate_model_snapshot(
                    fixture.root,
                    files=fixture.files,
                    expected_manifest_sha256=fixture.manifest_sha256,
                )

    def test_stage_verify_and_cleanup_records(self):
        payloads = {"model_index.json": b"index", "unet/model.bin": b"weights"}
        with tempfile.TemporaryDirectory() as directory:
            fixture = SnapshotFixture(directory, payloads)
            fixture.write_regular_sources()
            record = fixture.stage()
            stage_root = Path(record["path"])
            self.assertEqual(record["schema"], snapshot.MODEL_STAGE_SCHEMA)
            self.assertEqual(record["loaded_file_count"], 2)
            self.assertEqual(stat.S_IMODE(stage_root.stat().st_mode), 0o500)
            for name, value in payloads.items():
                staged = stage_root.joinpath(*Path(name).parts)
                self.assertFalse(staged.is_symlink())
                self.assertEqual(staged.read_bytes(), value)
                self.assertEqual(stat.S_IMODE(staged.stat().st_mode), 0o400)

            verification = fixture.verify(record)
            self.assertEqual(
                verification["schema"], snapshot.MODEL_STAGE_VERIFICATION_SCHEMA
            )
            self.assertEqual(verification["status"], "verified_unchanged")
            cleanup = snapshot.cleanup_staged_model_snapshot(record)
            self.assertEqual(cleanup["schema"], snapshot.MODEL_STAGE_CLEANUP_SCHEMA)
            self.assertEqual(cleanup["status"], "removed")
            self.assertFalse(stage_root.exists())

    @unittest.skipUnless(Path("/proc/self/fd").is_dir(), "requires procfs")
    def test_bound_loader_uses_pinned_root_and_rejects_restored_mutation(self):
        payloads = {"model.bin": b"weights"}
        with tempfile.TemporaryDirectory() as directory:
            fixture = SnapshotFixture(directory, payloads)
            fixture.write_regular_sources()
            record = fixture.stage()
            stage_root = Path(record["path"])
            observed = {}

            def mutate_and_restore(load_root):
                observed["load_root"] = load_root
                observed["loaded"] = (Path(load_root) / "model.bin").read_bytes()
                target = stage_root / "model.bin"
                target.chmod(0o600)
                target.write_bytes(b"evil")
                target.write_bytes(payloads["model.bin"])
                target.chmod(0o400)
                return object()

            try:
                with self.assertRaisesRegex(ValueError, "tree digest differs"):
                    snapshot.load_from_verified_staged_model_snapshot(
                        record,
                        mutate_and_restore,
                        files=fixture.files,
                        expected_manifest_sha256=fixture.manifest_sha256,
                    )
                self.assertTrue(observed["load_root"].startswith("/proc/self/fd/"))
                self.assertEqual(observed["loaded"], payloads["model.bin"])
            finally:
                snapshot.cleanup_staged_model_snapshot(record)

    @unittest.skipUnless(Path("/proc/self/fd").is_dir(), "requires procfs")
    def test_bound_loader_is_not_redirected_by_public_root_substitution(self):
        payloads = {"model.bin": b"weights"}
        with tempfile.TemporaryDirectory() as directory:
            fixture = SnapshotFixture(directory, payloads)
            fixture.write_regular_sources()
            record = fixture.stage()
            stage_root = Path(record["path"])
            displaced = stage_root.with_name(stage_root.name + "-held")

            def substitute_public_path(load_root):
                stage_root.rename(displaced)
                stage_root.mkdir(mode=0o700)
                (stage_root / "model.bin").write_bytes(b"evil")
                try:
                    return (Path(load_root) / "model.bin").read_bytes()
                finally:
                    (stage_root / "model.bin").unlink()
                    stage_root.rmdir()
                    displaced.rename(stage_root)

            try:
                with self.assertRaisesRegex(ValueError, "tree digest differs"):
                    snapshot.load_from_verified_staged_model_snapshot(
                        record,
                        substitute_public_path,
                        files=fixture.files,
                        expected_manifest_sha256=fixture.manifest_sha256,
                    )
            finally:
                snapshot.cleanup_staged_model_snapshot(record)

    def test_verification_rejects_byte_restoration_via_identity_signature(self):
        payloads = {"model.bin": b"weights"}
        with tempfile.TemporaryDirectory() as directory:
            fixture = SnapshotFixture(directory, payloads)
            fixture.write_regular_sources()
            record = fixture.stage()
            target = Path(record["path"]) / "model.bin"
            try:
                target.chmod(0o600)
                target.write_bytes(b"evil")
                target.write_bytes(payloads["model.bin"])
                target.chmod(0o400)
                with self.assertRaisesRegex(ValueError, "tree digest differs"):
                    fixture.verify(record)
            finally:
                snapshot.cleanup_staged_model_snapshot(record)

    def test_registered_huggingface_blob_symlink_is_safely_dereferenced(self):
        payloads = {"unet/model.bin": b"weights"}
        with tempfile.TemporaryDirectory() as directory:
            fixture = SnapshotFixture(directory, payloads)
            external_repository = fixture.root / "external-model-repository"
            external_repository.mkdir()
            fixture.model_repository.parent.mkdir(parents=True)
            fixture.model_repository.symlink_to(
                external_repository, target_is_directory=True
            )
            source, blob = fixture.write_hf_blob_source(
                "unet/model.bin", payloads["unet/model.bin"]
            )
            self.assertTrue(source.is_symlink())
            record = fixture.stage()
            try:
                source_row = record["source_snapshot"]["files"][0]
                self.assertEqual(source_row["kind"], "huggingface_blob_symlink")
                self.assertEqual(source_row["storage_path"], str(blob.resolve()))
                staged = Path(record["path"]) / "unet" / "model.bin"
                self.assertTrue(staged.is_file())
                self.assertFalse(staged.is_symlink())
                self.assertEqual(staged.read_bytes(), payloads["unet/model.bin"])
                fixture.verify(record)
            finally:
                snapshot.cleanup_staged_model_snapshot(record)

    def test_source_replacement_before_and_during_staging_fails(self):
        payloads = {"unet/model.bin": b"weights"}
        with tempfile.TemporaryDirectory() as directory:
            fixture = SnapshotFixture(directory, payloads)
            fixture.write_regular_sources()
            source = fixture.model_root / "unet" / "model.bin"
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                fixture.stage()
            self.assertEqual(list(fixture.staging_parent.iterdir()), [])

        with tempfile.TemporaryDirectory() as directory:
            fixture = SnapshotFixture(directory, payloads)
            fixture.write_regular_sources()
            source = fixture.model_root / "unet" / "model.bin"
            real_read = snapshot.os.read
            replaced = False

            def replace_after_first_read(descriptor, size):
                nonlocal replaced
                value = real_read(descriptor, size)
                if value and not replaced:
                    replacement = source.with_name("replacement.bin")
                    replacement.write_bytes(payloads["unet/model.bin"])
                    os.replace(replacement, source)
                    replaced = True
                return value

            with mock.patch.object(snapshot.os, "read", side_effect=replace_after_first_read):
                with self.assertRaisesRegex(ValueError, "changed while copied"):
                    fixture.stage()
            self.assertTrue(replaced)
            self.assertEqual(list(fixture.staging_parent.iterdir()), [])

    def test_source_replacement_after_staging_cannot_change_staged_bytes(self):
        payloads = {"unet/model.bin": b"weights"}
        with tempfile.TemporaryDirectory() as directory:
            fixture = SnapshotFixture(directory, payloads)
            fixture.write_regular_sources()
            record = fixture.stage()
            try:
                source = fixture.model_root / "unet" / "model.bin"
                replacement = source.with_name("replacement.bin")
                replacement.write_bytes(b"changed")
                os.replace(replacement, source)
                verification = fixture.verify(record)
                self.assertEqual(verification["status"], "verified_unchanged")
                self.assertEqual(
                    (Path(record["path"]) / "unet" / "model.bin").read_bytes(),
                    payloads["unet/model.bin"],
                )
            finally:
                snapshot.cleanup_staged_model_snapshot(record)

    def test_source_symlink_escape_absolute_link_and_hardlink_are_rejected(self):
        payloads = {"unet/model.bin": b"weights"}
        with tempfile.TemporaryDirectory() as directory:
            fixture = SnapshotFixture(directory, payloads)
            outside = fixture.root / "outside.bin"
            outside.write_bytes(payloads["unet/model.bin"])
            source = fixture.model_root / "unet" / "model.bin"
            source.parent.mkdir(parents=True)
            source.symlink_to(os.path.relpath(outside, source.parent))
            with self.assertRaisesRegex(ValueError, "escapes repository|inside repository blobs"):
                fixture.stage()

            source.unlink()
            source.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "target is invalid"):
                fixture.stage()

            source.unlink()
            os.link(outside, source)
            with self.assertRaisesRegex(ValueError, "hard-linked"):
                fixture.stage()

    def test_staged_extra_symlink_hardlink_and_mutation_are_rejected(self):
        payloads = {"a.bin": b"same", "b.bin": b"same"}
        cases = ("extra", "symlink", "hardlink", "mutation")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = SnapshotFixture(directory, payloads)
                fixture.write_regular_sources()
                record = fixture.stage()
                stage_root = Path(record["path"])
                try:
                    stage_root.chmod(0o700)
                    if case == "extra":
                        (stage_root / "extra.bin").write_bytes(b"extra")
                    elif case == "symlink":
                        (stage_root / "b.bin").unlink()
                        (stage_root / "b.bin").symlink_to("a.bin")
                    elif case == "hardlink":
                        (stage_root / "b.bin").unlink()
                        os.link(stage_root / "a.bin", stage_root / "b.bin")
                    else:
                        target = stage_root / "a.bin"
                        target.chmod(0o600)
                        target.write_bytes(b"evil")
                        target.chmod(0o400)
                    stage_root.chmod(0o500)
                    with self.assertRaisesRegex(
                        ValueError,
                        "inventory differs|symlink|hard-linked|SHA-256 differs",
                    ):
                        fixture.verify(record)
                finally:
                    snapshot.cleanup_staged_model_snapshot(record)

    def test_cleanup_rejects_forged_or_copied_stage_record(self):
        payloads = {"model.bin": b"weights"}
        with tempfile.TemporaryDirectory() as directory:
            fixture = SnapshotFixture(directory, payloads)
            fixture.write_regular_sources()
            record = fixture.stage()
            victim = fixture.staging_parent / "adaptive-oracle-model-victim"
            victim.mkdir()
            (victim / "keep.txt").write_text("keep", encoding="utf-8")
            forged = dict(record)
            forged["path"] = str(victim)
            forged["root_identity"] = snapshot._identity_record(victim.stat())
            try:
                with self.assertRaisesRegex(ValueError, "live creation record"):
                    snapshot.cleanup_staged_model_snapshot(forged)
                self.assertEqual(
                    (victim / "keep.txt").read_text(encoding="utf-8"), "keep"
                )
                with self.assertRaisesRegex(ValueError, "live creation record"):
                    snapshot.cleanup_staged_model_snapshot(dict(record))
            finally:
                snapshot.cleanup_staged_model_snapshot(record)

    def test_source_evidence_mutation_is_rejected(self):
        payloads = {"model.bin": b"weights"}
        with tempfile.TemporaryDirectory() as directory:
            fixture = SnapshotFixture(directory, payloads)
            fixture.write_regular_sources()
            record = fixture.stage()
            altered = dict(record)
            altered["source_snapshot"] = dict(record["source_snapshot"])
            altered["source_snapshot"]["path"] = str(fixture.root / "forged")
            altered["source_snapshot_sha256"] = snapshot.canonical_sha256(
                altered["source_snapshot"]
            )
            try:
                with self.assertRaisesRegex(ValueError, "live creation record"):
                    fixture.verify(altered)
            finally:
                snapshot.cleanup_staged_model_snapshot(record)

    def test_verification_detects_changes_after_inventory_and_file_hash(self):
        payloads = {"model.bin": b"weights"}
        for case in ("extra_after_inventory", "mutation_after_hash"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = SnapshotFixture(directory, payloads)
                fixture.write_regular_sources()
                record = fixture.stage()
                stage_root = Path(record["path"])
                real_listdir = snapshot.os.listdir
                calls = 0

                def mutate_after_listdir(descriptor):
                    nonlocal calls
                    names = real_listdir(descriptor)
                    calls += 1
                    if case == "extra_after_inventory" and calls == 1:
                        stage_root.chmod(0o700)
                        (stage_root / "extra.bin").write_bytes(b"extra")
                        stage_root.chmod(0o500)
                    elif case == "mutation_after_hash" and calls == 2:
                        target = stage_root / "model.bin"
                        target.chmod(0o600)
                        target.write_bytes(b"changed")
                        target.chmod(0o400)
                    return names

                try:
                    with mock.patch.object(
                        snapshot.os, "listdir", side_effect=mutate_after_listdir
                    ):
                        with self.assertRaisesRegex(
                            ValueError, "changed while verified|changed after verification"
                        ):
                            fixture.verify(record)
                finally:
                    snapshot.cleanup_staged_model_snapshot(record)

    @unittest.skipUnless(Path("/proc/self/fd").is_dir(), "requires procfs")
    def test_failure_paths_do_not_leak_file_descriptors(self):
        payloads = {"model.bin": b"weights"}
        with tempfile.TemporaryDirectory() as directory:
            fixture = SnapshotFixture(directory, payloads)
            fixture.write_regular_sources()
            missing_parent = fixture.root / "missing-parent"
            before = len(list(Path("/proc/self/fd").iterdir()))
            for _ in range(8):
                with self.assertRaises(FileNotFoundError):
                    snapshot.stage_model_snapshot(
                        fixture.root,
                        staging_parent=missing_parent,
                        files=fixture.files,
                        expected_manifest_sha256=fixture.manifest_sha256,
                    )
            self.assertEqual(len(list(Path("/proc/self/fd").iterdir())), before)

            record = fixture.stage()
            before_cleanup = len(list(Path("/proc/self/fd").iterdir()))
            with mock.patch.object(
                snapshot,
                "_remove_directory_contents",
                side_effect=RuntimeError("forced cleanup failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "forced cleanup failure"):
                    snapshot.cleanup_staged_model_snapshot(record)
            self.assertEqual(
                len(list(Path("/proc/self/fd").iterdir())), before_cleanup
            )
            snapshot.cleanup_staged_model_snapshot(record)

    def test_stage_creation_rejects_symlink_substitution_before_open(self):
        payloads = {"model.bin": b"weights"}
        with tempfile.TemporaryDirectory() as directory:
            fixture = SnapshotFixture(directory, payloads)
            fixture.write_regular_sources()
            real_mkdtemp = snapshot.tempfile.mkdtemp
            displaced = None

            def substitute_with_symlink(*args, **kwargs):
                nonlocal displaced
                created = Path(real_mkdtemp(*args, **kwargs))
                displaced = created.with_name(created.name + "-displaced")
                created.rename(displaced)
                created.symlink_to(displaced, target_is_directory=True)
                return str(created)

            with mock.patch.object(
                snapshot.tempfile, "mkdtemp", side_effect=substitute_with_symlink
            ):
                with self.assertRaisesRegex(
                    ValueError, "stage root|missing, linked, or invalid"
                ):
                    fixture.stage()
            self.assertIsNotNone(displaced)
            displaced.rmdir()

    def test_invalid_paths_and_digest_fail_closed(self):
        invalid = {
            "../escape": (1, "0" * 64),
            "/absolute": (1, "0" * 64),
            ".": (1, "0" * 64),
        }
        for path, value in invalid.items():
            with self.subTest(path=path), self.assertRaisesRegex(
                ValueError, "canonical"
            ):
                snapshot.expected_model_manifest({path: value})
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            snapshot.expected_model_manifest({"file": (1, "A" * 64)})


if __name__ == "__main__":
    unittest.main()
