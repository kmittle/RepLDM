import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "eval-pipeline" / "prestage_weights.py"


def load_prestage_module():
    spec = importlib.util.spec_from_file_location("repldm_prestage_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prestage = load_prestage_module()


class PrestageAssetClosureTest(unittest.TestCase):
    def test_main_returns_nonzero_and_never_claims_complete_after_failure(self):
        output = io.StringIO()

        def fail_iqa():
            raise RuntimeError("simulated TOPIQ failure")

        with mock.patch.object(prestage, "clip_models", return_value=None), \
            mock.patch.object(prestage, "imagereward", return_value=None), \
            mock.patch.object(prestage, "iqa", side_effect=fail_iqa), \
            mock.patch.object(prestage, "fix_hpsv2_vocab", return_value=None), \
            mock.patch.object(prestage, "hps", return_value=None), \
            mock.patch.object(prestage, "aesthetic", return_value=None), \
            contextlib.redirect_stdout(output):
            status = prestage.main()

        self.assertEqual(status, 1)
        self.assertIn("PRESTAGE FAILED", output.getvalue())
        self.assertIn("TOPIQ-NR checkpoint + timm ResNet-50 backbone", output.getvalue())
        self.assertNotIn("PRESTAGE COMPLETE", output.getvalue())

    def test_main_returns_zero_only_when_all_steps_succeed(self):
        output = io.StringIO()
        with mock.patch.object(prestage, "clip_models", return_value=None), \
            mock.patch.object(prestage, "imagereward", return_value=None), \
            mock.patch.object(prestage, "iqa", return_value=None), \
            mock.patch.object(prestage, "fix_hpsv2_vocab", return_value=None), \
            mock.patch.object(prestage, "hps", return_value=None), \
            mock.patch.object(prestage, "aesthetic", return_value=None), \
            contextlib.redirect_stdout(output):
            status = prestage.main()

        self.assertEqual(status, 0)
        self.assertIn("PRESTAGE COMPLETE", output.getvalue())

    def test_iqa_stages_topiq_at_pyiqa_path_and_resnet_backbone(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_topiq = root / "downloaded-topiq.pth"
            source_backbone = root / "downloaded-resnet.safetensors"
            source_topiq.write_bytes(b"topiq-checkpoint")
            source_backbone.write_bytes(b"resnet-backbone")
            target_dir = root / "torch" / "hub" / "pyiqa"
            target_path = target_dir / prestage.TOPIQ_FILENAME
            calls = []

            def fake_hf_download(repo_id, filename, **kwargs):
                calls.append((repo_id, filename, kwargs))
                if repo_id == prestage.TOPIQ_REPOSITORY:
                    return str(source_topiq)
                if repo_id == prestage.TIMM_REPOSITORY:
                    return str(source_backbone)
                raise AssertionError((repo_id, filename))

            huggingface_hub = types.ModuleType("huggingface_hub")
            huggingface_hub.hf_hub_download = fake_hf_download
            with mock.patch.dict(sys.modules, {"huggingface_hub": huggingface_hub}), \
                mock.patch.object(prestage, "TOPIQ_DIR", str(target_dir)), \
                mock.patch.object(prestage, "TOPIQ_PATH", str(target_path)):
                prestage.iqa()

            self.assertEqual(
                [(repo, filename) for repo, filename, _ in calls],
                [
                    (prestage.TOPIQ_REPOSITORY, prestage.TOPIQ_FILENAME),
                    (prestage.TIMM_REPOSITORY, prestage.TIMM_FILENAME),
                ],
            )
            self.assertEqual(target_path.read_bytes(), b"topiq-checkpoint")

    def test_imagereward_explicitly_resolves_all_bert_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_reward_dir = root / "ImageReward"
            calls = []

            def fake_hf_download(repo_id, filename, **kwargs):
                calls.append((repo_id, filename, kwargs))
                if kwargs.get("local_dir") is not None:
                    path = Path(kwargs["local_dir"]) / filename
                else:
                    path = root / "hf-cache" / repo_id / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(filename.encode("ascii"))
                return str(path)

            tokenizer_calls = []

            class FakeTokenizer:
                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    tokenizer_calls.append((args, kwargs))
                    return cls()

            huggingface_hub = types.ModuleType("huggingface_hub")
            huggingface_hub.hf_hub_download = fake_hf_download
            transformers = types.ModuleType("transformers")
            transformers.BertTokenizer = FakeTokenizer
            with mock.patch.dict(
                sys.modules,
                {"huggingface_hub": huggingface_hub, "transformers": transformers},
            ), mock.patch.object(prestage, "IMAGEREWARD_DIR", str(image_reward_dir)):
                prestage.imagereward()

            bert_calls = [
                filename
                for repo, filename, kwargs in calls
                if repo == prestage.BERT_REPOSITORY
            ]
            self.assertEqual(bert_calls, list(prestage.BERT_REQUIRED_FILES))
            self.assertEqual(
                tokenizer_calls,
                [
                    (
                        (prestage.BERT_REPOSITORY,),
                        {"local_files_only": True},
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
