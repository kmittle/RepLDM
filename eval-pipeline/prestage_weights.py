"""Pre-stage all eval reward weights into SHARED user caches — DECOUPLED from Sana.

Run ONCE in the scoring environment with network access:
    /home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/prestage_weights.py

Downloads (~7GB total) go to shared caches (~/.cache/{clip,hpsv2,aesthetic} and the
HF hub cache), NOT into another project tree. It uses upstream `clip` and `hpsv2`
packages and repairs the missing vocabulary in the vendored HPSv2 tokenizer. The
ImageReward step explicitly resolves all four BERT files used by the private
scorer, and the TOPIQ step stages both its pyiqa checkpoint and timm backbone.
After this completes, scoring runs fully offline.
"""
import os
import shutil
import traceback
import urllib.request

AESTHETIC_DIR = os.path.expanduser("~/.cache/aesthetic")
AESTHETIC_PATH = os.path.join(AESTHETIC_DIR, "sac+logos+ava1-l14-linearMSE.pth")
AESTHETIC_URL = ("https://github.com/christophschuhmann/improved-aesthetic-predictor"
                 "/raw/main/sac+logos+ava1-l14-linearMSE.pth")
IMAGEREWARD_DIR = os.path.expanduser("~/.cache/ImageReward")

# Keep these paths/repositories in sync with the private scorer asset contracts.
# TOPIQ is downloaded from the IQA-PyTorch weights repository, but pyiqa's
# legacy loader looks for it in torch's named cache, so we copy it there.
TOPIQ_REPOSITORY = "chaofengc/IQA-PyTorch-Weights"
TOPIQ_FILENAME = "cfanet_nr_koniq_res50-9a73138b.pth"
TOPIQ_DIR = os.path.expanduser("~/.cache/torch/hub/pyiqa")
TOPIQ_PATH = os.path.join(TOPIQ_DIR, TOPIQ_FILENAME)
TIMM_REPOSITORY = "timm/resnet50.a1_in1k"
TIMM_FILENAME = "model.safetensors"
BERT_REPOSITORY = "bert-base-uncased"
BERT_REQUIRED_FILES = (
    "vocab.txt",
    "tokenizer_config.json",
    "tokenizer.json",
    "config.json",
)


def _require_regular_file(path, label):
    """Fail the preparation step when an asset is absent or empty."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    if os.path.getsize(path) <= 0:
        raise RuntimeError(f"{label} is empty: {path}")
    return path


def step(name, fn):
    print(f"\n=== {name} ===", flush=True)
    try:
        result = fn()
        if result is False:
            raise RuntimeError("step reported failure")
        print(f"OK: {name}", flush=True)
        return True
    except Exception:
        print(f"FAIL: {name}\n{traceback.format_exc()}", flush=True)
        return False


def clip_models():
    import clip
    for m in ["ViT-B/32", "ViT-L/14"]:
        print(f"downloading OpenAI CLIP {m} -> ~/.cache/clip", flush=True)
        clip.load(m, device="cpu")


def imagereward():
    from huggingface_hub import hf_hub_download
    from transformers import BertTokenizer

    os.makedirs(IMAGEREWARD_DIR, exist_ok=True)
    for filename in ("ImageReward.pt", "med_config.json"):
        path = hf_hub_download(
            "THUDM/ImageReward", filename, local_dir=IMAGEREWARD_DIR
        )
        _require_regular_file(path, f"ImageReward {filename}")
        print(f"ImageReward asset: {path}", flush=True)
    # Calling from_pretrained alone is not a complete closure: depending on
    # the transformers version it may leave tokenizer.json/config.json out of
    # the cache. Resolve every file consumed by the private scorer explicitly.
    for filename in BERT_REQUIRED_FILES:
        path = hf_hub_download(BERT_REPOSITORY, filename)
        _require_regular_file(path, f"BERT tokenizer {filename}")
        print(f"BERT tokenizer asset: {path}", flush=True)
    BertTokenizer.from_pretrained(BERT_REPOSITORY, local_files_only=True)
    print("validated bert-base-uncased tokenizer and model config", flush=True)


def iqa():
    """Pre-stage TOPIQ-NR and its explicit timm ResNet-50 backbone."""
    from huggingface_hub import hf_hub_download

    os.makedirs(TOPIQ_DIR, exist_ok=True)
    downloaded_topiq = hf_hub_download(
        TOPIQ_REPOSITORY,
        TOPIQ_FILENAME,
    )
    _require_regular_file(downloaded_topiq, "TOPIQ-NR checkpoint")
    # `asset_sources()` and pyiqa both use this exact torch-hub path. Avoid a
    # second copy when the downloader was configured with that directory.
    if os.path.realpath(downloaded_topiq) != os.path.realpath(TOPIQ_PATH):
        temporary = TOPIQ_PATH + ".tmp"
        shutil.copyfile(downloaded_topiq, temporary)
        os.replace(temporary, TOPIQ_PATH)
    _require_regular_file(TOPIQ_PATH, "TOPIQ-NR checkpoint")
    print(f"TOPIQ-NR checkpoint: {TOPIQ_PATH}", flush=True)

    backbone = hf_hub_download(TIMM_REPOSITORY, TIMM_FILENAME)
    _require_regular_file(backbone, "TOPIQ timm ResNet-50 backbone")
    print(f"TOPIQ timm backbone: {backbone}", flush=True)


def hps():
    from huggingface_hub import hf_hub_download
    cp = hf_hub_download("xswu/HPSv2", "HPS_v2.1_compressed.pt")
    print("HPS_v2.1 checkpoint:", cp, flush=True)
    print("downloading open_clip ViT-H-14 laion2B backbone (~3.9GB)...", flush=True)
    from hpsv2.src.open_clip import create_model_and_transforms
    create_model_and_transforms("ViT-H-14", pretrained="laion2B-s32B-b79K",
                                precision="fp32", device="cpu")


def aesthetic():
    os.makedirs(AESTHETIC_DIR, exist_ok=True)
    if os.path.exists(AESTHETIC_PATH):
        print("aesthetic MLP already present:", AESTHETIC_PATH, flush=True)
        return
    print("downloading LAION aesthetic MLP ->", AESTHETIC_PATH, flush=True)
    urllib.request.urlretrieve(AESTHETIC_URL, AESTHETIC_PATH)


def fix_hpsv2_vocab():
    """The hpsv2 pkg ships a vendored open_clip missing its BPE vocab gz; copy it
    from OpenAI CLIP so HPSScorer can be imported before its weights are staged."""
    import shutil
    import clip
    import hpsv2
    src = os.path.join(os.path.dirname(clip.__file__), "bpe_simple_vocab_16e6.txt.gz")
    if not os.path.exists(src):
        raise FileNotFoundError(f"OpenAI CLIP tokenizer vocabulary is missing: {src}")
    dst = os.path.join(os.path.dirname(hpsv2.__file__), "src", "open_clip",
                       "bpe_simple_vocab_16e6.txt.gz")
    if os.path.exists(dst):
        print("hpsv2 vocab already present", flush=True)
    else:
        shutil.copy(src, dst)
        print("copied vocab into hpsv2 ->", dst, flush=True)


def _steps():
    """Return preparation steps at call time so tests/operators can override one."""
    return (
        ("CLIP ViT-B/32 + ViT-L/14", clip_models),
        ("ImageReward checkpoint and tokenizer", imagereward),
        ("TOPIQ-NR checkpoint + timm ResNet-50 backbone", iqa),
        ("HPSv2 tokenizer vocab fix", fix_hpsv2_vocab),
        ("HPSv2 (checkpoint + ViT-H-14 backbone)", hps),
        ("LAION aesthetic MLP", aesthetic),
    )


def main():
    """Run every preparation step and fail closed if any asset is incomplete."""
    failures = [name for name, fn in _steps() if not step(name, fn)]
    if failures:
        print(
            "\nPRESTAGE FAILED ({} step{}): {}".format(
                len(failures),
                "" if len(failures) == 1 else "s",
                ", ".join(failures),
            ),
            flush=True,
        )
        return 1
    print("\nPRESTAGE COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
