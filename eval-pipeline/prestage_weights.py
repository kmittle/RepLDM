"""Pre-stage all eval reward weights into SHARED user caches — DECOUPLED from Sana.

Run ONCE in the scoring environment with network access:
    /home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/prestage_weights.py

Downloads (~7GB total) go to shared caches (~/.cache/{clip,hpsv2,aesthetic} and the
HF hub cache), NOT into another project tree. It uses upstream `clip` and `hpsv2`
packages and repairs the missing vocabulary in the vendored HPSv2 tokenizer.
After this completes, scoring runs fully offline.
"""
import os
import traceback
import urllib.request

AESTHETIC_DIR = os.path.expanduser("~/.cache/aesthetic")
AESTHETIC_PATH = os.path.join(AESTHETIC_DIR, "sac+logos+ava1-l14-linearMSE.pth")
AESTHETIC_URL = ("https://github.com/christophschuhmann/improved-aesthetic-predictor"
                 "/raw/main/sac+logos+ava1-l14-linearMSE.pth")
IMAGEREWARD_DIR = os.path.expanduser("~/.cache/ImageReward")


def step(name, fn):
    print(f"\n=== {name} ===", flush=True)
    try:
        fn()
        print(f"OK: {name}", flush=True)
    except Exception:
        print(f"FAIL: {name}\n{traceback.format_exc()}", flush=True)


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
        print(f"ImageReward asset: {path}", flush=True)
    BertTokenizer.from_pretrained("bert-base-uncased")
    print("cached bert-base-uncased tokenizer", flush=True)


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


if __name__ == "__main__":
    step("CLIP ViT-B/32 + ViT-L/14", clip_models)
    step("ImageReward checkpoint and tokenizer", imagereward)
    step("HPSv2 tokenizer vocab fix", fix_hpsv2_vocab)
    step("HPSv2 (checkpoint + ViT-H-14 backbone)", hps)
    step("LAION aesthetic MLP", aesthetic)
    print("\nPRESTAGE COMPLETE", flush=True)
