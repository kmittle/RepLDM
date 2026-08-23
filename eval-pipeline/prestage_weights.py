"""Pre-stage all eval reward weights into SHARED user caches — DECOUPLED from Sana.

Run ONCE in env sana_cby with network access:
    /home/bycao/miniforge3/envs/sana_cby/bin/python eval-pipeline/prestage_weights.py

Downloads (~7GB total) go to shared caches (~/.cache/{clip,hpsv2,aesthetic} and the
HF hub cache), NOT into the Sana tree. Uses the same upstream pip packages already
installed in sana_cby (clip, open_clip, hpsv2) — this is not "importing Sana".
After this completes, scoring runs fully offline.
"""
import os
import traceback
import urllib.request

AESTHETIC_DIR = os.path.expanduser("~/.cache/aesthetic")
AESTHETIC_PATH = os.path.join(AESTHETIC_DIR, "sac+logos+ava1-l14-linearMSE.pth")
AESTHETIC_URL = ("https://github.com/christophschuhmann/improved-aesthetic-predictor"
                 "/raw/main/sac+logos+ava1-l14-linearMSE.pth")


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
    from the standard open_clip install so hpsv2's tokenizer works (else HPSScorer
    init fails with FileNotFoundError on bpe_simple_vocab_16e6.txt.gz)."""
    import glob
    import shutil
    import hpsv2
    import open_clip
    g = glob.glob(os.path.join(os.path.dirname(open_clip.__file__), "**",
                               "bpe_simple_vocab_16e6.txt.gz"), recursive=True)
    if not g:
        print("standard open_clip vocab not found; skip", flush=True); return
    dst = os.path.join(os.path.dirname(hpsv2.__file__), "src", "open_clip",
                       "bpe_simple_vocab_16e6.txt.gz")
    if os.path.exists(dst):
        print("hpsv2 vocab already present", flush=True)
    else:
        shutil.copy(g[0], dst)
        print("copied vocab into hpsv2 ->", dst, flush=True)


if __name__ == "__main__":
    step("CLIP ViT-B/32 + ViT-L/14", clip_models)
    step("HPSv2 (checkpoint + ViT-H-14 backbone)", hps)
    step("HPSv2 tokenizer vocab fix", fix_hpsv2_vocab)
    step("LAION aesthetic MLP", aesthetic)
    print("\nPRESTAGE COMPLETE", flush=True)
