"""HPSv2 (v2.1) human-preference scorer. Decoupled (official `hpsv2` pkg), model
loaded ONCE in __init__ (Sana's package-style score() reloads the checkpoint every
call — a perf bug we avoid). Score = diagonal of image@text features (raw cosine —
do NOT divide by logit_scale). §13.5: ViT-H-14@224, correlated with the other
CLIP-family metrics; not an independent detail witness.
"""
import os

import torch

from .base import Scorer, register_metric


@register_metric("hps")
class HPSScorer(Scorer):
    OUTPUT_KEYS = (("hpsv2", "higher"),)

    def __init__(self, device="cuda", **p):
        super().__init__(device, **p)
        from huggingface_hub import hf_hub_download
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        self.model, _, self.preprocess = create_model_and_transforms(
            "ViT-H-14", pretrained="laion2B-s32B-b79K", precision="amp",
            device=device, output_dict=True)
        cp = hf_hub_download("xswu/HPSv2", "HPS_v2.1_compressed.pt")
        sd = torch.load(cp, map_location="cpu")
        self.model.load_state_dict(sd["state_dict"])
        self.tokenizer = get_tokenizer("ViT-H-14")
        self.model = self.model.to(device).eval()

    @classmethod
    def weights_status(cls, **p):
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download("xswu/HPSv2", "HPS_v2.1_compressed.pt", local_files_only=True)
            hf_hub_download(
                "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                "open_clip_pytorch_model.bin",
                local_files_only=True,
            )
            import hpsv2
            vocab = os.path.join(
                os.path.dirname(hpsv2.__file__), "src", "open_clip",
                "bpe_simple_vocab_16e6.txt.gz",
            )
            if not os.path.exists(vocab):
                return False, f"HPSv2 tokenizer vocabulary is missing ({vocab})"
            return True, ""
        except Exception as e:
            return False, f"HPS_v2.1 / ViT-H-14 not cached ({e})"

    @torch.no_grad()
    def score_image(self, image, prompt):
        img = self.preprocess(image).unsqueeze(0).to(self.device)
        txt = self.tokenizer([prompt]).to(self.device)
        with torch.cuda.amp.autocast():
            out = self.model(img, txt)
            score = (out["image_features"] @ out["text_features"].T).diagonal()
        return {"hpsv2": float(score.item())}
