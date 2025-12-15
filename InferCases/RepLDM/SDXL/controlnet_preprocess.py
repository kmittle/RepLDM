import numpy as np
import cv2
from PIL import Image

import torch
from diffusers import ControlNetModel
from diffusers.utils import load_image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation


def preprocess_condition(cache_dir, device, condition_img_path, condition_img_type="canny", height=None, width=None):
    condition_img_type = condition_img_type.lower()
    condition_img = load_image(condition_img_path)
    if condition_img_type == "canny":
        # load controlnet
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True,
            cache_dir=cache_dir,
        ).to(device)
        # preprocess the condition image
        condition_img = np.array(condition_img)
        condition_img = cv2.Canny(condition_img, 100, 200)
        condition_img = condition_img[:, :, None]
        condition_img = np.concatenate([condition_img] * 3, axis=2)
        condition_img = Image.fromarray(condition_img)
        condition_img.save(condition_img_path.split(".")[0] + "-canny.png")
    elif condition_img_type == "depth":
        # load controlnet
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True,
            cache_dir=cache_dir,
        ).to(device)
        # preprocess the condition image
        depth_estimator = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-hybrid-midas",
            local_files_only=True,
            cache_dir=cache_dir,
        ).to(device)
        feature_extractor = DPTFeatureExtractor.from_pretrained(
            "Intel/dpt-hybrid-midas",
            local_files_only=True,
            cache_dir=cache_dir,
        )
        assert (height is not None) and (width is not None)
        condition_img = get_depth_map(condition_img, feature_extractor, depth_estimator, device, height, width)
        condition_img.save(condition_img_path.split(".")[0] + "-depth.png")
    else:
        raise NotImplementedError
    
    return condition_img, controlnet


def get_depth_map(image, feature_extractor, depth_estimator, device, height, width):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth
    
    aspect_ratio = height / width
    w_resized = int(round(1024 / aspect_ratio ** 0.5, 0))
    h_resized = int(round(1024 * aspect_ratio ** 0.5, 0))
    
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(h_resized, w_resized),
        mode="bicubic",
        align_corners=False,
    )
    
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)
    
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image
