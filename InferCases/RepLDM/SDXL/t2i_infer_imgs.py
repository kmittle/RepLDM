import os
from glob import glob
import time
import argparse
from ast import literal_eval
import random
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.multiprocessing as tmp
from torch.multiprocessing import Pool

from InferencePipelines import RepLDMSDXLPipeline


@ torch.no_grad()
def worker_process(config, save_dir, data, generator_seeds, device, task_queue):
    # define model
    torch.cuda.set_device(device)
    pipeline = RepLDMSDXLPipeline.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=config.cache_dir,
        local_files_only=True,
    ).to("cuda")
    # set inference args
    use_multi_decoder = config.resolution > 2048
    init_rates = [0.8] if config.resolution < 4096 else [0.9, 0.8]
    # if config.avoid_memory_fragment and config.resolution >= 4096:
        # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:16"
    # generation loop
    while True:
        try:
            idx = task_queue.get_nowait()
        except:
            break
        if idx is None:
            break
        seed = generator_seeds[idx].item()
        img_idx = data["index"][idx]
        prompt = data["TEXT"][idx]
        # sampling process
        torch.cuda.empty_cache()
        generator = torch.Generator("cuda").manual_seed(seed)
        images = pipeline(
            prompt, negative_prompt=config.negative_prompt, generator=generator,
            height=config.resolution, width=config.resolution,
            num_inference_steps=config.num_inference_steps, guidance_scale=config.guidance_scale,
            show_image=False,
            multi_decoder=use_multi_decoder, multi_encoder = True, models_to_cpu = True,
            num_resample_timesteps = 50,
            init_rates = init_rates,
            attn_type = 'vanilla',
            attn_guidance_scale = 0.004,
            attn_guidance_density = tuple([1]*47 + [0]*3),
            attn_guidance_decay = ('cosine', 0, 3),
            power_calibrate = 0,
            attn_guidance_filter = None,
        )
        save_path = os.path.join(save_dir, f"{img_idx}-{seed}.jpg")
        images[-1].save(save_path, "JPEG", quality=config.jpg_quality)
        del images
    
    print(f"process {device}: Generation task finished")


def get_config():
    def type_convert(x):
        if x.lower() in ["true", "false"]:
            x = True if x.lower() == "true" else False
        else:
            raise ValueError(f"Unknown type: {x}")
        return x
    # get args
    parser = argparse.ArgumentParser(description="running DemoFusion generation pipeline", add_help=False)
    
    parser.add_argument("--devices", type=str, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--framework", type=str, default="RepLDM")
    
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--negative_prompt", type=str, default="blurry, ugly, duplicate, poorly drawn, deformed, mosaic")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=int, default=7.5)
    
    parser.add_argument("--data_path", type=str, default="./download_laion_HR/laion_HR_inform_2000.csv", help="prompts, indices, and urls.")
    parser.add_argument("--jpg_quality", type=int, default=95)
    parser.add_argument("--avoid_memory_fragment", type=type_convert, default=True)
    parser.add_argument("--model_name", type=str, default= "stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--cache_dir", type=str, default="../huggingface_models")
    
    config, unparsed = parser.parse_known_args()
    return config


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = None
    
    # get args
    config = get_config()
    
    # build save dir
    save_dir = os.path.join("./output", config.framework, str(config.resolution), f"global_seed_{config.random_seed}")
    os.makedirs(save_dir, exist_ok=True)
    
    # get data & define model
    data = pd.read_csv(config.data_path)
    num_total_imgs = len(data)
    
    # check generated images
    generated_img_indices = []
    for img_format in ["jpg", "JPG","jpeg", "JPEG", "png", "PNG"]:
        generated_img_indices.extend(glob(os.path.join(save_dir, f"*.{img_format}")))
    num_generated_imgs = len(generated_img_indices)
    generated_img_indices = [int(os.path.basename(i).split("-")[0]) for i in generated_img_indices]
    num_generated_indices = len(set(generated_img_indices))
    assert num_generated_imgs == num_generated_indices
    
    # get devices
    devices = [f"cuda:{device}" for device in config.devices.split(",") if device != ""]
    assert devices != []
    num_tasks = len(devices)
    print(f"The number of processes: {num_tasks}")
    
    # get image indices to be generated
    todo_indices = [idx for idx in range(num_total_imgs) if int(data["index"][idx]) not in generated_img_indices]
    assert num_total_imgs == len(todo_indices) + num_generated_indices
    print(f"Total: {num_total_imgs}, Generated: {num_generated_indices}, To do: {len(todo_indices)}")
    
    # get random seeds
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    generator_seeds = torch.tensor([random.randint(0, int(1e8)) for _ in range(num_total_imgs)])
    
    # set task queue
    manager = tmp.Manager()
    task_queue = manager.Queue()
    for idx in todo_indices:
        task_queue.put(idx)
    
    # generation launch
    tmp.set_start_method('spawn', force=True)
    processes = []
    for device in devices:
        p = tmp.Process(target=worker_process, args=(config, save_dir, data, generator_seeds, device, task_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
