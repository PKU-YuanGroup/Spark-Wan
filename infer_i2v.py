import sys
import argparse

sys.path.append(".")
from typing import Literal, Optional
import os
import numpy as np
import json
import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan
from diffusers import WanImageToVideoPipeline
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from diffusers.image_processor import VaeImageProcessor
from spark_wan.models.transformer_wan import WanTransformer3DModel
from spark_wan.training_utils.load_model import replace_rmsnorm_with_fp32
from spark_wan.parrallel.env import init_sequence_parallel_group
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from safetensors.torch import load_file
from transformers import CLIPVisionModel


def init_env(sp_size: int = 8):
    # init env
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    init_sequence_parallel_group(sp_size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/pfs/checkpoints/Wan2.1-I2V-14B-720P-Diffusers/",
    )
    parser.add_argument("--lora_path", type=Optional[str], default=None)
    parser.add_argument("--weight_dtype", type=Literal["fp16", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--cfg", type=int, default=5.0)
    parser.add_argument("--prompt_file", type=str, default="scripts/prompt_i2v.json")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--flow_shift", type=float, default=5.0)
    parser.add_argument("--sp_size", type=int, default=8)
    parser.add_argument("--sampling_steps", type=int, default=32)


def infer(args):
    weight_dtype = torch.bfloat16
    seed = args.seed

    # Define weight dtype
    if args.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif args.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        raise ValueError("weight_dtype must be fp16 or bf16")
    # Make output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    image_encoder = CLIPVisionModel.from_pretrained(
        args.model_path, subfolder="image_encoder", torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(
        args.model_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
    )
    transformer = WanTransformer3DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )
    transformer = replace_rmsnorm_with_fp32(transformer)
    transformer.eval()

    # Make adaptor
    if args.lora_path:
        lora_target_modules = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_out",
            "ffn.net.0.proj",
            "ffn.net.2",
        ]
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=lora_target_modules,
        )
        transformer = get_peft_model(transformer, lora_config)

        state_dict = load_file(args.lora_path, device_map="cpu")

        _, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)
        assert len(unexpected_keys) == 0

    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=args.flow_shift,
    )

    pipe = WanImageToVideoPipeline.from_pretrained(
        args.model_path,
        image_encoder=image_encoder,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=weight_dtype,
    )

    pipe = pipe.to(device="cuda")

    # Get prompts
    max_area = args.height * args.width
    with open(args.prompt_file, "r") as file:
        prompts = json.load(file)
        for prompt in prompts:
            image = load_image(prompt["image"])
            aspect_ratio = image.height / image.width
            mod_value = (
                pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
            )
            height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
            image = image.resize((width, height))
            prompt["image"] = load_image(prompt["image"])

    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    idx = 0
    for prompt in tqdm(prompts):
        text_prompt = prompt["prompt"]
        image_prompt = prompt["image"]

        generator = torch.Generator(device="cuda").manual_seed(seed)
        pt_images = pipe(
            prompt=text_prompt,
            image=image_prompt,
            negative_prompt=negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.sampling_steps,
            num_frames=args.num_frames,
            guidance_scale=args.cfg,
            generator=generator,
            output_type="pt",
        ).frames[0]
        pt_images = torch.stack([pt_images[i] for i in range(pt_images.shape[0])])
        image_np = VaeImageProcessor.pt_to_numpy(pt_images)
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)
        if dist.get_rank() == 0:
            video_path = f"{args.output_dir}/output_{idx}.mp4"
            export_to_video(image_pil, video_path, fps=16)
        idx += 1

    # End inference
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    init_env(args.sp_size)
    infer(args)
