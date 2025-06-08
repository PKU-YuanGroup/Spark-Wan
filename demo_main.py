import os
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from diffusers import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from diffusers import WanPipeline
from diffusers.image_processor import VaeImageProcessor
from peft import LoraConfig, get_peft_model
import fcntl
from safetensors.torch import load_file

from diffusers.training_utils import free_memory

import sys
sys.path.append(".")
from spark_wan.models.transformer_wan import WanTransformer3DModel
from spark_wan.training_utils.load_model import replace_rmsnorm_with_fp32
from spark_wan.parrallel.env import init_sequence_parallel_group
from taehv.taehv import TAEHV

config_path = "config_temp.txt"
file_path = "file_temp.txt"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class TAEW2_1DiffusersWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float16
        self.device = "cuda"
        self.taehv = TAEHV("taehv/taew2_1.pth").to(self.dtype)
        self.temperal_downsample = [True, True, False] # [sic]
        self.config = DotDict(scaling_factor=1.0, latents_mean=torch.zeros(16), z_dim=16, latents_std=torch.ones(16))

    def decode(self, latents, return_dict=None):
        n, c, t, h, w = latents.shape
        # low-memory, set parallel=True for faster + higher memory
        return (self.taehv.decode_video(latents.transpose(1, 2), parallel=True).transpose(1, 2).mul_(2).sub_(1),)


def read_params_from_txt(file_path):
    params = {}
    try:
        with open(file_path, "r") as file:
            for line in file:
                key, value = line.strip().split("=")
                params[key] = value
    except Exception:
        pass
    return params


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
        default="/mnt/data/checkpoints/Wan-AI/Wan2.1-T2V-14B-Diffusers",
    )
    parser.add_argument("--transformer_subfolder", type=str, default="distill_4")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--weight_dtype", type=str, default="bf16")
    parser.add_argument("--seed", type=int, default=2002)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument("--prompt_file", type=str, default="scripts/sora.txt")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--flow_shift", type=float, default=7.0)
    parser.add_argument("--sp_size", type=int, default=8)
    parser.add_argument("--sampling_steps", type=int, default=4)
    return parser.parse_args()


def infer(args):
    weight_dtype = torch.bfloat16
    output_dir = args.output_dir

    # Define weight dtype
    if args.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif args.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        raise ValueError("weight_dtype must be fp16 or bf16")

    # Get prompts
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    # Make output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    vae = AutoencoderKLWan.from_pretrained(
        args.model_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
    )
    transformer = WanTransformer3DModel.from_pretrained(
        args.model_path,
        subfolder=args.transformer_subfolder,
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
            "time_embedder.linear_1",
            "time_embedder.linear_2",
            "time_proj",
            "patch_embedding",
        ]
        lora_config = LoraConfig(
            r=256,
            lora_alpha=512,
            target_modules=lora_target_modules,
        )
        transformer = get_peft_model(transformer, lora_config)

        state_dict = load_file(args.lora_path, device="cpu")

        _, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)
        assert len(unexpected_keys) == 0

    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=args.flow_shift,
    )

    pipe = WanPipeline.from_pretrained(
        args.model_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=weight_dtype,
    )

    pipe.vae = TAEW2_1DiffusersWrapper().to("cuda")

    pipe = pipe.to(device="cuda")

    while True:
        print("Waiting Input")
        while True:
            params = read_params_from_txt(config_path)
            if params != {}:
                print(params)
                break

        if int(params["height"]) == 960 and int(params["width"]) == 960:
            params["height"] = 720
            params["width"] = 1280
        args.prompt = (
            params["prompt"]
            if isinstance(params["prompt"], list)
            else [params["prompt"]]
        )
        args.height = int(params["height"])
        args.width = int(params["width"])
        args.sampling_steps = int(params["num_inference_steps"])
        args.num_frames = int(params["video_length"])
        args.seed = int(params["seed"])

        with torch.amp.autocast("cuda", dtype=weight_dtype):
            pt_images = pipe(
                prompt=args.prompt,
                negative_prompt=negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.sampling_steps,
                num_frames=args.num_frames,
                guidance_scale=args.cfg,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
                output_type="pt",
            ).frames[0]
        pt_images = torch.stack([pt_images[i] for i in range(pt_images.shape[0])])
        image_np = VaeImageProcessor.pt_to_numpy(pt_images)
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)
        if dist.get_rank() == 0:
            file_count = len(
                [
                    f
                    for f in os.listdir(output_dir)
                    if os.path.isfile(os.path.join(output_dir, f))
                ]
            )
            video_path = f"{output_dir}/{file_count:04d}.mp4"
            export_to_video(image_pil, video_path, fps=16)

            with open(file_path, "w") as file:
                fcntl.flock(file, fcntl.LOCK_EX)
                file.write(f"video_path={video_path}\n")
                fcntl.flock(file, fcntl.LOCK_UN)

        del pt_images
        with open(config_path, "w") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            file.write("")
            fcntl.flock(file, fcntl.LOCK_UN)
        free_memory()

    # idx = 0
    # for prompt in tqdm(prompts):
    #     video_path = f"{args.output_dir}/output_{idx}.mp4"
    #     if os.path.exists(video_path):
    #         idx += 1
    #         continue
    #     generator = torch.Generator(device="cuda").manual_seed(seed)
    #     with torch.amp.autocast("cuda", dtype=weight_dtype):
    #         pt_images = pipe(
    #             prompt=prompt,
    #             negative_prompt=negative_prompt,
    #             height=args.height,
    #             width=args.width,
    #             num_inference_steps=args.sampling_steps,
    #             num_frames=args.num_frames,
    #             guidance_scale=args.cfg,
    #             generator=generator,
    #             output_type="pt",
    #         ).frames[0]
    #     pt_images = torch.stack([pt_images[i] for i in range(pt_images.shape[0])])
    #     image_np = VaeImageProcessor.pt_to_numpy(pt_images)
    #     image_pil = VaeImageProcessor.numpy_to_pil(image_np)
    #     if dist.get_rank() == 0:
    #         export_to_video(image_pil, video_path, fps=16)
    #     idx += 1

    # End inference
    # dist.destroy_process_group()


if __name__ == "__main__":
    while True:
        try:
            args = parse_args()
            init_env(args.sp_size)
            infer(args)
        except:
            free_memory()
            continue
