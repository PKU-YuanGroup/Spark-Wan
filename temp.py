import sys
import argparse

sys.path.append(".")
import os

import time
import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from diffusers import WanPipeline
from diffusers.image_processor import VaeImageProcessor
from spark_wan.models.transformer_wan import WanTransformer3DModel
from spark_wan.training_utils.load_model import replace_rmsnorm_with_fp32
from spark_wan.parrallel.env import init_sequence_parallel_group
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from safetensors.torch import load_file

from diffusers import TorchAoConfig

model_path = "/mnt/workspace/ysh/Code/Wan-Distill/0_weight/Wan2.1-T2V-14B-Diffusers"
weight_dtype = torch.bfloat16

quantization_config = TorchAoConfig("int8wo")
transformer = WanTransformer3DModel.from_pretrained(
    model_path,
    subfolder="distill_8",
    # quantization_config=quantization_config,
    torch_dtype=weight_dtype,
)
transformer = replace_rmsnorm_with_fp32(transformer)
transformer.eval()
transformer.to("cuda")
print(f"Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")

# vae = AutoencoderKLWan.from_pretrained(
#         model_path,
#         subfolder="vae",
#         torch_dtype=weight_dtype,
#     )

# scheduler = UniPCMultistepScheduler(
#     prediction_type="flow_prediction",
#     use_flow_sigmas=True,
#     num_train_timesteps=1000,
#     flow_shift=7.0,
# )

# pipe = WanPipeline.from_pretrained(
#     model_path,
#     transformer=transformer,
#     vae=vae,
#     scheduler=scheduler,
#     torch_dtype=weight_dtype,
# )

# pipe = pipe.to(device="cuda")