from typing import List, Optional, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from spark_wan.models.autoencoder_wan import AutoencoderKLWan
from spark_wan.models.transformer_wan import WanTransformer3DModel, WanTransformerBlock
from spark_wan.modules.fp32_norm import FP32RMSNorm
from spark_wan.training_utils.fsdp2_utils import prepare_fsdp_model
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, UMT5EncoderModel
from transformers.models.umt5.modeling_umt5 import UMT5Block

from diffusers.models.normalization import RMSNorm


def replace_rmsnorm_with_fp32(model):
    for name, module in model.named_modules():
        if isinstance(module, RMSNorm):

            def new_forward(self, x):
                return FP32RMSNorm.forward(self, x)

            module.forward = new_forward.__get__(module, module.__class__)
    return model


def load_model(
    pretrained_model_name_or_path: str,
    is_train_lora: bool,
    fsdp_transformer: bool,
    fsdp_text_encoder: bool,
    weight_dtype: torch.dtype,
    device: torch.device,
    gradient_checkpointing: bool = True,
    compile_transformer: bool = False,
    lora_rank: Optional[int] = None,
    lora_alpha: Optional[float] = None,
    lora_dropout: Optional[float] = None,
    lora_target_modules: Optional[List[str]] = None,
    pretrained_lora_path: Optional[str] = None,
    find_unused_parameters: bool = False,
    reshard_after_forward: bool = False,  # Zero3
    transformer_subfolder: str = "transformer",
) -> Tuple[AutoTokenizer, UMT5EncoderModel, WanTransformer3DModel, AutoencoderKLWan]:

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer"
    )
    # Load text encoder
    text_encoder = UMT5EncoderModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )

    # Load transformer
    transformer = WanTransformer3DModel.from_pretrained(
        pretrained_model_name_or_path, subfolder=transformer_subfolder
    )
    transformer = replace_rmsnorm_with_fp32(transformer)

    # Load vae
    vae = AutoencoderKLWan.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae"
    )

    # Setup models
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device, dtype=torch.float32, non_blocking=True)

    if gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if is_train_lora:
        transformer.requires_grad_(False)
        transformer_lora_config = LoraConfig(
            r=lora_rank,
            target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=True,
        )
        if pretrained_lora_path is None:
            transformer = get_peft_model(transformer, transformer_lora_config)
        else:
            transformer = PeftModel.from_pretrained(
                transformer, pretrained_lora_path, is_trainable=True
            )

    # Compile transformer
    if compile_transformer:
        transformer = torch.compile(transformer)

    # FSDP
    if fsdp_transformer:
        prepare_fsdp_model(
            transformer,
            shard_conditions=[lambda n, m: isinstance(m, WanTransformerBlock)],
            cpu_offload=False,
            reshard_after_forward=reshard_after_forward,
            weight_dtype=weight_dtype,
        )
    else:
        transformer = transformer.to(device)
        transformer = DistributedDataParallel(
            transformer,
            device_ids=[device],
            find_unused_parameters=find_unused_parameters,
        )

    if fsdp_text_encoder:
        prepare_fsdp_model(
            text_encoder,
            shard_conditions=[lambda n, m: isinstance(m, (UMT5Block,))],
            cpu_offload=False,
            weight_dtype=weight_dtype,
        )
    else:
        text_encoder.to(device, non_blocking=True)

    return tokenizer, text_encoder, transformer, vae
