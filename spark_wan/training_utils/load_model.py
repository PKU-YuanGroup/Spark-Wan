from typing import List, Optional, Tuple, Union

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from spark_wan.models.autoencoder_wan import AutoencoderKLWan
from spark_wan.training_utils.train_config import ModelConfig, APTDistillConfig, StepDistillConfig, TrainingConfig
from spark_wan.models.transformer_wan import WanTransformer3DModel, WanTransformerBlock
from spark_wan.models.discriminator_wan import WanDiscriminator
from spark_wan.modules.fp32_norm import FP32RMSNorm
from spark_wan.training_utils.fsdp2_utils import prepare_fsdp_model
from spark_wan.utils.train_debug import output_grad_info
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
    model_config: ModelConfig,
    training_config: TrainingConfig,
    distill_config: Union[StepDistillConfig, APTDistillConfig],
    device: torch.device,
    weight_dtype: torch.dtype,
    lora_target_modules: List[str] = [],
    find_unused_parameters: bool = False,
) -> Tuple[AutoTokenizer, UMT5EncoderModel, WanTransformer3DModel, AutoencoderKLWan, WanDiscriminator]:

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    # Load text encoder
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_config.pretrained_model_name_or_path, subfolder="text_encoder"
    )

    # Load transformer
    transformer = WanTransformer3DModel.from_pretrained(
        model_config.pretrained_model_name_or_path, subfolder=model_config.transformer_subfolder
    )
    transformer = replace_rmsnorm_with_fp32(transformer)

    # Load vae
    vae = AutoencoderKLWan.from_pretrained(
        model_config.pretrained_model_name_or_path, subfolder="vae"
    )
    
    # Load discriminator
    enable_discriminator = True
    if hasattr(distill_config, "is_gan_distill"):
        enable_discriminator = distill_config.is_gan_distill
    
    discriminator = None
    if enable_discriminator:
        dic_model_config = transformer.config
        dic_model_config["num_layers"] = distill_config.discriminator_copy_num_layers
        dic_model_config["cnn_dropout"] = distill_config.discriminator_dropout
        dic_model_config["head_type"] = distill_config.discriminator_head_type
        dic_model_config["seaweed_output_layer"] = distill_config.discriminator_seaweed_output_layer
        
        discriminator = WanDiscriminator(
            **dic_model_config,
        )
        pretrained_checkpoint = transformer.state_dict()
        missing_keys, unexpected_keys = discriminator.load_state_dict(
            pretrained_checkpoint, strict=False
        )
        assert len(unexpected_keys) == 3, f"Unexpected keys: {unexpected_keys}" # ['scale_shift_table', 'proj_out.weight', 'proj_out.bias']
        
        discriminator = replace_rmsnorm_with_fp32(discriminator)
        if training_config.disc_gradient_checkpointing:
            discriminator.enable_gradient_checkpointing()

        if model_config.is_train_disc_lora:
            discriminator.requires_grad_(False)
            discriminator.disc_head.requires_grad_(True)
            if hasattr(discriminator, "seaweed_output_layers"):
                for layer in discriminator.seaweed_output_layers:
                    layer.embed.requires_grad_(True)
                    layer.requires_grad_(True)
            lora_config = LoraConfig(
                r=model_config.disc_lora_rank,
                target_modules=lora_target_modules,
                lora_alpha=model_config.disc_lora_alpha,
                lora_dropout=model_config.disc_lora_dropout,
                init_lora_weights=True,
            )
            discriminator = get_peft_model(discriminator, lora_config)
    
    # Setup models
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device, dtype=torch.float32, non_blocking=True)

    if training_config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if model_config.is_train_lora:
        transformer.requires_grad_(False)
        transformer_lora_config = LoraConfig(
            r=model_config.lora_rank,
            target_modules=lora_target_modules,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            init_lora_weights=True,
        )
        if model_config.pretrained_lora_path is None:
            print(transformer, transformer_lora_config)
            transformer = get_peft_model(transformer, transformer_lora_config)
        else:
            transformer = PeftModel.from_pretrained(
                transformer, model_config.pretrained_lora_path, is_trainable=True
            )
    else:
        transformer.requires_grad_(True)
    
    # For debug
    output_grad_info(transformer, "transformer_grad_info.txt")
    if discriminator:
        output_grad_info(discriminator, "discriminator_grad_info.txt")
    
    # Compile transformer
    if model_config.compile_transformer:
        transformer = torch.compile(transformer)

    # FSDP
    if model_config.fsdp_transformer:
        prepare_fsdp_model(
            transformer,
            shard_conditions=[lambda n, m: isinstance(m, WanTransformerBlock)],
            cpu_offload="transformer" in model_config.cpu_offload,
            reshard_after_forward="transformer" in model_config.reshard_after_forward,
            weight_dtype=weight_dtype,
        )
    else:
        transformer = transformer.to(device)
        transformer = DistributedDataParallel(
            transformer,
            device_ids=[device],
            find_unused_parameters=find_unused_parameters,
        )

    if discriminator:
        if model_config.fsdp_discriminator:
            prepare_fsdp_model(
                discriminator,
                shard_conditions=[lambda n, m: isinstance(m, WanTransformerBlock)],
                cpu_offload="discriminator" in model_config.cpu_offload,
                reshard_after_forward="discriminator" in model_config.reshard_after_forward,  # Discriminator need to reshard after forward.
                weight_dtype=weight_dtype,
            )
        else:
            discriminator = discriminator.to(device)
            discriminator = DistributedDataParallel(discriminator, device_ids=[device])
    
    if model_config.fsdp_text_encoder:
        prepare_fsdp_model(
            text_encoder,
            shard_conditions=[lambda n, m: isinstance(m, (UMT5Block,))],
            cpu_offload="text_encoder" in model_config.cpu_offload,
            reshard_after_forward="text_encoder" in model_config.reshard_after_forward,
            weight_dtype=weight_dtype,
        )
    else:
        text_encoder.to(device, non_blocking=True)

    return tokenizer, text_encoder, transformer, vae, discriminator
