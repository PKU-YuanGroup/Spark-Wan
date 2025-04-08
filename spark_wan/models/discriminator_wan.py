# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch._tensor import Tensor


from torch._tensor import Tensor


from torch._tensor import Tensor
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.distributed as dist
from spark_wan.models.transformer_wan import WanTransformer3DModel, WanAttnProcessor2_0
from spark_wan.parrallel.env import get_sequence_parallel_group
from spark_wan.parrallel.sp_modules import SplitAndScatter, Gather

from diffusers.configuration_utils import register_to_config
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.attention_processor import Attention

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SeaweedOutputHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str,
        eps: float,
        embed_seq_len: int = 1024,
    ):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(1, embed_seq_len, hidden_dim))
        self.output_cross_attn = Attention(
            query_dim=hidden_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=WanAttnProcessor2_0(),
        )
        self.output_mlp = nn.Sequential(
            nn.RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, hidden_states: torch.Tensor):
        batch_size = hidden_states.shape[0]
        repeated_embed: Tensor = self.embed.repeat(batch_size, 1, 1)

        sequence_parallel_group = get_sequence_parallel_group()

        if sequence_parallel_group:
            repeated_embed = SplitAndScatter.apply(
                sequence_parallel_group, repeated_embed, 1
            )

        cross_attention_output = self.output_cross_attn(
            hidden_states=repeated_embed, encoder_hidden_states=hidden_states
        )
        updated_hidden_states: Tensor = cross_attention_output + repeated_embed
        mlp_output: Tensor = self.output_mlp(updated_hidden_states)

        return mlp_output + updated_hidden_states


class WanDiscriminator(WanTransformer3DModel):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
    ]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        cnn_dropout: float = 0.0,
        head_type: str = "complex",
        seaweed_output_layer: List[int] = [-1],
        embed_seq_len: int = 1024,
        **kwargs,
    ) -> None:
        self.patch_size = patch_size
        super().__init__(
            patch_size,
            num_attention_heads,
            attention_head_dim,
            in_channels,
            out_channels,
            text_dim,
            freq_dim,
            ffn_dim,
            num_layers,
            cross_attn_norm,
            qk_norm,
            eps,
            image_dim,
            added_kv_proj_dim,
            rope_max_seq_len,
            is_partial_layer=True,
        )
        if head_type == "complex":
            self.dis_head = nn.Sequential(
                nn.Conv3d(
                    num_attention_heads * attention_head_dim,
                    512,
                    kernel_size=(3, 3, 3),
                    stride=1,
                    padding=1,
                ),
                nn.SiLU(True),
                nn.Dropout(cnn_dropout),
                nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=2, padding=1),
                nn.GroupNorm(32, 512),
                nn.SiLU(True),
                nn.Dropout(cnn_dropout),
                nn.Conv3d(512, 256, kernel_size=(3, 3, 3), stride=1, padding=1),
                nn.GroupNorm(32, 256),
                nn.SiLU(True),
                nn.Dropout(cnn_dropout),
                nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=2, padding=1),
                nn.GroupNorm(32, 256),
                nn.SiLU(True),
                nn.Dropout(cnn_dropout),
                nn.Conv3d(256, 1, kernel_size=(3, 3, 3), stride=1, padding=1),
            )
        elif head_type == "simple":
            self.dis_head = nn.Sequential(
                nn.Conv3d(
                    num_attention_heads * attention_head_dim,
                    128,
                    kernel_size=(4, 4, 4),
                    stride=2,
                    padding=1,
                ),
                nn.SiLU(True),
                nn.Dropout(cnn_dropout),
                nn.Conv3d(128, 1, kernel_size=(4, 4, 4), stride=2, padding=1),
            )
        elif head_type == "seaweed":
            hidden_dim = num_attention_heads * attention_head_dim
            self.seaweed_output_layer_idx = sorted(
                [
                    (self.num_layers + layer) if layer < 0 else layer
                    for layer in seaweed_output_layer
                ]
            )
            print("seaweed_output_layer_idx", self.seaweed_output_layer_idx)

            self.seaweed_output_layers = nn.ModuleList(
                [
                    SeaweedOutputHead(
                        hidden_dim=hidden_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        qk_norm=qk_norm,
                        eps=eps,
                    )
                    for _ in self.seaweed_output_layer_idx
                ]
            )
            self.disc_head = nn.Sequential(
                nn.RMSNorm(hidden_dim),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"Invalid head type: {head_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, latent_height, latent_width = (
            hidden_states.shape
        )
        assert (
            latent_height % self.patch_size[1] == 0
            and latent_width % self.patch_size[2] == 0
        )
        hidden_states_height = latent_height // self.patch_size[1]
        hidden_states_width = latent_width // self.patch_size[2]

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        sequence_parallel_group = get_sequence_parallel_group()
        if sequence_parallel_group:
            sequence_parallel_group_size = dist.get_world_size(sequence_parallel_group)

        if (
            sequence_parallel_group
            and hidden_states.size(1) % sequence_parallel_group_size != 0
        ):
            raise ValueError(
                "hidden_states.size(1) % sequence_parallel_group_size != 0"
            )
        if (
            sequence_parallel_group
            and encoder_hidden_states.size(1) % sequence_parallel_group_size != 0
        ):
            raise ValueError(
                "encoder_hidden_states.size(1) % sequence_parallel_group_size != 0"
            )
        if sequence_parallel_group:
            hidden_states = SplitAndScatter.apply(
                sequence_parallel_group, hidden_states, 1
            )  # b s d -> b s/p d

        if sequence_parallel_group:
            encoder_hidden_states = SplitAndScatter.apply(
                sequence_parallel_group, encoder_hidden_states, 1
            )  # b s d -> b s/p d

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep, encoder_hidden_states, encoder_hidden_states_image
            )
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        # 4. Transformer blocks
        hidden_states_list = []
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for idx, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )
                if idx in self.seaweed_output_layer_idx:
                    hidden_states_list.append(hidden_states)
        else:
            for idx, block in enumerate(self.blocks):
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
                if idx in self.seaweed_output_layer_idx:
                    hidden_states_list.append(hidden_states)

        if self.config.head_type == "seaweed":
            output_list = []
            
            for idx, hidden_states in enumerate(hidden_states_list):
                output_list.append(self.seaweed_output_layers[idx](hidden_states))

            output = torch.cat(output_list, dim=1)
            output = self.disc_head(output)
            if sequence_parallel_group:
                output = Gather.apply(sequence_parallel_group, output, 1)
            output = output.squeeze(-1)
        else:
            # GAN Distill Need
            if sequence_parallel_group:
                output = Gather.apply(sequence_parallel_group, output, 1)

            hidden_states = hidden_states.reshape(
                batch_size, num_frames, hidden_states_height, hidden_states_width, -1
            )  # torch.Size([1, 21, 60, 104, 384])
            hidden_states = hidden_states.permute(
                0, 4, 1, 2, 3
            )  # torch.Size([1, 384, 21, 60, 104])
            output = self.dis_head(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        return output


def convert_module_dtype(model, device, weight_dtype, norm_classes=(FP32LayerNorm,)):
    for module in model.modules():
        if isinstance(module, norm_classes):
            module.to(device, dtype=torch.float32)
        else:
            module.to(device, dtype=weight_dtype)
    return model
