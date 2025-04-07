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

import copy
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from spark_wan.parrallel.env import get_sequence_parallel_group
from spark_wan.parrallel.sp_modules import Gather, SplitAndAllToAll, SplitAndScatter

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.training_utils import free_memory
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)  # b, s, h*d
        key = attn.to_k(encoder_hidden_states)  # b, s, h*d
        value = attn.to_v(encoder_hidden_states)  # b, s, h*d

        # Pad head
        sequence_parallel_group = get_sequence_parallel_group()
        if sequence_parallel_group:
            sequence_parallel_group_size = dist.get_world_size(sequence_parallel_group)
            if attn.heads % sequence_parallel_group_size != 0:
                raise ValueError(
                    f"Heads {attn.heads} must be divisible by the number of GPUs in sequence parallel group {sequence_parallel_group_size}"
                )

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if sequence_parallel_group:
            query = SplitAndAllToAll.apply(sequence_parallel_group, query, 2, 1)
            key = SplitAndAllToAll.apply(sequence_parallel_group, key, 2, 1)
            value = SplitAndAllToAll.apply(sequence_parallel_group, value, 2, 1)
            query = torch.unflatten(
                query, 2, (attn.heads // sequence_parallel_group_size, -1)
            ).transpose(1, 2)
            key = torch.unflatten(
                key, 2, (attn.heads // sequence_parallel_group_size, -1)
            ).transpose(1, 2)
            value = torch.unflatten(
                value, 2, (attn.heads // sequence_parallel_group_size, -1)
            ).transpose(1, 2)
        else:
            query = torch.unflatten(query, 2, (attn.heads, -1)).transpose(1, 2)
            key = torch.unflatten(key, 2, (attn.heads, -1)).transpose(1, 2)
            value = torch.unflatten(value, 2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(
                    hidden_states.to(torch.float64).unflatten(3, (-1, 2))
                )
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        if sequence_parallel_group:
            hidden_states = SplitAndAllToAll.apply(
                sequence_parallel_group, hidden_states, 1, 2
            )

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim
        )
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh"
        )

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        timestep = self.timesteps_proj(timestep)

        # time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        time_embedder_dtype = torch.int8
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
            timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image
            )

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


@torch.amp.autocast("cuda", enabled=False)
class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=False,
                repeat_interleave_real=False,
                freqs_dtype=torch.float64,
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        self.freqs = self.freqs.to(hidden_states.device)
        freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(
            1, 1, ppf * pph * ppw, -1
        )
        return freqs


class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        attention_cls = Attention

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = attention_cls(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=WanAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = attention_cls(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=WanAttnProcessor2_0(),
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.float32):
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        attn_output = self.attn1(
            hidden_states=norm_hidden_states, rotary_emb=rotary_emb
        )
        with torch.amp.autocast("cuda", dtype=torch.float32):
            hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(
                hidden_states
            )

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            hidden_states = (
                hidden_states.float() + ff_output.float() * c_gate_msa
            ).type_as(hidden_states)

        return hidden_states


class WanTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin
):
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
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

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
        is_partial_layer: bool = False,
        partial_layer_idx: Tuple[int, ...] = None,
        reserve_layers: int = 0,
        self_distill_layers_idx: Tuple[int, ...] = None,
    ) -> None:
        super().__init__()

        self.is_partial_layer = is_partial_layer
        self.self_distill_layers_idx = self_distill_layers_idx

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels
        self.inner_dim = inner_dim

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        if not self.is_partial_layer:
            self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
            self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
            self.scale_shift_table = nn.Parameter(
                torch.randn(1, 2, inner_dim) / inner_dim**0.5
            )

        self.gradient_checkpointing = False

        self.is_partial_layer = is_partial_layer
        self.partial_layer_idx = partial_layer_idx
        self.reserve_layers = reserve_layers

    def load_partial_layers(
        self,
        base_layer_idx: Tuple[int, ...],
        mini_layer_idx: Tuple[int, ...],
        lora_path: str = None,
        is_delete_layer: bool = True,
    ) -> None:
        logger.info(f"Loading partial layers: {mini_layer_idx} from {lora_path}")

        # Create a mini_transformer and load the weights
        if lora_path:
            mini_transformer = copy.deepcopy(self)
            mini_transformer.set_partial_layers(mini_layer_idx)
            mini_transformer = PeftModel.from_pretrained(mini_transformer, lora_path)
            mini_transformer = mini_transformer.merge_and_unload()
        else:
            # (250309 YSH) need to be fixed
            pass

        # Get mini_transformer blocks and update the full model with corresponding layers
        mini_blocks = mini_transformer.blocks
        mini_layer_idx = sorted(mini_layer_idx)

        # Replace the full model's blocks with the mini_transformer's corresponding blocks
        for i, block_idx in enumerate(mini_layer_idx):
            self.blocks[block_idx] = copy.deepcopy(mini_blocks[i])

        # Delete unnecessary middle layers
        if is_delete_layer:
            delete_idx = set(base_layer_idx) - set(mini_layer_idx)
            self.blocks = nn.ModuleList(
                [
                    block
                    for idx, block in enumerate(self.blocks)
                    if idx not in delete_idx
                ]
            )

        # Clean up resources
        del mini_blocks
        del mini_transformer
        free_memory()

    def set_partial_layers(
        self,
        partial_layer_idx: Tuple[int, ...],
        is_train: bool = True,
        reserve_layers: int = 0,
    ) -> None:
        # Register partial layers
        logger.info(f"Setting partial layers: {self.partial_layer_idx}")
        self.is_partial_layer = is_train
        self.partial_layer_idx = tuple(sorted(partial_layer_idx))
        self.reserve_layers = reserve_layers

        if reserve_layers > 0:
            last_idx = self.partial_layer_idx[-1]
            new_indices = []

            for i in range(1, reserve_layers + 1):
                new_idx = last_idx + i
                if new_idx <= self.config.num_layers - 1:
                    new_indices.append(new_idx)

            self.partial_layer_idx = self.partial_layer_idx + tuple(new_indices)

            for idx in new_indices:
                for param in self.blocks[idx].parameters():
                    param.requires_grad = False

        logger.info("Registering partial layers in config.")
        self.register_to_config(is_partial_layer=self.is_partial_layer)
        self.register_to_config(partial_layer_idx=self.partial_layer_idx)
        self.register_to_config(reserve_layers=self.reserve_layers)

        # Store mapping from new block indices to original indices
        # self.layer_mapping = {new_idx: orig_idx for new_idx, orig_idx in enumerate(self.partial_layer_idx)}
        self.layer_mapping = {
            orig_idx: new_idx for new_idx, orig_idx in enumerate(self.partial_layer_idx)
        }

        # Get mini_transformer blocks
        self.blocks = nn.ModuleList([self.blocks[i] for i in self.partial_layer_idx])

        # Clean up resources
        if is_train:
            self.norm_out = None
            self.proj_out = None
            self.scale_shift_table = None
            del self.norm_out
            del self.proj_out
            del self.scale_shift_table
        free_memory()

    def set_self_distill_layers(
        self,
        self_distill_layers_idx: Tuple[int, ...],
    ) -> None:
        # Register partial layers
        logger.info(f"Setting self distill layers: {self.self_distill_layers_idx}")
        self.self_distill_layers_idx = tuple(sorted(self_distill_layers_idx))
        self.register_to_config(self_distill_layers_idx=self.self_distill_layers_idx)

        self.alphas = nn.ParameterList(
            [
                nn.Parameter(
                    torch.tensor([1.0], dtype=self.dtype, device=self.device),
                    requires_grad=False,
                )
                for _ in range(len(self.self_distill_layers_idx))
            ]
        )

    def step_alpha_decay(
        self, step: int, zero_step: int = 1000, scheduler: str = "linear", **kwargs
    ):
        step = torch.tensor(step, dtype=self.dtype, device=self.device)
        zero_step = torch.tensor(zero_step, dtype=self.dtype, device=self.device)

        if scheduler == "linear":
            if step < zero_step:
                for alpha in self.alphas:
                    alpha.copy_((zero_step - step) / zero_step)
            else:
                for alpha in self.alphas:
                    alpha.copy_(0.0)
        elif scheduler == "step":
            num_steps = kwargs["num_steps"]
            w = zero_step / num_steps
            for alpha in self.alphas:
                alpha.copy_(1 - torch.floor(step / w) * (1 / num_steps))

    # @torch.compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        output_hidden_states: bool = False,
        output_hidden_states_idx: Optional[List[int]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        assert not self.is_partial_layer, "Partial layer is not supported. You need to implement a new forward function."
        
        
        if output_hidden_states_idx is None:
            output_hidden_states_idx = []

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

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

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
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            # For Layer distill
            hidden_states_list = []
            alpha_idx = 0

            # Main Loop
            for i, block in enumerate(self.blocks):
                # Self distill: create a path for the hidden_states.
                if i in (self.self_distill_layers_idx or []):
                    degrading_hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        timestep_proj,
                        rotary_emb,
                    )
                    hidden_states = degrading_hidden_states * self.alphas[
                        alpha_idx
                    ] + hidden_states * (1 - self.alphas[alpha_idx])
                    alpha_idx += 1
                else:
                    hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        timestep_proj,
                        rotary_emb,
                    )

                # For Layer distill
                if output_hidden_states and i in output_hidden_states_idx:
                    hidden_states_list.append(hidden_states)
        else:
            hidden_states_list = []

            # Main Loop
            for i, block in enumerate(self.blocks):
                # When some block is float32, we need to convert the hidden_states and encoder_hidden_states to float32
                # and then convert back to the original dtype.
                if block.scale_shift_table.data.dtype == torch.float32:
                    origin_dtype = hidden_states.dtype
                    hidden_states = hidden_states.to(torch.float32)
                    encoder_hidden_states = encoder_hidden_states.to(torch.float32)
                    hidden_states = block(
                        hidden_states,
                        encoder_hidden_states,
                        timestep_proj,
                        rotary_emb,
                    )
                    hidden_states = hidden_states.to(origin_dtype)
                    encoder_hidden_states = encoder_hidden_states.to(origin_dtype)
                else:
                    hidden_states = block(
                        hidden_states,
                        encoder_hidden_states,
                        timestep_proj,
                        rotary_emb,
                    )

                if output_hidden_states and i in output_hidden_states_idx:
                    hidden_states_list.append(hidden_states)

        # 5. Output norm, projection & unpatchify
        with torch.amp.autocast("cuda", dtype=torch.float32):
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
            hidden_states = (
                self.norm_out(hidden_states.float()) * (1 + scale) + shift
            ).type_as(hidden_states)
            hidden_states = self.proj_out(hidden_states)

        hidden_states = Gather.apply(sequence_parallel_group, hidden_states, 1)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return {
            "sample": output,
            "hidden_states": hidden_states_list
        }
