import html
import re
from typing import List, Optional, Union

import ftfy
import torch


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


def _get_t5_prompt_embeds(
    tokenizer: None,
    text_encoder: None,
    prompt: Union[str, List[str]] = None,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_embeds = text_encoder(
        text_input_ids.to(device), mask.to(device)
    ).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device, non_blocking=True)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [
            torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: None,
    text_encoder: None,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    device = device or text_encoder.device
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt_embeds is None:
        prompt_embeds = _get_t5_prompt_embeds(
            tokenizer,
            text_encoder,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

    return prompt_embeds
