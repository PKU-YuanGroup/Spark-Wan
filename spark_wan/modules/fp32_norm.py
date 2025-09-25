import torch.nn as nn

from diffusers.models.normalization import RMSNorm
from diffusers.utils import is_torch_npu_available, is_torch_version


class FP32RMSNorm(RMSNorm):
    def forward(self, hidden_states):
        if is_torch_npu_available():
            raise ValueError("FP32RMSNorm is not available on NPU")

        if not is_torch_version(">=", "2.4"):
            raise ValueError("FP32RMSNorm is only available in PyTorch 2.4 or higher")

        original_dtype = hidden_states.dtype
        hidden_states = nn.functional.rms_norm(
            hidden_states.float(),
            normalized_shape=(hidden_states.shape[-1],),
            weight=self.weight.float(),
            eps=self.eps,
        )
        if self.bias is not None:
            hidden_states = hidden_states + self.bias.float()

        return hidden_states.to(original_dtype)
