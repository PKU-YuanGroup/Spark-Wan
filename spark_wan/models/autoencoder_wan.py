import torch

from diffusers import AutoencoderKLWan as AutoencoderKLWanDiffusers


class AutoencoderKLWan(AutoencoderKLWanDiffusers):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def encode(self, video):
        """Normalize latents after encoding"""
        latent_dist = super()._encode(video)
        mu, logvar = torch.chunk(latent_dist, 2, dim=1)
        mu = self._normalize_latents(
            mu,
            torch.tensor(self.config.latents_mean, device=self.device),
            1 / torch.tensor(self.config.latents_std, device=self.device),
        )
        return mu
