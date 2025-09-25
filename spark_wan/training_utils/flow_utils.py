import torch


def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=torch.cuda.current_device(), dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(torch.cuda.current_device())
    timesteps = timesteps.to(torch.cuda.current_device())
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)

    return sigma
