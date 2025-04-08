from diffusers import UniPCMultistepScheduler

scheduler = UniPCMultistepScheduler(
    prediction_type="flow_prediction",
    use_flow_sigmas=True,
    num_train_timesteps=1000,
    flow_shift=12
)
set_timesteps = scheduler.set_timesteps(1000)
print(scheduler.timesteps)