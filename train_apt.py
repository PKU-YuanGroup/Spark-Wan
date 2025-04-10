import sys


sys.path.append(".")

import argparse
import logging
import math
import os

import torch
import torch.distributed as dist
import transformers
from spark_wan.parrallel.env import setup_sequence_parallel_group
from spark_wan.training_utils.fsdp2_utils import (
    load_model_state,
    load_optimizer_state,
    load_state,
    save_state,
    unwrap_model,
)
from spark_wan.training_utils.gan_utils import hinge_d_loss, d_loss
from spark_wan.training_utils.input_process import encode_prompt
from spark_wan.training_utils.load_dataset import load_easyvideo_dataset
from spark_wan.training_utils.load_model import (
    load_model,
)
from spark_wan.training_utils.load_optimizer import get_optimizer
from spark_wan.training_utils.train_config import Args
from torch.amp import GradScaler
from tqdm.auto import tqdm


import diffusers
from diffusers import (
    UniPCMultistepScheduler,
    WanPipeline,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, free_memory, set_seed
from diffusers.utils import check_min_version, export_to_video, is_wandb_available

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")


def setup_distributed_env():
    dist.init_process_group(backend="cuda:nccl,cpu:gloo")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_distributed_env():
    dist.destroy_process_group()


def main(args: Args):
    setup_distributed_env()

    set_seed(args.seed)

    global_rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.cuda.current_device()
    world_size = dist.get_world_size()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if local_rank == 0:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if global_rank == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Mixed precision training
    weight_dtype = torch.float32
    if args.training_config.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.training_config.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif args.training_config.mixed_precision == "no":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load model
    tokenizer, text_encoder, transformer, vae, discriminator = load_model(
        model_config=args.model_config,
        training_config=args.training_config,
        distill_config=args.apt_distill_config,
        weight_dtype=weight_dtype,
        device=device,
    )

    # Setup distillation parameters
    if args.apt_distill_config.scheduler_type == "UniPC":
        scheduler_type = UniPCMultistepScheduler
        scheduler_kwargs = {
            "prediction_type": "flow_prediction",
            "use_flow_sigmas": True,
            "num_train_timesteps": 1000,
            "flow_shift": args.model_config.flow_shift,
        }
    else:
        raise ValueError(
            f"Scheduler type {args.apt_distill_config.scheduler_type} not supported"
        )
    print(f"Scheduler type: {args.apt_distill_config.scheduler_type}")

    student_noise_scheduler = scheduler_type(**scheduler_kwargs)
    discriminator_noise_scheduler = scheduler_type(**scheduler_kwargs)
    student_steps = args.apt_distill_config.student_step

    # Make sure the trainable params are in float32.
    if args.training_config.mixed_precision == "fp16":
        # only upcast trainable parameters into fp32
        cast_training_params([transformer], dtype=torch.float32)
        cast_training_params([discriminator], dtype=torch.float32)

    # Resume model state from checkpoint
    if args.training_config.resume_from_checkpoint:
        load_model_state(
            unwrap_model(transformer),
            args.training_config.resume_from_checkpoint,
            is_fsdp=args.model_config.fsdp_transformer,
            device=device,
            fsdp_cpu_offload=False,
        )

    # Setup optimizer
    transformer_lora_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters())
    )
    transformer_parameters_with_lr = {
        "params": transformer_lora_parameters,
        "lr": args.training_config.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]
    optimizer = get_optimizer(
        optimizer=args.training_config.optimizer,
        learning_rate=args.training_config.learning_rate,
        adam_beta1=args.training_config.adam_beta1,
        adam_beta2=args.training_config.adam_beta2,
        adam_epsilon=args.training_config.adam_epsilon,
        adam_weight_decay=args.training_config.adam_weight_decay,
        params_to_optimize=params_to_optimize,
    )
    disc_params = filter(lambda p: p.requires_grad, discriminator.parameters())
    disc_params_with_lr = {
        "params": disc_params,
        "lr": args.training_config.learning_rate,
    }
    disc_params_to_optimize = [disc_params_with_lr]
    disc_optimizer = get_optimizer(
        optimizer=args.training_config.optimizer,
        learning_rate=args.training_config.learning_rate,
        adam_beta1=args.training_config.adam_beta1,
        adam_beta2=args.training_config.adam_beta2,
        adam_epsilon=args.training_config.adam_epsilon,
        adam_weight_decay=args.training_config.adam_weight_decay,
        params_to_optimize=disc_params_to_optimize,
    )
    # Resume optimizer state from checkpoint
    if args.training_config.resume_from_checkpoint:
        load_optimizer_state(
            optimizer,
            args.training_config.resume_from_checkpoint,
            is_fsdp=args.model_config.fsdp_transformer,
        )

    # Setup gradient scaler
    scaler = GradScaler()

    # Setup sequence parallel group
    sp_group_index, sp_group_local_rank, dp_rank, dp_size = (
        setup_sequence_parallel_group(args.parallel_config.sp_size)
    )
    set_seed(args.seed + dp_rank)

    # Load dataset
    train_dataloader, sampler = load_easyvideo_dataset(
        height=args.data_config.height,
        width=args.data_config.width,
        max_num_frames=args.data_config.max_num_frames,
        instance_data_root=args.data_config.instance_data_root,
        train_batch_size=args.training_config.train_batch_size,
        dataloader_num_workers=args.data_config.dataloader_num_workers,
        dp_rank=dp_rank,
        dp_size=dp_size,
        seed=args.seed,
    )

    # Initialize tracker
    if global_rank == 0:
        wandb.init(
            project=args.report_to.project_name,
            name=args.report_to.wandb_name,
            notes=args.report_to.wandb_notes,
            sync_tensorboard=True,
        )
        wandb.config.update(OmegaConf.to_container(args, resolve=True))

    # Scheduler.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.training_config.gradient_accumulation_steps
    )
    if args.training_config.max_train_steps is None:
        args.training_config.max_train_steps = (
            args.training_config.num_train_epochs * num_update_steps_per_epoch
        )
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.training_config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.training_config.lr_warmup_steps * world_size,
        num_training_steps=args.training_config.max_train_steps * world_size,
        num_cycles=args.training_config.lr_num_cycles,
        power=args.training_config.lr_power,
    )

    # Resume state from checkpoint
    if args.training_config.resume_from_checkpoint:
        global_step = load_state(
            args.training_config.resume_from_checkpoint,
            dataloader=train_dataloader,
            sampler=sampler,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
        )
    else:
        global_step = 0

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.training_config.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.training_config.max_train_steps = (
            args.training_config.num_train_epochs * num_update_steps_per_epoch
        )
    # Afterwards we recalculate our number of training epochs
    args.training_config.num_train_epochs = math.ceil(
        args.training_config.max_train_steps / num_update_steps_per_epoch
    )

    # Train!
    total_batch_size = (
        args.training_config.train_batch_size
        * dp_size
        * args.training_config.gradient_accumulation_steps
    )
    num_trainable_parameters = sum(
        param.numel() for model in params_to_optimize for param in model["params"]
    )

    print("***** Running training *****")
    print(f"  Num trainable parameters = {num_trainable_parameters}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num epochs = {args.training_config.num_train_epochs}")
    print(
        f"  Instantaneous batch size per device = {args.training_config.train_batch_size}"
    )
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    print(
        f"  Gradient accumulation steps = {args.training_config.gradient_accumulation_steps}"
    )
    print(f"  Total optimization steps = {args.training_config.max_train_steps}")
    first_epoch = 0
    initial_global_step = global_step
    first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.training_config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not local_rank == 0,
    )

    for epoch in range(first_epoch, args.training_config.num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        discriminator.train()
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            # models_to_accumulate = [transformer]

            with torch.no_grad():
                # Forward & backward
                # Get data samplex
                videos = batch["videos"]
                videos = videos.to(device, dtype=vae.dtype)
                videos = vae.encode(videos)
                # Get latents (z_0)
                model_input = videos.to(
                    memory_format=torch.contiguous_format, dtype=weight_dtype
                )
                prompts = batch["prompts"]

                # encode prompts
                prompt_embeds = encode_prompt(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    prompt=prompts,
                    device=device,
                    dtype=weight_dtype,
                )

            # Sample noise that we'll add to the latents
            # Get noise (epsilon)
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]

            # Get timesteps
            discriminator_noise_scheduler.set_timesteps(1000)
            student_noise_scheduler.set_timesteps(student_steps)

            prob = torch.ones(1000) / (1000)
            discriminator_timestep_idx = torch.multinomial(prob, 1)
            discriminator_timestep = discriminator_noise_scheduler.timesteps[
                discriminator_timestep_idx
            ]
            discriminator_timesteps = torch.tensor(
                [discriminator_timestep], device=device
            ).repeat(bsz)

            # start_idx = 0
            # start_timestep = student_noise_scheduler.timesteps[start_idx]

            noisy_sample_init = noise

            latents_student = noisy_sample_init
            for t in student_noise_scheduler.timesteps:
                timestep = torch.tensor([t], device=device).repeat(bsz)
                latents_student_input = student_noise_scheduler.scale_model_input(
                    latents_student, t
                )
                with torch.amp.autocast("cuda", dtype=weight_dtype):
                    model_pred = transformer(
                        hidden_states=latents_student_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]
                latents_student = student_noise_scheduler.step(
                    model_pred, t, latents_student, return_dict=False
                )[0]

            real_global_step = (
                global_step + 1
            ) // args.training_config.gradient_accumulation_steps
            is_generator_step = (
                real_global_step % args.apt_distill_config.disc_interval == 0
            )
            if is_generator_step:
                unwrap_model(discriminator).requires_grad_(False)
                score_student = discriminator(
                    hidden_states=latents_student,
                    timestep=discriminator_timesteps,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )
                if args.apt_distill_config.disc_loss_type == "hinge":
                    g_loss = -torch.mean(score_student)
                else:
                    g_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        score_student,
                        torch.ones_like(score_student),
                        reduction="mean",
                    )
                loss = g_loss / args.training_config.gradient_accumulation_steps
                scaler.scale(loss).backward()

                if (
                    global_step + 1
                ) % args.training_config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        transformer_lora_parameters, args.training_config.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()
            else:
                unwrap_model(discriminator).requires_grad_(True)
                score_teacher = discriminator(
                    hidden_states=model_input.detach(),
                    timestep=discriminator_timesteps,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )
                score_student = discriminator(
                    hidden_states=latents_student.detach(),
                    timestep=discriminator_timesteps,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )
                if args.apt_distill_config.disc_loss_type == "hinge":
                    loss = hinge_d_loss(score_teacher, score_student)
                else:
                    loss = d_loss(score_teacher, score_student)

                loss = loss / args.training_config.gradient_accumulation_steps
                scaler.scale(loss).backward()
                if (
                    global_step + 1
                ) % args.training_config.gradient_accumulation_steps == 0:
                    scaler.unscale_(disc_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        disc_params, args.training_config.max_grad_norm
                    )
                    scaler.step(disc_optimizer)
                    scaler.update()
                    disc_optimizer.zero_grad(set_to_none=True)

            global_step += 1
            real_global_step = (
                global_step // args.training_config.gradient_accumulation_steps
            )
            is_real_step = (
                global_step % args.training_config.gradient_accumulation_steps == 0
            )

            if not is_real_step:
                continue

            progress_bar.update(1)

            # Log to tracker
            if global_rank == 0:
                if is_generator_step:
                    logs = {
                        "gen_loss": loss.detach().cpu().item(),
                        "g_loss": g_loss.detach().cpu().item(),
                    }
                    progress_bar.set_postfix(**logs)
                    wandb.log(logs, step=real_global_step)
                else:
                    logs = {
                        "disc_loss": loss.detach().cpu().item(),
                        "score_teacher": score_teacher.mean().detach().cpu().item(),
                        "score_student": score_student.mean().detach().cpu().item(),
                    }
                    progress_bar.set_postfix(**logs)
                    wandb.log(logs, step=real_global_step)

            if real_global_step % args.training_config.checkpointing_steps == 0:
                dist.barrier()
                checkpoint_path = os.path.join(
                    args.output_dir, f"checkpoint-{real_global_step}"
                )
                save_state(
                    output_dir=checkpoint_path,
                    global_step=real_global_step,
                    model=unwrap_model(transformer),
                    is_fsdp=args.model_config.fsdp_transformer,
                    optimizer=optimizer,
                    dataloader=train_dataloader,
                    sampler=sampler,
                    save_key_filter="lora" if args.model_config.is_train_lora else None,
                    scaler=scaler,
                    lr_scheduler=lr_scheduler,
                )
                save_state(
                    output_dir=checkpoint_path,
                    global_step=real_global_step,
                    model=unwrap_model(discriminator),
                    is_fsdp=args.model_config.fsdp_discriminator,
                    optimizer=disc_optimizer,
                    save_key_filter=(
                        "lora" if args.model_config.is_train_disc_lora else None
                    ),
                    save_name_prefix="discriminator",
                )

            # Free memory
            del model_pred
            del videos
            del prompt_embeds
            del model_input
            del noise
            del loss
            del latents_student
            free_memory()

            if (
                real_global_step % args.validation_config.validation_steps == 0
                or real_global_step == 1
            ):
                print(f"Validation {global_rank}")
                noise_scheduler_valid = scheduler_type(**scheduler_kwargs)
                pipe = WanPipeline.from_pretrained(
                    args.model_config.pretrained_model_name_or_path,
                    transformer=unwrap_model(transformer),
                    vae=vae,
                    text_encoder=text_encoder,
                    scheduler=noise_scheduler_valid,
                    torch_dtype=weight_dtype,
                )
                validation_prompts = args.validation_config.validation_prompt.split(
                    args.validation_config.validation_prompt_separator
                )
                step = args.apt_distill_config.student_step
                cfg = 0.0
                for validation_prompt in validation_prompts:
                    pipeline_args = {
                        "prompt": validation_prompt,
                        "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                        "guidance_scale": cfg,
                        "num_frames": args.data_config.max_num_frames,
                        "height": args.data_config.height,
                        "width": args.data_config.width,
                        "num_inference_steps": step,
                    }
                    with torch.no_grad() and torch.amp.autocast(
                        "cuda", dtype=weight_dtype
                    ):
                        log_validation(
                            pipe=pipe,
                            args=args,
                            pipeline_args=pipeline_args,
                            global_step=real_global_step,
                            phase_name="student/validation",
                            global_rank=global_rank,
                        )

            if global_step >= args.training_config.max_train_steps:
                break

    # Save the lora layers
    dist.barrier()
    cleanup_distributed_env()


@torch.inference_mode()
def log_validation(
    pipe,
    args: Args,
    pipeline_args,
    global_step: int,
    phase_name="",
    global_rank=0,
):
    print(
        f"Running validation... \n Generating {args.validation_config.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )

    generator = (
        torch.Generator(device="cuda").manual_seed(args.seed) if args.seed else None
    )

    videos = []
    for _ in range(args.validation_config.num_validation_videos):
        pt_images = pipe(**pipeline_args, generator=generator, output_type="pt").frames[
            0
        ]
        pt_images = torch.stack([pt_images[i] for i in range(pt_images.shape[0])])

        image_np = VaeImageProcessor.pt_to_numpy(pt_images)
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)

        videos.append(image_pil)

    video_filenames = []
    for i, video in enumerate(videos):
        prompt = (
            pipeline_args["prompt"][:25]
            .replace(" ", "_")
            .replace(" ", "_")
            .replace("'", "_")
            .replace('"', "_")
            .replace("/", "_")
        )
        filename = os.path.join(
            args.output_dir, f"{phase_name.replace('/', '_')}_video_{i}_{prompt}.mp4"
        )
        if global_rank == 0:
            export_to_video(video, filename, fps=8)
        video_filenames.append(filename)
    if global_rank == 0:
        wandb.log(
            {
                phase_name: [
                    wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                    for i, filename in enumerate(video_filenames)
                ]
            },
            step=global_step,
        )

    del pipe
    free_memory()

    return videos


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    schema = OmegaConf.structured(Args)
    conf = OmegaConf.merge(schema, config)
    main(conf)
