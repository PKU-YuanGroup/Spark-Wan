import sys


sys.path.append(".")

import argparse
import logging
import math
import os

import torch
import torch.distributed as dist
import transformers
from spark_wan.models.discriminator_wan import WanDiscriminator
from spark_wan.parrallel.env import setup_sequence_parallel_group
from spark_wan.training_utils.fsdp2_utils import (
    load_model_state,
    load_optimizer_state,
    load_state,
    prepare_fsdp_model,
    save_state,
    unwrap_model,
)
from spark_wan.training_utils.gan_utils import calculate_adaptive_weight, hinge_d_loss
from spark_wan.training_utils.input_process import encode_prompt
from spark_wan.training_utils.load_dataset import load_easyvideo_dataset
from spark_wan.training_utils.load_model import (
    DistributedDataParallel,
    WanTransformerBlock,
    load_model,
    replace_rmsnorm_with_fp32,
)
from spark_wan.training_utils.load_optimizer import get_optimizer
from spark_wan.training_utils.train_config import Args
from torch.amp import GradScaler
from tqdm.auto import tqdm

from peft import LoraConfig, get_peft_model

import diffusers
from diffusers import (
    UniPCMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
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
    lora_target_modules = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "proj_out",
        "ffn.net.0.proj",
        "ffn.net.2",
    ]
    tokenizer, text_encoder, transformer, vae = load_model(
        pretrained_model_name_or_path=args.model_config.pretrained_model_name_or_path,
        compile_transformer=args.model_config.compile_transformer,
        fsdp_transformer=args.model_config.fsdp_transformer,
        fsdp_text_encoder=args.model_config.fsdp_text_encoder,
        weight_dtype=weight_dtype,
        device=device,
        is_train_lora=True,
        lora_rank=args.model_config.lora_rank,
        lora_alpha=args.model_config.lora_alpha,
        lora_dropout=args.model_config.lora_dropout,
        lora_target_modules=lora_target_modules,
        pretrained_lora_path=args.model_config.pretrained_lora_path,
        reshard_after_forward=args.parallel_config.reshard_after_forward,
        transformer_subfolder=args.model_config.transformer_subfolder,
    )
    # Make sure the reshard_after_forward is not True when using adaptive weight.
    # assert not (args.parallel_config.reshard_after_forward and args.step_distill_config.adaptive_weight)

    if args.step_distill_config.is_gan_distill:
        model_config = transformer.config
        model_config["num_layers"] = (
            args.step_distill_config.discriminator_copy_num_layers
        )
        model_config["cnn_dropout"] = args.step_distill_config.discriminator_dropout
        model_config["head_type"] = args.step_distill_config.discriminator_head_type
        discriminator = WanDiscriminator(
            **model_config,
        )
        pretrained_checkpoint = transformer.state_dict()
        missing_keys, unexpected_keys = discriminator.load_state_dict(
            pretrained_checkpoint, strict=False
        )
        print(
            f"DISC: missing_keys {len(missing_keys)} {missing_keys}, unexpected_keys {len(unexpected_keys)}"
        )
        print(
            f"DISC: Successfully load {len(pretrained_checkpoint) - len(missing_keys)}/{len(pretrained_checkpoint)} keys!"
        )
        discriminator = replace_rmsnorm_with_fp32(discriminator)
        if args.training_config.disc_gradient_checkpointing:
            discriminator.enable_gradient_checkpointing()
        
        if args.model_config.is_train_disc_lora:
            discriminator.requires_grad_(False)
            discriminator.dis_head.requires_grad_(True)
            lora_config = LoraConfig(
                r=args.model_config.disc_lora_rank,
                target_modules=lora_target_modules,
                lora_alpha=args.model_config.disc_lora_alpha,
                lora_dropout=args.model_config.disc_lora_dropout,
                init_lora_weights=True,
            )
            discriminator = get_peft_model(discriminator, lora_config)
        
        if args.model_config.fsdp_discriminator:
            prepare_fsdp_model(
                discriminator,
                shard_conditions=[lambda n, m: isinstance(m, WanTransformerBlock)],
                cpu_offload=True,
                reshard_after_forward=True, # Discriminator need to reshard after forward.
                weight_dtype=weight_dtype,
            )
        else:
            discriminator = discriminator.to(device)
            discriminator = DistributedDataParallel(discriminator, device_ids=[device])

    # Setup distillation parameters
    if args.step_distill_config.scheduler_type == "UniPC":
        scheduler_type = UniPCMultistepScheduler
        scheduler_kwargs = {
            "prediction_type": "flow_prediction",
            "use_flow_sigmas": True,
            "num_train_timesteps": 1000,
            "flow_shift": args.model_config.flow_shift,
        }
    else:
        raise ValueError(
            f"Scheduler type {args.step_distill_config.scheduler_type} not supported"
        )
    print(f"Scheduler type: {args.step_distill_config.scheduler_type}")

    teacher_noise_scheduler = scheduler_type(**scheduler_kwargs)
    student_noise_scheduler = scheduler_type(**scheduler_kwargs)
    teacher_steps = args.step_distill_config.teacher_step
    student_steps = args.step_distill_config.student_step
    guidance_scale_min = args.step_distill_config.guidance_scale_min
    guidance_scale_max = args.step_distill_config.guidance_scale_max

    # Make sure the trainable params are in float32.
    if args.training_config.mixed_precision == "fp16":
        # only upcast trainable parameters into fp32
        cast_training_params([transformer], dtype=torch.float32)
        if args.step_distill_config.is_gan_distill:
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
    if args.step_distill_config.is_gan_distill:
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

    sample_neg_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    uncond_text = [sample_neg_prompt] * args.training_config.train_batch_size
    uncond_context = encode_prompt(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=uncond_text,
        device=device,
        dtype=weight_dtype,
    )

    for epoch in range(first_epoch, args.training_config.num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)

        if args.step_distill_config.is_gan_distill:
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
            teacher_noise_scheduler.set_timesteps(teacher_steps)
            student_noise_scheduler.set_timesteps(student_steps)
            num_teacher_steps = len(teacher_noise_scheduler.timesteps)
            num_student_steps = len(student_noise_scheduler.timesteps)
            times = num_teacher_steps // num_student_steps
            prob = torch.ones(num_student_steps) / (num_student_steps)
            start_idx = torch.multinomial(prob, 1)
            start_timestep = student_noise_scheduler.timesteps[start_idx]
            if args.step_distill_config.is_gan_distill:
                if start_idx + 1 < len(student_noise_scheduler.timesteps):
                    end_timestep = student_noise_scheduler.timesteps[start_idx + 1]
                else:
                    end_timestep = 0
                end_timesteps = torch.tensor([end_timestep], device=device).repeat(bsz)
                start_timesteps = torch.tensor([start_timestep], device=device).repeat(
                    bsz
                )
            else:
                start_timesteps = torch.tensor([start_timestep], device=device).repeat(
                    bsz
                )
            teacher_timesteps = teacher_noise_scheduler.timesteps[
                int(start_idx * times) : int((start_idx + 1) * times)
            ]
            assert start_timestep == teacher_timesteps[0]

            # Get z_t (SDXL lightning 3.5 Fix the Schedule)
            if start_idx == 0:
                noisy_sample_init = noise
            else:
                # Add noise to sample
                noisy_sample_init = teacher_noise_scheduler.add_noise(
                    model_input, noise, start_timesteps
                )
            if guidance_scale_max == 0 and guidance_scale_min == 0:
                guidance_scale = 0
            else:
                guidance_scale = (
                    torch.rand(1).to(device, dtype=weight_dtype)
                    * (guidance_scale_max - guidance_scale_min)
                    + guidance_scale_min
                )

            # Teacher step
            unwrap_model(transformer).disable_adapter_layers()
            latents_teacher = noisy_sample_init
            with torch.no_grad():
                for t in teacher_timesteps:
                    if guidance_scale != 0:
                        timestep = torch.tensor([t], device=device).repeat(bsz * 2)
                        latents_teacher_input = teacher_noise_scheduler.scale_model_input(
                            latents_teacher, t
                        )
                        latents_teacher_input = torch.concatenate(
                            [latents_teacher_input, latents_teacher_input], dim=0
                        )
                        input_context = torch.concatenate(
                            [prompt_embeds, uncond_context], dim=0
                        )
                        model_pred = transformer(
                            hidden_states=latents_teacher_input,
                            timestep=timestep,
                            encoder_hidden_states=input_context,
                            return_dict=False,
                        )[0]
                        cond_pred, uncond_pred = model_pred[:bsz], model_pred[bsz:]
                        noise_pred = (
                            guidance_scale * cond_pred + (1 - guidance_scale) * uncond_pred
                        )
                        latents_teacher = teacher_noise_scheduler.step(
                            noise_pred, t, latents_teacher, return_dict=False
                        )[0]
                    else:
                        timestep = torch.tensor([t], device=device).repeat(bsz)
                        latents_teacher_input = teacher_noise_scheduler.scale_model_input(
                            latents_teacher, t
                        )
                        model_pred = transformer(
                            hidden_states=latents_teacher_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            return_dict=False,
                        )[0]
                        latents_teacher = teacher_noise_scheduler.step(
                            model_pred, t, latents_teacher, return_dict=False
                        )[0]
            unwrap_model(transformer).enable_adapter_layers()

            # Student step
            latents_student = noisy_sample_init
            latents_student_input = student_noise_scheduler.scale_model_input(
                latents_student, start_timestep
            )
            model_pred = transformer(
                hidden_states=latents_student_input,
                timestep=start_timesteps,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            latents_student = student_noise_scheduler.step(
                model_pred, start_timestep, latents_student, return_dict=False
            )[0]

            if (
                args.step_distill_config.is_gan_distill
            ):  # If training with discriminator
                is_generator_step = (
                    step % args.step_distill_config.disc_interval == 0
                    or step < args.step_distill_config.disc_start
                )
                if is_generator_step:
                    unwrap_model(discriminator).requires_grad_(False)

                    if args.step_distill_config.distance_weight > 0:
                        if args.step_distill_config.reduce_func == "sum":
                            reduce_func = torch.sum
                        elif args.step_distill_config.reduce_func == "mean":
                            reduce_func = torch.mean
                        else:
                            raise ValueError(
                                f"Invalid reduce function: {args.step_distill_config.reduce_func}"
                            )

                        rec_loss = reduce_func(
                            torch.abs(latents_student - latents_teacher)
                        )
                    else:  # No distance loss
                        rec_loss = torch.tensor(0.0)

                    if step >= args.step_distill_config.disc_start:
                        score_student = discriminator(
                            hidden_states=latents_student,
                            timestep=end_timesteps,
                            encoder_hidden_states=prompt_embeds,
                            return_dict=False,
                        )
                        g_loss = -torch.mean(score_student)

                        if args.step_distill_config.adaptive_weight:
                            if args.model_config.fsdp_transformer:
                                transformer.set_reshard_after_backward(False)
                            adaptive_disc_weight = calculate_adaptive_weight(
                                rec_loss,
                                g_loss,
                                last_layer=[
                                    unwrap_model(transformer)
                                    .base_model.blocks[-1]
                                    .ffn.net[2]
                                    .lora_A.default.weight,
                                    unwrap_model(transformer)
                                    .base_model.blocks[-1]
                                    .ffn.net[2]
                                    .lora_B.default.weight,
                                ],
                            )
                            if args.model_config.fsdp_transformer:
                                transformer.set_reshard_after_backward(True)
                        else:
                            adaptive_disc_weight = torch.tensor(args.step_distill_config.default_disc_weight)
                    else:
                        adaptive_disc_weight = torch.tensor(0.0)
                        g_loss = torch.tensor(0.0)

                    loss = (
                        args.step_distill_config.distance_weight * rec_loss
                        + adaptive_disc_weight
                        * args.step_distill_config.disc_weight
                        * g_loss
                    )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        transformer_lora_parameters, args.training_config.max_grad_norm
                    )
                    scaler.step(optimizer)
                else:
                    unwrap_model(discriminator).requires_grad_(True)
                    score_teacher = discriminator(
                        hidden_states=latents_teacher.detach(),
                        timestep=end_timesteps,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )
                    score_student = discriminator(
                        hidden_states=latents_student.detach(),
                        timestep=end_timesteps,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )
                    loss = hinge_d_loss(score_teacher, score_student)
                    scaler.scale(loss).backward()
                    scaler.unscale_(disc_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        disc_params, args.training_config.max_grad_norm
                    )
                    scaler.step(disc_optimizer)
            else:  # No gan
                adaptive_disc_weight = torch.tensor(0.0)

                rec_loss = torch.sum(
                    torch.abs(latents_student.float() - latents_teacher.float())
                )
                loss = rec_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    transformer_lora_parameters, args.training_config.max_grad_norm
                )
                scaler.step(optimizer)

            scaler.update()
            lr_scheduler.step()
            if args.step_distill_config.is_gan_distill:
                disc_optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            progress_bar.update(1)
            global_step += 1

            # Log to tracker
            if global_rank == 0:
                if args.step_distill_config.is_gan_distill:
                    if is_generator_step:
                        logs = {
                            "gen_loss": loss.detach().cpu().item(),
                            "rec_loss": rec_loss.detach().cpu().item(),
                            "g_loss": g_loss.detach().cpu().item(),
                            "adaptive_disc_weight": adaptive_disc_weight.detach()
                            .cpu()
                            .item(),
                        }
                        progress_bar.set_postfix(**logs)
                        wandb.log(logs, step=global_step)
                    else:
                        logs = {
                            "disc_loss": loss.detach().cpu().item(),
                            "adaptive_disc_weight": adaptive_disc_weight.detach()
                            .cpu()
                            .item(),
                            "score_teacher": score_teacher.mean().detach().cpu().item(),
                            "score_student": score_student.mean().detach().cpu().item(),
                        }
                        progress_bar.set_postfix(**logs)
                        wandb.log(logs, step=global_step)
                else:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    progress_bar.set_postfix(**logs)
                    wandb.log(logs, step=global_step)

            if global_step % args.training_config.checkpointing_steps == 0:
                dist.barrier()
                checkpoint_path = os.path.join(
                    args.output_dir, f"checkpoint-{global_step}"
                )
                save_state(
                    output_dir=checkpoint_path,
                    global_step=global_step,
                    model=unwrap_model(transformer),
                    is_fsdp=args.model_config.fsdp_transformer,
                    optimizer=optimizer,
                    dataloader=train_dataloader,
                    sampler=sampler,
                    save_key_filter="lora",
                    scaler=scaler,
                    lr_scheduler=lr_scheduler,
                )

            # Free memory
            del model_pred
            del videos
            del prompt_embeds
            del model_input
            del noise
            del loss
            del latents_teacher
            del latents_student
            free_memory()

            if (
                global_step % args.validation_config.validation_steps == 0
                or global_step == 1
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
                step = args.step_distill_config.student_step
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
                    with torch.no_grad():
                        log_validation(
                            pipe=pipe,
                            args=args,
                            pipeline_args=pipeline_args,
                            global_step=global_step,
                            phase_name="student/validation",
                            global_rank=global_rank,
                        )
                    if global_step == 1 and args.validation_config.log_teacher_sample:
                        unwrap_model(transformer).disable_adapter_layers()
                        pipeline_args["guidance_scale"] = 5.0
                        with torch.no_grad():
                            pipeline_args["num_inference_steps"] = teacher_steps
                            log_validation(
                                pipe=pipe,
                                args=args,
                                pipeline_args=pipeline_args,
                                global_step=global_step,
                                phase_name="teacher/teacher_step",
                                global_rank=global_rank,
                            )
                            pipeline_args["num_inference_steps"] = student_steps
                            log_validation(
                                pipe=pipe,
                                args=args,
                                pipeline_args=pipeline_args,
                                global_step=global_step,
                                phase_name="teacher/student_step",
                                global_rank=global_rank,
                            )
                        unwrap_model(transformer).enable_adapter_layers()
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
