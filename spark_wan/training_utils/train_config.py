from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReportTo:
    wandb: bool = field(default=True)
    project_name: str = field(default="Spark-Wan")
    wandb_name: str = field(default="testrun")
    wandb_notes: str = field(default="")


@dataclass
class DataConfig:
    instance_data_root: str = field(default="", metadata={"required": True})
    height: int = field(default=480)
    width: int = field(default=720)
    fps: int = field(default=8)
    max_num_frames: int = field(default=49)
    dataloader_num_workers: int = field(default=0)


@dataclass
class ModelConfig:
    pretrained_lora_path: Optional[str] = field(default=None)
    pretrained_model_name_or_path: Optional[str] = field(
        default=None,
    )
    transformer_subfolder: str = field(default="transformer")
    compile_transformer: bool = field(default=False)
    fsdp_transformer: bool = field(default=False)
    fsdp_discriminator: bool = field(default=False)
    fsdp_text_encoder: bool = field(default=False)
    cpu_offload: List[str] = field(default_factory=lambda: [])
    reshard_after_forward: List[str] = field(default_factory=lambda: [])
    is_train_lora: bool = field(default=True)
    lora_rank: int = field(default=128)
    lora_alpha: float = field(default=128)
    lora_dropout: float = field(default=0.0)
    lora_target_modules: Optional[str] = field(default=None)
    is_train_disc_lora: bool = field(default=False)
    disc_lora_rank: int = field(default=64)
    disc_lora_alpha: float = field(default=128)
    disc_lora_dropout: float = field(default=0.0)
    disc_target_modules: Optional[str] = field(default=None)
    flow_shift: float = field(default=8.0)


@dataclass
class ValidationConfig:
    validation_steps: int = field(default=100)
    validation_prompt: Optional[str] = field(
        default="A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
    )
    validation_prompt_separator: str = field(default=":::")
    num_validation_videos: int = field(default=1)
    log_teacher_sample: bool = field(default=False)


@dataclass
class TrainingConfig:
    resume_from_checkpoint: Optional[str] = field(default=None)
    gradient_accumulation_steps: int = field(default=1)
    graident_clipping: bool = field(default=True)
    max_train_steps: Optional[int] = field(default=None)
    checkpointing_steps: int = field(default=500)
    train_batch_size: int = field(default=4)
    num_train_epochs: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)
    disc_gradient_checkpointing: bool = field(default=False)
    mixed_precision: str = field(
        default="bf16",
        metadata={"choices": ["no", "fp16", "bf16"]},
    )
    learning_rate: float = field(default=1e-4)
    lr_scheduler: str = field(
        default="constant",
        metadata={
            "choices": [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ]
        },
    )
    lr_warmup_steps: int = field(default=500)
    lr_num_cycles: int = field(default=1)
    lr_power: float = field(default=1.0)
    optimizer: str = field(
        default="adam",
        metadata={
            "choices": ["adam", "adamw", "prodigy"],
        },
    )
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    adam_weight_decay: float = field(default=1e-04)
    adam_epsilon: float = field(default=1e-08)
    max_grad_norm: float = field(default=1.0)
    # ---- Flow Matching ----
    weighting_scheme: str = field(default="logit_normal")
    logit_mean: float = field(default=0.0)
    logit_std: float = field(default=1.0)
    mode_scale: float = field(default=1.29)


@dataclass
class ParallelConfig:
    sp_size: int = field(default=1)


@dataclass
class StepDistillConfig:
    teacher_step: int = field(default=32)
    student_step: int = field(default=16)
    guidance_scale_max: float = field(default=8.0)
    guidance_scale_min: float = field(default=4.0)
    is_gan_distill: bool = field(default=False)
    disc_interval: int = field(default=2)
    disc_start: int = field(default=0)
    distance_weight: float = field(default=1.0)
    disc_weight: float = field(default=1.0)
    scheduler_type: str = field(default="UniPC")
    discriminator_copy_num_layers: int = field(default=4)
    adaptive_weight: bool = field(default=False)
    reduce_func: str = field(default="mean")
    discriminator_head_type: str = field(default="complex")
    discriminator_dropout: float = field(default=0.0)
    default_disc_weight: float = field(default=1e5)
    disc_loss_type: str = field(default="hinge")
    discriminator_seaweed_output_layer: List[int] = field(default_factory=lambda: [-1])


@dataclass
class APTDistillConfig:
    is_gan_distill: bool = field(default=True)
    student_step: int = field(default=4)
    disc_interval: int = field(default=2)
    disc_weight: float = field(default=1.0)
    scheduler_type: str = field(default="UniPC")
    discriminator_copy_num_layers: int = field(default=4)
    discriminator_head_type: str = field(default="complex")
    discriminator_dropout: float = field(default=0.0)
    default_disc_weight: float = field(default=1e5)
    disc_loss_type: str = field(default="hinge")
    discriminator_seaweed_output_layer: List[int] = field(default_factory=lambda: [-1])


@dataclass
class SelfLayerDistillConfig:
    layers_idx: List[int] = field(default_factory=list)
    scheduler: str = field(default="linear")
    zero_step: int = field(default=1000)
    scheduler_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Args:
    output_dir: str = field(default="wan-lora")
    seed: int = field(default=1234)
    report_to: ReportTo = field(default_factory=ReportTo)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    step_distill_config: Optional[StepDistillConfig] = field(default=None)
    apt_distill_config: Optional[APTDistillConfig] = field(default=None)
    self_layer_distill_config: Optional[SelfLayerDistillConfig] = field(default=None)
    logging_dir: str = field(default="logs")
