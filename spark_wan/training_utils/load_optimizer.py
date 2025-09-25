from typing import List

import torch


def get_optimizer(
    optimizer: str,
    learning_rate: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
    adam_weight_decay: float,
    params_to_optimize: List[torch.nn.Parameter],
    use_deepspeed: bool = False,
):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
            weight_decay=adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw"]
    if optimizer not in supported_optimizers:
        print(
            f"Unsupported choice of optimizer: {optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        optimizer = "adamw"

    if optimizer.lower() == "adamw":
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
            weight_decay=adam_weight_decay,
        )
    elif optimizer.lower() == "adam":
        optimizer_class = torch.optim.Adam
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
            weight_decay=adam_weight_decay,
        )

    return optimizer
