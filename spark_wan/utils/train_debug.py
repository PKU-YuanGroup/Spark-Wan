import torch.nn as nn


def output_grad_info(model: nn.Module, output_file="grad_info.txt"):
    with open(output_file, "w") as f:
        for name, param in model.named_parameters():
            f.write(f"{name}: {param.requires_grad}\n")
