import safetensors
import torch
from torch.distributed._tensor import DTensor

class EMA:
    def __init__(self, model, decay: float = 0.99, fsdp_resharded: bool = False):
        self.model = model
        self.decay = decay
        self.fsdp_resharded = fsdp_resharded
        self.shadow = {}
        self.backup = {}
        self.is_registered = False
        
    def register(self):
        if self.is_registered:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().cpu()
        self.is_registered = True
        
    def update(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                new_average = (1.0 - self.decay) * param.data.cpu() + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                shadow_tensor = self.shadow[name]
                self.backup[name] = param.data.clone()
                param.data.copy_(shadow_tensor.to(param.data.device))
                
    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.backup[name])
        
    def save_ckpt(self, path):
        state_dict = {}
        for name, param in self.shadow.items():
            if self.fsdp_resharded:
                state_dict[name] = param.full_tensor()
            else:
                state_dict[name] = param
        safetensors.torch.save_file(state_dict, path)