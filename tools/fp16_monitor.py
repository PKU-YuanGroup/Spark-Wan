import torch
from collections import defaultdict


class FP16Monitor:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.stats = defaultdict(dict)

        # 注册前向传播钩子
        for name, module in model.named_modules():
            hook = self._create_hook(name)
            self.hooks.append(module.register_forward_hook(hook))

    def _create_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if output.dtype == torch.float16:
                self.stats[name] = {
                    "max": output.max().item(),
                    "min": output.min().item(),
                    "has_inf": torch.isinf(output).any().item(),
                    "has_nan": torch.isnan(output).any().item(),
                }

        return hook

    def get_problematic_modules(self):
        return {
            name: stats
            for name, stats in self.stats.items()
            if stats["has_inf"]
            or stats["has_nan"]
            or abs(stats["max"]) > 1e4
            or abs(stats["min"]) > 1e4
        }

    def __del__(self):
        for hook in self.hooks:
            hook.remove()
