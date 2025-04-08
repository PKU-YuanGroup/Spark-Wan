from diffusers import WanTransformer3DModel
from peft import get_peft_model, LoraConfig
from safetensors.torch import load_file

state_dict_path = (
    "/mnt/data/lzj/codes/spark-wan/D1_3B_16_8_formal/checkpoint-600/model.safetensors"
)
transformer = WanTransformer3DModel.from_pretrained(
    "/mnt/data/checkpoints/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    subfolder="distill_16",
)
lora_target_modules = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "proj_out",
    "ffn.net.0.proj",
    "ffn.net.2",
]
lora_config = LoraConfig(r=256, lora_alpha=512, target_modules=lora_target_modules)
transformer = get_peft_model(transformer, lora_config)
state_dict = load_file(
    state_dict_path,
    device="cpu",
)
print(state_dict.keys())
missing_keys, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)
transformer = transformer.merge_and_unload()
transformer.save_pretrained(
    "/mnt/data/checkpoints/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/distill_8"
)
