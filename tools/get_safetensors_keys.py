from safetensors import safe_open
from safetensors.torch import save_file
def get_safetensors_keys(filename):
    with safe_open(filename, framework="pt", device="cpu") as f:
        return list(f.keys())
    
def resave_safetensors_keys(old_filename, filename, keys):
    with safe_open(old_filename, framework="pt", device="cpu") as f:
        new_dict = {}
        for key in keys:
            new_dict[key.replace("model.", "")] = f.get_tensor(key)
            
    save_file(new_dict, filename)
    
model_path = "/mnt/data/lzj/codes/spark-wan/1_3B_64_16_formal/checkpoint-800/model.safetensors"
output_path = "/mnt/workspace/checkpoints/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/distill_16/diffusion_pytorch_model.safetensors"
keys = get_safetensors_keys(model_path)
print(keys)
# resave_safetensors_keys(model_path, output_path, keys)


