import sys

sys.path.append(".")

import os

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from diffusers import WanPipeline
from diffusers.image_processor import VaeImageProcessor
from spark_wan.models.transformer_wan import WanTransformer3DModel
from spark_wan.training_utils.load_model import replace_rmsnorm_with_fp32
from spark_wan.parrallel.env import init_sequence_parallel_group
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from safetensors.torch import load_file
from tools.fp16_monitor import FP16Monitor

# 确保推理确定性
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

dist.init_process_group(backend="nccl")
torch.cuda.set_device(dist.get_rank())
sp_group_index, sp_group_local_rank = init_sequence_parallel_group(8)
weight_dtype = torch.bfloat16
seed = 2002
flow_shift = 5.0
prompts = [
    "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, along red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is dampand reflective, creating a mirror effect of thecolorful lights. Many pedestrians walk about.",
    # "Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered tree sand dramatic snow capped mountains in the distance,mid afternoon lightwith wispy cloud sand a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field",
    # "A movie trailer featuring the adventures ofthe 30 year old spacemanwearing a redwool knitted motorcycle helmet, bluesky, saltdesert, cinematic style, shoton 35mm film, vivid colors. ",
    # "Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach.The crashing blue waters create white-tipped waves,while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green",
    # "shrubbery covers the cliffs edge. The steep drop from the road down to the beach is adramatic feat, with the cliff's edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.",
    # "Animated scene features a close-up of a short fluffy monster kneeling beside a melting red candle.The art style is 3D and realistic,with a focus on lighting and texture.The mood of the painting is one of wonder and curiosity, as the monster gazes at the flame with wide eyes and",
    # "open mouth. lts pose and expression convey a sense of innocence and playfulness, as if it is exploring the world around it for the first time.The use of warm colors and dramatic lighting further enhances the cozy atmosphere of the image.",
    # "A gorgeously rendered papercraft world of a coral reef,rife with colorful fish and sea creatures.",
    # "This close-up shot of a Victoria crowned pigeon showcases its striking blue plumage and red chest. Its crest is made of delicate, lacy feathers, while its eye is a striking red color. The bird's head is tilted slightly to the side,giving the impression of it looking regal and majestic. The background is blurred,drawing attention to the bird's striking appearance.",
    # "Photorealistic closeup video of two pirate ships battling each other as they sail inside a cup of coffee.",
    # "A young man at his 20s is sitting on a piece of cloud in the sky, reading a book.",
    # "A petri dish with a bamboo forest growing within it that has tiny red pandas running around.",
    # "The camera rotates around a large stack of vintage televisions all showing different programs-1950s sci-fi movies, horror movies, news, static, a 1970s sitcom, etc, set inside a large New York museum gallery.",
    # "3D animation of a small, round, fluffy creature with big, expressive eyes explores a vibrant, enchanted forest. The creature, a whimsical blend of a rabbit and a squirrel, has soft blue fur and a bushy, striped tail. It hops along a sparkling stream,its eyes wide with wonder. The forest is alive with magical elements: flowers that glow and change colors, trees with leaves in shades of purple and silver, and small floating lights that resemble fireflies. The creature stops to interact playfully with a group of tiny, fairy-like beings dancing around a mushroom ring. The creature looks up in awe at a large, glowing tree that seems to be the heart of the forest.",
    # "Historical footage of California during the gold rush.",
    # "A close up view of a glass sphere that has a zen garden within it. There is a small dwarf in the sphere who is raking the zen garden and creating patterns in the sand.",
    # "Extreme close up of a 24 year old woman's eye blinking, standing in Marrakech during magic hour, cinematic film shot in 70mm, depth of field,vivid colors, cinematic.",
    # "A cartoon kangaroo disco dances.",
    # "A beautiful homemade video showing the people of Lagos, Nigeria in the year 2056. Shot with a mobile phone camera.",
    # "A cat waking up its sleeping owner demanding breakfast.The owner tries to ignore the cat, but the cat tries new tactics and finally the owner pulls out a secret stash of treats from under the pillow to hold the cat off a little longer.",
]
# negative_prompt = ""
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
steps = 32
output_video_path = f"output/origin_{steps}_seed_{seed}_flow_{flow_shift}5/"
state_dict_path = None
# state_dict_path = f"/data/pfs/checkpoints/chestnutlzj/Spark-Wan-16steps/model.safetensors"
model_path = "/data/pfs/checkpoints/Wan2.1-T2V-14B-Diffusers/"

os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
vae = AutoencoderKLWan.from_pretrained(
    model_path,
    subfolder="vae",
    torch_dtype=weight_dtype,
)
transformer = WanTransformer3DModel.from_pretrained(
    model_path,
    subfolder="transformer",
    torch_dtype=weight_dtype,
)
transformer = replace_rmsnorm_with_fp32(transformer)
cfg = 5.0
if state_dict_path is not None:
    cfg = 0.0
    lora_target_modules = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "proj_out",
        "ffn.net.0.proj",
        "ffn.net.2",
    ]
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=lora_target_modules
    )
    transformer = get_peft_model(transformer, lora_config)  
    state_dict = load_file(
        state_dict_path,
        device="cpu",
    )
    missing_keys, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)
    print("unexpected_keys: ", unexpected_keys)
    print("loading lora done!")

transformer.eval()

print("cfg: ", cfg, "steps: ", steps)
scheduler = UniPCMultistepScheduler(
    prediction_type="flow_prediction",
    use_flow_sigmas=True,
    num_train_timesteps=1000,
    flow_shift=flow_shift,
)

pipe = WanPipeline.from_pretrained(
    model_path,
    transformer=transformer,
    vae=vae,
    scheduler=scheduler,
    torch_dtype=weight_dtype,
)

pipe = pipe.to(weight_dtype, device="cuda")

videos = []

idx = 0
for prompt in tqdm(prompts):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        pt_images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=720,
            width=1280,
            num_inference_steps=steps,
            num_frames=81,
            guidance_scale=cfg,
            generator=generator,
            output_type="pt",
        ).frames[0]
    pt_images = torch.stack([pt_images[i] for i in range(pt_images.shape[0])])
    image_np = VaeImageProcessor.pt_to_numpy(pt_images)
    image_pil = VaeImageProcessor.numpy_to_pil(image_np)
    if dist.get_rank() == 0:
        video_path = f"{output_video_path}/output_{idx}.mp4"
        export_to_video(image_pil, video_path, fps=16)
    idx += 1

dist.barrier()
