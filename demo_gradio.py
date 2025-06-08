import gradio as gr
import os
import random

import fcntl

os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

config_path = "config_temp.txt"
file_path = "file_temp.txt"
with open(config_path, "w") as file:
    fcntl.flock(file, fcntl.LOCK_EX)
    file.write("")
    fcntl.flock(file, fcntl.LOCK_UN)

with open(file_path, "w") as file:
    fcntl.flock(file, fcntl.LOCK_EX)
    file.write("")
    fcntl.flock(file, fcntl.LOCK_UN)

def write_params_to_txt(
    file_path, prompt, width, height, video_length, seed, num_inference_steps
):
    with open(file_path, "w") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.write(f"prompt={prompt}\n")
        file.write(f"width={width}\n")
        file.write(f"height={height}\n")
        file.write(f"video_length={video_length}\n")
        file.write(f"seed={seed}\n")
        file.write(f"num_inference_steps={num_inference_steps}\n")
        fcntl.flock(file, fcntl.LOCK_UN)


def generate_video(prompt, resolution, video_length, seed, num_inference_steps):
    width, height = map(int, resolution.split("x"))

    if seed == -1:
        seed = random.randint(0, 1_000_000)
    write_params_to_txt(
        config_path, prompt, width, height, video_length, seed, num_inference_steps
    )

    flag = 0
    while True:
        with open(file_path, "r") as file:
            for line in file:
                if line.startswith("video_path="):
                    video_path = line.strip().split("=", 1)[1]
                    print(f"video path is :{video_path}")
                    flag = 1
                    break
        if flag == 1:
            break

    with open(config_path, "w") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.write("")
        fcntl.flock(file, fcntl.LOCK_UN)

    with open(file_path, "w") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.write("")
        fcntl.flock(file, fcntl.LOCK_UN)

    return video_path


t2v_prompt_examples = [
    "无人机镜头从空中俯瞰，渐渐接近雄伟的大雁塔，塔身在夕阳的金色余晖中显得尤为庄严，犹如一座古老的灯塔，守护着这片历史的土地。周围的广场和街道逐渐显现出繁忙的景象，游客在塔下驻足，微弱的步伐声和交谈声轻轻传入耳中。无人机镜头缓慢向上升起，展示出整个大唐不夜城的壮丽景观。远处，现代高楼大厦与古老的建筑融为一体，霓虹灯的光辉闪烁，与古塔的静谧形成鲜明对比。镜头继续拉远，穿越城市的空中，鸟瞰下方那些交错的街道和熙熙攘攘的人群。随着夜幕的降临，整个大唐不夜城逐渐点亮，霓虹和灯光在夜空中汇成一片璀璨的星海。镜头飞掠过街头，捕捉到各色灯光下人们的笑脸和忙碌的脚步，音乐也在背景中渐渐转换成融合了古风和现代节奏的动感旋律，完美呈现出一个既古老又现代的城市风貌。",
    "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, along red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is dampand reflective, creating a mirror effect of thecolorful lights. Many pedestrians walk about.",
    "动画场景特写中，一个矮小、毛茸茸的怪物跪在一根融化的红蜡烛旁。三维写实的艺术风格注重光照和纹理的相互作用，在整个场景中投射出引人入胜的阴影。怪物睁着好奇的大眼睛注视着火焰，它的皮毛在温暖闪烁的光芒中轻轻拂动。镜头慢慢拉近，捕捉到怪物皮毛的复杂细节和精致的熔蜡液滴。怪物试探性地伸出一只爪子，似乎想要触碰火焰，而烛光则在它周围闪烁舞动，气氛充满了惊奇和好奇。",
    "A drone camera gracefully circles a historic church perched on a rugged outcropping along the Amalfi Coast, capturing its magnificent architectural details and tiered pathways and patios. Below, waves crash against the rocks, while the horizon stretches out over the coastal waters and hilly landscapes of Italy. Distant figures stroll and enjoy the breathtaking ocean views from the patios, creating a dynamic scene. The warm glow of the afternoon sun bathes the scene in a magical and romantic light, casting long shadows and adding depth to the stunning vista. The camera occasionally zooms in to highlight the intricate details of the church, then pans out to showcase the expansive coastline, creating a captivating visual narrative.",
    "一个特写镜头捕捉到一位 60 多岁、留着胡子的白发老人，他坐在巴黎的一家咖啡馆里陷入沉思，思考着宇宙的历史。他的眼睛紧紧盯着屏幕外走动的人们，而自己却一动不动。他身着羊毛大衣、纽扣衬衫、棕色贝雷帽，戴着一副眼镜，散发着教授的风范。他偶尔瞥一眼四周，目光停留在背景中熙熙攘攘的巴黎街道和城市景观上。场景沐浴在金色的光线中，让人联想到 35 毫米电影胶片。当他微微前倾时，眼睛睁大，露出顿悟的瞬间，并微微闭口微笑，暗示他已经找到了生命奥秘的答案。景深营造出光影交错的动态效果，烘托出智慧沉思的氛围。",
    "A cheerful otter confidently balances on a surfboard, donning a bright yellow lifejacket, as it glides through the shimmering turquoise waters near lush tropical islands. The scene is rendered in a 3D digital art style, with the sunlight casting playful shadows on the water's surface. The otter occasionally dips its paws into the water, sending up sprays of droplets that catch the light, adding a sense of motion and excitement to the tranquil atmosphere.",
]

t2v_videos_examples = [
    "./examples/output_0.mp4",
    "./examples/output_1.mp4",
    "./examples/output_2.mp4",
    "./examples/output_3.mp4",
    "./examples/output_4.mp4",
    "./examples/output_5.mp4",
]

t2v_examples = [
    [prompt, video] for prompt, video in zip(t2v_prompt_examples, t2v_videos_examples)
]


def create_demo():

    with gr.Blocks() as demo:
        # gr.Markdown("# Open-Sora Plan v1.5 Video Generation")
        LOGO = """
            <center><img src='https://s21.ax1x.com/2024/07/14/pk5pLBF.jpg' alt='Open-Sora Plan logo' style="width:220px; margin-bottom:1px"></center>
        """
        TITLE = """
            <div style="text-align: center; font-size: 45px; font-weight: bold; margin-bottom: 5px;">
                Open-Sora Plan🤗
            </div>
        """
        DESCRIPTION = """
            <div style="text-align: center; font-size: 16px; font-weight: bold; margin-bottom: 5px;">
                Support Chinese and English; 支持中英双语
            </div>
            <div style="text-align: center; font-size: 16px; font-weight: bold; margin-bottom: 5px;">
                Welcome to Star🌟 our <a href='https://github.com/PKU-YuanGroup/Open-Sora-Plan' target='_blank'><b>GitHub</b></a>
            </div>
        """
        gr.HTML(LOGO)
        gr.HTML(TITLE)
        gr.HTML(DESCRIPTION)

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    value="A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, along red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is dampand reflective, creating a mirror effect of thecolorful lights. Many pedestrians walk about.",
                )
                with gr.Row():
                    resolution = gr.Dropdown(
                        choices=[
                            # 720p
                            ("1280x720", "1280x720"),
                            ("720x1280", "720x1280"),
                        ],
                        value="1280x720",
                        label="Resolution",
                    )
                    video_length = gr.Dropdown(
                        label="Video Length",
                        choices=[
                            ("5s (81 frames)", 81),
                        ],
                        value=81,
                    )
                num_inference_steps = gr.Slider(
                    1, 100, value=4, step=1, label="Number of Inference Steps"
                )
                seed = gr.Number(value=-1, label="Seed (-1 for random)")
                generate_btn = gr.Button("Generate")

            with gr.Column():
                output = gr.Video(label="Generated Video")

        generate_btn.click(
            fn=lambda *inputs: generate_video(*inputs),
            inputs=[
                prompt,
                resolution,
                video_length,
                seed,
                num_inference_steps,
            ],
            outputs=output,
        )

        gr.Examples(
            examples=t2v_examples,
            inputs=[prompt, output],
            label="Prompt & Video Examples",
        )

    return demo


if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    server_name = os.getenv("SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("SERVER_PORT", "7860"))
    demo = create_demo()
    demo.launch(server_name=server_name, server_port=server_port, share=True)
    # demo.launch(share=True)