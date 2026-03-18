import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:5",
    model_configs=[
        ModelConfig(path="models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(path=[
            "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
            "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
            "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors",
        ], offload_device="cpu"),
        ModelConfig(path="models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()

# # Text-to-video
# video = pipe(
#     prompt="A man walking from left to right in a city street. A dog walking from right to left in the same street. The dog and man meet at the center of the street.",
#     # negative_prompt="",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     num_frames=121,
# )
# save_video(video, "video1.mp4", fps=15, quality=5)

# Image-to-video
# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=["data/examples/wan/cat_fightning.jpg"]
# )
# input_image = Image.open("data/examples/wan/cat_fightning.jpg").resize((1248, 704))
# video = pipe(
#     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "video2.mp4", fps=15, quality=5)


"""
Generate a realistic future video from the ego vehicle's front camera. The colored trajectory overlaid on the image represents the intended driving path of the ego car over time. Predict the future scene by following this trajectory strictly.
"""
input_image = Image.open("/home/duzl/hlc/DiffSynth-Studio/image.png").resize((1248, 704))

video = pipe(
    prompt = """
            Generate a realistic future video from the ego vehicle's front camera.
            The colored trajectory overlaid on the image represents the intended driving path of the ego car.
            The color transitions from cold to warm indicate temporal progression, from current position to future waypoints.

            Predict the future scene by following this trajectory strictly.
            Ensure the ego vehicle motion and camera movement align smoothly with the trajectory over time.

            Maintain scene consistency, including surrounding vehicles, pedestrians, buildings, and traffic lights.
            Ensure natural motion of all dynamic objects and realistic temporal continuity.
            """,
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    # negative_prompt 最好不要改，因为训练的时候全用的这个。
    # negative_prompt = """
    #         blurry, low resolution, pixelated, noisy,
    #         distorted geometry, warped cars, stretched objects,
    #         flickering, temporal inconsistency, jittery motion, frame jumps,
    #         camera static, no camera movement, wrong camera motion,
    #         trajectory ignored, off-path motion, inconsistent ego motion,
    #         moving backward, reverse driving, backward motion, negative speed,
    #         duplicate vehicles, disappearing objects, popping artifacts,
    #         unrealistic lighting, overexposed, underexposed, still
    #         """,
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "video_front_no_nega_prompt.mp4", fps=15, quality=5)
