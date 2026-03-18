# Training Guide: Wan2.2-TI2V-5B LoRA from Scratch

This guide covers every step from a fresh clone to a trained LoRA model.

---

## 1. Environment Setup

### Prerequisites
- Linux (Ubuntu 22.04 recommended)
- Python 3.8+
- CUDA 12.x compatible GPU (minimum ~24 GB VRAM recommended for this 5B model)
- `pip` package manager
- `conda` or `mamba` package manager

### Create Conda Environment

```bash
conda create -n diffsynth python=3.10 -y
conda activate diffsynth
```

### Install DiffSynth-Studio

```bash
cd /home/duzl/hlc/DiffSynth-Studio
pip install setuptools
pip install -e .
```

### Install Dependencies

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Key packages installed:
- `torch>=2.0.0`, `torchvision`
- `cupy-cuda12x` — CUDA 12.x bindings
- `accelerate` — multi-GPU / distributed training
- `transformers`, `safetensors`, `sentencepiece`, `protobuf`
- `imageio[ffmpeg]` — video reading/writing
- `modelscope` — model downloading
- `peft` — LoRA injection (required by trainer, not in requirements.txt)

Install `peft` separately if not already present:
```bash
pip install peft
```

### Configure Accelerate

```bash
accelerate config
```

For a single-GPU setup, select:
- "This machine"
- "No distributed training" (or "multi-GPU" if you have multiple)
- Mixed precision: `bf16` (recommended for Wan2.2)

---

## 2. Model Download

The training script downloads model weights automatically from ModelScope on first run via the `model_id_with_origin_paths` argument. The three components it fetches from `Wan-AI/Wan2.2-TI2V-5B` are:

| File pattern | Description |
|---|---|
| `diffusion_pytorch_model*.safetensors` | DiT (the diffusion transformer) |
| `models_t5_umt5-xxl-enc-bf16.pth` | T5 text encoder |
| `Wan2.2_VAE.pth` | VAE |

Downloaded files are cached to `./models/` by default.

**If you are on a machine without internet access**, download these files manually from:
- https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B

Place them under `models/Wan-AI/Wan2.2-TI2V-5B/` and replace the `--model_id_with_origin_paths` argument with `--model_paths` pointing to local paths (see training script comments).

---

## 3. Prepare Training Data

The training script expects a dataset directory with **video files** and a **CSV metadata file**.

### Directory Structure

```
data/example_video_dataset/
├── video1.mp4
├── video2.mp4
├── ...
└── metadata.csv
```

### metadata.csv Format

The CSV must have at minimum two columns: `video` and `prompt`.

```csv
video,prompt
video1.mp4,a small town at sunset with lights reflecting on a river
video2.mp4,a dog running in a green meadow under blue sky
```

- `video` — filename relative to `--dataset_base_path`
- `prompt` — text description of the video content

### Video Requirements

Since the training script uses `--extra_inputs "input_image"`, this model is **Text+Image-to-Video (TI2V)**. The first frame of each video will be extracted automatically and used as `input_image`.

- **Format**: `.mp4`, `.avi`, `.mov`, `.wmv`, `.mkv`, `.flv`, `.webm`
- **Resolution**: The script is configured to train at **480×832** (height×width). Videos will be center-cropped and resized automatically.
- **Length**: At least **49 frames** recommended (matches `--num_frames 49`). Videos with fewer frames will use all available frames.
- **Minimum dataset size**: Even 1 video works for a quick test, but 10–100+ clips give better results. With `--dataset_repeat 100`, each video is seen 100 times per epoch.

### Quick Test Dataset

To test the pipeline with a single video:

```bash
mkdir -p data/example_video_dataset

# copy any short .mp4 into the folder
cp /path/to/your/video.mp4 data/example_video_dataset/video1.mp4

# create metadata
echo "video,prompt" > data/example_video_dataset/metadata.csv
echo "video1.mp4,your description of the video here" >> data/example_video_dataset/metadata.csv
```

---

## 4. Run Training

From the **repo root** (`/home/duzl/hlc/DiffSynth-Studio`):

```bash
bash examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B.sh
```

The script runs:

```bash
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image"
```

### Key Parameters Explained

| Parameter | Value | Meaning |
|---|---|---|
| `--height / --width` | 480 / 832 | Training resolution |
| `--num_frames` | 49 | Frames per video sample |
| `--dataset_repeat` | 100 | Dataset repetitions per epoch (inflate small datasets) |
| `--num_epochs` | 5 | Training epochs |
| `--lora_rank` | 32 | LoRA rank — higher = more capacity, more VRAM |
| `--lora_target_modules` | `q,k,v,o,ffn.0,ffn.2` | Attention + FFN layers get LoRA adapters |
| `--learning_rate` | 1e-4 | AdamW learning rate |
| `--extra_inputs` | `input_image` | Enables TI2V mode (first frame as conditioning image) |

### Output

After each epoch a checkpoint is saved to:
```
models/train/Wan2.2-TI2V-5B_lora/epoch-0.safetensors
models/train/Wan2.2-TI2V-5B_lora/epoch-1.safetensors
...
models/train/Wan2.2-TI2V-5B_lora/epoch-4.safetensors
```

---

## 5. VRAM Considerations

| Situation | Suggested Change |
|---|---|
| OOM at training | Add `--use_gradient_checkpointing_offload` flag to the script |
| Reduce memory further | Lower `--lora_rank` (e.g., 16 or 8) |
| Multiple GPUs | `accelerate config` → select multi-GPU; no script change needed |

---

## 6. Validate / Run Inference with Trained LoRA

After training, use the provided validation script:

```bash
python examples/wanvideo/model_training/validate_lora/Wan2.2-TI2V-5B.py
```

This script:
1. Loads the base `Wan-AI/Wan2.2-TI2V-5B` model
2. Loads your LoRA from `models/train/Wan2.2-TI2V-5B_lora/epoch-4.safetensors`
3. Runs inference with `input_image` taken from the first frame of your dataset video
4. Saves the result to `video_Wan2.2-TI2V-5B.mp4`

To use a **different epoch or prompt**, edit the validation script directly:
```python
pipe.load_lora(pipe.dit, "models/train/Wan2.2-TI2V-5B_lora/epoch-2.safetensors", alpha=1)
```

---

## 7. Step-by-Step Checklist

- [ ] `pip install -e .` — install DiffSynth-Studio
- [ ] `pip install -r requirements.txt` — install dependencies
- [ ] `pip install peft` — install LoRA library
- [ ] `accelerate config` — configure accelerate
- [ ] Create `data/example_video_dataset/` with videos + `metadata.csv`
- [ ] `bash examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B.sh` — start training
- [ ] Wait for training to finish (checkpoints saved per epoch)
- [ ] `python examples/wanvideo/model_training/validate_lora/Wan2.2-TI2V-5B.py` — validate
