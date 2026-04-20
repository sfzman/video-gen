from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
TASK_QUEUE_DIR = ROOT_DIR / "task_queue"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"

MODEL_ANISORA = "anisora-v3.2"
MODEL_LTX = "ltx-2.3"
MODEL_CHOICES = [MODEL_ANISORA, MODEL_LTX]

ANISORA_REPO_ROOT = Path("/Users/fangzhou/Workspace/Index-anisora/anisoraV3.2")
LTX_PIPELINES_SRC = Path("/Users/fangzhou/Workspace/LTX-2/packages/ltx-pipelines/src")
LTX_CORE_SRC = Path("/Users/fangzhou/Workspace/LTX-2/packages/ltx-core/src")

DEFAULT_ANISORA_NEGATIVE_PROMPT = (
    "oversaturated, overexposed, static, blurry details, subtitles, style reference, artwork, painting, "
    "frame freeze, gray overall tone, worst quality, low quality, jpeg artifacts, ugly, mutilated, extra fingers, "
    "bad hands, bad face, deformed limbs, fused fingers, cluttered background, extra legs, many background people"
)

DEFAULT_LTX_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy "
    "texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial "
    "features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, "
    "artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background clutter, "
    "harsh shadows, inconsistent lighting, color banding, stylized filters, AI artifacts"
)

DEFAULT_ANISORA_CKPT_DIR = os.getenv("ANISORA_CKPT_DIR", "")
DEFAULT_ANISORA_LOW_SUBDIR = os.getenv("ANISORA_LOW_SUBDIR", "low_noise_model")
DEFAULT_ANISORA_HIGH_SUBDIR = os.getenv("ANISORA_HIGH_SUBDIR", "high_noise_model")

DEFAULT_LTX_CHECKPOINT_PATH = os.getenv("LTX_CHECKPOINT_PATH", "")
DEFAULT_LTX_GEMMA_ROOT = os.getenv("LTX_GEMMA_ROOT", "")
DEFAULT_LTX_DISTILLED_LORA_PATH = os.getenv("LTX_DISTILLED_LORA_PATH", "")
DEFAULT_LTX_SPATIAL_UPSAMPLER_PATH = os.getenv("LTX_SPATIAL_UPSAMPLER_PATH", "")

DEFAULT_SERVER_NAME = os.getenv("VIDEO_GEN_SERVER_NAME", "127.0.0.1")
DEFAULT_SERVER_PORT = int(os.getenv("VIDEO_GEN_SERVER_PORT", "7860"))

ASPECT_RATIO_PRESETS: dict[str, tuple[int, int]] = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "4:3": (1024, 768),
    "3:4": (768, 1024),
    "1:1": (960, 960),
    "832x480": (832, 480),
    "480x832": (480, 832),
}


def ensure_runtime_dirs() -> None:
    TASK_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
