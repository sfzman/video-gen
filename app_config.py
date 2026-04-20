from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def _parse_dotenv_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        quote = value[0]
        value = value[1:-1]
        if quote == '"':
            value = (
                value.replace(r"\\", "\\")
                .replace(r"\n", "\n")
                .replace(r"\r", "\r")
                .replace(r"\t", "\t")
                .replace(r"\"", '"')
            )
    return value


def load_dotenv_file(dotenv_path: Path = ROOT_DIR / ".env") -> None:
    if not dotenv_path.is_file():
        return

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        if "=" not in stripped:
            continue

        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _parse_dotenv_value(raw_value)


def _env_path(name: str, default: Path | str) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


load_dotenv_file()

TASK_QUEUE_DIR = _env_path("VIDEO_GEN_TASK_QUEUE_DIR", ROOT_DIR / "task_queue")
DEFAULT_OUTPUT_DIR = _env_path("VIDEO_GEN_OUTPUT_DIR", ROOT_DIR / "outputs")

MODEL_ANISORA = "anisora-v3.2"
MODEL_LTX = "ltx-2.3"
MODEL_CHOICES = [MODEL_ANISORA, MODEL_LTX]

ANISORA_REPO_ROOT = _env_path("ANISORA_REPO_ROOT", "/Users/fangzhou/Workspace/Index-anisora/anisoraV3.2")
LTX_PIPELINES_SRC = _env_path("LTX_PIPELINES_SRC", "/Users/fangzhou/Workspace/LTX-2/packages/ltx-pipelines/src")
LTX_CORE_SRC = _env_path("LTX_CORE_SRC", "/Users/fangzhou/Workspace/LTX-2/packages/ltx-core/src")

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
