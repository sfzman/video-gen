from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from app_config import (
    ANISORA_REPO_ROOT,
    LTX_CORE_SRC,
    LTX_PIPELINES_SRC,
    MODEL_ANISORA,
    MODEL_LTX,
)
from sage_attention import patch_anisora_attention, patch_ltx_attention

logger = logging.getLogger(__name__)

_ANISORA_PIPELINES: dict[tuple[str, str, str, bool], Any] = {}
_LTX_PIPELINES: dict[tuple[str, str, str, str, float, bool], Any] = {}


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for video generation, but no CUDA device is available.")


def _insert_sys_path(path: Path) -> None:
    path_str = str(path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _normalize_seed(seed: int) -> int:
    return seed if seed >= 0 else random.randint(0, 2**31 - 1)


def _conditioning_items(params: dict[str, Any]) -> list[dict[str, Any]]:
    items = [item for item in params.get("conditioning_images", []) if item.get("path")]
    items.sort(key=lambda item: item.get("position", 0.0))
    return items


def _load_pil_images(items: list[dict[str, Any]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for item in items:
        with Image.open(item["path"]) as img:
            images.append(img.convert("RGB"))
    return images


def _get_anisora_pipeline(params: dict[str, Any]) -> Any:
    _require_cuda()
    _insert_sys_path(ANISORA_REPO_ROOT)

    import wan  # type: ignore
    from wan.configs import WAN_CONFIGS  # type: ignore

    use_sageattention = bool(params.get("anisora_use_sageattention"))
    patch_status = patch_anisora_attention() if use_sageattention else "disabled by request"
    logger.info("Anisora attention backend: %s", patch_status)

    checkpoint_dir = params["anisora_ckpt_dir"]
    low_subdir = params.get("anisora_low_subdir", "low_noise_model")
    high_subdir = params.get("anisora_high_subdir", "high_noise_model")
    cache_key = (checkpoint_dir, low_subdir, high_subdir, use_sageattention)
    if cache_key not in _ANISORA_PIPELINES:
        device_id = torch.cuda.current_device()
        _ANISORA_PIPELINES[cache_key] = wan.WanI2V(
            config=WAN_CONFIGS["i2v-A14B"],
            checkpoint_dir=checkpoint_dir,
            checkpoint_dir_lowname=low_subdir,
            checkpoint_dir_highname=high_subdir,
            device_id=device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            convert_model_dtype=False,
        )
    return _ANISORA_PIPELINES[cache_key]


@torch.inference_mode()
def _generate_anisora(params: dict[str, Any], work_dir: Path) -> tuple[str, str, dict[str, Any]]:
    _insert_sys_path(ANISORA_REPO_ROOT)

    from wan.utils.utils import save_video  # type: ignore

    items = _conditioning_items(params)
    if not items:
        raise ValueError("Anisora requires at least one conditioning image.")

    pipeline = _get_anisora_pipeline(params)
    prompt = params["prompt"].strip()
    negative_prompt = (params.get("negative_prompt") or "").strip()
    seed = _normalize_seed(int(params.get("seed", -1)))
    images = _load_pil_images(items)
    positions = [float(item["position"]) for item in items]

    video = pipeline.generate(
        prompt,
        images,
        positions,
        max_area=int(params["width"]) * int(params["height"]),
        frame_num=int(params["num_frames"]),
        shift=float(params.get("anisora_sample_shift", 5.0)),
        sample_solver=str(params.get("anisora_solver", "unipc")),
        sampling_steps=int(params["num_inference_steps"]),
        guide_scale=float(params.get("anisora_guide_scale", 3.5)),
        n_prompt=negative_prompt,
        seed=seed,
        offload_model=bool(params.get("anisora_offload_model", True)),
    )

    output_path = work_dir / f"{MODEL_ANISORA}_{seed}.mp4"
    save_video(
        tensor=video[None],
        save_file=str(output_path),
        fps=int(params["fps"]),
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    torch.cuda.synchronize()

    meta = {
        "seed": seed,
        "actual_frames": int(video.shape[1]),
        "actual_height": int(video.shape[2]),
        "actual_width": int(video.shape[3]),
    }
    message = (
        f"Anisora generation finished. Seed={seed}, output={meta['actual_width']}x{meta['actual_height']}, "
        f"frames={meta['actual_frames']}."
    )
    return str(output_path), message, meta


def _get_ltx_pipeline(params: dict[str, Any]) -> Any:
    _require_cuda()
    _insert_sys_path(LTX_CORE_SRC)
    _insert_sys_path(LTX_PIPELINES_SRC)

    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps  # type: ignore
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline  # type: ignore

    use_sageattention = bool(params.get("ltx_use_sageattention"))
    patch_status = patch_ltx_attention() if use_sageattention else "disabled by request"
    logger.info("LTX attention backend: %s", patch_status)

    checkpoint_path = params["ltx_checkpoint_path"]
    gemma_root = params["ltx_gemma_root"]
    distilled_lora_path = params["ltx_distilled_lora_path"]
    spatial_upsampler_path = params["ltx_spatial_upsampler_path"]
    distilled_lora_strength = float(params.get("ltx_distilled_lora_strength", 0.8))
    cache_key = (
        checkpoint_path,
        gemma_root,
        distilled_lora_path,
        spatial_upsampler_path,
        distilled_lora_strength,
        use_sageattention,
    )
    if cache_key not in _LTX_PIPELINES:
        distilled_lora = (
            LoraPathStrengthAndSDOps(
                distilled_lora_path,
                distilled_lora_strength,
                LTXV_LORA_COMFY_RENAMING_MAP,
            ),
        )
        _LTX_PIPELINES[cache_key] = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_root,
            loras=(),
            quantization=None,
            torch_compile=bool(params.get("ltx_compile", False)),
        )
    return _LTX_PIPELINES[cache_key]


@torch.inference_mode()
def _generate_ltx(params: dict[str, Any], work_dir: Path) -> tuple[str, str, dict[str, Any]]:
    _insert_sys_path(LTX_CORE_SRC)
    _insert_sys_path(LTX_PIPELINES_SRC)

    from ltx_core.components.guiders import MultiModalGuiderParams  # type: ignore
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number  # type: ignore
    from ltx_pipelines.utils.args import ImageConditioningInput  # type: ignore
    from ltx_pipelines.utils.media_io import encode_video  # type: ignore

    pipeline = _get_ltx_pipeline(params)
    items = _conditioning_items(params)
    seed = _normalize_seed(int(params.get("seed", -1)))
    num_frames = int(params["num_frames"])

    image_inputs = [
        ImageConditioningInput(
            path=item["path"],
            frame_idx=min(
                num_frames - 1,
                max(0, round((num_frames - 1) * float(item["position"]))),
            ),
            strength=float(item.get("strength", 1.0)),
        )
        for item in items
    ]

    tiling_config = TilingConfig.default()
    video, audio = pipeline(
        prompt=params["prompt"].strip(),
        negative_prompt=(params.get("negative_prompt") or "").strip(),
        seed=seed,
        height=int(params["height"]),
        width=int(params["width"]),
        num_frames=num_frames,
        frame_rate=float(params["fps"]),
        num_inference_steps=int(params["num_inference_steps"]),
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=float(params.get("ltx_video_cfg_scale", 3.0)),
            stg_scale=float(params.get("ltx_video_stg_scale", 1.0)),
            rescale_scale=float(params.get("ltx_video_rescale_scale", 0.7)),
            modality_scale=float(params.get("ltx_a2v_guidance_scale", 3.0)),
            skip_step=int(params.get("ltx_video_skip_step", 0)),
            stg_blocks=[28],
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=float(params.get("ltx_audio_cfg_scale", 7.0)),
            stg_scale=float(params.get("ltx_audio_stg_scale", 1.0)),
            rescale_scale=float(params.get("ltx_audio_rescale_scale", 0.7)),
            modality_scale=float(params.get("ltx_v2a_guidance_scale", 3.0)),
            skip_step=int(params.get("ltx_audio_skip_step", 0)),
            stg_blocks=[28],
        ),
        images=image_inputs,
        tiling_config=tiling_config,
        enhance_prompt=bool(params.get("ltx_enhance_prompt", False)),
        streaming_prefetch_count=(
            int(params["ltx_streaming_prefetch_count"])
            if int(params.get("ltx_streaming_prefetch_count", 0)) > 0
            else None
        ),
        max_batch_size=max(1, int(params.get("ltx_max_batch_size", 1))),
    )

    output_path = work_dir / f"{MODEL_LTX}_{seed}.mp4"
    encode_video(
        video=video,
        fps=int(params["fps"]),
        audio=audio,
        output_path=str(output_path),
        video_chunks_number=get_video_chunks_number(num_frames, tiling_config),
    )
    torch.cuda.synchronize()

    meta = {
        "seed": seed,
        "actual_frames": num_frames,
        "actual_height": int(params["height"]),
        "actual_width": int(params["width"]),
    }
    message = (
        f"LTX generation finished. Seed={seed}, output={meta['actual_width']}x{meta['actual_height']}, "
        f"frames={meta['actual_frames']}."
    )
    return str(output_path), message, meta


def generate_with_model(params: dict[str, Any], work_dir: Path) -> tuple[str, str, dict[str, Any]]:
    model_id = params["model_id"]
    work_dir.mkdir(parents=True, exist_ok=True)
    if model_id == MODEL_ANISORA:
        return _generate_anisora(params, work_dir)
    if model_id == MODEL_LTX:
        return _generate_ltx(params, work_dir)
    raise ValueError(f"Unsupported model_id: {model_id}")
