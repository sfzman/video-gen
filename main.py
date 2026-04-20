from __future__ import annotations

import atexit
import signal
import threading
from pathlib import Path
from typing import Any

try:
    import gradio as gr
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    gr = None
    _GRADIO_IMPORT_ERROR = exc
else:
    _GRADIO_IMPORT_ERROR = None

from app_config import (
    ASPECT_RATIO_PRESETS,
    DEFAULT_ANISORA_CKPT_DIR,
    DEFAULT_ANISORA_HIGH_SUBDIR,
    DEFAULT_ANISORA_LOW_SUBDIR,
    DEFAULT_ANISORA_NEGATIVE_PROMPT,
    DEFAULT_LTX_CHECKPOINT_PATH,
    DEFAULT_LTX_DISTILLED_LORA_PATH,
    DEFAULT_LTX_GEMMA_ROOT,
    DEFAULT_LTX_NEGATIVE_PROMPT,
    DEFAULT_LTX_SPATIAL_UPSAMPLER_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SERVER_NAME,
    DEFAULT_SERVER_PORT,
    MODEL_ANISORA,
    MODEL_CHOICES,
    MODEL_LTX,
    ensure_runtime_dirs,
)
from preview_utils import delete_task_files, load_task_preview, refresh_preview_list
from sage_attention import detect_sageattention
from task_queue import enqueue_task, kill_worker_subprocess, start_task_worker, stop_task_worker

_shutdown_lock = threading.Lock()
_shutdown_done = False


def _graceful_shutdown(reason: str = "unknown") -> None:
    global _shutdown_done
    with _shutdown_lock:
        if _shutdown_done:
            return
        _shutdown_done = True
    try:
        stop_task_worker()
    except Exception:
        pass
    print(f"[shutdown] cleanup finished ({reason})")


def _signal_handler(signum: int, frame: Any) -> None:  # pragma: no cover - signal path
    _graceful_shutdown(reason=signal.Signals(signum).name)
    raise SystemExit(0)


for _sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_sig, _signal_handler)
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, _signal_handler)
atexit.register(_graceful_shutdown, reason="atexit")


def update_dimensions(aspect_ratio: str) -> tuple[int, int]:
    width, height = ASPECT_RATIO_PRESETS.get(aspect_ratio, (1280, 720))
    return height, width


def get_runtime_status() -> str:
    sage_status = detect_sageattention()
    try:
        import torch

        if not torch.cuda.is_available():
            return f"CUDA unavailable. SageAttention: {sage_status}"
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        allocated = torch.cuda.memory_allocated(device_index) / 1024**3
        reserved = torch.cuda.memory_reserved(device_index) / 1024**3
        return (
            f"CUDA device: {props.name}\n"
            f"Allocated: {allocated:.2f} GiB\n"
            f"Reserved: {reserved:.2f} GiB\n"
            f"SageAttention: {sage_status}"
        )
    except Exception as exc:
        return f"Runtime status unavailable: {exc}. SageAttention: {sage_status}"


def _normalize_output_dir(output_dir: str) -> str:
    if not output_dir:
        return str(DEFAULT_OUTPUT_DIR)
    return str(Path(output_dir).expanduser())


def _build_conditioning_images(*items: Any) -> list[dict[str, Any]]:
    image_specs: list[dict[str, Any]] = []
    flat = list(items)
    for index in range(0, len(flat), 3):
        path = flat[index]
        position = flat[index + 1]
        strength = flat[index + 2]
        if not path:
            continue
        image_specs.append(
            {
                "path": str(path),
                "position": float(position),
                "strength": float(strength),
            }
        )
    return image_specs


def _validate_request(params: dict[str, Any]) -> None:
    if not params.get("prompt", "").strip():
        raise ValueError("Prompt is required.")
    num_frames = int(params["num_frames"])
    if (num_frames - 1) % 8 != 0:
        raise ValueError("num_frames must satisfy 8k+1 for both Anisora and LTX pipelines.")

    model_id = params["model_id"]
    conditioning_images = params.get("conditioning_images") or []

    if model_id == MODEL_ANISORA:
        if not conditioning_images:
            raise ValueError("Anisora requires at least one conditioning image.")
        if not params.get("anisora_ckpt_dir"):
            raise ValueError("Anisora checkpoint root is required.")
    elif model_id == MODEL_LTX:
        if int(params["height"]) % 64 != 0 or int(params["width"]) % 64 != 0:
            raise ValueError("LTX two-stage pipeline requires width and height divisible by 64.")
        required_keys = [
            "ltx_checkpoint_path",
            "ltx_gemma_root",
            "ltx_distilled_lora_path",
            "ltx_spatial_upsampler_path",
        ]
        missing = [key for key in required_keys if not params.get(key)]
        if missing:
            raise ValueError(f"Missing LTX settings: {', '.join(missing)}")
    else:
        raise ValueError(f"Unsupported model: {model_id}")


def handle_submit(
    model_id: str,
    prompt: str,
    negative_prompt: str,
    image_1: str | None,
    image_1_position: float,
    image_1_strength: float,
    image_2: str | None,
    image_2_position: float,
    image_2_strength: float,
    image_3: str | None,
    image_3_position: float,
    image_3_strength: float,
    width: float,
    height: float,
    num_frames: float,
    fps: float,
    seed: float,
    num_inference_steps: float,
    save_folder_path: str,
    anisora_ckpt_dir: str,
    anisora_low_subdir: str,
    anisora_high_subdir: str,
    anisora_guide_scale: float,
    anisora_sample_shift: float,
    anisora_offload_model: bool,
    anisora_use_sageattention: bool,
    ltx_checkpoint_path: str,
    ltx_gemma_root: str,
    ltx_distilled_lora_path: str,
    ltx_spatial_upsampler_path: str,
    ltx_distilled_lora_strength: float,
    ltx_video_cfg_scale: float,
    ltx_audio_cfg_scale: float,
    ltx_video_stg_scale: float,
    ltx_audio_stg_scale: float,
    ltx_video_rescale_scale: float,
    ltx_audio_rescale_scale: float,
    ltx_a2v_guidance_scale: float,
    ltx_v2a_guidance_scale: float,
    ltx_streaming_prefetch_count: float,
    ltx_max_batch_size: float,
    ltx_use_sageattention: bool,
    ltx_compile: bool,
) -> str:
    params = {
        "model_id": model_id,
        "prompt": prompt,
        "negative_prompt": negative_prompt.strip() or (
            DEFAULT_ANISORA_NEGATIVE_PROMPT if model_id == MODEL_ANISORA else DEFAULT_LTX_NEGATIVE_PROMPT
        ),
        "conditioning_images": _build_conditioning_images(
            image_1,
            image_1_position,
            image_1_strength,
            image_2,
            image_2_position,
            image_2_strength,
            image_3,
            image_3_position,
            image_3_strength,
        ),
        "width": int(width),
        "height": int(height),
        "num_frames": int(num_frames),
        "fps": int(fps),
        "seed": int(seed),
        "num_inference_steps": int(num_inference_steps),
        "save_folder_path": _normalize_output_dir(save_folder_path),
        "anisora_ckpt_dir": anisora_ckpt_dir.strip(),
        "anisora_low_subdir": anisora_low_subdir.strip() or "low_noise_model",
        "anisora_high_subdir": anisora_high_subdir.strip() or "high_noise_model",
        "anisora_guide_scale": float(anisora_guide_scale),
        "anisora_sample_shift": float(anisora_sample_shift),
        "anisora_offload_model": bool(anisora_offload_model),
        "anisora_use_sageattention": bool(anisora_use_sageattention),
        "ltx_checkpoint_path": ltx_checkpoint_path.strip(),
        "ltx_gemma_root": ltx_gemma_root.strip(),
        "ltx_distilled_lora_path": ltx_distilled_lora_path.strip(),
        "ltx_spatial_upsampler_path": ltx_spatial_upsampler_path.strip(),
        "ltx_distilled_lora_strength": float(ltx_distilled_lora_strength),
        "ltx_video_cfg_scale": float(ltx_video_cfg_scale),
        "ltx_audio_cfg_scale": float(ltx_audio_cfg_scale),
        "ltx_video_stg_scale": float(ltx_video_stg_scale),
        "ltx_audio_stg_scale": float(ltx_audio_stg_scale),
        "ltx_video_rescale_scale": float(ltx_video_rescale_scale),
        "ltx_audio_rescale_scale": float(ltx_audio_rescale_scale),
        "ltx_a2v_guidance_scale": float(ltx_a2v_guidance_scale),
        "ltx_v2a_guidance_scale": float(ltx_v2a_guidance_scale),
        "ltx_streaming_prefetch_count": int(ltx_streaming_prefetch_count),
        "ltx_max_batch_size": int(ltx_max_batch_size),
        "ltx_use_sageattention": bool(ltx_use_sageattention),
        "ltx_compile": bool(ltx_compile),
    }
    try:
        _validate_request(params)
        return enqueue_task(params)
    except Exception as exc:
        return f"Submit failed: {exc}"


def create_preview_tab() -> None:
    with gr.Column():
        gr.Markdown("## Generated Videos")
        preview_output_dir = gr.Textbox(label="Output directory", value=str(DEFAULT_OUTPUT_DIR), scale=3)
        refresh_btn = gr.Button("Refresh")

        initial_gallery, _, initial_tasks = refresh_preview_list(str(DEFAULT_OUTPUT_DIR))
        preview_tasks_state = gr.State(initial_tasks)
        selected_task_index_state = gr.State(None)

        task_gallery = gr.Gallery(
            label="Completed tasks",
            value=initial_gallery,
            columns=4,
            rows=2,
            height="auto",
            allow_preview=True,
            interactive=True,
        )
        preview_video = gr.Video(label="Video preview", height=420)
        params_summary = gr.Markdown(value="Select a thumbnail to inspect its task parameters.")
        task_json_display = gr.Code(label="Task JSON", language="json", lines=18, interactive=False)
        delete_btn = gr.Button("Delete selected task", variant="stop")
        action_status = gr.Textbox(label="Action status", interactive=False)

        def refresh_list(output_dir: str) -> tuple[gr.Update, list[dict[str, Any]]]:
            gallery_items, _, tasks = refresh_preview_list(output_dir)
            return gr.update(value=gallery_items), tasks

        refresh_btn.click(fn=refresh_list, inputs=[preview_output_dir], outputs=[task_gallery, preview_tasks_state])
        preview_output_dir.submit(fn=refresh_list, inputs=[preview_output_dir], outputs=[task_gallery, preview_tasks_state])

        def load_preview(evt: gr.SelectData, tasks: list[dict[str, Any]], output_dir: str) -> tuple[str | None, str, str, int | None]:
            if evt is None or evt.index is None or not tasks:
                return None, "No task selected.", "{}", None
            video_path, summary, task_json = load_task_preview(evt.index, output_dir, cached_tasks=tasks)
            return video_path, summary, task_json, evt.index

        task_gallery.select(
            fn=load_preview,
            inputs=[preview_tasks_state, preview_output_dir],
            outputs=[preview_video, params_summary, task_json_display, selected_task_index_state],
        )

        def delete_selected_task(
            tasks: list[dict[str, Any]],
            selected_index: int | None,
            output_dir: str,
        ) -> tuple[gr.Update, list[dict[str, Any]], int | None, gr.Update, gr.Update, gr.Update, str]:
            if not tasks or selected_index is None or selected_index >= len(tasks):
                return gr.update(), tasks, selected_index, gr.update(), gr.update(), gr.update(), "No valid task selected."
            success, message = delete_task_files(tasks[selected_index])
            gallery_items, _, new_tasks = refresh_preview_list(output_dir)
            return (
                gr.update(value=gallery_items),
                new_tasks,
                None if success else selected_index,
                gr.update(value=None) if success else gr.update(),
                gr.update(value="Select a thumbnail to inspect its task parameters.") if success else gr.update(),
                gr.update(value="{}") if success else gr.update(),
                message,
            )

        delete_btn.click(
            fn=delete_selected_task,
            inputs=[preview_tasks_state, selected_task_index_state, preview_output_dir],
            outputs=[
                task_gallery,
                preview_tasks_state,
                selected_task_index_state,
                preview_video,
                params_summary,
                task_json_display,
                action_status,
            ],
        )


def create_interface() -> gr.Blocks:
    if gr is None:
        raise RuntimeError(
            "Gradio is not installed in the current Python environment. "
            "Install the packages from requirements.txt before launching the UI."
        ) from _GRADIO_IMPORT_ERROR
    with gr.Blocks(title="Video Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Async Video Generator")
        gr.Markdown(
            "This rebuild keeps only native Anisora V3.2 and LTX 2.3 inference backends, plus asynchronous queueing and browser preview."
        )

        with gr.Row():
            runtime_status = gr.Textbox(label="Runtime status", value=get_runtime_status(), lines=4, interactive=False)
            refresh_runtime_btn = gr.Button("Refresh runtime")
            kill_worker_btn = gr.Button("Stop current worker", variant="stop")

        refresh_runtime_btn.click(fn=get_runtime_status, outputs=[runtime_status])
        kill_worker_btn.click(fn=kill_worker_subprocess, outputs=[runtime_status])

        with gr.Tabs():
            with gr.TabItem("Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_id = gr.Dropdown(label="Model", choices=MODEL_CHOICES, value=MODEL_ANISORA)
                        prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the video you want to generate.")
                        negative_prompt = gr.Textbox(label="Negative prompt", lines=3, placeholder="Optional negative prompt.")

                        gr.Markdown("### Conditioning images")
                        image_1 = gr.Image(label="Image 1", type="filepath", height=220)
                        with gr.Row():
                            image_1_position = gr.Slider(label="Image 1 position", minimum=0.0, maximum=1.0, value=0.0, step=0.01)
                            image_1_strength = gr.Slider(label="Image 1 strength", minimum=0.0, maximum=2.0, value=1.0, step=0.05)

                        image_2 = gr.Image(label="Image 2", type="filepath", height=220)
                        with gr.Row():
                            image_2_position = gr.Slider(label="Image 2 position", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                            image_2_strength = gr.Slider(label="Image 2 strength", minimum=0.0, maximum=2.0, value=1.0, step=0.05)

                        image_3 = gr.Image(label="Image 3", type="filepath", height=220)
                        with gr.Row():
                            image_3_position = gr.Slider(label="Image 3 position", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                            image_3_strength = gr.Slider(label="Image 3 strength", minimum=0.0, maximum=2.0, value=1.0, step=0.05)

                    with gr.Column(scale=1):
                        aspect_ratio = gr.Dropdown(label="Aspect ratio preset", choices=list(ASPECT_RATIO_PRESETS.keys()), value="16:9")
                        with gr.Row():
                            width = gr.Number(label="Width", value=1280, precision=0)
                            height = gr.Number(label="Height", value=720, precision=0)
                        with gr.Row():
                            num_frames = gr.Number(label="Frames", value=121, precision=0)
                            fps = gr.Number(label="FPS", value=24, precision=0)
                        with gr.Row():
                            seed = gr.Number(label="Seed", value=-1, precision=0)
                            num_inference_steps = gr.Number(label="Inference steps", value=30, precision=0)
                        save_folder_path = gr.Textbox(label="Output directory", value=str(DEFAULT_OUTPUT_DIR))

                        aspect_ratio.change(fn=update_dimensions, inputs=[aspect_ratio], outputs=[height, width])

                        with gr.Accordion("Anisora settings", open=True):
                            anisora_ckpt_dir = gr.Textbox(label="Checkpoint root", value=DEFAULT_ANISORA_CKPT_DIR)
                            with gr.Row():
                                anisora_low_subdir = gr.Textbox(label="Low-noise subdir", value=DEFAULT_ANISORA_LOW_SUBDIR)
                                anisora_high_subdir = gr.Textbox(label="High-noise subdir", value=DEFAULT_ANISORA_HIGH_SUBDIR)
                            with gr.Row():
                                anisora_guide_scale = gr.Slider(label="Guide scale", minimum=1.0, maximum=8.0, value=3.5, step=0.1)
                                anisora_sample_shift = gr.Slider(label="Sample shift", minimum=0.0, maximum=10.0, value=5.0, step=0.1)
                            with gr.Row():
                                anisora_offload_model = gr.Checkbox(label="Offload model", value=True)
                                anisora_use_sageattention = gr.Checkbox(label="Try SageAttention", value=False)
                            gr.Markdown(
                                "Anisora output size follows the first conditioning image aspect ratio; width and height only define the target area."
                            )

                        with gr.Accordion("LTX 2.3 settings", open=False):
                            ltx_checkpoint_path = gr.Textbox(label="Checkpoint path", value=DEFAULT_LTX_CHECKPOINT_PATH)
                            ltx_gemma_root = gr.Textbox(label="Gemma root", value=DEFAULT_LTX_GEMMA_ROOT)
                            ltx_distilled_lora_path = gr.Textbox(label="Distilled LoRA path", value=DEFAULT_LTX_DISTILLED_LORA_PATH)
                            ltx_spatial_upsampler_path = gr.Textbox(label="Spatial upsampler path", value=DEFAULT_LTX_SPATIAL_UPSAMPLER_PATH)
                            ltx_distilled_lora_strength = gr.Slider(label="Distilled LoRA strength", minimum=0.0, maximum=2.0, value=0.8, step=0.05)
                            with gr.Row():
                                ltx_video_cfg_scale = gr.Slider(label="Video CFG", minimum=1.0, maximum=10.0, value=3.0, step=0.1)
                                ltx_audio_cfg_scale = gr.Slider(label="Audio CFG", minimum=1.0, maximum=12.0, value=7.0, step=0.1)
                            with gr.Row():
                                ltx_video_stg_scale = gr.Slider(label="Video STG", minimum=0.0, maximum=3.0, value=1.0, step=0.1)
                                ltx_audio_stg_scale = gr.Slider(label="Audio STG", minimum=0.0, maximum=3.0, value=1.0, step=0.1)
                            with gr.Row():
                                ltx_video_rescale_scale = gr.Slider(label="Video rescale", minimum=0.0, maximum=2.0, value=0.7, step=0.05)
                                ltx_audio_rescale_scale = gr.Slider(label="Audio rescale", minimum=0.0, maximum=2.0, value=0.7, step=0.05)
                            with gr.Row():
                                ltx_a2v_guidance_scale = gr.Slider(label="A2V guidance", minimum=1.0, maximum=6.0, value=3.0, step=0.1)
                                ltx_v2a_guidance_scale = gr.Slider(label="V2A guidance", minimum=1.0, maximum=6.0, value=3.0, step=0.1)
                            with gr.Row():
                                ltx_streaming_prefetch_count = gr.Number(label="Streaming prefetch", value=0, precision=0)
                                ltx_max_batch_size = gr.Number(label="Max batch size", value=1, precision=0)
                            with gr.Row():
                                ltx_use_sageattention = gr.Checkbox(label="Try SageAttention", value=False)
                                ltx_compile = gr.Checkbox(label="torch.compile", value=False)

                        submit_btn = gr.Button("Submit async task", variant="primary")
                        submission_status = gr.Textbox(label="Submission status", lines=6, interactive=False)

                        submit_btn.click(
                            fn=handle_submit,
                            inputs=[
                                model_id,
                                prompt,
                                negative_prompt,
                                image_1,
                                image_1_position,
                                image_1_strength,
                                image_2,
                                image_2_position,
                                image_2_strength,
                                image_3,
                                image_3_position,
                                image_3_strength,
                                width,
                                height,
                                num_frames,
                                fps,
                                seed,
                                num_inference_steps,
                                save_folder_path,
                                anisora_ckpt_dir,
                                anisora_low_subdir,
                                anisora_high_subdir,
                                anisora_guide_scale,
                                anisora_sample_shift,
                                anisora_offload_model,
                                anisora_use_sageattention,
                                ltx_checkpoint_path,
                                ltx_gemma_root,
                                ltx_distilled_lora_path,
                                ltx_spatial_upsampler_path,
                                ltx_distilled_lora_strength,
                                ltx_video_cfg_scale,
                                ltx_audio_cfg_scale,
                                ltx_video_stg_scale,
                                ltx_audio_stg_scale,
                                ltx_video_rescale_scale,
                                ltx_audio_rescale_scale,
                                ltx_a2v_guidance_scale,
                                ltx_v2a_guidance_scale,
                                ltx_streaming_prefetch_count,
                                ltx_max_batch_size,
                                ltx_use_sageattention,
                                ltx_compile,
                            ],
                            outputs=[submission_status],
                        )

            with gr.TabItem("Preview"):
                create_preview_tab()

    return demo


if __name__ == "__main__":
    ensure_runtime_dirs()
    start_task_worker()
    app = create_interface()
    app.launch(server_name=DEFAULT_SERVER_NAME, server_port=DEFAULT_SERVER_PORT, show_api=False)
