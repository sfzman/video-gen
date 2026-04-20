from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


def scan_generation_directories(output_dir: str = "./outputs") -> list[dict[str, Any]]:
    output_path = Path(output_dir).expanduser().resolve()
    if not output_path.exists():
        return []

    tasks: list[dict[str, Any]] = []
    for generation_dir in sorted(output_path.glob("generation_*"), reverse=True):
        if not generation_dir.is_dir():
            continue
        for task_json_file in generation_dir.rglob("task_*.json"):
            try:
                with open(task_json_file, "r", encoding="utf-8") as handle:
                    task_data = json.load(handle)
            except Exception:
                continue

            result = task_data.get("result") or {}
            moved_video = result.get("moved_video")
            video_path = moved_video if moved_video and Path(moved_video).exists() else None
            if video_path is None:
                candidates = list(task_json_file.parent.glob("*.mp4"))
                if candidates:
                    video_path = str(candidates[0])

            tasks.append(
                {
                    "generation_dir": str(generation_dir),
                    "task_id": task_data.get("id", "unknown"),
                    "created_at": task_data.get("created_at"),
                    "finished_at": task_data.get("finished_at"),
                    "status": task_data.get("status"),
                    "video_path": video_path,
                    "task_data": task_data,
                }
            )
    return tasks


def _is_valid_image_file(image_path: str) -> bool:
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return False
    try:
        with open(path, "rb") as handle:
            if not handle.read(1):
                return False
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def extract_video_thumbnail(video_path: str, output_path: str | None = None) -> str | None:
    if not video_path or not Path(video_path).exists():
        return None
    try:
        if output_path is None:
            video_file = Path(video_path)
            output_path = str(video_file.parent / f"{video_file.stem}_thumb.jpg")
        output_obj = Path(output_path)
        if output_obj.exists() and _is_valid_image_file(str(output_obj)):
            return str(output_obj.resolve())
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-vf",
                "scale=320:-1",
                "-frames:v",
                "1",
                "-y",
                output_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        if _is_valid_image_file(output_path):
            return str(Path(output_path).resolve())
        return None
    except Exception:
        return None


def get_task_params_summary(task_data: dict[str, Any]) -> str:
    params = (task_data or {}).get("params", {})
    if not params:
        return "No parameter summary available."

    lines = [
        "### Parameters",
        f"- Model: {params.get('model_id', 'n/a')}",
        f"- Resolution: {params.get('width', 'n/a')} x {params.get('height', 'n/a')}",
        f"- Frames: {params.get('num_frames', 'n/a')}",
        f"- FPS: {params.get('fps', 'n/a')}",
        f"- Steps: {params.get('num_inference_steps', 'n/a')}",
        f"- Seed: {params.get('seed', 'n/a')}",
    ]

    conditioning_images = params.get("conditioning_images") or []
    if conditioning_images:
        lines.append("\n### Conditioning Images")
        for index, image in enumerate(conditioning_images, start=1):
            lines.append(
                f"- #{index}: pos={image.get('position', 0.0):.2f}, strength={image.get('strength', 1.0):.2f}, "
                f"file={Path(image.get('path', '')).name}"
            )

    prompt = params.get("prompt")
    negative_prompt = params.get("negative_prompt")
    if prompt:
        lines.append("\n### Prompt")
        lines.append(prompt)
    if negative_prompt:
        lines.append("\n### Negative Prompt")
        lines.append(negative_prompt)

    if task_data.get("duration_seconds") is not None:
        lines.append("\n### Runtime")
        lines.append(f"- Duration: {task_data['duration_seconds']} seconds")
    if task_data.get("finished_at"):
        lines.append(f"- Finished at: {task_data['finished_at']}")

    return "\n".join(lines)


def refresh_preview_list(output_dir: str = "./outputs") -> tuple[list[tuple[str, str]], int | None, list[dict[str, Any]]]:
    tasks = scan_generation_directories(output_dir)
    if not tasks:
        return [], None, []

    gallery_items: list[tuple[str, str]] = []
    visible_tasks: list[dict[str, Any]] = []
    for task in tasks:
        video_path = task.get("video_path")
        task_id = task.get("task_id", "unknown")
        generation_dir = task.get("generation_dir", "")
        caption = Path(generation_dir).name or task_id

        thumbnail_path = extract_video_thumbnail(video_path) if video_path else None
        if not thumbnail_path:
            placeholder = Image.new("RGB", (320, 180), color=(80, 80, 80))
            placeholder_path = Path(tempfile.gettempdir()) / f"video_gen_placeholder_{task_id}.jpg"
            placeholder.save(placeholder_path)
            thumbnail_path = str(placeholder_path)

        if _is_valid_image_file(thumbnail_path):
            gallery_items.append((str(Path(thumbnail_path).resolve()), caption))
            visible_tasks.append(task)

    return gallery_items, (0 if gallery_items else None), visible_tasks


def load_task_preview(
    selected_index: int,
    output_dir: str = "./outputs",
    cached_tasks: list[dict[str, Any]] | None = None,
) -> tuple[str | None, str, str]:
    tasks = cached_tasks if cached_tasks is not None else scan_generation_directories(output_dir)
    if not tasks or selected_index is None or selected_index >= len(tasks):
        return None, "No task found.", "{}"

    task = tasks[selected_index]
    task_data = task.get("task_data", {})
    video_path = task.get("video_path")
    if video_path and not Path(video_path).exists():
        video_path = None
    return video_path, get_task_params_summary(task_data), json.dumps(task_data, ensure_ascii=False, indent=2)


def delete_task_files(task: dict[str, Any]) -> tuple[bool, str]:
    generation_dir = task.get("generation_dir")
    if not generation_dir:
        return False, "Task has no generation directory."
    generation_path = Path(generation_dir)
    if not generation_path.exists():
        return False, f"Directory does not exist: {generation_path}"
    try:
        shutil.rmtree(generation_path)
        return True, f"Deleted: {generation_path.name}"
    except Exception as exc:
        return False, f"Delete failed: {exc}"
