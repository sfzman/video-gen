from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import shutil
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app_config import TASK_QUEUE_DIR, ensure_runtime_dirs

TASK_STATUS_PENDING = "pending"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_DONE = "done"
TASK_STATUS_FAILED = "failed"
MAX_RETRIES = 1
TASK_TIMEOUT = 60 * 60 * 2

_worker_thread: threading.Thread | None = None
_worker_stop_event = threading.Event()

_mp_ctx = mp.get_context("spawn")
_worker_proc: mp.Process | None = None
_task_q: Any = None
_result_q: Any = None


def _worker_subprocess_fn(task_q: Any, result_q: Any) -> None:
    import queue as queue_mod

    while True:
        try:
            params = task_q.get(timeout=1.0)
        except queue_mod.Empty:
            continue
        except (EOFError, OSError):
            break

        if params is None:
            break

        try:
            from generation_runner import run_generation_task

            output_path, message, meta = run_generation_task(params)
            result_q.put(("ok", output_path, message, meta))
        except Exception as exc:  # pragma: no cover - worker process path
            result_q.put(("error", str(exc), traceback.format_exc(), None))


def _start_worker_subprocess() -> None:
    global _worker_proc, _task_q, _result_q
    kill_worker_subprocess()
    _task_q = _mp_ctx.Queue()
    _result_q = _mp_ctx.Queue()
    _worker_proc = _mp_ctx.Process(
        target=_worker_subprocess_fn,
        args=(_task_q, _result_q),
        daemon=True,
    )
    _worker_proc.start()


def kill_worker_subprocess() -> str:
    global _worker_proc, _task_q, _result_q
    if _worker_proc is not None:
        if _worker_proc.is_alive():
            _worker_proc.terminate()
            _worker_proc.join(5)
            if _worker_proc.is_alive():
                _worker_proc.kill()
                _worker_proc.join(2)
        _worker_proc = None
    for current_queue in (_task_q, _result_q):
        if current_queue is not None:
            try:
                current_queue.close()
            except Exception:
                pass
    _task_q = None
    _result_q = None
    return "Worker subprocess terminated."


def _atomic_write_json(file_path: Path, data: dict[str, Any]) -> None:
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, file_path)


def _load_json(file_path: Path) -> dict[str, Any] | None:
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _recover_orphan_running_tasks() -> None:
    for task_file in TASK_QUEUE_DIR.rglob("task_*.json"):
        data = _load_json(task_file)
        if not data:
            continue
        if data.get("status") == TASK_STATUS_RUNNING:
            data["status"] = TASK_STATUS_PENDING
            _atomic_write_json(task_file, data)


def _iter_pending_tasks() -> list[tuple[Path, dict[str, Any]]]:
    tasks: list[tuple[Path, dict[str, Any]]] = []
    for task_file in TASK_QUEUE_DIR.rglob("task_*.json"):
        data = _load_json(task_file)
        if not data:
            continue
        retries = int(data.get("retries", 0))
        status = data.get("status")
        if status in (TASK_STATUS_PENDING, TASK_STATUS_FAILED) and retries <= MAX_RETRIES:
            tasks.append((task_file, data))
    tasks.sort(key=lambda item: item[1].get("created_at", ""))
    return tasks


def _copy_file_to_task(src_path: str, dst_dir: Path, file_name: str) -> str:
    src = Path(src_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Input file does not exist: {src}")
    dst = dst_dir / file_name
    shutil.copy2(str(src), str(dst))
    return str(dst)


def _copy_conditioning_images(images: list[dict[str, Any]], task_dir: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for index, image in enumerate(images, start=1):
        original_path = image.get("path")
        if not original_path:
            continue
        suffix = Path(original_path).suffix or ".png"
        copied_path = _copy_file_to_task(original_path, task_dir, f"conditioning_{index:02d}{suffix}")
        copied.append(
            {
                "path": copied_path,
                "position": float(image.get("position", 0.0)),
                "strength": float(image.get("strength", 1.0)),
            }
        )
    return copied


def _move_generation_artifacts(task_file: Path, task: dict[str, Any]) -> tuple[str | None, str | None, Path]:
    params = task.get("params", {})
    output_path = task.get("result", {}).get("output_video")
    save_folder = Path(params.get("save_folder_path") or "./outputs").expanduser().resolve()
    save_folder.mkdir(parents=True, exist_ok=True)
    task_dir = task_file.parent.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_root = save_folder / f"generation_{timestamp}_{task.get('id', 'unknown')}"
    dest_root.mkdir(parents=True, exist_ok=True)
    dest_task_dir = dest_root / task_dir.name
    shutil.move(str(task_dir), str(dest_task_dir))

    moved_video: str | None = None
    if output_path:
        output_name = Path(output_path).name
        candidate = dest_task_dir / output_name
        if candidate.exists():
            moved_video = str(candidate)

    dest_task_file = dest_task_dir / task_file.name
    return str(dest_task_dir), moved_video, dest_task_file


def _task_worker_loop() -> None:
    ensure_runtime_dirs()
    _recover_orphan_running_tasks()

    while not _worker_stop_event.is_set():
        try:
            pending = _iter_pending_tasks()
            if not pending:
                _worker_stop_event.wait(1.0)
                continue

            task_file, task = pending[0]
            task_id = task.get("id")
            task["status"] = TASK_STATUS_RUNNING
            start_time = datetime.now()
            task["started_at"] = start_time.isoformat()
            _atomic_write_json(task_file, task)

            params = task.get("params", {})
            if _worker_proc is None or not _worker_proc.is_alive():
                _start_worker_subprocess()

            _task_q.put(params)
            result = None
            elapsed = 0.0

            while elapsed < TASK_TIMEOUT and not _worker_stop_event.is_set():
                if _worker_proc is not None and not _worker_proc.is_alive():
                    result = ("error", "worker subprocess exited unexpectedly", f"exitcode={_worker_proc.exitcode}", None)
                    break
                try:
                    result = _result_q.get(timeout=1.0)
                    break
                except queue.Empty:
                    elapsed += 1.0

            if _worker_stop_event.is_set():
                break

            if result is None:
                kill_worker_subprocess()
                task["status"] = TASK_STATUS_FAILED
                task["result"] = {
                    "output_video": None,
                    "message": f"Task timed out after {TASK_TIMEOUT} seconds.",
                    "metadata": None,
                }
            elif result[0] == "ok":
                task["status"] = TASK_STATUS_DONE
                task["result"] = {
                    "output_video": result[1],
                    "message": result[2],
                    "metadata": result[3],
                }
            else:
                task["status"] = TASK_STATUS_FAILED
                task["result"] = {
                    "output_video": None,
                    "message": f"{result[1]}\n\n{result[2]}",
                    "metadata": result[3],
                }

            task["finished_at"] = datetime.now().isoformat()
            task["duration_seconds"] = round((datetime.now() - start_time).total_seconds(), 2)
            task["retries"] = int(task.get("retries", 0)) + (0 if task["status"] == TASK_STATUS_DONE else 1)

            write_target = task_file
            moved_dir = None
            moved_video = None
            if task["status"] == TASK_STATUS_DONE:
                try:
                    moved_dir, moved_video, write_target = _move_generation_artifacts(task_file, task)
                except Exception as exc:
                    task["result"]["message"] += f"\n\nArtifact move failed: {exc}"

            task["result"]["moved_task_dir"] = moved_dir
            task["result"]["moved_video"] = moved_video
            _atomic_write_json(write_target, task)
        except Exception:
            traceback.print_exc()
            _worker_stop_event.wait(1.0)

    kill_worker_subprocess()


def start_task_worker() -> str:
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return "Task worker thread is already running."
    ensure_runtime_dirs()
    _worker_stop_event.clear()
    _worker_thread = threading.Thread(target=_task_worker_loop, name="video-gen-task-worker", daemon=True)
    _worker_thread.start()
    return "Task worker thread started."


def stop_task_worker() -> str:
    _worker_stop_event.set()
    kill_worker_subprocess()
    return "Task worker thread stopped."


def enqueue_task(params: dict[str, Any]) -> str:
    ensure_runtime_dirs()
    task_id = str(uuid.uuid4())
    task_dir = TASK_QUEUE_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    persisted_params = dict(params)
    persisted_params["conditioning_images"] = _copy_conditioning_images(params.get("conditioning_images", []), task_dir)
    persisted_params["_task_dir"] = str(task_dir)

    task = {
        "id": task_id,
        "created_at": datetime.now().isoformat(),
        "status": TASK_STATUS_PENDING,
        "retries": 0,
        "max_retries": MAX_RETRIES,
        "params": persisted_params,
        "result": None,
    }
    task_file = task_dir / f"task_{task_id}.json"
    _atomic_write_json(task_file, task)
    start_task_worker()
    return f"Task queued: {task_id}\nQueue dir: {task_dir}\nThe worker will process it asynchronously."
