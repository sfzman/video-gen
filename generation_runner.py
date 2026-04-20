from __future__ import annotations

from pathlib import Path
from typing import Any

from engines import generate_with_model


def run_generation_task(params: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    task_dir = Path(params["_task_dir"]).resolve()
    clean_params = dict(params)
    clean_params.pop("_task_dir", None)
    return generate_with_model(clean_params, task_dir)
