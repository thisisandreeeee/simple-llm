"""Shared Modal setup for training jobs."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import modal

HF_CACHE_DIR = "/cache/huggingface"
TRAINING_DIR = "/training"
PACKAGE_DIR = Path(__file__).resolve().parent
RUN_NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]*"


def validate_run_name(run_name: str) -> str:
    """Validate and return a Modal training run name."""

    if not re.fullmatch(RUN_NAME_PATTERN, run_name):
        raise ValueError(
            "Run name may contain only letters, numbers, '.', '_', and '-'; "
            "it must start with a letter or number"
        )
    return run_name


def get_training_volumes() -> dict[str, modal.Volume]:
    """Return the shared Hugging Face cache and training volumes."""

    return {
        HF_CACHE_DIR: modal.Volume.from_name(
            "simple-llm-huggingface-cache", create_if_missing=True
        ),
        TRAINING_DIR: modal.Volume.from_name(
            "simple-llm-training", create_if_missing=True
        ),
    }


def build_training_image(
    packages: Iterable[str],
    local_files: Iterable[tuple[Path, str]] = (),
) -> modal.Image:
    """Build the common Python image and mount package/data files."""

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(*packages)
        .env({"HF_HOME": HF_CACHE_DIR, "HF_XET_HIGH_PERFORMANCE": "1"})
    )
    for local_path, remote_path in local_files:
        image = image.add_local_file(local_path, remote_path, copy=True)
    return image.add_local_dir(
        PACKAGE_DIR, "/root/simple_llm", ignore=["**/__pycache__/**"]
    )


def register_tensorboard_app(
    app: modal.App,
    image: modal.Image,
    training_volume: modal.Volume,
) -> object:
    """Register the shared TensorBoard endpoint on a training app."""

    class VolumeMiddleware:
        def __init__(self, wsgi_app):
            self.wsgi_app = wsgi_app

        def __call__(self, environ, start_response):
            if environ.get("PATH_INFO") == "/":
                try:
                    training_volume.reload()
                except Exception as error:
                    print(f"Could not reload TensorBoard logs: {error}")
            return self.wsgi_app(environ, start_response)

    @app.function(
        serialized=True,
        image=image,
        volumes={TRAINING_DIR: training_volume},
        max_containers=1,
    )
    @modal.wsgi_app()
    def tensorboard_app():
        import tensorboard

        board = tensorboard.program.TensorBoard()
        board.configure(logdir=TRAINING_DIR, load_fast="false")
        data_provider, deprecated_multiplexer = board._make_data_provider()
        return tensorboard.backend.application.TensorBoardWSGIApp(
            board.flags,
            board.plugin_loaders,
            data_provider,
            board.assets_zip_provider,
            deprecated_multiplexer,
            experimental_middlewares=[VolumeMiddleware],
        )

    return tensorboard_app
