"""vLLM-independent helpers for the Modal vLLM inference backend."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from .local import MAX_NEW_TOKENS

SCALED_ADAPTER_MARKER = ".simple-llm-scaled-adapter.json"


def format_prompt(tokenizer: Any, prompt: str, system_prompt: str | None) -> str:
    """Apply the shared chat template without importing vLLM."""
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_sampling_params(
    SamplingParams: Any,
    *,
    eos_token_ids: list[int],
    seed: int,
    presence_penalty: float | None,
    repetition_penalty: float | None,
) -> Any:
    """Construct vLLM sampling parameters from the shared generation settings."""
    kwargs: dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": MAX_NEW_TOKENS,
        "seed": seed,
        "stop_token_ids": eos_token_ids,
        "skip_special_tokens": True,
    }
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty
    return SamplingParams(**kwargs)


def prepare_scaled_adapter(adapter_path: Path, scale: float, destination: Path) -> Path:
    """Atomically cache an adapter whose LoRA alpha has the requested scale."""
    if scale < 0:
        raise ValueError("adapter scale must be non-negative")
    config, alpha = _adapter_config(adapter_path)
    scaled_alpha = alpha * scale
    source_manifest = _adapter_manifest(adapter_path)
    if _valid_scaled_adapter(
        destination, scale, scaled_alpha, source_manifest
    ):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    backup: Path | None = None
    try:
        shutil.copytree(adapter_path, stage, dirs_exist_ok=True)
        config["lora_alpha"] = scaled_alpha
        (stage / "adapter_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        marker = {
            "version": 1,
            "scale": scale,
            "source_manifest": source_manifest,
            "scaled_manifest": _adapter_manifest(stage),
        }
        (stage / SCALED_ADAPTER_MARKER).write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8"
        )
        if not _valid_scaled_adapter(stage, scale, scaled_alpha, source_manifest):
            raise ValueError(f"Invalid scaled adapter: {stage}")

        if destination.exists():
            backup = destination.with_name(
                f".{destination.name}.stale-{uuid.uuid4().hex}"
            )
            os.replace(destination, backup)
        try:
            os.replace(stage, destination)
        except Exception:
            if backup is not None and not destination.exists():
                os.replace(backup, destination)
                backup = None
            raise
        if backup is not None:
            _remove_path(backup)
    finally:
        if stage.exists():
            _remove_path(stage)
    return destination


def _adapter_config(adapter_path: Path) -> tuple[dict[str, Any], int | float]:
    config_path = adapter_path / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        alpha = config["lora_alpha"]
    except (json.JSONDecodeError, KeyError, OSError, TypeError) as exc:
        raise ValueError(f"Invalid adapter config: {config_path}") from exc
    if not isinstance(config, dict) or isinstance(alpha, bool) or not isinstance(
        alpha, (int, float)
    ):
        raise ValueError(f"Invalid adapter config: {config_path}")
    return config, alpha


def _adapter_manifest(adapter_path: Path) -> dict[str, str]:
    manifest = {}
    for path in sorted(adapter_path.rglob("*")):
        if not path.is_file() or path.name == SCALED_ADAPTER_MARKER:
            continue
        manifest[str(path.relative_to(adapter_path))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return manifest


def _valid_scaled_adapter(
    destination: Path,
    scale: float,
    scaled_alpha: int | float,
    source_manifest: dict[str, str],
) -> bool:
    try:
        marker = json.loads(
            (destination / SCALED_ADAPTER_MARKER).read_text(encoding="utf-8")
        )
        _, alpha = _adapter_config(destination)
        return (
            marker.get("version") == 1
            and marker.get("scale") == scale
            and marker.get("source_manifest") == source_manifest
            and marker.get("scaled_manifest") == _adapter_manifest(destination)
            and alpha == scaled_alpha
        )
    except (AttributeError, json.JSONDecodeError, OSError, TypeError, ValueError):
        return False


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def normalize_vllm_output(
    output: Any, started: float, max_tokens: int
) -> dict[str, Any]:
    """Translate a completed vLLM request result to the shared result shape."""
    if not output.outputs:
        raise ValueError(f"vLLM request {output.request_id} returned no output choices")
    choice = output.outputs[0]
    output_tokens = len(choice.token_ids)
    return {
        "response": choice.text.strip(),
        "input_tokens": len(output.prompt_token_ids),
        "output_tokens": output_tokens,
        "generation_seconds": time.perf_counter() - started,
        "truncated": output_tokens >= max_tokens,
    }
