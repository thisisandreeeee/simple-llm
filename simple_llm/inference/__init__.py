"""Shared inference API and backend implementations."""

from .local import (
    GENERATION,
    AsyncGenerator,
    Generator,
    PresencePenaltyLogitsProcessor,
    async_generate_predictions,
    device,
    generate,
    generate_predictions,
    generation_eos_token_ids,
    local_generator,
)

__all__ = [
    "GENERATION",
    "AsyncGenerator",
    "Generator",
    "PresencePenaltyLogitsProcessor",
    "async_generate_predictions",
    "device",
    "generate",
    "generate_predictions",
    "generation_eos_token_ids",
    "local_generator",
]
