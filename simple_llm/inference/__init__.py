"""Shared inference API and backend implementations."""

from .local import (
    GENERATION,
    Generator,
    PresencePenaltyLogitsProcessor,
    device,
    generate,
    generate_predictions,
    generation_eos_token_ids,
    local_generator,
)

__all__ = [
    "GENERATION",
    "Generator",
    "PresencePenaltyLogitsProcessor",
    "device",
    "generate",
    "generate_predictions",
    "generation_eos_token_ids",
    "local_generator",
]
