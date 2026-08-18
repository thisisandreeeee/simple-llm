"""Small shared helpers for calling teacher and judge language models."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar


TEACHER_MODEL = "deepseek-v4-pro"
T = TypeVar("T")
R = TypeVar("R")


def create_deepseek_model(
    *,
    temperature: float = 0.7,
    generation_kwargs: dict[str, object] | None = None,
) -> object:
    """Construct the configured DeepSeek model lazily."""

    from deepeval.models import DeepSeekModel

    return DeepSeekModel(
        model=TEACHER_MODEL,
        temperature=temperature,
        generation_kwargs=generation_kwargs,
    )


async def run_concurrently(
    items: Iterable[T],
    worker: Callable[[T], Awaitable[R]],
    *,
    concurrency: int,
    on_result: Callable[[R], Awaitable[None]] | None = None,
) -> list[R]:
    """Run async work with a bounded number of in-flight calls."""

    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(item: T) -> R:
        async with semaphore:
            return await worker(item)

    tasks = [asyncio.create_task(run_one(item)) for item in items]
    results: list[R] = []
    try:
        for completed in asyncio.as_completed(tasks):
            result = await completed
            results.append(result)
            if on_result is not None:
                await on_result(result)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return results


__all__ = ["TEACHER_MODEL", "create_deepseek_model", "run_concurrently"]
