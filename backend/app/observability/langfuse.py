"""Fail-open, privacy-conscious Langfuse instrumentation helpers."""
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional

from ..config import settings


@lru_cache(maxsize=1)
def get_langfuse_client():
    """Return a configured Langfuse client, or None when observability is off."""
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            base_url=settings.LANGFUSE_BASE_URL,
        )
    except Exception as exc:
        # Monitoring must never prevent a financial-research request from running.
        print(f"[OmniBrain] Langfuse disabled: {exc}")
        return None


@contextmanager
def observe(name: str, observation_type: str = "span", **kwargs: Any) -> Iterator[Optional[Any]]:
    """Create a Langfuse observation when configured; otherwise act as a no-op."""
    client = get_langfuse_client()
    if client is None:
        yield None
        return
    try:
        manager = client.start_as_current_observation(
            name=name, as_type=observation_type, **kwargs
        )
    except Exception as exc:
        print(f"[OmniBrain] Langfuse observation failed: {exc}")
        yield None
        return
    # Deliberately do not catch exceptions raised by the research workflow.
    # A telemetry wrapper must not hide failures in the actual application.
    with manager as observation:
        yield observation


def update(observation: Optional[Any], **kwargs: Any) -> None:
    """Best-effort updates; telemetry errors are intentionally non-fatal."""
    if observation is None:
        return
    try:
        observation.update(**kwargs)
    except Exception as exc:
        print(f"[OmniBrain] Langfuse update failed: {exc}")


def flush() -> None:
    client = get_langfuse_client()
    if client is not None:
        try:
            client.flush()
        except Exception as exc:
            print(f"[OmniBrain] Langfuse flush failed: {exc}")
