from __future__ import annotations

from typing import Any


def build_learning_router(*args: Any, **kwargs: Any):
    """Load the FastAPI router only when the application actually composes it.

    Keeping package import dependency-light prevents unrelated contract-only test
    discovery from requiring the full runtime dependency set.
    """

    from .api import build_learning_router as _build_learning_router

    return _build_learning_router(*args, **kwargs)


__all__ = ["build_learning_router"]
