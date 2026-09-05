"""name -> Editor factory, so grids can name editors as plain strings."""
from __future__ import annotations

from typing import Callable

from .base import Editor

_REGISTRY: dict[str, Callable[..., Editor]] = {}


def register(name: str, factory: Callable[..., Editor]) -> None:
    _REGISTRY[name] = factory


def available() -> list[str]:
    return sorted(_REGISTRY)


def build(name: str, **kwargs) -> Editor:
    if name not in _REGISTRY:
        raise KeyError(f"editor {name!r} is not registered; have {available()}")
    return _REGISTRY[name](**kwargs)


def register_defaults() -> None:
    """Register the stub plus every EasyEdit method under its bare name."""
    from .easyedit_adapter import SUPPORTED_METHODS, EasyEditAdapter
    from .stub import StubEditor

    register("stub", StubEditor)
    for method in SUPPORTED_METHODS:
        register(method, lambda method=method, **kw: EasyEditAdapter(method=method, **kw))
