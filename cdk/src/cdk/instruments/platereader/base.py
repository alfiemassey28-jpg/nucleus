from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .result import PlateReaderResult

_REGISTRY: dict[str, type[PlateReaderLoader]] = {}
_EXT_REGISTRY: dict[str, str] = {}  # extension -> loader name


def plate_loader(name: str, extensions: list[str]):
    """Register a PlateReaderLoader subclass by name and supported extensions."""
    def decorator(cls: type[PlateReaderLoader]) -> type[PlateReaderLoader]:
        _REGISTRY[name] = cls
        for ext in extensions:
            _EXT_REGISTRY[ext.lower()] = name
        return cls
    return decorator


class PlateReaderLoader(ABC):
    @classmethod
    @abstractmethod
    def load(cls, path: Path | str) -> PlateReaderResult:
        ...
