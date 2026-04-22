from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def normalize_log_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return normalize_log_value(value.model_dump())
    if is_dataclass(value):
        return normalize_log_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): normalize_log_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [normalize_log_value(item) for item in value]
    if isinstance(value, list):
        return [normalize_log_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return normalize_log_value({key: item for key, item in vars(value).items() if not key.startswith("_")})
    return value


def structured_extra(event: str, **fields: Any) -> dict[str, Any]:
    extra = {"event": event}
    for key, value in fields.items():
        extra[key] = normalize_log_value(value)
    return extra

