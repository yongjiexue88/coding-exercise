"""Query input normalization and validation helpers."""

from __future__ import annotations

import re
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_and_validate_query(value: Any) -> str:
    """Normalize user query text and enforce basic input constraints."""
    if not isinstance(value, str):
        raise ValueError("Query must be a string.")

    normalized = _WHITESPACE_RE.sub(" ", value).strip()
    if not normalized:
        raise ValueError("Query cannot be empty or whitespace only.")

    return normalized
