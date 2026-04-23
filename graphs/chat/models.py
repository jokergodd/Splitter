from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class ChatGraphInput(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    top_k: int = Field(default=10, gt=0)
    candidate_limit: int = Field(default=30, gt=0)
    max_queries: int = Field(default=4, gt=0)
    parent_limit: int = Field(default=5, gt=0)
    request_id: str | None = None
