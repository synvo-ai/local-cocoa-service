from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional

class PairRequest(BaseModel):
    chat_id: int = Field(..., gt=0)


class TestMessageRequest(BaseModel):
    chat_id: Optional[int] = Field(default=None, gt=0)
    text: Optional[str] = None


class TokenRequest(BaseModel):
    token: str = ""
