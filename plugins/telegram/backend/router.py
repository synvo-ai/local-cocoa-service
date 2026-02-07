from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from .service import telegram_service

router = APIRouter(tags=["plugin-telegram"])


class PairRequest(BaseModel):
    chat_id: int = Field(..., gt=0)


class TestMessageRequest(BaseModel):
    chat_id: int | None = Field(default=None, gt=0)
    text: str | None = None

class TokenRequest(BaseModel):
    token: str = ""


@router.on_event("startup")
async def _startup() -> None:
    await telegram_service.start()


@router.on_event("shutdown")
async def _shutdown() -> None:
    await telegram_service.stop()


@router.get("/status")
async def status():
    return await telegram_service.get_status()


@router.get("/messages")
async def messages(limit: int = 50):
    return {"messages": await telegram_service.list_messages(limit=limit)}


@router.post("/pair")
async def pair(payload: PairRequest):
    state = await telegram_service.pair_chat(payload.chat_id)
    return {"paired_chats": sorted(state.paired_chat_ids)}


@router.post("/unpair")
async def unpair(payload: PairRequest):
    state = await telegram_service.unpair_chat(payload.chat_id)
    return {"paired_chats": sorted(state.paired_chat_ids)}


@router.post("/restart")
async def restart():
    await telegram_service.restart()
    return {"ok": True}

@router.post("/token")
async def set_token(payload: TokenRequest):
    try:
        status_data = await telegram_service.set_token(payload.token)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"ok": True, "status": status_data}


@router.post("/test-message")
async def test_message(payload: TestMessageRequest):
    try:
        await telegram_service.send_test_message(payload.chat_id, payload.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"ok": True}
