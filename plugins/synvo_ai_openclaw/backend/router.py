from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status, FastAPI
from .service import get_telegram_service
from .models import PairRequest, TestMessageRequest, TokenRequest

router = APIRouter(tags=["plugin-openclaw"])


async def on_startup(app: FastAPI):
    """Lifecycle hook called when the plugin is started"""
    pass


async def on_stop(app: FastAPI):
    """Lifecycle hook called when the plugin is stopped"""
    await get_telegram_service().stop()


@router.get("/telegram/status")
async def status(service=Depends(get_telegram_service)):
    return await service.get_status()


@router.get("/telegram/messages")
async def messages(limit: int = 50, service=Depends(get_telegram_service)):
    return {"messages": await service.list_messages(limit=limit)}


@router.post("/telegram/pair")
async def pair(payload: PairRequest, service=Depends(get_telegram_service)):
    state = await service.pair_chat(payload.chat_id)
    return {"paired_chats": sorted(state.paired_chat_ids)}


@router.post("/telegram/unpair")
async def unpair(payload: PairRequest, service=Depends(get_telegram_service)):
    state = await service.unpair_chat(payload.chat_id)
    return {"paired_chats": sorted(state.paired_chat_ids)}


@router.post("/telegram/restart")
async def restart(service=Depends(get_telegram_service)):
    await service.restart()
    return {"ok": True}

@router.post("/telegram/token")
async def set_token(payload: TokenRequest, service=Depends(get_telegram_service)):
    try:
        status_data = await service.set_token(payload.token)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"ok": True, "status": status_data}


@router.post("/telegram/test-message")
async def test_message(payload: TestMessageRequest, service=Depends(get_telegram_service)):
    try:
        await service.send_test_message(payload.chat_id, payload.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"ok": True}
