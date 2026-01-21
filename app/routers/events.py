from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from core.events import bus

router = APIRouter()


@router.get("/api/events")
async def event_stream():
    return EventSourceResponse(bus.subscribe())
