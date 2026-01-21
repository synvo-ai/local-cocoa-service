import asyncio
from typing import AsyncGenerator, Any
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Event:
    type: str
    data: Any


class EventBus:
    _subscribers: list[asyncio.Queue] = field(default_factory=list)

    def __init__(self):
        self._subscribers = []

    async def publish(self, type: str, data: Any):
        event = Event(type=type, data=data)
        for queue in self._subscribers:
            await queue.put(event)

    async def subscribe(self) -> AsyncGenerator[str, None]:
        queue = asyncio.Queue()
        self._subscribers.append(queue)
        try:
            while True:
                event: Event = await queue.get()
                yield f"event: {event.type}\ndata: {json.dumps(event.data)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if queue in self._subscribers:
                self._subscribers.remove(queue)


# Global event bus
bus = EventBus()
