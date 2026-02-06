"""Storage service for the local RAG agent."""

from .apikeys import ApiKeyMixin
from .base import StorageBase
from .chats import ChatMixin
from .files import FileMixin
from .search import SearchMixin
from .memory import MemoryMixin


class IndexStorage(
    StorageBase,
    ApiKeyMixin,
    ChatMixin,
    FileMixin,
    SearchMixin,
    MemoryMixin,
):
    """
    Storage handling file index, folders, email accounts/messages, notes,
    chat sessions, messages, activity logs, and user memories.
    """
    pass


# Export the class properly
__all__ = ["IndexStorage"]
