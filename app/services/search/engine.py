from __future__ import annotations

import logging
from typing import Optional

import asyncio

from core.config import settings
from services.llm.client import EmbeddingClient, LlmClient, RerankClient
from core.models import QaRequest, QaResponse, SearchHit, SearchResponse
from services.storage import IndexStorage
from core.vector_store import VectorStore, get_vector_store

from .types import EmbeddingUnavailableError, StepRecorder, QueryRewriteResult
from .search import SearchMixin
from .qa import QaMixin

from .components.intent import IntentComponent
from .components.verification import VerificationComponent
from .components.synthesis import SynthesisComponent
from .components.vision_answer import VisionAnswerComponent
from .pipelines.standard import StandardPipeline
from .pipelines.multipath import MultiPathPipeline

logger = logging.getLogger(__name__)


class SearchEngine(SearchMixin, QaMixin):
    def __init__(
        self,
        storage: IndexStorage,
        embedding_client: EmbeddingClient,
        rerank_client: RerankClient,
        llm_client: LlmClient,
        *,
        vectors: Optional[VectorStore] = None,
    ) -> None:
        self.storage = storage
        self.embedding_client = embedding_client
        self.rerank_client = rerank_client
        self.llm_client = llm_client
        self.vector_store = vectors or get_vector_store()
        
        # Components
        self.intent_component = IntentComponent(llm_client)
        self.verification_component = VerificationComponent(llm_client)
        self.synthesis_component = SynthesisComponent(llm_client)
        self.vision_answer_component = VisionAnswerComponent(llm_client)
        
        # Pipelines
        self.standard_pipeline = StandardPipeline(
            self, 
            self.verification_component,
            vision_answer=self.vision_answer_component,
        )
        self.multipath_pipeline = MultiPathPipeline(
            self, 
            self.intent_component, 
            self.synthesis_component, 
            self.standard_pipeline
        )

    # Methods from Mixins are available here:
    # search(), stream_search(), answer(), stream_answer(), etc.
