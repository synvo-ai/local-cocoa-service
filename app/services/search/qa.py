from __future__ import annotations
import logging
import json
import time
import asyncio
import re
from typing import Any, Iterable, List, AsyncIterable, AsyncGenerator, Dict, TYPE_CHECKING, Optional
import os


from services.llm.client import EmbeddingClient, LlmClient, RerankClient
from core.config import settings
from core.model_manager import get_model_manager, ModelType, ModelState
from core.models import (
    QaRequest,
    QaResponse,
    SearchHit,
    SearchResponse,
    AgentStepFile,
    SubQueryResult,
)
from services.storage import IndexStorage
from core.vector_store import VectorStore
from .types import (
    EmbeddingUnavailableError,
    StepRecorder,
    QueryRewriteResult,
    SubQuestion,
    Candidate,
    VerifyResult,
    VerifiedCandidate,
    DebugStep,
    SubQuestionAnswer,
    RetrievalLimits,
)

if TYPE_CHECKING:
    from .engine import SearchEngine

logger = logging.getLogger(__name__)


class QaMixin:
    """
    Mixin containing QA/Answer logic for SearchEngine.
    """

    async def answer(self: 'SearchEngine', payload: QaRequest) -> QaResponse:
        started = time.perf_counter()

        full_answer = ""
        all_hits = []
        # StepRecorder expects structured steps - use recorder to track
        recorder = StepRecorder()
        thinking_steps: list[dict[str, Any]] = []

        def _match_hit(hit: SearchHit, result: dict[str, Any], idx: int) -> bool:
            hit_chunk_id = hit.chunk_id or hit.metadata.get("chunk_id") or hit.metadata.get("chunkId")
            result_chunk_id = result.get("chunk_id") or result.get("chunkId")
            hit_index = hit.metadata.get("index") if isinstance(hit.metadata, dict) else None
            result_index = result.get("index")

            if hit_chunk_id and result_chunk_id and hit_chunk_id == result_chunk_id:
                return True
            if result_index is not None and hit_index is not None and result_index == hit_index:
                return True
            if result_index is not None and result_index == (idx + 1):
                return True
            return False

        def _apply_analysis_to_hits(results: list[dict[str, Any]]) -> None:
            if not results:
                return
            for idx, hit in enumerate(all_hits):
                for result in results:
                    if _match_hit(hit, result, idx):
                        hit.has_answer = result.get("has_answer")
                        hit.analysis_comment = result.get("comment")
                        hit.analysis_confidence = result.get("confidence")
                        break

            # Update any cached hits inside thinking steps, if present
            for step in thinking_steps:
                hits = step.get("hits")
                if not isinstance(hits, list):
                    continue
                for h_idx, h in enumerate(hits):
                    for result in results:
                        hit_chunk_id = h.get("chunkId") or h.get("chunk_id") or h.get("metadata", {}).get("chunkId") or h.get("metadata", {}).get("chunk_id")
                        result_chunk_id = result.get("chunk_id") or result.get("chunkId")
                        hit_index = h.get("metadata", {}).get("index")
                        result_index = result.get("index")

                        chunk_match = hit_chunk_id and result_chunk_id and hit_chunk_id == result_chunk_id
                        index_match = (result_index is not None) and (hit_index is not None) and (result_index == hit_index)
                        fallback_match = (result_index is not None) and (result_index == (h_idx + 1))

                        if chunk_match or index_match or fallback_match:
                            h["hasAnswer"] = result.get("has_answer")
                            h["analysisComment"] = result.get("comment")
                            h["analysisConfidence"] = result.get("confidence")
                            break

        try:
            async for event_str in self.stream_answer(payload):
                try:
                    event = json.loads(event_str)
                    etype = event.get("type")
                    data = event.get("data")

                    if etype == "token":
                        full_answer += data
                    elif etype == "hits":
                        # data is list of dicts, convert back to SearchHit?
                        # Or stream_answer emits raw dicts?
                        # Looking at pipelines, they emit objects or dicts.
                        # stream_answer does: yield json.dumps(event)
                        # StandardPipeline yields "hits" as list of SearchHit objects usually,
                        # but json.dumps requires serializable.
                        # Let's check StandardPipeline. It yields hits as Pydantic models dumped to dict.
                        # So data is list of dicts.
                        for h in data:
                            all_hits.append(SearchHit(**h))

                    elif etype == "thinking_step":
                        # data is dict with id, title, status etc.
                        # StepRecorder.add only accepts specific keys, and 'type' is not one of them
                        # We need to filter 'type' out and map 'summary' to 'detail' if needed?
                        # StepRecorder.add(id, title, detail, status, queries, items, files, duration_ms)
                        # thinking_step data usually has: id, type, title, status, summary, items?

                        # Map keys valid for StepRecorder.add
                        valid_keys = {"id", "title", "detail", "status", "queries", "items", "files", "duration_ms"}

                        # If 'summary' is present but 'detail' is not, use 'summary' as 'detail'
                        if "summary" in data and "detail" not in data:
                            data["detail"] = data["summary"]

                        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
                        recorder.add(**filtered_data)
                        if isinstance(data, dict):
                            thinking_steps.append(dict(data))

                    elif etype == "error":
                        logger.error(f"Stream error: {data}")
                        full_answer += f"\nError: {data}"
                    elif etype == "chunk_analysis":
                        if isinstance(data, list):
                            _apply_analysis_to_hits(data)
                    elif etype == "chunk_progress":
                        if isinstance(data, dict) and isinstance(data.get("chunk_result"), dict):
                            _apply_analysis_to_hits([data["chunk_result"]])

                except json.JSONDecodeError:
                    pass

        except Exception as e:
            logger.error(f"Answer execution failed: {e}")
            return QaResponse(
                answer=f"An error occurred: {e}",
                hits=[],
                latency_ms=int((time.perf_counter() - started) * 1000),
                diagnostics=recorder.snapshot(),
                thinking_steps=thinking_steps or None,
            )

        return QaResponse(
            answer=full_answer.strip(),
            hits=all_hits,
            latency_ms=int((time.perf_counter() - started) * 1000),
            diagnostics=recorder.snapshot(),
            thinking_steps=thinking_steps or None,
        )

    async def stream_answer(self: 'SearchEngine', payload: QaRequest) -> AsyncGenerator[str, None]:
        """
        Stream answer generation with progressive steps.
        """
        logger.info(f"stream_answer called: mode={getattr(payload, 'mode', None)}, search_mode={getattr(payload, 'search_mode', 'auto')}")

        # Check if LLM client is available
        if not self.llm_client:
            yield json.dumps({"type": "error", "data": "LLM service is not configured. Please check your settings."}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return

        # Get search_mode from payload, default to "auto"
        search_mode = getattr(payload, 'search_mode', 'auto')

        started = time.perf_counter()
        limit = payload.limit or settings.qa_context_limit

        # === Scope Isolation: Two mechanisms ===
        # 1. @filename syntax in query (user-facing, inline in question)
        # 2. folder_ids parameter (API-facing, for benchmarks/test mode)

        target_file_ids: Optional[List[str]] = None
        query = payload.query

        # Mechanism 1: Parse @filename mentions from query
        # Supports: @filename or @"filename with spaces"
        file_filters = []
        matches = re.findall(r'@(?:"([^"]+)"|(\S+))', query)
        for m in matches:
            name = m[0] if m[0] else m[1]
            if name:
                file_filters.append(name)

        if file_filters:
            # Remove @mentions from query for cleaner search
            query = re.sub(r'@(?:"[^"]+"|[^\s]+)', '', query).strip()
            query = re.sub(r'\s+', ' ', query).strip()  # Clean up double spaces

            # Find matching files
            target_file_ids = []
            for fname in file_filters:
                files = self.storage.find_files_by_name(fname)
                target_file_ids.extend([f.id for f in files])

            if target_file_ids:
                logger.info(f"@mention filter: restricting to {len(target_file_ids)} files matching {file_filters}")
            else:
                logger.warning(f"@mention filter: no files found matching {file_filters}")

        # Mechanism 2: folder_ids parameter (API-level, for benchmarks)
        if payload.folder_ids:
            folder_file_ids = []
            for folder_id in payload.folder_ids:
                folder_files = self.storage.get_files_in_folder(folder_id)
                folder_file_ids.extend([f.id for f in folder_files])

            if folder_file_ids:
                if target_file_ids:
                    # Intersect with @mention filter if both present
                    target_file_ids = list(set(target_file_ids) & set(folder_file_ids))
                    logger.info(f"Combined filter: {len(target_file_ids)} files (intersection of @mention and folder)")
                else:
                    target_file_ids = folder_file_ids
                    logger.info(f"Folder filter: restricting to {len(target_file_ids)} files in {len(payload.folder_ids)} folder(s)")
            else:
                logger.warning(f"Folder filter: no files found in folders {payload.folder_ids}")

        # Handle "direct" search_mode - skip document search entirely
        if search_mode == "direct" or payload.mode == "chat":
            # Check if model is hibernated and emit loading status, then wait for it
            manager = get_model_manager()
            logger.info(f"Model manager enabled: {settings.model_manager.enabled}")
            model_was_stopped = False
            if settings.model_manager.enabled:
                vision_model = manager.get_model_instance(ModelType.VISION)
                logger.info(f"Vision model state: {vision_model.state}")
                if vision_model.state != ModelState.RUNNING:
                    yield json.dumps({"type": "status", "data": "loading_model"}) + "\n"
                    try:
                        await manager.ensure_model(ModelType.VISION)
                        model_was_stopped = True
                    except Exception as e:
                        yield json.dumps({"type": "error", "data": f"Failed to start AI model: {e}"}) + "\n"
                        yield json.dumps({"type": "done"}) + "\n"
                        return

            if not model_was_stopped:
                yield json.dumps({"type": "status", "data": "answering"}) + "\n"

            try:
                system_message = (
                    "You are Local Cocoa, a helpful AI assistant for a local document workspace. "
                    "You were created at NTU Singapore (Nanyang Technological University).\n\n"
                    "Respond to the user naturally and helpfully."
                )
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": payload.query}
                ]

                is_first_token = True
                async for chunk in self.llm_client.stream_chat_complete(messages, max_tokens=1024):
                    if is_first_token:
                        if model_was_stopped:
                            # Now that we're actually generating, update status
                            yield json.dumps({"type": "status", "data": "answering"}) + "\n"
                        is_first_token = False
                    yield json.dumps({"type": "token", "data": chunk}) + "\n"
            except RuntimeError as e:
                error_msg = str(e)
                if "failed to start" in error_msg.lower() or "timeout" in error_msg.lower():
                    yield json.dumps({"type": "error", "data": f"AI model failed to start: {error_msg}"}) + "\n"
                else:
                    yield json.dumps({"type": "error", "data": f"LLM error: {error_msg}"}) + "\n"
            except Exception as e:
                logger.exception("LLM generation failed")
                yield json.dumps({"type": "error", "data": f"LLM generation failed: {type(e).__name__}"}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return

        # Check if models (Embedding & LLM) need loading
        # We need Embedding for search and Vision/LLM for intent routing + answering
        manager = get_model_manager()
        if settings.model_manager.enabled:
            embedding_model = manager.get_model_instance(ModelType.EMBEDDING)
            vision_model = manager.get_model_instance(ModelType.VISION)

            models_to_load = []
            if embedding_model.state != ModelState.RUNNING:
                models_to_load.append(ModelType.EMBEDDING)
            if vision_model.state != ModelState.RUNNING:
                models_to_load.append(ModelType.VISION)

            if models_to_load:
                logger.info(f"Models need loading: {models_to_load}")
                yield json.dumps({"type": "status", "data": "loading_model"}) + "\n"
                try:
                    for m_type in models_to_load:
                        await manager.ensure_model(m_type)
                except Exception as e:
                    logger.error(f"Failed to ensure models {models_to_load}: {e}")
                    yield json.dumps({"type": "error", "data": f"Failed to start AI models: {e}"}) + "\n"
                    yield json.dumps({"type": "done"}) + "\n"
                    return

        yield json.dumps({"type": "status", "data": "searching"}) + "\n"

        # For "knowledge" mode, always search docs (skip intent routing)
        if search_mode == "knowledge":
            intent = "document"
            call_tools = True
        else:
            # Auto mode: use query routing via Component
            # Use cleaned query (without @mentions) for routing
            routing_result = await self.intent_component.query_intent_routing(query)
            intent = routing_result.get("intent", "document")
            call_tools = routing_result.get("call_tools", True)
        # Path 1: Does not require tools - direct answer
        if not call_tools:
            # Check if model is hibernated and emit loading status, then wait for it
            manager = get_model_manager()
            model_was_stopped = False
            if settings.model_manager.enabled:
                vision_model = manager.get_model_instance(ModelType.VISION)
                if vision_model.state != ModelState.RUNNING:
                    yield json.dumps({"type": "status", "data": "loading_model"}) + "\n"
                    try:
                        await manager.ensure_model(ModelType.VISION)
                        model_was_stopped = True
                    except Exception as e:
                        yield json.dumps({"type": "error", "data": f"Failed to start AI model: {e}"}) + "\n"
                        yield json.dumps({"type": "done"}) + "\n"
                        return

            if not model_was_stopped:
                yield json.dumps({"type": "status", "data": "direct_answer"}) + "\n"

            if intent == "greeting":
                system_message = (
                    "You are Local Cocoa, a friendly AI assistant for a local document workspace.\n\n"
                    "Name: Local Cocoa\nOrigin: Born at NTU Singapore\n"
                    "When greeting users, be warm and briefly introduce yourself."
                )
            else:
                system_message = (
                    "You are Local Cocoa, a helpful AI assistant for a local document workspace. "
                    "Respond to the user naturally and concisely."
                )

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": payload.query}
            ]

            try:
                is_first_token = True
                async for chunk in self.llm_client.stream_chat_complete(messages, max_tokens=512):
                    if is_first_token:
                        if model_was_stopped:
                            yield json.dumps({"type": "status", "data": "direct_answer"}) + "\n"
                        is_first_token = False
                    yield json.dumps({"type": "token", "data": chunk}) + "\n"
            except RuntimeError as e:
                error_msg = str(e)
                if "failed to start" in error_msg.lower() or "timeout" in error_msg.lower():
                    yield json.dumps({"type": "error", "data": f"AI model failed to start: {error_msg}"}) + "\n"
                else:
                    yield json.dumps({"type": "error", "data": f"LLM error: {error_msg}"}) + "\n"
            except Exception as e:
                logger.exception("LLM generation failed")
                yield json.dumps({"type": "error", "data": f"LLM generation failed: {type(e).__name__}"}) + "\n"

            yield json.dumps({"type": "done"}) + "\n"
            return
        # Path 2: Requires tools - search docs

        # Step 0: Smart Analysis for Multi-path (Async)
        # Use cleaned query (without @mentions) for analysis and search
        analysis = {"needs_decomposition": False, "sub_queries": [query], "strategy": "SINGLE", "keywords": []}

        # Emit thinking step: analyzing query
        analyze_step_id = f"analyze_{int(time.perf_counter()*1000)}"
        yield json.dumps({"type": "thinking_step", "data": {
            "id": analyze_step_id,
            "type": "analyze",
            "title": "Analyzing Query",
            "status": "running",
            "summary": "Extracting keywords and analyzing query structure...",
            "timestamp_ms": int((time.perf_counter() - started) * 1000)
        }}) + "\n"

        try:
            analysis = await self.intent_component.analyze_query(query)
            # Emit thinking step: analysis complete
            sub_queries = analysis.get("sub_queries", [query])
            needs_decomp = analysis.get("needs_decomposition", False)
            
            if needs_decomp:
                summary = f"Split into {len(sub_queries)} sub-queries"
            else:
                summary = "Query analyzed"

            yield json.dumps({"type": "thinking_step", "data": {
                "id": analyze_step_id,
                "type": "analyze",
                "title": "Analyzing Query",
                "status": "complete",
                "summary": summary,
                "timestamp_ms": int((time.perf_counter() - started) * 1000),
            }}) + "\n"
        except Exception as e:
            logger.warning(f"Intent analysis failed: {e}")
            yield json.dumps({"type": "thinking_step", "data": {
                "id": analyze_step_id,
                "type": "analyze",
                "title": "Analyzing Query",
                "status": "complete",
                "summary": "Using default analysis",
                "timestamp_ms": int((time.perf_counter() - started) * 1000)
            }}) + "\n"

        # Get vision answering option from payload
        use_vision_for_answer = getattr(payload, 'use_vision_for_answer', False)
        logger.info(f"[VISION DEBUG] qa.py received use_vision_for_answer={use_vision_for_answer}, payload={payload}")

        if analysis["needs_decomposition"]:
            # Delegate to MultiPathPipeline
            async for event in self.multipath_pipeline.execute(
                payload, limit,
                analysis=analysis,
                target_file_ids=target_file_ids,
                cleaned_query=query,  # Pass cleaned query (without @mentions)
                use_vision_for_answer=use_vision_for_answer,
            ):
                yield event
            return

        # Standard Single-Path Flow via StandardPipeline
        step_counter = 0

        def standard_next_step():
            nonlocal step_counter
            step_counter += 1
            return f"single_{step_counter}"

        # Setup context parts for simple synthesis later
        valid_sub_answers = []
        valid_chunks = 0

        async for event in self.standard_pipeline.execute(
            query=query,  # Use cleaned query (without @mentions)
            limit=limit,
            step_generator=standard_next_step,
            target_file_ids=target_file_ids,
            use_vision_for_answer=use_vision_for_answer,
        ):
            if event["type"] in ["thinking_step", "status", "chunk_progress", "chunk_analysis"]:
                yield json.dumps(event) + "\n"
            elif event["type"] == "hits":
                yield json.dumps(event) + "\n"
                # Keep hits? Pipeline handles processing loop.
            elif event.get("type") == "done_internal":
                # Handle no results
                if event["data"] == "No matching files found.":
                    yield json.dumps({"type": "done", "data": "I couldn't find any relevant documents."}) + "\n"
                    return
            elif event.get("type") == "sub_answers":
                pass  # Pipeline returns sub-answers, but we needcontext for final synthesis

            # Note: StandardPipeline yields chunks progress. But synthesis?
            # StandardPipeline verifies chunks but doesn't synthesize the final answer text string.
            # We need to collect valid chunks for synthesis here.
            # StandardPipeline emits "sub_answers".
            if event["type"] == "sub_answers":
                sub_answers = event["data"]
                # Collect valid sub_answers for synthesis
                for ans in sub_answers:
                    if ans.get("has_answer"):
                        # Keep track of valid answers for synthesis later
                        # We need 'valid_sub_answers' list which was missing
                        valid_sub_answers.append(ans)
        synthesis_inputs = []
        for ans in valid_sub_answers:
            synthesis_inputs.append({
                "index": ans.get("index"),
                "source": ans.get("source"),
                "content": ans.get("content"),
                "confidence": ans.get("confidence", 1.0)
            })

        try:
            synth_id = standard_next_step()  # Define synth_id before use
            async for chunk in self.synthesis_component.stream_simple_aggregation(query, synthesis_inputs):
                yield json.dumps({"type": "token", "data": chunk}) + "\n"

            yield json.dumps({"type": "thinking_step", "data": {
                "id": synth_id,
                "type": "synthesize",
                "title": "Synthesizing Answer",
                "status": "complete",
                "summary": "Answer generated",
                "timestamp_ms": int((time.perf_counter() - started) * 1000)
            }}) + "\n"

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"

        yield json.dumps({"type": "done"}) + "\n"
