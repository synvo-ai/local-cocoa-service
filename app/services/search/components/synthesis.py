from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, AsyncIterable

logger = logging.getLogger(__name__)

class SynthesisComponent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def deduplicate_sub_answers(self, sub_answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes duplicate or highly similar sub-answers.
        """
        # Helper to check negative response logic roughly (could inject VerificationComponent or duplicate simple check)
        # For independence, strict "negative response" check might be better passed in or duplicated.
        # I'll implement a simple local check to reuse logic without circular dependency.
        def _is_neg(content):
            if not content: return True
            c = content.upper()
            return "NO_ANSWER" in c or "NO ANSWER" in c or "I DON'T KNOW" in c

        if len(sub_answers) <= 3:
            return [
                ans for ans in sub_answers 
                if ans.get("has_answer") and not _is_neg(ans.get("content", ""))
            ]
        
        unique_answers = []
        seen_contents = []
        
        for answer in sub_answers:
            if not answer.get("has_answer"):
                continue
            
            content = answer.get("content", "")
            if _is_neg(content):
                continue
            
            content_lower = content.lower()
            words = set(re.findall(r'\w+', content_lower))
            
            if not words:
                continue
            
            # Check if this answer is a duplicate
            is_duplicate = False
            for seen_words in seen_contents:
                overlap = len(words & seen_words)
                similarity = overlap / max(len(words), len(seen_words), 1)
                if similarity > 0.75:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_answers.append(answer)
                seen_contents.append(words)
        
        return unique_answers

    async def aggregate_sub_answers(
        self, 
        query: str, 
        sub_answers: List[Dict[str, Any]]
    ) -> str:
        """
        Optimized aggregation.
        """
        unique_answers = self.deduplicate_sub_answers(sub_answers)
        
        if not unique_answers:
            return "I cannot find a clear answer in your documents."
        
        unique_answers.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        if len(unique_answers) <= 8:
            return await self.simple_aggregation(query, unique_answers)
        
        return await self.hierarchical_aggregation(query, unique_answers)

    def _format_evidence_line(self, ans: Dict[str, Any]) -> str:
        """Format a single evidence line with file context."""
        if not ans.get('content'):
            return ""
        
        # Build source info with file name if available
        source = ans.get('source', 'Unknown')
        metadata = ans.get('metadata', {})
        file_name = metadata.get('file_name') or metadata.get('name')
        
        if file_name and file_name not in source:
            source_info = f"{file_name} - {source}"
        else:
            source_info = source
        
        confidence = ans.get('confidence', 0)
        return f"[{ans['index']}] (Source: {source_info}, Confidence: {confidence:.1f})\n{ans['content']}"

    async def simple_aggregation(
        self, 
        query: str, 
        sub_answers: List[Dict[str, Any]]
    ) -> str:
        """
        Aggregate multiple sub-answers into a comprehensive response.
        """
        evidence_lines = [
            self._format_evidence_line(ans)
            for ans in sub_answers
            if ans.get('content')
        ]
        evidence_lines = [line for line in evidence_lines if line]  # Remove empty lines
        evidence_text = "\n\n".join(evidence_lines)
        
        if not evidence_text.strip():
            return "I cannot find a clear answer in the provided documents."
        
        system_prompt = (
            "You are a strict, fact-based assistant. Answer the question based ONLY on the provided evidence.\n"
            "If the evidence does not contain the specific information requested (e.g., specific year or figure), "
            "explicitly state that the information is not available.\n"
            "Do NOT infer, guess, or use data from other years to fill gaps.\n"
            "CRITICAL: Use ONLY the exact bracket format [N] for citations where N is the evidence number.\n"
            "Do NOT write 'Evidence [N]' or '(source N)' - just [N].\n"
            "Place citation [N] immediately after each fact.\n"
            "IMPORTANT: Do NOT repeat yourself. Give a single, concise answer.\n"
            "If evidence conflicts, use the most specific/reliable source and ignore vague statements.\n"
            "Be coherent - do not contradict yourself.\n"
            "Write naturally and fluently, with government-style professional formatting.\n"
        )
        
        user_prompt = (
            f"Question: {query}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            "Cite each fact with [N] right after it. Do not repeat. Answer coherently, do not contradict yourself. Answer the question directly."
        )
        
        try:
            final_answer = await self.llm_client.chat_complete(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
            )
            return final_answer.strip()
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            if sub_answers:
                best = max(sub_answers, key=lambda x: x.get('confidence', 0))
                return f"{best['content']} [{best['index']}]"
            return "Synthesis error."

    async def stream_simple_aggregation(
        self,
        query: str,
        sub_answers: List[Dict[str, Any]],
        subquery_summaries: List[Dict[str, Any]] | None = None
    ) -> AsyncIterable[str]:
        """
        Streaming version of simple_aggregation.
        
        Args:
            query: The original user question
            sub_answers: Chunk-level evidence with references
            subquery_summaries: Optional list of sub-query level answers (already synthesized)
        """
        evidence_lines = [
            self._format_evidence_line(ans)
            for ans in sub_answers
            if ans.get('content')
        ]
        evidence_lines = [line for line in evidence_lines if line]  # Remove empty lines
        evidence_text = "\n\n".join(evidence_lines)

        if not evidence_text.strip():
             yield "I cannot find a clear answer in the provided documents."
             return

        # Build sub-query summaries section if available
        summaries_section = ""
        if subquery_summaries:
            summary_lines = []
            for sq in subquery_summaries:
                summary_lines.append(f"• {sq['sub_query']}\n  → {sq['answer']}")
            summaries_section = "KEY FINDINGS FROM SUB-QUERIES:\n" + "\n".join(summary_lines) + "\n\n"

        system_prompt = (
            "You are a strict, fact-based assistant. Answer the question based ONLY on the provided evidence.\n"
            "If the evidence does not contain the specific information requested (e.g., specific year or figure), "
            "explicitly state that the information is not available.\n"
            "Do NOT infer, guess, or use data from other years to fill gaps.\n"
            "CRITICAL: Use ONLY the exact bracket format [N] for citations where N is the evidence number.\n"
            "Do NOT write 'Evidence [N]' or '(source N)' - just [N].\n"
            "Place citation [N] immediately after each fact.\n"
            "IMPORTANT: Do NOT repeat yourself. Give a single, concise answer.\n"
            "Meta Answer is provided, but you should also refer to the raw evidence to answer the question with citations."
            "If evidence conflicts, use the most specific/reliable source and ignore vague statements.\n"
            "Be coherent - do not contradict yourself.\n"
            "Write naturally and fluently, with government-style professional formatting.\n"
        )

        user_prompt = (
            f"Question: {query}\n\n"
            f"Meta Answer: {summaries_section}"
            f"EVIDENCE (for citations):\n{evidence_text}\n\n"
            "Cite each fact with [N] right after it. Do not repeat. Answer coherently, do not contradict yourself. Answer the question directly."
        )

        # DEBUG LOGGING
        logger.warning("=" * 80)
        logger.warning("[SYNTHESIS DEBUG] stream_simple_aggregation called")
        logger.warning(f"[SYNTHESIS DEBUG] Query: {query}")
        logger.warning(f"[SYNTHESIS DEBUG] Number of sub_answers: {len(sub_answers)}")
        logger.warning(f"[SYNTHESIS DEBUG] System Prompt:\n{system_prompt}")
        logger.warning(f"[SYNTHESIS DEBUG] User Prompt:\n{user_prompt}")
        logger.warning(f"[SYNTHESIS DEBUG] max_tokens=250")
        logger.warning("=" * 80)

        collected_answer = []
        try:
            async for token in self.llm_client.stream_chat_complete(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
            ):
                collected_answer.append(token)
                yield token
            
            # Log final answer
            logger.warning(f"[SYNTHESIS DEBUG] Final Answer: {''.join(collected_answer)}")
            logger.warning("=" * 80)
        except Exception as e:
            logger.error(f"Streaming aggregation failed: {e}")
            if sub_answers:
                best = max(sub_answers, key=lambda x: x.get('confidence', 0))
                yield f"{best['content']} [{best['index']}]"
            else:
                yield "Synthesis error."

    async def hierarchical_aggregation(
        self, 
        query: str, 
        sub_answers: List[Dict[str, Any]]
    ) -> str:
        """
        Two-stage hierarchical aggregation.
        """
        batch_size = 6
        intermediate_summaries = []
        
        for i in range(0, len(sub_answers), batch_size):
            batch = sub_answers[i:i + batch_size]
            group_num = i // batch_size + 1
            try:
                summary = await self.simple_aggregation(query, batch)
                intermediate_summaries.append({
                    "content": summary,
                    "source": f"Group {group_num}",
                    "index": group_num,
                    "has_answer": True,
                    "confidence": 0.8
                })
            except Exception as e:
                continue
        
        if not intermediate_summaries:
            return "Unable to synthesize answer from the provided documents."
        
        if len(intermediate_summaries) == 1:
            return intermediate_summaries[0]["content"]
        
        summaries_text = "\n\n".join(
            f"[Group {s['index']}]\n{s['content']}"
            for s in intermediate_summaries
        )
        
        system_prompt = (
            "You are a strict, fact-based assistant. Answer the question based ONLY on the provided evidence.\n"
            "If the evidence does not contain the specific information requested (e.g., specific year or figure), "
            "explicitly state that the information is not available.\n"
            "Do NOT infer, guess, or use data from other years to fill gaps.\n"
            "CRITICAL: Use ONLY the exact bracket format [N] for citations where N is the evidence number.\n"
            "Do NOT write 'Evidence [N]' or '(source N)' - just [N].\n"
            "Place citation [N] immediately after each fact.\n"
            "IMPORTANT: Do NOT repeat yourself. Give a single, concise answer.\n"
            "If evidence conflicts, use the most specific/reliable source and ignore vague statements.\n"
            "Be coherent - do not contradict yourself.\n"
            "Write naturally and fluently, with government-style professional formatting.\n"
        )
        
        user_prompt = (
            f"Question: {query}\n\n"
            f"Group Summaries:\n{summaries_text}\n\n"
            "Cite each fact with [N] right after it. Do not repeat. Answer coherently, do not contradict yourself. Answer the question directly."
        )
        
        try:
            final_answer = await self.llm_client.chat_complete(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,
            )
            return final_answer.strip()
        except Exception as e:
            logger.error(f"Final synthesis failed: {e}")
            return intermediate_summaries[0]["content"]
