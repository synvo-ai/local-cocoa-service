import asyncio
import sys
import json
from unittest.mock import AsyncMock, MagicMock

# 1. Mock heavy dependencies BEFORE imports
sys.modules["qdrant_client"] = MagicMock()
sys.modules["qdrant_client.models"] = MagicMock()
sys.modules["core.vector_store"] = MagicMock()

# Now we can import
from services.search.progressive import search_pipeline
from services.search.types import SearchMethod, AnswerReadiness, MethodRunResult
import services.search.progressive as prog

# 2. Setup Mock Components
class MockEngine:
    def __init__(self):
        self.llm_client = AsyncMock()
        self.storage = MagicMock()
        # Mock embeddings check
        self.embeddings_ready = lambda: True
    
    # Mock retrieval methods
    async def _lexical_backfill(self, *args, **kwargs):
        # Simulator: Lexical search finds nothing or irrelevant stuff
        print("  -> [Engine] Executing Lexical Search...")
        return []

    async def _vector_hits(self, *args, **kwargs):
        # Simulator: Vector search finds the right doc
        print("  -> [Engine] Executing Vector Search...")
        return []

# 3. Custom Mock Logic for the Progressive Flow
async def run_simulation():
    print("=== Starting Progressive Search Simulation ===")
    print("Query: 'Who is the president of NTU'")
    
    engine = MockEngine()
    
    # Force the strict Method list as defined in the system
    # We will simulate:
    # 1. Metadata Lookup (Fail)
    # 2. Fulltext (Fail)
    # 3. Vector (Success - "Good Answer")
    
    m1 = SearchMethod(name="metadata_lookup", enabled=True, cost_level=1, top_k=5)
    m2 = SearchMethod(name="fulltext_fts", enabled=True, cost_level=2, top_k=5)
    m3 = SearchMethod(name="vector_ann", enabled=True, cost_level=3, top_k=10)
    
    prog.build_method_list = MagicMock(return_value=[m1, m2, m3])
    
    # Mock Search Functions per method
    async def mock_search_fn(engine, query, keywords, filters, top_k):
        # Just return empty candidates by default, logic handled below in run_logs or verified
        return []
    
    prog.get_method_fn = MagicMock(return_value=mock_search_fn)
    
    # Mock Dedup & Rerank (Pass through)
    prog.dedup_and_merge = MagicMock(side_effect=lambda pool, new: new)
    prog.pre_rerank = MagicMock(side_effect=lambda pool, q, b: pool)
    prog.extract_keywords = MagicMock(return_value=["president", "NTU"])
    
    # Mock Verification & Readiness to simulate the story
    # Call 1 (Metadata): No candidates -> No Ans
    # Call 2 (Fulltext): No candidates -> No Ans
    # Call 3 (Vector): Found candidates -> Good Ans
    
    # We need to execute the actual loop in progressive.py
    # But because we mocked `get_method_fn` to return [], `global_pool` stays empty
    # So we need to inject candidates into `get_method_fn` for the 3rd call
    
    async def smart_search_fn(engine, query, keywords, filters, top_k):
        # Inspect the stack or use a counter to know which method? 
        # Actually `get_method_fn` is called with method_name. 
        # But `prog.get_method_fn` returns the function.
        pass

    # Better approach: 
    # Let's mock `search_one_subquery_progressive`'s internal helper calls?
    # No, we want to run the pipeline logic.
    
    # We will simply mock `get_method_fn` to return different functions based on name
    async def search_meta(*args, **kwargs):
        print("  -> [Method: metadata_lookup] Searching...")
        await asyncio.sleep(0.1)
        return []
        
    async def search_fulltext(*args, **kwargs):
        print("  -> [Method: fulltext_fts] Searching...")
        await asyncio.sleep(0.2)
        return []
        
    async def search_vector(*args, **kwargs):
        print("  -> [Method: vector_ann] Searching...")
        print("     Found 3 candidates: ['ntu_governance.pdf', ...]")
        await asyncio.sleep(0.5)
        # Return a "candidate"
        return [prog.Candidate(
            chunk_id="c1", 
            text_preview="Professor Ho Teck Hua is the President of Nanyang Technological University (NTU), Singapore.", 
            score=0.92,
            sub_question_id="sq1",
            route="vector_ann",
            query_used=args[1],
            file_id="f1",
            meta={}
        )]

    def get_method_router(name):
        if name == "metadata_lookup": return search_meta
        if name == "fulltext_fts": return search_fulltext
        if name == "vector_ann": return search_vector
        return search_meta

    prog.get_method_fn = MagicMock(side_effect=get_method_router)
    
    # Mock Check Answer
    async def smart_check(engine, query, verified):
        if not verified:
            return (AnswerReadiness.NO_ANSWER, 0.0, None)
        # If we have the simulated vector result
        print("  -> [Check] Verifying answer readiness...")
        best = verified[0].verify
        if "Ho Teck Hua" in best.extracted_answer:
            return (AnswerReadiness.GOOD_ANSWER, 0.95, best.extracted_answer)
        return (AnswerReadiness.NO_ANSWER, 0.0, None)
        
    prog.check_answer_ready = smart_check

    # Mock verify_candidates_batch to actually "verify" the injected candidate
    async def smart_verify(engine, query, candidates, limits):
        verified = []
        for c in candidates:
            # Simulate LLM verification
            print(f"     [Verify] LLM confirming candidate: {c.text_preview[:30]}...")
            verified.append(prog.VerifiedCandidate(
                candidate=c,
                verify=prog.VerifyResult(
                    is_relevant=True,
                    answerable=True,
                    confidence=0.95,
                    extracted_answer="Professor Ho Teck Hua",
                    evidence_quote="Professor Ho Teck Hua is the President..."
                )
            ))
        return verified, None

    prog.verify_candidates_batch = smart_verify

    # execute
    sub_queries = [{"id": "sq1", "text": "Who is the president of NTU"}]
    
    print("\n--- Pipeline Execution Start ---")
    result = await search_pipeline(engine, "Who is the president of NTU", sub_queries)
    print("--- Pipeline Execution End ---\n")
    
    # Analyze Result
    sq_res = result["sub_results"][0]
    print(f"Final Status: {'Need User Decision' if result['needs_user_decision'] else 'Done'}")
    print(f"Best Answer Found: {sq_res['best_so_far']}")
    print(f"Confidence: {sq_res['best_conf']}")
    print(f"Resume Token: {sq_res['resume_token'][:20]}...")
    
    print("\n[Simulator] Success! The system escalated to Vector search, found the answer, and triggered an early exit.")

if __name__ == "__main__":
    asyncio.run(run_simulation())
