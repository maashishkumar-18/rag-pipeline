"""
Retrieval Debug Server (throwaway)

Minimal local HTTP wrapper around RetrievalOrchestrator, built so a
frontend or curl/Postman can exercise retrieval quality manually against a
running Pinecone index. No auth, no sessions, no error contracts, single
process, single worker — for local debugging, not a production API layer.

Run:
    uvicorn tools.retrieval_debug_server:app --reload --port 8500
(run from the repo root so the `src` package resolves)
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ingestion.vector_store import VectorStore
from src.retrieval.orchestrator import RetrievalOrchestrator
from src.retrieval.query_rewriter import ConversationState
from src.retrieval.pipelines import PIPELINE_REGISTRY

app = FastAPI(title="RAG Pipeline Retrieval Debug Server")

app.add_middleware(
    CORSMiddleware,
    # Vite auto-increments the port (5173, 5174, ...) if one is taken, so
    # match any localhost port instead of hardcoding 5173.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

_vector_store: Optional[VectorStore] = None
_orchestrator: Optional[RetrievalOrchestrator] = None


def get_orchestrator() -> RetrievalOrchestrator:
    global _vector_store, _orchestrator
    if _orchestrator is None:
        _vector_store = VectorStore()
        _orchestrator = RetrievalOrchestrator(vector_store=_vector_store)
    return _orchestrator


class QueryRequest(BaseModel):
    query: str
    namespace: str
    pipeline: str = "learning"
    subject: Optional[str] = None
    module: Optional[str] = None
    chapter: Optional[str] = None
    current_topic: Optional[str] = None
    current_concept: Optional[str] = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/namespaces")
def namespaces() -> Dict[str, List[str]]:
    try:
        vector_store = VectorStore() if _vector_store is None else _vector_store
        return {"namespaces": vector_store.list_namespaces()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipelines")
def pipelines() -> Dict[str, List[str]]:
    return {"pipelines": list(PIPELINE_REGISTRY.keys())}


@app.post("/query")
def query(req: QueryRequest) -> Dict[str, Any]:
    if req.pipeline not in PIPELINE_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown pipeline '{req.pipeline}'")

    orchestrator = get_orchestrator()

    state = ConversationState(
        session_id="retrieval_debug_ui",
        subject=req.subject or None,
        module=req.module or None,
        chapter=req.chapter or None,
        current_topic=req.current_topic or None,
        current_concept=req.current_concept or None,
    )

    result = orchestrator.retrieve(
        query=req.query,
        pipeline_config=PIPELINE_REGISTRY[req.pipeline],
        conversation_state=state,
        namespace=req.namespace,
    )

    chunks = [
        {
            "chunk_id": c.chunk_id,
            "content": c.content,
            "raw_content": c.raw_content,
            "score": c.score,
            "semantic_score": c.semantic_score,
            "keyword_score": c.keyword_score,
            "metadata_score": c.metadata_score,
            "course_name": c.course_name,
            "chapter_title": c.chapter_title,
            "topic": c.topic,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "chunk_type": c.chunk_type,
            "filename": c.filename,
            "file_type": c.file_type,
        }
        for c in result.assembled_context.chunks
    ]

    return {
        "chunks": chunks,
        "formatted_context": result.assembled_context.formatted_context,
        "diagnostics": {
            "pipeline_name": result.pipeline_name,
            "routing_decision": result.routing_decision,
            "agent_used": result.agent_used,
            "original_query": result.original_query,
            "rewritten_query": result.rewritten_query,
            "expanded_query": result.expanded_query,
            "keywords": result.keywords,
            "candidates_retrieved": result.candidates_retrieved,
            "candidates_reranked": result.candidates_reranked,
            "sources_used": result.sources_used,
            "total_time_ms": result.total_time_ms,
            "timing_breakdown": result.timing_breakdown,
            "retry_count": result.retry_count,
            "is_fallback_result": result.is_fallback_result,
            "error_details": result.error_details,
            "failed_components": result.failed_components,
            "confidence": {
                "score": result.confidence.score,
                "level": result.confidence.level,
                "action": result.confidence.action,
                "feature_scores": result.confidence.feature_scores,
                "recommendations": result.confidence.recommendations,
                "should_retry": result.confidence.should_retry,
            },
        },
    }
