"""
Shared data contracts used by both the retrieval and generation layers.

`RetrievedChunk` used to be defined twice — once in `src.retrieval.config`
and once in `src.generation.config` — with an application-layer adapter
function converting between the two on every request. Since both layers
represent the exact same concept (a chunk retrieved from the knowledge
base, carried through to generation), it's unified here as a single
source of truth. Both `src.retrieval.config` and `src.generation.config`
re-export it under the same name so existing imports keep working.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum


class CitationLocationType(str, Enum):
    """Type of location in the source document."""
    PAGE = "page"
    SLIDE = "slide"
    SECTION = "section"
    CHAPTER = "chapter"
    UNKNOWN = "unknown"


@dataclass
class RetrievedChunk:
    """
    A single chunk retrieved from the knowledge base.

    Produced by the retrieval layer's ContextBuilder and consumed
    unchanged by the generation layer's GenerationRequest — no
    conversion step between the two.
    """
    chunk_id: str
    content: str                       # Enriched content (with source prefix)
    score: float                       # Final relevance score (post-rerank)

    # Source traceability
    course_name: str = ""
    chapter_title: str = ""
    topic: str = ""
    location_type: CitationLocationType = CitationLocationType.UNKNOWN
    location_start: int = 0
    location_end: int = 0
    filename: str = ""
    file_type: str = ""                # e.g. "pptx", "pdf" — determines location_type
    chunk_type: str = ""
    source_prefix: str = ""

    # Original content without source-prefix enrichment, when available
    raw_content: str = ""

    # Optional per-source component scores from hybrid search, when tracked
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    metadata_score: Optional[float] = None

    # Additional metadata (flexible for provider-specific data)
    metadata: Dict[str, Any] = field(default_factory=dict)
