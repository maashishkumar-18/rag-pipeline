"""
Post Processor
Extracts citations, validates grounding, cleans artifacts,
and formats the final answer from generated text.

Architecture:
    GeneratedAnswer + RetrievedChunks → PostProcessor.process() → GenerationResponse

Features:
- Citation extraction from generated text
- Grounding validation (keyword overlap)
- Artifact removal (markdown cleanup)
- Answer formatting
- Citation-to-source mapping

Configuration-driven — settings from config/generation/post_processing.yaml.
No hardcoded patterns.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path

from src.generation.config import (
    GeneratedAnswer,
    GenerationResponse,
    Citation,
    RetrievedChunk,
    RetrievalMetadata,
    GenerationMode,
    AnswerFormat,
    CitationStyle,
    CitationLocationType,
    PostProcessingConfig,
    CitationFormatter,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Citation Extractor
# ============================================================================

class CitationExtractor:
    """
    Extracts citations from generated text.
    
    Recognizes patterns like:
    - [Course | Chapter | Slide 5]
    - [Course | Chapter | Pages 5-7]
    - [Source: ...]
    """
    
    def __init__(self, pattern: Optional[str] = None):
        """
        Initialize citation extractor.
        
        Args:
            pattern: Regex pattern for citation detection.
                     Defaults to bracket-based citation format.
        """
        self.pattern = pattern or r'\[([^\]]+)\]'
        self._compiled = re.compile(self.pattern)
    
    def extract(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract citations from text.
        
        Args:
            text: Generated answer text
            
        Returns:
            List of (citation_string, start_position, end_position) tuples
        """
        citations = []
        for match in self._compiled.finditer(text):
            citation_str = match.group(1)
            # Filter out non-citation brackets (e.g., markdown links)
            if self._is_citation(citation_str):
                citations.append((citation_str, match.start(), match.end()))
        return citations
    
    def _is_citation(self, text: str) -> bool:
        """
        Check if a bracketed string is likely a citation.
        
        Citations typically contain pipes (|) separating fields
        or reference to page/slide numbers.
        """
        # Contains pipe separators (Course | Chapter | Slide X)
        if "|" in text:
            return True
        
        # Contains page/slide reference
        if re.search(r'(slide|page|slides|pages)\s*\d+', text, re.IGNORECASE):
            return True
        
        return False
    
    def parse_citation(self, citation_str: str) -> Optional[Citation]:
        """
        Parse a citation string into a Citation object.
        
        Args:
            citation_str: Raw citation string (e.g., "SCM | Inventory | Slide 5")
            
        Returns:
            Citation object or None if parsing fails
        """
        parts = [p.strip() for p in citation_str.split("|")]
        
        if len(parts) < 3:
            return None
        
        course_name = parts[0] if len(parts) > 0 else ""
        chapter_title = parts[1] if len(parts) > 1 else ""
        location_str = parts[2] if len(parts) > 2 else ""
        
        # Parse location
        location_type = CitationLocationType.UNKNOWN
        location_start = 0
        location_end = 0
        
        location_lower = location_str.lower()
        
        # Slide patterns
        slide_match = re.search(r'slides?\s*(\d+)(?:\s*[-–]\s*(\d+))?', location_lower)
        if slide_match:
            location_type = CitationLocationType.SLIDE
            location_start = int(slide_match.group(1))
            location_end = int(slide_match.group(2)) if slide_match.group(2) else location_start
        
        # Page patterns
        page_match = re.search(r'pages?\s*(\d+)(?:\s*[-–]\s*(\d+))?', location_lower)
        if page_match:
            location_type = CitationLocationType.PAGE
            location_start = int(page_match.group(1))
            location_end = int(page_match.group(2)) if page_match.group(2) else location_start
        
        # Section patterns
        section_match = re.search(r'sections?\s*(\d+)', location_lower)
        if section_match:
            location_type = CitationLocationType.SECTION
            location_start = int(section_match.group(1))
        
        return Citation(
            index=0,  # Will be assigned after extraction
            chunk_id="",  # Will be matched later
            course_name=course_name,
            chapter_title=chapter_title,
            location_type=location_type,
            location_start=location_start,
            location_end=location_end,
        )


# ============================================================================
# Grounding Validator
# ============================================================================

class GroundingValidator:
    """
    Validates that generated answers are grounded in the retrieved context.
    
    Uses keyword overlap to check if answer claims are supported by
    the source chunks.
    """
    
    def __init__(
        self,
        method: str = "keyword_overlap",
        min_overlap: float = 0.3
    ):
        """
        Initialize grounding validator.
        
        Args:
            method: Validation method ("keyword_overlap", "llm_check", "both")
            min_overlap: Minimum keyword overlap ratio for grounding
        """
        self.method = method
        self.min_overlap = min_overlap
    
    def validate(
        self,
        answer: str,
        chunks: List[RetrievedChunk]
    ) -> Tuple[bool, float]:
        """
        Validate grounding of answer against retrieved chunks.
        
        Args:
            answer: Generated answer text
            chunks: Source chunks used for generation
            
        Returns:
            Tuple of (is_grounded, confidence)
        """
        if not chunks or not answer:
            return False, 0.0
        
        if self.method == "keyword_overlap":
            return self._keyword_overlap_check(answer, chunks)
        elif self.method == "both":
            kw_grounded, kw_conf = self._keyword_overlap_check(answer, chunks)
            return kw_grounded, kw_conf
        else:
            # "llm_check" — placeholder for future implementation
            return self._keyword_overlap_check(answer, chunks)
    
    def _keyword_overlap_check(
        self,
        answer: str,
        chunks: List[RetrievedChunk]
    ) -> Tuple[bool, float]:
        """
        Check grounding using keyword overlap.
        
        Extracts meaningful keywords from the answer and checks
        what fraction appear in the source chunks.
        """
        # Combine all chunk content
        source_text = " ".join(chunk.content.lower() for chunk in chunks)
        
        # Extract keywords from answer (exclude common words)
        answer_words = self._extract_keywords(answer.lower())
        
        if not answer_words:
            return False, 0.0
        
        # Count matches
        matched = sum(1 for word in answer_words if word in source_text)
        overlap_ratio = matched / len(answer_words)
        
        is_grounded = overlap_ratio >= self.min_overlap
        confidence = min(1.0, overlap_ratio / max(self.min_overlap, 0.1))
        
        return is_grounded, confidence
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """
        Extract meaningful keywords from text.
        
        Excludes common stopwords and short words.
        """
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
            'and', 'but', 'or', 'not', 'no', 'if', 'then', 'than',
            'also', 'very', 'just', 'about', 'each', 'all', 'both',
            'such', 'only', 'other', 'more', 'some', 'most',
        }
        
        words = re.findall(r'\b[a-z]{3,}\b', text)
        return {w for w in words if w not in stopwords}


# ============================================================================
# Answer Cleaner
# ============================================================================

class AnswerCleaner:
    """
    Cleans generated answer text.
    
    Removes artifacts, normalizes whitespace, and formats output.
    """
    
    def __init__(
        self,
        remove_artifacts: bool = True,
        normalize_whitespace: bool = True,
        max_length: Optional[int] = None
    ):
        """
        Initialize answer cleaner.
        
        Args:
            remove_artifacts: Remove [HIDE], <THINK> and similar tags
            normalize_whitespace: Normalize whitespace
            max_length: Maximum answer length (truncates if exceeded)
        """
        self.remove_artifacts = remove_artifacts
        self.normalize_whitespace = normalize_whitespace
        self.max_length = max_length
    
    def clean(self, text: str) -> str:
        """
        Clean generated answer text.
        
        Args:
            text: Raw generated text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove artifacts
        if self.remove_artifacts:
            text = self._remove_artifacts(text)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        # Truncate if needed
        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length].rsplit(' ', 1)[0] + "..."
        
        return text.strip()
    
    def _remove_artifacts(self, text: str) -> str:
        """Remove generation artifacts."""
        # Remove [HIDE]...[/HIDE] blocks
        text = re.sub(r'\[HIDE\].*?\[/HIDE\]', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove <THINK>...</THINK> blocks
        text = re.sub(r'<THINK>.*?</THINK>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove trailing "---" separators
        text = re.sub(r'\n*---\s*$', '', text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        # Remove trailing whitespace per line
        lines = [line.rstrip() for line in text.split('\n')]
        
        # Remove empty lines at start and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        # Normalize multiple blank lines to max 2
        result = []
        blank_count = 0
        for line in lines:
            if not line.strip():
                blank_count += 1
                if blank_count <= 2:
                    result.append('')
            else:
                blank_count = 0
                result.append(line)
        
        return '\n'.join(result)


# ============================================================================
# Post Processor
# ============================================================================

class PostProcessor:
    """
    Processes generated answers into final GenerationResponse objects.
    
    Composes CitationExtractor, GroundingValidator, and AnswerCleaner
    to produce clean, validated, citation-mapped output.
    """
    
    def __init__(self, config: Optional[PostProcessingConfig] = None):
        """
        Initialize post processor.
        
        Args:
            config: PostProcessingConfig (loads from YAML if None)
        """
        if config is None:
            config = PostProcessingConfig(
                extract_citations=True,
                citation_style=CitationStyle.INLINE,
                validate_grounding=True,
                grounding_check_method="keyword_overlap",
                remove_artifacts=True,
                normalize_whitespace=True,
                format_as_markdown=True,
            )
        
        self.config = config
        
        self.citation_extractor = CitationExtractor()
        self.grounding_validator = GroundingValidator(
            method=config.grounding_check_method,
            min_overlap=config.min_keyword_overlap,
        )
        self.answer_cleaner = AnswerCleaner(
            remove_artifacts=config.remove_artifacts,
            normalize_whitespace=config.normalize_whitespace,
            max_length=config.max_answer_length,
        )
        self.citation_formatter = CitationFormatter()
    
    def process(
        self,
        generated: GeneratedAnswer,
        chunks: List[RetrievedChunk],
        retrieval_metadata: RetrievalMetadata,
        mode: GenerationMode,
        request_id: str,
        generation_id: str,
        warnings: Optional[List[str]] = None,
        sources_used: Optional[List[str]] = None,
    ) -> GenerationResponse:
        """
        Process generated answer into final response.
        
        Args:
            generated: Raw GeneratedAnswer from LLM
            chunks: Source chunks used for generation
            retrieval_metadata: Metadata from retrieval
            mode: Generation mode used
            request_id: Request identifier
            generation_id: Generation identifier
            warnings: Optional pre-existing warnings
            sources_used: Optional retrieval sources used
            
        Returns:
            GenerationResponse ready for API response
        """
        all_warnings = list(warnings) if warnings else []
        
        # Step 1: Clean the answer
        cleaned_answer = self.answer_cleaner.clean(generated.content)
        
        # Step 2: Extract citations
        citations = []
        if self.config.extract_citations:
            raw_citations = self.citation_extractor.extract(cleaned_answer)
            
            for i, (citation_str, start, end) in enumerate(raw_citations):
                parsed = self.citation_extractor.parse_citation(citation_str)
                if parsed:
                    parsed.index = i + 1
                    parsed.position_start = start
                    parsed.position_end = end
                    
                    # Try to match citation to a source chunk
                    matched_chunk = self._match_citation_to_chunk(parsed, chunks)
                    if matched_chunk:
                        parsed.chunk_id = matched_chunk.chunk_id
                    
                    citations.append(parsed)
            
            if not citations:
                all_warnings.append("No citations found in generated answer")
        
        # Step 3: Validate grounding
        is_grounded = False
        grounding_confidence = 0.0
        
        if self.config.validate_grounding:
            is_grounded, grounding_confidence = self.grounding_validator.validate(
                answer=cleaned_answer,
                chunks=chunks,
            )
            
            if not is_grounded:
                all_warnings.append(
                    f"Answer may not be fully grounded in context "
                    f"(confidence: {grounding_confidence:.2f})"
                )
        
        # Step 4: Build final response
        return GenerationResponse(
            answer=cleaned_answer,
            mode=mode,
            answer_format=AnswerFormat.MARKDOWN if self.config.format_as_markdown else AnswerFormat.PLAIN_TEXT,
            citations=citations,
            is_grounded=is_grounded,
            grounding_confidence=round(grounding_confidence, 4),
            retrieval_metadata=retrieval_metadata,
            model_info=generated.model_info,
            usage=generated.usage,
            generation_time_ms=generated.generation_time_ms,
            total_time_ms=0,  # Will be set by orchestrator
            request_id=request_id,
            generation_id=generation_id,
            warnings=all_warnings,
            sources_used=sources_used or [],
        )
    
    def _match_citation_to_chunk(
        self,
        citation: Citation,
        chunks: List[RetrievedChunk]
    ) -> Optional[RetrievedChunk]:
        """
        Match a parsed citation to a source chunk.
        
        Matches by course name, chapter, and page/slide range.
        """
        for chunk in chunks:
            if (chunk.course_name.lower() == citation.course_name.lower() and
                chunk.chapter_title.lower() == citation.chapter_title.lower() and
                citation.location_start >= chunk.location_start and
                citation.location_end <= chunk.location_end):
                return chunk
        
        # Relaxed match: course + chapter only
        for chunk in chunks:
            if (chunk.course_name.lower() == citation.course_name.lower() and
                chunk.chapter_title.lower() == citation.chapter_title.lower()):
                return chunk
        
        return None


# ============================================================================
# Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Post Processor - Validation")
    print("=" * 60)
    
    from src.generation.config import (
        GeneratedAnswer,
        RetrievedChunk,
        RetrievalMetadata,
        GenerationMode,
        CitationLocationType,
        ConfidenceLevel,
        UsageStats,
        ModelInfo,
    )
    
    # Sample generated answer with citations
    sample_answer = """
## Definition

[Supply Chain Management Strategy | Inventory Management | Slide 5]
Inventory refers to the stock of goods, materials, and supplies that a business holds for the purpose of resale, production, or use in providing services.

## Types of Inventory

[Supply Chain Management Strategy | Inventory Management | Slide 6]
The main types of inventory include:

- **Raw Materials**: Basic materials used in production
- **Work-in-Progress (WIP)**: Partially completed products
- **Finished Goods**: Completed products ready for sale
- **MRO Supplies**: Maintenance, repair, and operations supplies

## Example

[Supply Chain Management Strategy | Inventory Management | Slide 7]
For example, a car manufacturer holds raw materials (steel, rubber), WIP (partially assembled vehicles), and finished goods (completed cars ready for sale).
"""
    
    # Sample source chunks
    sample_chunks = [
        RetrievedChunk(
            chunk_id="doc1_0001",
            content="Inventory refers to the stock of goods, materials, and supplies that a business holds.",
            score=0.92,
            course_name="Supply Chain Management Strategy",
            chapter_title="Inventory Management",
            location_type=CitationLocationType.SLIDE,
            location_start=5,
            location_end=5,
            metadata={"chunk_type": "definition"}
        ),
        RetrievedChunk(
            chunk_id="doc1_0002",
            content="Types of inventory: raw materials, work-in-progress, finished goods, MRO supplies.",
            score=0.88,
            course_name="Supply Chain Management Strategy",
            chapter_title="Inventory Management",
            location_type=CitationLocationType.SLIDE,
            location_start=6,
            location_end=6,
            metadata={"chunk_type": "explanation"}
        ),
        RetrievedChunk(
            chunk_id="doc1_0003",
            content="Car manufacturer example: raw materials (steel), WIP (partial vehicles), finished goods (completed cars).",
            score=0.82,
            course_name="Supply Chain Management Strategy",
            chapter_title="Inventory Management",
            location_type=CitationLocationType.SLIDE,
            location_start=7,
            location_end=7,
            metadata={"chunk_type": "example"}
        ),
    ]
    
    # Process the answer
    processor = PostProcessor()
    
    generated = GeneratedAnswer(
        content=sample_answer,
        model_info=ModelInfo(provider="deepseek", model_name="deepseek-chat"),
        usage=UsageStats(input_tokens=200, output_tokens=150, total_tokens=350),
        generation_time_ms=1200.0,
        finish_reason="stop",
    )
    
    retrieval_metadata = RetrievalMetadata(
        confidence_score=0.87,
        confidence_level=ConfidenceLevel.HIGH,
        retrieval_time_ms=150.0,
        total_chunks_retrieved=3,
        namespace="supply_chain_test",
    )
    
    response = processor.process(
        generated=generated,
        chunks=sample_chunks,
        retrieval_metadata=retrieval_metadata,
        mode=GenerationMode.CONTEXT_AWARE,
        request_id="test-001",
        generation_id="gen-001",
    )
    
    print(f"\n📊 Processing Results:")
    print(f"   Cleaned answer length: {len(response.answer)} chars")
    print(f"   Citations extracted: {len(response.citations)}")
    print(f"   Is grounded: {response.is_grounded}")
    print(f"   Grounding confidence: {response.grounding_confidence:.2f}")
    print(f"   Warnings: {response.warnings}")
    
    print(f"\n📋 Extracted Citations:")
    for citation in response.citations:
        formatted = processor.citation_formatter.format_citation(citation)
        print(f"   [{citation.index}] {formatted}")
        print(f"       Chunk ID: {citation.chunk_id or 'unmatched'}")
        print(f"       Position: {citation.position_start}-{citation.position_end}")
    
    print(f"\n✅ Post Processor ready for integration")