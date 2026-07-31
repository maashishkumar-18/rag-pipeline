"""
Step 5: Chunk Enrichment
Injects source traceability (slide/page references) directly into chunk content
so each chunk is self-contained for retrieval and generation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import tiktoken

from src.ingestion.chunker import Chunk

# Constants
ENRICHMENT_VERSION = "1.0"
DEFAULT_MAX_PREFIX_LENGTH = 200
EMBEDDING_MODEL = "text-embedding-3-small"
METADATA_VERSION = "1.0"
PIPELINE_STAGE = "Step 5"


@dataclass
class EnricherConfig:
    """Configuration for the chunk enricher."""
    add_source_prefix: bool = True
    validate_chunks: bool = True
    max_prefix_length: int = DEFAULT_MAX_PREFIX_LENGTH


@dataclass
class ValidationReport:
    """
    Structured validation report for enriched chunks.
    """
    valid: bool
    total_chunks: int
    errors: List[str]
    warnings: List[str]


@dataclass
class EnrichedChunk:
    """
    A fully enriched chunk ready for embedding and vector storage.
    Every chunk is self-contained with all metadata and source references.
    """
    chunk_id: str
    content: str                                    # Enriched text with source prefix
    raw_content: str                                # Original content without enrichment
    chunk_type: str
    page_start: int
    page_end: int
    topic: str
    token_count: int                                # Token count of enriched content
    element_ids: List[int]
    parent_chunk_id: str

    # Document metadata (carried forward)
    filename: str
    file_type: str
    course_name: str
    chapter_title: str
    subject_area: str
    global_context: str
    key_topics: List[str]
    extraction_confidence: float

    # Enrichment metadata
    source_prefix: Optional[str] = None
    enrichment_version: str = ENRICHMENT_VERSION
    pipeline_stage: str = PIPELINE_STAGE

    # Free-form document category, set per-document at ingestion time (see
    # ChunkEnricher.enrich(document_category=...)), not derived from chunk
    # content — every chunk from one upload shares the same value. Callers
    # can use this for whatever corpus-specific taxonomy they need (e.g.
    # "manual" vs "changelog", or leave it unset) — retrieval can filter on
    # it via metadata_filter.py without the ingestion/enrichment layer
    # needing to know what the categories mean.
    document_category: Optional[str] = None
    
    # Embedding metadata (to be populated by EmbeddingGenerator)
    embedding_ready: bool = False
    embedding_model: str = EMBEDDING_MODEL
    embedding_dimensions: Optional[int] = None
    embedding_version: Optional[str] = None
    metadata_version: str = METADATA_VERSION


class ChunkEnricher:
    """
    Enriches chunks with source traceability prefixes and validates completeness.

    Each chunk gets a source prefix like:
    "[Course: CS101 | Chapter: Trees | Slides 5-7 | Topic: Binary Search Trees]"
    """

    def __init__(self, config: Optional[EnricherConfig] = None):
        """
        Initialize the enricher with optional configuration.

        Args:
            config: Optional EnricherConfig object
        """
        self.config = config or EnricherConfig()
        
        # Pre-compute max field length for optimization
        self.max_field_len = self.config.max_prefix_length // 4
        
        # Initialize token counter
        self.encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
        
        # Storage for last validation report
        self._last_validation_report: Optional[ValidationReport] = None

    def enrich(self, chunks: List[Chunk], document_category: Optional[str] = None) -> List[EnrichedChunk]:
        """
        Enrich all chunks with source traceability and validate completeness.

        Args:
            chunks: List of Chunk objects from Step 4
            document_category: Optional free-form category for the document
                this whole batch of chunks came from. Stamped onto every
                chunk unchanged; meaning is entirely caller-defined.

        Returns:
            List of EnrichedChunk objects ready for embedding
        """
        # Enrich all chunks using list comprehension
        enriched_chunks = [
            self._enrich_single(chunk, document_category)
            for chunk in chunks
        ]

        # Validate if configured and store report
        if self.config.validate_chunks:
            self._last_validation_report = self.get_validation_report(enriched_chunks)

        return enriched_chunks

    def _enrich_single(self, chunk: Chunk, document_category: Optional[str] = None) -> EnrichedChunk:
        """
        Enrich a single chunk with source traceability prefix.
        """
        # Build source prefix
        source_prefix = self._build_source_prefix(chunk)

        # Build enriched content
        enriched_content = self._build_enriched_content(chunk, source_prefix)

        # Count tokens for enriched content
        token_count = self._count_tokens(enriched_content)

        return EnrichedChunk(
            chunk_id=chunk.chunk_id,
            content=enriched_content,
            raw_content=chunk.content,
            chunk_type=chunk.chunk_type,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            topic=chunk.topic,
            token_count=token_count,
            element_ids=chunk.element_ids,
            parent_chunk_id=chunk.parent_chunk_id,
            filename=chunk.filename,
            file_type=chunk.file_type,
            course_name=chunk.course_name,
            chapter_title=chunk.chapter_title,
            subject_area=chunk.subject_area,
            global_context=chunk.global_context,
            key_topics=chunk.key_topics,
            extraction_confidence=chunk.extraction_confidence,
            source_prefix=source_prefix,
            document_category=document_category,
        )

    def _build_source_prefix(self, chunk: Chunk) -> Optional[str]:
        """
        Build a source traceability prefix for a chunk.

        Format depends on document type:
        - PPTX: "[Course: X | Chapter: Y | Slides 5-7 | Topic: Z]"
        - PDF/DOCX: "[Course: X | Chapter: Y | Pages 5-7 | Topic: Z]"
        """
        if not self.config.add_source_prefix:
            return None

        return self._format_source_prefix(chunk)

    def _format_source_prefix(self, chunk: Chunk) -> str:
        """
        Format the source prefix with proper truncation of individual fields.
        """
        parts = []

        # Course name
        if chunk.course_name and chunk.course_name != "Unknown":
            course = self._truncate_field(chunk.course_name, self.max_field_len)
            parts.append(f"Course: {course}")

        # Chapter title
        if chunk.chapter_title and chunk.chapter_title != "Unknown":
            chapter = self._truncate_field(chunk.chapter_title, self.max_field_len)
            parts.append(f"Chapter: {chapter}")

        # Page/Slide range
        location_label = "Slides" if chunk.file_type == "pptx" else "Pages"
        if chunk.page_start == chunk.page_end:
            parts.append(f"{location_label}: {chunk.page_start}")
        else:
            parts.append(f"{location_label}: {chunk.page_start}-{chunk.page_end}")

        # Topic
        if chunk.topic and chunk.topic != "General":
            topic = self._truncate_field(chunk.topic, self.max_field_len)
            parts.append(f"Topic: {topic}")

        prefix = " | ".join(parts)
        prefix = f"[{prefix}]"

        # Final safety truncation (should rarely trigger now)
        if len(prefix) > self.config.max_prefix_length:
            prefix = prefix[:self.config.max_prefix_length - 3] + "...]"

        return prefix

    def _truncate_field(self, text: str, max_length: int) -> str:
        """Truncate a field to max_length if needed."""
        if len(text) > max_length:
            return text[:max_length - 3] + "..."
        return text

    def _build_enriched_content(self, chunk: Chunk, source_prefix: Optional[str]) -> str:
        """
        Build the final enriched content by prepending source prefix.
        Context is NOT added to content (stored in metadata only).
        """
        # Check if already enriched with the same prefix
        if source_prefix and chunk.content.startswith(source_prefix):
            return chunk.content
        
        # If no source prefix, just return original content
        if not source_prefix:
            return chunk.content

        # Add source prefix to content
        return f"{source_prefix}\n{chunk.content}"

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the embedding model's tokenizer."""
        return len(self.encoding.encode(text))

    def get_validation_report(self, enriched_chunks: List[EnrichedChunk]) -> ValidationReport:
        """
        Validate that all chunks meet minimum requirements.
        Returns structured validation report.

        Returns:
            ValidationReport object with valid flag, total chunks, errors, and warnings
        """
        if not enriched_chunks:
            return ValidationReport(
                valid=False,
                total_chunks=0,
                errors=["No chunks to validate"],
                warnings=[]
            )

        errors = []
        warnings = []

        for chunk in enriched_chunks:
            # Check required fields
            if not chunk.chunk_id:
                errors.append("Missing chunk_id")
            
            if not chunk.content or len(chunk.content.strip()) < 10:
                errors.append(f"{chunk.chunk_id}: Content too short ({len(chunk.content)} chars)")
            
            if chunk.token_count == 0:
                warnings.append(f"{chunk.chunk_id}: Token count is 0")
            
            if chunk.course_name == "Unknown" and chunk.chapter_title == "Unknown":
                warnings.append(f"{chunk.chunk_id}: Both course and chapter are Unknown")
            
            if chunk.page_start == 0 and chunk.page_end == 0:
                warnings.append(f"{chunk.chunk_id}: Page range is 0-0")
            
            # Check enrichment
            if self.config.add_source_prefix:
                if not chunk.source_prefix:
                    errors.append(f"{chunk.chunk_id}: Missing source prefix")
                elif chunk.chunk_type != "metadata":  # Skip metadata chunks
                    # Check if content starts with the source prefix
                    if not chunk.content.startswith(chunk.source_prefix):
                        warnings.append(f"{chunk.chunk_id}: Content doesn't start with source prefix")

        return ValidationReport(
            valid=len(errors) == 0,
            total_chunks=len(enriched_chunks),
            errors=errors,
            warnings=warnings
        )

    def get_last_validation_report(self) -> Optional[ValidationReport]:
        """
        Get the last validation report generated during enrich().
        
        Returns:
            ValidationReport if validation was run, None otherwise
        """
        return self._last_validation_report

    def get_enrichment_summary(self, enriched_chunks: List[EnrichedChunk]) -> Dict[str, Any]:
        """
        Generate a summary of enriched chunks for pipeline diagnostics.
        """
        if not enriched_chunks:
            return {"total_chunks": 0}

        total_tokens = sum(c.token_count for c in enriched_chunks)
        courses = set(c.course_name for c in enriched_chunks if c.course_name != "Unknown")
        chapters = set(c.chapter_title for c in enriched_chunks if c.chapter_title != "Unknown")
        topics = set(c.topic for c in enriched_chunks if c.topic != "General")

        return {
            "total_chunks": len(enriched_chunks),
            "total_tokens": total_tokens,
            "avg_tokens": total_tokens / len(enriched_chunks),
            "unique_courses": list(courses),
            "unique_chapters": list(chapters),
            "unique_topics": list(topics),
            "file_types": list(set(c.file_type for c in enriched_chunks)),
            "split_chunks": sum(1 for c in enriched_chunks if c.parent_chunk_id),
            "page_range": f"{min(c.page_start for c in enriched_chunks)}-{max(c.page_end for c in enriched_chunks)}",
            "enrichment_version": ENRICHMENT_VERSION,
            "embedding_model": EMBEDDING_MODEL
        }
    
    @staticmethod
    def to_metadata(chunk: EnrichedChunk) -> Dict[str, Any]:
        """
        Convert enriched chunk to metadata dictionary for vector store.

        This is used by VectorStore to extract metadata when adding chunks.
        
        Args:
            chunk: EnrichedChunk object

        Returns:
            Dictionary of metadata fields for vector store
        """
        metadata = {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "raw_content": chunk.raw_content,
            "filename": chunk.filename,
            "file_type": chunk.file_type,
            "topic": chunk.topic,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "course_name": chunk.course_name,
            "chapter_title": chunk.chapter_title,
            "subject_area": chunk.subject_area,
            "global_context": chunk.global_context,
            "key_topics": chunk.key_topics,
            "parent_chunk_id": chunk.parent_chunk_id,
            "chunk_type": chunk.chunk_type,
            "token_count": chunk.token_count,
            "source_prefix": chunk.source_prefix,
            "enrichment_version": chunk.enrichment_version,
            "metadata_version": chunk.metadata_version,
        }
        # Only stamp document_category when the caller actually set one —
        # Pinecone metadata fields are typically expected to be non-null,
        # and most corpora won't need this field at all.
        if chunk.document_category is not None:
            metadata["document_category"] = chunk.document_category
        return metadata


# ============================================================================
# Step 5 Validation
# ============================================================================

if __name__ == "__main__":
    import os
    from pathlib import Path
    from src.ingestion.parser import DocumentParser
    from src.ingestion.cleaner import TextCleaner
    from src.ingestion.metadata_extractor import MetadataExtractor
    from src.ingestion.chunker import SemanticChunker

    # Full pipeline
    parser = DocumentParser()
    cleaner = TextCleaner()
    extractor = MetadataExtractor()
    chunker = SemanticChunker()

    # Create enricher with default config
    enricher = ChunkEnricher()

    # Bundled sample corpus file — swap in your own files to test other formats.
    SAMPLE_FILE = str(Path(__file__).resolve().parents[2] / "sample_data" / "sample_lecture.pptx")
    test_files = [
        SAMPLE_FILE
    ]

    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n{'='*60}")
            print(f"Enriching: {file_path}")
            print(f"{'='*60}")

            # Steps 1-4
            parsed = parser.parse(file_path)
            cleaned = cleaner.clean(parsed)
            metadata = extractor.extract_with_retry(cleaned)
            chunks = chunker.chunk(cleaned, metadata)

            # Step 5
            enriched = enricher.enrich(chunks)

            # Get validation report (stored in enricher)
            validation = enricher.get_last_validation_report()
            if validation:
                print(f"\n📋 Validation Report:")
                print(f"   Valid: {validation.valid}")
                print(f"   Total chunks: {validation.total_chunks}")
                if validation.errors:
                    print(f"   Errors: {len(validation.errors)}")
                    for error in validation.errors[:3]:
                        print(f"     - {error}")
                if validation.warnings:
                    print(f"   Warnings: {len(validation.warnings)}")
                    for warning in validation.warnings[:3]:
                        print(f"     - {warning}")

            # Summary
            summary = enricher.get_enrichment_summary(enriched)
            print(f"\n📊 Enrichment Summary:")
            print(f"   Total chunks: {summary['total_chunks']}")
            print(f"   Total tokens: {summary['total_tokens']}")
            print(f"   Avg tokens/chunk: {summary['avg_tokens']:.1f}")
            print(f"   Split chunks: {summary['split_chunks']}")
            print(f"   Courses: {summary['unique_courses']}")
            print(f"   Chapters: {summary['unique_chapters']}")
            print(f"   Topics: {summary['unique_topics'][:5]}...")

            # Show sample enriched content
            print(f"\n📄 Sample Enriched Chunk:")
            if enriched:
                sample = enriched[0]
                print(f"   Source Prefix: {sample.source_prefix}")
                print(f"   Content Preview: {sample.content[:200]}...")
                print(f"   Token Count: {sample.token_count}")
                print(f"   Full Metadata: course={sample.course_name}, chapter={sample.chapter_title}")
                print(f"   Embedding Ready: {sample.embedding_ready}")
                print(f"   Embedding Model: {sample.embedding_model}")
                
                # Show metadata export
                print(f"\n📦 Metadata Export (for vector store):")
                metadata_dict = enricher.to_metadata(sample)
                for key, value in list(metadata_dict.items())[:7]:
                    print(f"   {key}: {value}")
        else:
            print(f"⚠️  Test file not found: {file_path}")