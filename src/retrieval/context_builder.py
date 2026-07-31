"""
Context Builder
Assembles reranked chunks into structured, generation-ready context.

Organizes content by template (definition → explanation → example),
injects source citations, and enforces token limits.

Configuration-driven — all templates, labels, and assembly rules
loaded from config/retrieval/context_builder.yaml with environment
variable overrides.
"""

import os
import re
import yaml
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

from src.retrieval.config import ContextTemplate
from src.common.types import RetrievedChunk, CitationLocationType
from src.retrieval.reranker import RerankedChunk

# Optional imports for tokenization
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TemplateConfig:
    """Configuration for a single context template."""
    order: List[str] = field(default_factory=list)
    section_labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class ClassificationHeuristic:
    """Rule-based patterns for chunk classification."""
    patterns: List[str] = field(default_factory=list)


@dataclass
class ContextBuilderConfig:
    """Configuration for context building — loaded from YAML."""
    templates: Dict[str, TemplateConfig] = field(default_factory=dict)
    classification_enabled: bool = True
    classification_method: str = "metadata"
    heuristics: Dict[str, ClassificationHeuristic] = field(default_factory=dict)
    citations_enabled: bool = True
    citation_format: str = "[{course} | {chapter} | {location} {page}]"
    include_citations_in_context: bool = True
    include_citations_separate: bool = False
    max_total_tokens: int = 3000
    max_chunks: int = 5
    include_metadata_header: bool = True
    metadata_header_format: str = "Context from: {course} - {chapter}"
    separator: str = "\n\n---\n\n"
    truncation_marker: str = "... [truncated]"
    sort_by: str = "template_order"
    remove_duplicate_lines: bool = True
    normalize_whitespace: bool = True
    strip_source_prefix: bool = False
    tokenizer_model: str = "gpt-4o"  # Model for token counting
    diversity_enabled: bool = True   # Whether to enforce chunk diversity
    diversity_threshold: float = 0.3 # Similarity threshold for diversity
    truncate_at_sentence: bool = True # Whether to truncate at sentence boundaries

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "ContextBuilderConfig":
        """
        Load configuration from YAML file.
        
        Priority:
        1. Explicit path argument
        2. RAGPIPE_CONTEXT_BUILDER_CONFIG env var
        3. config/retrieval/context_builder.yaml (default)
        """
        config_path = (
            path
            or os.getenv("RAGPIPE_CONTEXT_BUILDER_CONFIG")
            or str(Path(__file__).parent.parent.parent / "config" / "retrieval" / "context_builder.yaml")
        )
        
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
        else:
            data = cls._default_config()
        
        data = cls._apply_env_overrides(data)
        
        # Parse templates
        templates = {}
        for name, template_data in data.get("templates", {}).items():
            templates[name] = TemplateConfig(
                order=template_data.get("order", []),
                section_labels=template_data.get("section_labels", {}),
            )
        
        # Parse heuristics
        heuristics = {}
        for name, h_data in data.get("classification", {}).get("heuristics", {}).items():
            heuristics[name] = ClassificationHeuristic(
                patterns=h_data.get("patterns", [])
            )
        
        assembly = data.get("assembly", {})
        cleaning = data.get("cleaning", {})
        citations = data.get("citations", {})
        classification = data.get("classification", {})
        diversity = data.get("diversity", {})
        tokenization = data.get("tokenization", {})
        
        return cls(
            templates=templates,
            classification_enabled=bool(classification.get("enabled", True)),
            classification_method=str(classification.get("method", "metadata")),
            heuristics=heuristics,
            citations_enabled=bool(citations.get("enabled", True)),
            citation_format=str(citations.get("format", "[{course} | {chapter} | {location} {page}]")),
            include_citations_in_context=bool(citations.get("include_in_context", True)),
            include_citations_separate=bool(citations.get("include_separate_section", False)),
            max_total_tokens=int(assembly.get("max_total_tokens", 3000)),
            max_chunks=int(assembly.get("max_chunks", 5)),
            include_metadata_header=bool(assembly.get("include_metadata_header", True)),
            metadata_header_format=str(assembly.get("metadata_header_format", "Context from: {course} - {chapter}")),
            separator=str(assembly.get("separator", "\n\n---\n\n")),
            truncation_marker=str(assembly.get("truncation_marker", "... [truncated]")),
            sort_by=str(assembly.get("sort_by", "template_order")),
            remove_duplicate_lines=bool(cleaning.get("remove_duplicate_lines", True)),
            normalize_whitespace=bool(cleaning.get("normalize_whitespace", True)),
            strip_source_prefix=bool(cleaning.get("strip_source_prefix_from_content", False)),
            tokenizer_model=str(tokenization.get("model", "gpt-4o")),
            diversity_enabled=bool(diversity.get("enabled", True)),
            diversity_threshold=float(diversity.get("threshold", 0.3)),
            truncate_at_sentence=bool(assembly.get("truncate_at_sentence", True)),
        )
    
    @classmethod
    def _apply_env_overrides(cls, data: Dict) -> Dict:
        """Apply environment variable overrides."""
        env_mapping = {
            "RAGPIPE_CONTEXT_MAX_TOKENS": ("assembly", "max_total_tokens"),
            "RAGPIPE_CONTEXT_MAX_CHUNKS": ("assembly", "max_chunks"),
            "RAGPIPE_CONTEXT_SORT": ("assembly", "sort_by"),
            "RAGPIPE_CONTEXT_CITATIONS": ("citations", "enabled"),
            "RAGPIPE_CONTEXT_DIVERSITY": ("diversity", "enabled"),
            "RAGPIPE_CONTEXT_TOKENIZER": ("tokenization", "model"),
        }
        
        for env_var, (section, key) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in data:
                    data[section] = {}
                try:
                    if key == "enabled":
                        data[section][key] = value.lower() in ["true", "1", "yes"]
                    else:
                        data[section][key] = int(value)
                except ValueError:
                    data[section][key] = value
        
        return data
    
    @staticmethod
    def _default_config() -> Dict:
        """Minimal fallback configuration."""
        return {
            "templates": {
                "raw": {
                    "order": [],
                    "section_labels": {},
                }
            },
            "classification": {
                "enabled": False,
                "method": "metadata",
            },
            "citations": {
                "enabled": True,
                "format": "[{course} | {chapter} | {location} {page}]",
                "include_in_context": True,
            },
            "assembly": {
                "max_total_tokens": 3000,
                "max_chunks": 5,
                "separator": "\n\n---\n\n",
                "sort_by": "score",
                "truncate_at_sentence": True,
            },
            "diversity": {
                "enabled": True,
                "threshold": 0.3,
            },
            "tokenization": {
                "model": "gpt-4o",
            },
        }
    
    def get_template(self, template_name: str) -> TemplateConfig:
        """Get a template by name, falling back to raw."""
        template_key = template_name.value if isinstance(template_name, ContextTemplate) else str(template_name)
        return self.templates.get(template_key, TemplateConfig())


# ============================================================================
# Token Counter
# ============================================================================

class TokenCounter:
    """
    Handles token counting for context assembly.
    
    Uses the specified tokenizer for accurate counting, with fallback
    to character-based estimation.
    """
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self._encoder = None
        
        if TIKTOKEN_AVAILABLE:
            try:
                self._encoder = tiktoken.encoding_for_model(model_name)
            except Exception:
                # Fall back to cl100k_base if specific model encoding not found
                try:
                    self._encoder = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self._encoder = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a string."""
        if not text:
            return 0
        
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                # Fall back to estimation if encoding fails
                return self._estimate_tokens(text)
        
        # Fall back to estimation
        return self._estimate_tokens(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens using character-based approximation."""
        # Rough approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for multiple texts efficiently."""
        if self._encoder is not None:
            try:
                # Use batch encoding if available
                combined = "\n".join(texts)
                all_tokens = self._encoder.encode(combined)
                
                # This is a simplified approach; for precise per-text counts,
                # we'd need to track positions. For our use case, individual
                # counts are fine.
                return [self.count_tokens(t) for t in texts]
            except Exception:
                pass
        
        return [self._estimate_tokens(t) for t in texts]


# ============================================================================
# Context Builder
# ============================================================================

@dataclass
class ClassifiedChunk:
    """A chunk with its classification for template ordering."""
    chunk: RerankedChunk
    section_type: str           # "definition", "example", "formula", etc.
    citation: str               # Formatted source citation
    display_content: str        # Content with citation prepended
    token_count: int = 0        # Token count of display_content


@dataclass
class AssembledContext:
    """Final assembled context ready for the generation layer."""
    formatted_context: str
    chunks_used: int
    tokens_used: int
    sections: List[str]
    citations: List[str]
    template_used: str
    metadata_header: str = ""
    truncated: bool = False
    diversity_applied: bool = False
    # Raw per-chunk data backing `citations`/`formatted_context`, in the same
    # order as `citations`. This is what downstream layers (e.g. the
    # Application layer's Generation bridge) must use to build real
    # RetrievedChunk objects — never parse `citations` strings for structure,
    # they are display-only formatted text, not a data contract.
    chunks: List[RetrievedChunk] = field(default_factory=list)


class ContextBuilder:
    """
    Assembles reranked chunks into structured, generation-ready context.
    
    Organizes by template, classifies chunks into sections,
    injects citations, enforces token limits, and ensures
    content diversity.
    
    Configuration-driven — all templates and rules from YAML.
    """
    
    def __init__(
        self,
        config: Optional[ContextBuilderConfig] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the context builder.
        
        Args:
            config: ContextBuilderConfig object
            config_path: Path to YAML config file
        """
        self.config = config or ContextBuilderConfig.from_yaml(config_path)
        self.token_counter = TokenCounter(self.config.tokenizer_model)
        
        # Pre-compile heuristic patterns
        self._compiled_heuristics = {}
        for section_type, heuristic in self.config.heuristics.items():
            self._compiled_heuristics[section_type] = [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in heuristic.patterns
            ]
    
    def build(
        self,
        chunks: List[RerankedChunk],
        template: ContextTemplate,
        pipeline_name: str = "learning"
    ) -> AssembledContext:
        """
        Assemble reranked chunks into structured context.
        
        Args:
            chunks: Reranked chunks from the reranker
            template: Context template for organization
            pipeline_name: Pipeline name for diagnostics
            
        Returns:
            AssembledContext with formatted context string
        """
        if not chunks:
            return AssembledContext(
                formatted_context="",
                chunks_used=0,
                tokens_used=0,
                sections=[],
                citations=[],
                template_used=template.value,
                truncated=False,
                diversity_applied=False
            )
        
        template_config = self.config.get_template(template)
        
        # Step 1: Classify chunks into sections
        classified = self._classify_chunks(chunks, template_config)
        
        # Step 2: Sort by template order (primary) and score (secondary)
        if self.config.sort_by == "template_order" and template_config.order:
            classified = self._sort_by_template(classified, template_config.order)
        elif self.config.sort_by == "score":
            classified.sort(key=lambda c: c.chunk.rerank_score, reverse=True)
        
        # Step 3: Apply diversity filtering (if enabled)
        diversity_applied = False
        if self.config.diversity_enabled:
            classified, diversity_applied = self._apply_diversity_filtering(classified)
        
        # Step 4: Limit to max chunks
        classified = classified[:self.config.max_chunks]
        
        # Step 5: Pre-calculate token counts
        for c in classified:
            c.token_count = self.token_counter.count_tokens(c.display_content)
        
        # Step 6: Group by section type
        sections = self._group_by_section(classified, template_config)
        
        # Step 7: Build formatted context with token limiting
        formatted, tokens_used, truncated = self._build_formatted_context(
            sections, template_config, chunks
        )
        
        # Step 8: Extract citations
        citations = [c.citation for c in classified if c.citation]

        # Step 9: Build raw per-chunk data (real content/score/source —
        # what the Generation layer's `RetrievedChunk` contract needs)
        retrieved_chunks = [self._to_retrieved_chunk(c) for c in classified]

        # Step 10: Build metadata header
        metadata_header = self._build_metadata_header(chunks)

        return AssembledContext(
            formatted_context=formatted,
            chunks_used=len(classified),
            tokens_used=tokens_used,
            sections=list(sections.keys()),
            citations=citations,
            template_used=template.value,
            metadata_header=metadata_header,
            truncated=truncated,
            diversity_applied=diversity_applied,
            chunks=retrieved_chunks,
        )

    def _to_retrieved_chunk(self, classified: ClassifiedChunk) -> RetrievedChunk:
        """
        Convert a classified chunk to the shared RetrievedChunk contract.

        Mirrors the exact metadata keys `_build_citation` reads, so the
        structured data here always agrees with the formatted citation
        string for the same chunk.
        """
        chunk = classified.chunk
        metadata = chunk.metadata
        file_type = metadata.get("file_type", "")
        location_type = CitationLocationType.SLIDE if file_type == "pptx" else CitationLocationType.PAGE

        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            raw_content=chunk.raw_content,
            score=chunk.rerank_score,
            course_name=metadata.get("course_name", ""),
            chapter_title=metadata.get("chapter_title", ""),
            topic=metadata.get("topic", ""),
            location_type=location_type,
            location_start=self._to_display_int(metadata.get("page_start")) or 0,
            location_end=self._to_display_int(metadata.get("page_end")) or 0,
            chunk_type=metadata.get("chunk_type", ""),
            filename=metadata.get("filename", ""),
            file_type=file_type,
            source_prefix=metadata.get("source_prefix", ""),
        )
    
    # ========================================================================
    # Diversity Filtering
    # ========================================================================
    
    def _apply_diversity_filtering(
        self,
        classified: List[ClassifiedChunk]
    ) -> Tuple[List[ClassifiedChunk], bool]:
        """
        Apply diversity filtering to ensure chunks from different sources.
        
        Uses a simple heuristic: prefer chunks from different pages/sections.
        """
        if len(classified) <= 1:
            return classified, False
        
        # Group chunks by source (page, section, or file)
        source_groups: Dict[str, List[ClassifiedChunk]] = defaultdict(list)
        
        for c in classified:
            # Create a source key based on available metadata
            metadata = c.chunk.metadata
            page = metadata.get("page_start", "")
            chapter = metadata.get("chapter_title", "")
            chunk_type = metadata.get("chunk_type", "")
            
            # Prefer page-level diversity, then chapter, then chunk type
            if page:
                key = f"page_{page}"
            elif chapter:
                key = f"chapter_{chapter}"
            elif chunk_type:
                key = f"type_{chunk_type}"
            else:
                key = "unknown"
            
            source_groups[key].append(c)
        
        # If all chunks are from the same source, no diversity to apply
        if len(source_groups) <= 1:
            return classified, False
        
        # Diversify: take best from each source, then fill remaining slots
        diversified = []
        source_keys = list(source_groups.keys())
        
        # Round-robin: take the top chunk from each source
        source_iterators = {
            key: iter(sorted(group, key=lambda c: c.chunk.rerank_score, reverse=True))
            for key, group in source_groups.items()
        }
        
        # Keep track of which sources have been used
        active_sources = set(source_keys)
        remaining_slots = self.config.max_chunks
        
        while active_sources and remaining_slots > 0:
            # Cycle through active sources
            for source in list(active_sources):
                if remaining_slots <= 0:
                    break
                
                try:
                    chunk = next(source_iterators[source])
                    diversified.append(chunk)
                    remaining_slots -= 1
                except StopIteration:
                    # This source is exhausted
                    active_sources.remove(source)
        
        # If we didn't fill all slots, take the next best from any source
        if remaining_slots > 0 and len(diversified) < len(classified):
            # Get chunks that weren't selected
            selected_ids = {c.chunk.chunk_id for c in diversified}
            remaining_chunks = [c for c in classified if c.chunk.chunk_id not in selected_ids]
            
            # Add more chunks sorted by score
            for c in remaining_chunks[:remaining_slots]:
                diversified.append(c)
        
        return diversified, True
    
    # ========================================================================
    # Chunk Classification
    # ========================================================================
    
    def _classify_chunks(
        self,
        chunks: List[RerankedChunk],
        template: TemplateConfig
    ) -> List[ClassifiedChunk]:
        """Classify each chunk into a section type."""
        classified = []
        
        for chunk in chunks:
            # Determine section type
            section_type = self._determine_section_type(chunk)
            
            # Build citation
            citation = self._build_citation(chunk)
            
            # Prepare display content
            display_content = self._prepare_display_content(chunk, citation)
            
            classified.append(ClassifiedChunk(
                chunk=chunk,
                section_type=section_type,
                citation=citation,
                display_content=display_content,
                token_count=0  # Will be set later
            ))
        
        return classified
    
    def _determine_section_type(self, chunk: RerankedChunk) -> str:
        """Determine which section a chunk belongs to."""
        # Method 1: Use metadata chunk_type from ingestion
        if self.config.classification_method == "metadata":
            chunk_type = chunk.metadata.get("chunk_type", "")
            if chunk_type in self.config.heuristics:
                return chunk_type
        
        # Method 2: Heuristic pattern matching
        if self.config.classification_enabled:
            content = chunk.raw_content or chunk.content
            
            for section_type, patterns in self._compiled_heuristics.items():
                for pattern in patterns:
                    if pattern.search(content):
                        return section_type
        
        # Fallback
        return "general"
    
    @staticmethod
    def _to_display_int(value: Any) -> Optional[int]:
        """Coerce a possibly-float metadata number (e.g. Pinecone's 6.0) to a clean int for display."""
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _build_citation(self, chunk: RerankedChunk) -> str:
        """Build a formatted citation string for a chunk."""
        if not self.config.citations_enabled:
            return ""
        
        metadata = chunk.metadata

        course = metadata.get("course_name", "")
        chapter = metadata.get("chapter_title", "")
        # Pinecone stores all numeric metadata as float, so page_start/page_end
        # come back as e.g. 6.0 even though they're conceptually ints — cast
        # back before formatting to avoid citations like "Slide 6.0-7.0".
        page_start = self._to_display_int(metadata.get("page_start"))
        page_end = self._to_display_int(metadata.get("page_end"))
        file_type = metadata.get("file_type", "")

        # Determine location label
        location = "Slide" if file_type == "pptx" else "Page"

        # Format page range
        if page_start and page_end and page_start != page_end:
            page = f"{page_start}-{page_end}"
        elif page_start:
            page = str(page_start)
        else:
            page = "N/A"
        
        return self.config.citation_format.format(
            course=course or "Unknown",
            chapter=chapter or "Unknown",
            location=location,
            page=page
        )
    
    def _prepare_display_content(self, chunk: RerankedChunk, citation: str) -> str:
        """Prepare chunk content for display with optional citation."""
        content = chunk.content
        
        # Optionally strip source prefix that was added during enrichment
        if self.config.strip_source_prefix:
            source_prefix = chunk.metadata.get("source_prefix", "")
            if source_prefix and content.startswith(source_prefix):
                content = content[len(source_prefix):].strip()
        
        # Add citation if enabled
        if self.config.include_citations_in_context and citation:
            content = f"{citation}\n{content}"
        
        return content
    
    # ========================================================================
    # Sorting & Grouping
    # ========================================================================
    
    def _sort_by_template(
        self,
        classified: List[ClassifiedChunk],
        order: List[str]
    ) -> List[ClassifiedChunk]:
        """Sort chunks according to template section order."""
        # Create priority map
        priority = {section: i for i, section in enumerate(order)}
        
        # Sort: by template priority, then by rerank score within same section
        classified.sort(key=lambda c: (
            priority.get(c.section_type, len(order)),
            -c.chunk.rerank_score
        ))
        
        return classified
    
    def _group_by_section(
        self,
        classified: List[ClassifiedChunk],
        template: TemplateConfig
    ) -> Dict[str, List[ClassifiedChunk]]:
        """Group classified chunks by section type."""
        sections: Dict[str, List[ClassifiedChunk]] = {}
        
        for c in classified:
            if c.section_type not in sections:
                sections[c.section_type] = []
            sections[c.section_type].append(c)
        
        # If template has an order, preserve it
        if template.order:
            ordered_sections = {}
            for section_type in template.order:
                if section_type in sections:
                    ordered_sections[section_type] = sections[section_type]
            # Add any remaining sections not in template order
            for section_type, items in sections.items():
                if section_type not in ordered_sections:
                    ordered_sections[section_type] = items
            return ordered_sections
        
        return sections
    
    # ========================================================================
    # Context Assembly
    # ========================================================================
    
    def _build_formatted_context(
        self,
        sections: Dict[str, List[ClassifiedChunk]],
        template: TemplateConfig,
        original_chunks: List[RerankedChunk]
    ) -> Tuple[str, int, bool]:
        """
        Build the final formatted context string with token limiting.
        
        Returns:
            Tuple of (formatted_context, token_count, was_truncated)
        """
        parts = []
        total_tokens = 0
        max_tokens = self.config.max_total_tokens
        
        # Add metadata header
        if self.config.include_metadata_header:
            header = self._build_metadata_header(original_chunks)
            if header:
                header_tokens = self.token_counter.count_tokens(header)
                if total_tokens + header_tokens <= max_tokens:
                    parts.append(header)
                    total_tokens += header_tokens
                else:
                    # Can't even include header, return empty
                    return "", 0, True
        
        # Build each section incrementally, checking token budget
        truncated = False
        
        for section_type, items in sections.items():
            # Section label
            label = template.section_labels.get(section_type, section_type.upper().replace("_", " "))
            section_label = f"## {label}"
            
            label_tokens = self.token_counter.count_tokens(section_label)
            if total_tokens + label_tokens > max_tokens:
                truncated = True
                break
            
            section_parts = [section_label]
            section_tokens = label_tokens
            
            for item in items:
                # Clean content
                content = item.display_content
                if self.config.normalize_whitespace:
                    content = self._normalize_whitespace(content)
                if self.config.remove_duplicate_lines:
                    content = self._remove_duplicate_lines(content)
                
                content_tokens = self.token_counter.count_tokens(content)
                
                # Check if we can add this chunk
                if section_tokens + content_tokens > max_tokens - total_tokens:
                    # Try to truncate at sentence boundary if possible
                    if self.config.truncate_at_sentence:
                        truncated_content, truncated_tokens = self._truncate_at_sentence(
                            content, 
                            max_tokens - total_tokens - section_tokens
                        )
                        if truncated_content:
                            section_parts.append(truncated_content)
                            section_tokens += truncated_tokens
                            truncated = True
                    break
                
                section_parts.append(content)
                section_tokens += content_tokens
            
            # Add this section if we have content
            if len(section_parts) > 1:
                section_text = "\n".join(section_parts)
                parts.append(section_text)
                total_tokens += section_tokens
        
        # Join sections
        formatted = self.config.separator.join(parts)
        
        # Add truncation marker if truncated
        if truncated:
            formatted += self.config.truncation_marker
        
        return formatted, total_tokens, truncated
    
    def _truncate_at_sentence(self, text: str, max_tokens: int) -> Tuple[str, int]:
        """
        Truncate text at a sentence boundary within the token budget.
        
        Returns:
            Tuple of (truncated_text, token_count)
        """
        if max_tokens <= 0:
            return "", 0
        
        # Split into sentences (simple heuristic)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        truncated_parts = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.token_counter.count_tokens(sentence)
            if current_tokens + sentence_tokens <= max_tokens:
                truncated_parts.append(sentence)
                current_tokens += sentence_tokens
            else:
                # Can't fit this whole sentence; try to add partial sentence
                # but only if we have at least some content
                if truncated_parts:
                    break
                else:
                    # If we can't even fit one sentence, take a character-based truncation
                    chars_to_take = max_tokens * 4  # Approximate
                    truncated = text[:chars_to_take]
                    if truncated:
                        return truncated + "...", self.token_counter.count_tokens(truncated + "...")
                    return "", 0
        
        if not truncated_parts:
            return "", 0
        
        result = " ".join(truncated_parts)
        return result, self.token_counter.count_tokens(result)
    
    def _build_metadata_header(self, chunks: List[RerankedChunk]) -> str:
        """Build a metadata header from chunk information."""
        if not chunks or not self.config.include_metadata_header:
            return ""
        
        # Get metadata from first chunk (all should be from same document)
        first = chunks[0].metadata
        
        course = first.get("course_name", "")
        chapter = first.get("chapter_title", "")
        
        if not course and not chapter:
            return ""
        
        return self.config.metadata_header_format.format(
            course=course or "Unknown Course",
            chapter=chapter or "Unknown Chapter"
        )
    
    # ========================================================================
    # Content Cleaning
    # ========================================================================
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace without collapsing intentional structure."""
        # Remove trailing whitespace per line
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)
        
        # Replace multiple spaces (but not newlines)
        text = re.sub(r' {2,}', ' ', text)
        
        # Normalize multiple newlines to max 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _remove_duplicate_lines(self, text: str) -> str:
        """Remove duplicate consecutive lines."""
        lines = text.split('\n')
        unique = []
        prev = None
        
        for line in lines:
            stripped = line.strip()
            if stripped != prev or not stripped:
                unique.append(line)
            prev = stripped
        
        return '\n'.join(unique)
    
    # ========================================================================
    # Public Utility Methods
    # ========================================================================
    
    def count_tokens(self, text: str) -> int:
        """Public method to count tokens using the configured tokenizer."""
        return self.token_counter.count_tokens(text)
    
    def estimate_context_size(self, chunks: List[RerankedChunk]) -> int:
        """
        Estimate the token size of the context that would be built.
        
        Useful for pre-validation before building.
        """
        if not chunks:
            return 0
        
        total = 0
        
        # Header
        if self.config.include_metadata_header:
            header = self._build_metadata_header(chunks)
            if header:
                total += self.token_counter.count_tokens(header)
        
        # Chunks
        for chunk in chunks:
            content = chunk.content
            if self.config.normalize_whitespace:
                content = self._normalize_whitespace(content)
            total += self.token_counter.count_tokens(content)
        
        # Separators
        num_separators = min(len(chunks) - 1, 0)
        total += num_separators * self.token_counter.count_tokens(self.config.separator)
        
        return total


# ============================================================================
# Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Context Builder - Validation (Enhanced)")
    print("=" * 60)
    
    config = ContextBuilderConfig.from_yaml()
    builder = ContextBuilder(config=config)
    
    print(f"\n📋 Configuration loaded:")
    print(f"   Templates: {list(config.templates.keys())}")
    print(f"   Classification: {config.classification_method}")
    print(f"   Citations: {config.citations_enabled}")
    print(f"   Max tokens: {config.max_total_tokens}")
    print(f"   Max chunks: {config.max_chunks}")
    print(f"   Sort by: {config.sort_by}")
    print(f"   Diversity enabled: {config.diversity_enabled}")
    print(f"   Diversity threshold: {config.diversity_threshold}")
    print(f"   Tokenizer model: {config.tokenizer_model}")
    print(f"   Truncate at sentence: {config.truncate_at_sentence}")
    
    # Create sample reranked chunks
    from src.retrieval.reranker import RerankedChunk
    
    sample_chunks = [
        RerankedChunk(
            chunk_id="doc1_0001",
            content="Deadlock is a situation where two or more processes are unable to proceed because each is waiting for a resource held by another.",
            raw_content="Deadlock is a situation where two or more processes are unable to proceed because each is waiting for a resource held by another.",
            original_score=0.85,
            rerank_score=0.95,
            rank=1,
            metadata={
                "course_name": "Operating Systems",
                "chapter_title": "Process Management",
                "topic": "Deadlock",
                "page_start": 12,
                "page_end": 12,
                "file_type": "pptx",
                "chunk_type": "definition"
            }
        ),
        RerankedChunk(
            chunk_id="doc1_0002",
            content="A deadlock occurs when four conditions hold simultaneously: mutual exclusion, hold and wait, no preemption, and circular wait.",
            raw_content="A deadlock occurs when four conditions hold simultaneously: mutual exclusion, hold and wait, no preemption, and circular wait.",
            original_score=0.78,
            rerank_score=0.88,
            rank=2,
            metadata={
                "course_name": "Operating Systems",
                "chapter_title": "Process Management",
                "topic": "Deadlock",
                "page_start": 13,
                "page_end": 13,
                "file_type": "pptx",
                "chunk_type": "explanation"
            }
        ),
        RerankedChunk(
            chunk_id="doc1_0003",
            content="For example, consider two processes P1 and P2. P1 holds Resource A and waits for Resource B. P2 holds Resource B and waits for Resource A. Neither can proceed.",
            raw_content="For example, consider two processes P1 and P2...",
            original_score=0.72,
            rerank_score=0.82,
            rank=3,
            metadata={
                "course_name": "Operating Systems",
                "chapter_title": "Process Management",
                "topic": "Deadlock",
                "page_start": 14,
                "page_end": 14,
                "file_type": "pptx",
                "chunk_type": "example"
            }
        ),
        RerankedChunk(
            chunk_id="doc1_0004",
            content="To prevent deadlock, use resource allocation graphs or the banker's algorithm. Alternatively, break one of the four necessary conditions.",
            raw_content="To prevent deadlock, use resource allocation graphs...",
            original_score=0.60,
            rerank_score=0.75,
            rank=4,
            metadata={
                "course_name": "Operating Systems",
                "chapter_title": "Process Management",
                "topic": "Deadlock",
                "page_start": 15,
                "page_end": 15,
                "file_type": "pptx",
                "chunk_type": "important_notes"
            }
        ),
        RerankedChunk(
            chunk_id="doc1_0005",
            content="Round Robin scheduling assigns each process a fixed time quantum in a circular order, ensuring fairness among all processes.",
            raw_content="Round Robin scheduling assigns each process...",
            original_score=0.65,
            rerank_score=0.70,
            rank=5,
            metadata={
                "course_name": "Operating Systems",
                "chapter_title": "Process Management",
                "topic": "CPU Scheduling",
                "page_start": 20,
                "page_end": 20,
                "file_type": "pptx",
                "chunk_type": "definition"
            }
        ),
        RerankedChunk(
            chunk_id="doc1_0006",
            content="Priority scheduling can lead to starvation of low-priority processes if not handled properly with aging.",
            raw_content="Priority scheduling can lead to starvation...",
            original_score=0.55,
            rerank_score=0.65,
            rank=6,
            metadata={
                "course_name": "Operating Systems",
                "chapter_title": "Process Management",
                "topic": "CPU Scheduling",
                "page_start": 21,
                "page_end": 21,
                "file_type": "pptx",
                "chunk_type": "important_notes"
            }
        ),
    ]
    
    # Test each template
    test_templates = [
        ContextTemplate.LEARNING,
        ContextTemplate.REVISION,
        ContextTemplate.QA,
        ContextTemplate.COMPARISON,
        ContextTemplate.RAW,
    ]
    
    for template in test_templates:
        print(f"\n{'─' * 50}")
        print(f"Template: {template.value}")
        
        result = builder.build(sample_chunks, template)
        
        print(f"   Chunks used: {result.chunks_used}")
        print(f"   Sections: {result.sections}")
        print(f"   Citations: {len(result.citations)}")
        print(f"   Estimated tokens: {result.tokens_used}")
        print(f"   Diversity applied: {result.diversity_applied}")
        print(f"   Truncated: {result.truncated}")
        
        if result.metadata_header:
            print(f"   Header: {result.metadata_header}")
        
        # Show first 200 chars of formatted context
        preview = result.formatted_context[:300].replace('\n', '\\n')
        print(f"   Preview: {preview}...")
    
    print(f"\n{'=' * 60}")
    print("✅ Enhanced Context Builder ready for integration")
    print("=" * 60)