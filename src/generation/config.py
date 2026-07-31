"""
Generation Layer — Shared Contracts
All dataclasses, enums, and configuration types for the generation pipeline.

No implementation logic — pure data contracts.
No UUID generation inside contracts.
No business logic in dataclasses.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.common.types import RetrievedChunk, CitationLocationType  # noqa: F401 — re-exported

load_dotenv()


# ============================================================================
# Enums
# ============================================================================

class GenerationMode(str, Enum):
    """Supported generation modes."""
    CONTEXT_AWARE = "context_aware"           # Precise, citation-backed
    SIMPLE_EXPLANATION = "simple_explanation"  # Analogies, simple language
    TOPIC_EXTRACTION = "topic_extraction"      # Strict-JSON topic/concept extraction
    QUIZ_GENERATION = "quiz_generation"        # Strict-JSON multiple-choice quiz generation


class AnswerFormat(str, Enum):
    """Output format for generated answers."""
    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"
    STRUCTURED = "structured"          # Sections with headers


class CitationStyle(str, Enum):
    """How citations appear in the answer."""
    INLINE = "inline"                  # [Course | Chapter | Slide 5]
    FOOTNOTE = "footnote"             # Superscript numbers with footer
    APPENDED = "appended"             # All citations at the end
    NONE = "none"


class ConfidenceLevel(str, Enum):
    """Confidence level for retrieval and grounding."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


# ============================================================================
# Core Data Contracts — Pure Data, No Logic
# ============================================================================

@dataclass
class ChatTurn:
    """A single turn in a conversation."""
    role: str                          # "user", "assistant", "system"
    content: str
    timestamp: Optional[str] = None    # ISO format timestamp


@dataclass
class RetrievalMetadata:
    """
    Metadata about the retrieval process.
    
    Owned by the retrieval layer, passed through generation.
    """
    confidence_score: float
    confidence_level: ConfidenceLevel
    retrieval_time_ms: float
    total_chunks_retrieved: int
    namespace: str
    top_k: int = 5
    threshold: Optional[float] = None
    retrieval_method: Optional[str] = None  # e.g., "semantic", "hybrid"
    
    # Additional provider-specific metadata
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    """
    Input to the generation pipeline.
    
    This is the contract between the retrieval layer and generation layer.
    IDs are provided by the orchestrator, not generated here.
    """
    request_id: str                      # Unique identifier (owned by orchestrator)
    query: str                           # Original user question
    mode: GenerationMode                 # Context-aware or simple explanation
    
    # Retrieved context (raw chunks, not assembled)
    chunks: List[RetrievedChunk]         # Raw chunks from retrieval
    retrieval_metadata: RetrievalMetadata  # Metadata about retrieval
    
    # Optional: conversation context for multi-turn
    conversation_history: Optional[List[ChatTurn]] = None

    # Optional free-form answer-shaping instructions (e.g.
    # {"length": "Target length: 250-350 words...", "style": "Structure the
    # answer as numbered steps."}). Each value is appended verbatim as a
    # line under an "## ANSWER REQUIREMENTS" section in the prompt — the
    # generation layer has no opinion on what keys mean or how a caller
    # derives them (marks-per-question, reading level, output length, etc.
    # are all just callers populating this dict differently).
    answer_hints: Optional[Dict[str, str]] = None


@dataclass
class UsageStats:
    """
    Token usage statistics from the LLM provider.
    
    Designed to be provider-agnostic while preserving provider-specific details.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Provider-specific metadata (e.g., cached_tokens, reasoning_tokens)
    provider_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """
    Information about the LLM model used.
    
    Provider-agnostic representation.
    """
    provider: str                      # "deepseek", "gemini", "openai", "anthropic", "local"
    model_name: str                    # "deepseek-chat", "gemini-2.5-flash", "gpt-4o", etc.
    api_version: Optional[str] = None  # "v1", "v1beta", etc.
    
    # Additional provider-specific details
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedAnswer:
    """
    Raw output from the LLM before post-processing.
    
    Contains the full model response plus usage metadata.
    """
    content: str                       # Raw LLM output
    model_info: ModelInfo              # Model used
    usage: UsageStats                  # Token usage
    generation_time_ms: float          # Time spent in LLM call
    finish_reason: str = "stop"        # "stop", "length", "error", "tool_calls"

    # Populated only when finish_reason == "error" — lets callers show a
    # specific message ("quota exceeded, retry in 20s") instead of a bare
    # empty answer. error_type is one of "rate_limited", "credential_error",
    # "timeout", "api_error", or None.
    error_type: Optional[str] = None
    retry_after_seconds: Optional[float] = None
    is_daily_quota: bool = False

    # Raw provider response (for debugging/tracing)
    raw_response: Any = None


@dataclass
class Citation:
    """
    A single source citation extracted from the generated answer.
    
    Pure data — formatting is handled by a formatter.
    """
    index: int                         # Citation number in the answer
    chunk_id: str                      # Source chunk identifier (maps to RetrievedChunk)
    
    # Source details (denormalized from RetrievedChunk for the final output)
    course_name: str
    chapter_title: str
    location_type: CitationLocationType
    location_start: int
    location_end: int
    
    # Where the citation appears in the answer
    position_start: int = 0            # Character position in answer
    position_end: int = 0


@dataclass
class GenerationResponse:
    """
    Complete output of the generation pipeline.
    
    This is the final contract — everything the caller receives.
    IDs are provided by the orchestrator.
    """
    # Core answer
    answer: str                        # Final answer text (post-processed)
    mode: GenerationMode
    answer_format: AnswerFormat
    
    # Citations and grounding
    citations: List[Citation]
    is_grounded: bool
    grounding_confidence: float
    
    # Retrieval metadata (owned by retrieval, passed through)
    retrieval_metadata: RetrievalMetadata
    
    # Generation metadata
    model_info: ModelInfo
    usage: UsageStats
    generation_time_ms: float
    total_time_ms: float
    
    # Traceability — IDs owned by orchestrator
    request_id: str
    generation_id: str
    
    # Version information for reproducibility
    prompt_version: Optional[str] = None
    config_version: Optional[str] = None
    
    # Diagnostics
    warnings: List[str] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)

    # Populated only on generation failure (see GeneratedAnswer) — lets
    # API consumers distinguish "the LLM provider is rate-limited, retry
    # in N seconds" from an ordinary ungrounded/empty answer.
    error_type: Optional[str] = None
    retry_after_seconds: Optional[float] = None
    is_daily_quota: bool = False


# ============================================================================
# Prompt Contract
# ============================================================================

@dataclass
class Prompt:
    """
    Structured prompt object.
    
    Built by PromptBuilder from YAML templates.
    Pure data — no concatenation logic.
    """
    system_prompt: str
    user_prompt: str
    template_id: str

    context_token_count: int
    system_token_count: int
    user_token_count: int
    total_token_count: int

    prompt_id: str
    request_id: str

    template_version: Optional[str] = None
    mode: Optional[GenerationMode] = None
    assembled_at: Optional[str] = None


# ============================================================================
# Configuration Types
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for an LLM model."""
    provider: str                      # "deepseek", "gemini", "openai", "anthropic"
    model_name: str                    # "deepseek-chat", "gemini-2.5-flash", "gpt-4o", etc.
    api_version: Optional[str] = None
    temperature: float = 0.3
    max_output_tokens: int = 2048
    top_p: float = 0.95
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    # Provider-specific configuration
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_model_info(self) -> ModelInfo:
        """Convert to ModelInfo (pure data transformation)."""
        return ModelInfo(
            provider=self.provider,
            model_name=self.model_name,
            api_version=self.api_version,
            extra=self.extra
        )


@dataclass
class ModeConfig:
    """Configuration for a generation mode."""
    mode: GenerationMode
    prompt_id: str                     # Identifier for prompt template (not path)
    model_key: str                     # Key in models.yaml
    answer_format: AnswerFormat
    citation_style: CitationStyle
    include_context_header: bool
    include_source_prefix: bool
    max_context_tokens: int
    system_instructions: str           # Brief description for logging
    
    # Optional prompt version constraint
    prompt_version_constraint: Optional[str] = None  # e.g., ">=1.0.0,<2.0.0"


@dataclass
class PostProcessingConfig:
    """Configuration for answer post-processing."""
    extract_citations: bool
    citation_style: CitationStyle
    validate_grounding: bool
    grounding_check_method: str        # "keyword_overlap", "llm_check", "both"
    remove_artifacts: bool             # Remove [HIDE], <THINK> tags
    normalize_whitespace: bool
    max_answer_length: Optional[int] = None  # Truncate if exceeded
    format_as_markdown: bool = True
    
    # Grounding validation thresholds
    min_keyword_overlap: float = 0.3
    min_grounding_confidence: float = 0.6


@dataclass
class GenerationConfig:
    """
    Complete generation pipeline configuration.
    
    Loaded from config/generation/ YAML files.
    """
    config_version: str = "1.0.0"
    default_mode: GenerationMode = GenerationMode.CONTEXT_AWARE
    default_model_key: str = "deepseek_chat"
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    modes: Dict[str, ModeConfig] = field(default_factory=dict)
    post_processing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    
    # Version information for reproducibility
    prompt_version: Optional[str] = None
    generation_version: str = "1.0.0"

    @classmethod
    def from_yaml(cls, config_dir: Optional[str] = None) -> "GenerationConfig":
        """
        Load all generation configuration from YAML files.
        
        Priority:
        1. Explicit config_dir argument
        2. RAGPIPE_GENERATION_CONFIG_DIR env var
        3. config/generation/ (default)
        """
        config_dir = (
            config_dir
            or os.getenv("RAGPIPE_GENERATION_CONFIG_DIR")
            or str(Path(__file__).parent.parent.parent / "config" / "generation")
        )
        config_dir = Path(config_dir)
        
        # Load generation.yaml
        gen_path = config_dir / "generation.yaml"
        gen_data = {}
        if gen_path.exists():
            with open(gen_path, "r") as f:
                gen_data = yaml.safe_load(f) or {}
        gen_data = cls._apply_env_overrides(gen_data)
        
        # Load models.yaml
        models_path = config_dir / "models.yaml"
        models_data = {}
        if models_path.exists():
            with open(models_path, "r") as f:
                models_data = yaml.safe_load(f) or {}
        
        # Parse models
        models = {}
        for key, model_data in models_data.get("models", {}).items():
            models[key] = ModelConfig(
                provider=str(model_data.get("provider", "deepseek")),
                model_name=str(model_data.get("model_name", "deepseek-chat")),
                api_version=model_data.get("api_version"),
                temperature=float(model_data.get("temperature", 0.3)),
                max_output_tokens=int(model_data.get("max_output_tokens", 2048)),
                top_p=float(model_data.get("top_p", 0.95)),
                timeout_seconds=int(model_data.get("timeout_seconds", 30)),
                max_retries=int(model_data.get("max_retries", 3)),
                retry_delay_seconds=float(model_data.get("retry_delay_seconds", 1.0)),
                extra=model_data.get("extra", {})
            )
        
        # Parse modes
        modes = {}
        for mode_key, mode_data in gen_data.get("modes", {}).items():
            modes[mode_key] = ModeConfig(
                mode=GenerationMode(mode_data.get("mode", "context_aware")),
                prompt_id=str(mode_data.get("prompt_id", "context_aware")),
                model_key=str(mode_data.get("model_key", "deepseek_chat")),
                answer_format=AnswerFormat(mode_data.get("answer_format", "markdown")),
                citation_style=CitationStyle(mode_data.get("citation_style", "inline")),
                include_context_header=bool(mode_data.get("include_context_header", True)),
                include_source_prefix=bool(mode_data.get("include_source_prefix", True)),
                max_context_tokens=int(mode_data.get("max_context_tokens", 3000)),
                system_instructions=str(mode_data.get("system_instructions", "")),
                prompt_version_constraint=mode_data.get("prompt_version_constraint")
            )
        
        # Load post_processing.yaml
        pp_path = config_dir / "post_processing.yaml"
        pp_data = {}
        if pp_path.exists():
            with open(pp_path, "r") as f:
                pp_data = yaml.safe_load(f) or {}
        
        post_processing = PostProcessingConfig(
            extract_citations=bool(pp_data.get("extract_citations", True)),
            citation_style=CitationStyle(pp_data.get("citation_style", "inline")),
            validate_grounding=bool(pp_data.get("validate_grounding", True)),
            grounding_check_method=str(pp_data.get("grounding_check_method", "keyword_overlap")),
            remove_artifacts=bool(pp_data.get("remove_artifacts", True)),
            normalize_whitespace=bool(pp_data.get("normalize_whitespace", True)),
            max_answer_length=pp_data.get("max_answer_length"),
            format_as_markdown=bool(pp_data.get("format_as_markdown", True)),
            min_keyword_overlap=float(pp_data.get("min_keyword_overlap", 0.3)),
            min_grounding_confidence=float(pp_data.get("min_grounding_confidence", 0.6)),
        )
        
        return cls(
            config_version=str(gen_data.get("config_version", "1.0.0")),
            default_mode=GenerationMode(gen_data.get("default_mode", "context_aware")),
            default_model_key=str(gen_data.get("default_model_key", "deepseek_chat")),
            models=models,
            modes=modes,
            post_processing=post_processing,
            prompt_version=gen_data.get("prompt_version"),
            generation_version=str(gen_data.get("generation_version", "1.0.0")),
        )
    
    @classmethod
    def _apply_env_overrides(cls, data: Dict) -> Dict:
        """Apply environment variable overrides."""
        env_mapping = {
            "RAGPIPE_GENERATION_DEFAULT_MODE": "default_mode",
            "RAGPIPE_GENERATION_DEFAULT_MODEL": "default_model_key",
        }
        for env_var, key in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                data[key] = value
        return data
    
    def get_model_config(self, model_key: Optional[str] = None) -> ModelConfig:
        """Get model configuration by key."""
        key = model_key or self.default_model_key
        if key not in self.models:
            raise ValueError(f"Unknown model key: {key}. Available: {list(self.models.keys())}")
        return self.models[key]
    
    def get_mode_config(self, mode: GenerationMode) -> ModeConfig:
        """Get mode configuration."""
        key = mode.value
        if key not in self.modes:
            raise ValueError(f"Unknown mode: {key}. Available: {list(self.modes.keys())}")
        return self.modes[key]


# ============================================================================
# Formatter (Separate from Data Contracts)
# ============================================================================

class CitationFormatter:
    """
    Formats citations from pure data objects.
    
    This is a utility class, not part of the core contracts.
    Separates formatting logic from data representation.
    """
    
    @staticmethod
    def format_chunk(chunk: RetrievedChunk, style: CitationStyle = CitationStyle.INLINE) -> str:
        """Format a RetrievedChunk as a citation string."""
        location_str = CitationFormatter._format_location(
            chunk.location_type,
            chunk.location_start,
            chunk.location_end
        )
        
        if style == CitationStyle.INLINE:
            return f"[{chunk.course_name} | {chunk.chapter_title} | {location_str}]"
        elif style == CitationStyle.NONE:
            return ""
        else:
            return f"{chunk.course_name}, {chunk.chapter_title}, {location_str}"
    
    @staticmethod
    def format_citation(citation: Citation, style: CitationStyle = CitationStyle.INLINE) -> str:
        """Format a Citation object as a citation string."""
        location_str = CitationFormatter._format_location(
            citation.location_type,
            citation.location_start,
            citation.location_end
        )
        
        if style == CitationStyle.INLINE:
            return f"[{citation.course_name} | {citation.chapter_title} | {location_str}]"
        elif style == CitationStyle.NONE:
            return ""
        else:
            return f"{citation.course_name}, {citation.chapter_title}, {location_str}"
    
    @staticmethod
    def _format_location(
        location_type: CitationLocationType,
        start: int,
        end: int
    ) -> str:
        """Format location information."""
        if location_type == CitationLocationType.PAGE:
            if end and end != start:
                return f"Pages {start}-{end}"
            return f"Page {start}"
        elif location_type == CitationLocationType.SLIDE:
            if end and end != start:
                return f"Slides {start}-{end}"
            return f"Slide {start}"
        elif location_type == CitationLocationType.SECTION:
            return f"Section {start}"
        elif location_type == CitationLocationType.CHAPTER:
            return f"Chapter {start}"
        else:
            return "Source"