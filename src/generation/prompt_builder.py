"""
Prompt Builder
Loads YAML prompt templates, formats retrieved chunks into context,
injects variables, and returns a structured Prompt object.

Architecture:
    GenerationRequest → PromptBuilder.build() → Prompt

Features:
- Template resolution by prompt_id (not file path)
- Chunk formatting via CitationFormatter
- Token counting for context budget management
- Template version tracking
- Configurable context assembly with pluggable strategies
- Variable validation with warnings for debugging
"""

import os
import re
import hashlib
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Protocol, runtime_checkable
from pathlib import Path
from datetime import datetime, timezone
from abc import ABC, abstractmethod

import yaml
from dotenv import load_dotenv

from src.generation.config import (
    GenerationRequest,
    GenerationMode,
    ModeConfig,
    Prompt,
    RetrievedChunk,
    CitationFormatter,
    CitationStyle,
)

load_dotenv()

# Setup logging for warnings
logger = logging.getLogger(__name__)


# ============================================================================
# Section Classification Strategy
# ============================================================================

@runtime_checkable
class SectionClassifier(Protocol):
    """Protocol for section classification strategies."""
    
    def classify(self, chunk: RetrievedChunk) -> str:
        """
        Classify a chunk into a section type.
        
        Args:
            chunk: RetrievedChunk to classify
            
        Returns:
            Section identifier string (e.g., "definition", "example")
        """
        ...


class MetadataBasedClassifier:
    """
    Classifies chunks based on metadata fields.
    
    Uses chunk_type, topic, and other metadata fields to determine
    the section type. Fully configurable via classification_rules.
    """
    
    def __init__(self, rules: Optional[Dict[str, List[str]]] = None):
        """
        Initialize with classification rules.
        
        Args:
            rules: Dictionary mapping section names to lists of keywords
                   that trigger classification. Defaults to built-in rules.
        """
        self.rules = rules or {
            "definition": ["definition", "defines", "is defined as"],
            "example": ["example", "e.g.", "for example", "illustration"],
            "formula": ["formula", "equation", "calculation", "="],
            "key_points": ["key point", "important", "key takeaway"],
            "explanation": ["explanation", "explains", "overview"],
        }
    
    def classify(self, chunk: RetrievedChunk) -> str:
        """
        Classify chunk based on metadata and content heuristics.
        
        Priority:
        1. Explicit chunk_type from metadata
        2. Topic-based classification
        3. Content-based keyword matching
        4. Default to "general"
        """
        metadata = chunk.metadata
        
        # Check explicit chunk_type
        chunk_type = metadata.get("chunk_type", "").lower()
        if chunk_type:
            for section, keywords in self.rules.items():
                if any(keyword in chunk_type for keyword in keywords):
                    return section
        
        # Check topic
        topic = metadata.get("topic", "").lower()
        if topic:
            for section, keywords in self.rules.items():
                if any(keyword in topic for keyword in keywords):
                    return section
        
        # Content-based heuristic
        content_lower = chunk.content.lower()[:200]
        for section, keywords in self.rules.items():
            if any(keyword in content_lower for keyword in keywords):
                return section
        
        return "general"
    
    def add_rule(self, section: str, keywords: List[str]) -> None:
        """Add or update classification rules for a section."""
        self.rules[section] = keywords


class SemanticGroupingClassifier:
    """
    Groups chunks semantically using embeddings or similarity.
    
    Placeholder for future implementation with vector embeddings.
    Currently falls back to metadata-based classification.
    """
    
    def __init__(self, embedding_model: Optional[Any] = None):
        """
        Initialize with optional embedding model.
        
        Args:
            embedding_model: Model for computing embeddings
        """
        self.embedding_model = embedding_model
        self._fallback = MetadataBasedClassifier()
    
    def classify(self, chunk: RetrievedChunk) -> str:
        """Classify using semantic similarity (placeholder)."""
        # TODO: Implement semantic grouping with embeddings
        # For now, fall back to metadata-based classification
        return self._fallback.classify(chunk)


class ChronologicalOrderingClassifier:
    """
    Classifies based on position/order in the document.
    
    Useful for maintaining narrative flow when chunks are from
    sequential sections.
    """
    
    def __init__(self):
        """Initialize chronological classifier."""
        self._counter = 0
    
    def classify(self, chunk: RetrievedChunk) -> str:
        """
        Classify based on position information.
        
        Returns sequential section names to maintain order.
        """
        # Use position metadata if available
        position = chunk.metadata.get("position", self._counter)
        self._counter += 1
        
        # Sequential sections with position-based naming
        sections = ["beginning", "early", "middle", "late", "conclusion"]
        idx = min(position // 5, len(sections) - 1)
        return sections[idx]


class ExamAnswerOrderingClassifier:
    """
    Orders chunks in an exam-answer format.
    
    Structures response as: Thesis → Evidence → Examples → Conclusion.
    Useful for exam preparation and structured answers.
    """
    
    def __init__(self):
        """Initialize exam answer classifier."""
        self._section_order = [
            "thesis",
            "evidence",
            "examples",
            "conclusion"
        ]
    
    def classify(self, chunk: RetrievedChunk) -> str:
        """
        Classify for exam answer structure.
        
        Uses content analysis to determine where each chunk fits
        in the exam answer structure.
        """
        content = chunk.content.lower()
        
        # Thesis: first chunk or content with "this means", "in essence"
        if any(phrase in content for phrase in ["this means", "in essence", "therefore", "thus"]):
            return "thesis"
        
        # Evidence: factual statements, data, citations
        if any(phrase in content for phrase in ["according to", "data shows", "research indicates"]):
            return "evidence"
        
        # Examples: explicit examples
        if any(phrase in content for phrase in ["example", "for instance", "case study"]):
            return "examples"
        
        # Conclusion: summary statements
        if any(phrase in content for phrase in ["in conclusion", "to summarize", "overall"]):
            return "conclusion"
        
        # Default to middle sections
        return "evidence"


# ============================================================================
# Context Assembly Strategy
# ============================================================================

class ContextAssemblyStrategy(ABC):
    """
    Abstract strategy for context assembly.
    
    Defines how chunks are processed, grouped, and formatted.
    """
    
    @abstractmethod
    def assemble(
        self,
        chunks: List[RetrievedChunk],
        citation_style: CitationStyle,
        max_tokens: int,
        token_counter: 'TokenCounter'
    ) -> Tuple[str, int, List[str]]:
        """
        Assemble chunks into a formatted context string.
        
        Args:
            chunks: Retrieved chunks to assemble
            citation_style: How to format citations
            max_tokens: Maximum tokens allowed
            token_counter: TokenCounter for budget management
            
        Returns:
            Tuple of (formatted_context, token_count, section_labels)
        """
        ...


class StandardContextAssembly(ContextAssemblyStrategy):
    """
    Standard context assembly with section grouping.
    
    Groups chunks by section (definition, example, etc.) and
    applies consistent formatting.
    """
    
    def __init__(
        self,
        classifier: Optional[SectionClassifier] = None,
        section_headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize standard assembly strategy.
        
        Args:
            classifier: SectionClassifier instance (defaults to MetadataBasedClassifier)
            section_headers: Custom section header mapping
        """
        self.classifier = classifier or MetadataBasedClassifier()
        self.citation_formatter = CitationFormatter()
        
        self.section_headers = section_headers or {
            "definition": "## DEFINITION",
            "explanation": "## EXPLANATION",
            "example": "## EXAMPLE",
            "formula": "## FORMULA",
            "key_points": "## KEY POINTS",
            "general": "## ADDITIONAL INFORMATION",
            "thesis": "## THESIS",
            "evidence": "## EVIDENCE",
            "conclusion": "## CONCLUSION",
            "beginning": "## INTRODUCTION",
            "early": "## EARLY CONTEXT",
            "middle": "## MAIN CONTENT",
            "late": "## LATER DEVELOPMENT",
            "conclusion": "## CONCLUSION",
        }
    
    def assemble(
        self,
        chunks: List[RetrievedChunk],
        citation_style: CitationStyle,
        max_tokens: int,
        token_counter: 'TokenCounter'
    ) -> Tuple[str, int, List[str]]:
        """Assemble chunks with section-based grouping."""
        if not chunks:
            return "", 0, []
        
        # Group chunks by section
        sections: Dict[str, List[str]] = {}
        total_tokens = 0
        
        for chunk in chunks:
            # Classify section
            section = self.classifier.classify(chunk)
            
            # Format the chunk with citation
            citation = self.citation_formatter.format_chunk(chunk, citation_style)
            formatted = f"{citation}\n{chunk.content}" if citation else chunk.content
            
            # Check token budget
            chunk_tokens = token_counter.count(formatted)
            if total_tokens + chunk_tokens > max_tokens:
                break
            
            # Add to section
            if section not in sections:
                sections[section] = []
            sections[section].append(formatted)
            total_tokens += chunk_tokens
        
        # Build context with section headers
        context_parts = []
        section_labels_used = []
        
        for section_name, contents in sections.items():
            header = self.section_headers.get(section_name, f"## {section_name.upper()}")
            context_parts.append(header)
            context_parts.extend(contents)
            section_labels_used.append(section_name)
        
        formatted = "\n\n".join(context_parts)
        return formatted, total_tokens, section_labels_used
    
    def set_classifier(self, classifier: SectionClassifier) -> None:
        """Change the classifier strategy."""
        self.classifier = classifier
    
    def add_section_header(self, section: str, header: str) -> None:
        """Add or update a section header."""
        self.section_headers[section] = header


class MinimalContextAssembly(ContextAssemblyStrategy):
    """
    Minimal context assembly without section grouping.
    
    Each chunk is formatted with its citation and concatenated
    with minimal formatting.
    """
    
    def __init__(self):
        """Initialize minimal assembly strategy."""
        self.citation_formatter = CitationFormatter()
    
    def assemble(
        self,
        chunks: List[RetrievedChunk],
        citation_style: CitationStyle,
        max_tokens: int,
        token_counter: 'TokenCounter'
    ) -> Tuple[str, int, List[str]]:
        """Assemble chunks with minimal formatting."""
        if not chunks:
            return "", 0, []
        
        formatted_chunks = []
        total_tokens = 0
        
        for chunk in chunks:
            citation = self.citation_formatter.format_chunk(chunk, citation_style)
            formatted = f"{citation}\n{chunk.content}" if citation else chunk.content
            
            chunk_tokens = token_counter.count(formatted)
            if total_tokens + chunk_tokens > max_tokens:
                break
            
            formatted_chunks.append(formatted)
            total_tokens += chunk_tokens
        
        context = "\n\n---\n\n".join(formatted_chunks)
        return context, total_tokens, ["all"]


class ChronologicalContextAssembly(ContextAssemblyStrategy):
    """
    Chronological context assembly preserving document order.
    
    Uses position metadata to maintain the original sequence.
    """
    
    def __init__(self, classifier: Optional[SectionClassifier] = None):
        """
        Initialize chronological assembly.
        
        Args:
            classifier: SectionClassifier for grouping (defaults to ChronologicalOrderingClassifier)
        """
        self.classifier = classifier or ChronologicalOrderingClassifier()
        self.citation_formatter = CitationFormatter()
    
    def assemble(
        self,
        chunks: List[RetrievedChunk],
        citation_style: CitationStyle,
        max_tokens: int,
        token_counter: 'TokenCounter'
    ) -> Tuple[str, int, List[str]]:
        """Assemble chunks in chronological order."""
        if not chunks:
            return "", 0, []
        
        # Sort by position if available
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.metadata.get("position", 0)
        )
        
        # Delegate to standard assembly with chronological classifier
        standard = StandardContextAssembly(self.classifier)
        return standard.assemble(sorted_chunks, citation_style, max_tokens, token_counter)


class ExamAnswerContextAssembly(ContextAssemblyStrategy):
    """
    Exam-answer context assembly for structured responses.
    
    Organizes chunks in: Thesis → Evidence → Examples → Conclusion format.
    Ideal for exam preparation and answer generation.
    """
    
    def __init__(self, classifier: Optional[SectionClassifier] = None):
        """
        Initialize exam answer assembly.
        
        Args:
            classifier: SectionClassifier for exam structure (defaults to ExamAnswerOrderingClassifier)
        """
        self.classifier = classifier or ExamAnswerOrderingClassifier()
        self.citation_formatter = CitationFormatter()
    
    def assemble(
        self,
        chunks: List[RetrievedChunk],
        citation_style: CitationStyle,
        max_tokens: int,
        token_counter: 'TokenCounter'
    ) -> Tuple[str, int, List[str]]:
        """Assemble chunks in exam answer structure."""
        if not chunks:
            return "", 0, []
        
        # Delegate to standard assembly with exam classifier
        standard = StandardContextAssembly(self.classifier)
        return standard.assemble(chunks, citation_style, max_tokens, token_counter)


# ============================================================================
# Token Counter
# ============================================================================

class TokenCounter:
    """
    Counts tokens in text for context budget management.
    
    Uses tiktoken if available, falls back to character-based estimation.
    """
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize token counter.
        
        Args:
            model_name: Model name for tiktoken encoding lookup
        """
        self.model_name = model_name
        self._encoder = None
        
        try:
            import tiktoken
            try:
                self._encoder = tiktoken.encoding_for_model(model_name)
            except Exception:
                self._encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            self._encoder = None
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass
        
        # Fallback: ~4 chars per token
        return len(text) // 4
    
    def count_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for multiple texts."""
        return [self.count(t) for t in texts]


# ============================================================================
# Template Loader
# ============================================================================

class TemplateLoader:
    """
    Loads and caches prompt templates from YAML files.
    
    Templates are referenced by prompt_id, not file path.
    Resolution: prompt_id → config/generation/prompts/{prompt_id}.yaml
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize template loader.
        
        Args:
            templates_dir: Path to prompt templates directory.
                           Defaults from env var or config/generation/prompts/
        """
        self.templates_dir = Path(
            templates_dir
            or os.getenv("RAGPIPE_PROMPT_TEMPLATES_DIR")
            or str(Path(__file__).parent.parent.parent / "config" / "generation" / "prompts")
        )
        
        if not self.templates_dir.exists():
            raise FileNotFoundError(
                f"Prompt templates directory not found: {self.templates_dir}"
            )
        
        # Template cache: prompt_id → (version, system_prompt, user_prompt)
        self._cache: Dict[str, Tuple[str, str, str]] = {}
    
    def load(self, prompt_id: str) -> Tuple[str, str, str]:
        """
        Load a prompt template by its identifier.
        
        Args:
            prompt_id: Template identifier (e.g., "context_aware")
            
        Returns:
            Tuple of (version, system_prompt, user_prompt)
            
        Raises:
            FileNotFoundError: If template file doesn't exist
            ValueError: If template is missing required fields
        """
        # Check cache
        if prompt_id in self._cache:
            return self._cache[prompt_id]
        
        # Resolve file path
        template_path = self.templates_dir / f"{prompt_id}.yaml"
        
        if not template_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {template_path}\n"
                f"Available templates: {self._list_available()}"
            )
        
        # Load and parse
        with open(template_path, "r") as f:
            data = yaml.safe_load(f) or {}
        
        version = str(data.get("version", "1.0.0"))
        system_prompt = str(data.get("system", ""))
        user_prompt = str(data.get("user", ""))
        
        if not system_prompt and not user_prompt:
            raise ValueError(
                f"Template '{prompt_id}' missing both 'system' and 'user' fields"
            )
        
        # Cache
        self._cache[prompt_id] = (version, system_prompt, user_prompt)
        
        return version, system_prompt, user_prompt
    
    def get_version(self, prompt_id: str) -> str:
        """Get the version of a template without loading the full content."""
        version, _, _ = self.load(prompt_id)
        return version
    
    def _list_available(self) -> List[str]:
        """List available template IDs."""
        if not self.templates_dir.exists():
            return []
        return [
            p.stem for p in self.templates_dir.glob("*.yaml")
        ]
    
    def clear_cache(self) -> None:
        """Clear the template cache (useful for hot-reloading)."""
        self._cache.clear()


# ============================================================================
# Context Assembler
# ============================================================================

class ContextAssembler:
    """
    Assembles retrieved chunks into formatted context strings.
    
    Strategy-driven architecture allows pluggable assembly strategies
    for different use cases (standard, minimal, chronological, exam-answer).
    """
    
    def __init__(
        self,
        token_counter: Optional[TokenCounter] = None,
        strategy: Optional[ContextAssemblyStrategy] = None
    ):
        """
        Initialize context assembler.
        
        Args:
            token_counter: TokenCounter instance (creates default if None)
            strategy: ContextAssemblyStrategy (defaults to StandardContextAssembly)
        """
        self.token_counter = token_counter or TokenCounter()
        self.strategy = strategy or StandardContextAssembly()
        self.citation_formatter = CitationFormatter()
    
    def assemble(
        self,
        chunks: List[RetrievedChunk],
        mode_config: ModeConfig,
        max_tokens: int
    ) -> Tuple[str, int, List[str]]:
        """
        Assemble chunks into a formatted context string.
        
        Args:
            chunks: Retrieved chunks from the retrieval layer
            mode_config: Mode configuration (citation style, etc.)
            max_tokens: Maximum tokens for the context
            
        Returns:
            Tuple of (formatted_context, token_count, section_labels)
        """
        return self.strategy.assemble(
            chunks=chunks,
            citation_style=mode_config.citation_style,
            max_tokens=max_tokens,
            token_counter=self.token_counter
        )
    
    def set_strategy(self, strategy: ContextAssemblyStrategy) -> None:
        """
        Change the assembly strategy.
        
        Allows runtime strategy switching for different use cases.
        """
        self.strategy = strategy


# ============================================================================
# Prompt Builder
# ============================================================================

class PromptBuilder:
    """
    Builds structured Prompt objects from YAML templates and retrieved chunks.
    
    Single responsibility: assemble a GenerationRequest into a Prompt.
    
    Usage:
        builder = PromptBuilder()
        prompt = builder.build(request, mode_config)
    """
    
    def __init__(
        self,
        templates_dir: Optional[str] = None,
        token_counter: Optional[TokenCounter] = None,
        assembler: Optional[ContextAssembler] = None,
        log_missing_variables: bool = True
    ):
        """
        Initialize the prompt builder.
        
        Args:
            templates_dir: Path to prompt templates directory
            token_counter: TokenCounter instance
            assembler: ContextAssembler instance (creates default if None)
            log_missing_variables: Whether to log warnings for missing template variables
        """
        self.template_loader = TemplateLoader(templates_dir)
        self.token_counter = token_counter or TokenCounter()
        self.context_assembler = assembler or ContextAssembler(self.token_counter)
        self.log_missing_variables = log_missing_variables
    
    def build(
        self,
        request: GenerationRequest,
        mode_config: ModeConfig
    ) -> Prompt:
        """
        Build a Prompt from a GenerationRequest.
        
        Args:
            request: GenerationRequest with query and chunks
            mode_config: Mode configuration with template reference
            
        Returns:
            Prompt object ready for LLMClient
        """
        # Load template
        version, system_template, user_template = self.template_loader.load(
            mode_config.prompt_id
        )
        
        # Assemble context from chunks
        context, context_tokens, sections = self.context_assembler.assemble(
            chunks=request.chunks,
            mode_config=mode_config,
            max_tokens=mode_config.max_context_tokens
        )
        
        # Build context header
        context_header = ""
        if mode_config.include_context_header and request.chunks:
            first_chunk = request.chunks[0]
            context_header = f"Context from: {first_chunk.course_name} - {first_chunk.chapter_title}"
        
        # Prepare variables for template substitution
        variables = {
            "context": context,
            "context_header": context_header,
            "query": request.query,
            "course": request.chunks[0].course_name if request.chunks else "",
            "chapter": request.chunks[0].chapter_title if request.chunks else "",
            "answer_hints": self._build_answer_hints_instruction(request.answer_hints),
        }
        
        # Substitute variables in templates with validation
        system_prompt = self._substitute_with_validation(system_template, variables)
        user_prompt = self._substitute_with_validation(user_template, variables)
        
        # Count tokens
        system_tokens = self.token_counter.count(system_prompt)
        user_tokens = self.token_counter.count(user_prompt)
        total_tokens = system_tokens + user_tokens
        
        # Build full prompt
        full_prompt = self._assemble_full_prompt(system_prompt, user_prompt)
        
        # Generate prompt ID
        prompt_id = self._generate_prompt_id(request.request_id, mode_config.prompt_id)
        
        return Prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            template_id=mode_config.prompt_id,
            template_version=version,
            context_token_count=context_tokens,
            system_token_count=system_tokens,
            user_token_count=user_tokens,
            total_token_count=total_tokens,
            prompt_id=prompt_id,
            request_id=request.request_id,
            mode=request.mode,
            assembled_at=datetime.now(timezone.utc).isoformat()
        )
    
    def _build_answer_hints_instruction(self, answer_hints: Optional[Dict[str, str]]) -> str:
        """
        Turn caller-supplied answer_hints into an "## ANSWER REQUIREMENTS"
        block appended to the prompt, so hints are never silently ignored
        (see GenerationRequest.answer_hints).

        The generation layer has no domain opinion here — it just renders
        whatever instruction strings the caller provides (e.g. a length
        target derived from marks-per-question, a reading-level target, a
        step-by-step vs. narrative style preference). Deriving those
        strings from domain data is entirely the caller's responsibility.

        Returns "" when no hints are set — existing behavior for every
        caller that doesn't populate it is unchanged.
        """
        if not answer_hints:
            return ""

        lines = ["## ANSWER REQUIREMENTS"] + [v for v in answer_hints.values() if v]

        if len(lines) == 1:
            return ""  # answer_hints given but every value was empty

        return "\n" + "\n".join(lines) + "\n"

    def _substitute_with_validation(self, template: str, variables: Dict[str, str]) -> str:
        """
        Substitute variables in a template string with validation.
        
        Uses {variable} syntax. Missing variables are left as-is but
        logged as warnings for debugging.
        
        Args:
            template: Template string with {variable} placeholders
            variables: Dictionary of variable values
            
        Returns:
            Substituted template string
        """
        result = template
        
        # Find all placeholders in the template
        placeholders = re.findall(r'\{([^}]+)\}', template)
        
        for placeholder in placeholders:
            if placeholder in variables:
                # Substitute the variable
                result = result.replace(f"{{{placeholder}}}", variables[placeholder])
            else:
                # Log missing variable if configured
                if self.log_missing_variables:
                    logger.warning(
                        f"Missing template variable: '{{{placeholder}}}' in template. "
                        f"Available variables: {list(variables.keys())}"
                    )
                # Leave placeholder as-is (debug-friendly)
        
        return result
    
    def _assemble_full_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Assemble system and user prompts into a single string."""
        parts = []
        if system_prompt:
            parts.append(f"[SYSTEM]\n{system_prompt}")
        if user_prompt:
            parts.append(f"[USER]\n{user_prompt}")
        return "\n\n".join(parts)
    
    def _generate_prompt_id(self, request_id: str, template_id: str) -> str:
        """Generate a deterministic prompt ID."""
        raw = f"{request_id}:{template_id}:{time.time()}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]
    
    def set_assembly_strategy(self, strategy: ContextAssemblyStrategy) -> None:
        """Change the context assembly strategy."""
        self.context_assembler.set_strategy(strategy)


# ============================================================================
# Convenience Factory Functions
# ============================================================================

def create_standard_assembler(
    classifier: Optional[SectionClassifier] = None,
    section_headers: Optional[Dict[str, str]] = None
) -> ContextAssembler:
    """Create a ContextAssembler with standard assembly strategy."""
    strategy = StandardContextAssembly(classifier, section_headers)
    return ContextAssembler(strategy=strategy)


def create_minimal_assembler() -> ContextAssembler:
    """Create a ContextAssembler with minimal assembly strategy."""
    strategy = MinimalContextAssembly()
    return ContextAssembler(strategy=strategy)


def create_chronological_assembler(
    classifier: Optional[SectionClassifier] = None
) -> ContextAssembler:
    """Create a ContextAssembler with chronological assembly strategy."""
    strategy = ChronologicalContextAssembly(classifier)
    return ContextAssembler(strategy=strategy)


def create_exam_assembler(
    classifier: Optional[SectionClassifier] = None
) -> ContextAssembler:
    """Create a ContextAssembler with exam-answer assembly strategy."""
    strategy = ExamAnswerContextAssembly(classifier)
    return ContextAssembler(strategy=strategy)


# ============================================================================
# Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Prompt Builder - Validation")
    print("=" * 60)
    
    # Configure logging to show warnings
    logging.basicConfig(level=logging.WARNING)
    
    builder = PromptBuilder()
    
    # List available templates
    available = builder.template_loader._list_available()
    print(f"\n📋 Available templates: {available}")
    
    for template_id in available:
        version, system, user = builder.template_loader.load(template_id)
        print(f"\n{'─' * 50}")
        print(f"Template: {template_id} (v{version})")
        print(f"System prompt length: {len(system)} chars")
        print(f"User prompt length: {len(user)} chars")
        print(f"System preview: {system[:100]}...")
    
    # Build a sample prompt
    from src.generation.config import (
        GenerationRequest,
        GenerationMode,
        RetrievedChunk,
        RetrievalMetadata,
        CitationLocationType,
        ConfidenceLevel,
    )
    
    sample_chunks = [
        RetrievedChunk(
            chunk_id="doc1_0001",
            content="Inventory refers to the stock of goods, materials, and supplies that a business holds for the purpose of resale, production, or use in providing services.",
            score=0.92,
            course_name="Supply Chain Management Strategy",
            chapter_title="Inventory Management",
            location_type=CitationLocationType.SLIDE,
            location_start=5,
            location_end=5,
            metadata={"chunk_type": "definition", "topic": "Inventory", "position": 0}
        ),
        RetrievedChunk(
            chunk_id="doc1_0002",
            content="The main types of inventory include: raw materials, work-in-progress (WIP), finished goods, and MRO (maintenance, repair, and operations) supplies.",
            score=0.88,
            course_name="Supply Chain Management Strategy",
            chapter_title="Inventory Management",
            location_type=CitationLocationType.SLIDE,
            location_start=6,
            location_end=6,
            metadata={"chunk_type": "explanation", "topic": "Inventory Types", "position": 1}
        ),
        RetrievedChunk(
            chunk_id="doc1_0003",
            content="For example, a car manufacturer holds raw materials (steel, rubber), WIP (partially assembled vehicles), and finished goods (completed cars ready for sale).",
            score=0.82,
            course_name="Supply Chain Management Strategy",
            chapter_title="Inventory Management",
            location_type=CitationLocationType.SLIDE,
            location_start=7,
            location_end=7,
            metadata={"chunk_type": "example", "topic": "Inventory Example", "position": 2}
        ),
    ]
    
    from src.generation.config import ModeConfig, AnswerFormat, CitationStyle
    
    mode_config = ModeConfig(
        mode=GenerationMode.CONTEXT_AWARE,
        prompt_id="context_aware",
        model_key="deepseek_chat",
        answer_format=AnswerFormat.MARKDOWN,
        citation_style=CitationStyle.INLINE,
        include_context_header=True,
        include_source_prefix=True,
        max_context_tokens=3000,
        system_instructions="Answer using ONLY the provided context."
    )
    
    request = GenerationRequest(
        request_id="test-001",
        query="What is inventory and what are its types?",
        mode=GenerationMode.CONTEXT_AWARE,
        chunks=sample_chunks,
        retrieval_metadata=RetrievalMetadata(
            confidence_score=0.87,
            confidence_level=ConfidenceLevel.HIGH,
            retrieval_time_ms=150.0,
            total_chunks_retrieved=3,
            namespace="supply_chain_test"
        )
    )
    
    # Test different assembly strategies
    strategies = {
        "Standard": StandardContextAssembly(),
        "Chronological": ChronologicalContextAssembly(),
        "Exam Answer": ExamAnswerContextAssembly(),
        "Minimal": MinimalContextAssembly()
    }
    
    for strategy_name, strategy in strategies.items():
        print(f"\n{'=' * 60}")
        print(f"Testing {strategy_name} Assembly Strategy")
        print(f"{'=' * 60}")
        
        # Create assembler with this strategy
        assembler = ContextAssembler(strategy=strategy)
        builder.set_assembly_strategy(strategy)
        
        # Build prompt
        prompt = builder.build(request, mode_config)
        
        print(f"Prompt ID: {prompt.prompt_id}")
        print(f"Template: {prompt.template_id} v{prompt.template_version}")
        print(f"Tokens - Context: {prompt.context_token_count}, "
              f"System: {prompt.system_token_count}, "
              f"User: {prompt.user_token_count}, "
              f"Total: {prompt.total_token_count}")
        print(f"\n{'─' * 50}")
        print("USER PROMPT:")
        print(f"{'─' * 50}")
        print(prompt.user_prompt[:500] + "..." if len(prompt.user_prompt) > 500 else prompt.user_prompt)
    
    # Test variable validation with missing variable
    print(f"\n{'=' * 60}")
    print("Testing Variable Validation")
    print(f"{'=' * 60}")
    
    # Create a template with a missing variable
    test_template = "Answer based on: {context}\nCourse: {course}\nMissing: {missing_var}"
    variables = {
        "context": "Sample context",
        "course": "Sample course"
    }
    
    print("\nTemplate with missing variable '{missing_var}':")
    print(test_template)
    print("\nAvailable variables:", list(variables.keys()))
    print("\nSubstitution result:")
    result = builder._substitute_with_validation(test_template, variables)
    print(result)
    print("\n✅ Missing variable left as-is for debugging, with warning logged")
    
    print(f"\n✅ Prompt Builder ready for integration")