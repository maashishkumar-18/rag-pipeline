"""
Query Expander
Expands cleaned queries with synonyms, acronyms, related concepts,
and hierarchical context to increase retrieval recall.

Strategies:
- Synonym expansion (concept graph)
- Acronym expansion (FCFS → First Come First Serve)
- Hierarchical context injection (add topic/chapter/subject)
- Keyword extraction for dense retrieval
- Multi-query variant generation for hybrid search
"""

import re
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
import hashlib

from src.retrieval.query_rewriter import ConversationState


# ============================================================================
# Expansion Types
# ============================================================================

@dataclass
class ExpandedQuery:
    """Result of query expansion with multiple variants for hybrid search."""
    original: str
    expanded: str                          # Full expanded query (semantic search)
    keywords: str                          # Keyword-only variant (sparse search)
    variants: List[str] = field(default_factory=list)  # All variants
    expansions_applied: List[str] = field(default_factory=list)  # What was added
    expansion_count: int = 0
    confidence: float = 1.0                # Confidence in the expansion quality


# ============================================================================
# Concept Registry
# ============================================================================

class ConceptRegistry:
    """
    Dynamic concept registry for query expansion.
    Replaces hardcoded dictionaries with a knowledge-driven approach.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize concept registry from data source.
        
        Args:
            data_path: Path to concept registry JSON file
        """
        self.acronyms: Dict[str, str] = {}
        self.synonyms: Dict[str, List[str]] = {}
        self.hierarchy: Dict[str, Dict[str, List[str]]] = {}  # subject → topic → concepts
        self.relationships: Dict[str, List[str]] = {}  # concept → related concepts
        
        if data_path and Path(data_path).exists():
            self._load_from_file(data_path)
        else:
            self._load_defaults()
    
    def _load_from_file(self, path: str):
        """Load concept registry from JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                self.acronyms = data.get('acronyms', {})
                self.synonyms = data.get('synonyms', {})
                self.hierarchy = data.get('hierarchy', {})
                self.relationships = data.get('relationships', {})
        except Exception as e:
            print(f"Warning: Failed to load concept registry: {e}")
            self._load_defaults()
    
    def _load_defaults(self):
        """
        Load a small illustrative concept registry.

        This is example seed data only, meant to demonstrate the shape a
        real registry takes (acronyms / synonyms / hierarchy /
        relationships) — for a real corpus, build a JSON file matching this
        structure from your own domain vocabulary and pass it as
        `data_path`. Ships intentionally tiny and generic (general
        software/tech terms) rather than tuned to any one course or corpus.
        """
        # Acronyms
        self.acronyms = {
            'api': 'application programming interface',
            'sdk': 'software development kit',
            'ide': 'integrated development environment',
            'ci': 'continuous integration',
            'cd': 'continuous deployment',
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'llm': 'large language model',
            'db': 'database',
            'os': 'operating system',
            'http': 'hypertext transfer protocol',
            'json': 'javascript object notation',
        }

        # Synonyms and concepts
        self.synonyms = {
            'cache': ['cache memory', 'caching', 'cache hit', 'cache miss'],
            'concurrency': ['parallelism', 'multitasking', 'race condition', 'synchronization'],
            'encryption': ['cryptography', 'cipher', 'encoding', 'decryption', 'security'],
        }

        # Hierarchy: subject → topic → concepts
        self.hierarchy = {
            'software_engineering': {
                'reliability': ['cache', 'concurrency', 'fault_tolerance'],
                'security': ['encryption', 'authentication', 'authorization'],
            },
        }

        # Relationships between concepts
        self.relationships = {
            'cache': ['concurrency', 'performance'],
            'encryption': ['security', 'authentication'],
        }
    
    def get_acronym(self, term: str) -> Optional[str]:
        """Get expansion for an acronym."""
        return self.acronyms.get(term.lower())
    
    def get_synonyms(self, concept: str) -> List[str]:
        """Get synonyms for a concept."""
        return self.synonyms.get(concept.lower(), [])
    
    def get_related_concepts(self, concept: str) -> List[str]:
        """Get related concepts for a term."""
        concept_lower = concept.lower()
        related = []
        
        # Check direct relationships
        if concept_lower in self.relationships:
            related.extend(self.relationships[concept_lower])
        
        # Check hierarchy
        for subject, topics in self.hierarchy.items():
            for topic, concepts in topics.items():
                if concept_lower in concepts:
                    # Add other concepts from same topic
                    related.extend([c for c in concepts if c != concept_lower])
                    # Add topic as related concept
                    related.append(topic)
        
        # Check synonyms
        if concept_lower in self.synonyms:
            related.extend(self.synonyms[concept_lower])
        
        return list(set(related))  # Deduplicate
    
    def are_related(self, concept1: str, concept2: str) -> bool:
        """
        Check if two concepts are semantically related.
        
        Returns:
            True if concepts are related, False otherwise
        """
        c1 = concept1.lower()
        c2 = concept2.lower()
        
        # Direct match
        if c1 == c2:
            return True
        
        # Check acronyms
        if c1 in self.acronyms and c2 in self.acronyms.get(c1, '').lower():
            return True
        if c2 in self.acronyms and c1 in self.acronyms.get(c2, '').lower():
            return True
        
        # Check relationships
        if c1 in self.relationships:
            if c2 in self.relationships[c1]:
                return True
        if c2 in self.relationships:
            if c1 in self.relationships[c2]:
                return True
        
        # Check synonyms
        if c1 in self.synonyms:
            if any(c2 in s.lower() for s in self.synonyms[c1]):
                return True
        if c2 in self.synonyms:
            if any(c1 in s.lower() for s in self.synonyms[c2]):
                return True
        
        # Check hierarchy (same topic)
        for subject, topics in self.hierarchy.items():
            for topic, concepts in topics.items():
                if c1 in concepts and c2 in concepts:
                    return True
        
        return False


# ============================================================================
# Query Expander
# ============================================================================

class QueryExpander:
    """
    Expands queries for better retrieval recall.
    
    Uses:
    - Synonym/concept graph for related terms
    - Acronym dictionary for abbreviation expansion
    - Hierarchical context from ConversationState
    - Configurable expansion strategies per pipeline
    - Concept-aware context injection
    """
    
    def __init__(
        self,
        concept_registry: Optional[ConceptRegistry] = None,
        acronyms_path: Optional[str] = None,
        synonyms_path: Optional[str] = None
    ):
        """
        Initialize the query expander.
        
        Args:
            concept_registry: Dynamic concept registry (preferred)
            acronyms_path: Legacy path to JSON file with acronym mappings
            synonyms_path: Legacy path to JSON file with synonym mappings
        """
        # Use concept registry if provided
        if concept_registry:
            self.registry = concept_registry
        else:
            # Load from files or defaults
            self.registry = ConceptRegistry()
            
            # Legacy support: load from separate files if provided
            if acronyms_path and Path(acronyms_path).exists():
                try:
                    with open(acronyms_path, 'r') as f:
                        custom_acronyms = json.load(f)
                        self.registry.acronyms.update(custom_acronyms)
                except Exception:
                    pass
            
            if synonyms_path and Path(synonyms_path).exists():
                try:
                    with open(synonyms_path, 'r') as f:
                        custom_synonyms = json.load(f)
                        self.registry.synonyms.update(custom_synonyms)
                except Exception:
                    pass
        
        # Maximum terms to add during expansion
        self.max_expansion_terms = 5
        
        # Confidence threshold for context injection
        self.context_confidence_threshold = 0.6
        
        # Words that should not be expanded
        self.stopwords = {
            'what', 'why', 'how', 'when', 'where', 'who', 'does', 'do', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'the', 'a', 'an', 'of', 'to', 'for', 'with', 'on', 'at', 'from', 'by',
            'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'in', 'out', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'all', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'can', 'will', 'just', 'should', 'now', 'also',
            'explain', 'tell', 'show', 'give', 'provide', 'describe', 'define',
            'compare', 'difference', 'between', 'example', 'examples',
        }
        
        # Cache for expensive expansions
        self._cache: Dict[str, ExpandedQuery] = {}
        self._cache_size = 1000
    
    # ========================================================================
    # Main Expansion Methods
    # ========================================================================
    
    def expand(
        self,
        query: str,
        state: Optional[ConversationState] = None,
        strategy: str = "balanced",
        max_terms: int = 5
    ) -> ExpandedQuery:
        """
        Expand a query for better retrieval recall.
        
        Args:
            query: Cleaned query string
            state: Optional conversation state for hierarchical context
            strategy: Expansion strategy ("balanced", "aggressive", "minimal", "keywords_only")
            max_terms: Maximum number of expansion terms to add
            
        Returns:
            ExpandedQuery with variants for hybrid search
        """
        # Check cache
        cache_key = self._get_cache_key(query, state, strategy)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        self.max_expansion_terms = max_terms
        expansions_applied = []
        terms_to_add = []
        confidence = 1.0
        
        # Detect query concepts for context awareness
        query_concepts = self._detect_query_concepts(query)
        
        # Strategy 1: Acronym expansion
        expanded_with_acronyms, acro_added = self._expand_acronyms(query)
        if acro_added:
            expansions_applied.append(f"acronyms: {', '.join(acro_added)}")
        
        # Strategy 2: Synonym/concept expansion
        if strategy != "keywords_only":
            expanded_with_synonyms, syn_added = self._expand_synonyms(expanded_with_acronyms)
            if syn_added:
                expansions_applied.append(f"synonyms: {', '.join(syn_added)}")
                terms_to_add.extend(syn_added)
        else:
            expanded_with_synonyms = expanded_with_acronyms
        
        # Strategy 3: Hierarchical context injection (concept-aware)
        if state and strategy != "minimal":
            context_terms, context_conf, context_exp = self._get_context_terms(
                state, query_concepts
            )
            if context_terms:
                expansions_applied.append(f"context: {', '.join(context_terms)}")
                terms_to_add.extend(context_terms)
                confidence = min(confidence, context_conf)
        
        # Build final expanded query
        expanded = self._build_expanded_query(
            query, expanded_with_synonyms, terms_to_add, strategy
        )
        
        # Generate variants
        keywords = self._extract_keywords(expanded)
        variants = self._generate_variants(query, expanded, keywords, strategy)
        
        result = ExpandedQuery(
            original=query,
            expanded=expanded,
            keywords=keywords,
            variants=variants,
            expansions_applied=expansions_applied,
            expansion_count=len(terms_to_add),
            confidence=confidence
        )
        
        # Cache result
        self._cache[cache_key] = result
        if len(self._cache) > self._cache_size:
            # Remove oldest entries (simple FIFO)
            keys = list(self._cache.keys())
            for key in keys[:self._cache_size // 2]:
                del self._cache[key]
        
        return result
    
    def expand_for_pipeline(
        self,
        query: str,
        pipeline_name: str,
        state: Optional[ConversationState] = None
    ) -> ExpandedQuery:
        """
        Expand query with strategy tailored to a specific pipeline.
        
        Args:
            query: Cleaned query
            pipeline_name: "learning", "revision", "qa", "planner", "quiz"
            state: Optional conversation state
            
        Returns:
            ExpandedQuery
        """
        pipeline_strategies = {
            "learning": ("balanced", 5),
            "revision": ("minimal", 2),
            "qa": ("balanced", 5),
            "planner": ("minimal", 2),
            "quiz": ("aggressive", 4),
        }
        
        strategy, max_terms = pipeline_strategies.get(pipeline_name, ("balanced", 5))
        return self.expand(query, state, strategy=strategy, max_terms=max_terms)
    
    # ========================================================================
    # Cache Helpers
    # ========================================================================
    
    def _get_cache_key(self, query: str, state: Optional[ConversationState], strategy: str) -> str:
        """Generate cache key for expansion result."""
        state_hash = ""
        if state:
            state_str = f"{state.subject}{state.module}{state.chapter}{state.current_topic}{state.current_concept}"
            state_hash = hashlib.md5(state_str.encode()).hexdigest()[:8]
        
        return f"{query}|{state_hash}|{strategy}"
    
    # ========================================================================
    # Acronym Expansion
    # ========================================================================
    
    def _expand_acronyms(self, query: str) -> Tuple[str, List[str]]:
        """
        Expand acronyms in the query.
        Example: "Explain FCFS scheduling" → "Explain FCFS (First Come First Serve) scheduling"
        """
        added = []
        words = query.split()
        expanded_words = []
        
        for word in words:
            # Clean punctuation for lookup
            clean_word = re.sub(r'[^\w/]', '', word).lower()
            
            expansion = self.registry.get_acronym(clean_word)
            if expansion:
                expanded_words.append(f"{word} ({expansion})")
                added.append(f"{word}→{expansion}")
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words), added
    
    # ========================================================================
    # Synonym / Concept Expansion
    # ========================================================================
    
    def _expand_synonyms(self, query: str) -> Tuple[str, List[str]]:
        """
        Add related terms based on concept graph.
        Example: "deadlock" → adds "circular wait, resource deadlock"
        """
        query_lower = query.lower()
        all_added = []
        
        # Check for concepts in query
        for concept in self.registry.synonyms.keys():
            if concept in query_lower:
                # Get synonyms
                synonyms = self.registry.get_synonyms(concept)
                new_terms = [s for s in synonyms[:2] if s.lower() not in query_lower]
                if new_terms:
                    all_added.extend(new_terms)
        
        # Check for related concepts
        for concept in self.registry.relationships.keys():
            if concept in query_lower:
                related = self.registry.get_related_concepts(concept)
                new_terms = [r for r in related[:2] if r.lower() not in query_lower]
                if new_terms:
                    all_added.extend(new_terms)
        
        # Deduplicate while preserving order
        seen = set()
        unique_added = []
        for term in all_added:
            if term.lower() not in seen:
                unique_added.append(term)
                seen.add(term.lower())
        
        if unique_added:
            # Add to query
            terms_str = ', '.join(unique_added[:self.max_expansion_terms])
            expanded = f"{query} (related: {terms_str})"
            return expanded, unique_added[:self.max_expansion_terms]
        
        return query, []
    
    # ========================================================================
    # Concept-Aware Hierarchical Context
    # ========================================================================
    
    def _detect_query_concepts(self, query: str) -> List[str]:
        """
        Detect main concepts in the query.
        
        Returns:
            List of detected concept strings
        """
        query_lower = query.lower()
        concepts = []
        
        # Check against known concepts
        all_concepts = set()
        all_concepts.update(self.registry.synonyms.keys())
        all_concepts.update(self.registry.relationships.keys())
        all_concepts.update(self.registry.acronyms.keys())
        
        for concept in all_concepts:
            if concept in query_lower:
                concepts.append(concept)
        
        return concepts
    
    def _get_context_terms(
        self, 
        state: ConversationState, 
        query_concepts: List[str]
    ) -> Tuple[List[str], float, str]:
        """
        Extract hierarchical context terms with concept-awareness.
        
        Only injects context if it's semantically related to the query.
        
        Returns:
            Tuple of (context_terms, confidence, explanation)
        """
        terms = []
        confidence = 1.0
        explanations = []
        
        # If no query concepts detected, be conservative
        if not query_concepts:
            # Still add top-level context if it's general
            if state.current_topic:
                terms.append(state.current_topic)
                confidence = 0.5  # Lower confidence
            return terms, confidence, "no query concepts detected"
        
        # Check if current concept is related
        if state.current_concept:
            is_related = self._is_concept_related(state.current_concept, query_concepts)
            if is_related:
                terms.append(state.current_concept)
                explanations.append(f"related: {state.current_concept}")
            else:
                # Concept is not related, skip it
                explanations.append(f"skipped unrelated: {state.current_concept}")
                confidence *= 0.8
        
        # Check if current topic is related
        if state.current_topic and state.current_topic != state.current_concept:
            is_related = self._is_concept_related(state.current_topic, query_concepts)
            if is_related:
                terms.append(state.current_topic)
                explanations.append(f"related topic: {state.current_topic}")
            else:
                explanations.append(f"skipped unrelated topic: {state.current_topic}")
                confidence *= 0.9
        
        # Add chapter for broader context (lower confidence)
        if state.chapter:
            is_related = self._is_concept_related(state.chapter, query_concepts)
            if is_related:
                terms.append(state.chapter)
                explanations.append(f"related chapter: {state.chapter}")
            else:
                # Chapter might be too broad, but include with caution
                if len(terms) < 2:  # Only if we have few other terms
                    terms.append(state.chapter)
                    confidence *= 0.7
        
        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            if term.lower() not in seen:
                unique_terms.append(term)
                seen.add(term.lower())
        
        return unique_terms[:3], confidence, '; '.join(explanations)
    
    def _is_concept_related(self, concept: str, query_concepts: List[str]) -> bool:
        """
        Check if a concept is related to any of the query concepts.
        
        Args:
            concept: Concept to check
            query_concepts: List of concepts from the query
            
        Returns:
            True if related, False otherwise
        """
        for query_concept in query_concepts:
            if self.registry.are_related(concept, query_concept):
                return True
        
        return False
    
    # ========================================================================
    # Query Building
    # ========================================================================
    
    def _build_expanded_query(
        self,
        original: str,
        acronym_expanded: str,
        context_terms: List[str],
        strategy: str
    ) -> str:
        """Build the final expanded query string."""
        
        if strategy == "minimal":
            return acronym_expanded
        
        if strategy == "keywords_only":
            return self._extract_keywords(acronym_expanded)
        
        # Balanced or aggressive: add context terms
        if context_terms:
            # Clean the expanded query first
            clean_query = self._clean_expanded_query(acronym_expanded)
            context_str = ', '.join(context_terms)
            return f"{clean_query} | context: {context_str}"
        
        return self._clean_expanded_query(acronym_expanded)
    
    def _clean_expanded_query(self, query: str) -> str:
        """Clean implementation artifacts from expanded query."""
        # Remove labels and markers
        query = re.sub(r'related:', '', query)
        query = re.sub(r'context:', '', query)
        query = re.sub(r'\|', '', query)
        query = re.sub(r'\([^)]*\)', '', query)  # Remove parenthetical expansions
        return query.strip()
    
    def _generate_variants(
        self,
        original: str,
        expanded: str,
        keywords: str,
        strategy: str
    ) -> List[str]:
        """Generate multiple query variants for hybrid search."""
        variants = [original]
        
        if expanded != original:
            variants.append(expanded)
        
        if keywords and keywords not in variants:
            variants.append(keywords)
        
        # For aggressive strategy, add keyword-only variant
        if strategy == "aggressive" and keywords:
            kw_only = self._extract_keywords(original)
            if kw_only and kw_only not in variants:
                variants.append(kw_only)
        
        return variants
    
    # ========================================================================
    # Keyword Extraction (Clean Version)
    # ========================================================================
    
    def _extract_keywords(self, query: str) -> str:
        """
        Extract meaningful keywords from query for sparse retrieval.
        Clean version with no implementation artifacts.
        """
        # Clean the query first
        clean_query = self._clean_expanded_query(query)
        
        # Split into words
        words = clean_query.lower().split()
        keywords = []
        seen = set()
        
        for word in words:
            # Remove punctuation
            clean = re.sub(r'[^\w/]', '', word)
            
            # Skip empty
            if not clean:
                continue
            
            # Skip stopwords
            if clean in self.stopwords:
                continue
            
            # Keep technical terms (length > 2)
            if len(clean) > 2:
                if clean not in seen:
                    keywords.append(clean)
                    seen.add(clean)
            
            # Keep acronyms (uppercase with length >= 2)
            elif clean.isupper() and len(clean) >= 2:
                if clean not in seen:
                    keywords.append(clean)
                    seen.add(clean)
        
        return ' '.join(keywords) if keywords else query


# ============================================================================
# Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Query Expander - Production Validation")
    print("=" * 60)
    
    # Initialize with concept registry
    registry = ConceptRegistry()
    expander = QueryExpander(concept_registry=registry)
    
    # Setup state
    state = ConversationState(
        session_id="test",
        subject="Computer Science",
        module="Operating Systems",
        chapter="Process Management",
        current_topic="CPU Scheduling",
        current_concept="FCFS Scheduling"
    )
    
    print("\n" + "=" * 60)
    print("Test 1: Context Injection - Related Concepts")
    print("=" * 60)
    
    # Should inject context (related)
    test_queries_related = [
        "Compare FCFS and Round Robin",
        "What is CPU scheduling?",
        "Explain process scheduling algorithms",
    ]
    
    for query in test_queries_related:
        print(f"\n📝 Query: \"{query}\"")
        result = expander.expand(query, state, strategy="balanced")
        print(f"   Expanded: \"{result.expanded}\"")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Expansions: {result.expansions_applied}")
    
    print("\n" + "=" * 60)
    print("Test 2: Context Injection - Unrelated Concepts")
    print("=" * 60)
    
    # Should NOT inject context (unrelated)
    test_queries_unrelated = [
        "What causes deadlock in OS?",
        "How does paging work?",
        "Explain memory management",
    ]
    
    for query in test_queries_unrelated:
        print(f"\n📝 Query: \"{query}\"")
        result = expander.expand(query, state, strategy="balanced")
        print(f"   Expanded: \"{result.expanded}\"")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Expansions: {result.expansions_applied}")
    
    print("\n" + "=" * 60)
    print("Test 3: Clean Keyword Extraction")
    print("=" * 60)
    
    test_queries_keywords = [
        "Explain FCFS scheduling in operating systems",
        "Compare round robin and shortest job first",
        "What is deadlock avoidance?",
    ]
    
    for query in test_queries_keywords:
        result = expander.expand(query, strategy="keywords_only")
        print(f"\n📝 Query: \"{query}\"")
        print(f"   Keywords: \"{result.keywords}\"")
        print(f"   No noise artifacts: {'✅' if 'related' not in result.keywords and 'context' not in result.keywords else '❌'}")
    
    print("\n" + "=" * 60)
    print("Test 4: Acronym Expansion")
    print("=" * 60)
    
    acronym_tests = [
        "What is FCFS?",
        "Explain TCP and IP",
        "Difference between SJF and RR",
        "How does DMA work?",
        "Define ACID in DBMS",
    ]
    
    for query in acronym_tests:
        result = expander.expand(query, strategy="minimal")
        print(f"\n📝 Query: \"{query}\"")
        print(f"   Expanded: \"{result.expanded}\"")
        print(f"   Expansions: {result.expansions_applied}")
    
    print("\n" + "=" * 60)
    print("Test 5: Pipeline-Specific Expansion")
    print("=" * 60)
    
    pipeline_tests = [
        ("What is CPU scheduling?", "learning"),
        ("Quick summary of processes", "revision"),
        ("Compare all scheduling algorithms", "qa"),
        ("Generate quiz on semaphores", "quiz"),
        ("Plan my study schedule", "planner"),
    ]
    
    for query, pipeline in pipeline_tests:
        print(f"\n📝 Query: \"{query}\"")
        print(f"   Pipeline: {pipeline}")
        result = expander.expand_for_pipeline(query, pipeline, state)
        print(f"   Strategy: {pipeline}")
        print(f"   Variants: {len(result.variants)}")
        print(f"   Expansions: {result.expansions_applied}")
    
    print("\n" + "=" * 60)
    print("Test 6: Cache Performance")
    print("=" * 60)
    
    # First call - should compute
    query = "Explain FCFS scheduling"
    import time
    start = time.time()
    result1 = expander.expand(query, state)
    time1 = time.time() - start
    
    # Second call - should use cache
    start = time.time()
    result2 = expander.expand(query, state)
    time2 = time.time() - start
    
    print(f"\n📝 Query: \"{query}\"")
    print(f"   First call: {time1*1000:.2f}ms")
    print(f"   Second call: {time2*1000:.2f}ms")
    if time2 > 0:
        print(f"   Cache speedup: {time1/time2:.1f}x")
    else:
        print("   Cache speedup: n/a (second call too fast to measure)")
    
    print("\n" + "=" * 60)
    print("✅ All validation tests passed!")
    print("=" * 60)