"""
Lightweight Query Rewriter
Handles deterministic, fast preprocessing only.
Complex reasoning and novel follow-ups are delegated to the Agent layer.

Keeps:
- Pronoun resolution (deterministic, fast)
- 6 core follow-up patterns
- 11 core academic intents
- Comparison extraction (regex)
- Self-containment check
- Keyword extraction
- Rewrite caching
- Dual retrieval variants

Delegates to Agent:
- Exam-specific intents (PYQ, MCQ, marks-based, interview, etc.)
- Novel/unpredictable follow-ups
- Ambiguity detection
- Query decomposition
- Complex multi-turn reasoning
"""

import re
import json
import logging
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from pathlib import Path

# Optional: FlashText for efficient term lookup
try:
    from flashtext import KeywordProcessor
    FLASHTEXT_AVAILABLE = True
except ImportError:
    FLASHTEXT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Enums — Core academic intents only
# ============================================================================

class IntentType(str, Enum):
    """Core academic intents. Exam-specific variants handled by Agent."""
    EXPLANATION = "explanation"
    EXAMPLE = "example"
    COMPARISON = "comparison"
    ELABORATION = "elaboration"
    REPEAT = "repeat"
    SUMMARY = "summary"
    FORMULA = "formula"
    ADVANTAGES = "advantages"
    DISADVANTAGES = "disadvantages"
    APPLICATIONS = "applications"
    DEFINITION = "definition"


class RewriteType(str, Enum):
    """Types of rewriting performed."""
    NO_REWRITE = "no_rewrite"
    PRONOUN_RESOLUTION = "pronoun_resolution"
    CONTEXT_APPEND = "context_append"
    INTENT_EXPANSION = "intent_expansion"
    COMPARISON_RESOLUTION = "comparison_resolution"
    FOLLOWUP_RESOLUTION = "followup_resolution"
    CACHE_HIT = "cache_hit"


# ============================================================================
# Conversation Memory (unchanged — Agent also needs this)
# ============================================================================

@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str
    content: str
    topic: Optional[str] = None
    concept: Optional[str] = None
    subconcept: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationState:
    """
    Tracks the current state of a student's learning session.
    Hierarchy: Subject → Module → Chapter → Topic → Concept → Subconcept
    """
    session_id: str
    subject: Optional[str] = None
    module: Optional[str] = None
    chapter: Optional[str] = None
    current_topic: Optional[str] = None
    current_concept: Optional[str] = None
    current_subconcept: Optional[str] = None
    history: deque = field(default_factory=lambda: deque(maxlen=20))
    
    last_explained_concept: Optional[str] = None
    last_compared_pair: Optional[Tuple[str, str]] = None
    concept_history: List[str] = field(default_factory=list)
    entity_history: List[str] = field(default_factory=list)
    
    rewrite_cache: Dict[str, Tuple[str, Dict[str, Any]]] = field(default_factory=dict)
    
    def add_turn(self, role: str, content: str,
                 topic: str = None, concept: str = None,
                 subconcept: str = None, entities: List[str] = None):
        """Add a conversation turn."""
        turn = ConversationTurn(
            role=role, content=content,
            topic=topic or self.current_topic,
            concept=concept or self.current_concept,
            subconcept=subconcept or self.current_subconcept,
            entities=entities or []
        )
        self.history.append(turn)
        
        if role == "assistant":
            if topic:
                self.current_topic = topic
            if concept:
                self.current_concept = concept
                self.last_explained_concept = concept
                self.concept_history.append(concept)
            if subconcept:
                self.current_subconcept = subconcept
            if entities:
                self.entity_history.extend(entities)
        
        if len(self.concept_history) > 10:
            self.concept_history = self.concept_history[-10:]
        if len(self.entity_history) > 20:
            self.entity_history = self.entity_history[-20:]
    
    def update_comparison(self, concept_a: str, concept_b: str):
        """Update the last compared pair."""
        self.last_compared_pair = (concept_a, concept_b)
    
    def get_cache_key(self, query: str) -> str:
        key_parts = [
            query,
            self.current_concept or "",
            self.current_subconcept or "",
            self.current_topic or "",
            str(self.last_compared_pair) if self.last_compared_pair else ""
        ]
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()
    
    def get_recent_concepts(self, n: int = 3) -> List[str]:
        concepts = []
        for turn in reversed(self.history):
            if turn.role == "assistant" and turn.concept:
                if turn.concept not in concepts:
                    concepts.append(turn.concept)
                if len(concepts) >= n:
                    break
        return concepts
    
    def get_recent_entities(self, n: int = 3) -> List[str]:
        return self.entity_history[-n:] if self.entity_history else []
    
    def get_primary_concept(self) -> Optional[str]:
        """Get most specific concept available."""
        return (self.current_subconcept or self.current_concept or 
                self.current_topic or self.chapter or self.module or self.subject)
    
    def get_context_summary(self) -> str:
        parts = []
        if self.subject: parts.append(f"Subject: {self.subject}")
        if self.module: parts.append(f"Module: {self.module}")
        if self.chapter: parts.append(f"Chapter: {self.chapter}")
        if self.current_topic: parts.append(f"Topic: {self.current_topic}")
        if self.current_concept: parts.append(f"Concept: {self.current_concept}")
        if self.current_subconcept: parts.append(f"Subconcept: {self.current_subconcept}")
        return " | ".join(parts) if parts else "No active context"


# ============================================================================
# Lightweight Query Rewriter
# ============================================================================

class QueryRewriter:
    """
    Lightweight query preprocessor. Handles deterministic rewrites only.
    Novel/complex queries are passed through to the Agent layer.
    """
    
    # Core incomplete patterns (universal academic patterns)
    INCOMPLETE_PATTERNS = [
        (re.compile(r'^(why|how|when|where|what)$', re.IGNORECASE), IntentType.EXPLANATION),
        (re.compile(r'^(example|give example|show example|illustrate|demonstrate|sample)$', re.IGNORECASE), IntentType.EXAMPLE),
        (re.compile(r'^(difference|diff|compare|vs\.?|versus)$', re.IGNORECASE), IntentType.COMPARISON),
        (re.compile(r'^(explain|elaborate|describe|tell me more|go on)$', re.IGNORECASE), IntentType.ELABORATION),
        (re.compile(r'^(again|repeat|re-explain|explain again|once more)$', re.IGNORECASE), IntentType.REPEAT),
        (re.compile(r'^(summary|summarize|tldr|brief|in short|recap)$', re.IGNORECASE), IntentType.SUMMARY),
        (re.compile(r'^(formula|equation|math|calculation)$', re.IGNORECASE), IntentType.FORMULA),
        (re.compile(r'^(advantages|pros|benefits|merits|strengths)$', re.IGNORECASE), IntentType.ADVANTAGES),
        (re.compile(r'^(disadvantages|cons|drawbacks|demerits|limitations|weaknesses)$', re.IGNORECASE), IntentType.DISADVANTAGES),
        (re.compile(r'^(applications|uses|applied|real world|practical|usage)$', re.IGNORECASE), IntentType.APPLICATIONS),
        (re.compile(r'^(definition|define|meaning|what is)$', re.IGNORECASE), IntentType.DEFINITION),
    ]
    
    # Only 6 core follow-up patterns — rest handled by Agent
    FOLLOWUP_PATTERNS = [
        re.compile(r'^why\s*(though|so|is that)?$', re.IGNORECASE),
        re.compile(r'^how\s*(so|come)?$', re.IGNORECASE),
        re.compile(r'^(example|give example|show example|illustrate|demonstrate|sample)$', re.IGNORECASE),
        re.compile(r'^(difference|diff|compare|vs\.?|versus)$', re.IGNORECASE),
        re.compile(r'^(explain|elaborate|describe|tell me more|go on)$', re.IGNORECASE),
        re.compile(r'^(again|repeat|re-explain|explain again|once more)$', re.IGNORECASE),
    ]
    
    # Comparison extraction regex
    COMPARISON_REGEX = re.compile(
        r'compare\s+(.+?)\s+(?:and|vs\.?|versus|with)\s+(.+?)(?:\?|$|\.)',
        re.IGNORECASE
    )
    DIFFERENCE_REGEX = re.compile(
        r'(?:difference|diff)\s+(?:between\s+)?(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)(?:\?|$|\.)',
        re.IGNORECASE
    )
    
    # Intent templates — core academic intents only
    INTENT_TEMPLATES = {
        IntentType.EXPLANATION: "Explain why {concept} works the way it does.",
        IntentType.EXAMPLE: "Provide a clear example of {concept}.",
        IntentType.COMPARISON: "Compare {concept_a} and {concept_b} in detail.",
        IntentType.ELABORATION: "Explain {concept} in detail.",
        IntentType.REPEAT: "Explain {concept} again with different examples.",
        IntentType.SUMMARY: "Summarize the key points of {concept}.",
        IntentType.FORMULA: "Key formulas and equations for {concept}.",
        IntentType.ADVANTAGES: "What are the main advantages of {concept}?",
        IntentType.DISADVANTAGES: "What are the main disadvantages of {concept}?",
        IntentType.APPLICATIONS: "What are the real-world applications of {concept}?",
        IntentType.DEFINITION: "Define {concept} clearly.",
    }
    
    def __init__(self, technical_terms_path: Optional[str] = None, max_cache_size: int = 1000):
        """
        Initialize the lightweight query rewriter.
        
        Args:
            technical_terms_path: Path to JSON file with technical terms
            max_cache_size: Maximum number of cached rewrites
        """
        self.max_cache_size = max_cache_size
        self.technical_terms = self._load_technical_terms(technical_terms_path)
        
        # Initialize FlashText if available
        self.use_flashtext = FLASHTEXT_AVAILABLE
        if self.use_flashtext:
            self.keyword_processor = KeywordProcessor()
            for term in self.technical_terms:
                self.keyword_processor.add_keyword(term)
        
        # Pronoun infrastructure
        self.pronouns = ["it", "they", "them", "this", "that", "these", "those"]
        self.pronoun_pattern = re.compile(
            r'\b(' + '|'.join(self.pronouns) + r')\b', re.IGNORECASE
        )
        self.plural_pronouns = {"they", "them", "these", "those"}
        
        logger.info(f"Lightweight QueryRewriter initialized with {len(self.technical_terms)} terms")
    
    def _load_technical_terms(self, path: Optional[str]) -> Set[str]:
        """Load technical terms from JSON or use defaults."""
        default_terms = {
            'deadlock', 'scheduling', 'paging', 'segmentation', 'virtual memory',
            'thread', 'process', 'semaphore', 'mutex', 'cache', 'cpu', 'memory',
            'disk', 'file system', 'interrupt', 'system call', 'kernel',
            'algorithm', 'data structure', 'network', 'protocol', 'tcp', 'ip',
            'os', 'operating system', 'database', 'sql', 'transaction',
            'oop', 'inheritance', 'polymorphism', 'encapsulation', 'abstraction',
            'compiler', 'parser', 'lexer', 'linked list', 'tree', 'graph',
            'hash table', 'queue', 'stack', 'fcfs', 'round robin',
            'priority scheduling', 'shortest job first'
        }
        
        if path and Path(path).exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return default_terms | set(data)
                    elif isinstance(data, dict):
                        return default_terms | set(data.get('terms', []))
            except Exception as e:
                logger.warning(f"Failed to load technical terms: {e}")
        
        return default_terms
    
    # ========================================================================
    # Main Rewrite Method
    # ========================================================================
    
    def rewrite(self, query: str, state: ConversationState) -> Tuple[str, Dict[str, Any]]:
        """
        Rewrite a query using deterministic rules only.
        Returns (rewritten_query, metadata).
        If query cannot be confidently rewritten, returns original query with
        needs_agent=True in metadata.
        """
        original = query.strip()
        
        # Check cache
        cache_key = state.get_cache_key(original)
        if cache_key in state.rewrite_cache:
            cached_query, cached_metadata = state.rewrite_cache[cache_key]
            cached_metadata["cache_hit"] = True
            cached_metadata["rewrite_type"] = RewriteType.CACHE_HIT
            return cached_query, cached_metadata
        
        metadata = {
            "original_query": original,
            "was_rewritten": False,
            "rewrite_type": RewriteType.NO_REWRITE,
            "rewrite_confidence": 1.0,
            "detected_intent": None,
            "needs_agent": False,           # Key: tells router if Agent is needed
            "agent_reason": "",             # Why Agent is needed (if applicable)
            "context_used": state.get_context_summary(),
        }
        
        # Step 1: Check if self-contained
        is_sc, reason = self._is_self_contained(original)
        if is_sc:
            metadata["reason"] = reason
            state.rewrite_cache[cache_key] = (original, metadata)
            self._trim_cache(state)
            return original, metadata
        
        # Step 2: Resolve pronouns
        if self._has_pronouns(original):
            resolved = self._resolve_pronouns(original, state)
            if resolved != original:
                query = resolved
                metadata["was_rewritten"] = True
                metadata["rewrite_type"] = RewriteType.PRONOUN_RESOLUTION
        
        # Step 3: Extract comparison
        comparison = self._extract_comparison(query)
        if comparison:
            query = comparison
            metadata["was_rewritten"] = True
            metadata["rewrite_type"] = RewriteType.COMPARISON_RESOLUTION
        
        # Step 4: Detect intent
        detected_intent = self._detect_intent(original)
        metadata["detected_intent"] = detected_intent.value if detected_intent else None
        
        # Step 5: Build intent query
        if detected_intent and not metadata["was_rewritten"]:
            rewritten = self._build_intent_query(original, detected_intent, state)
            if rewritten != original:
                query = rewritten
                metadata["was_rewritten"] = True
                metadata["rewrite_type"] = RewriteType.INTENT_EXPANSION
        
        # Step 6: Follow-up resolution
        if self._is_followup(original) and not metadata["was_rewritten"]:
            resolved = self._resolve_followup(original, state)
            if resolved != original:
                query = resolved
                metadata["was_rewritten"] = True
                metadata["rewrite_type"] = RewriteType.FOLLOWUP_RESOLUTION
        
        # Step 7: If still not rewritten and not self-contained → needs Agent
        if not metadata["was_rewritten"] and not is_sc:
            metadata["needs_agent"] = True
            metadata["agent_reason"] = f"Query not matched by rule patterns: '{original}'"
        
        # Calculate confidence
        metadata["rewrite_confidence"] = self._calculate_confidence(query, state, metadata)
        
        # Final check: if confidence is low and we have context → Agent can do better
        if metadata["rewrite_confidence"] < 0.5 and state.get_primary_concept():
            metadata["needs_agent"] = True
            metadata["agent_reason"] = f"Low rule confidence ({metadata['rewrite_confidence']:.2f})"
        
        metadata["rewritten_query"] = query
        
        # Cache
        state.rewrite_cache[cache_key] = (query, metadata)
        self._trim_cache(state)
        
        if metadata["was_rewritten"]:
            logger.info(f"Rewritten: '{original}' → '{query}' (confidence: {metadata['rewrite_confidence']:.2f})")
        
        return query, metadata
    
    def _trim_cache(self, state: ConversationState):
        """Trim cache if exceeds max size."""
        if len(state.rewrite_cache) > self.max_cache_size:
            items = list(state.rewrite_cache.items())
            for key, _ in items[:len(items) // 5]:
                del state.rewrite_cache[key]
    
    def rewrite_for_dual_retrieval(self, query: str, state: ConversationState) -> Tuple[List[str], Dict[str, Any]]:
        """Generate multiple query variants for hybrid retrieval."""
        rewritten, metadata = self.rewrite(query, state)
        variants = [query]
        if rewritten != query:
            variants.append(rewritten)
        keywords = self._extract_keywords(rewritten)
        if keywords and keywords not in variants:
            variants.append(keywords)
        metadata["query_variants"] = variants
        return variants, metadata
    
    # ========================================================================
    # Self-Containment Check
    # ========================================================================
    
    def _is_self_contained(self, query: str) -> Tuple[bool, str]:
        """Check if query is self-contained."""
        words = query.split()
        query_lower = query.lower()
        
        # Technical terms → self-contained
        if self.use_flashtext:
            if self.keyword_processor.extract_keywords(query_lower):
                return True, "contains_technical_term"
        else:
            for term in self.technical_terms:
                if term in query_lower:
                    return True, "contains_technical_term"
        
        if len(words) >= 8:
            return True, "long_query"
        if self.pronoun_pattern.search(query):
            return False, "contains_pronoun"
        for pattern, _ in self.INCOMPLETE_PATTERNS:
            if pattern.match(query):
                return False, "incomplete_pattern"
        if len(words) >= 5:
            return True, "medium_query_no_pronouns"
        if query.endswith('?') and len(words) >= 4:
            return True, "question_with_content"
        
        return False, "too_short_or_vague"
    
    # ========================================================================
    # Pronoun Resolution
    # ========================================================================
    
    def _has_pronouns(self, query: str) -> bool:
        return bool(self.pronoun_pattern.search(query))
    
    def _resolve_pronouns(self, query: str, state: ConversationState) -> str:
        concept = state.get_primary_concept()
        if not concept:
            return query
        
        entities = state.get_recent_entities(2)
        
        def replace_pronoun(match):
            pronoun = match.group(0).lower()
            if pronoun in self.plural_pronouns:
                if len(entities) >= 2:
                    return f"{entities[0]} and {entities[1]}"
                elif entities:
                    return f"{entities[0]} and related concepts"
                return f"{concept} and related concepts"
            else:
                return entities[0] if entities else concept
        
        resolved = self.pronoun_pattern.sub(replace_pronoun, query)
        if concept:
            resolved = re.sub(r'\b' + re.escape(concept) + r'\s+it\b', concept, resolved, flags=re.IGNORECASE)
            resolved = re.sub(r'\b' + re.escape(concept) + r'\s+this\b', concept, resolved, flags=re.IGNORECASE)
        
        return resolved
    
    # ========================================================================
    # Comparison Extraction
    # ========================================================================
    
    def _extract_comparison(self, query: str) -> Optional[str]:
        for pattern in [self.COMPARISON_REGEX, self.DIFFERENCE_REGEX]:
            match = pattern.search(query)
            if match:
                a = re.sub(r'[^\w\s]', '', match.group(1).strip())
                b = re.sub(r'[^\w\s]', '', match.group(2).strip())
                if a and b:
                    return f"Compare {a} and {b} in detail. What are the key differences and similarities?"
        return None
    
    # ========================================================================
    # Intent Detection
    # ========================================================================
    
    def _detect_intent(self, query: str) -> Optional[IntentType]:
        query_lower = query.lower()
        
        for pattern, intent in self.INCOMPLETE_PATTERNS:
            if pattern.match(query_lower):
                return intent
        
        if self.COMPARISON_REGEX.search(query_lower) or self.DIFFERENCE_REGEX.search(query_lower):
            return IntentType.COMPARISON
        
        for pattern in self.FOLLOWUP_PATTERNS:
            if pattern.match(query_lower):
                return IntentType.ELABORATION
        
        return None
    
    def _is_followup(self, query: str) -> bool:
        query_lower = query.lower()
        return any(p.match(query_lower) for p in self.FOLLOWUP_PATTERNS)
    
    def _resolve_followup(self, query: str, state: ConversationState) -> str:
        concept = state.get_primary_concept()
        if not concept:
            return query
        
        mapping = {
            r'^why': f"Why does {concept} work the way it does?",
            r'^how': f"How does {concept} work?",
            r'^example': f"Provide a clear example of {concept}.",
            r'^difference|^diff|^compare': f"Compare {concept} with related concepts.",
            r'^explain|^elaborate|^describe': f"Explain {concept} in detail.",
            r'^again|^repeat': f"Explain {concept} again with different examples.",
        }
        
        for pattern, template in mapping.items():
            if re.match(pattern, query.lower()):
                return template
        
        return f"Explain {concept} further."
    
    def _build_intent_query(self, query: str, intent: IntentType, state: ConversationState) -> str:
        template = self.INTENT_TEMPLATES.get(intent)
        if not template:
            return query
        
        if intent == IntentType.COMPARISON:
            concepts = self._extract_concepts_from_query(query, state)
            if len(concepts) >= 2:
                return template.format(concept_a=concepts[0], concept_b=concepts[1])
            elif state.last_compared_pair:
                a, b = state.last_compared_pair
                return template.format(concept_a=a, concept_b=b)
            concept = state.get_primary_concept()
            return template.format(concept_a=concept or "this", concept_b="related concepts")
        
        concept = state.get_primary_concept()
        if concept and "{concept}" in template:
            return template.format(concept=concept)
        
        return query
    
    def _extract_concepts_from_query(self, query: str, state: ConversationState) -> List[str]:
        concepts = []
        query_lower = query.lower()
        
        for pattern in [self.COMPARISON_REGEX, self.DIFFERENCE_REGEX]:
            match = pattern.search(query)
            if match:
                a, b = match.group(1).strip(), match.group(2).strip()
                if a and b:
                    return [a, b]
        
        if self.use_flashtext:
            found = self.keyword_processor.extract_keywords(query_lower)
            concepts = list(dict.fromkeys(found))
        else:
            concepts = [term for term in self.technical_terms if term in query_lower]
        
        if not concepts:
            concepts = [w for w in query.split() if w and w[0].isupper() and len(w) > 2]
        
        return concepts[:2]
    
    # ========================================================================
    # Keyword Extraction
    # ========================================================================
    
    def _extract_keywords(self, query: str) -> str:
        stopwords = {'what', 'why', 'how', 'when', 'where', 'does', 'do', 'is', 'are',
                    'the', 'a', 'an', 'of', 'to', 'for', 'with', 'on', 'at', 'from',
                    'can', 'you', 'me', 'explain', 'tell', 'show', 'give', 'provide'}
        
        words = query.lower().split()
        keywords = []
        i = 0
        while i < len(words):
            if self.use_flashtext:
                sub_text = ' '.join(words[i:i+3])
                found = self.keyword_processor.extract_keywords(sub_text)
                if found:
                    keywords.append(found[0])
                    i += len(found[0].split())
                    continue
            word = words[i]
            if word not in stopwords and len(word) > 2:
                keywords.append(word)
            i += 1
        
        return ' '.join(keywords) if keywords else query
    
    # ========================================================================
    # Confidence Calculation (simplified)
    # ========================================================================
    
    def _calculate_confidence(self, query: str, state: ConversationState, metadata: Dict) -> float:
        weights = {
            "context_available": 0.30,
            "pronoun_resolved": 0.25,
            "detected_intent": 0.20,
            "technical_terms": 0.15,
            "query_length": 0.10,
        }
        
        features = {
            "context_available": 1.0 if state.get_primary_concept() else 0.0,
            "pronoun_resolved": 1.0 if metadata["rewrite_type"] == RewriteType.PRONOUN_RESOLUTION else 0.0,
            "detected_intent": 1.0 if metadata["detected_intent"] else 0.0,
            "technical_terms": min(1.0, len(self._extract_concepts_from_query(query, state)) / 2.0),
            "query_length": min(1.0, len(query.split()) / 8.0),
        }
        
        return min(0.95, sum(features[k] * weights[k] for k in weights))


# ============================================================================
# Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lightweight Query Rewriter v4.0 - Validation")
    print("=" * 60)
    
    rewriter = QueryRewriter()
    
    state = ConversationState(
        session_id="test",
        subject="Computer Science",
        module="Operating Systems",
        chapter="Process Management",
        current_topic="Concurrency",
        current_concept="Deadlock"
    )
    
    state.add_turn("user", "What is deadlock?")
    state.add_turn("assistant", "Deadlock is a situation where...",
                   topic="Concurrency", concept="Deadlock")
    state.update_comparison("Deadlock", "Starvation")
    
    print(f"\nContext: {state.get_context_summary()}\n")
    
    test_queries = [
        # Should be rewritten by rules
        "Why?",
        "Give example",
        "Compare FCFS and Round Robin",
        "difference between paging and segmentation",
        "How do they differ?",
        
        # Should return needs_agent=True (not matched by core patterns)
        "Can this happen in real life?",
        "Which one would you choose for embedded systems?",
        "If I change the assumptions, would starvation still occur?",
        "Answer this from an exam perspective",
        "Make it suitable for a 5-mark answer",
        "Connect this with DBMS concepts",
        "Give me an analogy for this",
    ]
    
    for query in test_queries:
        rewritten, metadata = rewriter.rewrite(query, state)
        
        if metadata["needs_agent"]:
            print(f"🤖 AGENT | \"{query}\"")
            print(f"        → Reason: {metadata['agent_reason']}")
        elif metadata["was_rewritten"]:
            print(f"✅ RULE | \"{query}\"")
            print(f"        → \"{rewritten}\"")
        else:
            print(f"📌 PASS | \"{query}\" (self-contained)")
        print()