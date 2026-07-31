"""
Reranker
Cross-encoder and LLM-based reranking for candidate refinement.

Takes the candidate pool from hybrid search and selects the most
relevant chunks using a cross-encoder model or LLM-based scoring.

Backends:
- Cross-encoder: Fast, local model (default)
- LLM: DeepSeek for high-quality but slower reranking
- ColBERT: Token-level late interaction (future)
- Hybrid: Weighted combination of multiple rerankers

Configuration-driven — all models, thresholds, and strategies
loaded from config/retrieval/reranker.yaml with environment
variable overrides.
"""

import os
import time
import hashlib
import json
import re
import yaml
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from dotenv import load_dotenv
import numpy as np

from src.common.llm_client import simple_generate
from src.retrieval.config import RetrievedChunk, ConfidenceLevel

load_dotenv()


# ============================================================================
# Enhanced Cache Implementation
# ============================================================================

class LRUCache:
    """
    Thread-safe LRU cache with TTL support.
    
    Features:
    - LRU eviction when max size exceeded
    - TTL-based expiration
    - Thread-safe for concurrent access
    - Optional size limiting
    """
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        """
        Initialize the LRU cache.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from cache if not expired.
        
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            value, timestamp = self._cache[key]
            
            # Check TTL
            if self.ttl_seconds > 0 and (time.time() - timestamp) > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value
    
    def put(self, key: str, value: Any) -> None:
        """Store a value in cache."""
        with self._lock:
            if key in self._cache:
                # Update existing entry
                self._cache[key] = (value, time.time())
                self._cache.move_to_end(key)
            else:
                # Add new entry
                self._cache[key] = (value, time.time())
                
                # Evict if over limit
                while len(self._cache) > self.max_size:
                    self._cache.popitem(last=False)
    
    def invalidate(self, key: str) -> None:
        """Remove a specific key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache hit/miss statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "size": len(self._cache)
            }


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RerankerConfig:
    """Configuration for reranking — loaded from YAML."""
    backend_type: str = "cross_encoder"
    model_name: Optional[str] = None
    input_k: int = 40
    output_k: int = 5
    min_score_threshold: float = 0.1
    normalize_scores: bool = True
    normalize_method: str = "softmax"
    temperature: float = 1.0
    batch_size: int = 32
    use_fp16: bool = False
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    max_cache_entries: int = 10000
    cross_encoder_max_length: int = 512
    cross_encoder_show_progress: bool = False
    llm_confidence_threshold: float = 0.7
    hybrid_weights: Dict[str, float] = field(default_factory=dict)
    default_models: Dict[str, str] = field(default_factory=dict)
    parallel_hybrid: bool = True  # NEW: Execute hybrid rerankers in parallel
    max_parallel_workers: int = 4  # NEW: Max parallel threads for hybrid

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "RerankerConfig":
        """
        Load configuration from YAML file.
        
        Priority:
        1. Explicit path argument
        2. RAGPIPE_RERANKER_CONFIG env var
        3. config/retrieval/reranker.yaml (default)
        """
        config_path = (
            path
            or os.getenv("RAGPIPE_RERANKER_CONFIG")
            or str(Path(__file__).parent.parent.parent / "config" / "retrieval" / "reranker.yaml")
        )
        
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
        else:
            data = cls._default_config()
        
        data = cls._apply_env_overrides(data)
        
        backend = data.get("backend", {})
        candidates = data.get("candidates", {})
        scoring = data.get("scoring", {})
        performance = data.get("performance", {})
        ce_config = data.get("cross_encoder", {})
        llm_config = data.get("llm_reranker", {})
        hybrid_config = data.get("hybrid", {})
        
        return cls(
            backend_type=str(backend.get("type", "cross_encoder")),
            model_name=backend.get("model"),
            input_k=int(candidates.get("input_k", 40)),
            output_k=int(candidates.get("output_k", 5)),
            min_score_threshold=float(candidates.get("min_score_threshold", 0.1)),
            normalize_scores=bool(scoring.get("normalize", True)),
            normalize_method=str(scoring.get("method", "softmax")),
            temperature=float(scoring.get("temperature", 1.0)),
            batch_size=int(performance.get("batch_size", 32)),
            use_fp16=bool(performance.get("use_fp16", False)),
            cache_enabled=bool(performance.get("cache_enabled", True)),
            cache_ttl_seconds=int(performance.get("cache_ttl_seconds", 3600)),
            max_cache_entries=int(performance.get("max_cache_entries", 10000)),
            cross_encoder_max_length=int(ce_config.get("max_length", 512)),
            cross_encoder_show_progress=bool(ce_config.get("show_progress", False)),
            llm_confidence_threshold=float(llm_config.get("confidence_threshold", 0.7)),
            hybrid_weights=dict(hybrid_config.get("weights", {})),
            default_models=dict(backend.get("default_models", {})),
            parallel_hybrid=bool(hybrid_config.get("parallel", True)),
            max_parallel_workers=int(performance.get("max_parallel_workers", 4)),
        )
    
    @classmethod
    def _apply_env_overrides(cls, data: Dict) -> Dict:
        """Apply environment variable overrides."""
        env_mapping = {
            "RAGPIPE_RERANKER_BACKEND": ("backend", "type"),
            "RAGPIPE_RERANKER_MODEL": ("backend", "model"),
            "RAGPIPE_RERANKER_INPUT_K": ("candidates", "input_k"),
            "RAGPIPE_RERANKER_OUTPUT_K": ("candidates", "output_k"),
            "RAGPIPE_RERANKER_MIN_SCORE": ("candidates", "min_score_threshold"),
            "RAGPIPE_RERANKER_BATCH_SIZE": ("performance", "batch_size"),
            "RAGPIPE_RERANKER_PARALLEL": ("hybrid", "parallel"),
        }
        
        for env_var, (section, key) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in data:
                    data[section] = {}
                try:
                    if key in ["parallel"]:
                        data[section][key] = value.lower() in ["true", "1", "yes"]
                    elif "." in value:
                        data[section][key] = float(value)
                    else:
                        data[section][key] = int(value)
                except ValueError:
                    data[section][key] = value
        
        return data
    
    @staticmethod
    def _default_config() -> Dict:
        """Minimal fallback configuration."""
        return {
            "backend": {
                "type": "cross_encoder",
                "model": None,
                "default_models": {
                    "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "llm": "deepseek-chat",
                    "colbert": "colbert-ir/colbertv2.0",
                },
            },
            "candidates": {
                "input_k": 40,
                "output_k": 5,
                "min_score_threshold": 0.1,
            },
            "scoring": {
                "normalize": True,
                "method": "softmax",
                "temperature": 1.0,
            },
            "performance": {
                "batch_size": 32,
                "use_fp16": False,
                "cache_enabled": True,
                "cache_ttl_seconds": 3600,
                "max_cache_entries": 10000,
                "max_parallel_workers": 4,
            },
            "hybrid": {
                "parallel": True,
                "weights": {"cross_encoder": 0.7, "llm": 0.3},
            },
        }


# ============================================================================
# Reranker Result Types
# ============================================================================

@dataclass
class RerankedChunk:
    """A single reranked chunk with score and source traceability."""
    chunk_id: str
    content: str
    raw_content: str
    original_score: float        # Score from hybrid search
    rerank_score: float          # Score from reranker
    rank: int                    # Final rank (1-based)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def combined_score(self) -> float:
        """Weighted combination of original and rerank scores."""
        return 0.3 * self.original_score + 0.7 * self.rerank_score


@dataclass
class RerankerResult:
    """Complete reranker output."""
    chunks: List[RerankedChunk]
    total_input: int
    total_output: int
    backend_used: str
    model_used: str
    processing_time_ms: float
    cache_hit: bool = False
    cache_stats: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Enhanced Cross-Encoder Reranker
# ============================================================================

class CrossEncoderReranker:
    """
    Cross-encoder based reranker using sentence-transformers.
    
    Uses a pre-trained cross-encoder model to score (query, chunk) pairs.
    Significantly more accurate than embedding similarity for reranking.
    """
    
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.model_name = config.model_name or config.default_models.get(
            "cross_encoder", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self._model = None
        
        # Enhanced LRU cache
        self._cache = LRUCache(
            max_size=config.max_cache_entries,
            ttl_seconds=config.cache_ttl_seconds
        )
    
    @property
    def model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    self.model_name,
                    max_length=self.config.cross_encoder_max_length
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for cross-encoder reranking. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int
    ) -> Tuple[List[RerankedChunk], float, bool]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Search query
            candidates: List of candidate dicts from hybrid search
            top_k: Number of top results to return
            
        Returns:
            Tuple of (reranked_chunks, processing_time_ms, cache_hit)
        """
        start_time = time.time()
        cache_hit = False
        
        if not candidates:
            return [], 0.0, False
        
        # Check cache
        cache_key = self._get_cache_key(query, candidates)
        if self.config.cache_enabled:
            cached_scores = self._cache.get(cache_key)
            if cached_scores is not None:
                cache_hit = True
                processing_time = (time.time() - start_time) * 1000
                results = self._build_results(candidates, cached_scores, top_k)
                return results, processing_time, True
        
        # Prepare pairs for cross-encoder
        pairs = [(query, c.get("content", "")[:self.config.cross_encoder_max_length]) 
                 for c in candidates]
        
        # Score all pairs
        scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.cross_encoder_show_progress
        )
        
        # Convert to list
        scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        
        # Cache results
        if self.config.cache_enabled:
            self._cache.put(cache_key, scores_list)
        
        processing_time = (time.time() - start_time) * 1000
        results = self._build_results(candidates, scores_list, top_k)
        return results, processing_time, False
    
    def _build_results(
        self,
        candidates: List[Dict],
        scores: List[float],
        top_k: int
    ) -> List[RerankedChunk]:
        """Build RerankedChunk objects from scores."""
        # Pair candidates with scores and sort
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Apply normalization
        if self.config.normalize_scores:
            raw_scores = [s for _, s in scored]
            normalized = self._normalize(raw_scores)
            scored = [(c, n) for (c, _), n in zip(scored, normalized)]
        
        # Build results
        results = []
        for rank, (candidate, score) in enumerate(scored[:top_k], 1):
            if score < self.config.min_score_threshold:
                continue
            
            results.append(RerankedChunk(
                chunk_id=candidate.get("chunk_id", ""),
                content=candidate.get("content", ""),
                raw_content=candidate.get("raw_content", candidate.get("content", "")),
                original_score=candidate.get("score", 0.0),
                rerank_score=float(score),
                rank=rank,
                metadata=candidate.get("metadata", {})
            ))
        
        return results
    
    def _normalize(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return scores
        
        arr = np.array(scores)
        
        if self.config.normalize_method == "softmax":
            # Softmax with temperature
            exp_arr = np.exp((arr - arr.max()) / self.config.temperature)
            return (exp_arr / exp_arr.sum()).tolist()
        else:
            # MinMax
            min_val, max_val = arr.min(), arr.max()
            if max_val == min_val:
                return [1.0] * len(scores)
            return ((arr - min_val) / (max_val - min_val)).tolist()
    
    def _get_cache_key(self, query: str, candidates: List[Dict]) -> str:
        """Generate cache key from query and candidate IDs."""
        candidate_ids = sorted([c.get("chunk_id", "") for c in candidates[:20]])
        key_str = f"{query}|{'|'.join(candidate_ids)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return self._cache.get_stats()


# ============================================================================
# Enhanced LLM Reranker with Structured Output
# ============================================================================

class LLMReranker:
    """
    LLM-based reranker using DeepSeek.

    Higher quality than cross-encoder but slower and has API cost.
    Best used for small candidate sets or when maximum accuracy is required.
    """

    def __init__(self, config: RerankerConfig):
        self.config = config
        self.model_name = config.model_name or config.default_models.get(
            "llm", "deepseek-chat"
        )

        # Cache for LLM responses
        self._cache = LRUCache(
            max_size=config.max_cache_entries // 2,  # Smaller cache for LLM
            ttl_seconds=config.cache_ttl_seconds
        )

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int
    ) -> Tuple[List[RerankedChunk], float, bool]:
        """
        Rerank candidates using LLM relevance scoring.
        
        Args:
            query: Search query
            candidates: List of candidate dicts
            top_k: Number of top results
            
        Returns:
            Tuple of (reranked_chunks, processing_time_ms, cache_hit)
        """
        start_time = time.time()
        cache_hit = False
        
        if not candidates:
            return [], 0.0, False
        
        # Limit candidates for LLM (cost control)
        candidates_to_score = candidates[:min(len(candidates), self.config.input_k)]
        
        # Check cache
        cache_key = self._get_cache_key(query, candidates_to_score)
        if self.config.cache_enabled:
            cached_scores = self._cache.get(cache_key)
            if cached_scores is not None:
                cache_hit = True
                processing_time = (time.time() - start_time) * 1000
                results = self._build_results(candidates_to_score, cached_scores, top_k)
                return results, processing_time, True
        
        # Build prompt with structured output guidance
        prompt = self._build_prompt(query, candidates_to_score)
        
        try:
            response_text = simple_generate(prompt, self.model_name)
            scores = self._parse_scores(response_text, len(candidates_to_score))
        except Exception as e:
            print(f"LLM reranker failed: {e}. Returning original order.")
            scores = [1.0 - (i * 0.01) for i in range(len(candidates_to_score))]
        
        # Normalize
        if self.config.normalize_scores and scores:
            arr = np.array(scores)
            exp_arr = np.exp((arr - arr.max()) / self.config.temperature)
            scores = (exp_arr / exp_arr.sum()).tolist()
        
        # Cache results
        if self.config.cache_enabled:
            self._cache.put(cache_key, scores)
        
        processing_time = (time.time() - start_time) * 1000
        results = self._build_results(candidates_to_score, scores, top_k)
        return results, processing_time, False
    
    def _build_prompt(self, query: str, candidates: List[Dict]) -> str:
        """Build the LLM reranking prompt with structured output guidance."""
        candidates_text = ""
        for i, c in enumerate(candidates):
            content = c.get("content", "")[:200]
            candidates_text += f"\n[{i}] {content}\n"
        
        return f"""Rate the relevance of each text passage to the query on a scale of 0.0 to 1.0.

Query: {query}

Passages:{candidates_text}

Return ONLY a valid JSON array of scores in the same order as the passages.
Example: [0.95, 0.3, 0.8, 0.1]
Do NOT include any explanation, markdown, or additional text.

IMPORTANT: The response must be a valid JSON array with no trailing commas."""
    
    def _parse_scores(self, response_text: str, expected_count: int) -> List[float]:
        """
        Parse LLM response into scores with enhanced robustness.
        
        Handles:
        - Markdown code blocks
        - Extra prose
        - Malformed JSON
        - Trailing commas
        - Invalid numbers
        """
        text = response_text.strip()
        
        # Remove markdown code blocks
        if text.startswith("```"):
            # Find the actual content
            code_block_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', text)
            if code_block_match:
                text = code_block_match.group(1).strip()
            else:
                # Fallback: remove outer backticks
                text = text.strip('`')
                if text.startswith("json"):
                    text = text[4:].strip()
        
        # Extract JSON array - try multiple patterns
        scores = None
        
        # Pattern 1: Complete JSON array
        json_match = re.search(r'\[[\s\S]*?\]', text)
        if json_match:
            try:
                # Clean up trailing commas before parsing
                json_str = json_match.group()
                json_str = re.sub(r',\s*\]', ']', json_str)  # Remove trailing commas
                scores = json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Pattern 2: Try to parse any JSON-like array with numbers
        if scores is None:
            number_pattern = re.findall(r'[-+]?\d*\.?\d+', text)
            if number_pattern:
                scores = [float(x) for x in number_pattern]
        
        # Pattern 3: Fallback to expected count
        if scores is None or not scores:
            return [0.5] * expected_count
        
        # Convert all to float and clamp to [0, 1]
        scores = [max(0.0, min(1.0, float(x))) for x in scores]
        
        # Ensure correct length (pad or truncate)
        if len(scores) < expected_count:
            scores.extend([0.0] * (expected_count - len(scores)))
        else:
            scores = scores[:expected_count]
        
        return scores
    
    def _build_results(
        self,
        candidates: List[Dict],
        scores: List[float],
        top_k: int
    ) -> List[RerankedChunk]:
        """Build RerankedChunk objects from scores."""
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (candidate, score) in enumerate(scored[:top_k], 1):
            if score < self.config.min_score_threshold:
                continue
            results.append(RerankedChunk(
                chunk_id=candidate.get("chunk_id", ""),
                content=candidate.get("content", ""),
                raw_content=candidate.get("raw_content", candidate.get("content", "")),
                original_score=candidate.get("score", 0.0),
                rerank_score=float(score),
                rank=rank,
                metadata=candidate.get("metadata", {})
            ))
        
        return results
    
    def _get_cache_key(self, query: str, candidates: List[Dict]) -> str:
        """Generate cache key from query and candidate IDs."""
        candidate_ids = sorted([c.get("chunk_id", "") for c in candidates[:15]])
        key_str = f"llm_{query}|{'|'.join(candidate_ids)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return self._cache.get_stats()


# ============================================================================
# Main Reranker with Parallel Hybrid Support
# ============================================================================

class Reranker:
    """
    Production-ready reranker with multiple backend support.
    
    Backends:
    - cross_encoder: Fast, local, free (default)
    - llm: Higher quality, API cost
    - hybrid: Weighted combination (with optional parallel execution)
    
    Configuration-driven — all settings from YAML with env var overrides.
    """
    
    def __init__(
        self,
        config: Optional[RerankerConfig] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the reranker.
        
        Args:
            config: RerankerConfig object
            config_path: Path to YAML config file
        """
        self.config = config or RerankerConfig.from_yaml(config_path)
        self._backend = None
        self._backend_name = None
        self._executor = None
    
    @property
    def backend(self):
        """Lazy-load the appropriate backend."""
        if self._backend is None:
            backend_type = self.config.backend_type
            
            if backend_type == "cross_encoder":
                self._backend = CrossEncoderReranker(self.config)
                self._backend_name = "cross_encoder"
            elif backend_type == "llm":
                self._backend = LLMReranker(self.config)
                self._backend_name = "llm"
            elif backend_type == "hybrid":
                self._backend = {
                    "cross_encoder": CrossEncoderReranker(self.config),
                    "llm": LLMReranker(self.config),
                }
                self._backend_name = "hybrid"
            else:
                raise ValueError(f"Unknown reranker backend: {backend_type}")
        
        return self._backend
    
    @property
    def executor(self):
        """Get or create thread pool executor for parallel hybrid execution."""
        if self._executor is None and self.config.parallel_hybrid:
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.max_parallel_workers
            )
        return self._executor
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> RerankerResult:
        """
        Rerank candidates and return top results.
        
        Args:
            query: Search query
            candidates: List of candidate dicts from hybrid search
            top_k: Number of results to return (defaults to config.output_k)
            
        Returns:
            RerankerResult with reranked chunks
        """
        if top_k is None:
            top_k = self.config.output_k
        
        # Limit input candidates
        candidates = candidates[:self.config.input_k]
        
        # Use config directly instead of _backend_name
        if self.config.backend_type == "hybrid":
            return self._hybrid_rerank(query, candidates, top_k)

        # Single backend reranking
        results, processing_time, cache_hit = self.backend.rerank(query, candidates, top_k)
        
        # Get cache stats if available
        cache_stats = {}
        if hasattr(self.backend, 'get_cache_stats'):
            cache_stats = self.backend.get_cache_stats()
        
        return RerankerResult(
            chunks=results,
            total_input=len(candidates),
            total_output=len(results),
            backend_used=self._backend_name or self.config.backend_type,
            model_used=getattr(self.backend, 'model_name', 'unknown'),
            processing_time_ms=round(processing_time, 2),
            cache_hit=cache_hit,
            cache_stats=cache_stats
        )
    
    def _hybrid_rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int
    ) -> RerankerResult:
        """
        Combine multiple reranker backends with weighted scores.
        
        If parallel_hybrid is True, executes rerankers concurrently.
        """
        weights = self.config.hybrid_weights or {"cross_encoder": 0.7, "llm": 0.3}
        start_time = time.time()
        cache_hit = False
        total_cache_hits = 0
        
        if self.config.parallel_hybrid and len(candidates) > 0:
            # Parallel execution
            return self._parallel_hybrid_rerank(query, candidates, top_k, weights, start_time)
        else:
            # Sequential execution
            return self._sequential_hybrid_rerank(query, candidates, top_k, weights, start_time)
    
    def _parallel_hybrid_rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int,
        weights: Dict[str, float],
        start_time: float
    ) -> RerankerResult:
        """Execute hybrid rerankers in parallel."""
        total_cache_hits = 0

        ce_reranker = self.backend["cross_encoder"]
        llm_reranker = self.backend["llm"]
        
        # Submit both reranking tasks to thread pool
        futures = []
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_workers) as executor:
            # Submit cross-encoder task
            ce_future = executor.submit(
                ce_reranker.rerank, query, candidates, len(candidates)
            )
            futures.append(("cross_encoder", ce_future))
            
            # Submit LLM task
            llm_future = executor.submit(
                llm_reranker.rerank, query, candidates, len(candidates)
            )
            futures.append(("llm", llm_future))
            
            # Collect results as they complete
            results_map = {}
            for name, future in futures:
                try:
                    results, processing_time, cache_hit_flag = future.result(timeout=120)
                    results_map[name] = {
                        "results": results,
                        "processing_time": processing_time,
                        "cache_hit": cache_hit_flag,
                        "scores": {r.chunk_id: r.rerank_score for r in results}
                    }
                    if cache_hit_flag:
                        total_cache_hits += 1
                except Exception as e:
                    print(f"Parallel reranker {name} failed: {e}")
                    results_map[name] = {
                        "results": [],
                        "processing_time": 0.0,
                        "cache_hit": False,
                        "scores": {}
                    }
        
        # Combine scores
        cache_hit = total_cache_hits > 0
        return self._combine_hybrid_results(
            query, candidates, top_k, weights, results_map, start_time, cache_hit
        )
    
    def _sequential_hybrid_rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int,
        weights: Dict[str, float],
        start_time: float
    ) -> RerankerResult:
        """Execute hybrid rerankers sequentially."""
        results_map = {}
        total_cache_hits = 0
        
        # Cross-encoder scores
        ce_results, ce_time, ce_cache = self.backend["cross_encoder"].rerank(
            query, candidates, len(candidates)
        )
        results_map["cross_encoder"] = {
            "results": ce_results,
            "processing_time": ce_time,
            "cache_hit": ce_cache,
            "scores": {r.chunk_id: r.rerank_score for r in ce_results}
        }
        if ce_cache:
            total_cache_hits += 1
        
        # LLM scores
        llm_results, llm_time, llm_cache = self.backend["llm"].rerank(
            query, candidates, len(candidates)
        )
        results_map["llm"] = {
            "results": llm_results,
            "processing_time": llm_time,
            "cache_hit": llm_cache,
            "scores": {r.chunk_id: r.rerank_score for r in llm_results}
        }
        if llm_cache:
            total_cache_hits += 1
        
        cache_hit = total_cache_hits > 0
        return self._combine_hybrid_results(
            query, candidates, top_k, weights, results_map, start_time, cache_hit
        )
    
    def _combine_hybrid_results(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int,
        weights: Dict[str, float],
        results_map: Dict[str, Any],
        start_time: float,
        cache_hit: bool
    ) -> RerankerResult:
        """Combine results from multiple rerankers."""
        # Combine scores
        combined = []
        ce_weight = weights.get("cross_encoder", 0.7)
        llm_weight = weights.get("llm", 0.3)
        
        for candidate in candidates:
            chunk_id = candidate.get("chunk_id", "")
            ce_score = results_map.get("cross_encoder", {}).get("scores", {}).get(chunk_id, 0.0)
            llm_score = results_map.get("llm", {}).get("scores", {}).get(chunk_id, 0.0)
            
            combined_score = (ce_weight * ce_score + llm_weight * llm_score)
            combined.append((candidate, combined_score))
        
        # Sort and build results
        combined.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (candidate, score) in enumerate(combined[:top_k], 1):
            if score < self.config.min_score_threshold:
                continue
            results.append(RerankedChunk(
                chunk_id=candidate.get("chunk_id", ""),
                content=candidate.get("content", ""),
                raw_content=candidate.get("raw_content", candidate.get("content", "")),
                original_score=candidate.get("score", 0.0),
                rerank_score=score,
                rank=rank,
                metadata=candidate.get("metadata", {})
            ))
        
        # Calculate total processing time
        total_time = sum(
            r.get("processing_time", 0.0) 
            for r in results_map.values()
        )
        if not self.config.parallel_hybrid:
            total_time = sum(
                r.get("processing_time", 0.0) 
                for r in results_map.values()
            )
        else:
            # In parallel mode, total time is max of the times, not sum
            total_time = max(
                r.get("processing_time", 0.0) 
                for r in results_map.values()
            )
        
        # Get cache stats from backends
        cache_stats = {}
        for name, backend in self.backend.items():
            if hasattr(backend, 'get_cache_stats'):
                cache_stats[name] = backend.get_cache_stats()
        
        return RerankerResult(
            chunks=results,
            total_input=len(candidates),
            total_output=len(results),
            backend_used="hybrid",
            model_used=f"ce+llm (parallel={self.config.parallel_hybrid})",
            processing_time_ms=round(total_time, 2),
            cache_hit=cache_hit,
            cache_stats=cache_stats        )
    
    def get_cache_stats(self) -> Dict[str, Dict[str, int]]:
        """Get cache statistics for all backends."""
        stats = {}
        
        if self._backend_name == "hybrid":
            for name, backend in self.backend.items():
                if hasattr(backend, 'get_cache_stats'):
                    stats[name] = backend.get_cache_stats()
        else:
            if hasattr(self.backend, 'get_cache_stats'):
                stats[self._backend_name] = self.backend.get_cache_stats()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        if self._backend_name == "hybrid":
            for backend in self.backend.values():
                if hasattr(backend, '_cache'):
                    backend._cache.clear()
        else:
            if hasattr(self.backend, '_cache'):
                self.backend._cache.clear()
    
    def close(self) -> None:
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None


# ============================================================================
# Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Reranker - Enhanced Validation")
    print("=" * 60)
    
    config = RerankerConfig.from_yaml()
    
    print(f"\n📋 Configuration loaded:")
    print(f"   Backend: {config.backend_type}")
    print(f"   Model: {config.model_name or config.default_models.get(config.backend_type, 'default')}")
    print(f"   Input K: {config.input_k}")
    print(f"   Output K: {config.output_k}")
    print(f"   Min score: {config.min_score_threshold}")
    print(f"   Normalize: {config.normalize_scores} ({config.normalize_method})")
    print(f"   Cache: {config.cache_enabled} (TTL: {config.cache_ttl_seconds}s, Max: {config.max_cache_entries})")
    print(f"   Parallel Hybrid: {config.parallel_hybrid}")
    
    # Check for sentence-transformers
    try:
        from sentence_transformers import CrossEncoder
        print(f"\n✅ sentence-transformers installed — cross-encoder available")
    except ImportError:
        print(f"\n⚠️  sentence-transformers not installed")
        print("   Install with: pip install sentence-transformers")
    
    # Create sample candidates
    sample_candidates = [
        {
            "chunk_id": "doc1_chunk_0001",
            "content": "Deadlock is a situation where two or more processes are unable to proceed because each is waiting for a resource held by another.",
            "score": 0.85,
            "metadata": {"topic": "Deadlock", "course_name": "Operating Systems"}
        },
        {
            "chunk_id": "doc1_chunk_0002",
            "content": "A deadlock occurs when four conditions hold simultaneously: mutual exclusion, hold and wait, no preemption, and circular wait.",
            "score": 0.78,
            "metadata": {"topic": "Deadlock", "course_name": "Operating Systems"}
        },
        {
            "chunk_id": "doc1_chunk_0003",
            "content": "Round Robin scheduling assigns each process a fixed time quantum in a circular order, ensuring fairness.",
            "score": 0.72,
            "metadata": {"topic": "CPU Scheduling", "course_name": "Operating Systems"}
        },
        {
            "chunk_id": "doc1_chunk_0004",
            "content": "Paging is a memory management scheme that eliminates the need for contiguous allocation of physical memory.",
            "score": 0.65,
            "metadata": {"topic": "Memory Management", "course_name": "Operating Systems"}
        },
        {
            "chunk_id": "doc1_chunk_0005",
            "content": "To prevent deadlock, one can use resource allocation graphs, banker's algorithm, or break one of the four necessary conditions.",
            "score": 0.60,
            "metadata": {"topic": "Deadlock", "course_name": "Operating Systems"}
        },
    ]
    
    print(f"\n{'─' * 50}")
    print(f"Sample candidates: {len(sample_candidates)}")
    print(f"Query: 'What is deadlock and how to prevent it?'")
    
    # Test with cross-encoder if available
    try:
        from sentence_transformers import CrossEncoder
        reranker = Reranker(config=config)
        
        # Test first run (cache miss)
        result = reranker.rerank(
            query="What is deadlock and how to prevent it?",
            candidates=sample_candidates
        )
        
        print(f"\n📊 First Run Results:")
        print(f"   Backend: {result.backend_used}")
        print(f"   Model: {result.model_used}")
        print(f"   Input: {result.total_input} candidates")
        print(f"   Output: {result.total_output} chunks")
        print(f"   Time: {result.processing_time_ms:.1f}ms")
        print(f"   Cache Hit: {result.cache_hit}")
        
        if result.cache_stats:
            print(f"   Cache Stats: {result.cache_stats}")
        
        print(f"\n📄 Reranked Chunks:")
        for chunk in result.chunks:
            print(f"\n   Rank {chunk.rank}: {chunk.chunk_id}")
            print(f"   Original: {chunk.original_score:.3f} → Rerank: {chunk.rerank_score:.3f}")
            print(f"   Content: {chunk.content[:100]}...")
            print(f"   Topic: {chunk.metadata.get('topic', 'N/A')}")
        
        # Test second run (cache hit)
        result2 = reranker.rerank(
            query="What is deadlock and how to prevent it?",
            candidates=sample_candidates
        )
        
        print(f"\n📊 Second Run (Cache Test):")
        print(f"   Time: {result2.processing_time_ms:.1f}ms")
        print(f"   Cache Hit: {result2.cache_hit}")
        
        # Test cache stats
        stats = reranker.get_cache_stats()
        if stats:
            print(f"   Cache Stats: {stats}")
        
        # Test hybrid if available
        if config.backend_type != "hybrid":
            print(f"\n{'─' * 50}")
            print("Testing Hybrid Mode (if LLM available)...")
            
            # Create hybrid config
            hybrid_config = RerankerConfig.from_yaml()
            hybrid_config.backend_type = "hybrid"
            hybrid_config.parallel_hybrid = True
            
            # Check if DeepSeek API key is set
            if os.getenv("DEEPSEEK_API_KEY"):
                hybrid_reranker = Reranker(config=hybrid_config)
                hybrid_result = hybrid_reranker.rerank(
                    query="What is deadlock and how to prevent it?",
                    candidates=sample_candidates
                )
                
                print(f"\n📊 Hybrid Results (Parallel={hybrid_config.parallel_hybrid}):")
                print(f"   Backend: {hybrid_result.backend_used}")
                print(f"   Model: {hybrid_result.model_used}")
                print(f"   Input: {hybrid_result.total_input} candidates")
                print(f"   Output: {hybrid_result.total_output} chunks")
                print(f"   Time: {hybrid_result.processing_time_ms:.1f}ms")
                print(f"   Cache Hit: {hybrid_result.cache_hit}")
                
                if hybrid_result.cache_stats:
                    print(f"   Cache Stats: {hybrid_result.cache_stats}")
            else:
                print("   ⚠️  DEEPSEEK_API_KEY not set — skipping hybrid test")
        
    except ImportError:
        print(f"\n⚠️  Skipping cross-encoder test (sentence-transformers not installed)")
        print("   LLM reranker is available as an alternative backend.")
    
    # Show environment variable overrides
    print(f"\n{'─' * 50}")
    print("Environment variable overrides available:")
    for var in ["RAGPIPE_RERANKER_BACKEND", "RAGPIPE_RERANKER_MODEL",
                "RAGPIPE_RERANKER_INPUT_K", "RAGPIPE_RERANKER_OUTPUT_K",
                "RAGPIPE_RERANKER_MIN_SCORE", "RAGPIPE_RERANKER_BATCH_SIZE",
                "RAGPIPE_RERANKER_PARALLEL"]:
        value = os.getenv(var)
        if value:
            print(f"   {var} = {value} ✅")
        else:
            print(f"   {var} = (not set)")
    
    print(f"\n✅ Enhanced Reranker ready for integration")