"""
Hybrid Search Engine
Combines semantic (dense vector), keyword (sparse BM25), and metadata search
into a unified candidate pool for reranking.

ARCHITECTURE NOTE:
- Semantic: Dense vector retrieval via Pinecone embeddings
- Keyword: Sparse retrieval using BM25 (production) or semantic fallback (development)
- Metadata: Filter-based retrieval using Pinecone metadata

Configuration-driven — all weights, thresholds, and strategies
loaded from config/retrieval/hybrid_search.yaml with environment
variable overrides.
"""

import os
import time
import yaml
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

from src.ingestion.vector_store import VectorStore
from src.retrieval.config import PipelineConfig, RetrievedChunk, HybridWeights

load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search — loaded from YAML."""
    weights: HybridWeights
    semantic_k: int = 40
    keyword_k: int = 20
    max_total_candidates: int = 60
    dedup_threshold: float = 0.85
    dedup_strategy: str = "score"
    merge_strategy: str = "weighted"
    rrf_k: int = 60
    normalize_method: str = "minmax"
    metadata_enabled: bool = True
    metadata_fields: List[str] = field(default_factory=list)
    metadata_match_threshold: float = 0.3
    
    # Keyword search settings
    keyword_backend: str = "bm25"  # "bm25", "elasticsearch", or "semantic_fallback"
    keyword_use_sparse: bool = True
    bm25_index_path: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "HybridSearchConfig":
        """
        Load configuration from YAML file.
        
        Priority:
        1. Explicit path argument
        2. RAGPIPE_HYBRID_SEARCH_CONFIG env var
        3. config/retrieval/hybrid_search.yaml (default)
        """
        config_path = (
            path
            or os.getenv("RAGPIPE_HYBRID_SEARCH_CONFIG")
            or str(Path(__file__).parent.parent.parent / "config" / "retrieval" / "hybrid_search.yaml")
        )
        
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
        else:
            data = cls._default_config()
        
        # Apply environment variable overrides
        data = cls._apply_env_overrides(data)
        
        return cls(
            weights=HybridWeights(**data.get("weights", {})),
            semantic_k=int(data.get("candidates", {}).get("semantic_k", 40)),
            keyword_k=int(data.get("candidates", {}).get("keyword_k", 20)),
            max_total_candidates=int(data.get("candidates", {}).get("max_total", 60)),
            dedup_threshold=float(data.get("dedup", {}).get("threshold", 0.85)),
            dedup_strategy=str(data.get("dedup", {}).get("strategy", "score")),
            merge_strategy=str(data.get("merge", {}).get("strategy", "weighted")),
            rrf_k=int(data.get("merge", {}).get("rrf_k", 60)),
            normalize_method=str(data.get("normalize", {}).get("method", "minmax")),
            metadata_enabled=bool(data.get("metadata", {}).get("enabled", True)),
            metadata_fields=list(data.get("metadata", {}).get("fields", [])),
            metadata_match_threshold=float(data.get("metadata", {}).get("match_threshold", 0.3)),
            keyword_backend=str(data.get("keyword", {}).get("backend", "bm25")),
            keyword_use_sparse=bool(data.get("keyword", {}).get("use_sparse", True)),
            bm25_index_path=data.get("keyword", {}).get("bm25_index_path"),
        )
    
    @classmethod
    def _apply_env_overrides(cls, data: Dict) -> Dict:
        """Apply environment variable overrides to config data."""
        env_mapping = {
            "RAGPIPE_SEMANTIC_K": ("candidates", "semantic_k"),
            "RAGPIPE_KEYWORD_K": ("candidates", "keyword_k"),
            "RAGPIPE_MAX_CANDIDATES": ("candidates", "max_total"),
            "RAGPIPE_DEDUP_THRESHOLD": ("dedup", "threshold"),
            "RAGPIPE_MERGE_STRATEGY": ("merge", "strategy"),
            "RAGPIPE_SEMANTIC_WEIGHT": ("weights", "semantic"),
            "RAGPIPE_KEYWORD_WEIGHT": ("weights", "keyword"),
            "RAGPIPE_METADATA_WEIGHT": ("weights", "metadata"),
            "RAGPIPE_KEYWORD_BACKEND": ("keyword", "backend"),
            "RAGPIPE_KEYWORD_USE_SPARSE": ("keyword", "use_sparse"),
        }
        
        for env_var, (section, key) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in data:
                    data[section] = {}
                try:
                    # Handle boolean values
                    if value.lower() in ['true', 'false']:
                        data[section][key] = value.lower() == 'true'
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
            "weights": {"semantic": 0.6, "keyword": 0.25, "metadata": 0.15},
            "candidates": {"semantic_k": 40, "keyword_k": 20, "max_total": 60},
            "dedup": {"threshold": 0.85, "strategy": "score"},
            "merge": {"strategy": "weighted", "rrf_k": 60},
            "normalize": {"method": "minmax"},
            "metadata": {
                "enabled": True,
                "fields": ["topic", "course_name", "chapter_title"],
                "match_threshold": 0.3,
            },
            "keyword": {
                "backend": "bm25",  # bm25, elasticsearch, semantic_fallback
                "use_sparse": True,
                "bm25_index_path": None,
            },
        }


# ============================================================================
# Search Result Types
# ============================================================================

@dataclass
class SearchCandidate:
    """A single candidate from any search method."""
    chunk_id: str
    content: str
    raw_content: str
    score: float                     # Normalized score [0, 1]
    source: str                      # "semantic", "keyword", or "metadata"
    original_rank: int               # Rank in its source list
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridSearchResult:
    """Complete hybrid search result."""
    candidates: List[SearchCandidate]
    total_retrieved: int
    sources_used: List[str]
    processing_time_ms: float
    query: str
    keyword_backend_used: str = "unknown"


# ============================================================================
# True Keyword Search with BM25
# ============================================================================

class BM25Index:
    """
    Lightweight BM25 index for sparse keyword retrieval.
    
    PRODUCTION NOTE: For large-scale deployments, replace with:
    - Elasticsearch/OpenSearch
    - Pinecone sparse-dense index
    - LanceDB with BM25
    - Tantivy or similar dedicated search engine
    
    This implementation uses rank_bm25 for demonstration and testing.
    """
    
    def __init__(self, index_path: Optional[str] = None):
        self.index_path = index_path
        self.index = None
        self.documents = []
        self.tokenizer = lambda x: self._tokenize(x)
        self._initialized = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25."""
        # Convert to lowercase and split on non-alphanumeric
        import re
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build BM25 index from a list of documents.
        
        Args:
            documents: List of dicts with 'id', 'content', and optional 'metadata'
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank_bm25 is required for BM25 indexing. "
                "Install with: pip install rank_bm25"
            )
        
        self.documents = documents
        tokenized_docs = [self.tokenizer(doc['content']) for doc in documents]
        self.index = BM25Okapi(tokenized_docs)
        self._initialized = True
        
        # Save index if path provided
        if self.index_path:
            self._save_index()
    
    def _save_index(self) -> None:
        """Save index to disk for persistence."""
        if not self.index_path:
            return
        
        # Save document metadata
        meta_path = Path(self.index_path).with_suffix('.meta.json')
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(meta_path, 'w') as f:
            json.dump({
                'documents': self.documents,
                'num_docs': len(self.documents)
            }, f)
    
    def load_index(self) -> bool:
        """Load index from disk."""
        if not self.index_path:
            return False
        
        meta_path = Path(self.index_path).with_suffix('.meta.json')
        if not meta_path.exists():
            return False
        
        try:
            with open(meta_path, 'r') as f:
                data = json.load(f)
                self.documents = data['documents']
                # Rebuild index from documents
                self.build_index(self.documents)
                return True
        except Exception as e:
            print(f"Failed to load BM25 index: {e}")
            return False
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Search the BM25 index.

        Args:
            query: Query text
            top_k: Number of results to return. If None, returns every
                non-zero-scored document ranked best-first — used by callers
                that need to post-filter (e.g. by namespace) before truncating.

        Returns:
            List of (document_index, score) tuples
        """
        if not self._initialized:
            raise RuntimeError("BM25 index not initialized. Call build_index() or load_index() first.")

        tokenized_query = self.tokenizer(query)
        scores = self.index.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]

        return [(i, scores[i]) for i in ranked_indices if scores[i] > 0]


class KeywordSearch:
    """
    Keyword-based search with BM25 sparse retrieval.
    
    ARCHITECTURE:
    Primary: BM25 sparse retrieval for true keyword matching
    Fallback: Semantic search (when BM25 unavailable)
    
    PRODUCTION NOTE: 
    - This implementation uses rank_bm25 for demonstration
    - For production at scale, replace with Elasticsearch or similar
    - The fallback behavior ensures graceful degradation
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        backend: str = "bm25",
        use_sparse: bool = True,
        bm25_index_path: Optional[str] = None,
        embedder: Optional[Any] = None,
    ):
        self.vector_store = vector_store
        self.backend = backend
        self.use_sparse = use_sparse
        self.bm25_index = BM25Index(bm25_index_path)
        self._bm25_available = False
        self._embedder = embedder
    
    def initialize_bm25(self, documents: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Initialize the BM25 index.
        
        Args:
            documents: Documents to index. If None, attempts to load from disk.
        """
        if documents:
            self.bm25_index.build_index(documents)
            self._bm25_available = True
        elif self.bm25_index.index_path:
            self._bm25_available = self.bm25_index.load_index()
        else:
            print("WARNING: No BM25 index provided. Keyword search will fall back to semantic search.")
            self._bm25_available = False
    
    def search(
        self,
        query_keywords: str,
        top_k: int,
        namespace: Union[str, List[str]],
        filter: Optional[Dict] = None,
        precomputed_vector: Optional[List[float]] = None,
    ) -> List[SearchCandidate]:
        """
        Keyword-based search with BM25 sparse retrieval.

        ⚠️  BEHAVIOR NOTE:
        If BM25 is unavailable, falls back to semantic search.
        This is a graceful degradation path, NOT the intended production behavior.

        Args:
            query_keywords: Keywords for sparse search
            top_k: Number of results to return
            namespace: Pinecone namespace
            filter: Optional metadata filter
            precomputed_vector: Pre-embedded query vector to reuse for the
                semantic fallback path, avoiding a redundant embedding call

        Returns:
            List of SearchCandidate objects
        """
        # Primary: BM25 sparse search
        if self.use_sparse and self._bm25_available:
            results = self._bm25_search(query_keywords, top_k, namespace, filter)
            if results:
                return results

        # Secondary: Try Elasticsearch backend if configured
        if self.backend == "elasticsearch":
            results = self._elasticsearch_search(query_keywords, top_k, namespace, filter)
            if results:
                return results

        # Fallback: Semantic search (development/placeholder behavior)
        print(f"WARNING: Keyword search using semantic fallback for query: {query_keywords}")
        return self._fallback_semantic_search(query_keywords, top_k, namespace, filter, precomputed_vector)
    
    def _bm25_search(
        self,
        query_keywords: str,
        top_k: int,
        namespace: Union[str, List[str]],
        filter: Optional[Dict] = None
    ) -> List[SearchCandidate]:
        """
        True BM25 sparse retrieval, scoped to the requested namespace(s) and
        metadata filter — mirrors Pinecone's per-namespace query semantics so
        one global BM25 corpus never leaks results across Exams/documents.
        """
        try:
            allowed_namespaces: Optional[set] = None
            if namespace:
                allowed_namespaces = set(namespace) if isinstance(namespace, list) else {namespace}

            # When post-filtering we can't rely on the index's own top_k cutoff
            # (the best-scored docs overall may not be in-namespace), so pull
            # every non-zero match and truncate ourselves after filtering.
            needs_post_filter = bool(allowed_namespaces) or bool(filter)
            results = self.bm25_index.search(query_keywords, top_k=None if needs_post_filter else top_k)

            if not results:
                return []

            candidates = []
            for doc_idx, score in results:
                doc = self.bm25_index.documents[doc_idx]
                doc_metadata = doc.get('metadata', {})

                if allowed_namespaces is not None and doc_metadata.get('namespace') not in allowed_namespaces:
                    continue
                if filter and not self._matches_filter(doc_metadata, filter):
                    continue

                candidates.append(SearchCandidate(
                    chunk_id=doc['id'],
                    content=doc.get('content', ''),
                    raw_content=doc.get('raw_content', doc.get('content', '')),
                    score=float(score),
                    source="keyword_bm25",
                    original_rank=doc_idx + 1,
                    metadata=doc_metadata
                ))
                if len(candidates) >= top_k:
                    break

            return candidates

        except Exception as e:
            print(f"BM25 search failed: {e}")
            return []

    @staticmethod
    def _matches_filter(metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """
        Evaluate a Pinecone-style metadata filter against a BM25 document's
        metadata. Supports the operators actually produced by
        `MetadataFilterBuilder` ($eq, $ne, $in, $nin); a bare value is
        treated as shorthand for $eq.
        """
        for field, condition in filter.items():
            actual = metadata.get(field)
            if isinstance(condition, dict):
                for op, expected in condition.items():
                    if op == "$eq" and actual != expected:
                        return False
                    elif op == "$ne" and actual == expected:
                        return False
                    elif op == "$in" and actual not in expected:
                        return False
                    elif op == "$nin" and actual in expected:
                        return False
            else:
                if actual != condition:
                    return False
        return True
    
    def _elasticsearch_search(
        self,
        query_keywords: str,
        top_k: int,
        namespace: str,
        filter: Optional[Dict] = None
    ) -> List[SearchCandidate]:
        """
        Elasticsearch-backed keyword search.
        
        TODO: Implement when Elasticsearch integration is added.
        """
        # Placeholder for Elasticsearch integration
        print("Elasticsearch backend not yet implemented")
        return []
    
    def _fallback_semantic_search(
        self,
        query_keywords: str,
        top_k: int,
        namespace: str,
        filter: Optional[Dict] = None,
        precomputed_vector: Optional[List[float]] = None,
    ) -> List[SearchCandidate]:
        """
        Fallback to semantic search when BM25 is unavailable.

        ⚠️  DEVELOPMENT ONLY: This is not true keyword search.
        """
        embedded = precomputed_vector
        if embedded is None:
            if self._embedder is None:
                from src.ingestion.embedder import EmbeddingGenerator
                self._embedder = EmbeddingGenerator()
            embedded = self._embedder.embed_query(query_keywords)

        if not embedded:
            return []
        
        results = self.vector_store.query(
            vector=embedded,
            top_k=top_k,
            namespace=namespace,
            filter=filter
        )
        
        candidates = []
        for i, result in enumerate(results):
            candidates.append(SearchCandidate(
                chunk_id=result["id"],
                content=result.get("metadata", {}).get("content", ""),
                raw_content=result.get("metadata", {}).get("raw_content", ""),
                score=result["score"],
                source="keyword_fallback",  # Differentiates from true keyword search
                original_rank=i + 1,
                metadata=result.get("metadata", {})
            ))
        
        return candidates


# ============================================================================
# Hybrid Search Engine
# ============================================================================

class HybridSearch:
    """
    Production-ready hybrid search combining semantic, keyword, and metadata search.
    
    ARCHITECTURE OVERVIEW:
    ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐
    │   Semantic      │  │    Keyword       │  │    Metadata     │
    │   (Dense)       │  │    (Sparse)      │  │    (Filter)     │
    ├─────────────────┤  ├──────────────────┤  ├─────────────────┤
    │ Pinecone Embed  │  │ BM25/Elasticsearch│  │ Pinecone Meta   │
    │ Vector Search   │  │ Sparse Retrieval │  │ Filter + Boost  │
    └─────────────────┘  └──────────────────┘  └─────────────────┘
           │                      │                      │
           └──────────────────────┼──────────────────────┘
                                  ▼
                      ┌─────────────────────┐
                      │   Weighted Merge    │
                      │   + Deduplication   │
                      └─────────────────────┘
                                  │
                                  ▼
                      ┌─────────────────────┐
                      │   Unified Results   │
                      └─────────────────────┘
    
    All configuration loaded from YAML with environment variable overrides.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        config: Optional[HybridSearchConfig] = None,
        config_path: Optional[str] = None,
        initialize_bm25: bool = True
    ):
        """
        Initialize hybrid search engine.
        
        Args:
            vector_store: VectorStore instance for Pinecone queries
            config: HybridSearchConfig object (overrides config file)
            config_path: Path to YAML config (overrides env var and default)
            initialize_bm25: Whether to initialize BM25 index on startup
        """
        self.vector_store = vector_store

        # Load configuration
        self.config = config or HybridSearchConfig.from_yaml(config_path)

        # Single shared embedder reused across semantic/keyword-fallback/metadata
        # search so a query is embedded once per API call instead of once per source.
        from src.ingestion.embedder import EmbeddingGenerator
        self._embedder = EmbeddingGenerator()

        # Initialize keyword search with BM25
        self.keyword_search = KeywordSearch(
            vector_store=vector_store,
            backend=self.config.keyword_backend,
            use_sparse=self.config.keyword_use_sparse,
            bm25_index_path=self.config.bm25_index_path,
            embedder=self._embedder,
        )

        # Track which backend is actually being used
        self._keyword_backend_used = "uninitialized"
    
    def initialize_indices(self, documents: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Initialize all search indices.
        
        Args:
            documents: Documents for BM25 indexing. If None, attempts to load from disk.
        """
        self.keyword_search.initialize_bm25(documents)
        self._keyword_backend_used = "bm25" if self.keyword_search._bm25_available else "semantic_fallback"
        
        if self._keyword_backend_used == "semantic_fallback":
            print("⚠️  WARNING: Keyword search using semantic fallback!")
            print("   For production, ensure BM25 index is properly initialized.")
    
    def search(
        self,
        query: str,
        keywords: str,
        namespace: Union[str, List[str]] = "default",
        metadata_filters: Optional[Dict[str, Any]] = None,
        pipeline_weights: Optional[HybridWeights] = None
    ) -> HybridSearchResult:
        """
        Execute hybrid search across all configured sources.
        
        Args:
            query: Full expanded query for semantic search
            keywords: Keyword-only variant for sparse search
            namespace: Pinecone namespace
            metadata_filters: Pinecone metadata filter dict
            pipeline_weights: Optional per-pipeline weight override
            
        Returns:
            HybridSearchResult with deduplicated, scored candidates
        """
        start_time = time.time()
        weights = pipeline_weights or self.config.weights

        candidates: List[SearchCandidate] = []
        sources_used = []

        # ------------------------------------------------------------------
        # 0. Batch-embed every query variant needed below in ONE API call.
        # Previously each of semantic / keyword-fallback / per-metadata-field
        # search embedded independently (up to 5 sequential round trips per
        # query, ~800-900ms each), which alone could exceed the retrieval
        # timeout. Embedding once and reusing the vectors collapses that to
        # a single round trip.
        # ------------------------------------------------------------------
        needs_semantic = weights.semantic > 0
        needs_keyword_fallback = (
            weights.keyword > 0 and bool(keywords)
            and not (self.keyword_search.use_sparse and self.keyword_search._bm25_available)
            and self.keyword_search.backend != "elasticsearch"
        )
        needs_metadata = weights.metadata > 0 and self.config.metadata_enabled
        metadata_field_queries = (
            [f"{query} {field}" for field in self.config.metadata_fields]
            if needs_metadata else []
        )

        embed_texts: List[str] = []
        if needs_semantic:
            embed_texts.append(query)
        if needs_keyword_fallback:
            embed_texts.append(keywords)
        embed_texts.extend(metadata_field_queries)

        vectors = self._embedder.embed_queries(embed_texts) if embed_texts else []

        offset = 0
        semantic_vector = None
        if needs_semantic:
            semantic_vector = vectors[offset]
            offset += 1
        keyword_vector = None
        if needs_keyword_fallback:
            keyword_vector = vectors[offset]
            offset += 1
        metadata_vectors = vectors[offset:offset + len(metadata_field_queries)]

        # ------------------------------------------------------------------
        # 1. Semantic Search (Pinecone Dense)
        # ------------------------------------------------------------------
        if needs_semantic:
            semantic_results = self._semantic_search(
                query=query,
                top_k=self.config.semantic_k,
                namespace=namespace,
                filter_dict=metadata_filters,
                precomputed_vector=semantic_vector
            )
            candidates.extend(semantic_results)
            sources_used.append("semantic")

        # ------------------------------------------------------------------
        # 2. Keyword Search (BM25 Sparse)
        # ------------------------------------------------------------------
        if weights.keyword > 0 and keywords:
            keyword_results = self.keyword_search.search(
                query_keywords=keywords,
                top_k=self.config.keyword_k,
                namespace=namespace,
                filter=metadata_filters,
                precomputed_vector=keyword_vector
            )
            # Normalize keyword scores to [0, 1] range
            keyword_results = self._normalize_scores(keyword_results)
            candidates.extend(keyword_results)

            # Track backend used
            if keyword_results:
                source_name = keyword_results[0].source
                if source_name == "keyword_bm25":
                    sources_used.append("keyword_bm25")
                elif source_name == "keyword_fallback":
                    sources_used.append("keyword_fallback")
                    print("⚠️  Using semantic fallback for keyword search - results may overlap with semantic branch")
                else:
                    sources_used.append("keyword")

        # ------------------------------------------------------------------
        # 3. Metadata Search
        # ------------------------------------------------------------------
        if needs_metadata:
            metadata_results = self._metadata_search(
                query=query,
                top_k=self.config.semantic_k // 2,
                namespace=namespace,
                base_filter=metadata_filters,
                precomputed_vectors=metadata_vectors
            )
            candidates.extend(metadata_results)
            sources_used.append("metadata")
        
        # ------------------------------------------------------------------
        # 4. Merge & Weight Scores
        # ------------------------------------------------------------------
        merged = self._merge_candidates(candidates, weights)
        
        # ------------------------------------------------------------------
        # 5. Deduplicate
        # ------------------------------------------------------------------
        deduped = self._deduplicate(merged)
        
        # ------------------------------------------------------------------
        # 6. Limit to max candidates
        # ------------------------------------------------------------------
        final = deduped[:self.config.max_total_candidates]
        
        processing_time = (time.time() - start_time) * 1000
        
        return HybridSearchResult(
            candidates=final,
            total_retrieved=len(final),
            sources_used=sources_used,
            processing_time_ms=round(processing_time, 2),
            query=query,
            keyword_backend_used=self._keyword_backend_used
        )
    
    # ========================================================================
    # Search Methods
    # ========================================================================
    
    def _semantic_search(
        self,
        query: str,
        top_k: int,
        namespace: str,
        filter_dict: Optional[Dict] = None,
        precomputed_vector: Optional[List[float]] = None
    ) -> List[SearchCandidate]:
        """Execute semantic search via Pinecone dense retrieval."""
        embedded = precomputed_vector
        if embedded is None:
            embedded = self._embedder.embed_query(query)

        if not embedded:
            return []
        
        # Query Pinecone
        results = self.vector_store.query(
            vector=embedded,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict
        )
        
        candidates = []
        for i, result in enumerate(results):
            candidates.append(SearchCandidate(
                chunk_id=result["id"],
                content=result.get("metadata", {}).get("content", query),
                raw_content=result.get("metadata", {}).get("raw_content", query),
                score=result["score"],
                source="semantic",
                original_rank=i + 1,
                metadata=result.get("metadata", {})
            ))
        
        return candidates
    
    def _metadata_search(
        self,
        query: str,
        top_k: int,
        namespace: str,
        base_filter: Optional[Dict] = None,
        precomputed_vectors: Optional[List[Optional[List[float]]]] = None
    ) -> List[SearchCandidate]:
        """
        Search with metadata field matching.
        Builds additional filters based on configured metadata fields.
        """
        candidates = []
        vectors = precomputed_vectors or [None] * len(self.config.metadata_fields)

        for field, vector in zip(self.config.metadata_fields, vectors):
            # Use semantic search with metadata field as additional context
            filter_dict = dict(base_filter) if base_filter else {}
            # Don't filter too aggressively — just boost by including field in query
            field_query = f"{query} {field}"

            field_results = self._semantic_search(
                query=field_query,
                top_k=max(5, top_k // len(self.config.metadata_fields)),
                namespace=namespace,
                filter_dict=filter_dict if filter_dict else None,
                precomputed_vector=vector
            )

            for c in field_results:
                c.source = "metadata"
                c.score *= self.config.metadata_match_threshold

            candidates.extend(field_results)

        return candidates
    
    # ========================================================================
    # Merging & Scoring
    # ========================================================================
    
    def _merge_candidates(
        self,
        candidates: List[SearchCandidate],
        weights: HybridWeights
    ) -> List[SearchCandidate]:
        """Merge candidates from different sources with weighted scoring."""
        
        if self.config.merge_strategy == "rrf":
            return self._reciprocal_rank_fusion(candidates, weights)
        else:
            return self._weighted_merge(candidates, weights)
    
    def _weighted_merge(
        self,
        candidates: List[SearchCandidate],
        weights: HybridWeights
    ) -> List[SearchCandidate]:
        """Merge by applying source-specific weights to scores."""
        source_weights = {
            "semantic": weights.semantic,
            "keyword": weights.keyword,
            "keyword_bm25": weights.keyword,
            "keyword_fallback": weights.keyword * 0.5,  # Penalize fallback
            "metadata": weights.metadata,
        }
        
        for candidate in candidates:
            w = source_weights.get(candidate.source, 0.0)
            candidate.score *= w
        
        # Sort by weighted score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates
    
    def _reciprocal_rank_fusion(
        self,
        candidates: List[SearchCandidate],
        weights: HybridWeights
    ) -> List[SearchCandidate]:
        """Merge using Reciprocal Rank Fusion."""
        # Group by source and rank within each source
        by_source: Dict[str, List[SearchCandidate]] = {}
        for c in candidates:
            by_source.setdefault(c.source, []).append(c)
        
        # Sort each source list by score
        for source in by_source:
            by_source[source].sort(key=lambda c: c.score, reverse=True)
        
        # Apply RRF formula: score = 1 / (k + rank)
        source_weights = {
            "semantic": weights.semantic,
            "keyword": weights.keyword,
            "keyword_bm25": weights.keyword,
            "keyword_fallback": weights.keyword * 0.5,
            "metadata": weights.metadata,
        }
        
        for source, source_candidates in by_source.items():
            w = source_weights.get(source, 0.0)
            for rank, candidate in enumerate(source_candidates):
                candidate.score = w / (self.config.rrf_k + rank + 1)
        
        # Sort all by new score
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates
    
    # ========================================================================
    # Deduplication
    # ========================================================================
    
    def _deduplicate(self, candidates: List[SearchCandidate]) -> List[SearchCandidate]:
        """Remove near-duplicate chunks.

        Near-duplicates are chunks that were split from the same source
        chunk (tracked via metadata["parent_chunk_id"], set by the chunker
        only when a chunk was split — empty otherwise). Chunk IDs themselves
        (e.g. "ch 9_chunk_0003") can't be used for this via string-splitting:
        every chunk in a file shares the same prefix once the trailing index
        is stripped, which would collapse the entire file into one chunk.
        """
        if not candidates:
            return []

        unique = []
        seen_ids = set()

        for candidate in candidates:
            if candidate.chunk_id in seen_ids:
                continue

            parent_id = candidate.metadata.get("parent_chunk_id") or None

            is_duplicate = False
            if parent_id:
                for existing in unique:
                    existing_parent_id = existing.metadata.get("parent_chunk_id") or None
                    if parent_id == existing_parent_id:
                        # Split from the same source chunk — keep the higher-scored one
                        if self.config.dedup_strategy == "score":
                            if candidate.score > existing.score:
                                unique.remove(existing)
                                unique.append(candidate)
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique.append(candidate)
                seen_ids.add(candidate.chunk_id)

        # Re-sort after dedup
        unique.sort(key=lambda c: c.score, reverse=True)
        return unique
    
    # ========================================================================
    # Score Normalization
    # ========================================================================
    
    def _normalize_scores(self, candidates: List[SearchCandidate]) -> List[SearchCandidate]:
        """Normalize scores to [0, 1] range."""
        if not candidates:
            return candidates
        
        scores = [c.score for c in candidates]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            for c in candidates:
                c.score = 1.0
            return candidates
        
        if self.config.normalize_method == "minmax":
            for c in candidates:
                c.score = (c.score - min_score) / (max_score - min_score)
        elif self.config.normalize_method == "zscore":
            mean = sum(scores) / len(scores)
            std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
            if std > 0:
                for c in candidates:
                    c.score = max(0.0, min(1.0, (c.score - mean) / (2 * std) + 0.5))
        
        return candidates
    
    # ========================================================================
    # Diagnostics
    # ========================================================================
    
    def get_retrieval_status(self) -> Dict[str, Any]:
        """Get status of all retrieval backends."""
        return {
            "semantic": {"status": "available", "backend": "pinecone"},
            "keyword": {
                "status": "available" if self.keyword_search._bm25_available else "fallback",
                "backend": self.keyword_search.backend,
                "uses_sparse": self.keyword_search.use_sparse,
                "bm25_available": self.keyword_search._bm25_available,
            },
            "metadata": {"status": "available", "fields": self.config.metadata_fields},
            "keyword_backend_used": self._keyword_backend_used,
        }


# ============================================================================
# Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Hybrid Search - Validation")
    print("=" * 60)
    
    # Load configuration
    config = HybridSearchConfig.from_yaml()
    print(f"\n📋 Configuration loaded:")
    print(f"   Weights: semantic={config.weights.semantic}, keyword={config.weights.keyword}, metadata={config.weights.metadata}")
    print(f"   Candidates: semantic_k={config.semantic_k}, keyword_k={config.keyword_k}, max={config.max_total_candidates}")
    print(f"   Merge: {config.merge_strategy}")
    print(f"   Dedup: threshold={config.dedup_threshold}, strategy={config.dedup_strategy}")
    print(f"   Metadata fields: {config.metadata_fields}")
    print(f"   Keyword backend: {config.keyword_backend}")
    print(f"   Keyword use_sparse: {config.keyword_use_sparse}")
    
    # Validate environment variable overrides
    print(f"\n🔧 Environment variable overrides available:")
    for var in ["RAGPIPE_SEMANTIC_K", "RAGPIPE_KEYWORD_K", "RAGPIPE_MAX_CANDIDATES",
                "RAGPIPE_DEDUP_THRESHOLD", "RAGPIPE_MERGE_STRATEGY",
                "RAGPIPE_SEMANTIC_WEIGHT", "RAGPIPE_KEYWORD_WEIGHT", "RAGPIPE_METADATA_WEIGHT",
                "RAGPIPE_KEYWORD_BACKEND", "RAGPIPE_KEYWORD_USE_SPARSE"]:
        value = os.getenv(var)
        if value:
            print(f"   {var} = {value} ✅")
        else:
            print(f"   {var} = (not set, using config file default)")
    
    # Check for BM25 dependency
    try:
        import rank_bm25
        print(f"\n✅ rank_bm25 installed - BM25 indexing available")
    except ImportError:
        print(f"\n⚠️  rank_bm25 not installed - install with: pip install rank_bm25")
        print("   Keyword search will fall back to semantic search.")
    
    print(f"\n📊 Hybrid Search Architecture:")
    print(f"   ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐")
    print(f"   │   Semantic      │  │    Keyword       │  │    Metadata     │")
    print(f"   │   (Dense)       │  │    (Sparse)      │  │    (Filter)     │")
    print(f"   ├─────────────────┤  ├──────────────────┤  ├─────────────────┤")
    print(f"   │ Pinecone Embed  │  │ BM25/{config.keyword_backend} │  │ Pinecone Meta   │")
    print(f"   │ Vector Search   │  │ Sparse Retrieval │  │ Filter + Boost  │")
    print(f"   └─────────────────┘  └──────────────────┘  └─────────────────┘")
    print(f"           │                      │                      │")
    print(f"           └──────────────────────┼──────────────────────┘")
    print(f"                                  ▼")
    print(f"                      ┌─────────────────────┐")
    print(f"                      │   Weighted Merge    │")
    print(f"                      │   + Deduplication   │")
    print(f"                      └─────────────────────┘")
    
    print(f"\n✅ Hybrid Search ready (requires VectorStore + Pinecone for full test)")
    print(f"   Full integration test available after orchestrator is built.")
    
    print(f"\n⚠️  IMPORTANT: Ensure BM25 index is initialized for production keyword search.")
    print(f"   Without BM25, keyword search falls back to semantic embeddings.")