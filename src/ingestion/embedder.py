"""
Step 6: Vector Embedding
Converts enriched chunks into vector embeddings using configurable providers.
Supports Google, OpenAI, and extensible for other providers.
"""

import os
import time
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque

from dotenv import load_dotenv

from src.ingestion.enricher import EnrichedChunk, ChunkEnricher

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Provider Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for an embedding model."""
    max_tokens: int
    default_dimensions: int
    provider: str
    pricing_per_1m_tokens: float = 0.0
    supports_custom_dimensions: bool = False

# Model configurations
MODEL_CONFIGS = {
    # Google models
    "gemini-embedding-001": ModelConfig(
        max_tokens=2048,
        default_dimensions=768,
        provider="google",
        pricing_per_1m_tokens=0.0,  # Free during preview
        supports_custom_dimensions=True,
    ),
    # OpenAI models
    "text-embedding-3-small": ModelConfig(
        max_tokens=8191,
        default_dimensions=1536,
        provider="openai",
        pricing_per_1m_tokens=0.02,
        supports_custom_dimensions=True,
    ),
    "text-embedding-3-large": ModelConfig(
        max_tokens=8191,
        default_dimensions=3072,
        provider="openai",
        pricing_per_1m_tokens=0.13,
        supports_custom_dimensions=True,
    ),
    "text-embedding-ada-002": ModelConfig(
        max_tokens=8191,
        default_dimensions=1536,
        provider="openai",
        pricing_per_1m_tokens=0.0001,
        supports_custom_dimensions=False,
    ),
    # Placeholder for future models
    "bge-large-en-v1.5": ModelConfig(
        max_tokens=512,
        default_dimensions=1024,
        provider="local",
        pricing_per_1m_tokens=0.0,
        supports_custom_dimensions=False,
    ),
}

# ============================================================================
# Provider Interface
# ============================================================================

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, model: str, dimensions: Optional[int] = None):
        self.model = model
        self.dimensions = dimensions
        self.config = MODEL_CONFIGS.get(model)
        if not self.config:
            raise ValueError(f"Unknown model: {model}")
        
        # Track actual dimensions from response
        self._actual_dimensions = dimensions or self.config.default_dimensions
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_usage(self) -> Dict[str, Any]:
        """Get usage statistics for the current session."""
        pass
    
    @abstractmethod
    def reset_usage(self) -> None:
        """Reset usage statistics."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider."""
        pass


# ============================================================================
# Google Provider Implementation
# ============================================================================

class GoogleProvider(EmbeddingProvider):
    """Google GenAI embedding provider."""
    
    def __init__(self, model: str, dimensions: Optional[int] = None, api_key: Optional[str] = None):
        super().__init__(model, dimensions)
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is required for Google provider.")
        
        from google import genai
        from google.genai import types
        
        self.client = genai.Client(api_key=self.api_key)
        self.types = types
        
        # Rate limiting state
        self._request_timestamps = deque()
        self._token_usage = deque()
        self._total_tokens = 0
        self._total_requests = 0
        self._errors = 0
        
        # Rate limits from config
        self.requests_per_minute = 60
        self.tokens_per_minute = 300000
    
    @property
    def provider_name(self) -> str:
        return "google"
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Google's API."""
        # Rate limiting check
        self._wait_if_needed(texts)
        
        config = None
        if self.dimensions is not None:
            config = self.types.EmbedContentConfig(
                output_dimensionality=self.dimensions
            )
        
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=config
        )
        
        # Update usage tracking
        self._total_requests += 1
        if hasattr(response, 'usage_metadata'):
            tokens_used = response.usage_metadata.total_token_count
            self._total_tokens += tokens_used
            current_time = time.time()
            self._request_timestamps.append((current_time, 1))  # 1 request
            self._token_usage.append((current_time, tokens_used))
        
        # Extract embeddings
        embeddings = []
        for i, embedding_data in enumerate(response.embeddings):
            if i == 0:  # First embedding sets the dimension
                self._actual_dimensions = len(embedding_data.values)
            embeddings.append(embedding_data.values)
        
        return embeddings
    
    def _wait_if_needed(self, texts: List[str]) -> None:
        """Dynamic rate limiting with rolling 60-second window."""
        current_time = time.time()
        window_seconds = 60
        
        # Clean old entries
        self._clean_old_entries(current_time, window_seconds)
        
        # Calculate tokens in current batch
        # Note: We don't know exact tokens, but can estimate
        # For simplicity, we'll use a rough estimate
        # In production, use proper token counting
        batch_tokens = sum(len(text) / 4 for text in texts)  # Rough estimate
        
        # Check request rate
        requests_in_window = len(self._request_timestamps)
        if requests_in_window >= self.requests_per_minute:
            oldest = self._request_timestamps[0][0]
            wait_time = window_seconds - (current_time - oldest) + 0.1
            if wait_time > 0:
                logger.debug(f"Rate limit: waiting {wait_time:.1f}s (requests: {requests_in_window}/{self.requests_per_minute})")
                time.sleep(wait_time)
                # Re-clean after wait
                self._clean_old_entries(time.time(), window_seconds)
        
        # Check token rate
        tokens_in_window = sum(count for _, count in self._token_usage)
        if tokens_in_window + batch_tokens > self.tokens_per_minute:
            if self._token_usage:
                oldest = self._token_usage[0][0]
                wait_time = window_seconds - (current_time - oldest) + 0.1
                if wait_time > 0:
                    logger.debug(f"Rate limit: waiting {wait_time:.1f}s (tokens: {tokens_in_window}/{self.tokens_per_minute})")
                    time.sleep(wait_time)
                    # Re-clean after wait
                    self._clean_old_entries(time.time(), window_seconds)
    
    def _clean_old_entries(self, current_time: float, window_seconds: int) -> None:
        """Remove entries older than the window."""
        cutoff = current_time - window_seconds
        
        while self._request_timestamps and self._request_timestamps[0][0] < cutoff:
            self._request_timestamps.popleft()
        
        while self._token_usage and self._token_usage[0][0] < cutoff:
            self._token_usage.popleft()
    
    def get_usage(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "provider": self.provider_name,
            "model": self.model,
            "total_tokens": self._total_tokens,
            "total_requests": self._total_requests,
            "errors": self._errors,
            "dimensions": self._actual_dimensions,
        }
    
    def reset_usage(self) -> None:
        """Reset usage statistics."""
        self._request_timestamps.clear()
        self._token_usage.clear()
        self._total_tokens = 0
        self._total_requests = 0
        self._errors = 0


# ============================================================================
# OpenAI Provider Implementation
# ============================================================================

class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, model: str, dimensions: Optional[int] = None, api_key: Optional[str] = None):
        super().__init__(model, dimensions)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider.")
        
        from openai import OpenAI
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Rate limiting state
        self._request_timestamps = deque()
        self._token_usage = deque()
        self._total_tokens = 0
        self._total_requests = 0
        self._errors = 0
        
        # Rate limits for OpenAI
        self.requests_per_minute = 60  # Depends on tier
        self.tokens_per_minute = 350000  # Depends on tier
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenAI's API."""
        # Rate limiting check
        self._wait_if_needed(texts)
        
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions if self.dimensions else self.config.default_dimensions
        )
        
        # Update usage tracking
        self._total_requests += 1
        current_time = time.time()
        self._request_timestamps.append((current_time, 1))
        
        if hasattr(response, 'usage'):
            tokens_used = response.usage.total_tokens
            self._total_tokens += tokens_used
            self._token_usage.append((current_time, tokens_used))
        
        # Extract embeddings
        embeddings = [data.embedding for data in response.data]
        if embeddings:
            self._actual_dimensions = len(embeddings[0])
        
        return embeddings
    
    def _wait_if_needed(self, texts: List[str]) -> None:
        """Dynamic rate limiting with rolling 60-second window."""
        current_time = time.time()
        window_seconds = 60
        
        # Clean old entries
        self._clean_old_entries(current_time, window_seconds)
        
        # Estimate tokens in batch (rough approximation)
        batch_tokens = sum(len(text) / 4 for text in texts)
        
        # Check request rate
        requests_in_window = len(self._request_timestamps)
        if requests_in_window >= self.requests_per_minute:
            oldest = self._request_timestamps[0][0]
            wait_time = window_seconds - (current_time - oldest) + 0.1
            if wait_time > 0:
                logger.debug(f"Rate limit: waiting {wait_time:.1f}s (requests: {requests_in_window}/{self.requests_per_minute})")
                time.sleep(wait_time)
                self._clean_old_entries(time.time(), window_seconds)
        
        # Check token rate
        tokens_in_window = sum(count for _, count in self._token_usage)
        if tokens_in_window + batch_tokens > self.tokens_per_minute:
            if self._token_usage:
                oldest = self._token_usage[0][0]
                wait_time = window_seconds - (current_time - oldest) + 0.1
                if wait_time > 0:
                    logger.debug(f"Rate limit: waiting {wait_time:.1f}s (tokens: {tokens_in_window}/{self.tokens_per_minute})")
                    time.sleep(wait_time)
                    self._clean_old_entries(time.time(), window_seconds)
    
    def _clean_old_entries(self, current_time: float, window_seconds: int) -> None:
        """Remove entries older than the window."""
        cutoff = current_time - window_seconds
        
        while self._request_timestamps and self._request_timestamps[0][0] < cutoff:
            self._request_timestamps.popleft()
        
        while self._token_usage and self._token_usage[0][0] < cutoff:
            self._token_usage.popleft()
    
    def get_usage(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "provider": self.provider_name,
            "model": self.model,
            "total_tokens": self._total_tokens,
            "total_requests": self._total_requests,
            "errors": self._errors,
            "dimensions": self._actual_dimensions,
        }
    
    def reset_usage(self) -> None:
        """Reset usage statistics."""
        self._request_timestamps.clear()
        self._token_usage.clear()
        self._total_tokens = 0
        self._total_requests = 0
        self._errors = 0


# ============================================================================
# Provider Factory
# ============================================================================

class ProviderFactory:
    """Factory for creating embedding providers."""
    
    _providers = {
        "google": GoogleProvider,
        "openai": OpenAIProvider,
    }
    
    @classmethod
    def create(cls, model: str, dimensions: Optional[int] = None, api_key: Optional[str] = None) -> EmbeddingProvider:
        """
        Create a provider instance for the given model.
        
        Args:
            model: Model name (e.g., "text-embedding-004")
            dimensions: Desired output dimensions (optional)
            api_key: API key for the provider (optional)
            
        Returns:
            EmbeddingProvider instance
        """
        config = MODEL_CONFIGS.get(model)
        if not config:
            raise ValueError(f"Unsupported model: {model}")
        
        provider_class = cls._providers.get(config.provider)
        if not provider_class:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        return provider_class(model, dimensions, api_key)


# ============================================================================
# Main Embedding Generator
# ============================================================================

@dataclass
class EmbeddingResult:
    """A single embedding result tied to its chunk."""
    chunk_id: str
    embedding: List[float]
    token_count: int
    model: str = ""
    dimensions: int = 0
    batch_index: Optional[int] = None


@dataclass
class EmbeddedChunk:
    """
    Complete chunk with embedding vector and all metadata.
    This is the final form before going to the vector database.
    """
    chunk: EnrichedChunk          # All enriched chunk data
    embedding: List[float]        # The vector embedding
    model: str = ""
    dimensions: int = 0
    version: str = "1.0"
    
    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id
    
    @property
    def content(self) -> str:
        return self.chunk.content
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return ChunkEnricher.to_metadata(self.chunk)


@dataclass
class EmbeddingBatchResult:
    """Results from embedding a batch of chunks."""
    chunks: List[EmbeddedChunk]   # Successfully embedded chunks
    total_tokens: int
    total_chunks: int
    failed_chunks: List[str]      # chunk_ids that failed
    processing_time: float        # seconds
    model: str = ""
    provider_usage: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_count(self) -> int:
        return len(self.chunks)


class EmbeddingGenerator:
    """
    Generates vector embeddings for enriched chunks using configurable providers.
    Handles batching, retries, rate limiting, and token validation.
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        dimensions: Optional[int] = None,
        batch_size: int = 20,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        api_key: Optional[str] = None,
        enable_logging: bool = True,
    ):
        """
        Initialize the embedding generator.

        Args:
            model: Embedding model to use (see MODEL_CONFIGS)
            dimensions: Desired output dimensions. None uses model default.
            batch_size: Number of texts to embed per API call
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Base delay between retries (exponential backoff applied)
            api_key: API key for the provider (optional, uses env var)
            enable_logging: Enable/disable logging output
        """
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_logging = enable_logging
        
        # Get model configuration
        self.config = MODEL_CONFIGS.get(model)
        if not self.config:
            raise ValueError(f"Unsupported model: {model}. Available: {list(MODEL_CONFIGS.keys())}")
        
        # Create provider
        self.provider = ProviderFactory.create(model, dimensions, api_key)
        
        # Set dimensions from config if not specified
        if self.dimensions is None:
            self.dimensions = self.config.default_dimensions
        
        # Version tracking
        self.version = "1.0"
        
        if self.enable_logging:
            logger.info(f"Initialized EmbeddingGenerator with model={model}, dimensions={self.dimensions}, provider={self.provider.provider_name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: List[EnrichedChunk]) -> EmbeddingBatchResult:
        """
        Generate embeddings for all enriched chunks.

        Args:
            chunks: List of EnrichedChunk objects from Step 5

        Returns:
            EmbeddingBatchResult with EmbeddedChunk objects and diagnostics
        """
        if not chunks:
            return EmbeddingBatchResult(
                chunks=[],
                total_tokens=0,
                total_chunks=0,
                failed_chunks=[],
                processing_time=0.0,
                model=self.model
            )

        start_time = time.time()

        # Validate all chunks before sending to API
        valid_chunks, oversized = self._validate_chunks(chunks)

        if oversized:
            logger.warning(
                f"{len(oversized)} chunk(s) exceed token limit ({self.config.max_tokens}) and will be skipped"
            )
            for chunk_id, tokens in oversized[:3]:
                logger.debug(f"Oversized chunk {chunk_id}: {tokens} tokens")

        # Generate embeddings in batches
        all_embedded_chunks = []
        failed = []
        retry_attempts = defaultdict(int)

        for i in range(0, len(valid_chunks), self.batch_size):
            batch = valid_chunks[i:i + self.batch_size]
            
            embedded, failed_in_batch = self._embed_batch_with_retry(
                batch, 
                batch_index=i // self.batch_size,
                retry_attempts=retry_attempts
            )
            
            all_embedded_chunks.extend(embedded)
            failed.extend(failed_in_batch)

        # Add oversized chunks to failed list
        failed.extend([chunk_id for chunk_id, _ in oversized])

        processing_time = time.time() - start_time

        # Update chunk metadata
        for embedded_chunk in all_embedded_chunks:
            chunk = embedded_chunk.chunk
            chunk.embedding_ready = True
            chunk.embedding_model = self.model
            chunk.embedding_version = self.version
            chunk.embedding_dimensions = embedded_chunk.dimensions

        # Mark failed chunks as not ready
        for chunk in chunks:
            if chunk.chunk_id in failed:
                chunk.embedding_ready = False

        return EmbeddingBatchResult(
            chunks=all_embedded_chunks,
            total_tokens=sum(ec.chunk.token_count for ec in all_embedded_chunks),
            total_chunks=len(chunks),
            failed_chunks=failed,
            processing_time=processing_time,
            model=self.model,
            provider_usage=self.provider.get_usage()
        )

    async def embed_chunks_async(self, chunks: List[EnrichedChunk]) -> EmbeddingBatchResult:
        """
        Async version for use within FastAPI endpoints.
        """
        return await asyncio.to_thread(self.embed_chunks, chunks)

    def embed_single(self, chunk: EnrichedChunk) -> Optional[EmbeddedChunk]:
        """
        Generate embedding for a single chunk (used for queries).
        """
        result = self._embed_batch_with_retry([chunk], batch_index=0)
        if result[0]:
            return result[0][0]
        return None
    
    def embed_query(self, query: str) -> Optional[List[float]]:
        """
        Generate an embedding vector for a user query.

        Unlike embed_single(), this method accepts a plain string instead of an
        EnrichedChunk and returns only the embedding vector, making it suitable
        for retrieval-time semantic search.

        Args:
            query: User query text.

        Returns:
            Embedding vector (List[float]) or None if embedding fails.
        """
        if not query or not query.strip():
            return None

        for attempt in range(self.max_retries):
            try:
                embeddings = self.provider.embed_batch([query])

                if embeddings:
                    return embeddings[0]

                return None

            except Exception as e:
                error = str(e).lower()
                retryable = any(
                    x in error
                    for x in (
                        "429",
                        "rate",
                        "quota",
                        "resource_exhausted",
                        "timeout",
                        "deadline_exceeded",
                    )
                )

                if attempt < self.max_retries - 1 and retryable:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Query embedding retry {attempt + 1}/{self.max_retries} "
                        f"in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    continue

                logger.error(f"Failed to embed query: {e}")
                return None

        return None

    def embed_queries(self, queries: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embedding vectors for multiple query strings in a single
        batched API call, instead of one round trip per query.

        Args:
            queries: List of query strings. Blank/empty entries map to None.

        Returns:
            List of embedding vectors (or None for blank/failed entries),
            in the same order as `queries`.
        """
        indexed = [(i, q) for i, q in enumerate(queries) if q and q.strip()]
        results: List[Optional[List[float]]] = [None] * len(queries)

        if not indexed:
            return results

        texts = [q for _, q in indexed]

        for attempt in range(self.max_retries):
            try:
                embeddings = self.provider.embed_batch(texts)

                for (original_idx, _), embedding in zip(indexed, embeddings):
                    results[original_idx] = embedding

                return results

            except Exception as e:
                error = str(e).lower()
                retryable = any(
                    x in error
                    for x in (
                        "429",
                        "rate",
                        "quota",
                        "resource_exhausted",
                        "timeout",
                        "deadline_exceeded",
                    )
                )

                if attempt < self.max_retries - 1 and retryable:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Batch query embedding retry {attempt + 1}/{self.max_retries} "
                        f"in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    continue

                logger.error(f"Failed to embed query batch: {e}")
                return results

        return results

    # ------------------------------------------------------------------
    # Core Embedding Logic
    # ------------------------------------------------------------------

    def _embed_batch_with_retry(
        self, 
        chunks: List[EnrichedChunk], 
        batch_index: int,
        retry_attempts: Optional[Dict[str, int]] = None
    ) -> Tuple[List[EmbeddedChunk], List[str]]:
        """
        Embed a batch of chunks with retry logic.

        Args:
            chunks: List of up to batch_size chunks
            batch_index: Index of this batch for tracking
            retry_attempts: Dictionary to track retry attempts per chunk

        Returns:
            Tuple of (embedded_chunks, failed_chunk_ids)
        """
        if not chunks:
            return [], []

        texts = [chunk.content for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]

        for attempt in range(self.max_retries):
            try:
                # Call provider
                embeddings = self.provider.embed_batch(texts)
                
                # Build EmbeddedChunk objects
                embedded_chunks = []
                for i, embedding in enumerate(embeddings):
                    embedded_chunk = EmbeddedChunk(
                        chunk=chunks[i],
                        embedding=embedding,
                        model=self.model,
                        dimensions=len(embedding),  # Actual dimensions from response
                        version=self.version
                    )
                    embedded_chunks.append(embedded_chunk)
                
                if self.enable_logging and len(embeddings) > 0:
                    logger.debug(f"Batch {batch_index}: embedded {len(embeddings)} chunks")
                
                return embedded_chunks, []

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(x in error_str for x in ["429", "rate", "quota", "resource_exhausted"])
                is_timeout = any(x in error_str for x in ["timeout", "deadline_exceeded"])
                
                if attempt < self.max_retries - 1:
                    if is_rate_limit or is_timeout:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limit/timeout. Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        # Non-retryable error or other issue
                        logger.error(f"Non-retryable error: {e}")
                        break
                else:
                    logger.error(f"Failed after {self.max_retries} attempts: {e}")
                    return [], chunk_ids

        return [], chunk_ids

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_chunks(self, chunks: List[EnrichedChunk]) -> Tuple[List[EnrichedChunk], List[Tuple[str, int]]]:
        """
        Validate chunks before embedding. Returns (valid_chunks, oversized_list).
        
        Uses precomputed token_count from EnrichedChunk to avoid redundant tokenization.
        Oversized chunks are those exceeding the model's token limit.
        """
        max_tokens = self.config.max_tokens
        valid = []
        oversized = []

        for chunk in chunks:
            if chunk.token_count > max_tokens:
                oversized.append((chunk.chunk_id, chunk.token_count))
            else:
                valid.append(chunk)

        return valid, oversized

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_embedding_summary(self, result: EmbeddingBatchResult) -> Dict[str, Any]:
        """
        Generate a human-readable summary of the embedding process.
        """
        return {
            "model": result.model,
            "dimensions": result.chunks[0].dimensions if result.chunks else self.dimensions,
            "version": self.version,
            "provider": self.provider.provider_name,
            "total_chunks": result.total_chunks,
            "successful": result.success_count,
            "failed": len(result.failed_chunks),
            "failed_chunk_ids": result.failed_chunks[:10],
            "total_tokens": result.total_tokens,
            "processing_time_seconds": round(result.processing_time, 2),
            "tokens_per_second": round(
                result.total_tokens / result.processing_time, 1
            ) if result.processing_time > 0 else 0,
            "provider_usage": result.provider_usage,
        }

    def estimate_cost(self, total_tokens: int) -> float:
        """
        Estimate cost for embedding using provider pricing.
        """
        price_per_1m = self.config.pricing_per_1m_tokens
        return (total_tokens / 1_000_000) * price_per_1m

    def switch_model(self, new_model: str, dimensions: Optional[int] = None) -> None:
        """
        Switch to a different model.
        
        Args:
            new_model: New model name
            dimensions: Optional new dimensions
        """
        self.model = new_model
        if dimensions is not None:
            self.dimensions = dimensions
        
        self.config = MODEL_CONFIGS.get(new_model)
        if not self.config:
            raise ValueError(f"Unsupported model: {new_model}")
        
        # Create new provider
        self.provider = ProviderFactory.create(new_model, self.dimensions)
        
        if self.enable_logging:
            logger.info(f"Switched to model={new_model}, dimensions={self.dimensions}, provider={self.provider.provider_name}")

    def set_logging_level(self, level: int = logging.INFO) -> None:
        """Configure logging level for the embedder."""
        logger.setLevel(level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)


# ============================================================================
# Step 6 Validation
# ============================================================================

if __name__ == "__main__":
    import os
    from pathlib import Path
    from src.ingestion.parser import DocumentParser
    from src.ingestion.cleaner import TextCleaner
    from src.ingestion.metadata_extractor import MetadataExtractor
    from src.ingestion.chunker import SemanticChunker
    from src.ingestion.enricher import ChunkEnricher

    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)

    # Full pipeline
    parser = DocumentParser()
    cleaner = TextCleaner()
    extractor = MetadataExtractor()
    chunker = SemanticChunker()
    enricher = ChunkEnricher()

    # Test with Google
    embedder = EmbeddingGenerator(
        model="gemini-embedding-001",
        dimensions=None,  # Use default
        batch_size=20,
        enable_logging=True
    )

    # Bundled sample corpus file — swap in your own files to test other formats.
    SAMPLE_FILE = str(Path(__file__).resolve().parents[2] / "sample_data" / "sample_lecture.pptx")
    test_files = [
        SAMPLE_FILE
    ]

    for file_path in test_files:
        if os.path.exists(file_path):
            logger.info(f"Processing: {file_path}")

            # Steps 1-5
            parsed = parser.parse(file_path)
            cleaned = cleaner.clean(parsed)
            metadata = extractor.extract_with_retry(cleaned)
            chunks = chunker.chunk(cleaned, metadata)
            enriched = enricher.enrich(chunks)

            logger.info(f"Pre-embedding: {len(enriched)} chunks, {sum(c.token_count for c in enriched)} total tokens")

            # Step 6
            result = embedder.embed_chunks(enriched)

            # Summary
            summary = embedder.get_embedding_summary(result)
            cost = embedder.estimate_cost(result.total_tokens)

            logger.info("Embedding Results:")
            logger.info(f"  Model: {summary['model']}")
            logger.info(f"  Provider: {summary['provider']}")
            logger.info(f"  Dimensions: {summary['dimensions']}")
            logger.info(f"  Successful: {summary['successful']}/{summary['total_chunks']}")
            logger.info(f"  Failed: {summary['failed']}")
            logger.info(f"  Total tokens: {summary['total_tokens']}")
            logger.info(f"  Processing time: {summary['processing_time_seconds']}s")
            logger.info(f"  Tokens/sec: {summary['tokens_per_second']}")
            logger.info(f"  Est. cost: ${cost:.6f}")
            logger.info(f"  Provider usage: {summary['provider_usage']}")

            # Show sample embedding
            if result.chunks:
                sample = result.chunks[0]
                logger.info("Sample Embedded Chunk:")
                logger.info(f"  Chunk ID: {sample.chunk_id}")
                logger.info(f"  Content preview: {sample.content[:100]}...")
                logger.info(f"  Vector dimensions: {len(sample.embedding)}")
                logger.info(f"  First 5 values: {sample.embedding[:5]}")
                
            # Test switching to OpenAI
            if os.getenv("OPENAI_API_KEY"):
                logger.info("\nTesting OpenAI provider...")
                embedder.switch_model("text-embedding-3-small", dimensions=768)
                result_openai = embedder.embed_chunks(enriched[:5])
                logger.info(f"OpenAI result: {len(result_openai.chunks)} chunks embedded")
        else:
            logger.warning(f"Test file not found: {file_path}")