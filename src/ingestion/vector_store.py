"""
Step 7: Vector Storage (Pinecone)
Stores embedded chunks in Pinecone with full metadata for retrieval,
filtering, and traceability. Handles index creation, upsert, and namespace
management.
"""

import os
import time
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

from pinecone import Pinecone, ServerlessSpec
from pinecone.core.client.exceptions import NotFoundException
from dotenv import load_dotenv

from src.ingestion.embedder import EmbeddedChunk, EmbeddingBatchResult
from src.ingestion.metadata_extractor import DocumentMetadata

load_dotenv()

# Constants
DIMENSIONS = 3072
METRIC = "cosine"
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")
BATCH_SIZE = 100                         # Pinecone recommends 100 vectors per upsert
MAX_RETRIES = 3
RETRY_DELAY = 1.0


@dataclass
class UpsertResult:
    """Results from a Pinecone upsert operation."""
    chunks_upserted: int
    chunks_failed: int
    total_chunks: int
    namespace: str
    index_name: str
    processing_time: float
    failed_chunk_ids: List[str] = field(default_factory=list)
    # Batch-level diagnostics
    total_vectors_attempted: int = 0
    pinecone_response_stats: Dict[str, Any] = field(default_factory=dict)
    failed_batch_indices: List[int] = field(default_factory=list)
    # Aggregated response stats
    total_upserted_count: int = 0
    avg_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = 0.0


class VectorStore:
    """
    Manages Pinecone vector storage for embedded chunks.
    Handles index creation, batch upsert, deletion, and namespace management.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        cloud: str = CLOUD,
        region: str = REGION
    ):
        """
        Initialize the vector store connection.

        Args:
            api_key: Pinecone API key. If None, loads from PINECONE_API_KEY env var.
            index_name: Name of the Pinecone index. If None, loads from env.
            cloud: Cloud provider (aws, gcp, azure)
            region: Cloud region
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is required. Set it in .env or pass directly.")

        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "rag-pipeline-index")
        self.cloud = cloud
        self.region = region

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)

        # Ensure index exists
        self._ensure_index()

        # Connect to index
        self.index = self.pc.Index(self.index_name)

    # ------------------------------------------------------------------
    # Index Management
    # ------------------------------------------------------------------

    def _ensure_index(self):
        """
        Create the index if it doesn't exist.
        Uses ServerlessSpec for cost efficiency.
        """
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"📦 Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=DIMENSIONS,
                metric=METRIC,
                spec=ServerlessSpec(
                    cloud=self.cloud,
                    region=self.region
                )
            )
            # Wait for index to be ready
            self._wait_for_index()
        else:
            print(f"📦 Using existing Pinecone index: {self.index_name}")

    def _wait_for_index(self, timeout: int = 60):
        """
        Wait for the index to become ready.
        """
        start = time.time()
        while time.time() - start < timeout:
            try:
                desc = self.pc.describe_index(self.index_name)
                if desc.status.get("ready", False):
                    print(f"✅ Index ready in {time.time() - start:.1f}s")
                    return
            except Exception:
                pass
            time.sleep(2)

        raise TimeoutError(f"Index {self.index_name} did not become ready within {timeout}s")

    def delete_index(self):
        """
        Delete the index (use with caution).
        """
        print(f"🗑️  Deleting index: {self.index_name}")
        self.pc.delete_index(self.index_name)

    # ------------------------------------------------------------------
    # Namespace Generation
    # ------------------------------------------------------------------

    @staticmethod
    def sanitize_namespace(name: str) -> str:
        """
        Sanitize a string for use as a Pinecone namespace.
        
        Pinecone namespaces must be lowercase alphanumeric with hyphens/underscores.
        """
        # Convert to lowercase and replace spaces with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9\s\-_]', '', name.lower())
        sanitized = re.sub(r'[\s]+', '_', sanitized)
        # Remove leading/trailing special chars
        sanitized = sanitized.strip('-_')
        # Ensure non-empty
        if not sanitized:
            sanitized = "default"
        return sanitized

    @staticmethod
    def generate_namespace_from_document(file_path: str, course_name: str = None) -> str:
        """
        Generate a deterministic namespace from document.
        
        Args:
            file_path: Absolute path to the document (for uniqueness)
            course_name: Optional course name for human-readable prefix
        
        Returns:
            Sanitized namespace string
        """
        # Use absolute path for uniqueness across users
        abs_path = os.path.abspath(file_path)
        
        # Create a hash of the absolute path
        # Using SHA256 for better collision resistance
        doc_hash = hashlib.sha256(abs_path.encode()).hexdigest()[:12]
        
        if course_name:
            course_slug = VectorStore.sanitize_namespace(course_name)
            return f"{course_slug}_{doc_hash}"
        else:
            # Use filename as prefix for readability
            filename = os.path.basename(file_path)
            file_prefix = VectorStore.sanitize_namespace(os.path.splitext(filename)[0])
            return f"{file_prefix}_{doc_hash}"

    @staticmethod
    def generate_namespace_from_uuid(document_uuid: str, course_name: str = None) -> str:
        """
        Generate a namespace from a UUID.
        
        Args:
            document_uuid: Unique identifier for the document
            course_name: Optional course name for human-readable prefix
        
        Returns:
            Sanitized namespace string
        """
        if course_name:
            course_slug = VectorStore.sanitize_namespace(course_name)
            return f"{course_slug}_{document_uuid[:8]}"
        else:
            return f"doc_{document_uuid[:8]}"

    # ------------------------------------------------------------------
    # Upsert Operations
    # ------------------------------------------------------------------

    def upsert(self, result: EmbeddingBatchResult, namespace: str = "default") -> UpsertResult:
        """
        Store all embedded chunks from an EmbeddingBatchResult.

        Args:
            result: EmbeddingBatchResult from Step 6
            namespace: Pinecone namespace for organization

        Returns:
            UpsertResult with upsert statistics
        """
        if not result.chunks:
            return UpsertResult(
                chunks_upserted=0,
                chunks_failed=0,
                total_chunks=0,
                namespace=namespace,
                index_name=self.index_name,
                processing_time=0.0,
                total_vectors_attempted=0,
                pinecone_response_stats={}
            )

        start_time = time.time()
        failed = []
        failed_batches = []
        
        # Track response stats for aggregation
        response_times = []
        total_upserted_count = 0

        # Batch upsert in groups of BATCH_SIZE
        chunks = result.chunks
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_idx, i in enumerate(range(0, len(chunks), BATCH_SIZE)):
            batch = chunks[i:i + BATCH_SIZE]
            
            # Prepare vectors with validation
            try:
                vectors = self._prepare_vectors(batch)
            except ValueError as e:
                print(f"❌ Vector preparation failed for batch {batch_idx}: {e}")
                failed.extend([c.chunk_id for c in batch])
                failed_batches.append(batch_idx)
                continue
            
            success, response_stats = self._upsert_batch_with_stats(vectors, namespace)
            
            if not success:
                failed.extend([c.chunk_id for c in batch])
                failed_batches.append(batch_idx)
            else:
                # Aggregate response stats
                if "upserted_count" in response_stats:
                    total_upserted_count += response_stats["upserted_count"]
                if "response_time" in response_stats:
                    response_times.append(response_stats["response_time"])
            
            # Log progress for large batches
            if total_batches > 5 and (batch_idx + 1) % 5 == 0:
                print(f"   Progress: {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks processed")

        processing_time = time.time() - start_time
        upserted = len(chunks) - len(failed)

        # Calculate aggregated stats
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        max_response_time = max(response_times) if response_times else 0.0
        min_response_time = min(response_times) if response_times else 0.0

        return UpsertResult(
            chunks_upserted=upserted,
            chunks_failed=len(failed),
            total_chunks=len(chunks),
            namespace=namespace,
            index_name=self.index_name,
            processing_time=processing_time,
            failed_chunk_ids=failed,
            total_vectors_attempted=len(chunks),
            pinecone_response_stats={
                "successful_batches": total_batches - len(failed_batches),
                "failed_batches": len(failed_batches),
                "total_batches": total_batches,
                "total_upserted": total_upserted_count,
                "avg_response_time": avg_response_time,
                "max_response_time": max_response_time,
                "min_response_time": min_response_time
            },
            failed_batch_indices=failed_batches,
            total_upserted_count=total_upserted_count,
            avg_response_time=avg_response_time,
            max_response_time=max_response_time,
            min_response_time=min_response_time
        )

    def upsert_single(self, chunk: EmbeddedChunk, namespace: str = "default") -> bool:
        """
        Store a single embedded chunk (used for incremental updates).
        """
        try:
            vectors = self._prepare_vectors([chunk])
            success, _ = self._upsert_batch_with_stats(vectors, namespace)
            return success
        except ValueError:
            return False

    def _prepare_vectors(self, chunks: List[EmbeddedChunk]) -> List[Dict[str, Any]]:
        """
        Convert EmbeddedChunk objects to Pinecone vector format.

        Each vector is a tuple of (id, embedding, metadata).
        Metadata includes all traceability fields for filtering.
        """
        vectors = []

        for chunk in chunks:
            # Validate embedding dimension
            if len(chunk.embedding) != DIMENSIONS:
                raise ValueError(
                    f"{chunk.chunk_id} has {len(chunk.embedding)} dimensions "
                    f"expected {DIMENSIONS}"
                )

            # Copy metadata to avoid mutation
            metadata = chunk.metadata.copy()

            # Remove large fields that aren't useful for filtering
            # (Pinecone has a 40KB metadata limit per vector)
            metadata.pop("global_context", None)
            metadata.pop("key_topics", None)
            
            # Store embedding model information for traceability
            # FIXED: Use chunk.model, chunk.version, chunk.dimensions
            metadata["embedding_model"] = chunk.model
            metadata["embedding_version"] = chunk.version
            metadata["embedding_dimensions"] = chunk.dimensions

            vectors.append({
                "id": chunk.chunk_id,
                "values": chunk.embedding,
                "metadata": metadata
            })

        return vectors

    def _upsert_batch_with_stats(self, vectors: List[Dict[str, Any]], namespace: str) -> tuple[bool, Dict[str, Any]]:
        """
        Upsert a batch of vectors with retry logic and statistics.
        
        Returns:
            (success, response_stats)
        """
        response_stats = {}
        
        for attempt in range(MAX_RETRIES):
            try:
                # Measure actual request duration
                request_start = time.time()
                response = self.index.upsert(
                    vectors=vectors,
                    namespace=namespace
                )
                request_elapsed = time.time() - request_start
                
                # Capture response statistics
                response_stats = {
                    "attempt": attempt + 1,
                    "vector_count": len(vectors),
                    "upserted_count": response.upserted_count if hasattr(response, 'upserted_count') else len(vectors),
                    "response_time": request_elapsed  # Actual duration in seconds
                }
                
                return True, response_stats

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    print(f"⚠️  Upsert failed (attempt {attempt + 1}): {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Upsert failed after {MAX_RETRIES} attempts: {e}")
                    response_stats = {
                        "attempt": MAX_RETRIES,
                        "vector_count": len(vectors),
                        "error": str(e)
                    }
                    return False, response_stats

        return False, response_stats

    # ------------------------------------------------------------------
    # Query Operations (for Step 8)
    # ------------------------------------------------------------------

    def query(
        self,
        vector: List[float],
        top_k: int = 20,
        namespace: Union[str, List[str]] = "default",
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the index for similar vectors.

        Pinecone only supports querying one namespace per call — each
        document is ingested into its own namespace (see
        `documents.py::_generate_namespace`) so callers that need to search
        everything a profile has uploaded pass a list here. In that case we
        fan out one query per namespace and merge the results by score,
        since there's no server-side cross-namespace query.

        Args:
            vector: Query embedding vector
            top_k: Number of results to return (across all namespaces combined)
            namespace: Single Pinecone namespace, or a list to search across
            filter: Optional metadata filter (e.g., {"course_name": "CS101"})

        Returns:
            List of matches with id, score, and metadata, sorted by score
            descending and truncated to top_k
        """
        namespaces = namespace if isinstance(namespace, list) else [namespace]
        namespaces = [ns for ns in dict.fromkeys(namespaces) if ns]  # dedupe, drop empties
        if not namespaces:
            return []

        def _query_one(ns: str) -> List[Dict[str, Any]]:
            try:
                results = self.index.query(
                    vector=vector,
                    top_k=top_k,
                    namespace=ns,
                    filter=filter,
                    include_metadata=True
                )
                return [
                    {"id": match.id, "score": match.score, "metadata": match.metadata}
                    for match in results.matches
                ]
            except Exception as e:
                print(f"❌ Query failed for namespace '{ns}': {e}")
                return []

        all_matches: List[Dict[str, Any]] = []
        if len(namespaces) == 1:
            all_matches = _query_one(namespaces[0])
        else:
            # Run per-namespace queries concurrently — these are independent
            # I/O-bound HTTP calls, and retrieval overall runs under a tight
            # time budget (see RetrievalConfig timeouts), so querying N
            # namespaces sequentially multiplies latency by N and can blow
            # the budget once a profile has more than one or two documents.
            with ThreadPoolExecutor(max_workers=min(8, len(namespaces))) as pool:
                for matches in pool.map(_query_one, namespaces):
                    all_matches.extend(matches)

        all_matches.sort(key=lambda m: m["score"], reverse=True)
        return all_matches[:top_k]

    # ------------------------------------------------------------------
    # Namespace Management
    # ------------------------------------------------------------------

    def list_namespaces(self) -> List[str]:
        """
        List all namespaces in the index.
        """
        stats = self.index.describe_index_stats()
        return list(stats.namespaces.keys())

    def delete_namespace(self, namespace: str):
        """
        Delete all vectors in a namespace.

        A namespace with zero vectors doesn't exist as far as Pinecone is
        concerned, so deleting it 404s even though the caller's goal (no
        vectors under this namespace) is already satisfied — treat that as
        success rather than surfacing it as a failure.
        """
        print(f"🗑️  Deleting namespace: {namespace}")
        try:
            self.index.delete(delete_all=True, namespace=namespace)
        except NotFoundException:
            print(f"ℹ️  Namespace already absent: {namespace}")

    def get_namespace_stats(self, namespace: str = "default") -> Dict[str, Any]:
        """
        Get statistics for a namespace.
        """
        stats = self.index.describe_index_stats()
        ns_stats = stats.namespaces.get(namespace, {})
        return {
            "namespace": namespace,
            "vector_count": ns_stats.get("vector_count", 0),
            "dimension": stats.dimension
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get overall index statistics.
        """
        stats = self.index.describe_index_stats()
        return {
            "index_name": self.index_name,
            "dimension": stats.dimension,
            "total_vector_count": stats.total_vector_count,
            "namespaces": list(stats.namespaces.keys()),
            "namespace_count": len(stats.namespaces)
        }


# ============================================================================
# Step 7 Validation
# ============================================================================

if __name__ == "__main__":
    import os
    from src.ingestion.parser import DocumentParser
    from src.ingestion.cleaner import TextCleaner
    from src.ingestion.metadata_extractor import MetadataExtractor
    from src.ingestion.chunker import SemanticChunker
    from src.ingestion.enricher import ChunkEnricher
    from src.ingestion.embedder import EmbeddingGenerator
    from pathlib import Path

    # Full pipeline
    parser = DocumentParser()
    cleaner = TextCleaner()
    extractor = MetadataExtractor()
    chunker = SemanticChunker()
    enricher = ChunkEnricher()
    embedder = EmbeddingGenerator()
    vector_store = VectorStore()

    # Use deterministic namespace generation. Bundled sample corpus file —
    # swap in your own files to test other formats.
    SAMPLE_FILE = str(Path(__file__).resolve().parents[2] / "sample_data" / "sample_lecture.pptx")
    test_files = [
        SAMPLE_FILE
    ]

    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n{'='*60}")
            print(f"Vector Storage: {file_path}")
            print(f"{'='*60}")

            # Steps 1-5
            parsed = parser.parse(file_path)
            cleaned = cleaner.clean(parsed)
            doc_metadata = extractor.extract_with_retry(cleaned)
            chunks = chunker.chunk(cleaned, doc_metadata)
            enriched = enricher.enrich(chunks)

            # Step 6
            embedding_result = embedder.embed_chunks(enriched)

            # Step 7 - Use deterministic namespace generation
            # FIXED: Use doc_metadata.course_name (not .get())
            course_name = getattr(doc_metadata, "course_name", "unknown")
            namespace = vector_store.generate_namespace_from_document(file_path, course_name)
            
            print(f"📍 Using namespace: {namespace}")
            
            upsert_result = vector_store.upsert(embedding_result, namespace=namespace)

            print(f"\n📊 Upsert Results:")
            print(f"   Index: {upsert_result.index_name}")
            print(f"   Namespace: {upsert_result.namespace}")
            print(f"   Upserted: {upsert_result.chunks_upserted}/{upsert_result.total_chunks}")
            print(f"   Failed: {upsert_result.chunks_failed}")
            print(f"   Processing time: {upsert_result.processing_time:.2f}s")
            print(f"   Batch stats: {upsert_result.pinecone_response_stats}")
            print(f"   Avg response time: {upsert_result.avg_response_time:.3f}s")
            print(f"   Total upserted count: {upsert_result.total_upserted_count}")

            # Verify storage
            ns_stats = vector_store.get_namespace_stats(namespace)
            print(f"\n📊 Namespace Stats:")
            print(f"   Vector count: {ns_stats['vector_count']}")
            print(f"   Dimension: {ns_stats['dimension']}")

            # Test query
            if embedding_result.chunks:
                test_vector = embedding_result.chunks[0].embedding
                results = vector_store.query(
                    vector=test_vector,
                    top_k=3,
                    namespace=namespace
                )
                print(f"\n🔍 Test Query (top 3):")
                for r in results:
                    print(f"   [{r['score']:.4f}] {r['id']}")
                    print(f"      Course: {r['metadata'].get('course_name', 'N/A')}")
                    print(f"      Chapter: {r['metadata'].get('chapter_title', 'N/A')}")
                    print(f"      Embedding Model: {r['metadata'].get('embedding_model', 'N/A')}")
                    print(f"      Embedding Version: {r['metadata'].get('embedding_version', 'N/A')}")
        else:
            print(f"⚠️  Test file not found: {file_path}")

    # Show overall index stats
    print(f"\n{'='*60}")
    print(f"Index Overview")
    print(f"{'='*60}")
    index_stats = vector_store.get_index_stats()
    for key, value in index_stats.items():
        print(f"   {key}: {value}")