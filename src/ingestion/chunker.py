"""
Step 4: Semantic Chunking
Uses DeepSeek to create intelligent, topic-aware chunks aligned with
slide boundaries (PPTX) or topic transitions (PDF/DOCX).
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
import tiktoken

from src.common.llm_client import simple_generate

# Try to import NLTK, fallback to regex if not available. Catches
# LookupError as well as ImportError — nltk being installed doesn't mean
# its data (punkt/punkt_tab, renamed between nltk versions) is actually
# present, and that failure mode should degrade to the regex fallback
# just like a missing package would, not crash the whole import chain.
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.data.find('tokenizers/punkt')
    NLTK_AVAILABLE = True
except (ImportError, LookupError):
    NLTK_AVAILABLE = False
    print("WARNING: NLTK not available. Using fallback sentence splitting.")

from src.ingestion.cleaner import CleanedDocument
from src.ingestion.metadata_extractor import DocumentMetadata

load_dotenv()


@dataclass
class Chunk:
    """A single semantic chunk enriched with document metadata."""
    # Required chunk metadata
    chunk_id: str
    content: str
    chunk_type: str

    page_start: int
    page_end: int

    topic: str

    # Required document metadata
    filename: str
    file_type: str

    course_name: str
    chapter_title: str
    subject_area: str
    global_context: str
    key_topics: List[str]

    parent_chunk_id: str = ""
    extraction_confidence: float = 0.0
    token_count: int = 0
    element_ids: List[int] = field(default_factory=list)


class SemanticChunker:
    """
    Creates intelligent, topic-aware chunks using DeepSeek.
    Preserves slide boundaries for PPTX and identifies topic transitions
    for text-based documents.

    Uses token-based chunk sizing for optimal compatibility with
    text-embedding-3-small.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_chunk_tokens: int = 512,
        min_chunk_tokens: int = 75
    ):
        """
        Initialize the semantic chunker.

        Args:
            api_key: DeepSeek API key. If None, loads from env var.
            max_chunk_tokens: Maximum tokens per chunk (default: 512)
            min_chunk_tokens: Minimum tokens per chunk (default: 75)
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is required.")

        # Token-based chunking configuration for text-embedding-3-small
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        
        # Initialize tiktoken for the embedding model
        self.encoding = tiktoken.encoding_for_model("text-embedding-3-small")
        
        # Slide boundary tolerance (still used for PPTX)
        self.slide_boundary_tolerance = 2
        
        # Sentence splitting configuration
        self.use_nltk = NLTK_AVAILABLE

    # ------------------------------------------------------------------
    # Token Utilities
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def _encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.encoding.encode(text)

    def _decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text."""
        return self.encoding.decode(tokens)

    def _get_token_info(self, text: str) -> Tuple[int, List[int]]:
        """Get both token count and token IDs in one operation."""
        tokens = self._encode(text)
        return len(tokens), tokens

    # ------------------------------------------------------------------
    # Main Chunking Pipeline
    # ------------------------------------------------------------------

    def chunk(self, cleaned_doc: CleanedDocument, metadata: DocumentMetadata) -> List[Chunk]:
        """
        Create semantic chunks from a cleaned document.

        For PPTX: Chunks align with slide boundaries (1+ slides per chunk).
        For PDF/DOCX: DeepSeek identifies topic transition points.

        Args:
            cleaned_doc: CleanedDocument from Step 2
            metadata: DocumentMetadata from Step 3

        Returns:
            List of Chunk objects with page ranges and topics
        """
        if cleaned_doc.file_type == "pptx":
            chunks = self._chunk_by_slides(cleaned_doc, metadata)
        else:
            chunks = self._chunk_by_topics(cleaned_doc, metadata)

        # First split oversized chunks to ensure all chunks respect max size
        chunks = self._split_oversized_chunks(chunks)

        # Assign sequential chunk IDs after all splitting is complete
        stem = Path(cleaned_doc.filename).stem
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"{stem}_chunk_{i:04d}"
            # Ensure token count is populated
            if chunk.token_count == 0:
                chunk.token_count = self._count_tokens(chunk.content)

        return chunks

    # ------------------------------------------------------------------
    # Sentence Splitting (with NLTK fallback)
    # ------------------------------------------------------------------

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK if available, otherwise fallback to regex.
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentences
        """
        if self.use_nltk:
            try:
                return sent_tokenize(text)
            except Exception as e:
                print(f"WARNING: NLTK sentence splitting failed: {e}. Using fallback.")
                return self._fallback_sentence_split(text)
        else:
            return self._fallback_sentence_split(text)

    def _fallback_sentence_split(self, text: str) -> List[str]:
        """
        Fallback sentence splitting using regex.
        Handles common abbreviations and punctuation.
        """
        # Pattern for sentence boundaries: ., !, ? followed by space or end of string
        # Exclude common abbreviations
        abbreviations = r'\b(?:Dr|Mr|Mrs|Ms|Prof|Rev|Hon|St|Ave|Blvd|Rd|Jr|Sr|vs|etc|e\.g|i\.e|U\.S|U\.S\.A|U\.K|U\.N)\b'
        
        # Replace abbreviations with placeholders
        placeholder = "___ABBR___"
        abbr_map = {}
        
        def replace_abbr(match):
            abbr = match.group(0)
            key = f"{placeholder}_{len(abbr_map)}"
            abbr_map[key] = abbr
            return key
        
        text_with_placeholders = re.sub(abbreviations, replace_abbr, text)
        
        # Split on sentence boundaries: ., !, ? followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', text_with_placeholders)
        
        # Restore abbreviations
        sentences = [
            self._restore_abbreviations(sent, abbr_map)
            for sent in sentences
            if sent.strip()
        ]
        
        return sentences

    def _restore_abbreviations(self, text: str, abbr_map: Dict[str, str]) -> str:
        """Restore abbreviation placeholders."""
        result = text
        for placeholder, abbr in abbr_map.items():
            result = result.replace(placeholder, abbr)
        return result

    # ------------------------------------------------------------------
    # Token-Based Chunk Size Management
    # ------------------------------------------------------------------

    def _split_oversized_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Split chunks that exceed max_chunk_tokens into smaller pieces.
        """
        if not chunks:
            return chunks

        final_chunks = []
        
        for chunk in chunks:
            token_count = self._count_tokens(chunk.content)
            if token_count <= self.max_chunk_tokens:
                chunk.token_count = token_count
                final_chunks.append(chunk)
            else:
                # Split oversized chunk
                split_chunks = self._split_chunk_by_tokens(chunk)
                final_chunks.extend(split_chunks)
        
        return final_chunks

    def _split_chunk_by_tokens(self, chunk: Chunk) -> List[Chunk]:
        """
        Split a single chunk into smaller chunks by token count.
        Attempts to split at sentence boundaries.
        
        Returns:
            List of split chunks with parent relationships preserved
        """
        content = chunk.content
        max_tokens = self.max_chunk_tokens
        
        # Get sentences using NLTK or fallback
        sentences = self._split_into_sentences(content)
        
        split_chunks = []
        current_sentences = []
        current_tokens = 0
        chunk_counter = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            # If this sentence alone exceeds max tokens, split it further by tokens
            if sentence_tokens > max_tokens:
                # Flush current chunk if exists
                if current_sentences:
                    split_chunks.append(
                        self._create_split_chunk(
                            chunk,
                            " ".join(current_sentences),
                            chunk.element_ids,
                            chunk_counter
                        )
                    )
                    chunk_counter += 1
                    current_sentences = []
                    current_tokens = 0
                
                # Split long sentence by token boundaries
                tokens = self._encode(sentence)
                for i in range(0, len(tokens), max_tokens):
                    part_tokens = tokens[i:i+max_tokens]
                    part_text = self._decode(part_tokens)
                    if part_text:
                        split_chunks.append(
                            self._create_split_chunk(
                                chunk,
                                part_text.strip(),
                                chunk.element_ids,
                                chunk_counter
                            )
                        )
                        chunk_counter += 1
                continue
            
            # Check if adding this sentence would exceed max tokens
            if current_tokens + sentence_tokens <= max_tokens:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
            else:
                # Flush current chunk and start new one
                if current_sentences:
                    split_chunks.append(
                        self._create_split_chunk(
                            chunk,
                            " ".join(current_sentences),
                            chunk.element_ids,
                            chunk_counter
                        )
                    )
                    chunk_counter += 1
                
                # Start new chunk with this sentence
                current_sentences = [sentence]
                current_tokens = sentence_tokens
        
        # Don't forget the last chunk
        if current_sentences:
            split_chunks.append(
                self._create_split_chunk(
                    chunk,
                    " ".join(current_sentences),
                    chunk.element_ids,
                    chunk_counter
                )
            )
        
        return split_chunks

    def _create_split_chunk(
        self, 
        original: Chunk, 
        content: str, 
        element_ids: List[int], 
        part_num: int
    ) -> Chunk:
        """
        Create a new chunk from an original chunk with modified content.
        Preserves parent relationship and calculates token count.
        """
        return Chunk(
            chunk_id="",  # Will be assigned later
            content=content,
            chunk_type=original.chunk_type + "_split",
            page_start=original.page_start,
            page_end=original.page_end,
            topic=original.topic,
            parent_chunk_id=original.chunk_id,  # Track parent relationship
            filename=original.filename,
            file_type=original.file_type,
            course_name=original.course_name,
            chapter_title=original.chapter_title,
            subject_area=original.subject_area,
            global_context=original.global_context,
            key_topics=original.key_topics,
            extraction_confidence=original.extraction_confidence,
            token_count=self._count_tokens(content),
            element_ids=element_ids
        )

    # ------------------------------------------------------------------
    # PPTX: Slide-Boundary-Aware Chunking
    # ------------------------------------------------------------------

    def _chunk_by_slides(self, cleaned_doc: CleanedDocument, metadata: DocumentMetadata) -> List[Chunk]:
        """
        Group elements by slide. Merge small adjacent slides into one chunk.
        Uses token counts for size decisions.
        """
        # Group elements by page_number (slide number for PPTX)
        slides: Dict[int, List] = {}
        for idx, element in enumerate(cleaned_doc.elements):
            page = element.page_number
            if page not in slides:
                slides[page] = []
            slides[page].append((idx, element))

        # Build initial slide chunks with token counts
        slide_chunks = []
        for page_num in sorted(slides.keys()):
            elements = slides[page_num]
            content = "\n".join(elem.content for _, elem in elements)
            element_ids = [idx for idx, _ in elements]
            token_count = self._count_tokens(content)

            slide_chunks.append({
                "page": page_num,
                "content": content,
                "element_ids": element_ids,
                "tokens": token_count
            })

        # Merge small adjacent slides
        merged = self._merge_small_chunks(slide_chunks)

        # Assign topics via DeepSeek (batch)
        topics = self._assign_topics_batch(
            [m["content"][:500] for m in merged],
            metadata
        )

        # Build Chunk objects
        chunks = []
        for i, m in enumerate(merged):
            chunk = Chunk(
                chunk_id="",

                content=m["content"],
                chunk_type="slide",

                page_start=m["page"],
                page_end=m["page"] if len(m["pages"]) == 1 else m["pages"][-1],

                topic=topics[i] if i < len(topics) else "General",

                parent_chunk_id="",  # No parent for original chunks

                filename=metadata.filename,
                file_type=metadata.file_type,

                course_name=metadata.course_name,
                chapter_title=metadata.chapter_title,
                subject_area=metadata.subject_area,
                global_context=metadata.global_context,
                key_topics=metadata.key_topics,

                extraction_confidence=metadata.extraction_confidence,

                token_count=m["tokens"],

                element_ids=m["element_ids"]
            )
            chunks.append(chunk)

        return chunks

    def _merge_small_chunks(self, slide_chunks: List[Dict]) -> List[Dict]:
        """
        Merge adjacent slides that are individually below min_chunk_tokens.
        Respects slide_boundary_tolerance limit.
        """
        if not slide_chunks:
            return []

        merged = []
        buffer = None

        for sc in slide_chunks:
            if buffer is None:
                buffer = {
                    "page": sc["page"],
                    "pages": [sc["page"]],
                    "content": sc["content"],
                    "element_ids": sc["element_ids"][:],
                    "tokens": sc["tokens"]
                }
            elif buffer["tokens"] < self.min_chunk_tokens:
                # Merge current into buffer
                pages_after_merge = buffer["pages"] + [sc["page"]]
                if len(pages_after_merge) <= self.slide_boundary_tolerance + 1:
                    buffer["pages"].append(sc["page"])
                    buffer["content"] += "\n\n" + sc["content"]
                    buffer["element_ids"].extend(sc["element_ids"])
                    buffer["tokens"] += sc["tokens"]
                else:
                    # Would exceed slide merge limit, flush buffer
                    merged.append(buffer)
                    buffer = {
                        "page": sc["page"],
                        "pages": [sc["page"]],
                        "content": sc["content"],
                        "element_ids": sc["element_ids"][:],
                        "tokens": sc["tokens"]
                    }
            else:
                # Buffer is large enough, flush it
                merged.append(buffer)
                buffer = {
                    "page": sc["page"],
                    "pages": [sc["page"]],
                    "content": sc["content"],
                    "element_ids": sc["element_ids"][:],
                    "tokens": sc["tokens"]
                }

        # Don't forget the last buffer
        if buffer is not None:
            # If last buffer is small, try merging with previous
            if buffer["tokens"] < self.min_chunk_tokens and merged:
                prev = merged[-1]
                total_pages = len(prev["pages"]) + len(buffer["pages"])
                if total_pages <= self.slide_boundary_tolerance + 1:
                    prev["pages"].extend(buffer["pages"])
                    prev["content"] += "\n\n" + buffer["content"]
                    prev["element_ids"].extend(buffer["element_ids"])
                    prev["tokens"] += buffer["tokens"]
                else:
                    merged.append(buffer)
            else:
                merged.append(buffer)

        return merged

    # ------------------------------------------------------------------
    # PDF / DOCX: Topic-Transition Chunking via DeepSeek
    # ------------------------------------------------------------------

    def _chunk_by_topics(self, cleaned_doc: CleanedDocument, metadata: DocumentMetadata) -> List[Chunk]:
        """
        Use DeepSeek to identify topic transition points, then split.
        """
        # Build indexed representation
        indexed_elements = []
        for idx, element in enumerate(cleaned_doc.elements):
            location = f"[Page {element.page_number}]"
            indexed_elements.append({
                "index": idx,
                "location": location,
                "content": element.content,
                "page": element.page_number
            })

        # Get transition points with topics from DeepSeek (single API call)
        chunk_definitions = self._detect_transitions_with_topics(
            indexed_elements,
            metadata
        )

        # Split into chunks at transition points with assigned topics
        chunks = self._split_with_topics(
            indexed_elements,
            chunk_definitions,
            metadata
        )

        return chunks

    def _detect_transitions_with_topics(
        self,
        elements: List[Dict],
        metadata: DocumentMetadata
    ) -> List[Dict]:
        """
        Ask DeepSeek to identify topic transitions AND assign topics in one call.
        Returns list of {start_index, topic} dictionaries.
        """
        # Build compact representation for DeepSeek
        lines = []
        for e in elements:
            lines.append(f"[{e['index']}] {e['location']}: {e['content'][:150]}")

        text_block = "\n".join(lines)

        # Build context with only known metadata
        context_parts = []
        if metadata.course_name and metadata.course_name != "Unknown":
            context_parts.append(f"Course: {metadata.course_name}")
        if metadata.chapter_title and metadata.chapter_title != "Unknown":
            context_parts.append(f"Chapter: {metadata.chapter_title}")
        if metadata.subject_area and metadata.subject_area != "Unknown":
            context_parts.append(f"Subject: {metadata.subject_area}")
        if metadata.filename:
            context_parts.append(f"Document: {metadata.filename}")
        
        context_text = "\n".join(context_parts) if context_parts else ""

        prompt = f"""
{context_text}

You are analyzing a document to find topic transition points for chunking.

Below are indexed elements from the document. Identify where the TOPIC changes significantly.
Return a JSON array of objects, where each object marks the START of a new chunk.

Format:
[
  {{"start": 0, "topic": "Main topic for first chunk"}},
  {{"start": 5, "topic": "Topic for second chunk"}},
  ...
]

Rules:
- Always include index 0 as the first chunk start
- Mark transitions where the subject matter clearly shifts
- Aim for chunks of roughly 300-500 tokens
- Topics should be concise (3-7 words)
- Return ONLY a JSON array
- Do NOT use markdown or code blocks

Elements:
{text_block[:5000]}
"""

        try:
            response_text = simple_generate(prompt).strip()

            # Clean markdown if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            chunk_defs = json.loads(response_text)
            
            # Validate and ensure required fields
            if isinstance(chunk_defs, list) and len(chunk_defs) > 0:
                # Ensure first chunk starts at 0
                if chunk_defs[0].get("start", 0) != 0:
                    chunk_defs.insert(0, {"start": 0, "topic": "Introduction"})
                
                # Validate each has start and topic
                for c in chunk_defs:
                    if "start" not in c:
                        c["start"] = 0
                    if "topic" not in c:
                        c["topic"] = "General"
                
                return chunk_defs
            
            # Fallback to uniform chunking
            return self._uniform_transitions_with_topics(elements)

        except Exception as e:
            print(f"⚠️  Topic detection failed: {e}. Using uniform chunking.")
            return self._uniform_transitions_with_topics(elements)

    def _uniform_transitions_with_topics(self, elements: List[Dict]) -> List[Dict]:
        """
        Fallback: Create uniform chunk boundaries with generic topics.
        """
        chunk_defs = [{"start": 0, "topic": "General"}]
        token_count = 0
        target_tokens = self.max_chunk_tokens

        for e in elements:
            token_count += self._count_tokens(e["content"])
            if token_count >= target_tokens:
                chunk_defs.append({"start": e["index"], "topic": "General"})
                token_count = 0

        return chunk_defs

    def _split_with_topics(
        self,
        elements: List[Dict],
        chunk_defs: List[Dict],
        metadata: DocumentMetadata
    ) -> List[Chunk]:
        """
        Split elements into chunks using predefined chunk definitions with topics.
        """
        if not chunk_defs:
            chunk_defs = [{"start": 0, "topic": "General"}]

        chunks = []

        for i, chunk_def in enumerate(chunk_defs):
            start_idx = chunk_def["start"]
            topic = chunk_def.get("topic", "General")
            
            # Find end index (start of next chunk, or end of elements)
            if i + 1 < len(chunk_defs):
                end_idx = chunk_defs[i + 1]["start"]
            else:
                end_idx = len(elements)

            # Collect elements for this chunk
            chunk_elements = [
                e for e in elements
                if start_idx <= e["index"] < end_idx
            ]

            if not chunk_elements:
                continue

            content = "\n".join(e["content"] for e in chunk_elements)
            element_ids = [e["index"] for e in chunk_elements]
            pages = [e["page"] for e in chunk_elements]

            chunk = Chunk(
                chunk_id="",

                content=content,
                chunk_type="topic_section",

                page_start=min(pages) if pages else 1,
                page_end=max(pages) if pages else 1,

                topic=topic,

                parent_chunk_id="",  # No parent for original chunks

                filename=metadata.filename,
                file_type=metadata.file_type,

                course_name=metadata.course_name,
                chapter_title=metadata.chapter_title,
                subject_area=metadata.subject_area,
                global_context=metadata.global_context,
                key_topics=metadata.key_topics,

                extraction_confidence=metadata.extraction_confidence,

                token_count=self._count_tokens(content),

                element_ids=element_ids
            )
            chunks.append(chunk)

        return chunks

    # ------------------------------------------------------------------
    # Topic Assignment (shared)
    # ------------------------------------------------------------------

    def _assign_topics_batch(
        self,
        samples: List[str],
        metadata: DocumentMetadata
    ) -> List[str]:
        """
        Assign topics to multiple chunks in one DeepSeek call.
        """
        if not samples:
            return []

        # Build context with only known metadata
        context_parts = []
        if metadata.course_name and metadata.course_name != "Unknown":
            context_parts.append(f"Course: {metadata.course_name}")
        if metadata.chapter_title and metadata.chapter_title != "Unknown":
            context_parts.append(f"Chapter: {metadata.chapter_title}")
        if metadata.subject_area and metadata.subject_area != "Unknown":
            context_parts.append(f"Subject: {metadata.subject_area}")
        if metadata.filename:
            context_parts.append(f"Document: {metadata.filename}")
        
        context_text = "\n".join(context_parts) if context_parts else ""

        # Build prompt with numbered samples
        samples_text = ""
        for i, sample in enumerate(samples):
            samples_text += f"\n[Chunk {i}]: {sample[:300]}\n"

        prompt = f"""
{context_text}

Assign a concise topic label to each chunk.

For each chunk below, return a concise topic that captures its main subject.

Return ONLY a JSON array of strings, same order as chunks.
Example: ["Introduction to Algorithms", "Sorting Methods", "Complexity Analysis"]
Do NOT use markdown or code blocks.

{samples_text}
"""

        try:
            response_text = simple_generate(prompt).strip()

            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            topics = json.loads(response_text)

            # Ensure same length
            while len(topics) < len(samples):
                topics.append("General")
            return topics[:len(samples)]

        except Exception as e:
            print(f"⚠️  Batch topic assignment failed: {e}")
            return ["General"] * len(samples)


# ============================================================================
# Step 4 Validation
# ============================================================================

if __name__ == "__main__":
    import os
    from src.ingestion.parser import DocumentParser
    from src.ingestion.cleaner import TextCleaner
    from src.ingestion.metadata_extractor import MetadataExtractor

    parser = DocumentParser()
    cleaner = TextCleaner()
    extractor = MetadataExtractor()
    chunker = SemanticChunker()

    # Bundled sample corpus file — swap in your own files to test other formats.
    SAMPLE_FILE = str(Path(__file__).resolve().parents[2] / "sample_data" / "sample_lecture.pptx")
    test_files = [
        SAMPLE_FILE
    ]

    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n{'='*60}")
            print(f"Chunking: {file_path}")
            print(f"{'='*60}")

            # Steps 1-3
            parsed = parser.parse(file_path)
            cleaned = cleaner.clean(parsed)
            metadata = extractor.extract_with_retry(cleaned)

            # Step 4
            chunks = chunker.chunk(cleaned, metadata)

            print(f"\n📦 Created {len(chunks)} semantic chunks:")
            print(f"   Max tokens per chunk: {chunker.max_chunk_tokens}")
            print(f"   Min tokens per chunk: {chunker.min_chunk_tokens}")
            
            # Statistics
            total_tokens = sum(chunk.token_count for chunk in chunks)
            avg_tokens = total_tokens / len(chunks) if chunks else 0
            
            print(f"\n📊 Token Statistics:")
            print(f"   Total tokens: {total_tokens}")
            print(f"   Average per chunk: {avg_tokens:.1f}")
            print(f"   Min: {min(c.token_count for c in chunks) if chunks else 0}")
            print(f"   Max: {max(c.token_count for c in chunks) if chunks else 0}")
            
            # Group chunks by parent to show splitting
            parent_groups = {}
            for chunk in chunks:
                parent = chunk.parent_chunk_id if chunk.parent_chunk_id else chunk.chunk_id
                if parent not in parent_groups:
                    parent_groups[parent] = []
                parent_groups[parent].append(chunk)
            
            print(f"\n📄 Chunk Details:")
            for chunk in chunks:
                page_range = (
                    f"Slide {chunk.page_start}"
                    if chunk.page_start == chunk.page_end
                    else f"Slides {chunk.page_start}-{chunk.page_end}"
                )
                
                # Show if this is a split chunk
                split_info = ""
                if chunk.parent_chunk_id:
                    split_info = f" (part of {chunk.parent_chunk_id})"
                
                print(f"\n   [{chunk.chunk_type}] {chunk.chunk_id}{split_info}")
                print(f"   Topic: {chunk.topic}")
                print(f"   {page_range} | {chunk.token_count} tokens | {len(chunk.content)} chars")
                print(f"   Course: {chunk.course_name}")
                print(f"   Chapter: {chunk.chapter_title}")
                print(f"   Confidence: {chunk.extraction_confidence:.2f}")
                print(f"   Preview: {chunk.content[:100]}...")
        else:
            print(f"⚠️  Test file not found: {file_path}")