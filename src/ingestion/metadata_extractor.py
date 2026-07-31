"""
Step 3: Metadata Extraction
Uses DeepSeek to extract course name, chapter title, and global context.
Processed once per document.
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv

from src.common.llm_client import simple_generate
from src.ingestion.cleaner import CleanedDocument

# Load environment variables
load_dotenv()


@dataclass
class DocumentMetadata:
    """Structured metadata extracted from a document."""

    filename: str
    file_type: str

    course_name: str
    chapter_title: str
    global_context: str

    subject_area: str = ""
    key_topics: List[str] = field(default_factory=list)
    extraction_confidence: float = 0.0


class MetadataExtractor:
    """
    Extracts high-level metadata from cleaned documents using DeepSeek.
    Builds a structured sample from document elements.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the metadata extractor.

        Args:
            api_key: DeepSeek API key. If None, loads from DEEPSEEK_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key is required. Set DEEPSEEK_API_KEY in .env or pass it directly."
            )

    def extract(self, cleaned_doc: CleanedDocument) -> DocumentMetadata:
        """
        Extract metadata from a cleaned document.

        Args:
            cleaned_doc: CleanedDocument from Step 2

        Returns:
            DocumentMetadata with course, chapter, and context
        """
        # Prepare text sample (first 3000 chars + filename hint)
        text_sample = self._prepare_sample(cleaned_doc)

        # Call DeepSeek
        response = self._call_llm(
            text_sample,
            cleaned_doc.filename
        )

        # Parse and validate response
        metadata = self._parse_response(
            response,
            cleaned_doc.filename,
            cleaned_doc.file_type
        )

        return metadata

    def _prepare_sample(self, cleaned_doc: CleanedDocument) -> str:
        """
        Build a representative sample while preserving page/slide boundaries.
        """

        sample_parts = []

        current_page = None

        for element in cleaned_doc.elements:
            if element.page_number != current_page:
                current_page = element.page_number

                location = (
                    "Slide"
                    if cleaned_doc.file_type == "pptx"
                    else "Page"
                )

                sample_parts.append(f"\n[{location} {current_page}]")

            sample_parts.append(element.content)

        text = "\n".join(sample_parts)

        return text[:4000]

    def _call_llm(self,
                 text_sample: str,
                 filename: str) -> Dict[str, Any]:
        """
        Call DeepSeek with a structured prompt for metadata extraction.
        """
        prompt = f"""
You are an academic document analyzer. Analyze the following document text and extract:

1. **course_name**: The course or subject this document belongs to (e.g., "Introduction to Psychology", "Data Structures and Algorithms")
2. **chapter_title**: The specific chapter or topic title (e.g., "Chapter 5: Memory Systems", "Binary Trees")
3. **global_context**: A 2-3 sentence summary of what this document covers overall
4. **subject_area**: The broad academic discipline (e.g., "Computer Science", "Biology", "Business")
5. **key_topics**: A list of 3-7 main topics covered in this document

Rules:
- If you cannot determine something with confidence, use "Unknown" for text fields or empty list for key_topics
- Base your analysis ONLY on the provided text
- Return ONLY a valid JSON object.
- Do NOT wrap it in markdown.
- Do NOT use ```json.
- Do NOT explain your reasoning.
- Do NOT output any text before or after the JSON.

Return this exact JSON structure:
{{
  "course_name": "string",
  "chapter_title": "string", 
  "global_context": "string",
  "subject_area": "string",
  "key_topics": ["topic1", "topic2", ...],
  "confidence": 0.0
}}

Filename:
{filename}

Document text:
{text_sample}
"""
        response_text = ""
        try:

            response_text = simple_generate(prompt).strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            return json.loads(response_text)

        except json.JSONDecodeError as e:
            # Fallback: try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Failed to parse DeepSeek response as JSON: {response_text[:200]}") from e

        except Exception as e:
            raise RuntimeError(f"DeepSeek API call failed: {str(e)}") from e

    def _parse_response(
        self,
        response: Dict[str, Any],
        filename: str,
        file_type: str
    ) -> DocumentMetadata:
        """
        Parse and validate the DeepSeek response into a DocumentMetadata object.
        """
        topics = response.get("key_topics", [])

        if not isinstance(topics, list):
            topics = []

        try:
            confidence = float(response.get("confidence", 0.0))
        except (ValueError, TypeError):
            confidence = 0.0

        metadata = DocumentMetadata(
            filename=filename,
            file_type=file_type,      # will be supplied below

            course_name=response.get("course_name", "Unknown"),
            chapter_title=response.get("chapter_title", "Unknown"),
            global_context=response.get(
                "global_context",
                "No context available."
            ),
            subject_area=response.get(
                "subject_area",
                "Unknown"
            ),
            key_topics=topics,
            extraction_confidence=confidence
        )

        return metadata

    def extract_with_retry(self, cleaned_doc: CleanedDocument, max_retries: int = 2) -> DocumentMetadata:
        """
        Extract metadata with automatic retry on failure.
        """
        for attempt in range(max_retries + 1):
            try:
                return self.extract(cleaned_doc)
            except Exception as e:
                if attempt == max_retries:
                    # Return fallback metadata on final failure
                    return DocumentMetadata(
                        filename=cleaned_doc.filename,
                        file_type=cleaned_doc.file_type,
                        course_name="Unknown",
                        chapter_title="Unknown",
                        global_context=f"Document: {cleaned_doc.filename}",
                        subject_area="Unknown",
                        key_topics=[],
                        extraction_confidence=0.0
                    )
                print(f"⚠️  Attempt {attempt + 1} failed: {e}. Retrying...")


# ============================================================================
# Step 3 Validation
# ============================================================================

if __name__ == "__main__":
    import os
    from pathlib import Path
    from src.ingestion.parser import DocumentParser
    from src.ingestion.cleaner import TextCleaner

    # Initialize pipeline
    parser = DocumentParser()
    cleaner = TextCleaner()
    extractor = MetadataExtractor()

    # Bundled sample corpus file — swap in your own files to test other formats.
    SAMPLE_FILE = str(Path(__file__).resolve().parents[2] / "sample_data" / "sample_lecture.pptx")
    test_files = [
        SAMPLE_FILE
    ]

    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n{'='*60}")
            print(f"Extracting Metadata: {file_path}")
            print(f"{'='*60}")

            # Steps 1 & 2
            parsed = parser.parse(file_path)
            cleaned = cleaner.clean(parsed)

            # Step 3
            metadata = extractor.extract_with_retry(cleaned)

            print(f"\n📚 Extracted Metadata:")
            print(f"   Course:        {metadata.course_name}")
            print(f"   Chapter:       {metadata.chapter_title}")
            print(f"   Subject Area:  {metadata.subject_area}")
            print(f"   Key Topics:    {', '.join(metadata.key_topics) if metadata.key_topics else 'None'}")
            print(f"   Confidence:    {metadata.extraction_confidence:.2f}")
            print(f"\n📝 Global Context:")
            print(f"   {metadata.global_context}")
        else:
            print(f"⚠️  Test file not found: {file_path}")