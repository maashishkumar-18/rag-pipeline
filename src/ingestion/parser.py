"""
Step 1: Document Parsing & Layout Analysis
Handles PDF, PPTX, DOCX files while preserving layout, structure, and slide/page boundaries.
"""

import os
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ParsedElement:
    """Represents a single parsed element from a document."""
    content: str
    element_type: str
    page_number: int
    metadata: Dict[str, Any]


@dataclass
class ParsedDocument:
    """Represents a fully parsed document with all elements and metadata."""
    filename: str
    file_type: str
    elements: List[ParsedElement] = field(default_factory=list)
    total_pages: int = 0
    raw_text: str = ""
    structure_markers: List[Dict[str, Any]] = field(default_factory=list)


class DocumentParser:
    """
    Unified parser for PDF, PPTX, and DOCX files using unstructured library.
    Preserves layout, slide/page boundaries, and hierarchical structure.
    """

    SUPPORTED_FORMATS = {".pdf", ".pptx", ".docx"}

    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a document and return a structured ParsedDocument object.
        
        Args:
            file_path: Path to the document file (PDF, PPTX, or DOCX)
            
        Returns:
            ParsedDocument with all elements, structure markers, and metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = file_path.suffix.lower()
        if file_extension not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {file_extension}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

        # Route to appropriate parser
        if file_extension == ".pdf":
            return self._parse_pdf(file_path)
        elif file_extension == ".pptx":
            return self._parse_pptx(file_path)
        elif file_extension == ".docx":
            return self._parse_docx(file_path)

    def _parse_pdf(self, file_path: Path) -> ParsedDocument:
        """Parse PDF while preserving page boundaries and layout."""
        from unstructured.partition.pdf import partition_pdf

        raw_elements = partition_pdf(
            filename=str(file_path),
            strategy="auto",  # auto-detects best strategy
        )

        return self._process_elements(
            elements=raw_elements,
            filename=file_path.name,
            file_type="pdf",
        )

    def _parse_pptx(self, file_path: Path) -> ParsedDocument:
        """Parse PowerPoint while preserving slide boundaries and layout."""
        from unstructured.partition.pptx import partition_pptx

        raw_elements = partition_pptx(
            filename=str(file_path),
            include_slide_notes=False,  # skip speaker notes
        )

        return self._process_elements(
            elements=raw_elements,
            filename=file_path.name,
            file_type="pptx",
        )

    def _parse_docx(self, file_path: Path) -> ParsedDocument:
        """Parse Word document while preserving structure."""
        from unstructured.partition.docx import partition_docx

        raw_elements = partition_docx(
            filename=str(file_path),
        )

        return self._process_elements(
            elements=raw_elements,
            filename=file_path.name,
            file_type="docx",
        )

    def _process_elements(
        self,
        elements: List,
        filename: str,
        file_type: str,
    ) -> ParsedDocument:
        """
        Process raw unstructured elements into our ParsedDocument format.
        Tracks page/slide numbers and identifies structural markers.
        """
        parsed_doc = ParsedDocument(
            filename=filename,
            file_type=file_type
        )

        full_text_parts = []

        for element in elements:
            element_text = str(element).strip()

            metadata = getattr(element, "metadata", None)

            page_number = getattr(metadata, "page_number", None)

            if page_number is None:
                page_number = 1

            if not element_text:
                continue


            # Map element category to our type
            element_type = self._map_element_type(element)


            # Create parsed element
            parsed_element = ParsedElement(
            content=element_text,
            element_type=element_type,
            page_number=page_number,
            metadata={
                "source_file": filename,
                "element_category": getattr(element, "category", "unknown"),
                "parent_id": getattr(metadata, "parent_id", None),
                "languages": getattr(metadata, "languages", None),
            }
        )

            parsed_doc.elements.append(parsed_element)
            full_text_parts.append(element_text)

        # Populate document-level metadata
        parsed_doc.total_pages = max(
            (element.page_number for element in parsed_doc.elements),
            default=0
        )
        parsed_doc.raw_text = "\n".join(full_text_parts)

        return parsed_doc

    def _map_element_type(self, element) -> str:
        """Map unstructured element category to our simplified types."""
        if not hasattr(element, 'category'):
            return "text"

        category = str(element.category)

        mapping = {
            "Title": "title",
            "NarrativeText": "text",
            "ListItem": "list_item",
            "Table": "table",
            "Image": "image_caption",
            "Header": "title",
            "Footer": "text"
        }

        return mapping.get(category, "text")


    def get_element_summary(self, parsed_doc: ParsedDocument) -> Dict[str, Any]:
        """Generate a summary of parsed document structure."""
        element_counts = {}
        for element in parsed_doc.elements:
            element_counts[element.element_type] = element_counts.get(
                element.element_type, 0
            ) + 1

        return {
            "filename": parsed_doc.filename,
            "file_type": parsed_doc.file_type,
            "total_pages": parsed_doc.total_pages,
            "total_elements": len(parsed_doc.elements),
            "element_counts": element_counts,
            "structure_markers": len(parsed_doc.structure_markers),
        }


# ============================================================================
# Step 1 Validation
# ============================================================================

if __name__ == "__main__":
    parser = DocumentParser()

    # Bundled sample corpus file — swap in your own files to test other formats.
    SAMPLE_FILE = str(Path(__file__).resolve().parents[2] / "sample_data" / "sample_lecture.pptx")
    test_files = [
        SAMPLE_FILE
        # "path/to/test.pptx",
        # "path/to/test.docx",
    ]

    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n{'='*60}")
            print(f"Parsing: {file_path}")
            print(f"{'='*60}")

            parsed = parser.parse(file_path)
            summary = parser.get_element_summary(parsed)

            print(f"✅ Successfully parsed!")
            print(f"   Pages: {summary['total_pages']}")
            location = "Slides" if parsed.file_type == "pptx" else "Pages"
            print(f"   {location}: {summary['total_pages']}")
            print(f"   Elements: {summary['total_elements']}")
            print(f"   Element types: {summary['element_counts']}")
            print(f"   Structure markers: {summary['structure_markers']}")

            # Show first 3 elements
            print(f"\n📄 First 3 elements:")
            for i, elem in enumerate(parsed.elements[:3]):
                location = "Slide" if parsed.file_type == "pptx" else "Page"
                print(
                f"[{elem.element_type}] {location} {elem.page_number}: "
                f"{elem.content[:100]}..."
            )
        else:
            print(f"⚠️  Test file not found: {file_path}")