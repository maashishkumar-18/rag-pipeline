"""
Step 2: Text Cleaning
Fixes encoding issues, standardizes formatting, and prepares text for metadata extraction.
Uses ftfy for robust text repair.
"""

import re
import ftfy
import unicodedata
from typing import List, Dict, Any
from dataclasses import dataclass, field

from src.ingestion.parser import ParsedDocument, ParsedElement


@dataclass
class CleanedDocument:
    """Represents a document after text cleaning."""
    filename: str
    file_type: str
    elements: List[ParsedElement] = field(default_factory=list)
    total_pages: int = 0
    raw_text: str = ""
    cleaned_text: str = ""
    structure_markers: List[Dict[str, Any]] = field(default_factory=list)
    cleaning_stats: Dict[str, Any] = field(default_factory=dict)


class TextCleaner:
    """
    Cleans and standardizes text extracted from documents.
    Handles encoding issues, normalizes whitespace, and fixes common OCR/text artifacts.
    """

    def __init__(self):
        self.default_config = {
            "fix_encoding": True,
            "normalize_whitespace": True,
            "remove_control_chars": True,
            "normalize_unicode": True,
            "fix_line_breaks": False,
            "ocr_document": False,
            "min_content_length": 2  # skip elements shorter than this
        }

    def clean(self, parsed_doc: ParsedDocument, config: Dict = None) -> CleanedDocument:
        """
        Clean a parsed document's text content.
        
        Args:
            parsed_doc: ParsedDocument from Step 1
            config: Optional cleaning configuration overrides
            
        Returns:
            CleanedDocument with fixed text and cleaning statistics
        """
        cfg = {**self.default_config, **(config or {})}


        cleaned_doc = CleanedDocument(
            filename=parsed_doc.filename,
            file_type=parsed_doc.file_type,
            total_pages=parsed_doc.total_pages,
            structure_markers = parsed_doc.structure_markers
        )

        stats = {
            "total_elements": len(parsed_doc.elements),
            "elements_fixed": 0,
            "elements_skipped": 0,
            "encoding_fixes": 0,
            "whitespace_fixes": 0,
            "control_chars_removed": 0
        }

        for element in parsed_doc.elements:
            original_text = element.content

            # Skip elements that are too short
            if len(original_text.strip()) < cfg["min_content_length"]:
                stats["elements_skipped"] += 1
                continue

            cleaned_text = self._clean_text(original_text, cfg, stats)

            # Update element content with cleaned text
            cleaned_element = ParsedElement(
                content=cleaned_text,
                element_type=element.element_type,
                page_number=element.page_number,
                metadata=element.metadata.copy()
            )

            cleaned_doc.elements.append(cleaned_element)

            if cleaned_text != original_text:
                stats["elements_fixed"] += 1

        # Build full cleaned text
        cleaned_doc.raw_text = parsed_doc.raw_text
        cleaned_doc.cleaned_text = "\n".join(
            elem.content for elem in cleaned_doc.elements
        )
        cleaned_doc.cleaning_stats = stats

        return cleaned_doc

    def _clean_text(self, text: str, cfg: Dict, stats: Dict) -> str:
        """
        Apply all configured cleaning steps to a single text string.
        """
        cleaned = text

        # Step 1: Fix encoding issues (mojibake, broken unicode)
        if cfg["fix_encoding"]:
            before = cleaned
            cleaned = ftfy.fix_text(cleaned)
            if cleaned != before:
                stats["encoding_fixes"] += 1

        # Step 2: Normalize unicode (fullwidth chars, compatibility chars)
        if cfg["normalize_unicode"]:
            cleaned = unicodedata.normalize("NFKC", cleaned)

        # Step 3: Remove control characters (except newlines and tabs)
        if cfg["remove_control_chars"]:
            before_len = len(cleaned)
            cleaned = self._remove_control_chars(cleaned)
            if len(cleaned) != before_len:
                stats["control_chars_removed"] += 1

        # Step 4: Normalize whitespace
        if cfg["normalize_whitespace"]:
            before = cleaned
            cleaned = self._normalize_whitespace(cleaned)
            if cleaned != before:
                stats["whitespace_fixes"] += 1

        # Step 5: Fix line breaks
        if cfg["fix_line_breaks"] and cfg["ocr_document"]:
            cleaned = self._fix_line_breaks(cleaned)

        return cleaned.strip()

    def _remove_control_chars(self, text: str) -> str:
        """
        Remove control characters while preserving newlines and tabs.
        """
        # Remove all control chars except \n and \t
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return cleaned

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace without collapsing intentional line breaks.
        - Replace multiple spaces with single space
        - Keep paragraph breaks (double newlines)
        - Remove trailing whitespace on each line
        """
        # Remove trailing whitespace per line
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)

        # Replace multiple spaces (but not newlines)
        text = re.sub(r' {2,}', ' ', text)

        # Normalize multiple newlines to max 2 (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text

    def _fix_line_breaks(self, text: str) -> str:
        """
        Fix common line break issues:
        - Remove line breaks in the middle of sentences
        - Preserve intentional paragraph breaks
        """
        # Replace single line breaks that appear mid-sentence
        # A line ending with a lowercase letter or comma likely continues
        text = re.sub(r'(?<=[a-z,])\n(?=[a-z])', ' ', text)

        return text

    def get_cleaning_report(self, cleaned_doc: CleanedDocument) -> str:
        """
        Generate a human-readable cleaning report.
        """
        stats = cleaned_doc.cleaning_stats
        total = stats["total_elements"]

        if total == 0:
            return "No elements to clean."

        fixed_pct = (stats["elements_fixed"] / total) * 100
        skipped_pct = (stats["elements_skipped"] / total) * 100

        report = f"""
        ╔══════════════════════════════════════╗
        ║       TEXT CLEANING REPORT           ║
        ╠══════════════════════════════════════╣
        ║ File: {cleaned_doc.filename:<30} ║
        ║ Total elements: {total:<20} ║
        ║ Elements fixed: {stats['elements_fixed']:<19} ║
        ║ Elements skipped (too short): {stats['elements_skipped']:<8} ║
        ║                                      ║
        ║ Fixes applied:                       ║
        ║   Encoding fixes: {stats['encoding_fixes']:<17} ║
        ║   Whitespace fixes: {stats['whitespace_fixes']:<16} ║
        ║   Control chars removed: {stats['control_chars_removed']:<12} ║
        ║                                      ║
        ║ Fixed: {fixed_pct:.1f}% of elements            ║
        ║ Skipped: {skipped_pct:.1f}% of elements          ║
        ╚══════════════════════════════════════╝
        """
        return report


# ============================================================================
# Step 2 Validation
# ============================================================================

if __name__ == "__main__":
    import os
    from pathlib import Path
    from src.ingestion.parser import DocumentParser

    # Initialize
    parser = DocumentParser()
    cleaner = TextCleaner()

    # Bundled sample corpus file — swap in your own files to test other formats.
    SAMPLE_FILE = str(Path(__file__).resolve().parents[2] / "sample_data" / "sample_lecture.pptx")
    test_files = [
        SAMPLE_FILE
    ]

    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n{'='*60}")
            print(f"Cleaning: {file_path}")
            print(f"{'='*60}")

            # Step 1: Parse
            parsed = parser.parse(file_path)

            # Step 2: Clean
            cleaned = cleaner.clean(parsed)

            # Show stats
            print(cleaner.get_cleaning_report(cleaned))

            # Show before/after samples
            print("\n📄 Before & After Samples:")
            parsed_elements = [e for e in parsed.elements if len(e.content.strip()) > 2]
            cleaned_elements = cleaned.elements

            for i in range(min(2, len(cleaned_elements))):
                original = parsed_elements[i].content[:120]
                fixed = cleaned_elements[i].content[:120]
                if original != fixed:
                    print(f"\n   BEFORE: {original}")
                    print(f"   AFTER:  {fixed}")
                else:
                    print(f"\n   UNCHANGED: {fixed}")
        else:
            print(f"⚠️  Test file not found: {file_path}")