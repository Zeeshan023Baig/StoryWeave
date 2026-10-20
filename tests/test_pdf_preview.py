"""
Tests for the PDF Preview feature.

Verifies:
1. PDF extraction succeeds and returns correct metadata.
2. Extracted text is accessible for display (at least 3000-5000 chars).
3. Empty / scanned PDF is handled cleanly with an error message.
4. PDF upload + preview does NOT call any Gemini function
   (generate_json / generate_text).
5. pdf_processor reuses existing PyMuPDF extractor, not a second implementation.

All Gemini API calls are strictly mocked -- zero real API quota consumed.
"""

import io
import unittest
from unittest.mock import patch, MagicMock


class TestPdfPreviewExtraction(unittest.TestCase):
    """
    Tests for modules.pdf_processor.extract_text_from_pdf.
    No real file I/O -- uses in-memory PDF bytes via fitz (PyMuPDF) mock.
    """

    @patch("modules.pdf_processor.fitz.open")
    def test_successful_extraction_returns_metadata(self, mock_fitz_open):
        """A PDF with readable text returns page_count, word_count, char_count, and text."""
        sample_text = "Photosynthesis is the process by which plants produce food using sunlight."

        mock_page = MagicMock()
        mock_page.get_text.return_value = sample_text
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.load_page.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        stream = MagicMock()
        stream.read.return_value = b"%PDF-1.4 fake"

        from modules.pdf_processor import extract_text_from_pdf
        result = extract_text_from_pdf(stream)

        self.assertIsNotNone(result)
        self.assertIsNone(result.get("error"))
        self.assertEqual(result["page_count"], 1)
        self.assertGreater(result["word_count"], 0)
        self.assertGreater(result["char_count"], 0)
        self.assertIn("Photosynthesis", result["text"])

    @patch("modules.pdf_processor.fitz.open")
    def test_multipage_extraction_concatenates_text(self, mock_fitz_open):
        """A multi-page PDF returns text from all pages concatenated."""
        page_texts = [
            "Chapter 1: Introduction to Biology.",
            "Chapter 2: Cell structure and function.",
            "Chapter 3: DNA and genetics.",
        ]

        def load_page(i):
            p = MagicMock()
            p.get_text.return_value = page_texts[i]
            return p

        mock_doc = MagicMock()
        mock_doc.page_count = 3
        mock_doc.load_page.side_effect = load_page
        mock_fitz_open.return_value = mock_doc

        stream = MagicMock()
        stream.read.return_value = b"%PDF-1.4 fake"

        from modules.pdf_processor import extract_text_from_pdf
        result = extract_text_from_pdf(stream)

        self.assertIsNone(result.get("error"))
        self.assertEqual(result["page_count"], 3)
        for snippet in ["Chapter 1", "Chapter 2", "Chapter 3"]:
            self.assertIn(snippet, result["text"])

    @patch("modules.pdf_processor.fitz.open")
    def test_large_pdf_text_is_fully_accessible(self, mock_fitz_open):
        """Extracted text from a large PDF must be at least 3000 characters for preview."""
        big_text = "Biology is fascinating. " * 200   # ~4800 chars

        mock_page = MagicMock()
        mock_page.get_text.return_value = big_text
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.load_page.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        stream = MagicMock()
        stream.read.return_value = b"%PDF-1.4 fake"

        from modules.pdf_processor import extract_text_from_pdf
        result = extract_text_from_pdf(stream)

        self.assertIsNone(result.get("error"))
        self.assertGreaterEqual(len(result["text"]), 3000,
            "Extracted text must be >= 3000 characters for preview display")

    @patch("modules.pdf_processor.fitz.open")
    def test_image_only_pdf_returns_error_message(self, mock_fitz_open):
        """A scanned/image-only PDF must return an error field and empty text."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = ""
        mock_doc = MagicMock()
        mock_doc.page_count = 5
        mock_doc.load_page.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        stream = MagicMock()
        stream.read.return_value = b"%PDF-1.4 fake"

        from modules.pdf_processor import extract_text_from_pdf
        result = extract_text_from_pdf(stream)

        self.assertIsNotNone(result)
        self.assertEqual(result["text"], "")
        self.assertEqual(result["word_count"], 0)
        self.assertIsNotNone(result.get("error"), "image-only PDF must return an error message")
        self.assertIn("image", result["error"].lower())

    @patch("modules.pdf_processor.fitz.open")
    def test_zero_page_pdf_returns_error(self, mock_fitz_open):
        """A completely empty PDF (0 pages) must return error, not crash."""
        mock_doc = MagicMock()
        mock_doc.page_count = 0
        mock_fitz_open.return_value = mock_doc

        stream = MagicMock()
        stream.read.return_value = b"%PDF-1.4 fake"

        from modules.pdf_processor import extract_text_from_pdf
        result = extract_text_from_pdf(stream)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.get("error"))
        self.assertEqual(result["text"], "")

    def test_none_stream_returns_none(self):
        """extract_text_from_pdf(None) returns None gracefully."""
        from modules.pdf_processor import extract_text_from_pdf
        result = extract_text_from_pdf(None)
        self.assertIsNone(result)

    @patch("modules.pdf_processor.fitz.open")
    def test_word_and_char_counts_match_text(self, mock_fitz_open):
        """word_count and char_count must be consistent with the returned text."""
        sample = "The quick brown fox jumps over the lazy dog."

        mock_page = MagicMock()
        mock_page.get_text.return_value = sample
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.load_page.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        stream = MagicMock()
        stream.read.return_value = b"%PDF-1.4 fake"

        from modules.pdf_processor import extract_text_from_pdf
        result = extract_text_from_pdf(stream)

        text = result["text"]
        self.assertEqual(result["char_count"], len(text))
        self.assertEqual(result["word_count"], len(text.split()))


class TestPdfPreviewNoGeminiCalls(unittest.TestCase):
    """
    Verifies that extracting and displaying PDF text does NOT trigger
    any Gemini API calls (generate_json / generate_text / get_client).
    """

    @patch("modules.pdf_processor.fitz.open")
    @patch("modules.combined_generator.generate_json")
    @patch("modules.combined_generator.generate_text")
    def test_pdf_extraction_does_not_call_generate_json(
        self, mock_gen_text, mock_gen_json, mock_fitz_open
    ):
        """extract_text_from_pdf must never call generate_json."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Chapter content about photosynthesis."
        mock_doc = MagicMock()
        mock_doc.page_count = 2
        mock_doc.load_page.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        stream = MagicMock()
        stream.read.return_value = b"%PDF-1.4 fake"

        from modules.pdf_processor import extract_text_from_pdf
        result = extract_text_from_pdf(stream)

        self.assertIsNone(result.get("error"))
        mock_gen_json.assert_not_called()
        mock_gen_text.assert_not_called()

    @patch("modules.pdf_processor.fitz.open")
    @patch("modules.combined_generator.generate_json")
    @patch("modules.combined_generator.generate_text")
    def test_image_only_pdf_does_not_call_gemini(
        self, mock_gen_text, mock_gen_json, mock_fitz_open
    ):
        """Even when extraction yields no text (image PDF), no Gemini call is made."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = ""
        mock_doc = MagicMock()
        mock_doc.page_count = 3
        mock_doc.load_page.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        stream = MagicMock()
        stream.read.return_value = b"%PDF-1.4 fake"

        from modules.pdf_processor import extract_text_from_pdf
        result = extract_text_from_pdf(stream)

        mock_gen_json.assert_not_called()
        mock_gen_text.assert_not_called()
        self.assertIsNotNone(result.get("error"))

    @patch("modules.pdf_processor.fitz.open")
    @patch("modules.llm_service.get_client")
    def test_pdf_extraction_does_not_call_gemini_client(
        self, mock_get_client, mock_fitz_open
    ):
        """The Gemini client must not be instantiated just to extract PDF text."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Educational content about DNA replication."
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.load_page.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        stream = MagicMock()
        stream.read.return_value = b"%PDF-1.4 fake"

        from modules.pdf_processor import extract_text_from_pdf
        result = extract_text_from_pdf(stream)

        self.assertIsNone(result.get("error"))
        mock_get_client.assert_not_called()


class TestPdfPreviewStateKeys(unittest.TestCase):
    """
    Verify that the session state keys required by the PDF Preview tab
    are declared in app.py's _STATE_KEYS dictionary.
    """

    def test_pdf_preview_state_keys_exist_in_app(self):
        """The five pdf_preview_* state keys must be present in app.py _STATE_KEYS."""
        import ast, pathlib

        source = pathlib.Path("app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        state_keys_found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_STATE_KEYS":
                        if isinstance(node.value, ast.Dict):
                            for key in node.value.keys:
                                if isinstance(key, ast.Constant):
                                    state_keys_found.add(key.value)

        required = {
            "pdf_preview_filename",
            "pdf_preview_pages",
            "pdf_preview_words",
            "pdf_preview_chars",
            "pdf_preview_error",
        }
        for key in required:
            self.assertIn(
                key, state_keys_found,
                f"Session state key '{key}' missing from _STATE_KEYS in app.py"
            )

    def test_pdf_preview_tab_defined_in_app(self):
        """The PDF Preview tab variable must be present in app.py."""
        import pathlib

        source = pathlib.Path("app.py").read_text(encoding="utf-8")
        self.assertIn("tab_pdf", source, "tab_pdf variable not found in app.py")
        self.assertIn("PDF Preview", source, "'PDF Preview' tab label not found in app.py")

    def test_pdf_preview_no_gemini_import_in_pdf_processor(self):
        """modules/pdf_processor.py must not import any Gemini module."""
        import pathlib

        source = pathlib.Path("modules/pdf_processor.py").read_text(encoding="utf-8")
        self.assertNotIn("google.genai", source, "pdf_processor must not import google.genai")
        self.assertNotIn("generate_json", source, "pdf_processor must not reference generate_json")
        self.assertNotIn("generate_text", source, "pdf_processor must not reference generate_text")


if __name__ == "__main__":
    unittest.main()
