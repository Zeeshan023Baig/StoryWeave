import pymupdf as fitz  # PyMuPDF
import re

def extract_text_from_pdf(pdf_stream):
    """
    Extracts text from a PDF file stream (e.g. from st.file_uploader).
    Returns a dictionary with text, page count, word count, character count.
    Handles empty or image-only PDFs by returning empty text.
    """
    if not pdf_stream:
        return None
        
    try:
        # Read the stream into a fitz Document
        pdf_bytes = pdf_stream.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Invalid PDF file: {str(e)}")

    page_count = doc.page_count
    
    if page_count == 0:
        return {
            "text": "",
            "page_count": 0,
            "word_count": 0,
            "char_count": 0,
            "error": "Empty PDF file"
        }

    full_text = ""
    for page_num in range(page_count):
        page = doc.load_page(page_num)
        text = page.get_text()
        full_text += text + "\n"

    # ── Token-efficient cleaning ──────────────────────────────────────────────
    # 1. Rejoin words hyphenated across line breaks (e.g. "pho-\nsynthesis" -> "photosynthesis")
    normalized_text = re.sub(r'(\b\w+)-\s*\n\s*(\w+\b)', r'\1\2', full_text)

    # 2. Filter out isolated page number markers and empty lines
    cleaned_lines = []
    for line in normalized_text.splitlines():
        trimmed = re.sub(r'[ \t]+', ' ', line).strip()
        if not trimmed:
            cleaned_lines.append("")
            continue
        # Skip lone page numbers like "12", "Page 3", "Page 4 of 25"
        if re.match(r'^(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?$', trimmed, re.IGNORECASE):
            continue
        cleaned_lines.append(trimmed)

    rejoined_text = "\n".join(cleaned_lines)
    # 3. Collapse multiple consecutive empty lines to at most two (paragraph separation)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', rejoined_text).strip()
    
    if not cleaned_text:
        return {
            "text": "",
            "page_count": page_count,
            "word_count": 0,
            "char_count": 0,
            "error": "No extractable text found (might be an image-only PDF)"
        }

    word_count = len(cleaned_text.split())
    char_count = len(cleaned_text)

    return {
        "text": cleaned_text,
        "page_count": page_count,
        "word_count": word_count,
        "char_count": char_count,
        "error": None
    }
