# StoryWeave
### A Multimodal Generative AI Framework for Adaptive and Inclusive Learning

> **B.Tech Applied Generative AI — Project (Phase 1 MVP)**
> Built with Python, Streamlit, PyMuPDF, and Google Gemini 2.5 Flash

---

## Project Overview

**StoryWeave** is an AI-powered educational platform that transforms any educational PDF chapter into multiple personalized learning formats using **Generative AI**.

The core idea is simple:
> A student uploads a textbook chapter → StoryWeave automatically reads it, understands it, and generates **adaptive notes**, an **educational comic story**, and a **quiz** — all tailored to the student's learning level.

This addresses a real-world problem: **not every student learns the same way**. Some need simple language, others want technical depth. Some learn through stories, others through quizzes. StoryWeave generates all of these from a single PDF — automatically.

---

## Problem Statement

Traditional educational content:
- Is written at **one fixed difficulty level**
- Does not adapt to individual learner needs
- Fails to engage visual or story-based learners
- Provides no instant assessment or feedback

**StoryWeave solves all four problems using Generative AI.**

---

## Core Pipeline

```
Educational PDF
      |
      v
 PyMuPDF Text Extraction
      |
      v
 Text Cleaning & Chunking
      |
      v
 Google Gemini 2.5 Flash (LLM)
      |
      |--------------------------------------------+
      v                                            v
 Gemini Call 1                           Gemini Call 2
 Chapter Analysis +                      6-Panel Comic Script +
 Adaptive Simplified Text                5-Question MCQ Quiz
 (Beginner / Intermediate /              (auto-graded with
  Advanced level)                         explanations)
      |
      v
 gTTS Audio Summary (non-fatal, optional)
      |
      v
 Pixazo Flux Comic Images (non-fatal, optional)
```

> **Optimisation:** The original 4 Gemini API calls have been merged into **2 combined calls** via combined_generator.py, reducing quota usage and latency by ~50%.

---

## System Architecture

```
StoryWeave/
|
|-- app.py                      <- Main Streamlit web application
|-- requirements.txt            <- Python dependencies
|-- .env                        <- API keys (not committed to GitHub)
|
|-- config/
|   |-- settings.py             <- Centralized config (model name, API key)
|
|-- modules/                    <- Core logic, one module per feature
|   |-- pdf_processor.py        <- PDF upload, text extraction (PyMuPDF)
|   |-- llm_service.py          <- Single Gemini API gateway
|   |-- combined_generator.py   <- Merged Gemini calls (analysis+simplified, comic+quiz)
|   |-- content_analyzer.py     <- (legacy) standalone chapter analysis
|   |-- simplified_text.py      <- (legacy) standalone adaptive text generation
|   |-- comic_generator.py      <- (legacy) standalone comic script generation
|   |-- quiz_generator.py       <- (legacy) standalone quiz generation
|   |-- image_generator.py      <- Comic panel image generation (Pixazo Flux API)
|   |-- audio_generator.py      <- Text-to-Speech audio via gTTS
|
|-- prompts/                    <- LLM prompt templates (separate from code)
|   |-- combined_analysis_prompt.txt
|   |-- combined_comic_quiz_prompt.txt
|   |-- analysis_prompt.txt
|   |-- simplified_prompt.txt
|   |-- comic_prompt.txt
|   |-- quiz_prompt.txt
|
|-- utils/
    |-- helpers.py              <- Shared utility functions (currently empty)
```

### Key Design Decisions
- **Centralized LLM Gateway**: All Gemini API calls go through modules/llm_service.py. The model can be swapped in one place (config/settings.py), error handling is centralized, and prompts are kept separate from logic.
- **Combined Generators**: combined_generator.py merges 4 original Gemini calls into 2, cutting API usage in half.
- **Non-fatal Optional Features**: Audio and image generation failures show a warning but never block core content display.
- **PDF Cache**: An MD5 hash of the uploaded PDF prevents redundant Gemini calls if the same file + level is re-submitted.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.11+** | Core language |
| **Streamlit** | Web UI and interactive dashboard |
| **PyMuPDF** | PDF parsing and text extraction |
| **Google Gemini 2.5 Flash** | LLM for all AI generation tasks |
| **gTTS** | Google Text-to-Speech for audio summaries |
| **Pixazo Flux API** | AI image generation for comic panels |
| **python-dotenv** | Secure API key management |
| **google-genai** | Official Google GenAI Python SDK |
| **requests** | HTTP client for Pixazo API |

---

## What is Working

| Feature | Status | Notes |
|---|---|---|
| PDF upload & text extraction | Working | PyMuPDF handles multi-page PDFs; graceful error for image-only PDFs |
| Chapter analysis (Gemini) | Working | Returns title, topic, key concepts, difficulty, learning objectives, prerequisites |
| Adaptive simplified text (3 levels) | Working | Beginner / Intermediate / Advanced — overview, explanation, key concepts, terms, summary |
| 6-panel comic script (Gemini) | Working | Scene, characters, narration, dialogue per panel; robust field normalization |
| 5-question MCQ quiz (Gemini) | Working | Auto-graded with per-question explanations; answer locked until form submit |
| Session-state quiz (no answer leaks) | Working | Uses st.form so answers are not revealed before submission |
| PDF hash cache | Working | Same PDF + level skips Gemini calls and re-uses stored results |
| Optimised 2-call Gemini pipeline | Working | combined_generator.py replaces 4 separate calls with 2 merged calls |
| Audio narration (gTTS) | Working | Reads overview + detailed explanation aloud; plays in-browser via st.audio |
| Friendly error handling | Working | 429 quota, 503 unavailable, empty PDF, scanned PDF, missing API key all handled |

---

## What is NOT Working / Known Issues

| Feature | Status | Details |
|---|---|---|
| **Comic panel image generation** | BROKEN | Pixazo Flux API endpoint (/flux/text-to-image) returns **404 Resource not found**. PIXAZO_API_KEY must also be set in .env. Comic script always displays; images are non-fatal and are simply skipped with a consolidated warning. |
| **Audio on slow/offline connections** | UNRELIABLE | gTTS requires an active outbound internet connection to Google TTS servers. Degrades silently to a warning banner when the connection fails or times out. |
| **Image-only / scanned PDFs** | NOT SUPPORTED | PyMuPDF extracts no text from purely scanned documents. The app shows a clear error but cannot process such files. |
| **Very large PDFs** | MAY FAIL | No chunking strategy is applied before sending text to Gemini. Extremely long chapters may exceed the model context window and cause a generation failure. |
| **Legacy standalone modules** | NOT USED | content_analyzer.py, simplified_text.py, comic_generator.py, quiz_generator.py are no longer called by app.py — replaced by combined_generator.py. They remain in the repo for reference only. |
| **utils/helpers.py** | EMPTY | The file exists but contains no utility functions yet. |
| **.env.example** | MISSING | The file does not exist in the repository. New contributors must manually create a .env file. |

---

## Module-by-Module Explanation

### 1. pdf_processor.py — PDF Engine
Uses **PyMuPDF** to:
- Accept PDF uploads through the browser
- Extract raw text from each page
- Clean text (remove extra whitespace, noise)
- Count pages, words, and characters
- Handle edge cases: empty PDFs, image-only PDFs gracefully

### 2. llm_service.py — Gemini Gateway
The **single point of contact** with Google Gemini:
- generate_text(prompt) — for plain text output
- generate_json(prompt) — forces structured JSON output
- Handles API errors and malformed JSON gracefully
- Uses the new google-genai SDK

### 3. combined_generator.py — Merged API Calls (Active)
Reduces Gemini API usage from 4 calls to 2:
- generate_analysis_and_simplified(text, level) returns (analysis_dict, simplified_dict)
- generate_comic_and_quiz(text) returns (comic_dict, quiz_dict)

### 4. image_generator.py — Comic Panel Images (BROKEN — API 404)
Sends visual scene descriptions to the **Pixazo Flux API** to generate PNG images per comic panel. Currently returns 404 from the Pixazo endpoint. The app degrades gracefully — comic script is always shown; images are skipped with a single consolidated warning.

### 5. audio_generator.py — Text-to-Speech
Uses **gTTS** (Google Text-to-Speech) to convert the simplified overview + detailed explanation into an MP3 audio clip, played inline via st.audio. Requires internet access. Failure is non-fatal.

### 6. Quiz — Assessment Engine
Generates **5 MCQ questions** grounded in the uploaded chapter:

```json
{
  "question": "...",
  "options": ["A", "B", "C", "D"],
  "correct_answer": "...",
  "explanation": "..."
}
```

The Streamlit quiz interface:
- Uses st.form so answers are **not revealed before submission**
- Tracks state with st.session_state
- Shows **Score: X / 5**, percentage, and explanation per question after submission

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- A Google Gemini API Key (free at https://aistudio.google.com/app/apikey)
- (Optional) A Pixazo API Key for comic images — currently broken endpoint, images will not generate

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Zeeshan023Baig/StoryWeave.git
cd StoryWeave

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create a .env file and add your keys:
# GEMINI_API_KEY=your_gemini_key_here
# PIXAZO_API_KEY=your_pixazo_key_here   (optional — image generation currently broken)

# 4. Run the application
streamlit run app.py
```

---

## User Flow

1. Open the app at http://localhost:8501
2. **Sidebar**: Upload an educational PDF
3. **Sidebar**: Select learning level (Beginner / Intermediate / Advanced)
4. Click **Generate Content**
5. **Overview tab** — View chapter analysis (topic, concepts, difficulty)
6. **Simplified Learning tab** — View adaptive notes + listen to audio summary
7. **Comic Story tab** — View the 6-panel educational comic script (images shown if Pixazo works)
8. **Quiz tab** — Answer 5 questions, submit, and see your score with explanations

---

## Current Features (Phase 1 MVP)

- [x] PDF upload and text extraction (PyMuPDF)
- [x] User-controlled AI Learning Assistant ("Ask StoryWeave") with custom prompt input
- [x] Strict PDF grounding and anti-hallucination prompt engineering
- [x] Gemini-powered chapter analysis (concise summary + key concepts)
- [x] Adaptive simplified text (Beginner, Intermediate, Advanced)
- [x] 6-panel educational comic script
- [x] Interactive 5-question MCQ quiz with auto-grading and answer locking
- [x] Retest functionality with unlimited attempts, exclusion accumulation, and level preservation
- [x] Audio narration via gTTS with customizable playback speed (0.75x, 1.0x, 1.25x, 1.5x, 2.0x)
- [x] PDF hash-based result caching (avoids redundant Gemini calls)
- [x] Large-PDF chunking and condensation
- [x] Optimised 2-call Gemini pipeline (with 503 transient retry)
- [x] Friendly error handling throughout
- [ ] Comic panel image generation (code integrated, Pixazo API endpoint is broken — 404)

---

## Future Features (Phase 2 & 3)

| Feature | Phase |
|---|---|
| Fix/replace Pixazo image API with a working provider (e.g. Stability AI, Google Imagen) | Phase 2 |
| User Authentication | Phase 3 |
| Database (save sessions/history) | Phase 3 |
| Learning Analytics Dashboard | Phase 3 |
| Content Recommendation System | Phase 3 |
| Cloud Deployment (Streamlit Cloud / GCP) | Phase 3 |


---

## Security

- API keys stored in .env (excluded from GitHub via .gitignore)
- No hardcoded credentials anywhere in the codebase
- Create a .env file manually with GEMINI_API_KEY=... before running

---

## Author

**Zeeshan Baig**
B.Tech — Applied Generative AI
Project: StoryWeave — Multimodal Generative AI for Adaptive and Inclusive Learning

---

## License

This project is for academic purposes.
