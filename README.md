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
Chapter Analysis                     Adaptive Simplified Text
(title, topics, key                  (Beginner / Intermediate
concepts, difficulty,                / Advanced level)
learning objectives)
      |
      |--------------------------------------------+
      v                                            v
6-Panel Comic Script               5-Question MCQ Quiz
(scene, characters,                (auto-graded with
dialogue, narration)               explanations)
```

---

## System Architecture

```
StoryWeave/
|
|-- app.py                  <- Main Streamlit web application
|-- requirements.txt        <- Python dependencies
|-- .env                    <- API key (not committed to GitHub)
|-- .env.example            <- Template for .env setup
|
|-- config/
|   |-- settings.py         <- Centralized config (model name, API key)
|
|-- modules/                <- Core logic, one module per feature
|   |-- pdf_processor.py    <- PDF upload, text extraction (PyMuPDF)
|   |-- llm_service.py      <- Single Gemini API gateway
|   |-- content_analyzer.py <- Chapter analysis using Gemini
|   |-- simplified_text.py  <- Adaptive text generation (3 levels)
|   |-- comic_generator.py  <- 6-panel comic script generation
|   |-- quiz_generator.py   <- 5-question MCQ generation
|
|-- prompts/                <- LLM prompt templates (separate from code)
|   |-- analysis_prompt.txt
|   |-- simplified_prompt.txt
|   |-- comic_prompt.txt
|   |-- quiz_prompt.txt
|
|-- utils/
    |-- helpers.py          <- Shared utility functions
```

### Key Design Decision: Centralized LLM Gateway
All Gemini API calls go through **one single file** — `modules/llm_service.py`. This means:
- The model can be swapped in one place (`config/settings.py`)
- Error handling is centralized
- Prompts are kept separate from code logic (in `/prompts`)

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.11+** | Core language |
| **Streamlit** | Web UI and interactive dashboard |
| **PyMuPDF** | PDF parsing and text extraction |
| **Google Gemini 2.5 Flash** | LLM for all AI generation tasks |
| **python-dotenv** | Secure API key management |
| **google-genai** | Official Google GenAI Python SDK |

---

## Module-by-Module Explanation

### 1. `pdf_processor.py` — PDF Engine
Uses **PyMuPDF** to:
- Accept PDF uploads through the browser
- Extract raw text from each page
- Clean text (remove extra whitespace, noise)
- Count pages, words, and characters
- Handle edge cases: empty PDFs, image-only PDFs gracefully

### 2. `llm_service.py` — Gemini Gateway
The **single point of contact** with Google Gemini:
- `generate_text(prompt)` — for plain text output
- `generate_json(prompt)` — forces structured JSON output
- Handles API errors and malformed JSON gracefully
- Uses the new `google-genai` SDK

### 3. `content_analyzer.py` — Chapter Intelligence
Sends the extracted text to Gemini and returns structured JSON:

```json
{
  "chapter_title": "...",
  "topic": "...",
  "key_concepts": ["..."],
  "key_terms": ["..."],
  "important_facts": ["..."],
  "learning_objectives": ["..."],
  "difficulty": "Beginner / Intermediate / Advanced",
  "prerequisite_concepts": ["..."]
}
```

### 4. `simplified_text.py` — Adaptive Learning
Generates level-specific content from the same chapter:
- **Beginner** — simple words, analogies, short sentences
- **Intermediate** — standard educational language, clear definitions
- **Advanced** — technical terminology, in-depth explanations

Output includes: Overview, Key Concepts, Detailed Explanation, Important Terms, Quick Summary.

### 5. `comic_generator.py` — Story Script Engine
Transforms chapter content into an engaging **6-panel comic script**:

```json
{
  "panel_number": 1,
  "scene_description": "...",
  "characters": ["..."],
  "narration": "...",
  "dialogue": "..."
}
```

> Note: This is a text script only. AI image generation is planned for Phase 2.

### 6. `quiz_generator.py` — Assessment Engine
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
- Uses `st.form` so answers are **not revealed before submission**
- Tracks state with `st.session_state`
- Shows **Score: X / 5**, percentage, and explanation per question after submission

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- A Google Gemini API Key (free at https://aistudio.google.com/app/apikey)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Zeeshan023Baig/StoryWeave.git
cd StoryWeave

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your API key
# Create a .env file and add:
# GEMINI_API_KEY=your_key_here

# 4. Run the application
streamlit run app.py
```

---

## User Flow

1. Open the app at `http://localhost:8501`
2. **Sidebar**: Upload an educational PDF
3. **Sidebar**: Select learning level (Beginner / Intermediate / Advanced)
4. Click **"Generate Content"**
5. **Overview tab** — View chapter analysis (topic, concepts, difficulty)
6. **Simplified Learning tab** — View adaptive notes for your level
7. **Comic Story tab** — View the 6-panel educational comic script
8. **Quiz tab** — Answer 5 questions, submit, and see your score with explanations

---

## Current Features (Phase 1 MVP)

- [x] PDF upload and text extraction
- [x] Gemini-powered chapter analysis
- [x] Adaptive simplified text (3 levels)
- [x] 6-panel educational comic script
- [x] Interactive 5-question MCQ quiz with scoring
- [x] Session state-based quiz (no answer leaks before submission)
- [x] Friendly error handling throughout

---

## Future Features (Phase 2 & 3)

| Feature | Phase |
|---|---|
| AI Comic Image Generation (per panel) | Phase 2 |
| Text-to-Speech audio narration | Phase 2 |
| User Authentication | Phase 3 |
| Database (save sessions/history) | Phase 3 |
| Learning Analytics Dashboard | Phase 3 |
| Content Recommendation System | Phase 3 |
| Cloud Deployment (Streamlit Cloud / GCP) | Phase 3 |

---

## Security

- API key stored in `.env` (excluded from GitHub via `.gitignore`)
- `.env.example` provided as a safe template
- No hardcoded credentials anywhere in the codebase

---

## Author

**Zeeshan Baig**
B.Tech — Applied Generative AI
Project: StoryWeave — Multimodal Generative AI for Adaptive and Inclusive Learning

---

## License

This project is for academic purposes.
