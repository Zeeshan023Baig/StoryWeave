import os
from dotenv import load_dotenv

load_dotenv()

# ── AI Provider Selection ─────────────────────────────────────────────────────
# Set AI_PROVIDER to "ollama" to route all LLM calls to the local Ollama server.
# Set AI_PROVIDER to "gemini" to use the Google Gemini cloud API.
# Switching providers requires only changing this single constant.
AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama")  # "ollama" | "gemini"

# ── Ollama Configuration ──────────────────────────────────────────────────────
# Ollama runs locally — no API key required.
# Install model:  ollama pull qwen3.5:2b
# Start server:   ollama serve   (or it starts automatically)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
OLLAMA_CHUNK_CHAR_LIMIT = int(os.getenv("OLLAMA_CHUNK_CHAR_LIMIT", "5000"))

# ── Gemini Configuration ──────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.8-flash"

# ── Thinking Configuration ────────────────────────────────────────────────────
# Gemini 3.8 Flash supports reasoning/thinking configuration (LOW, MEDIUM, HIGH).
# Setting the lowest appropriate thinking level ("LOW") reduces unnecessary
# reasoning latency and token usage while maintaining high structural quality.
THINKING_LEVEL = "LOW"

# ── API Timeout Configuration ─────────────────────────────────────────────────
# Timeout in seconds for Gemini API calls to prevent Streamlit UI from hanging.
GEMINI_TIMEOUT_SECONDS = 45

# ── PDF chunking ──────────────────────────────────────────────────────────────
# Maximum characters sent to Gemini in a single call.
# Gemini 3.8 Flash has a large context window, but sending massive PDFs
# increases latency and token consumption significantly.
# Setting a balanced limit ensures large texts are efficiently condensed
# before being passed to Gemini analysis and generation prompts.
CHUNK_CHAR_LIMIT = 15_000   # ~2 500–4 000 words — covers typical textbook chapters

# ── Quiz configuration ────────────────────────────────────────────────────────
QUIZ_QUESTION_COUNT = 5     # number of MCQ questions per quiz / retest

# ── Learner-level instructions ────────────────────────────────────────────────
# These are injected verbatim into the prompt so the model receives explicit,
# unambiguous per-level guidance.  Adding a new level only requires a new key.
LEVEL_INSTRUCTIONS = {
    "Beginner": (
        "TARGET AUDIENCE: A learner with little or no prior knowledge of this subject.\n"
        "LEVEL-SPECIFIC INSTRUCTIONS:\n"
        "- Use simple, everyday vocabulary. Avoid jargon.\n"
        "- When a technical term is unavoidable, define it immediately in plain language.\n"
        "- Write short sentences (aim for 15 words or fewer per sentence).\n"
        "- Explain concepts step by step, building from the simplest idea first.\n"
        "- Use relatable analogies or real-life comparisons where helpful.\n"
        "- Focus on understanding the core idea, not exhaustive detail.\n"
        "- Do NOT assume the reader knows prerequisite concepts."
    ),
    "Intermediate": (
        "TARGET AUDIENCE: A learner with basic familiarity with the subject.\n"
        "LEVEL-SPECIFIC INSTRUCTIONS:\n"
        "- Use standard subject-area terminology, but define important technical terms.\n"
        "- Write at a clear educational level — neither oversimplified nor highly technical.\n"
        "- Explain relationships and connections between concepts.\n"
        "- Include moderate depth: cover the 'how' and 'why', not just the 'what'.\n"
        "- Assume the reader knows basic vocabulary but not advanced mechanisms.\n"
        "- Avoid reducing important nuances to oversimplifications."
    ),
    "Advanced": (
        "TARGET AUDIENCE: A learner with strong prior knowledge who wants deeper understanding.\n"
        "LEVEL-SPECIFIC INSTRUCTIONS:\n"
        "- Preserve and use precise technical terminology from the source material.\n"
        "- Do NOT simplify language unnecessarily — the reader is comfortable with jargon.\n"
        "- Explain advanced relationships, mechanisms, and interactions between concepts.\n"
        "- Discuss nuances, edge cases, and deeper conceptual detail.\n"
        "- Focus on deeper understanding, critical analysis, and interconnections.\n"
        "- Assume strong foundational knowledge — skip basic definitions."
    ),
}

# ── Audio configuration ───────────────────────────────────────────────────────
# Supported client-side playback rates (safely applied at player level)
PLAYBACK_SPEED_OPTIONS = [0.75, 1.0, 1.25, 1.5, 2.0]
