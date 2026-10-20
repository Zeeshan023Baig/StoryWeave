"""
Combined Gemini generators — 2-call optimised pipeline.

Call 1: generate_analysis_and_simplified()
    Produces chapter analysis + level-adapted simplified content.
    Injects explicit per-level instructions from config.settings.LEVEL_INSTRUCTIONS
    so Beginner / Intermediate / Advanced produce meaningfully different output.

Call 2: generate_comic_and_quiz()
    Produces 6-panel comic script + 5-question MCQ quiz.

Retest: generate_retest_quiz()
    Produces a brand-new quiz that explicitly excludes the previous questions.

Large-PDF chunking:
    If extracted text exceeds CHUNK_CHAR_LIMIT characters the text is first
    condensed into a representative summary via a single lightweight Gemini call,
    then that summary is used for the main 2-call pipeline.  Documents under the
    limit bypass chunking entirely (the normal fast path).
"""

import os
from functools import lru_cache
from modules.llm_service import (
    generate_json,
    generate_text,
    _get_active_provider,
    _extract_balanced_json_object,
)
from config.settings import (
    CHUNK_CHAR_LIMIT,
    OLLAMA_CHUNK_CHAR_LIMIT,
    QUIZ_QUESTION_COUNT,
    LEVEL_INSTRUCTIONS,
)

# ── Prompt loading helpers ─────────────────────────────────────────────────────

def _load_prompt(filename):
    """Load a prompt template from the prompts/ directory."""
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', filename)
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


# ── Large-PDF chunking ─────────────────────────────────────────────────────────

_CONDENSED_CACHE = {}


def _condense_text_ollama(text):
    """
    Lightweight, zero-LLM-call extractive condensation for local 4B model.
    Bounds prompt size to OLLAMA_CHUNK_CHAR_LIMIT without burning local CPU cycles
    on extra LLM summarisation calls.
    """
    if len(text) <= OLLAMA_CHUNK_CHAR_LIMIT:
        return text

    limit = OLLAMA_CHUNK_CHAR_LIMIT
    head_len = int(limit * 0.50)
    mid_len = int(limit * 0.30)
    tail_len = int(limit * 0.20)

    head = text[:head_len]
    mid_start = max(head_len, (len(text) // 2) - (mid_len // 2))
    mid = text[mid_start:mid_start + mid_len]
    tail = text[-tail_len:] if tail_len > 0 else ""

    return f"{head}\n\n[...]\n\n{mid}\n\n[...]\n\n{tail}".strip()


def _condense_text(text):
    """
    Route condensation based on active provider:
    - Ollama: Fast zero-extra-call extractive condensation bounded to OLLAMA_CHUNK_CHAR_LIMIT.
    - Gemini: Multi-chunk LLM condensation using CHUNK_CHAR_LIMIT (15_000).
    Results are cached in-memory to prevent repeated work.
    """
    provider = _get_active_provider()
    cache_key = (hash(text), provider)
    if cache_key in _CONDENSED_CACHE:
        return _CONDENSED_CACHE[cache_key]

    if provider == "ollama":
        result = _condense_text_ollama(text)
        _CONDENSED_CACHE[cache_key] = result
        return result

    # ── Gemini provider path (original multi-chunk logic preserved) ───────────
    if len(text) <= CHUNK_CHAR_LIMIT:
        _CONDENSED_CACHE[cache_key] = text
        return text

    overlap = 500
    step = CHUNK_CHAR_LIMIT - overlap
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_CHAR_LIMIT, len(text))
        chunks.append(text[start:end])
        start += step

    # Summarise each chunk with a lightweight text call (not JSON)
    partial_summaries = []
    for i, chunk in enumerate(chunks):
        condensation_prompt = (
            "## ROLE\n"
            "You are a study-notes assistant.\n\n"
            "## TASK\n"
            f"Summarise the following section of a textbook chapter "
            f"(part {i + 1} of {len(chunks)}).\n"
            "Preserve ALL key concepts, important terms, facts, and educational content.\n"
            "Do not add information not present in the text.\n"
            "Be concise, direct, and structured — omit conversational filler or redundant prose.\n\n"
            "## SECTION TEXT\n"
            f"{chunk}"
        )
        summary = generate_text(
            condensation_prompt,
            system_instruction="You are a helpful educational assistant."
        )
        partial_summaries.append(summary.strip())

    if len(partial_summaries) == 1:
        _CONDENSED_CACHE[cache_key] = partial_summaries[0]
        return partial_summaries[0]

    # Final condensation — join partials into one coherent text
    joined = "\n\n".join(
        f"[Section {i + 1}]\n{s}" for i, s in enumerate(partial_summaries)
    )
    final_prompt = (
        "## ROLE\n"
        "You are a study-notes assistant.\n\n"
        "## TASK\n"
        "The following are summaries of consecutive sections of a textbook chapter.\n"
        "Merge them into ONE coherent educational text that preserves all key concepts,\n"
        "important terms, facts, and educational content from every section.\n"
        "Do not add information not present in the summaries.\n"
        "Be concise, direct, and structured — avoid conversational filler.\n\n"
        "## SECTION SUMMARIES\n"
        f"{joined}"
    )
    condensed = generate_text(
        final_prompt,
        system_instruction="You are a helpful educational assistant."
    )
    result = condensed.strip()
    _CONDENSED_CACHE[cache_key] = result
    return result


# ── Call 1: Analysis + Simplified Text ────────────────────────────────────────

def generate_analysis_and_simplified(text, level="Beginner"):
    """
    Single Gemini call that produces both chapter analysis and
    level-adapted simplified content.

    Prompt-engineering approach (visible for academic review):
    - ROLE        : expert educational AI assistant
    - CONTEXT     : grounded in uploaded chapter text only
    - LEARNER LEVEL: explicit per-level instructions from LEVEL_INSTRUCTIONS
    - TASK        : chapter analysis + adaptive rewrite
    - CONSTRAINTS : no hallucination, level-appropriate language enforced
    - OUTPUT FORMAT: strict JSON schema matching app.py expectations

    Returns: (analysis_dict, simplified_dict)
    Raises:  RuntimeError or ValueError on failure.
    """
    if not text or not text.strip():
        raise ValueError("Cannot analyze empty text.")

    # Large-PDF guard — condense first if needed
    working_text = _condense_text(text)

    # Resolve level instructions (fallback to Intermediate if unknown level passed)
    level_instructions = LEVEL_INSTRUCTIONS.get(level, LEVEL_INSTRUCTIONS["Intermediate"])

    provider = _get_active_provider()
    template_name = 'ollama_analysis_prompt.txt' if provider == "ollama" else 'combined_analysis_prompt.txt'
    prompt_template = _load_prompt(template_name)
    prompt = (
        prompt_template
        .replace("{level}", level)
        .replace("{level_instructions}", level_instructions)
        .replace("{text}", working_text)
    )

    result = generate_json(
        prompt,
        system_instruction="You are a helpful educational AI assistant."
    )

    if not isinstance(result, dict):
        raise ValueError("Invalid response format: expected a JSON object.")

    analysis = result.get("analysis")
    simplified = result.get("simplified")

    if not isinstance(analysis, dict):
        raise ValueError("Gemini response missing valid 'analysis' section.")
    if not isinstance(simplified, dict):
        raise ValueError("Gemini response missing valid 'simplified' section.")

    # Ensure required fields with fallbacks
    analysis.setdefault("chapter_title", "Chapter Overview")
    analysis.setdefault("topic", "General Topic")
    analysis.setdefault("key_concepts", [])
    analysis.setdefault("key_terms", [])
    analysis.setdefault("important_facts", [])
    analysis.setdefault("learning_objectives", [])
    analysis.setdefault("difficulty", level)
    analysis.setdefault("prerequisite_concepts", [])

    simplified.setdefault("chapter_title", analysis.get("chapter_title", "Chapter Overview"))
    simplified.setdefault("overview", "")
    simplified.setdefault("key_concepts", [])
    simplified.setdefault("detailed_explanation", "")
    simplified.setdefault("important_terms", [])
    simplified.setdefault("quick_summary", "")

    return analysis, simplified


# ── Call 2: Comic Script + Quiz ────────────────────────────────────────────────

def _validate_and_normalize_comic(comic: dict) -> dict | None:
    """Validate and normalise comic panels without fabricating data."""
    if not isinstance(comic, dict):
        return None
    panels = comic.get("panels")
    if not isinstance(panels, list) or len(panels) == 0:
        return None

    normalized_panels = []
    for i, p in enumerate(panels):
        if not isinstance(p, dict):
            continue
        panel_num = p.get("panel_number", i + 1)
        scene = p.get("scene", "")
        vis_desc = p.get("visual_description", "")
        scene_desc = p.get("scene_description", "") or vis_desc or scene

        normalized_panels.append({
            "panel_number": panel_num,
            "scene": scene or scene_desc,
            "visual_description": vis_desc or scene_desc,
            "scene_description": scene_desc,
            "characters": p.get("characters", []),
            "narration": p.get("narration", ""),
            "dialogue": p.get("dialogue", ""),
        })

    if not normalized_panels:
        return None

    return {
        "comic_title": comic.get("comic_title", "Educational Comic"),
        "panels": normalized_panels,
    }


def _validate_and_normalize_quiz(quiz: dict) -> dict | None:
    """Validate and normalise quiz questions without fabricating data."""
    if not isinstance(quiz, dict):
        return None
    questions = quiz.get("questions")
    if not isinstance(questions, list) or len(questions) == 0:
        return None

    normalized_questions = []
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            continue
        question_text = q.get("question")
        options = q.get("options")
        if not question_text or not isinstance(options, list) or len(options) < 2:
            continue
        correct_ans = q.get("correct_answer") or options[0]
        explanation = q.get("explanation", "")
        normalized_questions.append({
            "question": str(question_text),
            "options": [str(opt) for opt in options],
            "correct_answer": str(correct_ans),
            "explanation": str(explanation),
        })

    if not normalized_questions:
        return None

    return {
        "quiz_title": quiz.get("quiz_title", "Chapter Quiz"),
        "questions": normalized_questions,
    }


def _generate_quiz_fallback(working_text: str) -> dict | None:
    """
    Perform a dedicated quiz-only fallback call generating exactly 5 MCQs.
    Called when the combined Call 2 failed to produce a valid quiz.
    """
    prompt_template = _load_prompt('quiz_prompt.txt')
    prompt = prompt_template.replace("{text}", working_text)
    try:
        result = generate_json(
            prompt,
            system_instruction="You are an expert educational assessment creator. Output exactly 5 multiple choice questions in JSON."
        )
        if isinstance(result, dict):
            if "quiz" in result and isinstance(result["quiz"], dict):
                return result["quiz"]
            if "questions" in result and isinstance(result["questions"], list):
                return result
    except Exception:
        pass
    return None


def generate_comic_and_quiz(text):
    """
    Produces comic script and 5-question MCQ quiz.
    Comic generation is non-fatal; quiz generation is prioritized and recovered.

    Returns: (comic_dict | None, quiz_dict | None)
    Raises: ValueError on empty input or if both comic and quiz fail.
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate comic and quiz from empty text.")

    working_text = _condense_text(text)

    provider = _get_active_provider()
    template_name = 'ollama_comic_quiz_prompt.txt' if provider == "ollama" else 'combined_comic_quiz_prompt.txt'
    prompt_template = _load_prompt(template_name)
    prompt = (
        prompt_template
        .replace("{question_count}", str(QUIZ_QUESTION_COUNT))
        .replace("{text}", working_text)
    )

    comic = None
    quiz = None
    combined_err = None

    try:
        result = generate_json(
            prompt,
            system_instruction="You are a creative educational content creator."
        )
        if isinstance(result, dict):
            if "comic" in result and isinstance(result["comic"], dict):
                comic = _validate_and_normalize_comic(result["comic"])
            if "quiz" in result and isinstance(result["quiz"], dict):
                quiz = _validate_and_normalize_quiz(result["quiz"])
    except Exception as e:
        combined_err = e

    # 1. If quiz is not valid yet, attempt recovery from error text if present
    if quiz is None and combined_err is not None:
        recovered_quiz = _extract_balanced_json_object(str(combined_err), "quiz")
        if recovered_quiz:
            quiz = _validate_and_normalize_quiz(recovered_quiz)

    # 2. If comic is not valid yet, attempt recovery from error text if present
    if comic is None and combined_err is not None:
        recovered_comic = _extract_balanced_json_object(str(combined_err), "comic")
        if recovered_comic:
            comic = _validate_and_normalize_comic(recovered_comic)

    # 3. If quiz is STILL None, perform dedicated quiz-only fallback call
    if quiz is None:
        fallback_raw = _generate_quiz_fallback(working_text)
        if fallback_raw:
            quiz = _validate_and_normalize_quiz(fallback_raw)

    # 4. If BOTH comic and quiz failed completely, raise error
    if comic is None and quiz is None:
        if isinstance(combined_err, RuntimeError):
            raise combined_err
        if combined_err:
            raise ValueError(f"Comic and quiz generation failed: {combined_err}")
        else:
            raise ValueError("Gemini response missing valid 'comic' and 'quiz' sections.")

    return comic, quiz


# ── Retest: New Quiz Generation ────────────────────────────────────────────────

def generate_retest_quiz(text, previous_questions, level="Beginner"):
    """
    Generate a completely new quiz from the same chapter text, explicitly
    excluding the previous questions to ensure fresh coverage while preserving
    the same learner level.

    Prompt-engineering approach (visible for academic review):
    - ROLE        : expert educational assessment designer
    - CONTEXT     : grounded in uploaded chapter text only
    - LEARNER LEVEL: preserves the selected learner level
    - TASK        : new MCQ quiz avoiding previous questions
    - CONSTRAINTS : anti-repetition list injected into prompt; capped to prevent unbounded growth; no hallucination
    - OUTPUT FORMAT: same JSON schema as generate_comic_and_quiz quiz section

    Arguments:
        text               : the original extracted chapter text
        previous_questions : list of question dicts from previous quizzes
        level              : learner level (Beginner / Intermediate / Advanced)

    Returns: quiz_dict (same schema as generate_comic_and_quiz quiz output)
    Raises:  RuntimeError or ValueError on failure.
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate retest from empty text.")

    working_text = _condense_text(text)

    # Build a compact exclusion list — only question text, no full objects.
    # Deduplicate and cap to prevent prompt explosion across unlimited retests.
    if previous_questions:
        seen = set()
        unique_prev = []
        for q in previous_questions:
            q_text = q.get('question', '').strip() if isinstance(q, dict) else str(q).strip()
            if q_text and q_text not in seen:
                seen.add(q_text)
                unique_prev.append(q_text)
        max_exclusions = 25
        if len(unique_prev) > max_exclusions:
            unique_prev = unique_prev[-max_exclusions:]
        prev_q_text = "\n".join(f"- {q}" for q in unique_prev)
    else:
        prev_q_text = "(none)"

    level_str = level or "Beginner"
    level_instructions = LEVEL_INSTRUCTIONS.get(level_str, LEVEL_INSTRUCTIONS.get("Beginner", ""))

    provider = _get_active_provider()
    template_name = 'ollama_retest_prompt.txt' if provider == "ollama" else 'retest_quiz_prompt.txt'
    prompt_template = _load_prompt(template_name)
    prompt = (
        prompt_template
        .replace("{question_count}", str(QUIZ_QUESTION_COUNT))
        .replace("{previous_questions}", prev_q_text)
        .replace("{level}", level_str)
        .replace("{level_instructions}", level_instructions)
        .replace("{text}", working_text)
    )

    result = generate_json(
        prompt,
        system_instruction="You are an expert educational assessment designer."
    )

    if not isinstance(result, dict):
        raise ValueError("Invalid retest response format: expected a JSON object.")

    quiz = result.get("quiz")
    if not isinstance(quiz, dict):
        raise ValueError("Retest response missing valid 'quiz' section.")

    quiz.setdefault("quiz_title", "Retest — New Questions")

    questions = quiz.get("questions", [])
    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError("Retest returned no quiz questions.")

    normalized_questions = []
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            continue
        options = q.get("options", [])
        if not isinstance(options, list):
            options = []
        normalized_questions.append({
            "question": q.get("question", f"Question {i + 1}"),
            "options": options,
            "correct_answer": q.get("correct_answer", options[0] if options else ""),
            "explanation": q.get("explanation", ""),
        })
    quiz["questions"] = normalized_questions

    return quiz


# ── AI Learning Assistant ─────────────────────────────────────────────────────

def generate_assistant_response(text, user_prompt, level="Beginner"):
    """
    Generate an educational AI response to a user's custom instruction/question
    about the uploaded chapter, grounded in the chapter text and adapted to
    the learner's level.

    Prompt-engineering approach:
    - ROLE            : educational AI assistant
    - CONTEXT         : grounded strictly in uploaded textbook/chapter
    - LEARNER LEVEL   : explicit per-level instructions from LEVEL_INSTRUCTIONS
    - USER REQUEST    : user's custom instruction/question
    - GROUNDING RULES : primary source enforcement, no hallucination, clear refusal if absent
    - ADAPTATION RULES: level-specific vocabulary and depth constraints
    - OUTPUT          : clear educational response formatted for Streamlit display

    Arguments:
        text        : the extracted chapter text
        user_prompt : the custom user question / instruction
        level       : Beginner, Intermediate, or Advanced

    Returns: response string (markdown-formatted)
    Raises:  ValueError or RuntimeError on failure.
    """
    if not text or not text.strip():
        raise ValueError("Cannot answer questions without chapter text.")
    if not user_prompt or not user_prompt.strip():
        raise ValueError("User request cannot be empty.")

    working_text = _condense_text(text)

    level_str = level or "Beginner"
    level_instructions = LEVEL_INSTRUCTIONS.get(level_str, LEVEL_INSTRUCTIONS.get("Beginner", ""))

    provider = _get_active_provider()
    template_name = 'ollama_assistant_prompt.txt' if provider == "ollama" else 'assistant_prompt.txt'
    prompt_template = _load_prompt(template_name)
    prompt = (
        prompt_template
        .replace("{level}", level_str)
        .replace("{level_instructions}", level_instructions)
        .replace("{user_prompt}", user_prompt.strip())
        .replace("{text}", working_text)
    )

    response = generate_text(
        prompt,
        system_instruction="You are an educational AI assistant."
    )

    if not response or not response.strip():
        raise ValueError("Gemini returned an empty response.")

    return response.strip()


# ── Summary & Main Topics Extraction (Zero API Call Reuse) ────────────────────

def get_chapter_summary_and_topics(analysis=None, simplified=None):
    """
    Extract a concise summary and main topics / key concepts from existing
    chapter analysis and simplified data structures without making
    any additional Gemini API calls.

    Returns: dict with 'summary' (str) and 'main_topics' (list of str)
    """
    summary = ""
    main_topics = []

    if isinstance(simplified, dict):
        summary = (
            simplified.get("overview")
            or simplified.get("quick_summary")
            or ""
        )
        for item in simplified.get("key_concepts", []):
            if isinstance(item, dict):
                c = item.get("concept")
                if c and c not in main_topics:
                    main_topics.append(c)
            elif isinstance(item, str) and item not in main_topics:
                main_topics.append(item)

    if isinstance(analysis, dict):
        if not summary:
            summary = analysis.get("topic", "")
        for c in analysis.get("key_concepts", []):
            if isinstance(c, str) and c not in main_topics:
                main_topics.append(c)

    return {
        "summary": summary,
        "main_topics": main_topics,
    }
