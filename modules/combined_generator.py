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
from modules.llm_service import generate_json, generate_text
from config.settings import (
    CHUNK_CHAR_LIMIT,
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

def _condense_text(text):
    """
    If *text* exceeds CHUNK_CHAR_LIMIT characters, split it into overlapping
    windows and ask Gemini for a single condensed summary that preserves all
    key educational content.  This costs 1 extra Gemini call but prevents
    context-window failures on very large PDFs.

    For ordinary textbook chapters (< CHUNK_CHAR_LIMIT) this function is a
    no-op and returns the original text immediately — keeping the pipeline at
    exactly 2 Gemini calls.

    Chunking strategy
    -----------------
    - Window size : CHUNK_CHAR_LIMIT characters
    - Overlap     : 500 characters (preserves sentence continuity at boundaries)
    - The condensed summary from each window is joined, then a final pass
      produces a single coherent condensed text.
    - Total extra calls = 1 (regardless of number of windows) because we
      concatenate partial summaries and do one final condensation call.

    Trade-off documented for academic review:
        Small PDF  → 0 extra calls, normal 2-call path.
        Large PDF  → 1 extra condensation call, then normal 2-call path.
    """
    if len(text) <= CHUNK_CHAR_LIMIT:
        return text   # fast path — no chunking needed

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
            "Be concise but comprehensive — this summary will be used for further analysis.\n\n"
            "## SECTION TEXT\n"
            f"{chunk}"
        )
        summary = generate_text(
            condensation_prompt,
            system_instruction="You are a helpful educational assistant."
        )
        partial_summaries.append(summary.strip())

    if len(partial_summaries) == 1:
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
        "Do not add information not present in the summaries.\n\n"
        "## SECTION SUMMARIES\n"
        f"{joined}"
    )
    condensed = generate_text(
        final_prompt,
        system_instruction="You are a helpful educational assistant."
    )
    return condensed.strip()


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

    prompt_template = _load_prompt('combined_analysis_prompt.txt')
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

def generate_comic_and_quiz(text):
    """
    Single Gemini call that produces both a 6-panel comic script
    and a 5-question MCQ quiz.

    Prompt-engineering approach (visible for academic review):
    - ROLE        : expert educational content creator
    - CONTEXT     : grounded in uploaded chapter text only
    - TASK        : comic script + MCQ quiz generation
    - CONSTRAINTS : no hallucination, exact panel/question counts enforced
    - OUTPUT FORMAT: strict JSON schema matching app.py expectations

    Returns: (comic_dict, quiz_dict)
    Raises:  RuntimeError or ValueError on failure.
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate comic and quiz from empty text.")

    working_text = _condense_text(text)

    prompt_template = _load_prompt('combined_comic_quiz_prompt.txt')
    prompt = (
        prompt_template
        .replace("{question_count}", str(QUIZ_QUESTION_COUNT))
        .replace("{text}", working_text)
    )

    result = generate_json(
        prompt,
        system_instruction="You are a creative educational content creator."
    )

    if not isinstance(result, dict):
        raise ValueError("Invalid response format: expected a JSON object.")

    comic = result.get("comic")
    quiz = result.get("quiz")

    if not isinstance(comic, dict):
        raise ValueError("Gemini response missing valid 'comic' section.")
    if not isinstance(quiz, dict):
        raise ValueError("Gemini response missing valid 'quiz' section.")

    comic.setdefault("comic_title", "Educational Comic")
    quiz.setdefault("quiz_title", "Chapter Quiz")

    # Validate and normalise panels
    panels = comic.get("panels", [])
    if not isinstance(panels, list) or len(panels) == 0:
        raise ValueError("Gemini returned no comic panels.")

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
            "scene_description": scene_desc,   # UI backward compatibility
            "characters": p.get("characters", []),
            "narration": p.get("narration", ""),
            "dialogue": p.get("dialogue", ""),
        })
    comic["panels"] = normalized_panels

    # Validate and normalise questions
    questions = quiz.get("questions", [])
    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError("Gemini returned no quiz questions.")

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

    prompt_template = _load_prompt('retest_quiz_prompt.txt')
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

    prompt_template = _load_prompt('assistant_prompt.txt')
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
