"""
Combined Gemini generators that merge multiple tasks into fewer API calls.

Call 1: generate_analysis_and_simplified() — analysis + adaptive simplified text
Call 2: generate_comic_and_quiz()          — 6-panel comic script + 5-question quiz

This reduces Gemini API usage from 4 calls to 2 calls per Generate Content click.
"""

import os
from modules.llm_service import generate_json


def generate_analysis_and_simplified(text, level="Beginner"):
    """
    Single Gemini call that produces both chapter analysis and
    level-adapted simplified content.

    Returns: (analysis_dict, simplified_dict)
    Raises: RuntimeError or ValueError on failure.
    """
    if not text or not text.strip():
        raise ValueError("Cannot analyze empty text.")

    prompt_path = os.path.join(
        os.path.dirname(__file__), '..', 'prompts', 'combined_analysis_prompt.txt'
    )
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    prompt = prompt_template.replace("{level}", level).replace("{text}", text)

    result = generate_json(
        prompt,
        system_instruction="You are a helpful educational AI assistant."
    )

    if not isinstance(result, dict):
        raise ValueError("Invalid response format: expected a JSON object.")

    # Validate top-level keys
    analysis = result.get("analysis")
    simplified = result.get("simplified")

    if not isinstance(analysis, dict):
        raise ValueError("Gemini response missing valid 'analysis' section.")
    if not isinstance(simplified, dict):
        raise ValueError("Gemini response missing valid 'simplified' section.")

    # Ensure required fields in analysis with fallbacks
    analysis.setdefault("chapter_title", "Chapter Overview")
    analysis.setdefault("topic", "General Topic")
    analysis.setdefault("key_concepts", [])
    analysis.setdefault("key_terms", [])
    analysis.setdefault("important_facts", [])
    analysis.setdefault("learning_objectives", [])
    analysis.setdefault("difficulty", level)
    analysis.setdefault("prerequisite_concepts", [])

    # Ensure required fields in simplified with fallbacks
    simplified.setdefault("chapter_title", analysis.get("chapter_title", "Chapter Overview"))
    simplified.setdefault("overview", "")
    simplified.setdefault("key_concepts", [])
    simplified.setdefault("detailed_explanation", "")
    simplified.setdefault("important_terms", [])
    simplified.setdefault("quick_summary", "")

    return analysis, simplified


def generate_comic_and_quiz(text):
    """
    Single Gemini call that produces both a 6-panel comic script
    and a 5-question MCQ quiz.

    Returns: (comic_dict, quiz_dict)
    Raises: RuntimeError or ValueError on failure.
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate comic and quiz from empty text.")

    prompt_path = os.path.join(
        os.path.dirname(__file__), '..', 'prompts', 'combined_comic_quiz_prompt.txt'
    )
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    prompt = prompt_template.replace("{text}", text)

    result = generate_json(
        prompt,
        system_instruction="You are a creative educational content creator."
    )

    if not isinstance(result, dict):
        raise ValueError("Invalid response format: expected a JSON object.")

    # Validate top-level keys
    comic = result.get("comic")
    quiz = result.get("quiz")

    if not isinstance(comic, dict):
        raise ValueError("Gemini response missing valid 'comic' section.")
    if not isinstance(quiz, dict):
        raise ValueError("Gemini response missing valid 'quiz' section.")

    comic.setdefault("comic_title", "Educational Comic")
    quiz.setdefault("quiz_title", "Chapter Quiz")

    # Validate and normalize panels
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

        normalized_panel = {
            "panel_number": panel_num,
            "scene": scene or scene_desc,
            "visual_description": vis_desc or scene_desc,
            "scene_description": scene_desc,  # UI compatibility
            "characters": p.get("characters", []),
            "narration": p.get("narration", ""),
            "dialogue": p.get("dialogue", "")
        }
        normalized_panels.append(normalized_panel)

    comic["panels"] = normalized_panels

    # Validate questions
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
        normalized_q = {
            "question": q.get("question", f"Question {i + 1}"),
            "options": options,
            "correct_answer": q.get("correct_answer", options[0] if options else ""),
            "explanation": q.get("explanation", "")
        }
        normalized_questions.append(normalized_q)

    quiz["questions"] = normalized_questions

    return comic, quiz
