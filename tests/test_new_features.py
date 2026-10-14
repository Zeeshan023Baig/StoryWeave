"""
Tests for the four improvements added in Phase 1 Part 2:
  1. Prompt engineering — level instructions injected, ROLE/CONTEXT/CONSTRAINTS present
  2. Beginner / Intermediate / Advanced produce different prompts
  3. Retest — excludes previous questions, uses same chapter text
  4. Large-PDF chunking — small docs skip chunking; large docs are condensed

All Gemini/LLM calls are fully mocked. Zero real API credits consumed.
"""

import unittest
from unittest.mock import patch, MagicMock, call
from config.settings import LEVEL_INSTRUCTIONS, CHUNK_CHAR_LIMIT, QUIZ_QUESTION_COUNT


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_analysis_response(level="Beginner"):
    return {
        "analysis": {
            "chapter_title": "Test Chapter",
            "topic": "Test Topic",
            "key_concepts": ["Concept A"],
            "key_terms": ["Term X"],
            "important_facts": ["Fact 1"],
            "learning_objectives": ["Objective 1"],
            "difficulty": level,
            "prerequisite_concepts": [],
        },
        "simplified": {
            "chapter_title": "Test Chapter",
            "overview": f"Overview for {level}.",
            "key_concepts": [{"concept": "Concept A", "explanation": f"Explanation for {level}."}],
            "detailed_explanation": f"Detailed explanation written for {level} learner.",
            "important_terms": [{"term": "Term X", "definition": f"Definition for {level}."}],
            "quick_summary": f"Summary for {level}.",
        },
    }


def _make_quiz_response(prefix="Q"):
    return {
        "quiz": {
            "quiz_title": "Test Quiz",
            "questions": [
                {
                    "question": f"{prefix} Question {i}: What is concept {i}?",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": "A",
                    "explanation": "Because A.",
                }
                for i in range(1, QUIZ_QUESTION_COUNT + 1)
            ],
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1 & 2 — Prompt Engineering + Level Adaptation
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptEngineering(unittest.TestCase):
    """
    Verify that prompts contain explicit prompt-engineering elements and
    that level instructions differ across Beginner / Intermediate / Advanced.
    No real Gemini calls — generate_json is mocked.
    """

    SAMPLE_TEXT = "Photosynthesis is the process by which plants produce food using sunlight."

    def _capture_prompt(self, level):
        """Run generate_analysis_and_simplified with mocked JSON and return the prompt string."""
        from modules.combined_generator import generate_analysis_and_simplified
        captured = {}

        def fake_generate_json(prompt, system_instruction=None):
            captured["prompt"] = prompt
            captured["system_instruction"] = system_instruction
            return _make_analysis_response(level)

        with patch("modules.combined_generator.generate_json", side_effect=fake_generate_json):
            generate_analysis_and_simplified(self.SAMPLE_TEXT, level=level)

        return captured["prompt"]

    # ── Prompt structure ──────────────────────────────────────────────────────

    def test_prompt_contains_role(self):
        """Prompt must explicitly declare a ROLE."""
        prompt = self._capture_prompt("Beginner")
        self.assertIn("ROLE", prompt, "Prompt must contain a ROLE section")

    def test_prompt_contains_context(self):
        """Prompt must contain a CONTEXT grounding instruction."""
        prompt = self._capture_prompt("Beginner")
        self.assertIn("CONTEXT", prompt, "Prompt must contain a CONTEXT section")

    def test_prompt_contains_task(self):
        """Prompt must contain a TASK section."""
        prompt = self._capture_prompt("Beginner")
        self.assertIn("TASK", prompt, "Prompt must contain a TASK section")

    def test_prompt_contains_constraints(self):
        """Prompt must contain CONSTRAINTS to prevent hallucination."""
        prompt = self._capture_prompt("Beginner")
        self.assertIn("CONSTRAINTS", prompt, "Prompt must contain a CONSTRAINTS section")

    def test_prompt_contains_output_format(self):
        """Prompt must contain an OUTPUT FORMAT section."""
        prompt = self._capture_prompt("Beginner")
        self.assertIn("OUTPUT FORMAT", prompt, "Prompt must contain OUTPUT FORMAT section")

    def test_prompt_contains_no_hallucination_instruction(self):
        """Prompt must instruct model not to invent information."""
        prompt = self._capture_prompt("Beginner")
        self.assertTrue(
            "Do not invent" in prompt or "not supported by" in prompt,
            "Prompt must contain explicit anti-hallucination instruction"
        )

    def test_prompt_contains_learner_level(self):
        """The learner level string must appear in the built prompt."""
        for level in ("Beginner", "Intermediate", "Advanced"):
            with self.subTest(level=level):
                prompt = self._capture_prompt(level)
                self.assertIn(level, prompt, f"Prompt must contain the level '{level}'")

    def test_prompt_contains_level_specific_instructions(self):
        """The per-level instruction text from LEVEL_INSTRUCTIONS must be in the prompt."""
        for level, instructions in LEVEL_INSTRUCTIONS.items():
            with self.subTest(level=level):
                prompt = self._capture_prompt(level)
                # Check a distinctive line from each level's instructions
                first_line = instructions.split("\n")[0]
                self.assertIn(
                    first_line, prompt,
                    f"Level instruction for '{level}' not found in prompt"
                )

    # ── Level differentiation ─────────────────────────────────────────────────

    def test_beginner_and_advanced_prompts_differ(self):
        """Beginner and Advanced must produce different prompts."""
        beginner_prompt  = self._capture_prompt("Beginner")
        advanced_prompt  = self._capture_prompt("Advanced")
        self.assertNotEqual(
            beginner_prompt, advanced_prompt,
            "Beginner and Advanced prompts must be different"
        )

    def test_all_three_levels_produce_different_prompts(self):
        """All three level prompts must be mutually distinct."""
        prompts = {level: self._capture_prompt(level) for level in LEVEL_INSTRUCTIONS}
        prompt_list = list(prompts.values())
        for i in range(len(prompt_list)):
            for j in range(i + 1, len(prompt_list)):
                self.assertNotEqual(
                    prompt_list[i], prompt_list[j],
                    "Each level must produce a unique prompt"
                )

    def test_beginner_instructions_contain_simple_vocabulary(self):
        """Beginner instructions must mention simple vocabulary."""
        instructions = LEVEL_INSTRUCTIONS["Beginner"]
        self.assertIn("simple", instructions.lower(),
                      "Beginner instructions must mention simple vocabulary")

    def test_advanced_instructions_contain_technical_terminology(self):
        """Advanced instructions must mention technical terminology."""
        instructions = LEVEL_INSTRUCTIONS["Advanced"]
        self.assertIn("technical", instructions.lower(),
                      "Advanced instructions must mention technical terminology")

    def test_level_instructions_config_has_all_three_levels(self):
        """LEVEL_INSTRUCTIONS must define Beginner, Intermediate, and Advanced."""
        for level in ("Beginner", "Intermediate", "Advanced"):
            self.assertIn(level, LEVEL_INSTRUCTIONS,
                          f"LEVEL_INSTRUCTIONS must contain '{level}'")

    def test_system_instruction_set(self):
        """generate_analysis_and_simplified must pass a system_instruction to generate_json."""
        from modules.combined_generator import generate_analysis_and_simplified
        captured = {}

        def fake_generate_json(prompt, system_instruction=None):
            captured["system_instruction"] = system_instruction
            return _make_analysis_response("Beginner")

        with patch("modules.combined_generator.generate_json", side_effect=fake_generate_json):
            generate_analysis_and_simplified(self.SAMPLE_TEXT, level="Beginner")

        self.assertIsNotNone(captured.get("system_instruction"),
                             "A system_instruction must be passed to generate_json")


# ─────────────────────────────────────────────────────────────────────────────
# 3 — Retest / New Question Generation
# ─────────────────────────────────────────────────────────────────────────────

class TestRetestGeneration(unittest.TestCase):
    """
    Verify that generate_retest_quiz:
    - passes previous question text as exclusion context into the prompt
    - uses the same chapter text
    - returns a valid quiz dict (same schema as original quiz)
    - handles empty previous-questions list gracefully
    No real Gemini calls — generate_json is mocked.
    """

    SAMPLE_TEXT = "Photosynthesis is the process by which plants produce food using sunlight."

    PREVIOUS_QUESTIONS = [
        {
            "question": "What is photosynthesis?",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "explanation": "...",
        },
        {
            "question": "Where does photosynthesis occur?",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "B",
            "explanation": "...",
        },
    ]

    def _run_retest(self, previous_questions=None, mock_response=None):
        from modules.combined_generator import generate_retest_quiz
        if mock_response is None:
            mock_response = _make_quiz_response(prefix="Retest")
        captured = {}

        def fake_generate_json(prompt, system_instruction=None):
            captured["prompt"] = prompt
            return mock_response

        pq = previous_questions if previous_questions is not None else self.PREVIOUS_QUESTIONS
        with patch("modules.combined_generator.generate_json", side_effect=fake_generate_json):
            result = generate_retest_quiz(self.SAMPLE_TEXT, previous_questions=pq)
        return result, captured.get("prompt", "")

    def test_retest_returns_quiz_dict(self):
        """generate_retest_quiz must return a dict with 'questions' key."""
        quiz, _ = self._run_retest()
        self.assertIsInstance(quiz, dict)
        self.assertIn("questions", quiz)

    def test_retest_returns_correct_question_count(self):
        """Retest quiz must have QUIZ_QUESTION_COUNT questions."""
        quiz, _ = self._run_retest()
        self.assertEqual(len(quiz["questions"]), QUIZ_QUESTION_COUNT)

    def test_retest_prompt_contains_previous_questions(self):
        """Previous question texts must appear in the retest prompt."""
        _, prompt = self._run_retest()
        self.assertIn("What is photosynthesis?", prompt,
                      "Previous question text must be in the retest prompt")
        self.assertIn("Where does photosynthesis occur?", prompt,
                      "All previous questions must be listed in the retest prompt")

    def test_retest_prompt_contains_anti_repetition_instruction(self):
        """Prompt must explicitly tell the model not to repeat previous questions."""
        _, prompt = self._run_retest()
        self.assertTrue(
            "DO NOT repeat" in prompt or "Do not repeat" in prompt,
            "Retest prompt must contain an explicit anti-repetition instruction"
        )

    def test_retest_prompt_contains_chapter_text(self):
        """The chapter text must be present in the retest prompt."""
        _, prompt = self._run_retest()
        self.assertIn(self.SAMPLE_TEXT, prompt,
                      "Retest prompt must include the chapter text")

    def test_retest_uses_same_chapter_text(self):
        """Retest must pass the original chapter text to the prompt."""
        from modules.combined_generator import generate_retest_quiz
        captured_text = {}

        def fake_generate_json(prompt, system_instruction=None):
            captured_text["prompt"] = prompt
            return _make_quiz_response(prefix="Retest")

        with patch("modules.combined_generator.generate_json", side_effect=fake_generate_json):
            generate_retest_quiz(self.SAMPLE_TEXT, self.PREVIOUS_QUESTIONS)

        self.assertIn(self.SAMPLE_TEXT, captured_text["prompt"])

    def test_retest_empty_previous_questions_still_works(self):
        """Retest must not crash if previous_questions is an empty list."""
        quiz, _ = self._run_retest(previous_questions=[])
        self.assertIsInstance(quiz, dict)
        self.assertIn("questions", quiz)

    def test_retest_question_schema_matches_original(self):
        """Retest questions must have same fields as original quiz questions."""
        quiz, _ = self._run_retest()
        for q in quiz["questions"]:
            self.assertIn("question", q)
            self.assertIn("options", q)
            self.assertIn("correct_answer", q)
            self.assertIn("explanation", q)
            self.assertEqual(len(q["options"]), 4)

    def test_retest_empty_text_raises(self):
        """generate_retest_quiz must raise ValueError on empty text."""
        from modules.combined_generator import generate_retest_quiz
        with self.assertRaises(ValueError):
            generate_retest_quiz("", self.PREVIOUS_QUESTIONS)

    def test_retest_default_title(self):
        """Retest quiz title must default to 'Retest — New Questions' if not set."""
        response = {
            "quiz": {
                "questions": [
                    {
                        "question": "Q",
                        "options": ["A", "B", "C", "D"],
                        "correct_answer": "A",
                        "explanation": "x",
                    }
                    for _ in range(QUIZ_QUESTION_COUNT)
                ]
                # no quiz_title key
            }
        }
        quiz, _ = self._run_retest(mock_response=response)
        self.assertEqual(quiz.get("quiz_title"), "Retest — New Questions")

    def test_retest_preserves_learner_level_in_prompt(self):
        """Retest quiz must inject learner level instructions into the prompt."""
        from modules.combined_generator import generate_retest_quiz
        captured = {}

        def fake_generate_json(prompt, system_instruction=None):
            captured["prompt"] = prompt
            return _make_quiz_response(prefix="Retest")

        with patch("modules.combined_generator.generate_json", side_effect=fake_generate_json):
            generate_retest_quiz(self.SAMPLE_TEXT, self.PREVIOUS_QUESTIONS, level="Advanced")

        self.assertIn("## LEARNER LEVEL\nAdvanced", captured["prompt"])


# ─────────────────────────────────────────────────────────────────────────────
# 4 — Large-PDF Chunking
# ─────────────────────────────────────────────────────────────────────────────

class TestChunking(unittest.TestCase):
    """
    Verify the _condense_text helper:
    - Small text (< CHUNK_CHAR_LIMIT) is returned unchanged with 0 extra calls.
    - Large text (> CHUNK_CHAR_LIMIT) triggers generate_text condensation calls.
    No real Gemini calls — generate_text is mocked.
    """

    def test_small_text_bypasses_chunking(self):
        """Text under the limit must be returned as-is without any generate_text call."""
        from modules.combined_generator import _condense_text
        small_text = "Short chapter text." * 10   # well under CHUNK_CHAR_LIMIT

        with patch("modules.combined_generator.generate_text") as mock_gen_text:
            result = _condense_text(small_text)

        self.assertEqual(result, small_text, "Small text must be returned unchanged")
        mock_gen_text.assert_not_called()

    def test_large_text_triggers_condensation(self):
        """Text over the limit must trigger at least one generate_text call."""
        from modules.combined_generator import _condense_text
        # Create text slightly over the limit
        large_text = "Word " * (CHUNK_CHAR_LIMIT // 5 + 100)

        with patch("modules.combined_generator.generate_text", return_value="condensed summary") as mock_gen:
            result = _condense_text(large_text)

        mock_gen.assert_called()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_exactly_at_limit_bypasses_chunking(self):
        """Text exactly at CHUNK_CHAR_LIMIT must NOT trigger chunking."""
        from modules.combined_generator import _condense_text
        at_limit = "a" * CHUNK_CHAR_LIMIT

        with patch("modules.combined_generator.generate_text") as mock_gen:
            result = _condense_text(at_limit)

        mock_gen.assert_not_called()
        self.assertEqual(result, at_limit)

    def test_one_char_over_limit_triggers_chunking(self):
        """Text one character over the limit must trigger chunking."""
        from modules.combined_generator import _condense_text
        over_limit = "a" * (CHUNK_CHAR_LIMIT + 1)

        with patch("modules.combined_generator.generate_text", return_value="condensed") as mock_gen:
            _condense_text(over_limit)

        mock_gen.assert_called()

    def test_chunking_does_not_call_generate_json(self):
        """_condense_text must only use generate_text, never generate_json."""
        from modules.combined_generator import _condense_text
        large_text = "Word " * (CHUNK_CHAR_LIMIT // 5 + 100)

        with patch("modules.combined_generator.generate_text", return_value="condensed"):
            with patch("modules.combined_generator.generate_json") as mock_json:
                _condense_text(large_text)

        mock_json.assert_not_called()

    def test_normal_pipeline_calls_remain_2_for_small_pdf(self):
        """Small PDF must still use exactly 2 generate_json calls (no extra chunking call)."""
        from modules.combined_generator import generate_analysis_and_simplified, generate_comic_and_quiz
        small_text = "Short chapter about photosynthesis."

        mock_analysis_resp = _make_analysis_response("Beginner")
        mock_comic_quiz_resp = {
            "comic": {
                "comic_title": "Comic",
                "panels": [
                    {
                        "panel_number": i, "scene": f"Scene {i}",
                        "visual_description": f"Visual {i}",
                        "characters": [], "narration": "", "dialogue": ""
                    }
                    for i in range(1, 7)
                ],
            },
            "quiz": {
                "quiz_title": "Quiz",
                "questions": [
                    {"question": f"Q{i}", "options": ["A","B","C","D"],
                     "correct_answer": "A", "explanation": "x"}
                    for i in range(1, QUIZ_QUESTION_COUNT + 1)
                ],
            },
        }

        call_count = {"n": 0}
        def fake_generate_json(prompt, system_instruction=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return mock_analysis_resp
            return mock_comic_quiz_resp

        with patch("modules.combined_generator.generate_json", side_effect=fake_generate_json):
            with patch("modules.combined_generator.generate_text") as mock_gen_text:
                generate_analysis_and_simplified(small_text, level="Beginner")
                generate_comic_and_quiz(small_text)

        self.assertEqual(call_count["n"], 2, "Small PDF must make exactly 2 generate_json calls")
        mock_gen_text.assert_not_called()

    def test_chunk_limit_is_configurable(self):
        """CHUNK_CHAR_LIMIT must be importable from config.settings and be a positive int."""
        self.assertIsInstance(CHUNK_CHAR_LIMIT, int)
        self.assertGreater(CHUNK_CHAR_LIMIT, 0)


if __name__ == "__main__":
    unittest.main()
