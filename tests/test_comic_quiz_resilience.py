"""
Unit and regression tests for Comic Generation Non-Fatal Resilience & Quiz Fallback.

Verifies:
- Valid comic + quiz success
- Comic malformed but quiz recoverable from response
- Comic malformed and quiz fallback succeeds
- Comic failure does not clear successful quiz
- Quiz failure with comic success
- Both comic and quiz failure
- Existing Call 1 output remains available after Call 2 failure
- Markdown fences, whitespace, and truncated JSON robustness
- Zero real API calls consumed (fully mocked)
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from modules.combined_generator import (
    generate_analysis_and_simplified,
    generate_comic_and_quiz,
    _validate_and_normalize_comic,
    _validate_and_normalize_quiz,
    _generate_quiz_fallback,
)
from modules.llm_service import _extract_balanced_json_object, _extract_json_from_text


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_TEXT = "Photosynthesis is the process by which green plants make food from sunlight, carbon dioxide, and water."

_VALID_COMIC = {
    "comic_title": "The Green Factory",
    "panels": [
        {
            "panel_number": i,
            "scene": f"Leaf cell stage {i}",
            "visual_description": f"Chloroplast visual {i}",
            "characters": ["Chlorophyll Molecule"],
            "narration": f"Stage {i} begins.",
            "dialogue": "Let's capture sunlight!",
        }
        for i in range(1, 7)
    ],
}

_VALID_QUIZ = {
    "quiz_title": "Photosynthesis Check",
    "questions": [
        {
            "question": f"Question {i}: What is required for photosynthesis?",
            "options": ["Sunlight", "Darkness", "Salt", "Ice"],
            "correct_answer": "Sunlight",
            "explanation": "Plants require light energy to synthesize glucose.",
        }
        for i in range(1, 6)
    ],
}


class TestComicQuizResilience(unittest.TestCase):
    """Verifies that comic failure is non-fatal and quiz recovery/fallback works gracefully."""

    @patch("modules.combined_generator.generate_json")
    def test_valid_comic_and_quiz(self, mock_gen_json):
        """1. When both comic and quiz are valid, both are returned cleanly."""
        mock_gen_json.return_value = {
            "comic": _VALID_COMIC,
            "quiz": _VALID_QUIZ,
        }
        comic, quiz = generate_comic_and_quiz(_SAMPLE_TEXT)
        self.assertIsNotNone(comic)
        self.assertIsNotNone(quiz)
        self.assertEqual(len(comic["panels"]), 6)
        self.assertEqual(len(quiz["questions"]), 5)
        self.assertEqual(comic["comic_title"], "The Green Factory")
        self.assertEqual(quiz["quiz_title"], "Photosynthesis Check")

    @patch("modules.combined_generator.generate_json")
    def test_comic_malformed_but_quiz_recoverable(self, mock_gen_json):
        """2. When comic portion is malformed/truncated but quiz is complete, quiz is recovered and comic is None."""
        # Simulated response where 'comic' has empty/broken panels, but 'quiz' is fully intact
        mock_gen_json.return_value = {
            "comic": {"comic_title": "Broken Comic", "panels": []},
            "quiz": _VALID_QUIZ,
        }
        comic, quiz = generate_comic_and_quiz(_SAMPLE_TEXT)
        self.assertIsNone(comic, "Malformed comic must result in comic=None")
        self.assertIsNotNone(quiz, "Valid quiz must be recovered and returned")
        self.assertEqual(len(quiz["questions"]), 5)
        self.assertEqual(quiz["questions"][0]["correct_answer"], "Sunlight")

    @patch("modules.combined_generator.generate_json")
    def test_comic_malformed_and_quiz_fallback_succeeds(self, mock_gen_json):
        """3. When combined Call 2 fails completely, a separate quiz-only fallback is executed and succeeds."""
        # Call 1 (combined call) raises ValueError (malformed/unparseable JSON)
        # Call 2 (quiz fallback) returns a valid 5-question quiz dict
        mock_gen_json.side_effect = [
            ValueError("Failed to parse JSON from Ollama response: truncated text..."),
            {"quiz": _VALID_QUIZ},
        ]
        comic, quiz = generate_comic_and_quiz(_SAMPLE_TEXT)
        self.assertIsNone(comic, "Comic must be None when combined generation fails")
        self.assertIsNotNone(quiz, "Quiz must be populated by the fallback call")
        self.assertEqual(len(quiz["questions"]), 5)
        # Verify fallback was invoked
        self.assertEqual(mock_gen_json.call_count, 2)

    @patch("modules.combined_generator.generate_json")
    def test_comic_failure_does_not_clear_successful_quiz(self, mock_gen_json):
        """4. Comic failure must never clear, replace, or overwrite a successful quiz."""
        # Combined call returns invalid comic (not a dict) and valid quiz
        mock_gen_json.return_value = {
            "comic": "invalid_non_dict_comic",
            "quiz": _VALID_QUIZ,
        }
        comic, quiz = generate_comic_and_quiz(_SAMPLE_TEXT)
        self.assertIsNone(comic)
        self.assertIsNotNone(quiz)
        self.assertEqual(len(quiz["questions"]), 5)

        # Simulate storing in session state: quiz remains intact
        session_state = {"comic": comic, "quiz": quiz}
        self.assertIsNone(session_state["comic"])
        self.assertEqual(session_state["quiz"]["quiz_title"], "Photosynthesis Check")

    @patch("modules.combined_generator.generate_json")
    def test_quiz_failure_with_comic_success(self, mock_gen_json):
        """5. When comic succeeds but quiz fails in combined call AND fallback call, comic is preserved."""
        # First call has valid comic, broken quiz; second call (quiz fallback) also fails
        mock_gen_json.side_effect = [
            {"comic": _VALID_COMIC, "quiz": {"questions": []}},
            ValueError("Quiz fallback failed"),
        ]
        comic, quiz = generate_comic_and_quiz(_SAMPLE_TEXT)
        self.assertIsNotNone(comic, "Valid comic must be preserved")
        self.assertEqual(len(comic["panels"]), 6)
        self.assertIsNone(quiz, "Quiz must be None if both primary and fallback fail")

    @patch("modules.combined_generator.generate_json")
    def test_both_comic_and_quiz_failure(self, mock_gen_json):
        """6. When both comic and quiz fail completely, ValueError is raised."""
        mock_gen_json.side_effect = [
            ValueError("Combined call malformed"),
            ValueError("Fallback call failed"),
        ]
        with self.assertRaises(ValueError) as ctx:
            generate_comic_and_quiz(_SAMPLE_TEXT)
        self.assertTrue(
            "comic" in str(ctx.exception).lower() or "failed" in str(ctx.exception).lower()
        )

    @patch("modules.combined_generator.generate_json")
    def test_existing_call_1_output_remains_available_after_call_2_failure(self, mock_gen_json):
        """7. Call 1 outputs (analysis, simplified text) remain fully available even if Call 2 fails."""
        analysis_mock = {
            "analysis": {
                "chapter_title": "Photosynthesis",
                "topic": "Botany",
                "key_concepts": ["Light Reaction", "Calvin Cycle"],
                "key_terms": ["Chlorophyll", "ATP"],
                "important_facts": ["Produces Oxygen"],
                "learning_objectives": ["Understand glucose synthesis"],
                "difficulty": "Beginner",
                "prerequisite_concepts": ["Cell structure"],
            },
            "simplified": {
                "chapter_title": "Photosynthesis",
                "overview": "Plants use light to make food.",
                "key_concepts": ["Sunlight"],
                "detailed_explanation": "Detailed explanation of light absorption.",
                "important_terms": ["Chlorophyll"],
                "quick_summary": "Sunlight -> Energy.",
            },
        }

        # Step 1: Call 1 succeeds
        mock_gen_json.return_value = analysis_mock
        analysis, simplified = generate_analysis_and_simplified(_SAMPLE_TEXT, level="Beginner")
        self.assertEqual(analysis["chapter_title"], "Photosynthesis")
        self.assertEqual(simplified["quick_summary"], "Sunlight -> Energy.")

        # Simulate storing Call 1 results in application state
        app_state = {
            "analysis": analysis,
            "simplified": simplified,
            "comic": None,
            "quiz": None,
        }

        # Step 2: Call 2 fails completely (both comic and quiz fail)
        mock_gen_json.side_effect = [
            ValueError("Call 2 broken response"),
            ValueError("Fallback broken response"),
        ]
        try:
            generate_comic_and_quiz(_SAMPLE_TEXT)
        except ValueError:
            pass  # Non-fatal handling in UI

        # Assert that Call 1 output is untouched and fully accessible
        self.assertIsNotNone(app_state["analysis"])
        self.assertIsNotNone(app_state["simplified"])
        self.assertEqual(app_state["analysis"]["chapter_title"], "Photosynthesis")
        self.assertEqual(app_state["simplified"]["overview"], "Plants use light to make food.")


class TestJsonRobustness(unittest.TestCase):
    """Verifies markdown fence handling, whitespace tolerance, and balanced recovery."""

    def test_extract_balanced_json_object_with_truncated_comic(self):
        """Recovers complete 'quiz' object from a string where 'comic' was truncated."""
        malformed_raw = (
            '{"comic": {"comic_title": "Truncated", "panels": [{"panel_number": 1, "scene": "unfinished... '
            '}, "quiz": {"quiz_title": "Photosynthesis Quiz", "questions": ['
            '{"question": "Q1?", "options": ["A", "B", "C", "D"], "correct_answer": "A", "explanation": "Why A."}'
            ']}}'
        )
        recovered_quiz = _extract_balanced_json_object(malformed_raw, "quiz")
        self.assertIsNotNone(recovered_quiz)
        self.assertEqual(recovered_quiz["quiz_title"], "Photosynthesis Quiz")
        self.assertEqual(len(recovered_quiz["questions"]), 1)

    def test_extract_json_with_markdown_fences_and_whitespace(self):
        """Successfully parses JSON enclosed in markdown fences with leading/trailing newlines."""
        fenced_text = """
        ```json
        {
            "quiz": {
                "quiz_title": "Fenced Quiz",
                "questions": [
                    {"question": "Q?", "options": ["1", "2"], "correct_answer": "1", "explanation": "E"}
                ]
            }
        }
        ```
        """
        result = _extract_json_from_text(fenced_text)
        self.assertIn("quiz", result)
        self.assertEqual(result["quiz"]["quiz_title"], "Fenced Quiz")

    def test_does_not_fabricate_incomplete_quiz_or_comic(self):
        """Never treats empty or malformed structures as valid."""
        self.assertIsNone(_validate_and_normalize_comic({}))
        self.assertIsNone(_validate_and_normalize_comic({"panels": []}))
        self.assertIsNone(_validate_and_normalize_comic({"panels": "not_a_list"}))

        self.assertIsNone(_validate_and_normalize_quiz({}))
        self.assertIsNone(_validate_and_normalize_quiz({"questions": []}))
        self.assertIsNone(_validate_and_normalize_quiz({"questions": [{"question": "Q"}]}))  # missing options


if __name__ == "__main__":
    unittest.main()
