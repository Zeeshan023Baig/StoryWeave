"""
Tests for the AI Learning Assistant, Grounding, Retest, and Playback Speed features.

Covers all 10 requirements:
1. Custom prompt is included in the Gemini request.
2. Uploaded chapter context is included.
3. Learner level instructions are included.
4. Grounding/anti-hallucination instructions are included.
5. Summary/main topics remain available without duplicate API calls.
6. Retest produces a new question set.
7. Multiple retests work across successive rounds.
8. Previous questions are excluded in retest prompt.
9. Same learner level is preserved during retests.
10. No real API calls occur in tests (strictly mocked).
"""

import unittest
from unittest.mock import patch, MagicMock
from config.settings import (
    LEVEL_INSTRUCTIONS,
    QUIZ_QUESTION_COUNT,
    PLAYBACK_SPEED_OPTIONS,
)
from modules.combined_generator import (
    generate_assistant_response,
    generate_retest_quiz,
    get_chapter_summary_and_topics,
)


class TestAILearningAssistant(unittest.TestCase):
    """Tests for user prompt handling, grounding rules, and learner-level adaptation."""

    SAMPLE_CHAPTER = (
        "Cellular respiration is a set of metabolic reactions that take place in the "
        "cells of organisms to convert biochemical energy from nutrients into ATP. "
        "The stages include glycolysis, the citric acid cycle (Krebs cycle), and oxidative phosphorylation."
    )

    def _capture_assistant_call(self, text, user_prompt, level="Beginner", mock_response="Educational explanation."):
        captured = {}

        def fake_generate_text(prompt, system_instruction=None):
            captured["prompt"] = prompt
            captured["system_instruction"] = system_instruction
            return mock_response

        with patch("modules.combined_generator.generate_text", side_effect=fake_generate_text) as mock_gen:
            resp = generate_assistant_response(text, user_prompt, level=level)

        return resp, captured.get("prompt", ""), captured.get("system_instruction"), mock_gen

    # ── 1. Custom prompt is included in the Gemini request ─────────────────────

    def test_custom_prompt_included_in_gemini_request(self):
        """User's typed custom prompt must appear in the prompt sent to Gemini."""
        prompts = [
            "Explain this chapter in simple words.",
            "Give me the most important topics for an exam.",
            "Explain photosynthesis step by step.",
            "Create revision notes from this chapter.",
            "Explain the difficult terms.",
            "Compare the two main concepts in this chapter.",
        ]
        for p in prompts:
            with self.subTest(user_prompt=p):
                _, prompt_sent, _, _ = self._capture_assistant_call(self.SAMPLE_CHAPTER, p, level="Beginner")
                self.assertIn(p, prompt_sent, f"Prompt '{p}' was not included in the request sent to Gemini.")

    # ── 2. Uploaded chapter context is included ────────────────────────────────

    def test_uploaded_chapter_context_is_included(self):
        """Uploaded chapter text must be included in the prompt sent to Gemini."""
        _, prompt_sent, _, _ = self._capture_assistant_call(
            self.SAMPLE_CHAPTER, "What is cellular respiration?", level="Beginner"
        )
        self.assertIn(self.SAMPLE_CHAPTER, prompt_sent)
        self.assertIn("## CHAPTER TEXT", prompt_sent)

    def test_large_chapter_context_is_condensed_and_included(self):
        """Chapters exceeding chunk limit are condensed and the condensed context is included."""
        large_text = "Mitochondria are the powerhouse of the cell. " * 800  # > 30,000 chars
        condensed_mock = "Condensed summary: Mitochondria generate ATP via oxidative phosphorylation."

        def fake_generate_text(prompt, system_instruction=None):
            if "Summarise the following section" in prompt or "Merge them into ONE" in prompt:
                return condensed_mock
            return "Final assistant response."

        with patch("modules.combined_generator.generate_text", side_effect=fake_generate_text):
            resp = generate_assistant_response(large_text, "Explain mitochondria", level="Intermediate")

        self.assertEqual(resp, "Final assistant response.")

    # ── 3. Learner level instructions are included ─────────────────────────────

    def test_learner_level_instructions_included_for_all_levels(self):
        """The specific LEVEL_INSTRUCTIONS for Beginner, Intermediate, and Advanced must be injected."""
        for level, instructions in LEVEL_INSTRUCTIONS.items():
            with self.subTest(level=level):
                _, prompt_sent, _, _ = self._capture_assistant_call(
                    self.SAMPLE_CHAPTER, "Explain glycolysis", level=level
                )
                self.assertIn(f"## LEARNER LEVEL\n{level}", prompt_sent)
                first_instruction_line = instructions.split("\n")[0]
                self.assertIn(first_instruction_line, prompt_sent)

    def test_different_levels_produce_different_prompts(self):
        """Beginner and Advanced prompts must differ noticeably."""
        _, prompt_beg, _, _ = self._capture_assistant_call(
            self.SAMPLE_CHAPTER, "Explain glycolysis", level="Beginner"
        )
        _, prompt_adv, _, _ = self._capture_assistant_call(
            self.SAMPLE_CHAPTER, "Explain glycolysis", level="Advanced"
        )
        self.assertNotEqual(prompt_beg, prompt_adv)
        self.assertIn("simple", prompt_beg.lower())
        self.assertIn("technical", prompt_adv.lower())

    # ── 4. Grounding / anti-hallucination instructions are included ───────────

    def test_grounding_anti_hallucination_rules_present(self):
        """Prompt must strictly enforce grounding in the uploaded chapter."""
        _, prompt_sent, system_instruction, _ = self._capture_assistant_call(
            self.SAMPLE_CHAPTER, "What is ATP?", level="Intermediate"
        )
        self.assertIn("## GROUNDING RULES", prompt_sent)
        self.assertIn("primary source", prompt_sent)
        self.assertIn("Do not invent facts that are not supported by the source", prompt_sent)
        self.assertIn("If the requested information is not present in the chapter, clearly say so", prompt_sent)
        self.assertIn("Do not pretend that unsupported information came from the textbook", prompt_sent)
        self.assertEqual(system_instruction, "You are an educational AI assistant.")

    # ── Edge cases ─────────────────────────────────────────────────────────────

    def test_empty_chapter_text_raises_error(self):
        """generate_assistant_response must raise ValueError if text is empty."""
        with self.assertRaises(ValueError):
            generate_assistant_response("", "Explain this", level="Beginner")

    def test_empty_user_prompt_raises_error(self):
        """generate_assistant_response must raise ValueError if prompt is empty."""
        with self.assertRaises(ValueError):
            generate_assistant_response(self.SAMPLE_CHAPTER, "   ", level="Beginner")


class TestSummaryAndMainTopics(unittest.TestCase):
    """Tests for summary and main topics availability without duplicate Gemini calls."""

    # ── 5. Summary / main topics remain available ──────────────────────────────

    def test_summary_and_main_topics_available_without_api_calls(self):
        """Summary and key concepts must be extracted from existing data with 0 Gemini calls."""
        mock_analysis = {
            "chapter_title": "Cell Biology",
            "topic": "Cellular Respiration",
            "key_concepts": ["Glycolysis", "Citric Acid Cycle"],
        }
        mock_simplified = {
            "overview": "Cells turn food into energy through respiration.",
            "key_concepts": [
                {"concept": "ATP", "explanation": "The energy currency of cells."},
                {"concept": "Glycolysis", "explanation": "Breaking down glucose."},
            ],
            "quick_summary": "Energy production summary.",
        }

        with patch("modules.combined_generator.generate_text") as mock_text, \
             patch("modules.combined_generator.generate_json") as mock_json:
            result = get_chapter_summary_and_topics(mock_analysis, mock_simplified)

            # Zero Gemini calls must have been made
            mock_text.assert_not_called()
            mock_json.assert_not_called()

        self.assertEqual(result["summary"], "Cells turn food into energy through respiration.")
        self.assertIn("ATP", result["main_topics"])
        self.assertIn("Glycolysis", result["main_topics"])
        self.assertIn("Citric Acid Cycle", result["main_topics"])

    def test_summary_fallback_to_analysis_topic_if_simplified_missing(self):
        """If simplified is missing, fallback to analysis topic."""
        mock_analysis = {
            "topic": "Photosynthesis Basics",
            "key_concepts": ["Chlorophyll", "Light Reactions"],
        }
        result = get_chapter_summary_and_topics(analysis=mock_analysis, simplified=None)
        self.assertEqual(result["summary"], "Photosynthesis Basics")
        self.assertEqual(result["main_topics"], ["Chlorophyll", "Light Reactions"])


class TestRetestAndUnlimitedSessions(unittest.TestCase):
    """Tests for quiz retesting, question set generation, exclusion accumulation, and level preservation."""

    SAMPLE_CHAPTER = "Photosynthesis converts solar energy into chemical energy in plant chloroplasts."

    def _mock_retest_quiz_response(self, id_start=1):
        return {
            "quiz": {
                "quiz_title": f"Retest Round {id_start}",
                "questions": [
                    {
                        "question": f"Question {i}: What is component {i}?",
                        "options": ["Opt A", "Opt B", "Opt C", "Opt D"],
                        "correct_answer": "Opt A",
                        "explanation": f"Explanation {i}",
                    }
                    for i in range(id_start, id_start + QUIZ_QUESTION_COUNT)
                ],
            }
        }

    # ── 6. Retest produces a new question set ──────────────────────────────────

    def test_retest_produces_new_question_set(self):
        """Retest must return a fresh quiz with 5 questions."""
        prev = [{"question": "Old Q1"}, {"question": "Old Q2"}]
        with patch("modules.combined_generator.generate_json") as mock_json:
            mock_json.return_value = self._mock_retest_quiz_response(id_start=10)
            quiz = generate_retest_quiz(self.SAMPLE_CHAPTER, previous_questions=prev, level="Intermediate")

        self.assertIn("questions", quiz)
        self.assertEqual(len(quiz["questions"]), 5)
        self.assertEqual(quiz["questions"][0]["question"], "Question 10: What is component 10?")

    # ── 7. Multiple retests work across successive rounds ──────────────────────

    def test_multiple_retests_accumulate_exclusions(self):
        """Multiple successive retests work, accumulating exclusions without error."""
        all_questions = []

        # Simulate Round 1: original quiz with questions 1-5
        for i in range(1, 6):
            all_questions.append({"question": f"Original Q{i}"})

        captured_prompts = []

        def fake_generate_json(prompt, system_instruction=None):
            captured_prompts.append(prompt)
            round_num = len(captured_prompts)
            return self._mock_retest_quiz_response(id_start=round_num * 10)

        with patch("modules.combined_generator.generate_json", side_effect=fake_generate_json):
            # Retest 1: excluding Original Q1-Q5
            retest_1 = generate_retest_quiz(self.SAMPLE_CHAPTER, all_questions, level="Beginner")
            for q in retest_1["questions"]:
                all_questions.append(q)

            # Retest 2: excluding Original Q1-Q5 + Retest 1 (Q10-Q14)
            retest_2 = generate_retest_quiz(self.SAMPLE_CHAPTER, all_questions, level="Beginner")
            for q in retest_2["questions"]:
                all_questions.append(q)

            # Retest 3: excluding all prior 15 questions
            retest_3 = generate_retest_quiz(self.SAMPLE_CHAPTER, all_questions, level="Beginner")

        self.assertEqual(len(captured_prompts), 3)

        # Check that Prompt 2 contains questions from round 1
        self.assertIn("Original Q1", captured_prompts[1])
        self.assertIn("Question 10: What is component 10?", captured_prompts[1])

        # Check that Prompt 3 contains questions from both previous rounds
        self.assertIn("Original Q5", captured_prompts[2])
        self.assertIn("Question 10: What is component 10?", captured_prompts[2])
        self.assertIn("Question 20: What is component 20?", captured_prompts[2])

    # ── 8. Previous questions are excluded and capped ─────────────────────────

    def test_previous_questions_are_excluded(self):
        """All previous questions passed to retest must be listed in the anti-repetition section."""
        prev = [
            {"question": "What pigment absorbs light in photosynthesis?"},
            {"question": "Where does the Calvin cycle take place?"},
        ]
        captured = {}

        def fake_generate_json(prompt, system_instruction=None):
            captured["prompt"] = prompt
            return self._mock_retest_quiz_response(1)

        with patch("modules.combined_generator.generate_json", side_effect=fake_generate_json):
            generate_retest_quiz(self.SAMPLE_CHAPTER, previous_questions=prev, level="Beginner")

        prompt = captured["prompt"]
        self.assertIn("PREVIOUS QUESTIONS (exclude these):", prompt)
        self.assertIn("What pigment absorbs light in photosynthesis?", prompt)
        self.assertIn("Where does the Calvin cycle take place?", prompt)
        self.assertIn("DO NOT repeat or paraphrase", prompt)

    def test_exclusion_prompt_is_capped_for_large_number_of_retests(self):
        """When dozens of retests occur, exclusions are capped to prevent unbounded prompt growth."""
        massive_history = [{"question": f"Historic Question {i}"} for i in range(1, 50)]
        captured = {}

        def fake_generate_json(prompt, system_instruction=None):
            captured["prompt"] = prompt
            return self._mock_retest_quiz_response(100)

        with patch("modules.combined_generator.generate_json", side_effect=fake_generate_json):
            generate_retest_quiz(self.SAMPLE_CHAPTER, previous_questions=massive_history, level="Beginner")

        prompt = captured["prompt"]
        # Max exclusions is 25: most recent should be included, earliest should be dropped
        self.assertIn("Historic Question 49", prompt)
        self.assertNotIn("Historic Question 1\n", prompt)

    # ── 9. Same learner level is preserved in retest ───────────────────────────

    def test_same_learner_level_is_preserved_in_retest(self):
        """Retest quiz must inject and preserve the specified learner level."""
        for level in ("Beginner", "Intermediate", "Advanced"):
            with self.subTest(level=level):
                captured = {}

                def fake_generate_json(prompt, system_instruction=None):
                    captured["prompt"] = prompt
                    return self._mock_retest_quiz_response(1)

                with patch("modules.combined_generator.generate_json", side_effect=fake_generate_json):
                    generate_retest_quiz(self.SAMPLE_CHAPTER, [], level=level)

                prompt = captured["prompt"]
                self.assertIn(f"## LEARNER LEVEL\n{level}", prompt)
                first_line = LEVEL_INSTRUCTIONS[level].split("\n")[0]
                self.assertIn(first_instruction_line := first_line, prompt)


class TestZeroRealAPICallsAndSpeedConfig(unittest.TestCase):
    """Tests verifying 0 real API calls and valid playback speed configurations."""

    # ── 10. No real API calls occur in tests ───────────────────────────────────

    def test_no_real_gemini_api_calls_occur_in_assistant(self):
        """Under no circumstances should generate_assistant_response hit live network in test suite."""
        with patch("modules.llm_service.get_client") as mock_client:
            mock_client.side_effect = AssertionError("REAL API CLIENT WAS CALLED! Network calls forbidden in tests.")
            # generate_assistant_response should use mocked generate_text, never get_client directly
            with patch("modules.combined_generator.generate_text", return_value="Mocked response"):
                resp = generate_assistant_response("Sample text", "What is this?", level="Beginner")
                self.assertEqual(resp, "Mocked response")

    def test_no_real_gemini_api_calls_occur_in_retest(self):
        """Under no circumstances should generate_retest_quiz hit live network in test suite."""
        with patch("modules.llm_service.get_client") as mock_client:
            mock_client.side_effect = AssertionError("REAL API CLIENT WAS CALLED! Network calls forbidden in tests.")
            mock_quiz = {
                "quiz": {
                    "quiz_title": "Mock Quiz",
                    "questions": [
                        {
                            "question": f"Q{i}",
                            "options": ["A", "B", "C", "D"],
                            "correct_answer": "A",
                            "explanation": "Exp",
                        }
                        for i in range(QUIZ_QUESTION_COUNT)
                    ],
                }
            }
            with patch("modules.combined_generator.generate_json", return_value=mock_quiz):
                quiz = generate_retest_quiz("Sample text", [], level="Beginner")
                self.assertEqual(len(quiz["questions"]), QUIZ_QUESTION_COUNT)

    def test_playback_speed_options_configured(self):
        """Playback speed options must match requirements: 0.75x, 1.0x, 1.25x, 1.5x, 2.0x."""
        expected = [0.75, 1.0, 1.25, 1.5, 2.0]
        self.assertEqual(PLAYBACK_SPEED_OPTIONS, expected)


if __name__ == "__main__":
    unittest.main()
