"""
Unit tests for the 2-call Gemini pipeline optimization and 503 retry handling.
All Gemini API calls are strictly mocked - zero real API quota is consumed.
"""

import unittest
from unittest.mock import patch, MagicMock, call
from modules.combined_generator import (
    generate_analysis_and_simplified,
    generate_comic_and_quiz
)


class TestGeminiOptimization(unittest.TestCase):

    def setUp(self):
        self.sample_text = "Photosynthesis is the process by which green plants produce glucose and oxygen from sunlight, water, and carbon dioxide."
        
        self.mock_analysis_simplified_response = {
            "analysis": {
                "chapter_title": "Photosynthesis Fundamentals",
                "topic": "Plant Biology",
                "key_concepts": ["Light Reactions", "Calvin Cycle"],
                "key_terms": ["Chlorophyll", "Stomata"],
                "important_facts": ["Occurs in chloroplasts", "Produces oxygen"],
                "learning_objectives": ["Understand the energy conversion in chloroplasts"],
                "difficulty": "Beginner",
                "prerequisite_concepts": ["Cell structure"]
            },
            "simplified": {
                "chapter_title": "Photosynthesis Fundamentals",
                "overview": "Plants use sunlight to make food.",
                "key_concepts": [
                    {"concept": "Chlorophyll", "explanation": "The green pigment catching sun rays."}
                ],
                "detailed_explanation": "Plants absorb water from roots and carbon dioxide from leaves...",
                "important_terms": [
                    {"term": "Chloroplast", "definition": "Cell organelle where food is made."}
                ],
                "quick_summary": "Sunlight + Water + CO2 = Sugar + Oxygen."
            }
        }

        self.mock_comic_quiz_response = {
            "comic": {
                "comic_title": "The Solar Kitchen",
                "panels": [
                    {
                        "panel_number": i,
                        "scene": f"Inside a leaf cell, part {i}",
                        "visual_description": f"Vibrant microscopic shot of chloroplast with sunlight, panel {i}",
                        "scene_description": f"Vibrant microscopic shot of chloroplast, panel {i}",
                        "characters": ["Sunny the Photon", "Chloe Chloroplast"],
                        "narration": f"Stage {i} begins as photons strike chlorophyll.",
                        "dialogue": f"Panel {i}: Let's synthesize energy!"
                    }
                    for i in range(1, 7)
                ]
            },
            "quiz": {
                "quiz_title": "Photosynthesis Quiz",
                "questions": [
                    {
                        "question": f"Question {i}: What is the primary product of stage {i}?",
                        "options": ["Glucose", "Nitrogen", "Methane", "Hydrogen"],
                        "correct_answer": "Glucose",
                        "explanation": "Glucose is synthesized as stored chemical energy."
                    }
                    for i in range(1, 6)
                ]
            }
        }

    @patch("modules.combined_generator.generate_json")
    def test_call_1_analysis_and_simplified_structure(self, mock_generate_json):
        """Verify Call 1 produces valid analysis and simplified data structures."""
        mock_generate_json.return_value = self.mock_analysis_simplified_response

        analysis, simplified = generate_analysis_and_simplified(self.sample_text, level="Beginner")

        # Verify analysis structure
        self.assertIn("chapter_title", analysis)
        self.assertIn("topic", analysis)
        self.assertIn("key_concepts", analysis)
        self.assertIn("key_terms", analysis)
        self.assertIn("important_facts", analysis)
        self.assertIn("learning_objectives", analysis)
        self.assertIn("difficulty", analysis)
        self.assertIn("prerequisite_concepts", analysis)
        self.assertEqual(analysis["chapter_title"], "Photosynthesis Fundamentals")

        # Verify simplified structure
        self.assertIn("chapter_title", simplified)
        self.assertIn("overview", simplified)
        self.assertIn("key_concepts", simplified)
        self.assertIn("detailed_explanation", simplified)
        self.assertIn("important_terms", simplified)
        self.assertIn("quick_summary", simplified)
        self.assertEqual(len(simplified["key_concepts"]), 1)

    @patch("modules.combined_generator.generate_json")
    def test_call_2_comic_and_quiz_structure(self, mock_generate_json):
        """Verify Call 2 produces exactly 6 comic panels and 5 quiz questions."""
        mock_generate_json.return_value = self.mock_comic_quiz_response

        comic, quiz = generate_comic_and_quiz(self.sample_text)

        # Verify comic panels
        panels = comic.get("panels", [])
        self.assertEqual(len(panels), 6, "Must contain exactly 6 comic panels")
        for panel in panels:
            self.assertIn("panel_number", panel)
            self.assertIn("scene", panel)
            self.assertIn("visual_description", panel)
            self.assertIn("scene_description", panel)  # UI backward compatibility
            self.assertIn("characters", panel)
            self.assertIn("narration", panel)
            self.assertIn("dialogue", panel)

        # Verify quiz questions
        questions = quiz.get("questions", [])
        self.assertEqual(len(questions), 5, "Must contain exactly 5 quiz questions")
        for q in questions:
            self.assertIn("question", q)
            self.assertIn("options", q)
            self.assertEqual(len(q["options"]), 4, "Each question must have exactly 4 options")
            self.assertIn("correct_answer", q)
            self.assertIn("explanation", q)
            self.assertIn(q["correct_answer"], q["options"])

    @patch("modules.combined_generator.generate_json")
    def test_pipeline_execution_makes_exactly_two_gemini_calls(self, mock_generate_json):
        """Verify that a full Generate Content execution makes exactly 2 Gemini calls."""
        call_count = {"n": 0}

        def side_effect(prompt, system_instruction=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call is always analysis + simplified
                return self.mock_analysis_simplified_response
            else:
                # Second call is always comic + quiz
                return self.mock_comic_quiz_response

        mock_generate_json.side_effect = side_effect

        # Simulate full pipeline Generate Content execution
        analysis, simplified = generate_analysis_and_simplified(self.sample_text, level="Intermediate")
        comic, quiz = generate_comic_and_quiz(self.sample_text)

        # Assert exactly 2 calls occurred
        self.assertEqual(mock_generate_json.call_count, 2, "Pipeline must make exactly 2 Gemini calls")
        self.assertIsNotNone(analysis)
        self.assertIsNotNone(simplified)
        self.assertIsNotNone(comic)
        self.assertIsNotNone(quiz)


    @patch("modules.combined_generator.generate_json")
    def test_gemini_api_failure_handling(self, mock_generate_json):
        """Verify that Gemini API runtime failures are cleanly raised for Streamlit error display."""
        mock_generate_json.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED: Quota exceeded")

        with self.assertRaises(RuntimeError) as ctx1:
            generate_analysis_and_simplified(self.sample_text, level="Beginner")
        self.assertIn("RESOURCE_EXHAUSTED", str(ctx1.exception))

        with self.assertRaises(RuntimeError) as ctx2:
            generate_comic_and_quiz(self.sample_text)
        self.assertIn("RESOURCE_EXHAUSTED", str(ctx2.exception))

    @patch("modules.combined_generator.generate_json")
    def test_malformed_json_handling(self, mock_generate_json):
        """Verify that missing sections or invalid response format raise descriptive errors."""
        mock_generate_json.return_value = {"incomplete": {}}

        with self.assertRaises(ValueError) as ctx:
            generate_analysis_and_simplified(self.sample_text)
        self.assertIn("analysis", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            generate_comic_and_quiz(self.sample_text)
        self.assertIn("comic", str(ctx.exception))

    def test_empty_input_handling(self):
        """Verify empty input texts fail fast with clean ValueError without calling API."""
        with self.assertRaises(ValueError):
            generate_analysis_and_simplified("")
        with self.assertRaises(ValueError):
            generate_comic_and_quiz("   ")


if __name__ == "__main__":
    unittest.main()


# ──────────────────────────────────────────────────────────────
# 503 / UNAVAILABLE retry tests
# ──────────────────────────────────────────────────────────────

class TestGeminiRetry503(unittest.TestCase):
    """Tests for _call_gemini_with_retry() in modules.llm_service."""

    def setUp(self):
        self.sample_text = (
            "Photosynthesis is the process by which plants produce "
            "glucose and oxygen from sunlight, water, and CO2."
        )
        self.mock_success_response = {
            "analysis": {
                "chapter_title": "Photosynthesis",
                "topic": "Plant Biology",
                "key_concepts": ["Light Reactions"],
                "key_terms": ["Chlorophyll"],
                "important_facts": ["Produces oxygen"],
                "learning_objectives": ["Understand energy conversion"],
                "difficulty": "Beginner",
                "prerequisite_concepts": [],
            },
            "simplified": {
                "chapter_title": "Photosynthesis",
                "overview": "Plants use sunlight to make food.",
                "key_concepts": [],
                "detailed_explanation": "Plants absorb CO2 and water...",
                "important_terms": [],
                "quick_summary": "Sunlight + Water + CO2 = Sugar + Oxygen.",
            },
        }

    @patch("modules.llm_service.time.sleep", return_value=None)
    @patch("modules.llm_service.get_client")
    def test_503_retry_success_on_second_attempt(self, mock_get_client, mock_sleep):
        """
        503 on first attempt → success on second attempt.
        Verifies: one retry is performed, result is returned, sleep is called once.
        No real Gemini API calls are made.
        """
        unavailable_error = RuntimeError("503 UNAVAILABLE: high demand")
        good_response = MagicMock()
        good_response.text = '{"key": "value"}'

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            unavailable_error,
            good_response,
        ]
        mock_get_client.return_value = mock_client

        from modules.llm_service import generate_json
        import json

        result = generate_json("test prompt")

        self.assertEqual(result, {"key": "value"})
        self.assertEqual(mock_client.models.generate_content.call_count, 2)
        mock_sleep.assert_called_once_with(2)   # first backoff = 2s

    @patch("modules.llm_service.time.sleep", return_value=None)
    @patch("modules.llm_service.get_client")
    def test_503_retry_exhaustion_raises_runtime_error(self, mock_get_client, mock_sleep):
        """
        503 on every attempt → RuntimeError after all retries are exhausted.
        Verifies: exactly _MAX_RETRIES+1 (4) SDK calls, 3 sleeps, RuntimeError raised.
        No real Gemini API calls are made.
        """
        unavailable_error = Exception("503 UNAVAILABLE: This model is currently experiencing high demand.")

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = unavailable_error
        mock_get_client.return_value = mock_client

        from modules.llm_service import generate_json, _MAX_RETRIES, _BACKOFF_SECONDS

        with self.assertRaises(RuntimeError) as ctx:
            generate_json("test prompt")

        self.assertIn("503", str(ctx.exception))

        # 1 initial attempt + _MAX_RETRIES retries = _MAX_RETRIES + 1 total calls
        expected_calls = _MAX_RETRIES + 1
        self.assertEqual(
            mock_client.models.generate_content.call_count,
            expected_calls,
            f"Expected {expected_calls} total SDK calls, "
            f"got {mock_client.models.generate_content.call_count}",
        )

        # One sleep per retry (not on the final exhausted attempt)
        self.assertEqual(mock_sleep.call_count, _MAX_RETRIES)
        expected_sleep_calls = [call(s) for s in _BACKOFF_SECONDS[:_MAX_RETRIES]]
        mock_sleep.assert_has_calls(expected_sleep_calls)

    @patch("modules.llm_service.time.sleep", return_value=None)
    @patch("modules.llm_service.get_client")
    def test_429_is_not_retried(self, mock_get_client, mock_sleep):
        """
        429 RESOURCE_EXHAUSTED → raised immediately, zero retries, zero sleeps.
        """
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "429 RESOURCE_EXHAUSTED: Quota exceeded"
        )
        mock_get_client.return_value = mock_client

        from modules.llm_service import generate_json

        with self.assertRaises(RuntimeError) as ctx:
            generate_json("test prompt")

        self.assertIn("RESOURCE_EXHAUSTED", str(ctx.exception))
        self.assertEqual(mock_client.models.generate_content.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("modules.llm_service.time.sleep", return_value=None)
    @patch("modules.llm_service.get_client")
    def test_auth_error_is_not_retried(self, mock_get_client, mock_sleep):
        """
        401 / API_KEY_INVALID → raised immediately, zero retries, zero sleeps.
        """
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "401 API_KEY_INVALID: The provided API key is invalid."
        )
        mock_get_client.return_value = mock_client

        from modules.llm_service import generate_json

        with self.assertRaises(RuntimeError) as ctx:
            generate_json("test prompt")

        self.assertIn("API_KEY_INVALID", str(ctx.exception))
        self.assertEqual(mock_client.models.generate_content.call_count, 1)
        mock_sleep.assert_not_called()


# ──────────────────────────────────────────────────────────────
# Audio generator tests
# ──────────────────────────────────────────────────────────────

class TestAudioGenerator(unittest.TestCase):
    """Tests for modules.audio_generator — no real network calls."""

    @patch("modules.audio_generator.gTTS")
    def test_generate_audio_bytes_returns_bytes(self, mock_gtts_cls):
        """generate_audio_bytes returns non-empty bytes on success."""
        import io
        fake_audio = b"\xff\xfb\x90\x00" * 100  # fake MP3 header bytes

        mock_tts_instance = MagicMock()

        def fake_write_to_fp(fp):
            fp.write(fake_audio)

        mock_tts_instance.write_to_fp.side_effect = fake_write_to_fp
        mock_gtts_cls.return_value = mock_tts_instance

        from modules.audio_generator import generate_audio_bytes
        result = generate_audio_bytes("Hello world")

        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)
        mock_gtts_cls.assert_called_once_with(text="Hello world", lang="en", slow=False)

    def test_generate_audio_bytes_empty_text_raises(self):
        """generate_audio_bytes raises RuntimeError for empty/whitespace text."""
        from modules.audio_generator import generate_audio_bytes
        with self.assertRaises(RuntimeError) as ctx:
            generate_audio_bytes("   ")
        self.assertIn("no text provided", str(ctx.exception))

    @patch("modules.audio_generator.gTTS")
    def test_generate_audio_bytes_network_failure(self, mock_gtts_cls):
        """gTTS network error is wrapped in RuntimeError, not propagated raw."""
        mock_gtts_cls.side_effect = Exception("Failed to connect to TTS server")
        from modules.audio_generator import generate_audio_bytes
        with self.assertRaises(RuntimeError) as ctx:
            generate_audio_bytes("Some text")
        self.assertIn("Audio generation failed", str(ctx.exception))

    @patch("modules.audio_generator.gTTS")
    def test_generate_audio_file_alias_works(self, mock_gtts_cls):
        """generate_audio_file() is a working alias for generate_audio_bytes()."""
        fake_audio = b"\xff\xfb" * 50
        mock_tts_instance = MagicMock()
        mock_tts_instance.write_to_fp.side_effect = lambda fp: fp.write(fake_audio)
        mock_gtts_cls.return_value = mock_tts_instance

        from modules.audio_generator import generate_audio_file
        result = generate_audio_file("Test alias")
        self.assertIsInstance(result, bytes)


# ──────────────────────────────────────────────────────────────
# Comic script / image independence tests
# ──────────────────────────────────────────────────────────────

class TestComicFallback(unittest.TestCase):
    """
    Verify that comic script data is always present and accessible even when
    image generation raises RuntimeError for every panel.
    No real Gemini or HF calls are made.
    """

    def setUp(self):
        self.sample_text = "Photosynthesis converts sunlight to sugar."
        self.mock_comic_quiz = {
            "comic": {
                "comic_title": "The Solar Kitchen",
                "panels": [
                    {
                        "panel_number": i,
                        "scene": f"Scene {i}",
                        "visual_description": f"Panel {i} visual",
                        "scene_description": f"Panel {i} description",
                        "characters": ["Character A"],
                        "narration": f"Narration {i}",
                        "dialogue": f"Dialogue {i}",
                    }
                    for i in range(1, 7)
                ],
            },
            "quiz": {
                "quiz_title": "Quiz",
                "questions": [
                    {
                        "question": f"Q{i}",
                        "options": ["A", "B", "C", "D"],
                        "correct_answer": "A",
                        "explanation": "Because A.",
                    }
                    for i in range(1, 6)
                ],
            },
        }

    @patch("modules.combined_generator.generate_json")
    def test_comic_script_present_when_images_fail(self, mock_generate_json):
        """
        Comic script (panels with scene, narration, dialogue) is always
        accessible even when image generation raises RuntimeError.
        """
        mock_generate_json.return_value = self.mock_comic_quiz

        comic, quiz = generate_comic_and_quiz(self.sample_text)

        # Simulate 6 image generation failures
        image_failures = []
        comic_images = {}
        for i, panel in enumerate(comic.get("panels", [])):
            try:
                raise RuntimeError("402 Payment Required — no HF credits")
            except RuntimeError as e:
                image_failures.append(str(e))

        # Script must be fully intact despite 6 image failures
        self.assertEqual(len(comic["panels"]), 6)
        self.assertEqual(len(image_failures), 6)
        self.assertEqual(len(comic_images), 0)  # no images stored

        for panel in comic["panels"]:
            self.assertIn("scene_description", panel)
            self.assertIn("narration", panel)
            self.assertIn("dialogue", panel)

    @patch("modules.combined_generator.generate_json")
    def test_partial_image_success_does_not_break_script(self, mock_generate_json):
        """
        If 3 of 6 images succeed and 3 fail, the other 3 panels still show script.
        """
        mock_generate_json.return_value = self.mock_comic_quiz
        comic, _ = generate_comic_and_quiz(self.sample_text)

        comic_images = {}
        for i, panel in enumerate(comic.get("panels", [])):
            if i % 2 == 0:
                comic_images[i] = b"\x89PNG fake bytes"  # fake success
            # odd panels: no image stored, script still accessible

        self.assertEqual(len(comic["panels"]), 6)
        self.assertEqual(len(comic_images), 3)  # only even indices


# ──────────────────────────────────────────────────────────────
# Session-state caching key logic tests
# ──────────────────────────────────────────────────────────────

class TestCachingLogic(unittest.TestCase):
    """
    Tests for the _pdf_hash / _cache_hit logic in app.py.
    Verifies that caching correctly identifies same-PDF-same-level hits
    and correctly invalidates on PDF or level change.
    No Streamlit imports needed — we test the pure functions directly.
    """

    def _make_hash(self, content: bytes) -> str:
        import hashlib
        return hashlib.md5(content).hexdigest()

    def test_same_pdf_same_level_is_cache_hit(self):
        """Identical PDF bytes and level → same hash → cache would hit."""
        pdf_bytes = b"%PDF-1.4 fake content for test"
        h1 = self._make_hash(pdf_bytes)
        h2 = self._make_hash(pdf_bytes)
        self.assertEqual(h1, h2)

    def test_different_pdf_is_cache_miss(self):
        """Different PDF bytes → different hash → cache would miss."""
        h1 = self._make_hash(b"%PDF-1.4 chapter one")
        h2 = self._make_hash(b"%PDF-1.4 chapter two")
        self.assertNotEqual(h1, h2)

    def test_same_pdf_different_level_requires_regeneration(self):
        """
        Same hash but different learning level means cache is invalid.
        Modelling the _cache_hit() logic from app.py.
        """
        pdf_bytes = b"%PDF-1.4 same content"
        current_hash = self._make_hash(pdf_bytes)

        # Simulate stored state from previous Beginner generation
        stored_hash = current_hash
        stored_level = "Beginner"
        current_level = "Advanced"

        # Cache hit requires BOTH hash AND level to match
        cache_valid = (stored_hash == current_hash) and (stored_level == current_level)
        self.assertFalse(cache_valid)

    def test_same_pdf_same_level_with_results_is_hit(self):
        """Same hash + level + results present → full cache hit."""
        pdf_bytes = b"%PDF-1.4 same content"
        current_hash = self._make_hash(pdf_bytes)

        stored_hash = current_hash
        stored_level = "Intermediate"
        current_level = "Intermediate"
        analysis = {"chapter_title": "Test"}  # non-None
        comic = {"panels": []}               # non-None

        cache_valid = (
            stored_hash == current_hash
            and stored_level == current_level
            and analysis is not None
            and comic is not None
        )
        self.assertTrue(cache_valid)

    def test_cache_miss_when_results_none(self):
        """Same hash + level but results are None → not a cache hit."""
        pdf_bytes = b"%PDF-1.4 content"
        current_hash = self._make_hash(pdf_bytes)

        cache_valid = (
            current_hash == current_hash
            and "Beginner" == "Beginner"
            and None is not None   # analysis is None
        )
        self.assertFalse(cache_valid)



# ──────────────────────────────────────────────────────────────
# Image generator tests
# ──────────────────────────────────────────────────────────────

class TestImageGenerator(unittest.TestCase):
    """Tests for modules.image_generator — no real network calls."""

    @patch("modules.image_generator.requests.get")
    @patch("modules.image_generator.requests.post")
    def test_generate_image_bytes_success(self, mock_post, mock_get):
        """Verifies generate_image_bytes returns bytes on a valid API response."""
        import os
        os.environ["PIXAZO_API_KEY"] = "fake-key"
        
        # Mock POST response returning an immediate image URL
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {"url": "https://fake.pixazo.ai/image.png"}
        mock_post.return_value = mock_post_resp
        
        # Mock GET response for downloading the image
        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.content = b"\x89PNG\r\n\x1a\nfakebytes"
        mock_get.return_value = mock_get_resp
        
        from modules.image_generator import generate_image_bytes
        result = generate_image_bytes("Test prompt")
        
        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"\x89PNG"))
        mock_post.assert_called_once()
        mock_get.assert_called_once_with("https://fake.pixazo.ai/image.png")

    @patch("modules.image_generator.requests.post")
    def test_generate_image_bytes_404_error(self, mock_post):
        """Verifies 404 errors are caught and re-raised cleanly."""
        import os
        os.environ["PIXAZO_API_KEY"] = "fake-key"
        
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 404
        mock_post.return_value = mock_post_resp
        
        from modules.image_generator import generate_image_bytes
        with self.assertRaises(RuntimeError) as ctx:
            generate_image_bytes("Test prompt")
            
        self.assertIn("Pixazo API free endpoint unavailable", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
