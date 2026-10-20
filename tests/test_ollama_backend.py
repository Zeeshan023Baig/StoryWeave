"""
tests/test_ollama_backend.py
============================
Unit tests for the Ollama backend added to modules/llm_service.py.

Coverage
--------
 1. Ollama text generation (success path)
 2. Ollama JSON generation (success path, valid JSON)
 3. Valid JSON parsing — direct, fenced, prose-wrapped
 4. Malformed / unparseable JSON handling
 5. Ollama unavailable (ConnectionError)
 6. Request timeout
 7. Model not found (HTTP 404)
 8. HTTP error other than 404
 9. Empty response from model
10. Provider selection — "ollama" routes to Ollama, "gemini" routes to Gemini
11. generate_analysis_and_simplified works end-to-end with mocked Ollama
12. generate_comic_and_quiz works end-to-end with mocked Ollama
13. generate_retest_quiz works end-to-end with mocked Ollama
14. generate_assistant_response works end-to-end with mocked Ollama
15. _extract_json_from_text handles <think> tags
16. _extract_json_from_text handles markdown fences
17. JSON reminder appended in JSON mode
18. system_instruction forwarded as system message

DESIGN NOTES:
 - Zero real HTTP calls are made.  All requests.post() calls are patched.
 - Zero real Gemini calls are made.  get_client() is never invoked under Ollama.
 - Tests are isolated: AI_PROVIDER is patched per-test so the global state
   in the already-imported llm_service module is safely overridden.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import requests  # imported so we can use its exception types in mocks


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ollama_response(content: str, status_code: int = 200) -> MagicMock:
    """Build a fake requests.Response that Ollama would return."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {
        "model": "qwen3.5:2b",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
    mock_resp.text = json.dumps(mock_resp.json.return_value)
    return mock_resp


def _make_http_error_response(status_code: int, body: str = "error") -> MagicMock:
    """Build a fake requests.Response with a non-200 status."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = body
    mock_resp.json.side_effect = json.JSONDecodeError("no json", "", 0)
    return mock_resp


# ─────────────────────────────────────────────────────────────────────────────
# Sample payloads reused across multiple tests
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_TEXT = "Photosynthesis is the process by which plants make food from sunlight."

_ANALYSIS_JSON = json.dumps({
    "analysis": {
        "chapter_title": "Photosynthesis",
        "topic": "Plant Biology",
        "key_concepts": ["Light Reactions"],
        "key_terms": ["Chlorophyll"],
        "important_facts": ["Produces oxygen"],
        "learning_objectives": ["Understand ATP"],
        "difficulty": "Beginner",
        "prerequisite_concepts": [],
    },
    "simplified": {
        "chapter_title": "Photosynthesis",
        "overview": "Plants use sunlight to make food.",
        "key_concepts": [],
        "detailed_explanation": "Sunlight drives the process.",
        "important_terms": [],
        "quick_summary": "Sunlight + CO2 + Water = Sugar + O2.",
    },
})

_COMIC_QUIZ_JSON = json.dumps({
    "comic": {
        "comic_title": "The Green Machine",
        "panels": [
            {
                "panel_number": i,
                "scene": f"Scene {i}",
                "visual_description": f"Visual {i}",
                "characters": [],
                "narration": f"Narration {i}",
                "dialogue": f"Dialogue {i}",
            }
            for i in range(1, 7)
        ],
    },
    "quiz": {
        "quiz_title": "Photosynthesis Quiz",
        "questions": [
            {
                "question": f"Question {i}?",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "explanation": "Because A.",
            }
            for i in range(1, 6)
        ],
    },
})

_RETEST_JSON = json.dumps({
    "quiz": {
        "quiz_title": "Retest",
        "questions": [
            {
                "question": f"New Question {i}?",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "explanation": "Because A.",
            }
            for i in range(1, 6)
        ],
    }
})


# ═════════════════════════════════════════════════════════════════════════════
# 1 & 2 — Basic success paths
# ═════════════════════════════════════════════════════════════════════════════

class TestOllamaTextGeneration(unittest.TestCase):
    """Test #1 — Ollama text generation success path."""

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post")
    def test_generate_text_returns_string(self, mock_post):
        """generate_text with Ollama provider returns a non-empty string."""
        mock_post.return_value = _make_ollama_response("Hello, this is the response.")
        from modules.llm_service import generate_text
        result = generate_text("Summarise photosynthesis.")
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Hello, this is the response.")

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post")
    def test_generate_text_sends_user_prompt(self, mock_post):
        """generate_text sends the prompt as a user message in the payload."""
        mock_post.return_value = _make_ollama_response("OK")
        from modules.llm_service import generate_text
        generate_text("My special prompt", system_instruction="Be helpful.")
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1].get("json") or call_kwargs[0][1]
        messages = payload["messages"]
        roles = [m["role"] for m in messages]
        self.assertIn("user", roles)
        user_msg = next(m for m in messages if m["role"] == "user")
        self.assertIn("My special prompt", user_msg["content"])


class TestOllamaJsonGeneration(unittest.TestCase):
    """Test #2 — Ollama JSON generation success path."""

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post")
    def test_generate_json_returns_dict(self, mock_post):
        """generate_json with Ollama provider parses and returns a dict."""
        mock_post.return_value = _make_ollama_response('{"key": "value", "count": 42}')
        from modules.llm_service import generate_json
        result = generate_json("Return JSON with key and count.")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["count"], 42)

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post")
    def test_generate_json_sets_format_json(self, mock_post):
        """generate_json sets format='json' in the Ollama request payload."""
        mock_post.return_value = _make_ollama_response('{"ok": true}')
        from modules.llm_service import generate_json
        generate_json("Give JSON.")
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1].get("json") or call_kwargs[0][1]
        self.assertEqual(payload.get("format"), "json")

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post")
    def test_successful_json_response_parsing(self, mock_post):
        """Verify successful JSON response parsing for structured output with qwen3.5:2b."""
        payload_data = {"status": "ok", "message": "STORYWEAVE JSON TEST", "count": 3}
        mock_post.return_value = _make_ollama_response(json.dumps(payload_data))
        from modules.llm_service import generate_json
        result = generate_json("Generate test JSON")
        self.assertEqual(result, payload_data)


# ═════════════════════════════════════════════════════════════════════════════
# 3 — Valid JSON parsing variations
# ═════════════════════════════════════════════════════════════════════════════

class TestOllamaJsonParsing(unittest.TestCase):
    """Test #3 — _extract_json_from_text handles multiple valid formats."""

    def _parse(self, text):
        from modules.llm_service import _extract_json_from_text
        return _extract_json_from_text(text)

    def test_direct_json_object(self):
        result = self._parse('{"a": 1, "b": "hello"}')
        self.assertEqual(result, {"a": 1, "b": "hello"})

    def test_json_with_leading_whitespace(self):
        result = self._parse('   \n{"x": true}   ')
        self.assertEqual(result, {"x": True})

    def test_json_wrapped_in_prose(self):
        text = 'Sure! Here is the JSON:\n{"result": "ok"}\nHope that helps!'
        result = self._parse(text)
        self.assertEqual(result, {"result": "ok"})

    def test_json_in_markdown_code_fence(self):
        text = '```json\n{"answer": 42}\n```'
        result = self._parse(text)
        self.assertEqual(result, {"answer": 42})

    def test_json_in_plain_code_fence(self):
        text = '```\n{"answer": 42}\n```'
        result = self._parse(text)
        self.assertEqual(result, {"answer": 42})

    def test_nested_json(self):
        result = self._parse('{"outer": {"inner": [1, 2, 3]}}')
        self.assertEqual(result["outer"]["inner"], [1, 2, 3])


# ═════════════════════════════════════════════════════════════════════════════
# 4 — Malformed JSON handling
# ═════════════════════════════════════════════════════════════════════════════

class TestMalformedJsonHandling(unittest.TestCase):
    """Test #4 — Unparseable model output raises ValueError (not crashes)."""

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post")
    def test_malformed_json_raises_value_error(self, mock_post):
        """Malformed JSON from Ollama must raise ValueError, not crash."""
        mock_post.return_value = _make_ollama_response("This is not JSON at all.")
        from modules.llm_service import generate_json
        with self.assertRaises(ValueError) as ctx:
            generate_json("Return JSON.")
        self.assertIn("Failed to parse JSON", str(ctx.exception))

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post")
    def test_partial_json_raises_value_error(self, mock_post):
        """Truncated JSON must raise ValueError."""
        mock_post.return_value = _make_ollama_response('{"key": "value", "incomplete":')
        from modules.llm_service import generate_json
        with self.assertRaises(ValueError):
            generate_json("Return JSON.")

    def test_extract_json_raises_for_plain_text(self):
        """_extract_json_from_text raises ValueError for plain prose."""
        from modules.llm_service import _extract_json_from_text
        with self.assertRaises(ValueError):
            _extract_json_from_text("No JSON here, just words.")


# ═════════════════════════════════════════════════════════════════════════════
# 5 — Ollama unavailable (ConnectionError)
# ═════════════════════════════════════════════════════════════════════════════

class TestOllamaUnavailable(unittest.TestCase):
    """Test #5 — Connection refused raises a friendly RuntimeError."""

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post",
           side_effect=requests.exceptions.ConnectionError("Connection refused"))
    def test_connection_refused_raises_runtime_error(self, mock_post):
        from modules.llm_service import generate_text
        with self.assertRaises(RuntimeError) as ctx:
            generate_text("Hello")
        msg = str(ctx.exception)
        self.assertIn("Cannot connect to Ollama", msg)
        self.assertIn("ollama serve", msg)

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post",
           side_effect=requests.exceptions.ConnectionError("Connection refused"))
    def test_connection_refused_in_json_mode(self, mock_post):
        from modules.llm_service import generate_json
        with self.assertRaises(RuntimeError) as ctx:
            generate_json("Return JSON.")
        self.assertIn("Cannot connect to Ollama", str(ctx.exception))


# ═════════════════════════════════════════════════════════════════════════════
# 6 — Timeout
# ═════════════════════════════════════════════════════════════════════════════

class TestOllamaTimeout(unittest.TestCase):
    """Test #6 — Request timeout raises a friendly RuntimeError."""

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post",
           side_effect=requests.exceptions.Timeout("Read timed out"))
    def test_timeout_raises_runtime_error(self, mock_post):
        from modules.llm_service import generate_text
        with self.assertRaises(RuntimeError) as ctx:
            generate_text("Hello")
        msg = str(ctx.exception)
        self.assertIn("timed out", msg.lower())

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post",
           side_effect=requests.exceptions.Timeout("timed out"))
    def test_timeout_in_json_mode(self, mock_post):
        from modules.llm_service import generate_json
        with self.assertRaises(RuntimeError) as ctx:
            generate_json("Return JSON.")
        self.assertIn("timed out", str(ctx.exception).lower())


# ═════════════════════════════════════════════════════════════════════════════
# 7 — Model not found (HTTP 404)
# ═════════════════════════════════════════════════════════════════════════════

class TestOllamaModelNotFound(unittest.TestCase):
    """Test #7 — HTTP 404 from Ollama is translated to a clear RuntimeError."""

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post")
    def test_404_raises_runtime_error_with_install_hint(self, mock_post):
        mock_post.return_value = _make_http_error_response(404)
        from modules.llm_service import generate_text
        with self.assertRaises(RuntimeError) as ctx:
            generate_text("Hello")
        msg = str(ctx.exception)
        self.assertIn("not found", msg.lower())
        self.assertIn("ollama pull", msg)


# ═════════════════════════════════════════════════════════════════════════════
# 8 — HTTP error other than 404
# ═════════════════════════════════════════════════════════════════════════════

class TestOllamaHttpErrors(unittest.TestCase):
    """Test #8 — Non-404 HTTP errors raise RuntimeError with status code."""

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post")
    def test_500_raises_runtime_error(self, mock_post):
        mock_post.return_value = _make_http_error_response(500, "Internal Server Error")
        from modules.llm_service import generate_text
        with self.assertRaises(RuntimeError) as ctx:
            generate_text("Hello")
        self.assertIn("500", str(ctx.exception))

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post")
    def test_503_raises_runtime_error(self, mock_post):
        mock_post.return_value = _make_http_error_response(503, "Service Unavailable")
        from modules.llm_service import generate_text
        with self.assertRaises(RuntimeError) as ctx:
            generate_text("Hello")
        self.assertIn("503", str(ctx.exception))


# ═════════════════════════════════════════════════════════════════════════════
# 9 — Empty response from model
# ═════════════════════════════════════════════════════════════════════════════

class TestOllamaEmptyResponse(unittest.TestCase):
    """Test #9 — An empty model response handling and fallbacks."""

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post")
    def test_empty_content_raises_runtime_error(self, mock_post):
        mock_post.return_value = _make_ollama_response("")
        from modules.llm_service import generate_text
        with self.assertRaises(RuntimeError) as ctx:
            generate_text("Hello")
        self.assertIn("empty response", str(ctx.exception).lower())

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post")
    def test_empty_content_with_thinking_fallback_succeeds(self, mock_post):
        """When message.content is empty but thinking contains text, fallback uses thinking."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model": "qwen3.5:2b",
            "message": {"role": "assistant", "content": "", "thinking": '{"fallback": "success"}'},
            "done": True,
        }
        mock_post.return_value = mock_resp
        from modules.llm_service import generate_json
        result = generate_json("Get json")
        self.assertEqual(result, {"fallback": "success"})

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post")
    def test_empty_json_response_raises_runtime_error(self, mock_post):
        """When both content and thinking are completely empty, RuntimeError is raised."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model": "qwen3.5:2b",
            "message": {"role": "assistant", "content": "", "thinking": ""},
            "done": True,
        }
        mock_post.return_value = mock_resp
        from modules.llm_service import generate_json
        with self.assertRaises(RuntimeError) as ctx:
            generate_json("Get json")
        self.assertIn("empty response", str(ctx.exception).lower())


# ═════════════════════════════════════════════════════════════════════════════
# 10 — Provider selection
# ═════════════════════════════════════════════════════════════════════════════

class TestProviderSelection(unittest.TestCase):
    """Test #10 — AI_PROVIDER correctly routes to Ollama vs Gemini."""

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service._generate_text_ollama", return_value="ollama result")
    @patch("modules.llm_service._generate_text_gemini")
    def test_ollama_provider_routes_to_ollama_text(self, mock_gemini, mock_ollama):
        from modules.llm_service import generate_text
        result = generate_text("prompt")
        mock_ollama.assert_called_once()
        mock_gemini.assert_not_called()
        self.assertEqual(result, "ollama result")

    @patch("modules.llm_service.AI_PROVIDER", "gemini")
    @patch("modules.llm_service._generate_text_gemini", return_value="gemini result")
    @patch("modules.llm_service._generate_text_ollama")
    def test_gemini_provider_routes_to_gemini_text(self, mock_ollama, mock_gemini):
        from modules.llm_service import generate_text
        result = generate_text("prompt")
        mock_gemini.assert_called_once()
        mock_ollama.assert_not_called()
        self.assertEqual(result, "gemini result")

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service._generate_json_ollama", return_value={"ok": True})
    @patch("modules.llm_service._generate_json_gemini")
    def test_ollama_provider_routes_to_ollama_json(self, mock_gemini, mock_ollama):
        from modules.llm_service import generate_json
        result = generate_json("prompt")
        mock_ollama.assert_called_once()
        mock_gemini.assert_not_called()
        self.assertEqual(result, {"ok": True})

    @patch("modules.llm_service.AI_PROVIDER", "gemini")
    @patch("modules.llm_service._generate_json_gemini", return_value={"gemini": True})
    @patch("modules.llm_service._generate_json_ollama")
    def test_gemini_provider_routes_to_gemini_json(self, mock_ollama, mock_gemini):
        from modules.llm_service import generate_json
        result = generate_json("prompt")
        mock_gemini.assert_called_once()
        mock_ollama.assert_not_called()
        self.assertEqual(result, {"gemini": True})

    @patch("modules.llm_service.AI_PROVIDER", "OLLAMA")  # uppercase variant
    @patch("modules.llm_service._generate_text_ollama", return_value="uppercase ok")
    @patch("modules.llm_service._generate_text_gemini")
    def test_provider_is_case_insensitive(self, mock_gemini, mock_ollama):
        from modules.llm_service import generate_text
        result = generate_text("prompt")
        mock_ollama.assert_called_once()
        mock_gemini.assert_not_called()

    @patch("modules.llm_service.AI_PROVIDER", "unknown_value")
    @patch("modules.llm_service._generate_text_ollama", return_value="fallback")
    @patch("modules.llm_service._generate_text_gemini")
    def test_unknown_provider_falls_back_to_ollama(self, mock_gemini, mock_ollama):
        """Any unrecognised provider string falls back to Ollama (safe default)."""
        from modules.llm_service import generate_text
        generate_text("prompt")
        mock_ollama.assert_called_once()
        mock_gemini.assert_not_called()


# ═════════════════════════════════════════════════════════════════════════════
# 11 — generate_analysis_and_simplified with mocked Ollama
# ═════════════════════════════════════════════════════════════════════════════

class TestCombinedGeneratorWithOllama(unittest.TestCase):
    """Test #11 — generate_analysis_and_simplified works end-to-end with Ollama mock."""

    @patch("modules.combined_generator.generate_json")
    def test_analysis_and_simplified_with_ollama_mock(self, mock_gen_json):
        """generate_analysis_and_simplified returns valid (analysis, simplified) tuple."""
        mock_gen_json.return_value = json.loads(_ANALYSIS_JSON)
        from modules.combined_generator import generate_analysis_and_simplified
        analysis, simplified = generate_analysis_and_simplified(_SAMPLE_TEXT, level="Beginner")

        self.assertIsInstance(analysis, dict)
        self.assertIsInstance(simplified, dict)
        self.assertIn("chapter_title", analysis)
        self.assertIn("overview", simplified)

    @patch("modules.combined_generator.generate_json")
    def test_missing_analysis_section_raises(self, mock_gen_json):
        """Missing 'analysis' key in JSON response raises ValueError."""
        mock_gen_json.return_value = {"simplified": {}}
        from modules.combined_generator import generate_analysis_and_simplified
        with self.assertRaises(ValueError) as ctx:
            generate_analysis_and_simplified(_SAMPLE_TEXT)
        self.assertIn("analysis", str(ctx.exception))


# ═════════════════════════════════════════════════════════════════════════════
# 12 — generate_comic_and_quiz with mocked Ollama
# ═════════════════════════════════════════════════════════════════════════════

class TestComicAndQuizWithOllama(unittest.TestCase):
    """Test #12 — generate_comic_and_quiz works end-to-end with Ollama mock."""

    @patch("modules.combined_generator.generate_json")
    def test_comic_and_quiz_returns_correct_structure(self, mock_gen_json):
        """generate_comic_and_quiz returns (comic, quiz) with 6 panels and 5 questions."""
        mock_gen_json.return_value = json.loads(_COMIC_QUIZ_JSON)
        from modules.combined_generator import generate_comic_and_quiz
        comic, quiz = generate_comic_and_quiz(_SAMPLE_TEXT)

        self.assertEqual(len(comic["panels"]), 6)
        self.assertEqual(len(quiz["questions"]), 5)

        for panel in comic["panels"]:
            self.assertIn("panel_number", panel)
            self.assertIn("scene", panel)
            self.assertIn("narration", panel)
            self.assertIn("dialogue", panel)

        for q in quiz["questions"]:
            self.assertIn("question", q)
            self.assertIn("options", q)
            self.assertIn("correct_answer", q)
            self.assertIn("explanation", q)


# ═════════════════════════════════════════════════════════════════════════════
# 13 — generate_retest_quiz with mocked Ollama
# ═════════════════════════════════════════════════════════════════════════════

class TestRetestWithOllama(unittest.TestCase):
    """Test #13 — generate_retest_quiz works end-to-end with Ollama mock."""

    @patch("modules.combined_generator.generate_json")
    def test_retest_returns_fresh_quiz(self, mock_gen_json):
        """generate_retest_quiz returns a quiz dict with questions."""
        mock_gen_json.return_value = json.loads(_RETEST_JSON)
        prev = [{"question": "Old question 1?"}, {"question": "Old question 2?"}]
        from modules.combined_generator import generate_retest_quiz
        quiz = generate_retest_quiz(_SAMPLE_TEXT, previous_questions=prev)

        self.assertIn("questions", quiz)
        self.assertEqual(len(quiz["questions"]), 5)

    @patch("modules.combined_generator.generate_json")
    def test_retest_prompt_excludes_previous_questions(self, mock_gen_json):
        """Previous question texts must appear in the retest prompt."""
        captured = {}

        def capture_prompt(prompt, system_instruction=None):
            captured["prompt"] = prompt
            return json.loads(_RETEST_JSON)

        mock_gen_json.side_effect = capture_prompt
        prev = [{"question": "What is chlorophyll?"}, {"question": "Where does photosynthesis occur?"}]
        from modules.combined_generator import generate_retest_quiz
        generate_retest_quiz(_SAMPLE_TEXT, previous_questions=prev)

        self.assertIn("What is chlorophyll?", captured["prompt"])
        self.assertIn("Where does photosynthesis occur?", captured["prompt"])


# ═════════════════════════════════════════════════════════════════════════════
# 14 — generate_assistant_response with mocked Ollama
# ═════════════════════════════════════════════════════════════════════════════

class TestAssistantResponseWithOllama(unittest.TestCase):
    """Test #14 — generate_assistant_response works end-to-end with Ollama mock."""

    @patch("modules.combined_generator.generate_text")
    def test_assistant_returns_string(self, mock_gen_text):
        """generate_assistant_response returns a non-empty string."""
        mock_gen_text.return_value = "Photosynthesis is the process of converting light energy."
        from modules.combined_generator import generate_assistant_response
        result = generate_assistant_response(
            _SAMPLE_TEXT, "Explain photosynthesis briefly.", level="Beginner"
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    @patch("modules.combined_generator.generate_text")
    def test_assistant_prompt_contains_user_question(self, mock_gen_text):
        """The user's question must appear in the prompt sent to the LLM."""
        captured = {}

        def capture(prompt, system_instruction=None):
            captured["prompt"] = prompt
            return "Response text."

        mock_gen_text.side_effect = capture
        user_q = "What is the role of chlorophyll?"
        from modules.combined_generator import generate_assistant_response
        generate_assistant_response(_SAMPLE_TEXT, user_q, level="Intermediate")
        self.assertIn(user_q, captured["prompt"])

    @patch("modules.combined_generator.generate_text")
    def test_no_real_ollama_calls_in_test(self, mock_gen_text):
        """generate_assistant_response must never reach real HTTP layer in tests."""
        mock_gen_text.return_value = "Safe mock response"
        with patch("modules.llm_service.requests.post") as mock_post:
            mock_post.side_effect = AssertionError(
                "Real HTTP call made! Mocking failed."
            )
            from modules.combined_generator import generate_assistant_response
            # This should use the mocked generate_text, not real HTTP
            result = generate_assistant_response(_SAMPLE_TEXT, "What is ATP?")
        self.assertEqual(result, "Safe mock response")


# ═════════════════════════════════════════════════════════════════════════════
# 15 — <think> tag stripping
# ═════════════════════════════════════════════════════════════════════════════

class TestThinkTagStripping(unittest.TestCase):
    """Test #15 — _extract_json_from_text strips <think>...</think> wrappers."""

    def test_think_tags_stripped_before_parsing(self):
        text = '<think>I need to return a JSON object here.</think>{"result": "yes"}'
        from modules.llm_service import _extract_json_from_text
        result = _extract_json_from_text(text)
        self.assertEqual(result, {"result": "yes"})

    def test_multiline_think_tags_stripped(self):
        text = "<think>\nReasoning step 1.\nReasoning step 2.\n</think>\n{\"done\": true}"
        from modules.llm_service import _extract_json_from_text
        result = _extract_json_from_text(text)
        self.assertEqual(result, {"done": True})


# ═════════════════════════════════════════════════════════════════════════════
# 16 — Markdown fence handling (already covered in TestOllamaJsonParsing,
#      but explicit here for clarity in the test count)
# ═════════════════════════════════════════════════════════════════════════════

class TestMarkdownFenceHandling(unittest.TestCase):
    """Test #16 — _extract_json_from_text handles markdown code fences."""

    def test_json_code_fence_parsed(self):
        from modules.llm_service import _extract_json_from_text
        result = _extract_json_from_text("```json\n{\"answer\": 99}\n```")
        self.assertEqual(result["answer"], 99)

    def test_plain_code_fence_parsed(self):
        from modules.llm_service import _extract_json_from_text
        result = _extract_json_from_text("```\n{\"x\": \"y\"}\n```")
        self.assertEqual(result["x"], "y")


# ═════════════════════════════════════════════════════════════════════════════
# 17 — JSON reminder appended in JSON mode
# ═════════════════════════════════════════════════════════════════════════════

class TestJsonReminderInPrompt(unittest.TestCase):
    """Test #17 — JSON mode appends an explicit JSON output instruction to the prompt."""

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post")
    def test_json_mode_appends_reminder(self, mock_post):
        """The JSON reminder must be present in the payload sent to Ollama."""
        mock_post.return_value = _make_ollama_response('{"ok": true}')
        from modules.llm_service import generate_json
        generate_json("Base prompt text.")
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1].get("json") or call_kwargs[0][1]
        messages = payload["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        self.assertIn("CRITICAL OUTPUT REQUIREMENT", user_msg["content"])
        self.assertIn("valid JSON object", user_msg["content"])

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post")
    def test_text_mode_does_not_append_json_reminder(self, mock_post):
        """Text mode must NOT append the JSON output requirement to the prompt."""
        mock_post.return_value = _make_ollama_response("Plain text response.")
        from modules.llm_service import generate_text
        generate_text("Base prompt text.")
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1].get("json") or call_kwargs[0][1]
        messages = payload["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        self.assertNotIn("CRITICAL OUTPUT REQUIREMENT", user_msg["content"])


# ═════════════════════════════════════════════════════════════════════════════
# 18 — system_instruction forwarded as system message
# ═════════════════════════════════════════════════════════════════════════════

class TestSystemInstructionForwarding(unittest.TestCase):
    """Test #18 — system_instruction is forwarded as a system-role message to Ollama."""

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post")
    def test_system_instruction_sent_as_system_message(self, mock_post):
        mock_post.return_value = _make_ollama_response("Response with system msg.")
        from modules.llm_service import generate_text
        generate_text("User prompt.", system_instruction="You are a teacher.")
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1].get("json") or call_kwargs[0][1]
        messages = payload["messages"]
        roles = [m["role"] for m in messages]
        self.assertIn("system", roles)
        sys_msg = next(m for m in messages if m["role"] == "system")
        self.assertEqual(sys_msg["content"], "You are a teacher.")

    @patch("modules.llm_service.AI_PROVIDER", "ollama")
    @patch("modules.llm_service.requests.post")
    def test_no_system_message_when_instruction_is_none(self, mock_post):
        mock_post.return_value = _make_ollama_response("Response without system.")
        from modules.llm_service import generate_text
        generate_text("User prompt.")
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1].get("json") or call_kwargs[0][1]
        messages = payload["messages"]
        roles = [m["role"] for m in messages]
        self.assertNotIn("system", roles)


# ═════════════════════════════════════════════════════════════════════════════
# Bonus: Gemini retry infrastructure still intact under Ollama provider
# ═════════════════════════════════════════════════════════════════════════════

class TestGeminiRetryInfrastructurePreserved(unittest.TestCase):
    """
    Verify that the Gemini retry infrastructure is not broken by the Ollama
    additions.  These tests run _call_gemini_with_retry directly — no Ollama
    calls are made.
    """

    def test_call_gemini_with_retry_is_importable(self):
        from modules.llm_service import _call_gemini_with_retry
        self.assertTrue(callable(_call_gemini_with_retry))

    def test_max_retries_and_backoff_constants_present(self):
        from modules.llm_service import _MAX_RETRIES, _BACKOFF_SECONDS
        self.assertIsInstance(_MAX_RETRIES, int)
        self.assertGreater(_MAX_RETRIES, 0)
        self.assertIsInstance(_BACKOFF_SECONDS, list)
        self.assertEqual(len(_BACKOFF_SECONDS), _MAX_RETRIES)

    @patch("modules.llm_service.time.sleep", return_value=None)
    def test_retry_succeeds_on_second_attempt(self, mock_sleep):
        """_call_gemini_with_retry retries 503 and returns on success."""
        from modules.llm_service import _call_gemini_with_retry
        call_count = {"n": 0}

        def flaky():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception("503 UNAVAILABLE: high demand")
            return "success"

        result = _call_gemini_with_retry(flaky)
        self.assertEqual(result, "success")
        self.assertEqual(call_count["n"], 2)
        mock_sleep.assert_called_once()

    def test_timeout_error_not_retried(self):
        """TimeoutError is wrapped in RuntimeError immediately, no retry."""
        from modules.llm_service import _call_gemini_with_retry
        call_count = {"n": 0}

        def timeout_fn():
            call_count["n"] += 1
            raise TimeoutError("deadline exceeded")

        with self.assertRaises(RuntimeError) as ctx:
            _call_gemini_with_retry(timeout_fn)
        self.assertEqual(call_count["n"], 1)  # No retries
        self.assertIn("timed out", str(ctx.exception).lower())


# ═════════════════════════════════════════════════════════════════════════════
# 13 — Ollama 4B Local Model Optimizations
# ═════════════════════════════════════════════════════════════════════════════

class TestOllamaOptimizations(unittest.TestCase):
    """
    Tests ensuring Ollama 4B local model optimizations:
    1. OLLAMA_CHUNK_CHAR_LIMIT is defined and appropriately bounded.
    2. Zero extra LLM calls are made during Ollama text condensation.
    3. Large text is safely bounded by extractive condensation.
    4. Options dict (temperature=0.2, num_predict=1024) is sent in Ollama payload.
    5. Ollama-specific concise prompts are loaded when AI_PROVIDER='ollama'.
    6. In-memory condensation cache reuses results without duplicate processing.
    """

    def setUp(self):
        from modules.combined_generator import _CONDENSED_CACHE
        _CONDENSED_CACHE.clear()
        self._provider_patcher = patch("modules.llm_service.AI_PROVIDER", "ollama")
        self._provider_patcher.start()

    def tearDown(self):
        self._provider_patcher.stop()
        from modules.combined_generator import _CONDENSED_CACHE
        _CONDENSED_CACHE.clear()

    def test_ollama_chunk_limit_configured(self):
        from config.settings import OLLAMA_CHUNK_CHAR_LIMIT
        self.assertIsInstance(OLLAMA_CHUNK_CHAR_LIMIT, int)
        self.assertLessEqual(OLLAMA_CHUNK_CHAR_LIMIT, 6000)
        self.assertGreaterEqual(OLLAMA_CHUNK_CHAR_LIMIT, 2000)

    def test_ollama_condense_text_small_text_bypasses(self):
        from modules.combined_generator import _condense_text
        small_text = "Short chapter content."
        with patch("modules.combined_generator.generate_text") as mock_gen_text:
            result = _condense_text(small_text)
        self.assertEqual(result, small_text)
        mock_gen_text.assert_not_called()

    def test_ollama_condense_text_large_text_zero_llm_calls(self):
        from modules.combined_generator import _condense_text
        from config.settings import OLLAMA_CHUNK_CHAR_LIMIT
        large_text = "Paragraph about cell biology with definitions and facts. " * 200
        self.assertGreater(len(large_text), OLLAMA_CHUNK_CHAR_LIMIT)

        with patch("modules.combined_generator.generate_text") as mock_gen_text, \
             patch("modules.combined_generator.generate_json") as mock_gen_json:
            result = _condense_text(large_text)

        # Zero LLM calls must be made!
        mock_gen_text.assert_not_called()
        mock_gen_json.assert_not_called()
        self.assertLess(len(result), len(large_text))
        self.assertLessEqual(len(result), OLLAMA_CHUNK_CHAR_LIMIT + 50)

    def test_ollama_payload_includes_options(self):
        from modules.llm_service import _call_ollama
        with patch("modules.llm_service.requests.post") as mock_post:
            mock_post.return_value = _make_ollama_response("Hello")
            _call_ollama("Test prompt")
            call_kwargs = mock_post.call_args[1]
            payload = call_kwargs.get("json", {})
            self.assertIn("options", payload)
            self.assertEqual(payload["options"].get("temperature"), 0.2)
            self.assertEqual(payload["options"].get("num_predict"), 1024)
            self.assertIn("think", payload)
            self.assertFalse(payload["think"])

    def test_ollama_payload_disables_thinking(self):
        """Verify that _call_ollama explicitly passes think=False to prevent token exhaustion."""
        from modules.llm_service import _call_ollama
        with patch("modules.llm_service.requests.post") as mock_post:
            mock_post.return_value = _make_ollama_response("Hello")
            _call_ollama("Test prompt")
            call_kwargs = mock_post.call_args[1]
            payload = call_kwargs.get("json", {})
            self.assertIn("think", payload)
            self.assertFalse(payload["think"])

    def test_ollama_default_model_is_qwen35_2b(self):
        """Verify that the default Ollama model in settings is qwen3.5:2b."""
        from config.settings import OLLAMA_MODEL
        self.assertEqual(OLLAMA_MODEL, "qwen3.5:2b")

    def test_ollama_payload_uses_default_model(self):
        """Verify that _call_ollama sends qwen3.5:2b in its request payload."""
        from modules.llm_service import _call_ollama
        with patch("modules.llm_service.requests.post") as mock_post:
            mock_post.return_value = _make_ollama_response("Hello")
            _call_ollama("Test prompt")
            call_kwargs = mock_post.call_args[1]
            payload = call_kwargs.get("json", {})
            self.assertEqual(payload.get("model"), "qwen3.5:2b")

    def test_ollama_uses_concise_analysis_prompt(self):
        from modules.combined_generator import generate_analysis_and_simplified
        captured = {}
        def mock_gen_json(prompt, system_instruction=None):
            captured["prompt"] = prompt
            return {
                "analysis": {
                    "chapter_title": "T", "topic": "Top",
                    "key_concepts": [], "key_terms": [], "important_facts": [],
                    "learning_objectives": [], "difficulty": "Beginner",
                    "prerequisite_concepts": []
                },
                "simplified": {
                    "chapter_title": "T", "overview": "O",
                    "key_concepts": [], "detailed_explanation": "D",
                    "important_terms": [], "quick_summary": "S"
                }
            }
        with patch("modules.combined_generator.generate_json", side_effect=mock_gen_json):
            generate_analysis_and_simplified("Some text", level="Beginner")
        self.assertIn("under 120 words", captured["prompt"])

    def test_ollama_uses_concise_comic_quiz_prompt(self):
        from modules.combined_generator import generate_comic_and_quiz
        captured = {}
        def mock_gen_json(prompt, system_instruction=None):
            captured["prompt"] = prompt
            return {
                "comic": {
                    "comic_title": "C",
                    "panels": [{"panel_number": i, "scene": "S", "visual_description": "V", "characters": [], "narration": "N", "dialogue": "D"} for i in range(1, 7)]
                },
                "quiz": {
                    "quiz_title": "Q",
                    "questions": [{"question": "Q1?", "options": ["A", "B", "C", "D"], "correct_answer": "A", "explanation": "E"}]
                }
            }
        with patch("modules.combined_generator.generate_json", side_effect=mock_gen_json):
            generate_comic_and_quiz("Some text")
        self.assertIn("under 15 words", captured["prompt"])

    def test_condense_cache_prevents_duplicate_work(self):
        from modules.combined_generator import _condense_text, _CONDENSED_CACHE
        text = "Sample text for caching test"
        res1 = _condense_text(text)
        self.assertIn((hash(text), "ollama"), _CONDENSED_CACHE)
        res2 = _condense_text(text)
        self.assertEqual(res1, res2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

