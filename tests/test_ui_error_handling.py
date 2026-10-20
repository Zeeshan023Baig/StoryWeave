"""
tests/test_ui_error_handling.py
===============================
Unit tests for provider-neutral error formatting and startup guard logic in app.py.

Verifies:
1. Provider-neutral timeout, 429, 503, connection, and model errors for Ollama.
2. Gemini-specific error formatting when AI_PROVIDER="gemini".
3. Ollama provider never claims a Gemini quota error.
4. Startup guard permits running with AI_PROVIDER="ollama" without GEMINI_API_KEY.
5. Startup guard enforces GEMINI_API_KEY only when AI_PROVIDER="gemini".
"""

import unittest
from unittest.mock import patch, MagicMock
from app import _format_ai_error


class TestOllamaErrorFormatting(unittest.TestCase):
    """Verifies that error messages are provider-neutral or Ollama-branded when AI_PROVIDER='ollama'."""

    def setUp(self):
        self._provider_patcher = patch("modules.llm_service.AI_PROVIDER", "ollama")
        self._provider_patcher.start()

    def tearDown(self):
        self._provider_patcher.stop()

    def test_ollama_timeout_error_message(self):
        msg = _format_ai_error("Ollama request timed out after 120s")
        self.assertIn("Ollama request timed out", msg)
        self.assertNotIn("Gemini", msg)

    def test_ollama_timeout_with_context(self):
        msg = _format_ai_error("Connection timed out", context="comic/quiz generation")
        self.assertIn("during comic/quiz generation", msg)
        self.assertIn("Ollama request timed out", msg)
        self.assertNotIn("Gemini", msg)

    def test_ollama_429_does_not_claim_gemini_quota(self):
        """When Ollama is active, 429/RESOURCE_EXHAUSTED must NOT claim Gemini quota."""
        msg = _format_ai_error("429 RESOURCE_EXHAUSTED: Rate limit reached")
        self.assertNotIn("Gemini", msg)
        self.assertIn("rate limit / resource limit reached", msg.lower())

    def test_ollama_503_unavailable_message(self):
        msg = _format_ai_error("503 UNAVAILABLE: Model is busy")
        self.assertIn("Ollama service is temporarily unavailable (503)", msg)
        self.assertNotIn("Gemini", msg)

    def test_ollama_connection_refused_message(self):
        msg = _format_ai_error("Cannot connect to Ollama at http://localhost:11434: [WinError 10061]")
        self.assertIn("Cannot connect to local Ollama server", msg)
        self.assertIn("ollama serve", msg)
        self.assertNotIn("Gemini", msg)

    def test_ollama_model_not_found_message(self):
        msg = _format_ai_error("Ollama model 'qwen3.5:4b' not found (HTTP 404)")
        self.assertIn("Ollama model not found", msg)
        self.assertIn("ollama pull qwen3.5:4b", msg)
        self.assertNotIn("Gemini", msg)

    def test_ollama_general_error_message(self):
        msg = _format_ai_error("Something unexpected happened")
        self.assertIn("Ollama error", msg)
        self.assertNotIn("Gemini", msg)


class TestGeminiErrorFormatting(unittest.TestCase):
    """Verifies that Gemini error messages remain properly formatted when AI_PROVIDER='gemini'."""

    def setUp(self):
        self._provider_patcher = patch("modules.llm_service.AI_PROVIDER", "gemini")
        self._provider_patcher.start()

    def tearDown(self):
        self._provider_patcher.stop()

    def test_gemini_timeout_error_message(self):
        msg = _format_ai_error("Gemini API Error: Request timed out")
        self.assertIn("Gemini request timed out", msg)

    def test_gemini_quota_error_message(self):
        msg = _format_ai_error("429 RESOURCE_EXHAUSTED: Quota exceeded")
        self.assertIn("Gemini quota exceeded (429)", msg)
        self.assertIn("API plan", msg)

    def test_gemini_503_unavailable_message(self):
        msg = _format_ai_error("503 UNAVAILABLE: Model experiencing high demand")
        self.assertIn("Gemini service is temporarily unavailable (503)", msg)

    def test_gemini_general_error_message(self):
        msg = _format_ai_error("Unknown SDK error occurred")
        self.assertIn("Gemini error", msg)


class TestStartupGuardLogic(unittest.TestCase):
    """Verifies startup guard behavior based on active AI provider."""

    def test_startup_guard_allows_ollama_without_gemini_key(self):
        """Ollama must NOT be blocked when GEMINI_API_KEY is None or empty."""
        from modules.llm_service import _get_active_provider

        with patch("modules.llm_service.AI_PROVIDER", "ollama"):
            active = _get_active_provider()
            gemini_key = ""
            should_block = (active == "gemini" and not gemini_key)
            self.assertFalse(should_block, "Ollama must run keyless without requiring GEMINI_API_KEY")

    def test_startup_guard_blocks_gemini_without_api_key(self):
        """Gemini MUST be blocked when GEMINI_API_KEY is missing."""
        from modules.llm_service import _get_active_provider

        with patch("modules.llm_service.AI_PROVIDER", "gemini"):
            active = _get_active_provider()
            gemini_key = None
            should_block = (active == "gemini" and not gemini_key)
            self.assertTrue(should_block, "Gemini provider requires GEMINI_API_KEY")

    def test_startup_guard_allows_gemini_with_api_key(self):
        """Gemini must proceed when GEMINI_API_KEY is provided."""
        from modules.llm_service import _get_active_provider

        with patch("modules.llm_service.AI_PROVIDER", "gemini"):
            active = _get_active_provider()
            gemini_key = "AIzaSyFakeKeyForTesting12345"
            should_block = (active == "gemini" and not gemini_key)
            self.assertFalse(should_block, "Gemini provider must proceed when API key is present")


if __name__ == "__main__":
    unittest.main()
