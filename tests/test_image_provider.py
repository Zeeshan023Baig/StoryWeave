"""
Mocked tests for independent IMAGE_PROVIDER routing (Gemini images while Ollama
handles text). No real network / paid API calls are made.
"""
import base64
import unittest
from unittest.mock import patch, MagicMock

PNG = b"\x89PNG\r\n\x1a\nfake"


def _gemini_response(data):
    part = MagicMock()
    part.inline_data.data = data
    cand = MagicMock()
    cand.content.parts = [part]
    resp = MagicMock()
    resp.candidates = [cand]
    return resp


class TestImageProviderRouting(unittest.TestCase):
    @patch("modules.image_generator._generate_image_pixazo")
    @patch("modules.image_generator._generate_image_gemini")
    def test_image_provider_gemini_independent_of_ollama_text(self, mock_gem, mock_pix):
        mock_gem.return_value = PNG
        with patch("config.settings.IMAGE_PROVIDER", "gemini"), \
             patch("modules.llm_service.AI_PROVIDER", "ollama"), \
             patch("config.settings.AI_PROVIDER", "ollama"):
            from modules.image_generator import generate_image_bytes
            self.assertEqual(generate_image_bytes("scene"), PNG)
        mock_gem.assert_called_once_with("scene")
        mock_pix.assert_not_called()

    @patch("modules.image_generator._generate_image_pixazo")
    @patch("modules.image_generator._generate_image_gemini")
    def test_image_provider_pixazo_preserved(self, mock_gem, mock_pix):
        mock_pix.return_value = b"pix"
        with patch("config.settings.IMAGE_PROVIDER", "pixazo"):
            from modules.image_generator import generate_image_bytes
            self.assertEqual(generate_image_bytes("scene"), b"pix")
        mock_pix.assert_called_once_with("scene")
        mock_gem.assert_not_called()

    @patch("modules.image_generator._generate_image_pixazo")
    @patch("modules.image_generator._generate_image_gemini")
    def test_unset_image_provider_falls_back_to_pixazo_for_ollama(self, mock_gem, mock_pix):
        mock_pix.return_value = b"pix"
        with patch("config.settings.IMAGE_PROVIDER", ""), \
             patch("modules.llm_service.AI_PROVIDER", "ollama"), \
             patch("config.settings.AI_PROVIDER", "ollama"):
            from modules.image_generator import generate_image_bytes
            generate_image_bytes("scene")
        mock_pix.assert_called_once()
        mock_gem.assert_not_called()

    @patch("modules.image_generator._generate_image_pixazo")
    @patch("modules.image_generator._generate_image_gemini")
    def test_gemini_failure_never_calls_pixazo(self, mock_gem, mock_pix):
        """When IMAGE_PROVIDER is gemini and Gemini fails, it must NEVER fall back to Pixazo."""
        mock_gem.side_effect = RuntimeError("Gemini image generation failed: 402 credits")
        with patch("config.settings.IMAGE_PROVIDER", "gemini"):
            from modules.image_generator import generate_image_bytes
            with self.assertRaises(RuntimeError) as ctx:
                generate_image_bytes("scene")
            self.assertIn("Gemini image generation failed", str(ctx.exception))
            self.assertNotIn("Pixazo", str(ctx.exception))
        mock_gem.assert_called_once()
        mock_pix.assert_not_called()


class TestGeminiSixPanels(unittest.TestCase):
    def _panels(self):
        return [{"panel_number": i + 1,
                 "visual_description": f"Panel {i + 1} visual"} for i in range(6)]

    @patch("modules.llm_service.get_client")
    def test_six_separate_requests_and_display_data(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client
        client.models.generate_content.side_effect = [
            _gemini_response(PNG + bytes([i])) for i in range(6)
        ]
        from modules.image_generator import generate_image_bytes

        images = {}
        with patch("config.settings.IMAGE_PROVIDER", "gemini"):
            for i, p in enumerate(self._panels()):
                images[i] = generate_image_bytes(p["visual_description"])

        self.assertEqual(client.models.generate_content.call_count, 6)
        prompts = [c.kwargs["contents"] for c in client.models.generate_content.call_args_list]
        for i, pr in enumerate(prompts):
            self.assertIn(f"Panel {i + 1} visual", pr)
        self.assertEqual(len(set(images.values())), 6)          # six distinct images
        self.assertEqual(sorted(images), list(range(6)))        # keyed 0..5 for st.image
        self.assertTrue(all(isinstance(v, bytes) for v in images.values()))

    @patch("modules.llm_service.get_client")
    def test_byte_extraction_raw_and_base64(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client
        from modules.image_generator import _generate_image_gemini

        client.models.generate_content.return_value = _gemini_response(PNG)
        self.assertEqual(_generate_image_gemini("x"), PNG)

        client.models.generate_content.return_value = _gemini_response(
            base64.b64encode(PNG).decode())
        self.assertEqual(_generate_image_gemini("x"), PNG)

    @patch("modules.llm_service.get_client")
    def test_graceful_failure_other_panels_continue(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client
        client.models.generate_content.side_effect = [
            _gemini_response(PNG), Exception("429 RESOURCE_EXHAUSTED"),
            _gemini_response(PNG), _gemini_response(PNG),
            _gemini_response(PNG), _gemini_response(PNG),
        ]
        from modules.image_generator import generate_image_bytes

        images, failures = {}, []
        with patch("config.settings.IMAGE_PROVIDER", "gemini"):
            for i, p in enumerate(self._panels()):   # mirrors app.py's per-panel try/except
                try:
                    images[i] = generate_image_bytes(p["visual_description"])
                except RuntimeError as e:
                    failures.append(str(e))
        self.assertEqual(len(images), 5)
        self.assertEqual(len(failures), 1)
        self.assertIn("429", failures[0])

    @patch("modules.llm_service.get_client")
    def test_no_image_part_raises_runtime_error(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client
        part = MagicMock()
        part.inline_data = None
        resp = MagicMock()
        resp.candidates = [MagicMock()]
        resp.candidates[0].content.parts = [part]
        client.models.generate_content.return_value = resp
        from modules.image_generator import _generate_image_gemini
        with self.assertRaises(RuntimeError):
            _generate_image_gemini("x")


class TestOllamaTextUnchanged(unittest.TestCase):
    @patch("modules.llm_service.requests.post")
    def test_ollama_payload_unchanged_with_gemini_images(self, mock_post):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"message": {"content": "ok"}}
        mock_post.return_value = resp
        with patch("config.settings.IMAGE_PROVIDER", "gemini"):
            from modules.llm_service import _call_ollama
            self.assertEqual(_call_ollama("hi"), "ok")
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "qwen3.5:2b")
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["options"]["num_predict"], 1024)
        self.assertEqual(mock_post.call_args.kwargs["timeout"], 120)


if __name__ == "__main__":
    unittest.main()

