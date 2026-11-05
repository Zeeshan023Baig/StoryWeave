import os
import requests
import time
from pathlib import Path
from dotenv import load_dotenv

# Explicitly load .env from StoryWeave project root regardless of cwd
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

_PIXAZO_URL = "https://gateway.pixazo.ai/flux/text-to-image"
_STATUS_URL_TEMPLATE = "https://gateway.pixazo.ai/v2/requests/status/{request_id}"
_MAX_RETRIES = 10
_POLL_INTERVAL_SEC = 2


def _generate_image_pixazo(prompt: str) -> bytes:
    """
    Generates an educational comic illustration from a text prompt using the
    Pixazo Flux API and returns raw PNG/JPEG bytes for st.image().

    Requires PIXAZO_API_KEY in .env.
    Raises RuntimeError with an actionable message on any failure.
    """
    api_key = os.getenv("PIXAZO_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PIXAZO_API_KEY is not set in .env. "
            "Please configure it to enable comic image generation."
        )

    enhanced_prompt = (
        f"Educational comic-book illustration style, colorful educational illustration, "
        f"consistent comic/storybook style, clear characters and environment, "
        f"no copyrighted characters, no unnecessary text inside the image.\n\nDescription: {prompt}"
    )

    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/json"
    }
    
    data = {
        "prompt": enhanced_prompt
    }

    try:
        response = requests.post(_PIXAZO_URL, headers=headers, json=data)
        
        # Pixazo API might be down or endpoint might have changed if it returns 404
        if response.status_code == 404:
            raise RuntimeError(f"Pixazo API free endpoint unavailable (404 Resource not found).")
            
        response.raise_for_status()
        resp_json = response.json()
        
        # Handle async polling if a request ID is returned instead of direct URL
        if "id" in resp_json and "url" not in resp_json:
            request_id = resp_json["id"]
            status_url = _STATUS_URL_TEMPLATE.format(request_id=request_id)
            
            for _ in range(_MAX_RETRIES):
                time.sleep(_POLL_INTERVAL_SEC)
                status_resp = requests.get(status_url, headers=headers)
                status_resp.raise_for_status()
                status_data = status_resp.json()
                
                if status_data.get("status") == "completed" and "url" in status_data:
                    image_url = status_data["url"]
                    break
                elif status_data.get("status") in ["failed", "error"]:
                    raise RuntimeError(f"Pixazo async generation failed: {status_data}")
            else:
                raise RuntimeError("Pixazo API polling timed out.")
        else:
            image_url = resp_json.get("url") or resp_json.get("image_url")
            if not image_url:
                # If there's an image base64 directly returned
                if "image" in resp_json:
                    import base64
                    return base64.b64decode(resp_json["image"])
                raise RuntimeError(f"Pixazo API did not return an image URL. Response: {resp_json}")
        
        # Download the actual image bytes
        img_resp = requests.get(image_url)
        img_resp.raise_for_status()
        return img_resp.content

    except requests.exceptions.RequestException as e:
        if e.response is not None:
            if e.response.status_code == 401:
                raise RuntimeError("Invalid PIXAZO_API_KEY. Check your API token in .env.")
            if e.response.status_code == 429:
                raise RuntimeError("Pixazo rate limit or quota exceeded (429). Wait a moment and retry.")
        raise RuntimeError(f"Pixazo API error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during image generation: {str(e)[:300]}")


def _generate_image_gemini(prompt: str) -> bytes:
    """
    Generates an educational comic panel illustration using Gemini's
    native image-generation API and returns raw PNG/JPEG bytes for st.image().

    Requires GEMINI_API_KEY in .env.
    Raises RuntimeError with an actionable message on any failure.
    """
    from modules.llm_service import get_client
    from config.settings import GEMINI_IMAGE_MODEL

    client = get_client()

    enhanced_prompt = (
        f"Educational comic-book panel illustration, clean colorful comic art style, "
        f"clear characters and educational setting, high visual clarity, "
        f"no text, no speech bubbles inside the image.\n\nScene: {prompt}"
    )

    try:
        if "imagen" in GEMINI_IMAGE_MODEL.lower():
            result = client.models.generate_images(
                model=GEMINI_IMAGE_MODEL,
                prompt=enhanced_prompt,
                config=dict(number_of_images=1, output_mime_type="image/png"),
            )
            if result.generated_images:
                return result.generated_images[0].image.image_bytes
            raise RuntimeError("Gemini returned no generated images.")
        else:
            response = client.models.generate_content(
                model=GEMINI_IMAGE_MODEL,
                contents=enhanced_prompt,
            )
            if not response.candidates or not response.candidates[0].content:
                raise RuntimeError("Gemini returned an empty image generation response.")

            for part in response.candidates[0].content.parts:
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    data = part.inline_data.data
                    if isinstance(data, str):
                        import base64
                        return base64.b64decode(data)
                    return data

            raise RuntimeError("Gemini response did not contain image data.")

    except Exception as exc:
        err_str = str(exc)
        if "402" in err_str or "prepayment" in err_str.lower():
            raise RuntimeError(
                "Gemini image generation failed: Prepayment credits depleted (402). "
                "Please check your Google AI Studio billing/credits."
            ) from exc
        if "429" in err_str or "resource_exhausted" in err_str.lower():
            raise RuntimeError(
                "Gemini image generation failed: Rate limit or quota exceeded (429). "
                "Wait a moment and try again."
            ) from exc
        if "404" in err_str:
            raise RuntimeError(
                f"Gemini image generation failed: Model '{GEMINI_IMAGE_MODEL}' not found (404)."
            ) from exc
        raise RuntimeError(f"Gemini image generation failed: {err_str}") from exc


def get_image_provider(provider_override: str | None = None) -> str:
    """
    Resolve active image provider ('gemini' | 'pixazo').

    Resolution order:
      1. Explicit provider_override argument (if passed; 'ollama' maps to 'pixazo')
      2. Module-level IMAGE_PROVIDER (supports unittest.mock.patch("modules.image_generator.IMAGE_PROVIDER", ...))
      3. config.settings.IMAGE_PROVIDER (supports config-level patch or environment switch)
      4. Fallback based on text provider: 'gemini' if AI_PROVIDER == 'gemini' else 'pixazo'
    """
    if provider_override:
        p = str(provider_override).lower().strip()
        return "pixazo" if p == "ollama" else p

    mod_val = globals().get("IMAGE_PROVIDER")
    try:
        import config.settings as _settings
        cfg_val = getattr(_settings, "IMAGE_PROVIDER", None)
    except Exception:
        cfg_val = None

    if mod_val is not None and str(mod_val).strip():
        val = str(mod_val).lower().strip()
        return "pixazo" if val == "ollama" else val

    if cfg_val is not None and str(cfg_val).strip():
        val = str(cfg_val).lower().strip()
        return "pixazo" if val == "ollama" else val

    from modules.llm_service import _get_active_provider
    return "gemini" if _get_active_provider() == "gemini" else "pixazo"


def generate_image_bytes(prompt: str, provider: str | None = None) -> bytes:
    """
    Public entry point for image generation.
    Routes to Gemini when active provider is 'gemini', otherwise uses Pixazo.
    Never silently falls back from Gemini to Pixazo.
    """
    active = get_image_provider(provider)

    if active == "gemini":
        return _generate_image_gemini(prompt)
    else:
        return _generate_image_pixazo(prompt)
