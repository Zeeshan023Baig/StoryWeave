from google import genai
from google.genai import types
import json
import time
from config.settings import GEMINI_API_KEY, MODEL_NAME

# ──────────────────────────────────────────────────────────────
# Retry configuration
# ──────────────────────────────────────────────────────────────
_MAX_RETRIES = 3
_BACKOFF_SECONDS = [2, 5, 10]   # wait before attempt 2, 3, 4

# Error substrings that should NOT be retried (auth / validation)
_NO_RETRY_SUBSTRINGS = [
    "401", "403",
    "API_KEY_INVALID", "PERMISSION_DENIED", "INVALID_ARGUMENT", "400",
]

# Error substrings that signal 429 rate-limit / quota
_QUOTA_SUBSTRINGS = [
    "429", "RESOURCE_EXHAUSTED", "quota", "rate",
]

# Error substrings that signal 503 transient unavailability
_UNAVAILABLE_SUBSTRINGS = [
    "503", "UNAVAILABLE", "high demand", "temporarily",
]


def get_client():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing. Please configure it in .env.")
    return genai.Client(api_key=GEMINI_API_KEY)


def _is_no_retry_error(error_str: str) -> bool:
    """Return True for auth/bad-request errors that retrying cannot fix."""
    lower = error_str.lower()
    return any(s.lower() in lower for s in _NO_RETRY_SUBSTRINGS)


def _is_quota_error(error_str: str) -> bool:
    """Return True for 429 / quota-exhausted errors."""
    lower = error_str.lower()
    return any(s.lower() in lower for s in _QUOTA_SUBSTRINGS)


def _is_unavailable_error(error_str: str) -> bool:
    """Return True for 503 / high-demand transient errors."""
    lower = error_str.lower()
    return any(s.lower() in lower for s in _UNAVAILABLE_SUBSTRINGS)


def _call_gemini_with_retry(call_fn):
    """
    Execute call_fn() with automatic retry for 503 UNAVAILABLE errors.

    Retry policy
    ─────────────
    • 503 / UNAVAILABLE  → retry up to _MAX_RETRIES times with exponential
                           backoff (_BACKOFF_SECONDS[attempt]).
    • 429 / quota        → raise immediately as RuntimeError (no retry).
    • auth / bad-request → raise immediately as RuntimeError (no retry).
    • other errors       → raise immediately.

    Returns the value returned by call_fn() on success.
    """
    last_exc = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            return call_fn()
        except Exception as exc:
            error_str = str(exc)

            # Never retry auth / validation errors
            if _is_no_retry_error(error_str):
                raise RuntimeError(f"Gemini API Error: {error_str}") from exc

            # Never retry quota errors
            if _is_quota_error(error_str):
                raise RuntimeError(f"Gemini API Error: {error_str}") from exc

            # Retry only on 503 / UNAVAILABLE
            if _is_unavailable_error(error_str):
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    wait = _BACKOFF_SECONDS[attempt]
                    time.sleep(wait)
                    continue
                else:
                    raise RuntimeError(f"Gemini API Error: {error_str}") from exc

            # All other errors – raise immediately
            raise RuntimeError(f"Gemini API Error: {error_str}") from exc

    # Fallback (should never reach here)
    raise RuntimeError(f"Gemini API Error: {str(last_exc)}")


def generate_text(prompt, system_instruction=None):
    client = get_client()

    def _call():
        config_kwargs = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config
        )
        return response.text

    return _call_gemini_with_retry(_call)


def generate_json(prompt, system_instruction=None):
    client = get_client()

    def _call():
        config_kwargs = {"response_mime_type": "application/json"}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        config = types.GenerateContentConfig(**config_kwargs)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config
        )
        text_response = response.text
        try:
            return json.loads(text_response.strip())
        except json.JSONDecodeError:
            raise ValueError("Failed to parse JSON from Gemini response.")

    try:
        return _call_gemini_with_retry(_call)
    except ValueError:
        # Preserve JSON parse errors as ValueError (not RuntimeError)
        raise
