"""
LLM service — dual-provider implementation.

Supported providers
-------------------
  ollama  : Routes all calls to a local Ollama server (default).
            No API key required.  Model: qwen3.5:2b
  gemini  : Routes all calls to the Google Gemini cloud API.
            Requires GEMINI_API_KEY in .env.

The active provider is selected by AI_PROVIDER in config/settings.py.
Switching providers only requires changing that one constant — no other
file needs to be modified.

Public interface (unchanged — combined_generator.py imports these)
------------------------------------------------------------------
  generate_text(prompt, system_instruction=None) -> str
  generate_json(prompt, system_instruction=None) -> dict

All other names (_call_gemini_with_retry, get_client, etc.) are preserved
for backward compatibility with the existing test suite.
"""

from google import genai
from google.genai import types
import json
import re
import time
import requests

from config.settings import (
    AI_PROVIDER,
    GEMINI_API_KEY,
    MODEL_NAME,
    THINKING_LEVEL,
    GEMINI_TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
)

# ──────────────────────────────────────────────────────────────
# Retry configuration (shared)
# ──────────────────────────────────────────────────────────────
_MAX_RETRIES = 3
_BACKOFF_SECONDS = [2, 5, 10]   # wait before attempt 2, 3, 4

# Error substrings that should NOT be retried (auth / validation / not found)
_NO_RETRY_SUBSTRINGS = [
    "401", "403", "404",
    "API_KEY_INVALID", "PERMISSION_DENIED", "INVALID_ARGUMENT", "400", "NOT_FOUND",
]

# Error substrings that signal 429 rate-limit / quota
_QUOTA_SUBSTRINGS = [
    "429", "RESOURCE_EXHAUSTED", "quota", "rate",
]

# Error substrings that signal 503 transient unavailability
_UNAVAILABLE_SUBSTRINGS = [
    "503", "UNAVAILABLE", "high demand", "temporarily",
]

# Error substrings that signal timeout
_TIMEOUT_SUBSTRINGS = [
    "timeout", "timed out", "deadline exceeded", "504", "gateway timeout",
]


# ══════════════════════════════════════════════════════════════
# GEMINI PROVIDER — original implementation, fully preserved
# ══════════════════════════════════════════════════════════════

def get_client():
    """Return a configured Gemini client (raises if API key is missing)."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing. Please configure it in .env.")
    timeout_ms = int(GEMINI_TIMEOUT_SECONDS * 1000) if GEMINI_TIMEOUT_SECONDS else None
    http_opts = types.HttpOptions(timeout=timeout_ms) if timeout_ms else None
    return genai.Client(api_key=GEMINI_API_KEY, http_options=http_opts)


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


def _is_timeout_error(exc: Exception) -> bool:
    """Return True if exception represents a network or API timeout."""
    if isinstance(exc, TimeoutError):
        return True
    try:
        import httpx
        if isinstance(exc, (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout)):
            return True
    except ImportError:
        pass
    lower = str(exc).lower()
    return any(s in lower for s in _TIMEOUT_SUBSTRINGS)


def _call_gemini_with_retry(call_fn):
    """
    Execute call_fn() with automatic retry for 503 UNAVAILABLE errors.

    Retry policy
    ─────────────
    • 503 / UNAVAILABLE  → retry up to _MAX_RETRIES times with exponential
                           backoff (_BACKOFF_SECONDS[attempt]).
    • timeout            → raise immediately as RuntimeError (no retry, avoids UI hanging).
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

            # Never retry timeout errors — fail fast to avoid leaving UI hanging
            if _is_timeout_error(exc):
                raise RuntimeError(f"Gemini API Error: Request timed out. {error_str}") from exc

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


def _generate_text_gemini(prompt, system_instruction=None):
    """Generate plain text via Gemini (original implementation)."""
    client = get_client()

    def _call():
        config_kwargs = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if THINKING_LEVEL:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=THINKING_LEVEL)
        if GEMINI_TIMEOUT_SECONDS:
            config_kwargs["http_options"] = types.HttpOptions(timeout=int(GEMINI_TIMEOUT_SECONDS * 1000))
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config
        )
        return response.text

    return _call_gemini_with_retry(_call)


def _generate_json_gemini(prompt, system_instruction=None):
    """Generate structured JSON via Gemini (original implementation)."""
    client = get_client()

    def _call():
        config_kwargs = {"response_mime_type": "application/json"}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if THINKING_LEVEL:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=THINKING_LEVEL)
        if GEMINI_TIMEOUT_SECONDS:
            config_kwargs["http_options"] = types.HttpOptions(timeout=int(GEMINI_TIMEOUT_SECONDS * 1000))
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


# ══════════════════════════════════════════════════════════════
# OLLAMA PROVIDER — new local implementation
# ══════════════════════════════════════════════════════════════

def _extract_balanced_json_object(text: str, key: str) -> dict | None:
    """
    Search for a key (e.g. 'quiz', 'comic', 'analysis', 'simplified') and extract
    the matching balanced JSON dictionary starting at its opening '{'.
    Safe against strings, escape sequences, and nested braces.
    Returns parsed dict if valid, else None.
    """
    pattern = rf'"{re.escape(key)}"\s*:\s*\{{'
    match = re.search(pattern, text)
    if not match:
        return None
    start_idx = match.end() - 1

    depth = 0
    in_string = False
    escape = False
    end_idx = -1

    for i in range(start_idx, len(text)):
        char = text[i]
        if escape:
            escape = False
            continue
        if char == '\\' and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break

    if end_idx != -1:
        snippet = text[start_idx:end_idx].strip()
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return None


def _extract_json_from_text(text: str) -> dict:
    """
    Robustly extract and parse a JSON object from a model response string.

    Strategy (in order):
      1. Strip think-tags and whitespace, try direct json.loads.
      2. Strip markdown fences (```json ... ``` or unclosed ```) and try json.loads.
      3. Extract the first {...} block via regex.
      4. Partial-failure recovery: if top-level parse fails due to one malformed
         section, recover any complete balanced sub-objects for known keys
         ('comic', 'quiz', 'analysis', 'simplified').
      5. Raise ValueError with a clear message on failure.
    """
    # Strip common think-tag wrappers some models emit
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Attempt 1: parse cleaned text directly
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Attempt 2b: strip leading/trailing fence lines without strict pairing
    cleaned_no_fence = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned_no_fence = re.sub(r"\s*```$", "", cleaned_no_fence).strip()
    try:
        return json.loads(cleaned_no_fence)
    except json.JSONDecodeError:
        pass

    # Attempt 3: find first JSON object in the string
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Attempt 4: Partial recovery of balanced sub-objects for known compound structures
    recovered = {}
    for key in ("comic", "quiz", "analysis", "simplified"):
        sub_obj = _extract_balanced_json_object(cleaned, key)
        if sub_obj is not None:
            recovered[key] = sub_obj
    if recovered:
        return recovered

    raise ValueError(
        f"Failed to parse JSON from Ollama response. "
        f"Raw output (first 500 chars): {text[:500]!r}"
    )


def _build_ollama_messages(prompt: str, system_instruction: str | None) -> list:
    """Assemble the messages list for the Ollama /api/chat endpoint."""
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    return messages


def _call_ollama(
    prompt: str,
    system_instruction: str | None = None,
    json_mode: bool = False,
) -> str:
    """
    Make a single call to the local Ollama /api/chat endpoint.

    Parameters
    ----------
    prompt             : The user prompt text.
    system_instruction : Optional system-role message.
    json_mode          : If True, request JSON output format from Ollama.

    Returns
    -------
    str : The model's response text.

    Raises
    ------
    RuntimeError : On connection failure, timeout, non-2xx status, or model errors.
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    messages = _build_ollama_messages(prompt, system_instruction)

    payload: dict = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 1024,
        },
    }

    # Request JSON output format when needed — Ollama supports format="json"
    if json_mode:
        payload["format"] = "json"

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Ollama Error: Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            f"Is Ollama running? (ollama serve)  Detail: {exc}"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Ollama Error: Request timed out after {OLLAMA_TIMEOUT_SECONDS}s. "
            f"The model may still be loading or the prompt is too long.  Detail: {exc}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Ollama Error: Network error — {exc}") from exc

    if response.status_code == 404:
        raise RuntimeError(
            f"Ollama Error: Model '{OLLAMA_MODEL}' not found. "
            f"Install it with:  ollama pull {OLLAMA_MODEL}"
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama Error: HTTP {response.status_code} — {response.text[:300]}"
        )

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Ollama Error: Unexpected non-JSON response from server. Detail: {exc}"
        ) from exc

    # Ollama /api/chat returns {"message": {"role": "assistant", "content": "..."}}
    msg_obj = data.get("message", {})
    content = (
        msg_obj.get("content")
        or data.get("response")  # /api/generate fallback key
        or (msg_obj.get("thinking") if isinstance(msg_obj, dict) else "")
        or (msg_obj.get("reasoning") if isinstance(msg_obj, dict) else "")
        or ""
    )

    if not content:
        raise RuntimeError(
            "Ollama Error: Model returned an empty response. "
            "Check that the model is properly installed and not corrupted."
        )

    return content


def _generate_text_ollama(prompt: str, system_instruction: str | None = None) -> str:
    """Generate plain text via the local Ollama model."""
    return _call_ollama(prompt, system_instruction=system_instruction, json_mode=False)


def _generate_json_ollama(prompt: str, system_instruction: str | None = None) -> dict:
    """
    Generate structured JSON via the local Ollama model.

    Appends an explicit JSON instruction to the prompt and enables Ollama's
    format=json mode, then robustly parses the response with _extract_json_from_text.
    """
    # Append an explicit JSON reminder so the model knows what we expect
    json_reminder = (
        "\n\n## CRITICAL OUTPUT REQUIREMENT\n"
        "Your response MUST be a single valid JSON object only.\n"
        "Do NOT include any prose, explanation, markdown fences, or comments.\n"
        "Start your response with '{' and end with '}'.\n"
    )
    augmented_prompt = prompt + json_reminder

    raw_text = _call_ollama(
        augmented_prompt,
        system_instruction=system_instruction,
        json_mode=True,
    )

    try:
        return _extract_json_from_text(raw_text)
    except ValueError as exc:
        # Re-raise as ValueError so combined_generator.py error handling is preserved
        raise ValueError(str(exc)) from exc


# ══════════════════════════════════════════════════════════════
# PUBLIC INTERFACE — provider-dispatching wrappers
# ══════════════════════════════════════════════════════════════

def _get_active_provider(provider_override: str = None) -> str:
    """
    Return the active AI provider in lowercase ('gemini' or 'ollama').

    Resolution order:
      1. Explicit provider_override argument (if passed)
      2. Module-level AI_PROVIDER (supports unittest.mock.patch("modules.llm_service.AI_PROVIDER", ...))
      3. config.settings.AI_PROVIDER (supports config-level patch or environment switch)
      4. Default to 'ollama'
    """
    if provider_override:
        return str(provider_override).lower().strip()

    mod_val = globals().get("AI_PROVIDER")
    try:
        import config.settings as _settings
        cfg_val = getattr(_settings, "AI_PROVIDER", None)
    except Exception:
        cfg_val = None

    if mod_val is not None and str(mod_val).lower().strip() != "ollama":
        return str(mod_val).lower().strip()
    if cfg_val is not None:
        return str(cfg_val).lower().strip()
    return str(mod_val or "ollama").lower().strip()


def generate_text(prompt, system_instruction=None, provider=None) -> str:
    """
    Generate plain text from the configured AI provider.

    Parameters
    ----------
    prompt             : The user / task prompt.
    system_instruction : Optional system-role instruction (passed as a
                         system message for Ollama; GenerateContentConfig
                         system_instruction for Gemini).
    provider           : Optional provider override ('ollama' | 'gemini').

    Returns
    -------
    str : The model's plain-text response.

    Raises
    ------
    RuntimeError : On any backend error (connection, timeout, API error).
    ValueError   : On input validation failure.
    """
    active = _get_active_provider(provider)
    if active == "gemini":
        return _generate_text_gemini(prompt, system_instruction=system_instruction)
    else:
        # Default to Ollama for any unrecognised value as well as "ollama"
        return _generate_text_ollama(prompt, system_instruction=system_instruction)


def generate_json(prompt, system_instruction=None, provider=None) -> dict:
    """
    Generate a structured JSON dict from the configured AI provider.

    Parameters
    ----------
    prompt             : The user / task prompt (should describe JSON schema).
    system_instruction : Optional system-role instruction.
    provider           : Optional provider override ('ollama' | 'gemini').

    Returns
    -------
    dict : The parsed JSON response.

    Raises
    ------
    RuntimeError : On any backend error (connection, timeout, API error).
    ValueError   : On JSON parse failure or input validation failure.
    """
    active = _get_active_provider(provider)
    if active == "gemini":
        return _generate_json_gemini(prompt, system_instruction=system_instruction)
    else:
        return _generate_json_ollama(prompt, system_instruction=system_instruction)

