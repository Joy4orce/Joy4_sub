"""Google Gemini API translation wrapper.

Supports multiple API keys (one per line in the apikey input). On rate limit
or quota errors, automatically rotates to the next available key.
"""

import json
import os
import re
from datetime import datetime

import apikeyrotator
from settings import get_settings_path

ENGINE_NAME = "Gemini"
DEFAULT_MODEL = "gemini-2.0-flash"

# Errors / messages that indicate we should rotate to the next key
ROTATE_PATTERNS = [
    "quota",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "429",
    "exceeded",
    "resource exhausted",
    "resource_exhausted",
    "too many requests",
    "limit reached",
]


class GeminiAllKeysExhaustedError(Exception):
    """Raised when every provided Gemini API key has been exhausted."""


def append_runtime_log(message):
    runtime_log_path = os.path.join(os.path.dirname(get_settings_path()), "runtime.log")
    os.makedirs(os.path.dirname(runtime_log_path), exist_ok=True)
    with open(runtime_log_path, "a", encoding="utf-8") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {message}\n")


def _is_rotate_error(err_text):
    low = (err_text or "").lower()
    return any(p in low for p in ROTATE_PATTERNS)


def _parse_response(text, expected_count):
    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            return [str(x) for x in data]
    except json.JSONDecodeError:
        pass
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [str(x) for x in data]
        except json.JSONDecodeError:
            pass
    return None


def translateusinggemini(text, uiwrapper):
    try:
        target_lang = uiwrapper.get_trglanguagecodeinput().strip()
        if not target_lang:
            uiwrapper.update_percentagelabel_post("text", "target language code is empty")
            append_runtime_log("Gemini skipped: target language code empty")
            return None

        # Read keys from the engine-specific accessor (falls back to legacy getkey)
        raw_keys = uiwrapper.get_gemini_keys() if hasattr(uiwrapper, 'get_gemini_keys') else uiwrapper.getkey()
        keys = apikeyrotator.parse_keys(raw_keys)
        if not keys:
            uiwrapper.update_percentagelabel_post("text", "Gemini API key missing")
            append_runtime_log("Gemini skipped: no API keys provided")
            return None

        # Resolve the model name (allow per-call override via uiwrapper)
        model_name = DEFAULT_MODEL
        if hasattr(uiwrapper, 'get_gemini_model'):
            chosen = (uiwrapper.get_gemini_model() or "").strip()
            if chosen:
                model_name = chosen

        single_mode = isinstance(text, str)
        lines = [text] if single_mode else list(text)

        prompt = (
            "You are a professional subtitle translator. "
            f"Translate to {target_lang}. "
            f"Input has {len(lines)} lines, output must have exactly {len(lines)} lines. "
            "Return ONLY a JSON array of translated strings with the exact same length. "
            "Do not merge, split, add, or remove entries. "
            "Preserve empty strings as empty strings. "
            "Return only valid JSON with no other text.\n\n"
            + json.dumps(lines, ensure_ascii=False)
        )

        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            uiwrapper.update_percentagelabel_post("text", "google-genai not installed")
            append_runtime_log("Gemini failed: google-genai package not installed")
            return None

        # Iterate through keys, rotating on rate limit / quota errors
        while apikeyrotator.get_active_index(ENGINE_NAME) < len(keys):
            idx = apikeyrotator.get_active_index(ENGINE_NAME)
            key = keys[idx]
            try:
                client = genai.Client(api_key=key)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
                raw = (response.text or "").strip()
                parsed = _parse_response(raw, len(lines))
                if parsed is None:
                    append_runtime_log(f"Gemini parse failed (key #{idx + 1}, model={model_name}). Raw: {raw[:200]}")
                    uiwrapper.update_percentagelabel_post("text", "Gemini parse error")
                    return None

                append_runtime_log(f"Gemini translated {len(lines)} lines using key #{idx + 1} ({model_name})")
                return parsed[0] if single_mode else parsed
            except Exception as e:
                err_text = f"{type(e).__name__}: {e}"
                if _is_rotate_error(err_text):
                    append_runtime_log(f"Gemini key #{idx + 1} exhausted ({type(e).__name__}); advancing")
                    uiwrapper.update_percentagelabel_post("text", f"Gemini key #{idx + 1} exhausted, switching")
                    apikeyrotator.advance_key(ENGINE_NAME)
                    continue
                # Non-rate-limit error: bail out
                append_runtime_log(f"Gemini error (key #{idx + 1}): {err_text}")
                uiwrapper.update_percentagelabel_post("text", f"Gemini error: {type(e).__name__}")
                return None

        # All keys exhausted
        append_runtime_log(f"Gemini: all {len(keys)} keys exhausted")
        uiwrapper.update_percentagelabel_post("text", "All Gemini keys exhausted")
        should_retry = uiwrapper.wait_for_rate_limit_decision()
        if should_retry:
            apikeyrotator.reset(ENGINE_NAME)
            return translateusinggemini(text, uiwrapper)
        raise GeminiAllKeysExhaustedError("All Gemini API keys reached their quota")

    except GeminiAllKeysExhaustedError:
        raise
    except Exception as e:
        append_runtime_log(f"Gemini translation failed: {type(e).__name__}: {e}")
        uiwrapper.update_percentagelabel_post("text", f"Gemini error: {type(e).__name__}")
        return None
