"""OpenAI (ChatGPT) API translation wrapper.

Supports multiple API keys (one per line in the apikey input). On rate limit
or quota errors, automatically rotates to the next available key.
"""

import json
import os
import re
from datetime import datetime

import apikeyrotator
from settings import get_settings_path

ENGINE_NAME = "ChatGPT"
DEFAULT_MODEL = "gpt-4o-mini"

ROTATE_PATTERNS = [
    "rate limit",
    "rate_limit",
    "ratelimit",
    "429",
    "quota",
    "exceeded",
    "insufficient_quota",
    "too many requests",
    "limit reached",
    "billing",
]


class OpenAIAllKeysExhaustedError(Exception):
    """Raised when every provided OpenAI API key has been exhausted."""


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
        # OpenAI's json mode may wrap output in an object — try common keys
        if isinstance(data, dict):
            for key in ("translations", "lines", "result", "items"):
                if key in data and isinstance(data[key], list):
                    return [str(x) for x in data[key]]
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


def translateusingopenai(text, uiwrapper):
    try:
        target_lang = uiwrapper.get_trglanguagecodeinput().strip()
        if not target_lang:
            uiwrapper.update_percentagelabel_post("text", "target language code is empty")
            append_runtime_log("ChatGPT skipped: target language code empty")
            return None

        raw_keys = uiwrapper.get_openai_keys() if hasattr(uiwrapper, 'get_openai_keys') else uiwrapper.getkey()
        keys = apikeyrotator.parse_keys(raw_keys)
        if not keys:
            uiwrapper.update_percentagelabel_post("text", "ChatGPT API key missing")
            append_runtime_log("ChatGPT skipped: no API keys provided")
            return None

        model_name = DEFAULT_MODEL
        if hasattr(uiwrapper, 'get_openai_model'):
            chosen = (uiwrapper.get_openai_model() or "").strip()
            if chosen:
                model_name = chosen

        single_mode = isinstance(text, str)
        lines = [text] if single_mode else list(text)

        system_prompt = (
            "You are a professional subtitle translator. "
            f"Translate to {target_lang}. "
            f"Input has {len(lines)} lines, output must have exactly {len(lines)} lines. "
            "Return ONLY a JSON array of translated strings with the exact same length. "
            "Do not merge, split, add, or remove entries. "
            "Preserve empty strings as empty strings."
        )
        user_prompt = json.dumps(lines, ensure_ascii=False)

        try:
            from openai import OpenAI
        except ImportError:
            uiwrapper.update_percentagelabel_post("text", "openai package not installed")
            append_runtime_log("ChatGPT failed: openai package not installed")
            return None

        while apikeyrotator.get_active_index(ENGINE_NAME) < len(keys):
            idx = apikeyrotator.get_active_index(ENGINE_NAME)
            key = keys[idx]
            try:
                client = OpenAI(api_key=key)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                raw = (response.choices[0].message.content or "").strip()
                parsed = _parse_response(raw, len(lines))
                if parsed is None:
                    append_runtime_log(f"ChatGPT parse failed (key #{idx + 1}, model={model_name}). Raw: {raw[:200]}")
                    uiwrapper.update_percentagelabel_post("text", "ChatGPT parse error")
                    return None

                append_runtime_log(f"ChatGPT translated {len(lines)} lines using key #{idx + 1} ({model_name})")
                return parsed[0] if single_mode else parsed
            except Exception as e:
                err_text = f"{type(e).__name__}: {e}"
                if _is_rotate_error(err_text):
                    append_runtime_log(f"ChatGPT key #{idx + 1} exhausted ({type(e).__name__}); advancing")
                    uiwrapper.update_percentagelabel_post("text", f"ChatGPT key #{idx + 1} exhausted, switching")
                    apikeyrotator.advance_key(ENGINE_NAME)
                    continue
                append_runtime_log(f"ChatGPT error (key #{idx + 1}): {err_text}")
                uiwrapper.update_percentagelabel_post("text", f"ChatGPT error: {type(e).__name__}")
                return None

        append_runtime_log(f"ChatGPT: all {len(keys)} keys exhausted")
        uiwrapper.update_percentagelabel_post("text", "All ChatGPT keys exhausted")
        should_retry = uiwrapper.wait_for_rate_limit_decision()
        if should_retry:
            apikeyrotator.reset(ENGINE_NAME)
            return translateusingopenai(text, uiwrapper)
        raise OpenAIAllKeysExhaustedError("All ChatGPT API keys reached their quota")

    except OpenAIAllKeysExhaustedError:
        raise
    except Exception as e:
        append_runtime_log(f"ChatGPT translation failed: {type(e).__name__}: {e}")
        uiwrapper.update_percentagelabel_post("text", f"ChatGPT error: {type(e).__name__}")
        return None
