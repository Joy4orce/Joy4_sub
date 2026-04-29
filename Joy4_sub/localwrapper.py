"""Local LLM (OpenAI-compatible) translation wrapper.

Designed to work with any local server that exposes an OpenAI-compatible
chat-completions endpoint: koboldcpp, LM Studio, Ollama, llama.cpp, vLLM, etc.

Unlike the cloud engines, this wrapper:
- Talks to a single, user-configured `base_url` (no key rotation).
- Uses a user-supplied system prompt (the line-count guard is appended
  automatically so model output stays array-length stable).
- Sends an "API key" only because the OpenAI SDK requires one — most local
  servers ignore it; "sk-local" is used as a benign fallback.
"""

import json
import os
import re
from datetime import datetime

from settings import get_settings_path

ENGINE_NAME = "Local LLM"
DEFAULT_ENDPOINT = "http://localhost:5001/v1"
DEFAULT_MODEL = "local"
DEFAULT_SYSTEM_PROMPT = (
    "당신은 전문 일한 번역가입니다. 주어진 일본어를 한국어로 번역하세요."
)
DEFAULT_TEMPERATURE = 0.1
DEFAULT_API_KEY = "sk-local"

# Errors that indicate "the local server isn't running" — surfaced with a
# friendlier message so the user knows to start koboldcpp / LM Studio first.
CONNECTION_ERROR_PATTERNS = [
    "connection refused",
    "connectionerror",
    "connecterror",
    "max retries exceeded",
    "name or service not known",
    "failed to establish a new connection",
    "newconnectionerror",
    "remote end closed",
    "connection aborted",
]


def append_runtime_log(message):
    runtime_log_path = os.path.join(os.path.dirname(get_settings_path()), "runtime.log")
    os.makedirs(os.path.dirname(runtime_log_path), exist_ok=True)
    with open(runtime_log_path, "a", encoding="utf-8") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {message}\n")


def _is_connection_error(err_text):
    low = (err_text or "").lower()
    return any(p in low for p in CONNECTION_ERROR_PATTERNS)


def _parse_response(text, expected_count):
    """Identical strategy to openaiwrapper._parse_response.

    Local 12B-class models often add stray prose around the JSON; tolerate
    that by also doing a regex fallback for the first JSON array seen.
    """
    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            return [str(x) for x in data]
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


def _resolve_setting(uiwrapper, getter_name, default):
    """Read a setting from uiwrapper if the getter exists, else use the default."""
    if hasattr(uiwrapper, getter_name):
        try:
            value = getattr(uiwrapper, getter_name)()
        except Exception:
            value = None
        if value is None or (isinstance(value, str) and not value.strip()):
            return default
        return value.strip() if isinstance(value, str) else value
    return default


def translateusinglocal(text, uiwrapper):
    try:
        target_lang = uiwrapper.get_trglanguagecodeinput().strip()
        if not target_lang:
            uiwrapper.update_percentagelabel_post("text", "target language code is empty")
            append_runtime_log("Local LLM skipped: target language code empty")
            return None

        endpoint = _resolve_setting(uiwrapper, 'get_local_endpoint', DEFAULT_ENDPOINT)
        model_name = _resolve_setting(uiwrapper, 'get_local_model', DEFAULT_MODEL)
        user_system_prompt = _resolve_setting(
            uiwrapper, 'get_local_system_prompt', DEFAULT_SYSTEM_PROMPT
        )
        temperature_raw = _resolve_setting(
            uiwrapper, 'get_local_temperature', DEFAULT_TEMPERATURE
        )
        try:
            temperature = float(temperature_raw)
        except (TypeError, ValueError):
            temperature = DEFAULT_TEMPERATURE

        # API key: most local servers ignore it, but the OpenAI client requires one.
        api_key = ""
        if hasattr(uiwrapper, 'get_local_apikey'):
            try:
                api_key = (uiwrapper.get_local_apikey() or "").strip()
            except Exception:
                api_key = ""
        if not api_key:
            api_key = DEFAULT_API_KEY

        single_mode = isinstance(text, str)
        lines = [text] if single_mode else list(text)

        # Append the line-count guard to whatever the user provided.
        # The guard mirrors openaiwrapper.py so _parse_response works identically.
        suffix = (
            f" Translate to {target_lang}. "
            f"Input has {len(lines)} lines, output must have exactly {len(lines)} lines. "
            "Return ONLY a JSON array of translated strings with the exact same length. "
            "Do not merge, split, add, or remove entries. "
            "Preserve empty strings as empty strings."
        )
        system_prompt = (user_system_prompt or DEFAULT_SYSTEM_PROMPT).rstrip() + suffix
        user_prompt = json.dumps(lines, ensure_ascii=False)

        try:
            from openai import OpenAI
        except ImportError:
            uiwrapper.update_percentagelabel_post("text", "openai package not installed")
            append_runtime_log("Local LLM failed: openai package not installed")
            return None

        try:
            client = OpenAI(api_key=api_key, base_url=endpoint)
            # max_tokens=4096 prevents truncation at koboldcpp's 1024-token default,
            # which silently produces invalid JSON for batches above ~30 lines.
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=4096,
            )
            raw = (response.choices[0].message.content or "").strip()
            parsed = _parse_response(raw, len(lines))
            if parsed is None:
                append_runtime_log(
                    f"Local LLM parse failed (endpoint={endpoint}, model={model_name}). Raw: {raw[:200]}"
                )
                uiwrapper.update_percentagelabel_post("text", "Local LLM parse error")
                return None

            append_runtime_log(
                f"Local LLM translated {len(lines)} lines via {endpoint} ({model_name})"
            )
            return parsed[0] if single_mode else parsed
        except Exception as e:
            err_text = f"{type(e).__name__}: {e}"
            if _is_connection_error(err_text):
                append_runtime_log(
                    f"Local LLM connection refused: {err_text} (endpoint={endpoint})"
                )
                uiwrapper.update_percentagelabel_post(
                    "text", "Local LLM 서버 미실행 - 엔드포인트 확인"
                )
                return None
            append_runtime_log(f"Local LLM error: {err_text}")
            uiwrapper.update_percentagelabel_post("text", f"Local LLM error: {type(e).__name__}")
            return None

    except Exception as e:
        append_runtime_log(f"Local LLM translation failed: {type(e).__name__}: {e}")
        uiwrapper.update_percentagelabel_post("text", f"Local LLM error: {type(e).__name__}")
        return None
