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

# Maximum number of translate+verify retry attempts before giving up on a
# batch and falling back to keeping the source lines as-is. Each retry
# increases temperature to break out of deterministic attractor states
# (e.g. the Gemma echo bug). 3 is a balance between catching bad output
# and bounding the per-file slowdown — at most ~6x slower than no-verify.
MAX_RETRY_ATTEMPTS = 3
TEMPERATURE_BUMP_PER_ATTEMPT = 0.2
TEMPERATURE_CEILING = 0.9

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


def _strip_markdown_fences(text):
    """Strip ``` or ```json fences whether they're balanced or only-opening.

    Multilingual models (notably Gemma variants) like to wrap their JSON
    output in a markdown code block — sometimes with the closing fence,
    sometimes without (when output is truncated by max_tokens).
    """
    cleaned = text.strip()
    # Opening fence: ``` or ```json (with optional newline)
    cleaned = re.sub(r'^\s*```(?:json|JSON)?\s*\n?', '', cleaned)
    # Closing fence: ``` (anchored at end)
    cleaned = re.sub(r'\n?\s*```\s*$', '', cleaned)
    return cleaned.strip()


def _parse_response(text, expected_count):
    """Tolerant parser for local model output.

    Handles four common shapes:
      1. Raw JSON array: ["a", "b", ...]
      2. JSON object with translations/lines/result/items field
      3. Markdown-wrapped JSON: ```json\n[...]\n```
      4. Prose + JSON: "Here is the translation: [...]"
    """
    cleaned = _strip_markdown_fences(text)

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict):
            for key in ("translations", "lines", "result", "items"):
                if key in data and isinstance(data[key], list):
                    return [str(x) for x in data[key]]
    except json.JSONDecodeError:
        pass

    # Last-ditch: find the first balanced JSON array anywhere in the text.
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [str(x) for x in data]
        except json.JSONDecodeError:
            pass
    return None


def _attempt_translation(client, model_name, merged_prompt, temperature, expected_count):
    """One translation API call. Returns (parsed_list, raw_text, error_text).

    - parsed_list: list[str] on success, None on parse failure or API error
    - raw_text: model's raw response (or empty on API error)
    - error_text: short error description if API call itself failed, else ""
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": merged_prompt}],
            temperature=temperature,
            max_tokens=8192,
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = _parse_response(raw, expected_count)
        return parsed, raw, ""
    except Exception as e:
        return None, "", f"{type(e).__name__}: {e}"


def _verify_translation(client, model_name, source_lines, translated_lines, target_lang):
    """Ask the local model to judge whether `translated_lines` is a correct
    translation of `source_lines` into `target_lang`.

    Checks objective failure modes only — line count, target language presence,
    untranslated source pollution. Quality / nuance judgments are deliberately
    not asked because the same model translating and verifying biases toward
    accepting its own (possibly wrong) output on subjective criteria.

    Returns True on PASS, False on FAIL or unparseable response, or None if the
    verification call itself raised an exception (e.g. connection drop). The
    caller should treat None as "verification unavailable" — usually retry.
    """
    if len(source_lines) != len(translated_lines):
        # Trivially fails one of our checks; no need to ask the model.
        return False

    verification_prompt = (
        "You are a strict translation reviewer. Reply with exactly one word.\n\n"
        f"Source ({len(source_lines)} lines):\n"
        f"{json.dumps(source_lines, ensure_ascii=False)}\n\n"
        f"Translation (claimed {target_lang}, {len(translated_lines)} lines):\n"
        f"{json.dumps(translated_lines, ensure_ascii=False)}\n\n"
        "Verify ALL of these:\n"
        f"1. The translation has exactly {len(source_lines)} entries.\n"
        f"2. Every non-empty entry is written in {target_lang} (not the source language).\n"
        "3. No translated entry is identical to its corresponding source entry "
        "(unless the source was already in the target language, e.g. proper nouns).\n"
        "4. The general meaning is preserved.\n\n"
        "Reply with exactly one word: PASS or FAIL. No explanation, no punctuation."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": verification_prompt}],
            temperature=0.0,  # deterministic for binary judgment
            max_tokens=10,
        )
        verdict = (response.choices[0].message.content or "").strip().upper()
    except Exception:
        return None  # verification call broken — caller decides what to do

    # Tolerate trailing punctuation, whitespace, markdown, etc.
    verdict_token = re.match(r'\s*\W*([A-Z]+)', verdict)
    word = verdict_token.group(1) if verdict_token else verdict
    if word.startswith("PASS"):
        return True
    return False


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
        # Stricter than openaiwrapper because local multilingual models (Gemma etc.)
        # often wrap output in markdown or add prose unless explicitly forbidden.
        suffix = (
            f" Translate to {target_lang}. "
            f"Input has {len(lines)} lines, output must have exactly {len(lines)} lines. "
            "Return ONLY a JSON array of translated strings with the exact same length. "
            "Do not merge, split, add, or remove entries. "
            "Preserve empty strings as empty strings. "
            "Output ONLY the raw JSON array. "
            "Do NOT use markdown formatting. "
            "Do NOT wrap the output in ``` code fences. "
            "Do NOT add explanations, prose, or any text before or after the array."
        )
        system_prompt = (user_system_prompt or DEFAULT_SYSTEM_PROMPT).rstrip() + suffix
        user_prompt = json.dumps(lines, ensure_ascii=False)

        try:
            from openai import OpenAI
        except ImportError:
            uiwrapper.update_percentagelabel_post("text", "openai package not installed")
            append_runtime_log("Local LLM failed: openai package not installed")
            return None

        # Merge system + user into a single user turn.
        #
        # Gemma 2/3/3n/4 chat templates do NOT define a system role; the
        # OpenAI-compat layer in koboldcpp (and various local servers) may
        # silently drop the system message when rendering through such a
        # template, leaving the model with just the input — which then
        # echoes the source verbatim (Gemma-class models are very prone
        # to this on a 0.1-temperature decode).
        #
        # Concatenating into one user turn is the canonical workaround
        # and is safe across providers: GPT/Claude/Llama-Instruct etc.
        # still receive the same instructions, just without the system
        # role wrapper. Cloud engines (OpenAI/Claude/Gemini) are not
        # affected — they have their own wrappers that keep system role.
        merged_user_prompt = system_prompt + "\n\n" + user_prompt

        try:
            client = OpenAI(api_key=api_key, base_url=endpoint)
        except Exception as e:
            append_runtime_log(f"Local LLM client init failed: {type(e).__name__}: {e}")
            uiwrapper.update_percentagelabel_post("text", f"Local LLM error: {type(e).__name__}")
            return None

        # Single-line / one-shot path: skip verification (degenerate batch
        # of size 1; verification overhead isn't worth it).
        if single_mode:
            parsed, raw, err = _attempt_translation(
                client, model_name, merged_user_prompt, temperature, len(lines)
            )
            if err:
                if _is_connection_error(err):
                    append_runtime_log(
                        f"Local LLM connection refused: {err} (endpoint={endpoint})"
                    )
                    uiwrapper.update_percentagelabel_post(
                        "text", "Local LLM 서버 미실행 - 엔드포인트 확인"
                    )
                else:
                    append_runtime_log(f"Local LLM error: {err}")
                    uiwrapper.update_percentagelabel_post(
                        "text", f"Local LLM error: {err.split(':')[0]}"
                    )
                return None
            if parsed is None:
                append_runtime_log(
                    f"Local LLM parse failed (single mode). Raw: {raw[:200]}"
                )
                uiwrapper.update_percentagelabel_post("text", "Local LLM parse error")
                return None
            append_runtime_log(
                f"Local LLM translated 1 line via {endpoint} ({model_name})"
            )
            return parsed[0]

        # Batch path: translate -> verify -> retry on verification failure.
        # Each retry bumps temperature to escape deterministic bad outputs
        # (notably Gemma's tendency to echo source language at low temps).
        last_attempt_lines = None
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            attempt_temp = min(
                temperature + (attempt - 1) * TEMPERATURE_BUMP_PER_ATTEMPT,
                TEMPERATURE_CEILING,
            )
            uiwrapper.update_percentagelabel_post(
                "text",
                f"Local LLM 번역 중 ({attempt}/{MAX_RETRY_ATTEMPTS}, temp={attempt_temp:.2f})",
            )
            parsed, raw, err = _attempt_translation(
                client, model_name, merged_user_prompt, attempt_temp, len(lines)
            )

            if err:
                # API-level error: connection drops are special-cased so we
                # don't waste retries when the server is down.
                if _is_connection_error(err):
                    append_runtime_log(
                        f"Local LLM connection refused on attempt {attempt}: {err}"
                    )
                    uiwrapper.update_percentagelabel_post(
                        "text", "Local LLM 서버 미실행 - 엔드포인트 확인"
                    )
                    return None
                append_runtime_log(
                    f"Local LLM API error on attempt {attempt}: {err}"
                )
                continue  # transient — retry

            if parsed is None:
                append_runtime_log(
                    f"Local LLM parse failed on attempt {attempt} (temp={attempt_temp:.2f}). "
                    f"Raw: {raw[:200]}"
                )
                continue  # bad shape — retry with higher temp

            # Translation got a parseable JSON array. Now verify.
            uiwrapper.update_percentagelabel_post(
                "text",
                f"Local LLM 검수 중 ({attempt}/{MAX_RETRY_ATTEMPTS})",
            )
            verdict = _verify_translation(
                client, model_name, lines, parsed, target_lang
            )
            if verdict is True:
                append_runtime_log(
                    f"Local LLM PASS on attempt {attempt}/{MAX_RETRY_ATTEMPTS} "
                    f"({len(lines)} lines, temp={attempt_temp:.2f})"
                )
                return parsed
            if verdict is None:
                append_runtime_log(
                    f"Local LLM verification call failed on attempt {attempt}; treating as FAIL"
                )
            else:
                append_runtime_log(
                    f"Local LLM FAIL verification on attempt {attempt}/{MAX_RETRY_ATTEMPTS} "
                    f"(temp={attempt_temp:.2f}); will retry with higher temperature"
                )
            last_attempt_lines = parsed  # remember in case all retries exhaust

        # All attempts failed verification. Returning None so mywhisper's
        # _translate_one_batch falls back to keeping the source lines for
        # this batch — the rest of the file still completes.
        append_runtime_log(
            f"Local LLM exhausted {MAX_RETRY_ATTEMPTS} attempts without passing verification "
            f"({len(lines)} lines); batch will fall back to original text"
        )
        uiwrapper.update_percentagelabel_post(
            "text", f"Local LLM 검수 {MAX_RETRY_ATTEMPTS}회 실패 - 원문 유지"
        )
        # We deliberately return None (not last_attempt_lines): if we couldn't
        # verify the output, we'd rather show the source text than a bad
        # translation that looked right structurally.
        return None

    except Exception as e:
        append_runtime_log(f"Local LLM translation failed: {type(e).__name__}: {e}")
        uiwrapper.update_percentagelabel_post("text", f"Local LLM error: {type(e).__name__}")
        return None
