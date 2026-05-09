import argparse
import json
import os
import sys

from claudewrapper import ClaudeRateLimitError
from geminiwrapper import GeminiAllKeysExhaustedError
from openaiwrapper import OpenAIAllKeysExhaustedError
from mywhisper import (
    append_runtime_log,
    transcribe_from_mp3_fast_whisper,
    transcribe_from_mp3_whisper,
)


# Sentinel prefix for progress messages on stdout. The parent process matches
# on this prefix to forward worker-side progress events (text labels, bar
# values, batch counters) into multifile_queue in real time. Anything that
# doesn't start with the prefix is treated as ordinary stdout and just
# logged. ASCII-only so encoding never trips it up.
PROGRESS_SENTINEL = "__J4S_PROG__"


# Make sure stdout is UTF-8 and line-buffered so the parent can read complete
# JSON messages as soon as the worker emits them. Without this, Korean
# characters in progress text can get mangled on Windows code-page consoles
# (cp949), and buffered writes don't reach the parent until exit.
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass


def _emit_progress(method, label, value):
    """Print one JSON-encoded progress event to stdout for the parent to
    consume. `method` is which UI surface to update ('text' for the status
    label, 'bar' for the progress bar). `label` is the existing label
    discriminator used by the in-process queue ('text' / 'value' /
    'maximum' / etc.) so the parent's existing message handlers can dispatch
    on it without knowing about subprocess plumbing.
    """
    payload = {"m": method, "l": label, "v": value}
    try:
        line = PROGRESS_SENTINEL + json.dumps(payload, ensure_ascii=False)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    except Exception:
        # Never let a stdout failure break translation. Log and continue.
        try:
            append_runtime_log(f"Worker progress emit failed for {payload}")
        except Exception:
            pass


class WorkerUIWrapper:
    def __init__(self, api_key, cuda_enabled, model_name, source_lang, target_lang,
                 keep_original, fast_mode, translation_engine="DeepL",
                 deepl_key="", gemini_keys="", openai_keys="",
                 gemini_model="", openai_model="",
                 claude_pro_token="", claude_team_token="", claude_default_plan="pro",
                 local_endpoint="", local_model="", local_system_prompt="",
                 local_temperature=0.1, local_apikey=""):
        self.api_key = api_key  # legacy
        self.cuda_enabled = cuda_enabled
        self.model_name = model_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.keep_original = keep_original
        self.fast_mode = fast_mode
        self.translation_engine = translation_engine
        self.deepl_key = deepl_key
        self.gemini_keys = gemini_keys
        self.openai_keys = openai_keys
        self.gemini_model = gemini_model
        self.openai_model = openai_model
        self.claude_pro_token = claude_pro_token
        self.claude_team_token = claude_team_token
        self.claude_default_plan = (claude_default_plan or "pro").lower()
        self.local_endpoint = local_endpoint
        self.local_model = local_model
        self.local_system_prompt = local_system_prompt
        self.local_temperature = local_temperature
        self.local_apikey = local_apikey

    def update_percentagelabel_post(self, text, value):
        # Forward to parent UI through the stdout sentinel channel (when
        # available) AND keep logging to runtime.log for offline debugging.
        _emit_progress("text", text, value)
        append_runtime_log(f"Worker progress text={text} value={value}")

    def update_progressbar(self, text, value):
        _emit_progress("bar", text, value)
        append_runtime_log(f"Worker progress update {text}={value}")

    def getkey(self):
        engine = self.translation_engine
        if engine == "DeepL":
            return self.deepl_key
        if engine == "Gemini":
            return self.gemini_keys
        if engine == "ChatGPT":
            return self.openai_keys
        if engine == "Local LLM":
            return self.local_apikey
        return self.api_key

    def get_cuda_var(self):
        return self.cuda_enabled

    def get_translateoption_Var(self):
        return self.model_name

    def get_srclanguagecodeinput(self):
        return self.source_lang

    def get_trglanguagecodeinput(self):
        return self.target_lang

    def get_original_var(self):
        return self.keep_original

    def get_fast_var(self):
        return self.fast_mode

    def get_translation_engine(self):
        return self.translation_engine

    def get_deepl_key(self):
        return self.deepl_key

    def get_gemini_keys(self):
        return self.gemini_keys

    def get_openai_keys(self):
        return self.openai_keys

    def get_gemini_model(self):
        return self.gemini_model

    def get_openai_model(self):
        return self.openai_model

    def get_claude_pro_token(self):
        return self.claude_pro_token

    def get_claude_team_token(self):
        return self.claude_team_token

    def get_claude_default_plan(self):
        return self.claude_default_plan

    def get_local_endpoint(self):
        return self.local_endpoint

    def get_local_model(self):
        return self.local_model

    def get_local_system_prompt(self):
        return self.local_system_prompt

    def get_local_temperature(self):
        return self.local_temperature

    def get_local_apikey(self):
        return self.local_apikey

    def wait_for_rate_limit_decision(self, file_info=""):
        append_runtime_log(f"Worker subprocess: rate limit reached, signaling parent via exit code 2")
        return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--temp-audio", required=True)
    parser.add_argument("--api-key", default="")  # legacy compat
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-lang", default="")
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--keep-original", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--engine", default="DeepL")
    parser.add_argument("--deepl-key", default="")
    parser.add_argument("--gemini-keys", default="")
    parser.add_argument("--openai-keys", default="")
    parser.add_argument("--gemini-model", default="")
    parser.add_argument("--openai-model", default="")
    parser.add_argument("--claude-pro-token", default="")
    parser.add_argument("--claude-team-token", default="")
    parser.add_argument("--claude-default-plan", default="pro")
    parser.add_argument("--local-endpoint", default="")
    parser.add_argument("--local-model", default="")
    parser.add_argument("--local-system-prompt", default="")
    parser.add_argument("--local-temperature", default="0.1")
    parser.add_argument("--local-apikey", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    append_runtime_log(f"Worker subprocess started for {args.file}")
    try:
        local_temp = float(args.local_temperature)
    except (TypeError, ValueError):
        local_temp = 0.1

    uiwrapper = WorkerUIWrapper(
        api_key=args.api_key,
        cuda_enabled=args.cuda,
        model_name=args.model,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        keep_original=args.keep_original,
        fast_mode=args.fast,
        translation_engine=args.engine,
        deepl_key=args.deepl_key,
        gemini_keys=args.gemini_keys,
        openai_keys=args.openai_keys,
        gemini_model=args.gemini_model,
        openai_model=args.openai_model,
        claude_pro_token=args.claude_pro_token,
        claude_team_token=args.claude_team_token,
        claude_default_plan=args.claude_default_plan,
        local_endpoint=args.local_endpoint,
        local_model=args.local_model,
        local_system_prompt=args.local_system_prompt,
        local_temperature=local_temp,
        local_apikey=args.local_apikey,
    )

    try:
        if args.fast:
            transcribe_from_mp3_fast_whisper(args.file, args.temp_audio, uiwrapper)
        else:
            transcribe_from_mp3_whisper(args.file, args.temp_audio, uiwrapper)
    except ClaudeRateLimitError:
        append_runtime_log(f"Worker subprocess: Claude rate limit reached for {args.file}")
        sys.exit(2)
    except GeminiAllKeysExhaustedError:
        append_runtime_log(f"Worker subprocess: All Gemini keys exhausted for {args.file}")
        sys.exit(2)
    except OpenAIAllKeysExhaustedError:
        append_runtime_log(f"Worker subprocess: All ChatGPT keys exhausted for {args.file}")
        sys.exit(2)

    append_runtime_log(f"Worker subprocess finished for {args.file}")


if __name__ == "__main__":
    main()
