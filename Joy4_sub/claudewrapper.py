import json
import os
import re
import subprocess
import sys
from datetime import datetime

from settings import get_settings_path

RATE_LIMIT_PATTERNS = [
    'usage limit',
    'rate limit',
    'too many requests',
    'limit reached',
    'quota exceeded',
    'usage cap',
    'daily limit',
    'claude.ai/upgrade',
    'pro plan',
    'try again later',
    'limit has been reached',
    # Actual messages emitted by recent claude.exe builds:
    "you've hit your limit",
    'hit your limit',
    'hit the rate limit',
    'limit · resets',
    'limit. resets',
    'limit, resets',
    "you've reached your",
    'reached your usage',
    'reached your limit',
    'usage has been exceeded',
]

# Phrases Claude emits when it refuses to translate due to content policy.
# We treat these specially so the user gets a clear message and the batch
# is preserved (kept as source) instead of silently failing.
SAFETY_REFUSAL_PATTERNS = [
    "i can't translate",
    "i cannot translate",
    "i'm not able to translate",
    "i am not able to translate",
    "i won't translate",
    "i will not translate",
    "i can't assist",
    "i cannot assist",
    "i won't assist",
    "i'm unable to translate",
    "against my",
    "content policy",
    "won't translate",
]

PLAN_LABELS = {
    'pro': 'Pro',
    'team': 'Team',
}


class ClaudeRateLimitError(Exception):
    pass


class ClaudeSafetyRefusalError(Exception):
    """Raised when Claude refuses to translate a batch due to content policy."""

    def __init__(self, message, refusal_text=""):
        super().__init__(message)
        self.refusal_text = refusal_text


def append_runtime_log(message):
    runtime_log_path = os.path.join(os.path.dirname(get_settings_path()), "runtime.log")
    os.makedirs(os.path.dirname(runtime_log_path), exist_ok=True)
    with open(runtime_log_path, "a", encoding="utf-8") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {message}\n")


def _find_claude_exe():
    """Return full path to claude executable, searching PATH and common install locations."""
    import shutil
    found = shutil.which("claude")
    if found:
        return found
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".local", "bin", "claude.exe"),
        os.path.join(home, ".local", "bin", "claude"),
        os.path.join(os.environ.get("APPDATA", ""), "npm", "claude.cmd"),
        os.path.join(os.environ.get("APPDATA", ""), "npm", "claude"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "claude.exe"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return "claude"


def _is_rate_limit(stdout, stderr):
    combined = (stdout + stderr).lower()
    return any(p in combined for p in RATE_LIMIT_PATTERNS)


def _is_safety_refusal(text):
    """Return True if the Claude response looks like a content-policy refusal.

    We require both a refusal phrase AND the absence of a JSON array, so that
    legitimate translations that happen to include the word "translate" don't
    get misclassified.
    """
    if not text:
        return False
    lowered = text.lower()
    if not any(p in lowered for p in SAFETY_REFUSAL_PATTERNS):
        return False
    # If the response also contains a JSON array, treat it as a normal response.
    if re.search(r'\[.*\]', text, re.DOTALL):
        return False
    return True


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


def _get_fallback_oauth_token():
    """Return a Claude OAuth token from ~/.joy4sub_claude_token (or legacy ~/.vatsg_claude_token).

    Single-token fallback retained for backward compatibility.
    """
    home = os.path.expanduser("~")
    for filename in (".joy4sub_claude_token", ".vatsg_claude_token"):
        token_file = os.path.join(home, filename)
        try:
            if os.path.isfile(token_file):
                with open(token_file, "r", encoding="utf-8") as f:
                    token = f.read().strip()
                if token:
                    return token
        except Exception:
            continue
    return None


def _get_token_for_plan(uiwrapper, plan):
    if plan == 'pro':
        getter = getattr(uiwrapper, 'get_claude_pro_token', None)
    elif plan == 'team':
        getter = getattr(uiwrapper, 'get_claude_team_token', None)
    else:
        return ""
    if getter is None:
        return ""
    try:
        return (getter() or "").strip()
    except Exception:
        return ""


def _select_initial_plan(uiwrapper):
    """Return the plan to start with, preferring the user's default if its token is set.
    Returns None if neither plan has a token (caller falls back to claude.exe defaults)."""
    default = 'pro'
    getter = getattr(uiwrapper, 'get_claude_default_plan', None)
    if getter is not None:
        try:
            default = (getter() or 'pro').strip().lower() or 'pro'
        except Exception:
            default = 'pro'
    if default not in ('pro', 'team'):
        default = 'pro'
    pro_token = _get_token_for_plan(uiwrapper, 'pro')
    team_token = _get_token_for_plan(uiwrapper, 'team')
    if default == 'pro' and pro_token:
        return 'pro'
    if default == 'team' and team_token:
        return 'team'
    if pro_token:
        return 'pro'
    if team_token:
        return 'team'
    return None


def _other_plan(plan):
    return 'team' if plan == 'pro' else 'pro'


def _build_env(uiwrapper, plan):
    """Construct the environment for claude.exe, injecting the OAuth token for the chosen plan."""
    env = os.environ.copy()
    if "USERPROFILE" not in env:
        env["USERPROFILE"] = os.path.expanduser("~")
    if "HOME" not in env:
        env["HOME"] = os.path.expanduser("~")

    # When Joy4_sub runs under a Claude Code session, PROVIDER_MANAGED_BY_HOST=1
    # causes claude.exe to skip ~/.claude.json auth. Remove it so claude
    # uses its own stored credentials.
    env.pop("CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST", None)
    # An empty ANTHROPIC_API_KEY string takes precedence over OAuth; remove it.
    if not env.get("ANTHROPIC_API_KEY"):
        env.pop("ANTHROPIC_API_KEY", None)

    plan_token = _get_token_for_plan(uiwrapper, plan) if plan else ""
    if plan_token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = plan_token
    elif not env.get("CLAUDE_CODE_OAUTH_TOKEN"):
        # No per-plan token configured and nothing in env — fall back to legacy file.
        token = _get_fallback_oauth_token()
        if token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    return env


def _run_claude_once(prompt, env):
    """Execute claude.exe a single time with the given env, returning the CompletedProcess."""
    claude_exe = _find_claude_exe()
    use_shell = claude_exe.lower().endswith(('.cmd', '.bat'))
    kwargs = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 600,
        "shell": use_shell,
        "env": env,
    }
    if sys.platform == "win32" and not use_shell:
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.run([claude_exe, "-p", prompt], **kwargs)


def translateusingclaude(text, uiwrapper, _plan=None, _exhausted=None):
    """Translate text via the Claude CLI, automatically failing over between Pro and Team plans.

    On the first call, picks the user's preferred plan (whose token is configured) and runs.
    If that plan is rate-limited and the other plan has a token, retries with the other plan.
    Only after both configured plans are exhausted does it surface the wait-or-cancel dialog.
    """
    try:
        # Cancel checkpoint on entry. This guards each Claude batch call
        # AND each plan-failover recursion (the function calls itself with
        # the other plan on rate limit). Mid-claude.exe-run cancellation
        # isn't supported — that's a 30+ second subprocess we can't easily
        # signal — but bailing here keeps the user out of additional batches.
        if hasattr(uiwrapper, 'is_cancelled') and uiwrapper.is_cancelled():
            append_runtime_log("Claude CLI cancelled by user before dispatch")
            return None

        target_lang = uiwrapper.get_trglanguagecodeinput().strip()

        if not target_lang:
            uiwrapper.update_percentagelabel_post("text", "target language code is empty")
            append_runtime_log("Claude CLI skipped: target language code empty")
            return None

        if _exhausted is None:
            _exhausted = set()
        if _plan is None:
            _plan = _select_initial_plan(uiwrapper)
            # _plan may be None here — that's OK, we'll let claude.exe use its stored login.

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

        env = _build_env(uiwrapper, _plan)
        plan_label = PLAN_LABELS.get(_plan, "default") if _plan else "default"

        if _plan:
            uiwrapper.update_percentagelabel_post("text", f"Claude {plan_label} 플랜으로 번역 중...")
            append_runtime_log(f"Claude CLI invoking with {plan_label} plan")

        result = _run_claude_once(prompt, env)

        # Rate-limit messages can appear with either returncode 0 or 1 and
        # are sometimes echoed only to stdout (e.g. "You've hit your limit · resets ...").
        rate_limited = _is_rate_limit(result.stdout, result.stderr)

        if rate_limited:
            append_runtime_log(
                f"Claude rate limit detected on {plan_label} plan. "
                f"stderr={result.stderr.strip()[:200]} stdout={result.stdout.strip()[:200]}"
            )
            if _plan:
                _exhausted.add(_plan)

            # Try the other plan if it has a token and hasn't been tried yet.
            other = _other_plan(_plan) if _plan else None
            other_token = _get_token_for_plan(uiwrapper, other) if other else ""
            if other and other not in _exhausted and other_token:
                other_label = PLAN_LABELS.get(other, other)
                append_runtime_log(f"Claude rate limit on {plan_label}, falling back to {other_label}")
                uiwrapper.update_percentagelabel_post(
                    "text", f"Claude {plan_label} 한도 도달 → {other_label}로 전환"
                )
                return translateusingclaude(text, uiwrapper, _plan=other, _exhausted=_exhausted)

            # Both configured plans exhausted (or only one configured) — wait dialog.
            uiwrapper.update_percentagelabel_post("text", "Claude 모든 플랜 한도 도달 - 대기 중...")
            should_retry = uiwrapper.wait_for_rate_limit_decision()
            if should_retry:
                append_runtime_log("Retrying Claude translation after rate limit reset (resetting plan rotation)")
                return translateusingclaude(text, uiwrapper, _plan=None, _exhausted=None)
            raise ClaudeRateLimitError("Claude usage limit reached on all configured plans")

        raw = result.stdout.strip()

        # Detect content-policy refusals BEFORE checking returncode, since refusal
        # responses sometimes come back with returncode 0 and no JSON array.
        if _is_safety_refusal(raw):
            snippet = raw[:200].replace("\n", " ")
            append_runtime_log(f"Claude content-policy refusal on {plan_label}: {snippet}")
            uiwrapper.update_percentagelabel_post(
                "text", "Claude 콘텐츠 정책 거부 - 원문 유지"
            )
            raise ClaudeSafetyRefusalError(
                "Claude refused this batch due to content policy", refusal_text=raw
            )

        if result.returncode != 0:
            err = result.stderr.strip()[:300]
            out = result.stdout.strip()[:300]
            uiwrapper.update_percentagelabel_post("text", "Claude CLI error")
            append_runtime_log(f"Claude CLI exited {result.returncode}: stderr={err!r} stdout={out!r}")
            return None

        parsed = _parse_response(raw, len(lines))

        if parsed is None:
            # Re-check refusal patterns on the raw output as a last resort.
            if _is_safety_refusal(raw):
                snippet = raw[:200].replace("\n", " ")
                append_runtime_log(f"Claude content-policy refusal (post-parse) on {plan_label}: {snippet}")
                uiwrapper.update_percentagelabel_post(
                    "text", "Claude 콘텐츠 정책 거부 - 원문 유지"
                )
                raise ClaudeSafetyRefusalError(
                    "Claude refused this batch due to content policy", refusal_text=raw
                )
            append_runtime_log(f"Claude CLI response parse failed. Raw: {raw[:200]}")
            uiwrapper.update_percentagelabel_post("text", "Claude CLI parse error")
            return None

        append_runtime_log(f"Claude CLI translated {len(lines)} lines on {plan_label} plan")
        return parsed[0] if single_mode else parsed

    except ClaudeRateLimitError:
        raise
    except ClaudeSafetyRefusalError:
        raise
    except subprocess.TimeoutExpired:
        uiwrapper.update_percentagelabel_post("text", "Claude CLI timeout")
        append_runtime_log("Claude CLI TimeoutExpired")
        return None
    except FileNotFoundError:
        uiwrapper.update_percentagelabel_post("text", "Claude CLI not found in PATH")
        append_runtime_log("Claude CLI not found (FileNotFoundError)")
        return None
    except Exception as e:
        uiwrapper.update_percentagelabel_post("text", f"Claude CLI error: {type(e).__name__}")
        append_runtime_log(f"Claude CLI error: {type(e).__name__}: {e}")
        return None
