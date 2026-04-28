"""Shared API key rotation helper for translation engines.

Engines that accept multiple API keys (one per line in the apikey input) use
this module to track which key is currently active and to rotate to the next
key when a rate limit error is encountered.

State is module-level per engine. Because each video transcription runs in a
fresh worker subprocess, state naturally resets per video.
"""

# {engine_name: index} — index of the currently active key for that engine
_active_index = {}


def parse_keys(raw):
    """Split a multiline key string into a list of trimmed non-empty keys."""
    if not raw:
        return []
    return [line.strip() for line in raw.splitlines() if line.strip()]


def get_active_index(engine):
    return _active_index.get(engine, 0)


def advance_key(engine):
    """Move to the next key for this engine. Returns the new index."""
    _active_index[engine] = _active_index.get(engine, 0) + 1
    return _active_index[engine]


def reset(engine):
    _active_index[engine] = 0
