"""Qwen3-ASR (Alibaba Cloud) speech-to-text engine — optional alternative to
Whisper for the transcription stage.

Why: on Japanese ASMR, Whisper (large-v3) is prone to runaway repetition
hallucinations on non-verbal vocalisations (moans, gasps), whereas Qwen3-ASR
stays coherent and is slightly more accurate on ordinary speech. See the A/B
notes in the project history.

How: Silero VAD (reused from faster-whisper) splits the 16 kHz mono audio into
speech regions; adjacent regions are merged into <=~15 s windows. Windows are
transcribed in batches by Qwen3-ASR-1.7B, and each window's VAD bounds become
the subtitle timestamps. This deliberately avoids the heavier
Qwen3-ForcedAligner (5-minute cap, extra model) while still giving natural,
tightly-timed segments.

Cost: a heavier dependency stack than faster-whisper (transformers +
accelerate + librosa) and slightly slower decoding. The model is loaded once
and cached for the life of the process (like the retained Whisper models).
"""

import os
from datetime import datetime

from settings import get_settings_path

MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
_TARGET_SR = 16000

# VAD tuned for whispered ASMR: a low threshold keeps soft speech; a cap on
# speech duration prevents a single window from growing unbounded.
_VAD_OPTIONS = dict(
    threshold=0.2,
    min_silence_duration_ms=400,
    speech_pad_ms=200,
    max_speech_duration_s=20,
)
_MERGE_MAX_S = 15.0   # never merge a window longer than this (subtitle length)
_MERGE_GAP_S = 0.6    # bridge silences shorter than this into one window
_BATCH = 8            # Qwen windows transcribed per generate() call
_MAX_NEW_TOKENS = 448

_model = None
_processor = None


def append_runtime_log(message):
    runtime_log_path = os.path.join(os.path.dirname(get_settings_path()), "runtime.log")
    os.makedirs(os.path.dirname(runtime_log_path), exist_ok=True)
    with open(runtime_log_path, "a", encoding="utf-8") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {message}\n")


def _lazy_load():
    """Load (and cache) the Qwen3-ASR model + processor on first use."""
    global _model, _processor
    if _model is None:
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        append_runtime_log(f"Qwen3-ASR loading {MODEL_ID} ...")
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID, device_map="auto", dtype=torch.float16
        )
        append_runtime_log("Qwen3-ASR model ready")
    return _model, _processor


def _merge_windows(speech_ts, sr):
    """Merge Silero VAD speech regions into <=_MERGE_MAX_S windows, bridging
    silences shorter than _MERGE_GAP_S. Returns [(start_sample, end_sample)]."""
    max_len = int(_MERGE_MAX_S * sr)
    gap = int(_MERGE_GAP_S * sr)
    windows = []
    cur = None
    for seg in speech_ts:
        s, e = int(seg["start"]), int(seg["end"])
        if cur is None:
            cur = [s, e]
        elif (s - cur[1]) <= gap and (e - cur[0]) <= max_len:
            cur[1] = e
        else:
            windows.append((cur[0], cur[1]))
            cur = [s, e]
    if cur is not None:
        windows.append((cur[0], cur[1]))
    return windows


def transcribe_qwen(wav_path, uiwrapper):
    """Transcribe a 16 kHz mono WAV with Qwen3-ASR.

    Returns a list of (start_seconds, end_seconds, text) segments, ordered by
    time, ready to be written as an SRT. Empty list if no speech is found.
    """
    import librosa
    from faster_whisper.vad import get_speech_timestamps, VadOptions

    model, processor = _lazy_load()

    audio, sr = librosa.load(wav_path, sr=_TARGET_SR, mono=True)  # float32 in [-1, 1]
    speech_ts = get_speech_timestamps(audio, VadOptions(**_VAD_OPTIONS), sampling_rate=sr)
    windows = _merge_windows(speech_ts, sr)
    if not windows:
        append_runtime_log(f"Qwen3-ASR: VAD found no speech in {wav_path}")
        return []

    # Source language: honour the UI code (e.g. "ja"); None lets Qwen auto-detect.
    src_lang = None
    if hasattr(uiwrapper, "get_srclanguagecodeinput"):
        src_lang = (uiwrapper.get_srclanguagecodeinput() or "").strip() or None

    total = len(windows)
    total_seconds = len(audio) / sr
    uiwrapper.update_progressbar("maximum", total_seconds or 1)

    segments = []
    done = 0
    for i in range(0, total, _BATCH):
        if hasattr(uiwrapper, "is_cancelled") and uiwrapper.is_cancelled():
            append_runtime_log(f"Qwen3-ASR cancelled by user at window {done}/{total}")
            break
        batch = windows[i:i + _BATCH]
        arrays = [audio[s:e] for s, e in batch]
        inputs = processor.apply_transcription_request(
            audio=arrays, language=src_lang
        ).to(model.device, model.dtype)
        output_ids = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        texts = processor.decode(generated, return_format="transcription_only")

        for (s, e), text in zip(batch, texts):
            text = (text or "").strip()
            if text:
                segments.append((s / sr, e / sr, text))

        done += len(batch)
        uiwrapper.update_percentagelabel_post("text", f"Qwen3-ASR 전사 중 ({done}/{total})")
        uiwrapper.update_progressbar("value", batch[-1][1] / sr)

    append_runtime_log(
        f"Qwen3-ASR transcribed {len(segments)} segments from {total} windows "
        f"({total_seconds:.0f}s audio)"
    )
    return segments
