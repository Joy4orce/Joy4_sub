import glob
import os
import gc
from datetime import datetime

import stable_whisper
import torch
import whisper
from faster_whisper import WhisperModel
from whisper.utils import get_writer

from claudewrapper import ClaudeRateLimitError, ClaudeSafetyRefusalError, translateusingclaude
from deeplwrapper import translateusingapi, translateusingapifortest
from geminiwrapper import GeminiAllKeysExhaustedError, translateusinggemini
from openaiwrapper import OpenAIAllKeysExhaustedError, translateusingopenai
from localwrapper import translateusinglocal
from extractaudio import extract_audio_for_transcription, get_media_length_in_seconds
from settings import get_settings_path
from utility import format_seconds

retained_cuda_models = []


def dispatch_translate(text, uiwrapper):
    engine = uiwrapper.get_translation_engine()
    if engine == "Claude Haiku":
        return translateusingclaude(text, uiwrapper)
    if engine == "Gemini":
        return translateusinggemini(text, uiwrapper)
    if engine == "ChatGPT":
        return translateusingopenai(text, uiwrapper)
    if engine == "Local LLM":
        return translateusinglocal(text, uiwrapper)
    return translateusingapi(text, uiwrapper)


def append_runtime_log(message):
    runtime_log_path = os.path.join(os.path.dirname(get_settings_path()), "runtime.log")
    os.makedirs(os.path.dirname(runtime_log_path), exist_ok=True)
    with open(runtime_log_path, "a", encoding="utf-8") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {message}\n")


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            append_runtime_log(f"torch.cuda.empty_cache() failed: {type(exc).__name__}: {exc}")


def should_skip_immediate_gpu_cleanup(uiwrapper):
    return uiwrapper.get_cuda_var()


def release_retained_cuda_models():
    global retained_cuda_models
    retained_cuda_models.clear()
    clear_gpu_memory()


def align_translated_lines(timestamps, translated_lines):
    slot_count = len(timestamps)
    normalized = [(line or "").strip() for line in translated_lines]

    if len(normalized) == slot_count:
        return normalized

    non_empty = [line for line in normalized if line]
    if not non_empty:
        return [""] * slot_count

    if len(non_empty) > slot_count:
        merged = []
        step = len(non_empty) / slot_count
        for index in range(slot_count):
            start = round(index * step)
            end = round((index + 1) * step)
            chunk = non_empty[start:end] or [non_empty[min(start, len(non_empty) - 1)]]
            merged.append(" ".join(chunk).strip())
        return merged

    padded = list(non_empty)
    padded.extend([""] * (slot_count - len(padded)))
    return padded


def _translate_one_batch(batch, uiwrapper):
    """Translate a single batch, gracefully handling Claude content-policy refusals
    by falling back to the original (untranslated) lines so the rest of the file
    can still complete."""
    try:
        return dispatch_translate(batch, uiwrapper)
    except ClaudeSafetyRefusalError:
        append_runtime_log(
            f"Claude refused batch of {len(batch)} lines; keeping original text for those lines"
        )
        return list(batch)


def translate_subtitle_lines(lines, uiwrapper, max_bytes=20000):
    translated_lines = []
    batch = []
    batch_size = 0

    for line in lines:
        candidate_size = len(line.encode("utf-8"))
        if batch and batch_size + candidate_size > max_bytes:
            translated_batch = _translate_one_batch(batch, uiwrapper)
            if not translated_batch:
                return None
            translated_lines.extend(translated_batch)
            batch = []
            batch_size = 0

        batch.append(line)
        batch_size += candidate_size

    if batch:
        translated_batch = _translate_one_batch(batch, uiwrapper)
        if not translated_batch:
            return None
        translated_lines.extend(translated_batch)

    return translated_lines


def extract_subtitle_lines(srt_path):
    with open(srt_path, "r", encoding="utf-8", errors="ignore") as srt_file:
        blocks = srt_file.read().strip().split("\n\n")

    timestamps = []
    source_lines = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) >= 3:
            timestamps.append(lines[1])
            source_lines.append(" ".join(part.strip() for part in lines[2:] if part.strip()))

    return timestamps, source_lines


def write_srt(output_path, timestamps, translated_lines):
    with open(output_path, "w", encoding="utf-8") as srt_file:
        for index, (timestamp, text) in enumerate(zip(timestamps, translated_lines), start=1):
            srt_file.write(f"{index}\n{timestamp}\n{text}\n\n")


def translate_srt_file(transcribed_srt_path, translated_srt_path, uiwrapper):
    append_runtime_log(f"Preparing translation from {transcribed_srt_path}")
    timestamps, source_lines = extract_subtitle_lines(transcribed_srt_path)
    if not source_lines:
        append_runtime_log(f"No subtitle lines found in {transcribed_srt_path}")
        return False

    append_runtime_log(f"Submitting {len(source_lines)} subtitle lines to DeepL")
    translated_lines = translate_subtitle_lines(source_lines, uiwrapper)
    if translated_lines is None:
        append_runtime_log("DeepL returned no translated lines")
        return False

    aligned_lines = align_translated_lines(timestamps, translated_lines)
    write_srt(translated_srt_path, timestamps, aligned_lines)
    append_runtime_log(f"Wrote translated subtitle file: {translated_srt_path}")
    return True


def transcribe_fast_whisper(file, option, uiwrapper, output_base=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    append_runtime_log(f"Fast whisper started for {file} on {device.type}")

    model = None
    try:
        if uiwrapper.get_cuda_var() and device.type == "cuda":
            model = WhisperModel(option, device="cuda", compute_type="float16")
        else:
            model = WhisperModel(option, device="cpu", compute_type="int8")

        srtfile = (output_base or os.path.splitext(file)[0]) + ".srt"
        if uiwrapper.get_srclanguagecodeinput() == "":
            segments, _ = model.transcribe(file, beam_size=5, word_timestamps=True,
                                           condition_on_previous_text=False)
        else:
            segments, _ = model.transcribe(
                file,
                beam_size=5,
                word_timestamps=True,
                language=uiwrapper.get_srclanguagecodeinput(),
                condition_on_previous_text=False,
            )

        allseconds = get_media_length_in_seconds(file)
        uiwrapper.update_progressbar("maximum", allseconds)

        if os.path.exists(srtfile):
            os.remove(srtfile)

        try:
            with open(srtfile, "a", encoding="utf8") as srt_output:
                for index, segment in enumerate(segments, start=1):
                    if segment.no_speech_prob > 0.6:
                        continue
                    translated_string = dispatch_translate(segment.text, uiwrapper)
                    if translated_string:
                        srt_output.write(
                            f"{index}\n{format_seconds(segment.start)} --> {format_seconds(segment.end)}\n{translated_string}\n\n"
                        )
                    uiwrapper.update_progressbar("value", segment.end)
                    uiwrapper.update_percentagelabel_post("text", "{:.2f}%".format((segment.end / allseconds) * 100))
        except ClaudeRateLimitError:
            if os.path.exists(srtfile):
                os.remove(srtfile)
                append_runtime_log(f"Deleted incomplete SRT due to rate limit: {srtfile}")
            raise

        uiwrapper.update_percentagelabel_post("text", "Finished")
    finally:
        if model is not None:
            if should_skip_immediate_gpu_cleanup(uiwrapper):
                retained_cuda_models.append(model)
                append_runtime_log(f"Retained CUDA model after fast whisper for {file}")
            else:
                del model
        if should_skip_immediate_gpu_cleanup(uiwrapper):
            append_runtime_log(f"Skipping immediate GPU cleanup after fast whisper for {file}")
        else:
            clear_gpu_memory()


def transcribe_fast_whisper_differently(file, option, uiwrapper, output_base=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    append_runtime_log(f"Fast whisper batch mode started for {file} on {device.type}")

    model = None
    try:
        if uiwrapper.get_cuda_var() and device.type == "cuda":
            model = WhisperModel(option, device="cuda", compute_type="float16")
        else:
            model = WhisperModel(option, device="cpu", compute_type="int8")

        final_base = output_base or os.path.splitext(file)[0]
        translatedsrtfile = final_base + ".srt"
        transcribedsrtfile = final_base + "_original.srt"

        if uiwrapper.get_srclanguagecodeinput() == "":
            segments, _ = model.transcribe(file, beam_size=5, word_timestamps=True,
                                           condition_on_previous_text=False)
        else:
            segments, _ = model.transcribe(
                file,
                beam_size=5,
                word_timestamps=True,
                language=uiwrapper.get_srclanguagecodeinput(),
                condition_on_previous_text=False,
            )

        allseconds = get_media_length_in_seconds(file)
        uiwrapper.update_progressbar("maximum", allseconds)

        with open(transcribedsrtfile, "w", encoding="utf8") as srt_output:
            for index, segment in enumerate(segments, start=1):
                if segment.no_speech_prob > 0.6:
                    continue
                srt_output.write(
                    f"{index}\n{format_seconds(segment.start)} --> {format_seconds(segment.end)}\n{segment.text.strip()}\n\n"
                )
                uiwrapper.update_progressbar("value", segment.end)
                uiwrapper.update_percentagelabel_post("text", "{:.2f}%".format((segment.end / allseconds) * 100))

        append_runtime_log(f"Finished transcription stage for {file}")
        uiwrapper.update_percentagelabel_post("text", "Transcription done, starting translation...")

        try:
            translated = translate_srt_file(transcribedsrtfile, translatedsrtfile, uiwrapper)
        except ClaudeRateLimitError:
            if os.path.exists(translatedsrtfile):
                os.remove(translatedsrtfile)
                append_runtime_log(f"Deleted incomplete translated SRT: {translatedsrtfile}")
            if not uiwrapper.get_original_var() and os.path.exists(transcribedsrtfile):
                os.remove(transcribedsrtfile)
                append_runtime_log(f"Deleted intermediate transcribed SRT: {transcribedsrtfile}")
            raise

        if translated and not uiwrapper.get_original_var() and os.path.exists(transcribedsrtfile):
            os.remove(transcribedsrtfile)

        append_runtime_log(f"Leaving fast whisper batch function for {file}")
        uiwrapper.update_percentagelabel_post("text", "Finished")
    finally:
        if model is not None:
            if should_skip_immediate_gpu_cleanup(uiwrapper):
                retained_cuda_models.append(model)
                append_runtime_log(f"Retained CUDA model after fast whisper batch for {file}")
            else:
                del model
        if should_skip_immediate_gpu_cleanup(uiwrapper):
            append_runtime_log(f"Skipping immediate GPU cleanup after fast whisper batch for {file}")
        else:
            clear_gpu_memory()


def transcribe_whisper(file, option, uiwrapper, output_base=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    append_runtime_log(f"Whisper started for {file} on {device.type}")

    model = None
    try:
        if uiwrapper.get_cuda_var() and device.type == "cuda":
            model = whisper.load_model(option, device="cuda")
        else:
            model = whisper.load_model(option, device="cpu")

        final_base = output_base or os.path.splitext(file)[0]
        transcribedsrtfile = final_base + "_original.srt"
        translatedsrtfile = final_base + ".srt"

        if uiwrapper.get_srclanguagecodeinput() == "":
            result = model.transcribe(file, verbose=True)
        else:
            result = model.transcribe(file, verbose=True, language=uiwrapper.get_srclanguagecodeinput())

        writer = get_writer("srt", os.path.dirname(transcribedsrtfile) or ".")
        writer(result, transcribedsrtfile)

        allseconds = get_media_length_in_seconds(file)
        uiwrapper.update_progressbar("maximum", allseconds)

        append_runtime_log(f"Finished whisper transcription stage for {file}")
        uiwrapper.update_percentagelabel_post("text", "Transcription done, starting translation...")

        try:
            translated = translate_srt_file(transcribedsrtfile, translatedsrtfile, uiwrapper)
        except ClaudeRateLimitError:
            if os.path.exists(translatedsrtfile):
                os.remove(translatedsrtfile)
                append_runtime_log(f"Deleted incomplete translated SRT: {translatedsrtfile}")
            if not uiwrapper.get_original_var() and os.path.exists(transcribedsrtfile):
                os.remove(transcribedsrtfile)
                append_runtime_log(f"Deleted intermediate transcribed SRT: {transcribedsrtfile}")
            raise

        if translated and not uiwrapper.get_original_var() and os.path.exists(transcribedsrtfile):
            os.remove(transcribedsrtfile)

        uiwrapper.update_percentagelabel_post("text", "Finished")
    finally:
        if model is not None:
            if should_skip_immediate_gpu_cleanup(uiwrapper):
                retained_cuda_models.append(model)
                append_runtime_log(f"Retained CUDA model after whisper for {file}")
            else:
                del model
        if should_skip_immediate_gpu_cleanup(uiwrapper):
            append_runtime_log(f"Skipping immediate GPU cleanup after whisper for {file}")
        else:
            clear_gpu_memory()


def transcribe_stable_whisper(file, option, uiwrapper, output_base=None):
    def progress_notify(seek, total):
        uiwrapper.update_progressbar("maximum", total)
        uiwrapper.update_progressbar("value", seek)
        uiwrapper.update_percentagelabel_post("text", "{:.2f}%".format((seek / total) * 100))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    append_runtime_log(f"Stable whisper started for {file} on {device.type}")

    model = None
    try:
        if uiwrapper.get_cuda_var() and device.type == "cuda":
            model = stable_whisper.load_model(option, device="cuda")
        else:
            model = stable_whisper.load_model(option, device="cpu")

        final_base = output_base or os.path.splitext(file)[0]
        transcribedsrtfile = final_base + "_original.srt"
        translatedsrtfile = final_base + ".srt"

        if uiwrapper.get_srclanguagecodeinput() == "":
            result = model.transcribe(file, verbose=True, progress_callback=progress_notify)
        else:
            result = model.transcribe(
                file,
                verbose=True,
                language=uiwrapper.get_srclanguagecodeinput(),
                progress_callback=progress_notify,
            )
        result.to_srt_vtt(transcribedsrtfile, word_level=False)

        allseconds = get_media_length_in_seconds(file)
        uiwrapper.update_progressbar("maximum", allseconds)

        append_runtime_log(f"Finished stable whisper transcription stage for {file}")
        uiwrapper.update_percentagelabel_post("text", "Transcription done, starting translation...")

        try:
            translated = translate_srt_file(transcribedsrtfile, translatedsrtfile, uiwrapper)
        except ClaudeRateLimitError:
            if os.path.exists(translatedsrtfile):
                os.remove(translatedsrtfile)
                append_runtime_log(f"Deleted incomplete translated SRT: {translatedsrtfile}")
            if not uiwrapper.get_original_var() and os.path.exists(transcribedsrtfile):
                os.remove(transcribedsrtfile)
                append_runtime_log(f"Deleted intermediate transcribed SRT: {transcribedsrtfile}")
            raise

        if translated and not uiwrapper.get_original_var() and os.path.exists(transcribedsrtfile):
            os.remove(transcribedsrtfile)

        uiwrapper.update_percentagelabel_post("text", "Finished")
    finally:
        if model is not None:
            if should_skip_immediate_gpu_cleanup(uiwrapper):
                retained_cuda_models.append(model)
                append_runtime_log(f"Retained CUDA model after stable whisper for {file}")
            else:
                del model
        if should_skip_immediate_gpu_cleanup(uiwrapper):
            append_runtime_log(f"Skipping immediate GPU cleanup after stable whisper for {file}")
        else:
            clear_gpu_memory()


def transcribebypatterns(filepattern, options, uiwrapper):
    matching_files = glob.glob(filepattern)
    for matched_file in matching_files:
        audio_file = os.path.splitext(matched_file)[0] + ".mp3"
        transcribe_from_mp3_fast_whisper(matched_file, audio_file, uiwrapper)


def transcribe_from_mp3_whisper(file1, file2, uiwrapper):
    append_runtime_log(f"Entered transcribe_from_mp3_whisper with source={file1} temp_audio={file2}")
    try:
        extract_audio_for_transcription(file1, file2)
        append_runtime_log(f"Audio normalized for whisper path: {file2}")
        transcribe_stable_whisper(file2, uiwrapper.get_translateoption_Var(), uiwrapper, output_base=os.path.splitext(file1)[0])
    finally:
        if os.path.exists(file2):
            os.remove(file2)
            append_runtime_log(f"Removed temp audio for whisper path: {file2}")
    append_runtime_log(f"Leaving transcribe_from_mp3_whisper with source={file1}")


def transcribe_from_mp3_fast_whisper(file1, file2, uiwrapper):
    append_runtime_log(f"Entered transcribe_from_mp3_fast_whisper with source={file1} temp_audio={file2}")
    try:
        extract_audio_for_transcription(file1, file2)
        append_runtime_log(f"Audio normalized for fast whisper path: {file2}")
        transcribe_fast_whisper_differently(file2, uiwrapper.get_translateoption_Var(), uiwrapper, output_base=os.path.splitext(file1)[0])
    finally:
        if os.path.exists(file2):
            os.remove(file2)
            append_runtime_log(f"Removed temp audio for fast whisper path: {file2}")
    append_runtime_log(f"Leaving transcribe_from_mp3_fast_whisper with source={file1}")


def testfunc(transcribedsrcfile):
    timestamps, source_lines = extract_subtitle_lines(transcribedsrcfile)
    if not source_lines:
        print("empty")
        return

    translated_lines = []
    for line in source_lines:
        translated = translateusingapifortest(line)
        translated_lines.append(translated or "")

    write_srt(os.path.splitext(transcribedsrcfile)[0] + "_test.srt", timestamps, translated_lines)


def makesrtfortest(timestamp, strs, file):
    translated_lines = align_translated_lines(timestamp, strs.splitlines())
    write_srt(os.path.splitext(file)[0] + "_test.srt", timestamp, translated_lines)
