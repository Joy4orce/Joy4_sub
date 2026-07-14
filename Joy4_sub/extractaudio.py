from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from imageio_ffmpeg import get_ffmpeg_exe
import wave
import subprocess
import os
import re
import sys

SUPPORTED_AUDIO_EXTENSIONS = (
    ".mp3", ".wav", ".aac", ".m4a", ".flac", ".ogg", ".wma", ".opus", ".weba"
)
SUPPORTED_VIDEO_EXTENSIONS = (
    ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".mpeg", ".mpg", ".m4v",
    ".ts", ".flv", ".asf", ".ogv"
)


def extract_audio_from_video(input_video_path, output_audio_path):
    try:
        video_clip = VideoFileClip(input_video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(output_audio_path)
        audio_clip.close()
        video_clip.close()
        print("Audio extracted successfully.")
    except Exception as e:
        print(f"Error: {e}")


def extract_audio_for_transcription(input_media_path, output_audio_path):
    os.makedirs(os.path.dirname(output_audio_path) or ".", exist_ok=True)
    ffmpeg_path = get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_media_path,
        "-vn",
        "-map_metadata",
        "-1",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        output_audio_path,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffmpeg audio extraction failed")

def get_media_length_in_time(file_path):
    try:
        duration = int(get_media_length_in_seconds(file_path))
    except Exception:
        return "Unknown"
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes}:{seconds}"

def _probe_duration_with_ffmpeg(file_path):
    """Duration (seconds) via the bundled ffmpeg, parsing the
    'Duration: HH:MM:SS.ss' line from `ffmpeg -i` stderr.

    Replaces the old moviepy AudioFileClip/VideoFileClip probe, which raised
    (and spewed 'FFMPEG_AudioReader.__del__ ... has no attribute proc' noise)
    on many perfectly playable real-world files — non-RIFF .wav exports and
    mp3s whose ffmpeg banner moviepy's regex couldn't parse. ffmpeg itself
    reads all of those fine, so probing with it directly is both quieter and
    far more reliable. Returns 0.0 when the duration can't be determined."""
    ffmpeg_path = get_ffmpeg_exe()
    kwargs = dict(capture_output=True, text=True, encoding="utf-8", errors="replace")
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    # `ffmpeg -i` with no output file exits non-zero by design; we only want
    # the informational banner it writes to stderr.
    completed = subprocess.run([ffmpeg_path, "-i", file_path], **kwargs)
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", completed.stderr or "")
    if not match:
        return 0.0
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def get_media_length_in_seconds(file_path):
    # Fast path for standard RIFF .wav via the stdlib (no subprocess), but fall
    # back to the ffmpeg probe when the header isn't RIFF (some tools export
    # .wav that Python's wave module rejects yet ffmpeg plays fine).
    if get_file_extension(file_path) == ".wav":
        try:
            return get_wav_length_in_seconds(file_path)
        except Exception:
            pass
    return _probe_duration_with_ffmpeg(file_path)

def is_audio(file_path):
    return get_file_extension(file_path) in SUPPORTED_AUDIO_EXTENSIONS


def is_video(file_path):
    return get_file_extension(file_path) in SUPPORTED_VIDEO_EXTENSIONS


def is_supported_media(file_path):
    extension = get_file_extension(file_path)
    return extension in SUPPORTED_AUDIO_EXTENSIONS or extension in SUPPORTED_VIDEO_EXTENSIONS


def get_supported_media_extensions():
    return SUPPORTED_AUDIO_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS


def get_file_extension(file_path):
    return file_path[file_path.rfind("."):].lower() if "." in file_path else ""


def get_wav_length_in_seconds(file_path):
    with wave.open(file_path, "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        if not frame_rate:
            return 0.0
        return frame_count / float(frame_rate)
