from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from imageio_ffmpeg import get_ffmpeg_exe
import wave
import subprocess
import os

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

def get_media_length_in_seconds(file_path):
    if get_file_extension(file_path) == ".wav":
        return get_wav_length_in_seconds(file_path)

    if is_audio(file_path):
        clip = AudioFileClip(file_path)
    else:
        clip = VideoFileClip(file_path)

    try:
        return float(clip.duration)
    finally:
        clip.close()

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
