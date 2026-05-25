import locale
import tkinter
import tkinter.ttk
import uuid
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import tkinter.ttk as ttk

import threading
import json
import os
import re
import subprocess
import tqdm
import traceback
from datetime import datetime

from tkinterdnd2 import DND_FILES, TkinterDnD

import whisper
import stable_whisper

import UIwrapper
import localization
import sys

import queue

from extractaudio import get_media_length_in_time, get_supported_media_extensions, is_supported_media
from utility import get_file_size_in_mb, treeview_sort_column, sort_by_path, shorten_path
from settings import (
    load_settings, load_apikey, settingjson, save_apikey, get_settings_path,
    save_engine_apikey, load_engine_apikey, run_first_run_migrations,
)

run_first_run_migrations()

update_queue = queue.Queue()
multifile_queue = queue.Queue()
lock = threading.Lock()

multi_processing = False

_rate_limit_action_event = threading.Event()
_rate_limit_cancelled = False
_rate_limit_dialog_window = None

# Cooperative cancellation. Producers (transcription / translation loops,
# worker subprocesses) poll this between safe checkpoints; UI sets it when
# the user clicks the Cancel button. Cleared at the start of every new job
# so leftover state from a previous cancel can't carry over.
_cancel_event = threading.Event()

__version__ = '1.0'

defaultdir = "C:/Users"
runtime_log_path = os.path.join(os.path.dirname(get_settings_path()), "runtime.log")

def append_runtime_log(message):
    os.makedirs(os.path.dirname(runtime_log_path), exist_ok=True)
    with open(runtime_log_path, "a", encoding="utf-8") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {message}\n")


def _rate_limit_wait(q, file_info=""):
    """Worker thread: signals main thread to show dialog, then blocks until user decides or 5h timeout."""
    global _rate_limit_cancelled
    _rate_limit_cancelled = False
    _rate_limit_action_event.clear()
    q.put(("rate_limit", file_info))
    _rate_limit_action_event.wait(timeout=18001)
    return not _rate_limit_cancelled


def _show_rate_limit_dialog(file_info=""):
    """Main thread: shows countdown dialog. Sets _rate_limit_action_event when done."""
    global _rate_limit_dialog_window, _rate_limit_cancelled
    if _rate_limit_dialog_window is not None:
        return

    dialog = Toplevel(window)
    dialog.title("Claude 제한량 도달")
    dialog.geometry("420x230")
    dialog.resizable(False, False)
    dialog.protocol("WM_DELETE_WINDOW", lambda: None)
    dialog.grab_set()
    _rate_limit_dialog_window = dialog

    total_seconds = [18000]

    Label(dialog, text="Claude 사용량 한도에 도달했습니다.\n(설정된 모든 플랜에서 한도 초과)", font=("", 11, "bold")).pack(pady=(15, 2))
    if file_info:
        Label(dialog, text=f"파일: {file_info}").pack()
    countdown_label = Label(dialog, text="05:00:00", font=("", 24))
    countdown_label.pack(pady=5)
    Label(dialog, text="5시간 후 자동으로 작업을 재개합니다.\n취소하려면 아래 버튼을 클릭하세요.").pack()

    def _cancel():
        global _rate_limit_cancelled, _rate_limit_dialog_window
        _rate_limit_cancelled = True
        _rate_limit_action_event.set()
        dialog.destroy()
        _rate_limit_dialog_window = None

    Button(dialog, text="작업 취소 및 파일 삭제", command=_cancel).pack(pady=10)

    def _update_countdown():
        global _rate_limit_cancelled, _rate_limit_dialog_window
        if _rate_limit_dialog_window is None:
            return
        remaining = total_seconds[0]
        if remaining <= 0:
            _rate_limit_cancelled = False
            _rate_limit_action_event.set()
            dialog.destroy()
            _rate_limit_dialog_window = None
            return
        h = remaining // 3600
        m = (remaining % 3600) // 60
        s = remaining % 60
        countdown_label['text'] = f"{h:02d}:{m:02d}:{s:02d}"
        total_seconds[0] -= 1
        window.after(1000, _update_countdown)

    _update_countdown()


def on_engine_change():
    pass

def log_exception(context, exc_type, exc_value, exc_traceback):
    formatted = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)).strip()
    append_runtime_log(f"{context}: {formatted}")

def install_exception_hooks():
    def global_exception_hook(exc_type, exc_value, exc_traceback):
        log_exception("Unhandled exception", exc_type, exc_value, exc_traceback)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    def threading_exception_hook(args):
        log_exception(f"Thread exception in {args.thread.name}", args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = global_exception_hook
    threading.excepthook = threading_exception_hook

window = TkinterDnD.Tk()
window.title(localization.getstr('appname') + __version__ + " by whiw")
# Default size grew (1060x560 -> 1100x1000) to fit all five API Settings
# engine sections without clipping. The Local LLM frame (endpoint, model,
# 4-line system prompt, temperature, API key, preset toolbar, hint) is
# roughly 280px tall on its own; the previous 900px default was still
# cutting off the bottom of that section on standard DPI. minsize stays
# smaller so the app still launches on 1366x768 / 1440x900 laptops.
window.geometry('1100x1000')
window.minsize(1060, 700)

def report_tk_exception(exc_type, exc_value, exc_traceback):
    log_exception("Tkinter callback exception", exc_type, exc_value, exc_traceback)
    traceback.print_exception(exc_type, exc_value, exc_traceback)

install_exception_hooks()
window.report_callback_exception = report_tk_exception
append_runtime_log("Application started")

translateoption_var = tkinter.StringVar()
translateoption_var.set("large-v3-turbo")
trnanslateoptions = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo"]

cuda_var = tkinter.BooleanVar()

original_var = tkinter.BooleanVar()

fast_var = tkinter.BooleanVar(value=True)

translation_engine_var = tkinter.StringVar(value="DeepL")

# Opt-in: when ON, rename every media file whose basename contains
# Japanese characters to its translated form after that file's SRT
# is generated. Default OFF because rename is destructive and the
# user can't easily undo a bad batch of renames.
translate_filenames_var = tkinter.BooleanVar(value=False)

# Per-engine model selection
gemini_model_var = tkinter.StringVar(value="gemini-2.0-flash")
openai_model_var = tkinter.StringVar(value="gpt-4o-mini")

# Local LLM configuration (OpenAI-compatible: koboldcpp / LM Studio / Ollama / ...)
local_endpoint_var = tkinter.StringVar(value="http://localhost:5001/v1")
local_model_var = tkinter.StringVar(value="local")
local_temperature_var = tkinter.StringVar(value="0.1")

# Claude plan selection (Pro / Team) — both can be configured for auto-fallback
claude_default_plan_var = tkinter.StringVar(value="pro")

# Model option lists (cheap → expensive ordering for clarity)
GEMINI_MODEL_OPTIONS = [
    "gemini-2.5-flash-lite",  # cheapest
    "gemini-2.0-flash",       # cheap, recommended
    "gemini-2.5-flash",       # mid
    "gemini-1.5-pro",         # expensive
    "gemini-2.5-pro",         # most expensive, best quality
]
OPENAI_MODEL_OPTIONS = [
    "gpt-4o-mini",            # cheap, recommended
    "gpt-4.1-mini",           # mid
    "gpt-4.1",                # expensive
    "gpt-4o",                 # expensive, best quality
]

# Local LLM presets — quick fill for known model archetypes. Endpoint URL
# and API key are intentionally NOT touched; those are environment-specific
# and the user maintains them. Only the per-model behavior (system prompt,
# temperature, model label) is filled.
LOCAL_LLM_PRESETS = {
    "ja_ko_vn_12b": {
        "label_key": "local_preset_jakovn",
        "model": "ja-ko-vn-12b-v2",
        "temperature": "0.1",
        "system_prompt": (
            "당신은 전문 일한 번역가입니다. "
            "주어진 일본어를 한국어로 번역하세요."
        ),
    },
    "gemma4_uncensored": {
        "label_key": "local_preset_gemma4",
        "model": "Gemma-4-E4B-Uncensored",
        "temperature": "0.3",
        "system_prompt": (
            "당신은 전문 자막 번역가입니다. 주어진 본문을 직역 위주로 자연스러운 한국어로 옮기세요. "
            "설명, 주석, 사족, 메타 텍스트는 절대 추가하지 마세요. "
            "원문 외의 내용은 출력하지 마세요."
        ),
    },
}


def _get_text_widget_content(widget):
    """Return the content of a Text widget without the trailing newline."""
    try:
        return widget.get("1.0", "end-1c")
    except Exception:
        return ""


def get_deepl_key_input():
    """Return the DeepL API key (single line)."""
    try:
        return deepl_key_input.get().strip()
    except Exception:
        return ""


def get_gemini_keys_input():
    """Return the Gemini API keys (multiline)."""
    return _get_text_widget_content(gemini_keys_input)


def get_openai_keys_input():
    """Return the OpenAI API keys (multiline)."""
    return _get_text_widget_content(openai_keys_input)


def get_claude_pro_token_input():
    try:
        return claude_pro_token_input.get().strip()
    except Exception:
        return ""


def get_claude_team_token_input():
    try:
        return claude_team_token_input.get().strip()
    except Exception:
        return ""


def get_local_endpoint_input():
    return (local_endpoint_var.get() or "").strip()


def get_local_model_input():
    return (local_model_var.get() or "").strip()


def get_local_system_prompt_input():
    return _get_text_widget_content(local_system_prompt_input)


def get_local_temperature_input():
    raw = (local_temperature_var.get() or "").strip()
    try:
        return float(raw) if raw else 0.1
    except ValueError:
        return 0.1


def get_local_apikey_input():
    try:
        return local_apikey_input.get().strip()
    except Exception:
        return ""

file_list= []

# Japanese Unicode blocks: Hiragana + Katakana + CJK Unified Ideographs +
# Halfwidth Katakana. Anything matching means the filename is at least
# partially Japanese and is a candidate for translation. Other CJK scripts
# (full Korean, full Chinese without Japanese kana) are deliberately not
# matched — those are usually already in the user's preferred form.
_JAPANESE_CHAR_RE = re.compile(r'[぀-ゟ゠-ヿ一-鿿ｦ-ﾟ]')

# Characters Windows refuses in filenames + ASCII control chars.
_WINDOWS_ILLEGAL_RE = re.compile(r'[<>:"/\\|?*\x00-\x1F]')


def _has_japanese(text):
    return bool(_JAPANESE_CHAR_RE.search(text or ""))


def _sanitize_windows_filename(name):
    """Strip filesystem-illegal characters, collapse whitespace, trim
    trailing dots/spaces (which Windows silently drops). Always returns
    a non-empty string — falls back to 'untitled' if everything was
    stripped away."""
    sanitized = _WINDOWS_ILLEGAL_RE.sub('_', name or '').strip()
    sanitized = re.sub(r'\s+', ' ', sanitized).rstrip('. ')
    return sanitized or "untitled"


def _resolve_unique_path(directory, basename, extension):
    """Return a path that doesn't collide with anything already on disk.
    If `directory/basename<extension>` is free, returns that; otherwise
    appends ` (1)`, ` (2)`, ... until it finds an unused slot."""
    candidate = os.path.join(directory, basename + extension)
    if not os.path.exists(candidate):
        return candidate
    for n in range(1, 1000):
        candidate = os.path.join(directory, f"{basename} ({n}){extension}")
        if not os.path.exists(candidate):
            return candidate
    # Extremely unlikely fallback
    return os.path.join(directory, f"{basename}_{os.getpid()}{extension}")


def _batch_translate_filenames(file_list_snapshot, uiwrapper):
    """Translate every Japanese-containing basename in the file list with
    one dispatch_translate call. Returns {original_path: sanitized_new_basename}.
    Empty dict on any failure — caller falls back to keeping originals."""
    targets_paths = []
    targets_basenames = []
    for path, _spath in file_list_snapshot:
        basename = os.path.splitext(os.path.basename(path))[0]
        if _has_japanese(basename):
            targets_paths.append(path)
            targets_basenames.append(basename)

    if not targets_basenames:
        return {}

    append_runtime_log(
        f"Filename translation: {len(targets_basenames)} Japanese filenames "
        f"to translate via {uiwrapper.get_translation_engine()}"
    )

    try:
        from mywhisper import dispatch_translate
        translated = dispatch_translate(targets_basenames, uiwrapper)
    except Exception as exc:
        append_runtime_log(
            f"Filename batch translation failed: {type(exc).__name__}: {exc}"
        )
        return {}

    if not translated or len(translated) != len(targets_basenames):
        append_runtime_log(
            f"Filename translation count mismatch "
            f"(got {len(translated or [])}, expected {len(targets_basenames)}); skipping renames"
        )
        return {}

    return {
        path: _sanitize_windows_filename(t)
        for path, t in zip(targets_paths, translated)
    }


def _rename_pair_for_file(media_path, new_basename):
    """Rename media file and its sibling .srt to share `new_basename`.
    Skips silently if the new basename equals the old one (no-op) or the
    media file no longer exists. Conflicts are resolved by appending
    ` (n)`. Returns the new media path on success, None on no-op/failure."""
    if not new_basename:
        return None
    directory = os.path.dirname(media_path)
    old_basename, ext = os.path.splitext(os.path.basename(media_path))
    if new_basename == old_basename:
        return None
    if not os.path.exists(media_path):
        append_runtime_log(f"Filename rename skipped (media missing): {media_path}")
        return None

    new_media_path = _resolve_unique_path(directory, new_basename, ext)
    # Derive the SRT path from the new media path so the pair stays together.
    new_base_noext = os.path.splitext(new_media_path)[0]
    old_srt_path = os.path.splitext(media_path)[0] + ".srt"

    try:
        os.rename(media_path, new_media_path)
    except Exception as exc:
        append_runtime_log(
            f"Filename rename failed for media {media_path}: {type(exc).__name__}: {exc}"
        )
        return None

    if os.path.exists(old_srt_path):
        new_srt_path = new_base_noext + ".srt"
        try:
            os.rename(old_srt_path, new_srt_path)
        except Exception as exc:
            append_runtime_log(
                f"Filename rename failed for srt {old_srt_path}: {type(exc).__name__}: {exc}"
            )
            # Media already renamed; the SRT pair is now broken. Log and continue.

    append_runtime_log(f"Renamed: {media_path} -> {new_media_path}")
    return new_media_path
multifile_status_indicator = 0
supported_media_extensions = set(get_supported_media_extensions())


def get_supported_media_description():
    return ", ".join(ext.upper().lstrip(".") for ext in sorted(supported_media_extensions))


def show_unsupported_media_message(file_paths):
    if not file_paths:
        return
    supported_description = get_supported_media_description()
    joined_paths = "\n".join(file_paths[:5])
    if len(file_paths) > 5:
        joined_paths += f"\n... (+{len(file_paths) - 5})"
    messagebox.showerror(
        "Unsupported format",
        f"지원하지 않는 미디어 포맷입니다.\n\n{joined_paths}\n\n지원 포맷: {supported_description}",
    )


def split_dnd_files(raw_data):
    return [path for path in window.tk.splitlist(raw_data) if path]


def add_media_file_to_list(file_path):
    file_treeview.insert(
        parent='',
        index=tkinter.END,
        iid=uuid.uuid4(),
        values=[
            shorten_path(file_path),
            str(get_file_size_in_mb(file_path)) + "MB",
            get_media_length_in_time(file_path),
            "Undone",
        ],
    )
    file_list.append((file_path, shorten_path(file_path)))

def get_active_progress_queue():
    return multifile_queue if multi_processing else update_queue

class _CustomProgressBar(tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current = self.n  # Set the initial value

    def update(self, n):
        super().update(n)
        self._current += n
        active_queue = get_active_progress_queue()
        percentage = "{:.2f}".format(round((self._current / self.total) * 100, 2)) + "%"
        with lock:
            active_queue.put(("maximum", self.total))
            active_queue.put(("value", self._current))
            active_queue.put(("text", percentage))


transcribe_module = sys.modules['whisper.transcribe']
transcribe_module.tqdm.tqdm = _CustomProgressBar

#stable_transcribe_module = sys.modules['stable_whisper.whisper_word_level']
#stable_transcribe_module.tqdm = _CustomProgressBar



def on_delete_key_press(event):
    selected_items = file_treeview.selection()
    for item in selected_items:
        item_values = file_treeview.item(item, 'values')
        for path, spath in file_list:
            if spath == item_values[0]:
                file_list.remove((path, spath))
        file_treeview.delete(item)
    multifile_list_label['text'] = "0/" + str(len(file_list))

def check_multifile_status():
    # Build the spath -> path map once (O(N)) instead of scanning the
    # whole file_list inside each row.
    #
    # Skip rows already marked Done: the .srt won't disappear mid-batch,
    # so once Done they stay Done. As the batch progresses, the disk-check
    # workload on subsequent calls shrinks toward O(remaining_undone) —
    # the very last call only touches rows that genuinely haven't been
    # processed yet.
    path_by_spath = {spath: path for path, spath in file_list}

    for item in file_treeview.get_children():
        item_value = list(file_treeview.item(item, 'values'))
        if len(item_value) >= 4 and item_value[3] == "Done":
            continue
        path = path_by_spath.get(item_value[0])
        if path is None:
            continue
        base, _ = os.path.splitext(path)
        if os.path.exists(base + ".srt"):
            item_value[3] = "Done"
            file_treeview.item(item, values=tuple(item_value))

def list_label_indicate( number = 0):
    multifile_queue.put(('list', str(number) + "/" + str(len(file_list))))

def on_filetap_drop(event):
    dropped_files = split_dnd_files(event.data)
    file_path = dropped_files[0] if dropped_files else ""
    if file_path and not is_supported_media(file_path):
        show_unsupported_media_message([file_path])
        return
    targetfileEntry.delete(0, len(targetfileEntry.get()))
    targetfileEntry.insert(0, file_path)




def on_drop(event):
    dropped_files = split_dnd_files(event.data)
    unsupported_files = [file for file in dropped_files if not is_supported_media(file)]
    supported_files = [file for file in dropped_files if is_supported_media(file)]

    if unsupported_files:
        show_unsupported_media_message(unsupported_files)

    existing_paths = {os.path.normcase(os.path.abspath(p)) for p, _ in file_list}
    for file_path in supported_files:
        norm = os.path.normcase(os.path.abspath(file_path))
        if norm in existing_paths:
            continue
        existing_paths.add(norm)
        add_media_file_to_list(file_path)

    multifile_list_label['text'] = "0/" + str(len(file_list))


def _collect_media_files_recursively(folder_path):
    """Walk folder_path recursively and return a sorted list of supported media files."""
    collected = []
    for root, _dirs, files in os.walk(folder_path):
        for name in files:
            full = os.path.join(root, name)
            if is_supported_media(full):
                collected.append(full)
    collected.sort()
    return collected


def _has_existing_subtitle(media_path):
    """Return True if a translated subtitle (.srt or .vtt) already sits next
    to this media file. The recursive folder-add flow uses this to skip
    files that already have a finished translation, so re-importing a
    folder doesn't re-queue work that's already done.

    Two naming conventions are recognized side-by-side:
      1. Replace-extension: video.mp3 -> video.srt / video.vtt
      2. Append-extension:  video.mp3 -> video.mp3.srt / video.mp3.vtt
    The append form is common for tools that preserve the source extension
    so users can tell at a glance which media a subtitle belongs to.
    """
    base = os.path.splitext(media_path)[0]
    candidates = (
        base + ".srt",
        base + ".vtt",
        media_path + ".srt",
        media_path + ".vtt",
    )
    return any(os.path.exists(c) for c in candidates)


def add_folder_to_multifile():
    """Let the user pick a folder and bulk-add every supported media file inside it (recursive)."""
    folder = filedialog.askdirectory(
        title=localization.getstr('add_folder_dialog_title'),
        initialdir=defaultdir,
        mustexist=True,
    )
    if not folder:
        return

    found_files = _collect_media_files_recursively(folder)

    if not found_files:
        messagebox.showinfo(
            localization.getstr('add_folder'),
            localization.getstr('add_folder_no_files').format(
                formats=get_supported_media_description()
            ),
        )
        return

    existing_paths = {os.path.normcase(os.path.abspath(p)) for p, _ in file_list}
    added = 0
    duplicates = 0
    skipped_existing = 0
    for file_path in found_files:
        norm = os.path.normcase(os.path.abspath(file_path))
        if norm in existing_paths:
            duplicates += 1
            continue
        if _has_existing_subtitle(file_path):
            skipped_existing += 1
            continue
        existing_paths.add(norm)
        add_media_file_to_list(file_path)
        added += 1

    multifile_list_label['text'] = "0/" + str(len(file_list))
    append_runtime_log(
        f"Added {added} files from folder "
        f"(skipped: {duplicates} duplicate, {skipped_existing} already-subtitled): {folder}"
    )

    messagebox.showinfo(
        localization.getstr('add_folder'),
        localization.getstr('add_folder_summary').format(
            added=added,
            duplicates=duplicates,
            skipped_existing=skipped_existing,
            unsupported=0,
        ),
    )




def open_dialog():
    file = filedialog.askopenfilename(
        initialdir=defaultdir,
        filetypes=[
            ("Supported media", " ".join(f"*{ext}" for ext in sorted(supported_media_extensions))),
            ("All files", "*.*"),
        ],
    )
    if file and not is_supported_media(file):
        show_unsupported_media_message([file])
        return
    targetfileEntry.delete(0, len(targetfileEntry.get()))
    targetfileEntry.insert(0, file)

def should_use_fast_whisper():
    selected_model = translateoption_var.get()
    return fast_var.get() or selected_model in {"large-v3", "large-v3-turbo"}


def proceedfastwhisperthread():
    from mywhisper import transcribe_from_mp3_fast_whisper, transcribe_from_mp3_whisper, release_retained_cuda_models
    from claudewrapper import ClaudeRateLimitError
    targetfile = targetfileEntry.get()
    file_name_without_extension, _ = os.path.splitext(targetfile)
    targetfile_audio = file_name_without_extension + "_joy4sub_temp.wav"
    transferuiwrapper = UIwrapper.UIwrapper(
        update_queue, lock, "", cuda_var.get(),
        translateoption_var.get(), sourcelanguagecodeinput.get(),
        targetlanguagecodeinput.get(), original_var.get(), fast_var.get(),
        translation_engine=translation_engine_var.get(),
        rate_limit_wait_fn=lambda fi="": _rate_limit_wait(update_queue, fi),
        deepl_key=get_deepl_key_input(),
        gemini_keys=get_gemini_keys_input(),
        openai_keys=get_openai_keys_input(),
        gemini_model=gemini_model_var.get(),
        openai_model=openai_model_var.get(),
        claude_pro_token=get_claude_pro_token_input(),
        claude_team_token=get_claude_team_token_input(),
        claude_default_plan=claude_default_plan_var.get(),
        local_endpoint=get_local_endpoint_input(),
        local_model=get_local_model_input(),
        local_system_prompt=get_local_system_prompt_input(),
        local_temperature=get_local_temperature_input(),
        local_apikey=get_local_apikey_input(),
        cancel_event=_cancel_event,
        translate_filenames=translate_filenames_var.get(),
    )
    save_all_apikeys()
    settingjson(transferuiwrapper)
    append_runtime_log(f"Starting single file job: {targetfile}")
    try:
        if should_use_fast_whisper():
            transcribe_from_mp3_fast_whisper(targetfile, targetfile_audio, transferuiwrapper)
        else:
            transcribe_from_mp3_whisper(targetfile, targetfile_audio, transferuiwrapper)
    except ClaudeRateLimitError:
        release_retained_cuda_models()
        append_runtime_log(f"Claude rate limit cancelled during single file job: {targetfile}")
        update_queue.put(("text", "작업 취소됨"))
        update_queue.put(("text", "Finished"))
        return
    finally:
        # Always signal Finished so the UI resets the Cancel button back
        # to Generate, whether the job ended naturally, raised, or was
        # cancelled mid-transcription.
        if _cancel_event.is_set():
            update_queue.put(("text", localization.getstr('cancelled')))
        update_queue.put(("text", "Finished"))
    release_retained_cuda_models()
    append_runtime_log(f"Finished single file job: {targetfile}")

    # Single-file filename rename — same opt-in behavior as multifile mode.
    # Skip if the user cancelled (the .srt may be partial / missing) or if
    # the filename has no Japanese characters.
    if (translate_filenames_var.get()
            and not _cancel_event.is_set()
            and _has_japanese(os.path.splitext(os.path.basename(targetfile))[0])):
        srt_path = os.path.splitext(targetfile)[0] + ".srt"
        if os.path.exists(srt_path):
            try:
                rename_map = _batch_translate_filenames(
                    [(targetfile, targetfile)], transferuiwrapper
                )
                new_basename = rename_map.get(targetfile)
                if new_basename:
                    _rename_pair_for_file(targetfile, new_basename)
            except Exception as exc:
                append_runtime_log(
                    f"Single-file filename rename failed: {type(exc).__name__}: {exc}"
                )

# Worker subprocess emits progress events to stdout prefixed with this
# sentinel. _run_worker_with_progress() peels them off and forwards them
# into multifile_queue so the UI updates live during long worker runs
# (multifile + CUDA + fast whisper). Must match worker_subprocess.PROGRESS_SENTINEL.
WORKER_PROGRESS_SENTINEL = "__J4S_PROG__"


def _run_worker_with_progress(args, cwd, queue_obj):
    """Spawn worker_subprocess.py and stream its progress messages back into
    `queue_obj` in real time. Returns a (returncode, stdout_str, stderr_str)
    tuple after the worker exits — the same shape the caller used to get
    from subprocess.run(capture_output=True), so the surrounding rate-limit
    / failure handling didn't have to change.

    Lines on stdout that start with the worker's progress sentinel are
    JSON-decoded and pushed onto the queue as (text, value) tuples for the
    existing UI dispatcher to render. Anything else is captured verbatim
    and returned in stdout_str so it still ends up in runtime.log.
    """
    popen_kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "cwd": cwd,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,  # line-buffered on the parent side
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    proc = subprocess.Popen(args, **popen_kwargs)

    stdout_chunks = []
    stderr_chunks = []

    def _drain_stdout():
        try:
            for line in proc.stdout:
                if line.startswith(WORKER_PROGRESS_SENTINEL):
                    try:
                        payload = json.loads(line[len(WORKER_PROGRESS_SENTINEL):].strip())
                        # payload = {"m": "text"|"bar", "l": <label>, "v": <value>}
                        # Forward as (label, value) — the UI dispatcher in
                        # multifile_update_from_queue already handles
                        # "text"/"value"/"maximum" labels uniformly.
                        queue_obj.put((payload.get("l"), payload.get("v")))
                    except Exception:
                        # Mangled progress line — fall back to logging.
                        stdout_chunks.append(line)
                else:
                    stdout_chunks.append(line)
        except Exception:
            pass

    def _drain_stderr():
        try:
            for line in proc.stderr:
                stderr_chunks.append(line)
        except Exception:
            pass

    def _cancel_watcher():
        # Poll the global cancel event ~5 times per second while the worker
        # is alive. When the user hits Cancel mid-job, terminate() lets the
        # multifile + CUDA + fast-whisper path bail out within a second
        # instead of waiting for the current file's Whisper transcription
        # to complete (which can be many minutes for long audio).
        import time as _time
        while proc.poll() is None:
            if _cancel_event.is_set():
                try:
                    proc.terminate()
                except Exception:
                    pass
                return
            _time.sleep(0.2)

    t_out = threading.Thread(target=_drain_stdout, daemon=True)
    t_err = threading.Thread(target=_drain_stderr, daemon=True)
    t_cancel = threading.Thread(target=_cancel_watcher, daemon=True)
    t_out.start()
    t_err.start()
    t_cancel.start()

    proc.wait()
    t_out.join(timeout=2)
    t_err.join(timeout=2)
    # _cancel_watcher exits on its own once proc.poll() returns non-None.

    return proc.returncode, "".join(stdout_chunks), "".join(stderr_chunks)


def proceed_multifile_whisperthread():
    global multi_processing
    from mywhisper import transcribe_from_mp3_fast_whisper, transcribe_from_mp3_whisper, release_retained_cuda_models
    from claudewrapper import ClaudeRateLimitError

    try:
        transferuiwrapper = UIwrapper.UIwrapper(
            multifile_queue, lock, "", cuda_var.get(),
            translateoption_var.get(), sourcelanguagecodeinput.get(),
            targetlanguagecodeinput.get(), original_var.get(), fast_var.get(),
            translation_engine=translation_engine_var.get(),
            rate_limit_wait_fn=lambda fi="": _rate_limit_wait(multifile_queue, fi),
            deepl_key=get_deepl_key_input(),
            gemini_keys=get_gemini_keys_input(),
            openai_keys=get_openai_keys_input(),
            gemini_model=gemini_model_var.get(),
            openai_model=openai_model_var.get(),
            claude_pro_token=get_claude_pro_token_input(),
            claude_team_token=get_claude_team_token_input(),
            claude_default_plan=claude_default_plan_var.get(),
            local_endpoint=get_local_endpoint_input(),
            local_model=get_local_model_input(),
            local_system_prompt=get_local_system_prompt_input(),
            local_temperature=get_local_temperature_input(),
            local_apikey=get_local_apikey_input(),
        )
        save_all_apikeys()
        settingjson(transferuiwrapper)

        # If filename translation is enabled, batch-translate all Japanese
        # basenames up front in ONE engine call. 267 files of Claude
        # Haiku batched cost ~one extra call instead of 267 individual
        # calls. Result is a path -> new_basename map; missing or
        # non-Japanese entries are left out and stay as their originals.
        filename_rename_map = {}
        if translate_filenames_var.get():
            multifile_queue.put(("text", localization.getstr('translating_filenames')))
            try:
                filename_rename_map = _batch_translate_filenames(
                    list(file_list), transferuiwrapper
                )
                append_runtime_log(
                    f"Filename rename map: {len(filename_rename_map)} entries prepared"
                )
            except Exception as exc:
                append_runtime_log(
                    f"Filename pre-batch failed: {type(exc).__name__}: {exc}"
                )
                filename_rename_map = {}

        for i, (file, spath) in enumerate(file_list):
            if _cancel_event.is_set():
                append_runtime_log(f"Multifile cancelled by user at item {i + 1}/{len(file_list)}")
                multifile_queue.put(("text", localization.getstr('cancelled')))
                break
            append_runtime_log(f"Starting multifile item {i + 1}/{len(file_list)}: {file}")
            list_label_indicate(i+1)
            append_runtime_log(f"Updated list label for multifile item {i + 1}/{len(file_list)}")
            file_name_without_extension, _ = os.path.splitext(file)
            file_audio = file_name_without_extension + "_joy4sub_temp.wav"
            append_runtime_log(f"Prepared temp audio path for multifile item {i + 1}/{len(file_list)}: {file_audio}")
            if should_use_fast_whisper() and cuda_var.get():
                append_runtime_log(f"Dispatching subprocess fast whisper for multifile item {i + 1}/{len(file_list)}")
                multifile_queue.put(("text", f"Launching worker {i + 1}/{len(file_list)}"))
                worker_script = os.path.join(os.path.dirname(__file__), "worker_subprocess.py")
                worker_args = [
                    sys.executable,
                    worker_script,
                    "--file", file,
                    "--temp-audio", file_audio,
                    "--model", translateoption_var.get(),
                    "--target-lang", targetlanguagecodeinput.get(),
                    "--deepl-key", get_deepl_key_input(),
                    "--gemini-keys", get_gemini_keys_input(),
                    "--openai-keys", get_openai_keys_input(),
                    "--gemini-model", gemini_model_var.get(),
                    "--openai-model", openai_model_var.get(),
                    "--claude-pro-token", get_claude_pro_token_input(),
                    "--claude-team-token", get_claude_team_token_input(),
                    "--claude-default-plan", claude_default_plan_var.get(),
                    "--local-endpoint", get_local_endpoint_input(),
                    "--local-model", get_local_model_input(),
                    "--local-system-prompt", get_local_system_prompt_input(),
                    "--local-temperature", str(get_local_temperature_input()),
                    "--local-apikey", get_local_apikey_input(),
                ]
                if sourcelanguagecodeinput.get():
                    worker_args.extend(["--source-lang", sourcelanguagecodeinput.get()])
                if original_var.get():
                    worker_args.append("--keep-original")
                if fast_var.get():
                    worker_args.append("--fast")
                if cuda_var.get():
                    worker_args.append("--cuda")
                worker_args.extend(["--engine", translation_engine_var.get()])

                rate_limit_cancelled = False
                while True:
                    # _run_worker_with_progress streams the worker's progress
                    # events back into multifile_queue in real time, so the
                    # UI no longer freezes on the post-transcription text
                    # for the entire CUDA+fast-whisper run. Returns the same
                    # (returncode, stdout, stderr) shape the old subprocess.run
                    # did, with progress lines stripped from stdout.
                    returncode, worker_stdout, worker_stderr = _run_worker_with_progress(
                        worker_args,
                        cwd=os.path.dirname(__file__),
                        queue_obj=multifile_queue,
                    )
                    append_runtime_log(f"Worker process exited with code {returncode} for {file}")
                    if worker_stdout.strip():
                        append_runtime_log(f"Worker stdout for {file}: {worker_stdout.strip()}")
                    if worker_stderr.strip():
                        append_runtime_log(f"Worker stderr for {file}: {worker_stderr.strip()}")
                    if returncode == 2:
                        append_runtime_log(f"Claude rate limit at subprocess item {i + 1}/{len(file_list)}: {file}")
                        multifile_queue.put(("text", f"Claude 제한량 도달 ({i + 1}/{len(file_list)}) - 대기 중..."))
                        should_resume = _rate_limit_wait(multifile_queue, os.path.basename(file))
                        if should_resume:
                            append_runtime_log(f"Retrying subprocess after rate limit reset: {file}")
                            continue
                        else:
                            append_runtime_log(f"Rate limit cancelled at subprocess item {i + 1}/{len(file_list)}: {file}")
                            multifile_queue.put(("text", "작업 취소됨"))
                            rate_limit_cancelled = True
                            break
                    elif returncode != 0:
                        multifile_queue.put(("text", f"Worker failed: {os.path.basename(file)}"))
                        break
                    else:
                        break
                if rate_limit_cancelled:
                    break
            elif should_use_fast_whisper():
                append_runtime_log(f"Dispatching fast whisper for multifile item {i + 1}/{len(file_list)}")
                try:
                    transcribe_from_mp3_fast_whisper(file, file_audio, transferuiwrapper)
                except ClaudeRateLimitError:
                    append_runtime_log(f"Rate limit cancelled at multifile item {i + 1}/{len(file_list)}: {file}")
                    multifile_queue.put(("text", "작업 취소됨"))
                    break
            else:
                append_runtime_log(f"Dispatching stable whisper for multifile item {i + 1}/{len(file_list)}")
                try:
                    transcribe_from_mp3_whisper(file, file_audio, transferuiwrapper)
                except ClaudeRateLimitError:
                    append_runtime_log(f"Rate limit cancelled at multifile item {i + 1}/{len(file_list)}: {file}")
                    multifile_queue.put(("text", "작업 취소됨"))
                    break
            append_runtime_log(f"Returned from worker for multifile item {i + 1}/{len(file_list)}: {file}")
            append_runtime_log(f"Finished multifile item {i + 1}/{len(file_list)}: {file}")
            # Rename media + srt to translated basename, if this file was
            # in the pre-batched rename map AND the .srt was actually
            # produced. The disk check inside _rename_pair_for_file
            # prevents renaming when the media file is somehow gone.
            new_basename = filename_rename_map.get(file)
            if new_basename:
                srt_path = os.path.splitext(file)[0] + ".srt"
                if os.path.exists(srt_path):
                    _rename_pair_for_file(file, new_basename)
                else:
                    append_runtime_log(
                        f"Filename rename skipped (srt not produced): {file}"
                    )
            # Signal the UI consumer to refresh the Status column for this
            # file. We don't pass the path — check_multifile_status walks the
            # treeview and skips rows already marked Done, so it stays cheap
            # as the batch progresses. Emitted regardless of success/failure;
            # the disk check inside the refresher decides Done vs leave-alone.
            multifile_queue.put(("__file_done__", None))
        release_retained_cuda_models()
        append_runtime_log("Released retained CUDA models after multifile batch")
    except Exception as exc:
        append_runtime_log(f"Multifile worker failed: {type(exc).__name__}: {exc}")
        multifile_queue.put(("text", f"Failed: {type(exc).__name__}"))
    finally:
        # Use a dedicated sentinel for end-of-batch, NOT ("text", "Finished").
        # Inner transcribe functions emit ("text", "Finished") on every
        # per-file completion (a single-file-mode convention), and when
        # those leak into multifile_queue the consumer would treat the
        # first one as end-of-batch and stop rescheduling itself —
        # leaving the UI frozen on file 1 while disk work continued.
        # See multifile_update_from_queue for the matching consumer side.
        multifile_queue.put(("__job_done__", None))
        multi_processing = False


def multifile_update_from_queue():
    # Drain the entire queue greedily and apply only the most-recent value
    # per message kind. The old "one message per 100 ms" loop fell minutes
    # behind during long Claude / Local LLM batches across many files —
    # producers (engine wrappers, batch loops, worker subprocesses) can
    # emit dozens of updates per second, while the consumer was capped at
    # 10 per second, so the visible "1/267" lagged 30+ files behind reality.
    #
    # Collapsing to "latest wins" per kind is correct here because each kind
    # describes current state, not an event stream: the user only needs the
    # current bar value, current text, current list counter — not the
    # history. Special signals (rate_limit, Finished) are tracked separately
    # so they're not lost in the collapse.
    latest = {}
    rate_limit_payload = None
    finished = False
    file_done_pending = False
    drained = 0
    DRAIN_CAP = 500  # safety: never starve the Tk main loop on a flood

    while drained < DRAIN_CAP:
        try:
            msg = multifile_queue.get_nowait()
        except queue.Empty:
            break
        drained += 1
        kind, val = msg[0], msg[1]
        if kind == "rate_limit":
            rate_limit_payload = val
        elif kind == "__job_done__":
            # Real end-of-batch — only emitted by proceed_multifile_whisperthread's
            # finally block after the for-loop has finished (or been
            # cancelled). Consumer exits here.
            finished = True
        elif kind == "__file_done__":
            # One file just completed (success or failure). Defer the
            # treeview status refresh until after the drain so multiple
            # file completions in the same tick coalesce into one walk.
            file_done_pending = True
        elif kind == "text" and val == "Finished":
            # Per-file leak from inner transcribe functions. Inner
            # transcribe emits ("text", "Finished") at the end of each
            # file (single-file-mode convention); in multifile context
            # that's just "one of N files done" — explicitly ignore so
            # the consumer doesn't stop on the first file's completion.
            pass
        else:
            latest[kind] = val

    if "maximum" in latest:
        with lock:
            multifile_progressbar['maximum'] = latest["maximum"]
    if "value" in latest:
        with lock:
            multifile_progressbar['value'] = latest["value"]
    if "list" in latest:
        with lock:
            multifile_list_label['text'] = latest["list"]
            # NOTE: do NOT call check_multifile_status() here. It walks the
            # entire treeview against the file_list and does an os.path.exists
            # per row — on a 200+ file batch that blocked the Tk main thread
            # for seconds per drain, which in turn delayed every subsequent
            # tick and made the list counter look frozen even though the
            # worker was advancing through files normally. Status column
            # is refreshed at start of batch and on Finished (below) — those
            # are the only points where the disk state changes meaningfully.
    if "text" in latest:
        with lock:
            multifile_status_label['text'] = latest["text"]

    if file_done_pending:
        with lock:
            # Refresh the Status column. The function skips rows already
            # Done so per-call cost shrinks across the batch — only the
            # still-undone rows incur a disk check. Coalesced when multiple
            # files complete in the same drain tick.
            check_multifile_status()

    if rate_limit_payload is not None:
        _show_rate_limit_dialog(rate_limit_payload)

    if finished:
        with lock:
            all_done = multifile_list_label['text'].split("/")[0] == str(len(file_list))
            if all_done:
                multifile_status_label['text'] = "Finished"
            # Restore the Generate ↔ Cancel toggle to its idle state on
            # BOTH buttons regardless of which mode just finished —
            # otherwise the user-cancelled multifile button could be
            # stuck on "취소 중..." after the worker thread exits.
            _reset_generate_buttons()
            multifile_progressbar['value'] = 0
            check_multifile_status()
            file_treeview.bind('<Delete>', on_delete_key_press)
        return

    window.after(100, multifile_update_from_queue)



def update_ui_from_queue():
    # Single-file tab: same drain-and-collapse pattern as the multifile tab
    # (see multifile_update_from_queue docstring above for the rationale).
    # Single-file jobs emit fewer updates so backlog is less visible here,
    # but the same producers run so a slow Tk tick could still cause lag.
    latest = {}
    rate_limit_payload = None
    finished = False
    drained = 0
    DRAIN_CAP = 500

    while drained < DRAIN_CAP:
        try:
            msg = update_queue.get_nowait()
        except queue.Empty:
            break
        drained += 1
        kind, val = msg[0], msg[1]
        if kind == "rate_limit":
            rate_limit_payload = val
        elif kind == "text" and val == "Finished":
            finished = True
        elif val == "Finished":
            finished = True
        else:
            latest[kind] = val

    if "maximum" in latest:
        with lock:
            progressbar['maximum'] = latest["maximum"]
    if "value" in latest:
        with lock:
            progressbar['value'] = latest["value"]
    if "text" in latest:
        with lock:
            percentagelabel['text'] = latest["text"]

    if rate_limit_payload is not None:
        _show_rate_limit_dialog(rate_limit_payload)

    if finished:
        with lock:
            percentagelabel['text'] = "Finished"
            _reset_generate_buttons()
            progressbar['value'] = 0
        return

    window.after(100, update_ui_from_queue)


def initialize():
    # Always load API keys from per-engine credential storage (independent of settings.json)
    load_all_apikeys()
    if not os.path.exists(get_settings_path()):
        return
    settings = load_settings()
    cuda_var.set(settings.get("cuda_var", False))
    saved_model = settings.get("translateoption_var", "large-v3-turbo")
    if saved_model == "small":
        saved_model = "large-v3-turbo"
    translateoption_var.set(saved_model)
    sourcelanguagecodeinput.insert(0, settings.get("sourcelanguagecodeinput", ""))
    targetlanguagecodeinput.insert(0, settings.get("targetlanguagecodeinput", ""))
    original_var.set(settings.get("original", False))
    fast_var.set(settings.get("fast", True) or saved_model in {"large-v3", "large-v3-turbo"})
    translation_engine_var.set(settings.get("translation_engine", "DeepL"))
    saved_gemini_model = settings.get("gemini_model", "")
    if saved_gemini_model:
        gemini_model_var.set(saved_gemini_model)
    saved_openai_model = settings.get("openai_model", "")
    if saved_openai_model:
        openai_model_var.set(saved_openai_model)
    saved_claude_default_plan = (settings.get("claude_default_plan", "pro") or "pro").lower()
    if saved_claude_default_plan in ("pro", "team"):
        claude_default_plan_var.set(saved_claude_default_plan)
    translate_filenames_var.set(bool(settings.get("translate_filenames", False)))
    saved_local_endpoint = settings.get("local_endpoint", "")
    if saved_local_endpoint:
        local_endpoint_var.set(saved_local_endpoint)
    saved_local_model = settings.get("local_model", "")
    if saved_local_model:
        local_model_var.set(saved_local_model)
    saved_local_system_prompt = settings.get("local_system_prompt", "")
    if saved_local_system_prompt:
        local_system_prompt_input.delete("1.0", "end")
        local_system_prompt_input.insert("1.0", saved_local_system_prompt)
    saved_local_temperature = settings.get("local_temperature", None)
    if saved_local_temperature is not None:
        local_temperature_var.set(str(saved_local_temperature))


def request_cancel():
    """Mark the running job as cancelled. The worker thread sees this at
    the next checkpoint and shuts down — between files in the multifile
    loop, between batches in translate_subtitle_lines, between retries in
    the Local LLM / Claude wrappers. The subprocess worker is also killed
    immediately by the cancel watcher inside _run_worker_with_progress.

    The button is disabled and switched to "취소 중..." while cancellation
    propagates; the Finished handler restores it to "자막생성" when the
    worker thread actually exits."""
    if not _cancel_event.is_set():
        _cancel_event.set()
        append_runtime_log("User requested cancellation")
    try:
        proceedbutton.config(state=DISABLED, text=localization.getstr('cancelling'))
    except Exception:
        pass
    try:
        multifile_generation_button.config(state=DISABLED, text=localization.getstr('cancelling'))
    except Exception:
        pass


def _reset_generate_buttons():
    """Restore both Generate buttons to their idle state. Called from the
    Finished handler regardless of whether the job ended naturally or via
    cancel — same UI outcome either way."""
    try:
        proceedbutton.config(state=NORMAL, text=localization.getstr('generate'), command=proceed)
    except Exception:
        pass
    try:
        multifile_generation_button.config(state=NORMAL, text=localization.getstr('generate'), command=proceedmultifile)
    except Exception:
        pass


def proceed():
    target_file = targetfileEntry.get().strip()
    if not target_file or not os.path.exists(target_file):
        messagebox.showerror("File not found", "처리할 미디어 파일을 선택해 주세요.")
        return
    if not is_supported_media(target_file):
        show_unsupported_media_message([target_file])
        return
    # Fresh cancellation slate every run.
    _cancel_event.clear()
    # Generate -> Cancel toggle. Keeps the button enabled so the user can
    # actually stop a long job; multifile button is disabled to prevent
    # starting two jobs at once.
    proceedbutton.config(text=localization.getstr('cancel'), command=request_cancel)
    multifile_generation_button.config(state=DISABLED)
    update_ui_from_queue()
    thread = threading.Thread(target=proceedfastwhisperthread)
    thread.start()

def proceedmultifile():
    global multi_processing
    if multi_processing:
        append_runtime_log("Ignored duplicate multifile start request")
        return
    if not file_list:
        messagebox.showerror("No files", "처리할 미디어 파일을 먼저 추가해 주세요.")
        return
    invalid_files = [path for path, _ in file_list if not is_supported_media(path)]
    if invalid_files:
        show_unsupported_media_message(invalid_files)
        return

    multi_processing = True
    _cancel_event.clear()
    # Same Generate -> Cancel toggle pattern as single-file mode.
    multifile_generation_button.config(text=localization.getstr('cancel'), command=request_cancel)
    proceedbutton.config(state=DISABLED)
    file_treeview.unbind('<Delete>')
    multifile_update_from_queue()
    thread = threading.Thread(target=proceed_multifile_whisperthread)
    thread.start()

#window.drop_target_register(DND_FILES)
#window.dnd_bind('<<Drop>>', on_drop)

notebook = ttk.Notebook(window)
# Fill the entire window so each tab can stretch with the user's resize.
notebook.pack(expand=True, fill='both')

frame1 = Frame(window)
frame1.drop_target_register(DND_FILES)
frame1.dnd_bind('<<Drop>>', on_filetap_drop)

multifileframe = Frame(window)
multifileframe.drop_target_register(DND_FILES)
multifileframe.dnd_bind('<<Drop>>', on_drop)

apisettingsframe = Frame(window)


notebook.add(frame1, text="File")
notebook.add(multifileframe, text="Multifile")
notebook.add(apisettingsframe, text=localization.getstr('api_settings_tab'))

# File tab layout: a single centered form with right-aligned labels and
# left-aligned inputs, padded so it fills the larger default window
# without looking empty. Empty rows above/below the form give it
# vertical breathing room without committing to a fixed offset.
frame1.grid_columnconfigure(0, weight=1)
frame1.grid_rowconfigure(0, weight=1)
frame1.grid_rowconfigure(2, weight=1)

form_frame = Frame(frame1)
form_frame.grid(column=0, row=1, padx=24, pady=8)
# Two-column form: labels on the left, inputs on the right. minsize keeps
# the input column wide enough that long entries don't visually wrap.
form_frame.grid_columnconfigure(0, weight=0, minsize=140)
form_frame.grid_columnconfigure(1, weight=1, minsize=520)

_FORM_PADX = (0, 10)
_FORM_PADY = 5

# Row 0 — file picker
button = Button(form_frame, text=localization.getstr('selectfile'), command=open_dialog)
button.grid(column=0, row=0, sticky='e', padx=_FORM_PADX, pady=_FORM_PADY)
targetfileEntry = Entry(form_frame)
targetfileEntry.insert(0, localization.getstr('selectinstruction'))
targetfileEntry.grid(column=1, row=0, sticky='ew', pady=_FORM_PADY)

# Row 1 — Whisper model + flags
_model_label_box = Frame(form_frame)
_model_label_box.grid(column=0, row=1, sticky='e', padx=_FORM_PADX, pady=_FORM_PADY)
Label(_model_label_box, text=localization.getstr('choosemodel')).grid(column=0, row=0)
fastoption = Checkbutton(_model_label_box, text="Fast", variable=fast_var)
fastoption.grid(column=1, row=0, padx=(8, 0))

frame2 = Frame(form_frame)
frame2.grid(column=1, row=1, sticky='w', pady=_FORM_PADY)
modeldropdown = ttk.Combobox(frame2, textvariable=translateoption_var, values=trnanslateoptions, width=22)
modeldropdown.grid(column=0, row=0)
checkbox = ttk.Checkbutton(frame2, text="Cuda", variable=cuda_var)
checkbox.grid(column=1, row=0, padx=(10, 0))

# Row 2 — source language code
Label(form_frame, text=localization.getstr('sourcelangcode'), justify='left'
      ).grid(column=0, row=2, sticky='ne', padx=_FORM_PADX, pady=_FORM_PADY)
sourcelanguagecodeinput = Entry(form_frame)
sourcelanguagecodeinput.grid(column=1, row=2, sticky='ew', pady=_FORM_PADY)

# Row 3 — target language code
Label(form_frame, text=localization.getstr('targetlangcode')
      ).grid(column=0, row=3, sticky='e', padx=_FORM_PADX, pady=_FORM_PADY)
targetlanguagecodeinput = Entry(form_frame)
targetlanguagecodeinput.grid(column=1, row=3, sticky='ew', pady=_FORM_PADY)

# Row 4 — translation engine selector (spans both columns)
engine_frame = Frame(form_frame)
engine_frame.grid(column=0, row=4, columnspan=2, sticky='ew', pady=(12, 0))
Label(engine_frame, text=localization.getstr('translation_engine')
      ).grid(column=0, row=0, padx=(0, 10))
Radiobutton(engine_frame, text="DeepL", variable=translation_engine_var, value="DeepL",
            command=lambda: on_engine_change()).grid(column=1, row=0, padx=2)
Radiobutton(engine_frame, text="Claude Haiku", variable=translation_engine_var, value="Claude Haiku",
            command=lambda: on_engine_change()).grid(column=2, row=0, padx=2)
Radiobutton(engine_frame, text="Gemini", variable=translation_engine_var, value="Gemini",
            command=lambda: on_engine_change()).grid(column=3, row=0, padx=2)
Radiobutton(engine_frame, text="ChatGPT", variable=translation_engine_var, value="ChatGPT",
            command=lambda: on_engine_change()).grid(column=4, row=0, padx=2)
Radiobutton(engine_frame, text="Local LLM", variable=translation_engine_var, value="Local LLM",
            command=lambda: on_engine_change()).grid(column=5, row=0, padx=2)
Label(engine_frame, text=localization.getstr('apikey_multiline_hint'), fg="#666"
      ).grid(column=0, row=1, columnspan=6, sticky='w', pady=(4, 0))
# Filename translation toggle. Sits on its own row under the engine
# selector since it's a translation-behavior switch, not an engine
# pick. Applies to both single-file and multifile modes.
Checkbutton(engine_frame, text=localization.getstr('translate_filenames'),
            variable=translate_filenames_var
            ).grid(column=0, row=2, columnspan=6, sticky='w', pady=(2, 0))

# Row 5 — primary action: generate + original-too checkbox
frame3 = Frame(form_frame)
frame3.grid(column=0, row=5, columnspan=2, pady=(18, 4))
proceedbutton = Button(frame3, text=localization.getstr('generate'), command=proceed)
proceedbutton.grid(column=0, row=0, padx=(0, 12))
originalcheckbox = Checkbutton(frame3, text=localization.getstr('original'), variable=original_var)
originalcheckbox.grid(column=1, row=0)

# Row 6 — progress bar (spans both columns so it visually represents the file)
progressbar = ttk.Progressbar(form_frame, length=400, maximum=20)
progressbar.grid(column=0, row=6, columnspan=2, sticky='ew', pady=(10, 4))

# Row 7 — current-status text
percentagelabel = Label(form_frame, text="0%")
percentagelabel.grid(column=0, row=7, columnspan=2)

# Footer credit, anchored to the bottom of the tab.
Label(frame1, text="JoyForce", fg="#888").grid(column=0, row=3, pady=(0, 10))


# ============================================================
# API Settings tab — separate API key input per engine + model
# ============================================================
api_inner = Frame(apisettingsframe)
api_inner.pack(padx=10, pady=10, fill='both', expand=True)

Label(api_inner, text=localization.getstr('api_settings_intro'),
      fg="#444").grid(column=0, row=0, sticky='w', columnspan=2, pady=(0, 8))

# DeepL
deepl_frame = ttk.LabelFrame(api_inner, text="DeepL")
deepl_frame.grid(column=0, row=1, sticky='ew', padx=5, pady=5)
Label(deepl_frame, text=localization.getstr('apikey_label')).grid(column=0, row=0, sticky='w', padx=5, pady=4)
deepl_key_input = Entry(deepl_frame, width=50)
deepl_key_input.grid(column=1, row=0, sticky='ew', padx=5, pady=4)
deepl_frame.columnconfigure(1, weight=1)

# Gemini
gemini_frame = ttk.LabelFrame(api_inner, text="Google Gemini")
gemini_frame.grid(column=0, row=2, sticky='ew', padx=5, pady=5)
Label(gemini_frame, text=localization.getstr('apikey_label_multi')).grid(column=0, row=0, sticky='nw', padx=5, pady=4)
gemini_keys_input = tkinter.Text(gemini_frame, width=50, height=4, wrap="none")
gemini_keys_input.grid(column=1, row=0, sticky='ew', padx=5, pady=4)
Label(gemini_frame, text=localization.getstr('model_label')).grid(column=0, row=1, sticky='w', padx=5, pady=4)
gemini_model_combo = ttk.Combobox(gemini_frame, textvariable=gemini_model_var, values=GEMINI_MODEL_OPTIONS, width=30)
gemini_model_combo.grid(column=1, row=1, sticky='w', padx=5, pady=4)
gemini_frame.columnconfigure(1, weight=1)

# ChatGPT
openai_frame = ttk.LabelFrame(api_inner, text="ChatGPT (OpenAI)")
openai_frame.grid(column=0, row=3, sticky='ew', padx=5, pady=5)
Label(openai_frame, text=localization.getstr('apikey_label_multi')).grid(column=0, row=0, sticky='nw', padx=5, pady=4)
openai_keys_input = tkinter.Text(openai_frame, width=50, height=4, wrap="none")
openai_keys_input.grid(column=1, row=0, sticky='ew', padx=5, pady=4)
Label(openai_frame, text=localization.getstr('model_label')).grid(column=0, row=1, sticky='w', padx=5, pady=4)
openai_model_combo = ttk.Combobox(openai_frame, textvariable=openai_model_var, values=OPENAI_MODEL_OPTIONS, width=30)
openai_model_combo.grid(column=1, row=1, sticky='w', padx=5, pady=4)
openai_frame.columnconfigure(1, weight=1)

# Claude (Pro + Team OAuth tokens with auto-fallback)
claude_frame = ttk.LabelFrame(api_inner, text="Claude Haiku")
claude_frame.grid(column=0, row=4, sticky='ew', padx=5, pady=5)
Label(claude_frame, text=localization.getstr('claude_pro_token_label')).grid(column=0, row=0, sticky='w', padx=5, pady=4)
claude_pro_token_input = Entry(claude_frame, width=50, show="*")
claude_pro_token_input.grid(column=1, row=0, sticky='ew', padx=5, pady=4)
Label(claude_frame, text=localization.getstr('claude_team_token_label')).grid(column=0, row=1, sticky='w', padx=5, pady=4)
claude_team_token_input = Entry(claude_frame, width=50, show="*")
claude_team_token_input.grid(column=1, row=1, sticky='ew', padx=5, pady=4)
Label(claude_frame, text=localization.getstr('claude_default_plan_label')).grid(column=0, row=2, sticky='w', padx=5, pady=4)
claude_plan_frame = Frame(claude_frame)
claude_plan_frame.grid(column=1, row=2, sticky='w', padx=5, pady=4)
Radiobutton(claude_plan_frame, text="Pro", variable=claude_default_plan_var, value="pro").grid(column=0, row=0)
Radiobutton(claude_plan_frame, text="Team", variable=claude_default_plan_var, value="team").grid(column=1, row=0, padx=(10, 0))
Label(claude_frame, text=localization.getstr('claude_token_hint'), fg="#666", wraplength=520, justify='left').grid(
    column=0, row=3, columnspan=2, sticky='w', padx=5, pady=(0, 4))
claude_frame.columnconfigure(1, weight=1)

# Local LLM (OpenAI-compatible: koboldcpp / LM Studio / Ollama / llama.cpp / vLLM)
local_frame = ttk.LabelFrame(api_inner, text=localization.getstr('local_llm_section'))
local_frame.grid(column=0, row=5, sticky='ew', padx=5, pady=5)

Label(local_frame, text=localization.getstr('local_endpoint_label')).grid(column=0, row=0, sticky='w', padx=5, pady=4)
local_endpoint_input = Entry(local_frame, textvariable=local_endpoint_var, width=50)
local_endpoint_input.grid(column=1, row=0, sticky='ew', padx=5, pady=4)

Label(local_frame, text=localization.getstr('local_model_label')).grid(column=0, row=1, sticky='w', padx=5, pady=4)
local_model_input = Entry(local_frame, textvariable=local_model_var, width=50)
local_model_input.grid(column=1, row=1, sticky='ew', padx=5, pady=4)

Label(local_frame, text=localization.getstr('local_system_prompt_label')).grid(column=0, row=2, sticky='nw', padx=5, pady=4)
local_system_prompt_input = tkinter.Text(local_frame, width=50, height=4, wrap="word")
local_system_prompt_input.grid(column=1, row=2, sticky='ew', padx=5, pady=4)
local_system_prompt_input.insert("1.0", "당신은 전문 일한 번역가입니다. 주어진 일본어를 한국어로 번역하세요.")

Label(local_frame, text=localization.getstr('local_temperature_label')).grid(column=0, row=3, sticky='w', padx=5, pady=4)
local_temperature_input = Entry(local_frame, textvariable=local_temperature_var, width=10)
local_temperature_input.grid(column=1, row=3, sticky='w', padx=5, pady=4)

Label(local_frame, text=localization.getstr('local_apikey_label')).grid(column=0, row=4, sticky='w', padx=5, pady=4)
local_apikey_input = Entry(local_frame, width=50, show="*")
local_apikey_input.grid(column=1, row=4, sticky='ew', padx=5, pady=4)

def _apply_local_llm_preset(preset_key):
    """Fill the Local LLM model / temperature / system-prompt fields with
    a known preset. Endpoint URL and API key are deliberately untouched —
    those are environment-specific and the user maintains them."""
    preset = LOCAL_LLM_PRESETS.get(preset_key)
    if not preset:
        return
    local_model_var.set(preset["model"])
    local_temperature_var.set(preset["temperature"])
    local_system_prompt_input.delete("1.0", "end")
    local_system_prompt_input.insert("1.0", preset["system_prompt"])
    append_runtime_log(f"Local LLM preset applied: {preset_key}")


# Preset toolbar — quick fill for known model archetypes.
local_preset_frame = Frame(local_frame)
local_preset_frame.grid(column=0, row=5, columnspan=2, sticky='w', padx=5, pady=(2, 4))
Label(local_preset_frame, text=localization.getstr('local_preset_label')).grid(column=0, row=0, padx=(0, 6))
Button(
    local_preset_frame,
    text=localization.getstr('local_preset_jakovn'),
    command=lambda: _apply_local_llm_preset("ja_ko_vn_12b"),
).grid(column=1, row=0, padx=2)
Button(
    local_preset_frame,
    text=localization.getstr('local_preset_gemma4'),
    command=lambda: _apply_local_llm_preset("gemma4_uncensored"),
).grid(column=2, row=0, padx=2)

Label(local_frame, text=localization.getstr('local_llm_hint'), fg="#666", wraplength=520, justify='left').grid(
    column=0, row=6, columnspan=2, sticky='w', padx=5, pady=(0, 4))
local_frame.columnconfigure(1, weight=1)

api_inner.columnconfigure(0, weight=1)


def save_all_apikeys():
    """Persist API keys for all engines to Windows Credential Manager."""
    save_engine_apikey("deepl", get_deepl_key_input())
    save_engine_apikey("gemini", get_gemini_keys_input())
    save_engine_apikey("openai", get_openai_keys_input())
    save_engine_apikey("claude_pro", get_claude_pro_token_input())
    save_engine_apikey("claude_team", get_claude_team_token_input())
    save_engine_apikey("local", get_local_apikey_input())


def load_all_apikeys():
    """Load API keys from Windows Credential Manager into the UI fields."""
    deepl = load_engine_apikey("deepl") or ""
    gemini = load_engine_apikey("gemini") or ""
    openai_k = load_engine_apikey("openai") or ""
    claude_pro = load_engine_apikey("claude_pro") or ""
    claude_team = load_engine_apikey("claude_team") or ""
    if deepl:
        deepl_key_input.delete(0, "end")
        deepl_key_input.insert(0, deepl)
    if gemini:
        gemini_keys_input.delete("1.0", "end")
        gemini_keys_input.insert("1.0", gemini)
    if openai_k:
        openai_keys_input.delete("1.0", "end")
        openai_keys_input.insert("1.0", openai_k)
    if claude_pro:
        claude_pro_token_input.delete(0, "end")
        claude_pro_token_input.insert(0, claude_pro)
    if claude_team:
        claude_team_token_input.delete(0, "end")
        claude_team_token_input.insert(0, claude_team)
    local_k = load_engine_apikey("local") or ""
    if local_k:
        local_apikey_input.delete(0, "end")
        local_apikey_input.insert(0, local_k)
    # Migrate from legacy single key (set as DeepL by default if empty)
    if not deepl:
        legacy = load_apikey()
        if legacy:
            try:
                legacy_str = legacy.decode('utf-16-le')
            except Exception:
                legacy_str = ""
            if legacy_str:
                deepl_key_input.delete(0, "end")
                deepl_key_input.insert(0, legacy_str)


save_button_frame = Frame(api_inner)
save_button_frame.grid(column=0, row=6, sticky='e', pady=(8, 0))
Button(save_button_frame, text=localization.getstr('save_apikeys'), command=save_all_apikeys).pack()

tree_frame = Frame(multifileframe)
tree_frame.grid(column=0, row=0, sticky='nsew', padx=8, pady=8)
# Make the tree row absorb extra vertical space when the window grows; the
# header (row 0) and the bottom button strip (row 2) stay at their natural
# height. Column 0 fills horizontally for the same reason.
tree_frame.grid_rowconfigure(1, weight=1)
tree_frame.grid_columnconfigure(0, weight=1)
multifileframe.grid_rowconfigure(0, weight=1)
multifileframe.grid_columnconfigure(0, weight=1)

multifile_header_frame = Frame(tree_frame)
multifile_header_frame.grid(column=0, row=0, sticky='ew', pady=(2, 4))
multifile_header_frame.grid_columnconfigure(0, weight=1)

file_treeview_instruction = Label(
    multifile_header_frame,
    text=localization.getstr("multifile_instruction"),
    wraplength=560,
    justify='left',
)
file_treeview_instruction.grid(column=0, row=0, sticky='w')

add_folder_button = Button(
    multifile_header_frame,
    text=localization.getstr('add_folder'),
    command=add_folder_to_multifile,
)
add_folder_button.grid(column=1, row=0, padx=(8, 4), sticky='e')

file_treeview = ttk.Treeview(
    tree_frame,
    columns=(localization.getstr("path"), localization.getstr("size"),
             localization.getstr("length"), localization.getstr("status")),
    height=22,  # was 5 — fills the larger default window proportionally
)
file_treeview.grid(column=0, row=1, sticky='nsew')

# 각 열의 설정
file_treeview.heading(localization.getstr("path"), text=localization.getstr("path"), command=lambda: sort_by_path(file_treeview, 0, False))
file_treeview.heading(localization.getstr("size"), text=localization.getstr("size") + "(MB)", command=lambda: treeview_sort_column(file_treeview, localization.getstr("size"), False))
file_treeview.heading(localization.getstr("length"), text=localization.getstr("length") + "(hh:mm:ss)", command=lambda: treeview_sort_column(file_treeview, localization.getstr("length"), False))
file_treeview.heading(localization.getstr("status"), text=localization.getstr("status"))

file_treeview.column("#0", width=0, stretch=tkinter.NO)
# Path column absorbs extra width when the window is wider than the default;
# the small fixed-width columns (size / length / status) stay readable.
file_treeview.column(localization.getstr("path"), anchor=tkinter.W, width=620, stretch=tkinter.YES)
file_treeview.column(localization.getstr("size"), anchor=tkinter.W, width=80, stretch=tkinter.NO)
file_treeview.column(localization.getstr("length"), anchor=tkinter.W, width=100, stretch=tkinter.NO)
file_treeview.column(localization.getstr("status"), anchor=tkinter.W, width=80, stretch=tkinter.NO)



# Bottom strip: progress bar, generate button, status indicators.
# Sits below the treeview, full-width so the contents can be centered.
generationframe = Frame(tree_frame)
generationframe.grid(column=0, row=2, sticky='ew', pady=(8, 4))
generationframe.grid_columnconfigure(0, weight=1)
generationframe.grid_columnconfigure(2, weight=1)  # right side flex for visual balance

file_treeview.bind('<Delete>', on_delete_key_press)

# Progress bar grows with the window for a clearer visual cue at all widths.
multifile_progressbar = ttk.Progressbar(generationframe, length=500, maximum=20)
multifile_progressbar.grid(column=1, row=0, padx=(0, 12), pady=6, sticky='ew')

multifile_generation_button = Button(generationframe, text=localization.getstr('generate'), command=proceedmultifile)
multifile_generation_button.grid(column=2, row=0, padx=(0, 0), sticky='w')

multifile_indicator_frame = Frame(generationframe)
multifile_indicator_frame.grid(column=1, row=1, sticky='w')

multifile_status_label = Label(multifile_indicator_frame, text="0%")
multifile_status_label.grid(column=0, row=0, padx=(0, 12))

multifile_list_label = Label(multifile_indicator_frame, text="0/0", anchor='w')
multifile_list_label.grid(column=1, row=0, sticky='w')

# Footer credit, centered across the bottom of the multifile tab.
Label(generationframe, text="JoyForce", fg="#888").grid(
    column=0, row=2, columnspan=3, pady=(6, 0))


initialize()



if __name__ == '__main__':
    window.mainloop()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
