import locale
import tkinter
import tkinter.ttk
import uuid
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import tkinter.ttk as ttk

import threading
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
window.geometry('1060x560')
window.minsize(1060, 560)

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

# Per-engine model selection
gemini_model_var = tkinter.StringVar(value="gemini-2.0-flash")
openai_model_var = tkinter.StringVar(value="gpt-4o-mini")

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

file_list= []
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
    all_items = file_treeview.get_children()
    for item in all_items:
        item_value = list(file_treeview.item(item, 'values'))  # 튜플을 리스트로 변환
        item_spath = item_value[0]
        for path, spath in file_list:
            if spath == item_spath:
                file_name_without_extension, _ = os.path.splitext(path)
                if os.path.exists(file_name_without_extension + ".srt"):
                    item_value[3] = "Done"
                else:
                    item_value[3] = "Undone"
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
    release_retained_cuda_models()
    append_runtime_log(f"Finished single file job: {targetfile}")

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
        )
        save_all_apikeys()
        settingjson(transferuiwrapper)
        for i, (file, spath) in enumerate(file_list):
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
                    completed = subprocess.run(
                        worker_args,
                        cwd=os.path.dirname(__file__),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    append_runtime_log(f"Worker process exited with code {completed.returncode} for {file}")
                    if completed.stdout.strip():
                        append_runtime_log(f"Worker stdout for {file}: {completed.stdout.strip()}")
                    if completed.stderr.strip():
                        append_runtime_log(f"Worker stderr for {file}: {completed.stderr.strip()}")
                    if completed.returncode == 2:
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
                    elif completed.returncode != 0:
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
        release_retained_cuda_models()
        append_runtime_log("Released retained CUDA models after multifile batch")
    except Exception as exc:
        append_runtime_log(f"Multifile worker failed: {type(exc).__name__}: {exc}")
        multifile_queue.put(("text", f"Failed: {type(exc).__name__}"))
    finally:
        multifile_queue.put(("text", "Finished"))
        multi_processing = False


def multifile_update_from_queue():
    if not multifile_queue.empty():
        msg = multifile_queue.get()
        if msg[0] == "maximum":
            with lock:
                multifile_progressbar['maximum'] = msg[1]
        elif msg[0] == "value":
            with lock:
                multifile_progressbar['value'] = msg[1]
        elif msg[0] == "list":
            with lock:
                multifile_list_label['text'] = msg[1]
                check_multifile_status()
        elif msg[0] == "rate_limit":
            _show_rate_limit_dialog(msg[1])
        elif msg[0] == "text" and msg[1] != "Finished":
            with lock:
                multifile_status_label['text'] = msg[1]
        elif msg[1] == "Finished":
            with lock:
                all_done = multifile_list_label['text'].split("/")[0] == str(len(file_list))
                if all_done:
                    multifile_status_label['text'] = "Finished"
                multifile_generation_button.config(state=NORMAL)
                multifile_progressbar['value'] = 0
                check_multifile_status()
                file_treeview.bind('<Delete>', on_delete_key_press)
                return
    window.after(100, multifile_update_from_queue)



def update_ui_from_queue():
    if not update_queue.empty():
        msg = update_queue.get()
        if msg[0] == "maximum":
            with lock:
                progressbar['maximum'] = msg[1]
        elif msg[0] == "value":
            with lock:
                progressbar['value'] = msg[1]
        elif msg[0] == "rate_limit":
            _show_rate_limit_dialog(msg[1])
        elif msg[0] == "text" and msg[1] != "Finished":
            with lock:
                percentagelabel['text'] = msg[1]
        elif msg[1] == "Finished":
            with lock:
                percentagelabel['text'] = msg[1]
                proceedbutton.config(state=NORMAL)
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


def proceed():
    target_file = targetfileEntry.get().strip()
    if not target_file or not os.path.exists(target_file):
        messagebox.showerror("File not found", "처리할 미디어 파일을 선택해 주세요.")
        return
    if not is_supported_media(target_file):
        show_unsupported_media_message([target_file])
        return
    proceedbutton.config(state=DISABLED)
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
    multifile_generation_button.config(state=DISABLED)
    file_treeview.unbind('<Delete>')
    multifile_update_from_queue()
    thread = threading.Thread(target=proceed_multifile_whisperthread)
    thread.start()

#window.drop_target_register(DND_FILES)
#window.dnd_bind('<<Drop>>', on_drop)

notebook = ttk.Notebook(window)
notebook.pack()

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

targetfileEntry = Entry(frame1, width=30)
targetfileEntry.insert(0, localization.getstr('selectinstruction'))
targetfileEntry.grid(column=1, row=1)

button = Button(frame1, text=localization.getstr('selectfile'), command=open_dialog)
button.grid(column=0, row=1)

frame4 = Frame(frame1)
frame4.grid(column=0, row=2)

label = Label(frame4, text=localization.getstr('choosemodel'))
label.grid(column=0, row=0)

fastoption = Checkbutton(frame4, text="Fast", variable=fast_var)
fastoption.grid(column=1, row=0)

frame2 = Frame(frame1)
frame2.grid(column=1, row=2)

modeldropdown = ttk.Combobox(frame2, textvariable=translateoption_var, values=trnanslateoptions)
modeldropdown.grid(column=0, row=0)

checkbox = ttk.Checkbutton(frame2, text="Cuda", variable=cuda_var)
checkbox.grid(column=1, row=0)



sourcelanguagecodeinput = Entry(frame1, width=30)
sourcelanguagecodeinput.grid(column=1, row=3)


label = Label(frame1, text=localization.getstr('targetlangcode'))
label.grid(column=0, row=4)



targetlanguagecodeinput = Entry(frame1, width=30)
targetlanguagecodeinput.grid(column=1, row=4)

label = Label(frame1, text=localization.getstr('sourcelangcode'))
label.grid(column=0, row=3)


engine_frame = Frame(frame1)
engine_frame.grid(column=0, row=5, columnspan=2, sticky='w', padx=5)

Label(engine_frame, text=localization.getstr('translation_engine')).grid(column=0, row=0, padx=(0, 10))
Radiobutton(engine_frame, text="DeepL", variable=translation_engine_var, value="DeepL", command=lambda: on_engine_change()).grid(column=1, row=0)
Radiobutton(engine_frame, text="Claude Haiku", variable=translation_engine_var, value="Claude Haiku", command=lambda: on_engine_change()).grid(column=2, row=0)
Radiobutton(engine_frame, text="Gemini", variable=translation_engine_var, value="Gemini", command=lambda: on_engine_change()).grid(column=3, row=0)
Radiobutton(engine_frame, text="ChatGPT", variable=translation_engine_var, value="ChatGPT", command=lambda: on_engine_change()).grid(column=4, row=0)
Label(engine_frame, text=localization.getstr('apikey_multiline_hint'), fg="#666").grid(column=0, row=1, columnspan=5, sticky='w', pady=(2, 0))

frame3 = Frame(frame1)
frame3.grid(column=0, row=7)

proceedbutton = Button(frame3, text=localization.getstr('generate'), command=proceed)
proceedbutton.grid(column=0, row=0)

originalcheckbox = Checkbutton(frame3, text=localization.getstr('original'), variable=original_var)
originalcheckbox.grid(column=1, row=0)

progressbar = ttk.Progressbar(frame1, length=100, maximum=20)
progressbar.grid(column=1, row=7)

percentagelabel = Label(frame1, text="0%")
percentagelabel.grid(column=1, row=8)

Label(frame1, text="JoyForce").grid(column=0, row=9, columnspan=2)


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

api_inner.columnconfigure(0, weight=1)


def save_all_apikeys():
    """Persist API keys for all engines to Windows Credential Manager."""
    save_engine_apikey("deepl", get_deepl_key_input())
    save_engine_apikey("gemini", get_gemini_keys_input())
    save_engine_apikey("openai", get_openai_keys_input())
    save_engine_apikey("claude_pro", get_claude_pro_token_input())
    save_engine_apikey("claude_team", get_claude_team_token_input())


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
save_button_frame.grid(column=0, row=5, sticky='e', pady=(8, 0))
Button(save_button_frame, text=localization.getstr('save_apikeys'), command=save_all_apikeys).pack()

tree_frame = Frame(multifileframe, width=400, height=20)
tree_frame.grid(column=0, row=0, sticky='nsew')
tree_frame.grid_rowconfigure(1, weight=0)

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

file_treeview =ttk.Treeview(tree_frame, columns=( localization.getstr("path"), localization.getstr("size"), localization.getstr("length"),localization.getstr("status")), height=5)
file_treeview.grid(column=0, row=1, sticky='nsew')

# 각 열의 설정
file_treeview.heading(localization.getstr("path"), text=localization.getstr("path"), command=lambda: sort_by_path(file_treeview, 0, False))
file_treeview.heading(localization.getstr("size"), text=localization.getstr("size") + "(MB)", command=lambda: treeview_sort_column(file_treeview, localization.getstr("size"), False))
file_treeview.heading(localization.getstr("length"), text=localization.getstr("length") + "(hh:mm:ss)", command=lambda: treeview_sort_column(file_treeview, localization.getstr("length"), False))
file_treeview.heading(localization.getstr("status"), text=localization.getstr("status"))

file_treeview.column("#0", width=0, stretch=tkinter.NO)
file_treeview.column(localization.getstr("path"), anchor=tkinter.W, width=400)
file_treeview.column(localization.getstr("size"), anchor=tkinter.W, width=70)
file_treeview.column(localization.getstr("length"), anchor=tkinter.W, width=90)
file_treeview.column(localization.getstr("status"), anchor=tkinter.W, width=70)



# 임시 데이터 삽입
generationframe = Frame(tree_frame, width=460)
generationframe.grid(column=0, row=2)
generationframe.grid_columnconfigure(1, weight=0)

file_treeview.bind('<Delete>', on_delete_key_press)

multifile_progressbar = ttk.Progressbar(generationframe, length=400, maximum=20)
multifile_progressbar.grid(column=0, row=0, padx=(60, 90), pady=10, sticky='e')

multifile_generation_button = Button(generationframe, text=localization.getstr('generate'), command=proceedmultifile)
multifile_generation_button.grid(column=1, row=0,padx=(0, 80), sticky='w')

multifile_indicator_frame = Frame(generationframe)
multifile_indicator_frame.grid(column=0, row=1)

multifile_list_label = Label(multifile_indicator_frame, text="0/0", anchor='w')
multifile_list_label.grid(column=1, row=0, sticky='w')

multifile_status_label = Label(multifile_indicator_frame, text = "0%")
multifile_status_label.grid(column=0, row=0)


Label(multifile_indicator_frame, text="JoyForce").grid(column=0, row=3, columnspan=2)


initialize()



if __name__ == '__main__':
    window.mainloop()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
