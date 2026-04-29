# Joy4_sub

**Video Automatic Transcribed → translated Subtitle Generator**

A Windows desktop tool that transcribes video/audio files with Whisper and translates the subtitles into your target language. Supports four translation engines, batch processing, and recursive folder import.

> Renamed from VATSG (1.0.4). The repository now lives at [`Joy4orce/Joy4_sub`](https://github.com/Joy4orce/Joy4_sub).

---

## Features

- **Speech-to-text** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [openai-whisper](https://github.com/openai/whisper) (CUDA 12.1 supported)
- **Four translation engines**, switchable from the UI:
  - **DeepL** (API key)
  - **Claude Haiku** via Claude Code CLI (Pro/Team plan with automatic failover)
  - **Google Gemini** (API key, multi-key rotation on rate limit)
  - **OpenAI ChatGPT** (API key, multi-key rotation on rate limit)
- **Multi-file batch mode** with progress tracking
- **Recursive folder import**: select a folder, every supported media file inside (including sub-folders) is added automatically
- **Drag & drop** is fully retained
- **Rate-limit / safety-refusal handling** for Claude Haiku — automatic plan switching, batch-level fallback to original text on content-policy refusal
- **Korean / English** localization

Supported media: `.mp3 .wav .aac .m4a .flac .mp4 .mkv .mov .avi .webm .ogg .opus .wma .ts ...` (22 formats total)

---

## Setup (Windows)

### Prerequisites
- Windows 10 / 11
- Python 3.10 (the setup script will offer to install it via `winget` if missing)
- (Optional) NVIDIA GPU with CUDA 12.1 for fast Whisper transcription

### One-shot install

```cmd
Joy4_sub-Setup.bat
```

This creates `.venv-vatsg/`, installs all dependencies from `requirements-venv.txt`, and offers to launch the app on completion.

### Manual install

```bash
python -m venv .venv-vatsg
.venv-vatsg\Scripts\activate
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements-venv.txt
python Joy4_sub\Joy4_sub.py
```

---

## Usage

### Single file tab
1. Configure your API keys in the **API Settings** tab (DeepL / Gemini / ChatGPT / Claude Pro+Team OAuth tokens)
2. Drop a video/audio file or click **file open**
3. Pick a Whisper model and target language code
4. Click **generate** → an `.srt` file is written next to the source file

### Multifile tab
- Drop multiple files, **or** click **폴더 추가 / Add Folder** to recursively pull in every supported media file from a directory tree
- Press `Delete` to remove selected items
- Click **generate** to process the queue sequentially

### Claude Haiku translation (CLI mode)
This project uses the official Claude Code CLI for Claude translation, not the Anthropic SDK directly. Steps:

```bash
# In a normal PowerShell:
claude.exe login              # Authenticate the Pro account
claude.exe setup-token        # (Optional) get an OAuth token for Team plan
```

Paste the Pro / Team OAuth tokens into the **API Settings** tab. When one plan hits its rate limit, Joy4_sub switches to the other automatically.

### Language codes
- DeepL codes: <https://www.deepl.com/docs-api/translate-text>
- Reference files included in the repo: `srclangcode.txt`, `targetlangcode.txt`

---

## Building a distributable

```cmd
build.bat
```

Produces a stand-alone `dist/` folder via PyInstaller.

---

## Project layout

```
Joy4_sub-1.0/
├─ Joy4_sub/                 # Application source
│  ├─ Joy4_sub.py            # Entry point + Tk UI
│  ├─ UIwrapper.py           # State container passed to workers
│  ├─ mywhisper.py           # Whisper / translation pipeline
│  ├─ extractaudio.py        # Media probing (moviepy / mutagen)
│  ├─ claudewrapper.py       # Claude Code CLI wrapper (rate-limit + safety)
│  ├─ deeplwrapper.py        # DeepL API wrapper
│  ├─ geminiwrapper.py       # Google Gemini wrapper (multi-key)
│  ├─ openaiwrapper.py       # ChatGPT wrapper (multi-key)
│  ├─ apikeyrotator.py       # Multi-key rotation logic
│  ├─ settings.py            # Windows Credential Manager persistence
│  ├─ localization.py        # ko_KR / en strings
│  ├─ utility.py             # Treeview helpers, path shortening
│  └─ worker_subprocess.py   # Out-of-process worker for batch mode
├─ Asset/                    # Icons / images
├─ requirements-venv.txt     # Pinned runtime dependencies
├─ requirements.txt          # Alias of the above
├─ Joy4_sub-Setup.bat        # First-run installer
├─ Joy4_sub-Start.bat        # Launcher
└─ build.bat                 # PyInstaller build script
```

User settings and runtime logs live at:
```
%APPDATA%\VATSG\settings.json
%APPDATA%\VATSG\runtime.log
```

API tokens for Claude are stored encrypted in **Windows Credential Manager**.

---

## License

See `LICENSE.txt`. Third-party licenses bundled in `licenses.txt`.

---

## Contact

JoyForce — `powertjsl@gmail.com`
