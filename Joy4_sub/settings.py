import win32cred
import json
import shutil
import win32timezone
import platform
import os

APP_DIR_NAME = "Joy4_sub"
LEGACY_APP_DIR_NAME = "VATSG"

# API 키 저장

def save_apikey(apikey):
    cred_info = {
        'Type': win32cred.CRED_TYPE_GENERIC,
        'TargetName': APP_DIR_NAME,
        'CredentialBlob': apikey,
        'Persist': win32cred.CRED_PERSIST_LOCAL_MACHINE
    }
    win32cred.CredWrite(cred_info)
def load_apikey():
    try:
        cred_info = win32cred.CredRead(Type=win32cred.CRED_TYPE_GENERIC, TargetName=APP_DIR_NAME)
        api_key = cred_info['CredentialBlob']
        return api_key
    except Exception as e:
        return None


# Per-engine API key storage. Stored in Windows Credential Manager under
# distinct TargetNames so each engine's keys are kept separately.
_ENGINE_TARGET = {
    "deepl": "Joy4_sub_deepl_apikey",
    "gemini": "Joy4_sub_gemini_apikey",
    "openai": "Joy4_sub_openai_apikey",
    "claude_pro": "Joy4_sub_claude_pro_token",
    "claude_team": "Joy4_sub_claude_team_token",
    "local": "Joy4_sub_local_apikey",
}

# Mapping from current TargetName -> legacy TargetName (used for one-time migration).
_LEGACY_TARGET_MAP = {
    "Joy4_sub": "VATSG",
    "Joy4_sub_deepl_apikey": "VATSG_deepl_apikey",
    "Joy4_sub_gemini_apikey": "VATSG_gemini_apikey",
    "Joy4_sub_openai_apikey": "VATSG_openai_apikey",
    "Joy4_sub_claude_pro_token": "VATSG_claude_pro_token",
    "Joy4_sub_claude_team_token": "VATSG_claude_team_token",
}


def save_engine_apikey(engine, apikey):
    target = _ENGINE_TARGET.get(engine)
    if not target:
        return
    cred_info = {
        'Type': win32cred.CRED_TYPE_GENERIC,
        'TargetName': target,
        'CredentialBlob': apikey or "",
        'Persist': win32cred.CRED_PERSIST_LOCAL_MACHINE,
    }
    win32cred.CredWrite(cred_info)


def load_engine_apikey(engine):
    """Return the stored API key string for an engine, or empty string."""
    target = _ENGINE_TARGET.get(engine)
    if not target:
        return ""
    try:
        cred_info = win32cred.CredRead(Type=win32cred.CRED_TYPE_GENERIC, TargetName=target)
        blob = cred_info.get('CredentialBlob', b"")
        if isinstance(blob, bytes):
            try:
                return blob.decode('utf-16-le')
            except Exception:
                return blob.decode('utf-8', errors='replace')
        return str(blob)
    except Exception:
        return ""


def _decode_blob(blob):
    if isinstance(blob, bytes):
        try:
            return blob.decode('utf-16-le')
        except Exception:
            return blob.decode('utf-8', errors='replace')
    return str(blob)


def _credential_has_value(target_name):
    try:
        cred = win32cred.CredRead(Type=win32cred.CRED_TYPE_GENERIC, TargetName=target_name)
    except Exception:
        return False
    blob = cred.get('CredentialBlob', b"") if cred else b""
    return bool(_decode_blob(blob))


def migrate_legacy_credentials():
    """Copy any VATSG_* credential into its Joy4_sub_* counterpart on first run.

    The legacy entry is left in place so the user can roll back if needed.
    Skips entries whose new TargetName already has a non-empty value.
    """
    for new_name, old_name in _LEGACY_TARGET_MAP.items():
        if _credential_has_value(new_name):
            continue
        try:
            cred = win32cred.CredRead(Type=win32cred.CRED_TYPE_GENERIC, TargetName=old_name)
        except Exception:
            continue
        blob_str = _decode_blob(cred.get('CredentialBlob', b"") if cred else b"")
        if not blob_str:
            continue
        try:
            win32cred.CredWrite({
                'Type': win32cred.CRED_TYPE_GENERIC,
                'TargetName': new_name,
                'CredentialBlob': blob_str,
                'Persist': win32cred.CRED_PERSIST_LOCAL_MACHINE,
            })
        except Exception:
            pass


def _legacy_settings_dir():
    if platform.system() == 'Windows':
        return os.path.join(os.environ.get('APPDATA', ''), LEGACY_APP_DIR_NAME)
    if platform.system() == 'Darwin':
        return os.path.expanduser(f'~/Library/Application Support/{LEGACY_APP_DIR_NAME}')
    return os.path.expanduser(f'~/.config/{LEGACY_APP_DIR_NAME}')


def migrate_legacy_settings_dir():
    """Copy %APPDATA%\\VATSG\\* into %APPDATA%\\Joy4_sub\\* on first run.

    Only runs if the new directory is missing or empty. Original files are kept
    so the user can roll back. Best-effort — failures are silently ignored.
    """
    new_dir = os.path.dirname(get_settings_path())
    old_dir = _legacy_settings_dir()
    if not old_dir or not os.path.isdir(old_dir):
        return
    try:
        if os.path.isdir(new_dir) and any(os.scandir(new_dir)):
            return
    except Exception:
        pass
    try:
        os.makedirs(new_dir, exist_ok=True)
    except Exception:
        return
    try:
        for entry in os.listdir(old_dir):
            src = os.path.join(old_dir, entry)
            dst = os.path.join(new_dir, entry)
            if os.path.exists(dst):
                continue
            try:
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                elif os.path.isdir(src):
                    shutil.copytree(src, dst)
            except Exception:
                continue
    except Exception:
        return


def run_first_run_migrations():
    """Idempotent — copies legacy VATSG settings + credentials into the new namespace."""
    try:
        migrate_legacy_settings_dir()
    except Exception:
        pass
    try:
        migrate_legacy_credentials()
    except Exception:
        pass


def get_settings_path():
    if platform.system() == 'Windows':
        return os.path.join(os.environ['APPDATA'], APP_DIR_NAME, 'settings.json')
    elif platform.system() == 'Darwin':  # macOS
        return os.path.join(os.path.expanduser(f'~/Library/Application Support/{APP_DIR_NAME}'), 'settings.json')
    else:  # Linux and others
        return os.path.join(os.path.expanduser(f'~/.config/{APP_DIR_NAME}'), 'settings.json')

def settingjson(uiwrapper):
    settings_path = get_settings_path()

    settings_dir = os.path.dirname(settings_path)
    if not os.path.exists(settings_dir):
        os.makedirs(settings_dir)

    settings = {
        "cuda_var": uiwrapper.get_cuda_var(),
        "translateoption_var": uiwrapper.get_translateoption_Var(),
        "sourcelanguagecodeinput": uiwrapper.get_srclanguagecodeinput(),
        "targetlanguagecodeinput": uiwrapper.get_trglanguagecodeinput(),
        "original": uiwrapper.get_original_var(),
        "fast": uiwrapper.get_fast_var(),
        "translation_engine": uiwrapper.get_translation_engine(),
        "gemini_model": getattr(uiwrapper, 'gemini_model', '') or '',
        "openai_model": getattr(uiwrapper, 'openai_model', '') or '',
        "claude_default_plan": getattr(uiwrapper, 'claude_default_plan', 'pro') or 'pro',
        "local_endpoint": getattr(uiwrapper, 'local_endpoint', '') or '',
        "local_model": getattr(uiwrapper, 'local_model', '') or '',
        "local_system_prompt": getattr(uiwrapper, 'local_system_prompt', '') or '',
        "local_temperature": getattr(uiwrapper, 'local_temperature', 0.1),
    }
    with open(get_settings_path(), "w", encoding="utf-8") as f:
        json.dump(settings, f)


def load_settings():
    try:
        with open(get_settings_path(), "r", encoding="utf-8-sig") as f:
            settings = json.load(f)
            return settings
    except FileNotFoundError:
        return {}
