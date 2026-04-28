import os
from datetime import datetime

from deepl import Translator
from deepl import exceptions

from settings import get_settings_path


def append_runtime_log(message):
    runtime_log_path = os.path.join(os.path.dirname(get_settings_path()), "runtime.log")
    os.makedirs(os.path.dirname(runtime_log_path), exist_ok=True)
    with open(runtime_log_path, "a", encoding="utf-8") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {message}\n")


def translateusingapi(text, uiwrapper):
    try:
        target_lang = uiwrapper.get_trglanguagecodeinput().strip()
        if hasattr(uiwrapper, 'get_deepl_key'):
            api_key = (uiwrapper.get_deepl_key() or "").strip()
        else:
            api_key = uiwrapper.getkey().strip()

        if not api_key:
            uiwrapper.update_percentagelabel_post("text", "check your apikey again")
            append_runtime_log("DeepL skipped because api key is empty")
            return None

        if not target_lang:
            uiwrapper.update_percentagelabel_post("text", "target language code is empty")
            append_runtime_log("DeepL skipped because target language code is empty")
            return None

        translator = Translator(api_key)
        result = translator.translate_text(text, target_lang=target_lang)

        if isinstance(text, list):
            append_runtime_log(f"DeepL translated batch of {len(text)} lines")
            return [item.text for item in result]

        append_runtime_log("DeepL translated single text payload")
        return result.text
    except Exception as e:
        append_runtime_log(f"DeepL translation failed: {type(e).__name__}: {e}")
        if isinstance(e, exceptions.QuotaExceededException):
            uiwrapper.update_percentagelabel_post("text", "api Exceed Quota")
        elif isinstance(e, exceptions.AuthorizationException):
            uiwrapper.update_percentagelabel_post("text", "check your apikey again")
        elif isinstance(e, exceptions.TooManyRequestsException):
            uiwrapper.update_percentagelabel_post("text", "deepl request is busy")
        else:
            uiwrapper.update_percentagelabel_post("text", f"DeepL error: {type(e).__name__}")
        return None


def translateusingapifortest(text):
    try:
        translator = Translator("fortest")
        result = translator.translate_text(text, target_lang="KO", source_lang="ZH")
        return result.text
    except Exception:
        return None
