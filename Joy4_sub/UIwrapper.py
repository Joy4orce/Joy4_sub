class UIwrapper:
    def __init__(self, queue, lock, key, cuda_var, translateoption_var,
                 srclanguagecodeinput, targetlanguagecodinput, original_var,
                 fast_option, translation_engine="DeepL", rate_limit_wait_fn=None,
                 deepl_key="", gemini_keys="", openai_keys="",
                 gemini_model="", openai_model="",
                 claude_pro_token="", claude_team_token="", claude_default_plan="pro",
                 local_endpoint="", local_model="", local_system_prompt="",
                 local_temperature=0.1, local_apikey="",
                 cancel_event=None):
        self.queue = queue
        self.lock = lock
        self.key = key  # legacy: active engine's keys (kept for compat)
        self.cuda_var = cuda_var
        self.translateoption_var = translateoption_var
        self.srclanguagecodeinput = srclanguagecodeinput
        self.targetlanguagecodinput = targetlanguagecodinput
        self.original_var = original_var
        self.fast_option = fast_option
        self.translation_engine = translation_engine
        self.rate_limit_wait_fn = rate_limit_wait_fn
        self.deepl_key = deepl_key
        self.gemini_keys = gemini_keys
        self.openai_keys = openai_keys
        self.gemini_model = gemini_model
        self.openai_model = openai_model
        self.claude_pro_token = claude_pro_token
        self.claude_team_token = claude_team_token
        self.claude_default_plan = (claude_default_plan or "pro").lower()
        self.local_endpoint = local_endpoint
        self.local_model = local_model
        self.local_system_prompt = local_system_prompt
        self.local_temperature = local_temperature
        self.local_apikey = local_apikey
        # threading.Event shared across the worker thread and the wrappers
        # so any cooperative checkpoint (batch boundary, retry attempt,
        # per-line call) can bail out cleanly when the user hits Cancel.
        self.cancel_event = cancel_event

    def is_cancelled(self):
        """True when the user has requested cancellation. Cheap to call —
        callers should poll this at any natural stop point."""
        return self.cancel_event is not None and self.cancel_event.is_set()

    def update_percentagelabel_post(self, text, value):
        with self.lock:
            self.queue.put((text, value))

    def update_progressbar(self, text, value):
        with self.lock:
            self.queue.put((text, value))

    def getkey(self):
        """Return the API key string for the currently active engine."""
        engine = self.translation_engine
        if engine == "DeepL":
            return self.deepl_key
        if engine == "Gemini":
            return self.gemini_keys
        if engine == "ChatGPT":
            return self.openai_keys
        if engine == "Local LLM":
            return self.local_apikey
        return self.key  # fallback for legacy / unknown engines

    def get_progressbar(self):
        return self.progressbar

    def get_percentagelabel(self):
        return self.percentagelabel

    def get_cuda_var(self):
        return self.cuda_var

    def get_translateoption_Var(self):
        return self.translateoption_var

    def get_srclanguagecodeinput(self):
        return self.srclanguagecodeinput

    def get_trglanguagecodeinput(self):
        return self.targetlanguagecodinput

    def get_original_var(self):
        return self.original_var

    def get_fast_var(self):
        return self.fast_option

    def get_translation_engine(self):
        return self.translation_engine

    def get_deepl_key(self):
        return self.deepl_key

    def get_gemini_keys(self):
        return self.gemini_keys

    def get_openai_keys(self):
        return self.openai_keys

    def get_gemini_model(self):
        return self.gemini_model

    def get_openai_model(self):
        return self.openai_model

    def get_claude_pro_token(self):
        return self.claude_pro_token

    def get_claude_team_token(self):
        return self.claude_team_token

    def get_claude_default_plan(self):
        return self.claude_default_plan

    def get_local_endpoint(self):
        return self.local_endpoint

    def get_local_model(self):
        return self.local_model

    def get_local_system_prompt(self):
        return self.local_system_prompt

    def get_local_temperature(self):
        return self.local_temperature

    def get_local_apikey(self):
        return self.local_apikey

    def wait_for_rate_limit_decision(self, file_info=""):
        if self.rate_limit_wait_fn is None:
            return False
        return self.rate_limit_wait_fn(file_info)
