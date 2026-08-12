"""
GeminiKeyManager: holds a pool of Gemini API keys AND, per task, a fallback
chain of models.

- Keys are picked at random per call; a rate-limited key is retried on a
  different key, an invalid/denied key is dropped for the session.
- Models are looked up per task from MODEL_FALLBACKS. If the current model
  for a task 404s or stays rate-limited across all keys for two full
  rounds, the manager advances to the next model in that task's chain and
  persists the position to model_state.json.
- If every model in the chain is exhausted, raises AllModelsExhaustedError.
"""
import random
import time

from google import genai

from . import config
from .config import _load_model_state, _save_model_state


class AllModelsExhaustedError(Exception):
    """Raised when every model in a task's fallback chain has failed."""

    def __init__(self, task, chain):
        self.task = task
        self.chain = chain
        super().__init__(f"All models exhausted for task '{task}'. Tried: {chain}")


class GeminiKeyManager:
    def __init__(self, api_keys, model_fallbacks):
        real_keys = [k for k in api_keys if k]
        if not real_keys:
            raise ValueError("No Gemini API keys provided.")
        self.api_keys = real_keys
        self.disabled_keys = set()
        self._clients = {
            k: genai.Client(api_key=k, http_options={"timeout": 300_000})  # ms, i.e. 300s
            for k in real_keys
        }
        self.model_fallbacks = model_fallbacks
        self.state = _load_model_state()

    def current_model(self, task):
        chain = self.model_fallbacks[task]
        idx = self.state.get(task, {}).get("index", 0)
        idx = min(idx, len(chain) - 1)
        return chain[idx], idx

    def _advance_model(self, task):
        chain = self.model_fallbacks[task]
        _, idx = self.current_model(task)
        if idx + 1 >= len(chain):
            raise AllModelsExhaustedError(task, chain)
        new_idx = idx + 1
        self.state[task] = {"index": new_idx, "model": chain[new_idx]}
        _save_model_state(self.state)
        print(f"   🔁 switching task '{task}' -> model '{chain[new_idx]}' "
              f"(saved to model_state.json — future runs start here)")

    def _random_key(self, exclude):
        live = [k for k in self.api_keys if k not in self.disabled_keys]
        pool = [k for k in live if k not in exclude] or live or self.api_keys
        return random.choice(pool)

    @staticmethod
    def _classify(err):
        msg = str(err)
        if "404" in msg or "NOT_FOUND" in msg:
            return "not_found"
        if any(s in msg for s in ("403", "401", "PERMISSION_DENIED", "UNAUTHENTICATED", "denied access")):
            return "key_invalid"
        if any(s in msg.lower() for s in ("429", "resource_exhausted", "rate limit", "quota", "unavailable", "503")):
            return "rate_limit"
        return "other"

    def call(self, task, fn):
        while True:
            model, _ = self.current_model(task)
            tried = set()
            sleep_secs = config.RATE_LIMIT_SLEEP_SECS
            rounds_exhausted = 0
            attempt = 0

            while attempt < config.MAX_RETRIES_PER_KEY_ROUND:
                attempt += 1
                key = self._random_key(tried)
                tried.add(key)
                try:
                    return fn(self._clients[key], model)
                except Exception as err:
                    kind = self._classify(err)

                    if kind == "not_found":
                        print(f"   ✗ model '{model}' not available for your project (404).")
                        break

                    if kind == "key_invalid":
                        print(f"❌ key ...{key[-4:]} denied/invalid — disabling for this session")
                        self.disabled_keys.add(key)
                        if len(self.disabled_keys) >= len(self.api_keys):
                            raise RuntimeError(f"All API keys are invalid/denied: {err}") from err
                        attempt -= 1
                        continue

                    if kind == "rate_limit":
                        live_count = len(self.api_keys) - len(self.disabled_keys)
                        print(f"⚠️ rate limit on '{model}' / key ...{key[-4:]} "
                              f"(attempt {attempt}/{config.MAX_RETRIES_PER_KEY_ROUND})")
                        if len({k for k in tried if k not in self.disabled_keys}) >= max(live_count, 1):
                            rounds_exhausted += 1
                            tried = set()
                            if rounds_exhausted >= 2:
                                break
                            print(f"   all live keys rate-limited on '{model}' — sleeping {sleep_secs:.0f}s...")
                            time.sleep(sleep_secs)
                            sleep_secs *= config.RATE_LIMIT_BACKOFF
                        continue

                    if type(err).__name__ == "JSONRepairError":
                        # Model ignored the JSON constraint / got truncated.
                        # Worth a quick retry on a different key before we
                        # burn a model-rotation on what's often a one-off
                        # generation hiccup.
                        print(f"⚠️ Gemini returned unparsable JSON for task '{task}' — retrying...")
                    else:
                        print(f"⚠️ Gemini call failed ({type(err).__name__}: {err}) — retrying...")
                    time.sleep(2)
            else:
                # attempt budget exhausted without an explicit break -> try next model too
                pass

            self._advance_model(task)


class _LazyKeyManager:
    """
    Defers constructing the real GeminiKeyManager (which requires at least
    one real Gemini API key) until the RAG feature is actually used, instead
    of at import time. This lets the merged app start up fine even when
    GEMINI_API_KEY_1/2/3 aren't configured yet — ask-scholar's own routes
    don't need them, only /api/rag/* does.
    """

    def __init__(self):
        self._instance = None

    def _get(self):
        if self._instance is None:
            self._instance = GeminiKeyManager(config.GEMINI_API_KEYS, config.MODEL_FALLBACKS)
        return self._instance

    def call(self, task, fn):
        return self._get().call(task, fn)


key_manager = _LazyKeyManager()
