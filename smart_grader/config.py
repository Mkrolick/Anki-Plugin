"""Config accessor — reads from Anki's per-addon config (managed by Anki itself)."""
from aqt import mw


DEFAULTS = {
    "openai_api_key": "",
    "embedding_model": "text-embedding-3-small",
    "chat_model": "gpt-4o-mini",
    "n_good_paraphrases": 12,
    "n_bad_paraphrases": 6,
    "fallback_threshold": 0.82,
    "epsilon_above_bad": 0.01,
}


def get_config() -> dict:
    raw = mw.addonManager.getConfig(__name__.split(".")[0]) or {}
    merged = {**DEFAULTS, **raw}
    return merged
