"""Config accessor — reads from Anki's per-addon config (managed by Anki itself).

The aqt import is deferred to call time so this module can be imported
in test contexts where Anki isn't on the path. get_config() will fail
fast at call time if aqt isn't available.
"""

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
    from aqt import mw
    raw = mw.addonManager.getConfig(__name__.split(".")[0]) or {}
    merged = {**DEFAULTS, **raw}
    return merged
