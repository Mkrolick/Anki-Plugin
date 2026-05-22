"""Config accessor — reads from Anki's per-addon config (managed by Anki itself)."""
from aqt import mw

DEFAULTS = {
    "openai_api_key": "",
    "chat_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "chunk_pages": 5,
    "chunk_overlap": 1,
    "cluster_distance": 0.25,
    "max_cards_per_topic": 5,
    "default_max_cost_usd": 5.0,
    "skip_per_card_calibration": False,
}


def get_config() -> dict:
    raw = mw.addonManager.getConfig(__name__.split(".")[0]) or {}
    return {**DEFAULTS, **raw}
