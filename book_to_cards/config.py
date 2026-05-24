"""Config accessor — reads from Anki's per-addon config (managed by Anki itself).

aqt is imported lazily so this module can be imported in headless tests.
"""

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
    # Number of cards calibrated in parallel during the precalibrate stage.
    # Each worker issues ~2 chat completions + 1 batched embedding call per
    # card; OpenAI's per-second rate limits are well above 8 concurrent
    # requests on typical accounts. Lower to 1 to force serial behaviour.
    "calibrate_concurrency": 8,
}


def get_config() -> dict:
    from aqt import mw
    raw = mw.addonManager.getConfig(__name__.split(".")[0]) or {}
    return {**DEFAULTS, **raw}
