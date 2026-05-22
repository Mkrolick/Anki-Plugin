"""
Front-end injection.

The card-template HTML pipe in Anki is just string concatenation, so we append
a `<div>` (the input UI + a hidden data island holding reference + keywords +
threshold) plus a `<script>` block. The JS listens for Enter, calls back into
Python via `pycmd("smartgrader:...")`, and renders the result on the same page.

We avoid touching the front template the user wrote — everything we add is
in its own container with a `sg-` class prefix so it can be styled or hidden
via custom CSS if anyone wants.
"""

from __future__ import annotations

import html
import json


def build_input_html(payload: dict) -> str:
    """payload = {note_id, reference, keywords, threshold}. We embed it as JSON."""
    blob = html.escape(json.dumps(payload), quote=True)
    return f"""
<div id="sg-root">
  <style>
    #sg-root {{ margin-top: 1.5em; font-family: inherit; }}
    #sg-input {{
      width: 100%; box-sizing: border-box; padding: 0.6em;
      font-size: 1em; min-height: 4em; resize: vertical;
      border: 1px solid #888; border-radius: 6px;
    }}
    #sg-submit {{
      margin-top: 0.5em; padding: 0.5em 1.2em; font-size: 1em;
      cursor: pointer; border-radius: 6px;
    }}
    #sg-result {{ margin-top: 1em; font-size: 0.95em; }}
    .sg-pass {{ color: #1f7a1f; font-weight: 600; }}
    .sg-fail {{ color: #b00020; font-weight: 600; }}
    .sg-eq {{ color: inherit; }}
    .sg-ins {{ color: #1f7a1f; }}
    .sg-del {{ color: #b00020; text-decoration: line-through; }}
    .sg-meta {{ color: #666; font-size: 0.85em; margin-top: 0.5em; }}
    .sg-chip {{
      display: inline-block; padding: 0.1em 0.5em; margin: 0 0.2em;
      border-radius: 10px; font-size: 0.8em;
    }}
    .sg-chip-found {{ background: #d4edda; color: #155724; }}
    .sg-chip-missing {{ background: #f8d7da; color: #721c24; }}
  </style>
  <textarea id="sg-input" placeholder="Type your answer, then press Ctrl/Cmd+Enter…"></textarea>
  <button id="sg-submit">Check answer</button>
  <div id="sg-result"></div>
  <script type="application/json" id="sg-payload">{blob}</script>
</div>
"""


# Kept separate from build_input_html so we can ensure it's only injected once
# even if Anki re-renders. The submit handler shells out to Python via pycmd.
INJECT_JS = """
<script>
(function() {
  if (window.__smartGraderInstalled) return;
  window.__smartGraderInstalled = true;

  function getPayload() {
    var el = document.getElementById("sg-payload");
    return el ? JSON.parse(el.textContent) : null;
  }

  function submit() {
    var payload = getPayload();
    if (!payload) return;
    var ta = document.getElementById("sg-input");
    if (!ta || !ta.value.trim()) return;
    var msg = Object.assign({}, payload, { user_answer: ta.value });
    pycmd("smartgrader:" + JSON.stringify(msg));
  }

  window.smartGraderShowResult = function(result) {
    var box = document.getElementById("sg-result");
    if (!box) return;
    var chips = (result.keyword_check.found || []).map(function(k) {
      return '<span class="sg-chip sg-chip-found">' + k + '</span>';
    }).concat((result.keyword_check.missing || []).map(function(k) {
      return '<span class="sg-chip sg-chip-missing">' + k + '</span>';
    })).join("");

    var header;
    if (result.pass) {
      header = '<div class="sg-pass">\u2713 Passed</div>';
    } else if (result.reason === "missing_keywords") {
      header = '<div class="sg-fail">\u2717 Missing required keyword(s)</div>';
    } else if (result.reason === "embedding_error") {
      header = '<div class="sg-fail">\u26a0 Embedding error: ' + (result.error || "unknown") + '</div>';
    } else {
      header = '<div class="sg-fail">\u2717 Too dissimilar from the reference</div>';
    }

    var sim = (result.similarity !== null && result.similarity !== undefined)
      ? "similarity " + result.similarity.toFixed(3) + " / threshold " + result.threshold.toFixed(3)
      : "similarity not computed (keyword check failed first)";

    box.innerHTML =
      header +
      '<div class="sg-meta">Keywords: ' + chips + '</div>' +
      '<div class="sg-meta">' + sim + '</div>' +
      '<div class="sg-meta" style="margin-top:0.6em">Diff vs reference:</div>' +
      '<div>' + result.diff_html + '</div>';
  };

  // Bind once the DOM is ready. Anki shows cards via webview reloads, so we
  // attach inside DOMContentLoaded or immediately if already past that.
  function bind() {
    var btn = document.getElementById("sg-submit");
    var ta = document.getElementById("sg-input");
    if (btn) btn.addEventListener("click", submit);
    if (ta) {
      ta.addEventListener("keydown", function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") submit();
      });
      ta.focus();
    }
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }
})();
</script>
"""
