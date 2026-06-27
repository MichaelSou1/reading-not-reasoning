"""Prompt + code fingerprint for current numeric VLM evaluation runs."""
from __future__ import annotations

import hashlib

# Bump when runtime behavior changes in a way that should invalidate cached
# predictions even if prompt strings are unchanged.
AGENT_CODE_VERSION = "v22"


def prompt_fingerprint() -> str:
    """Short, stable hash of the current VLM QA prompt surface."""
    from app.vqa import QA_SYSTEM_PROMPT

    payload = "\x1e".join(
        [
            QA_SYSTEM_PROMPT,
            AGENT_CODE_VERSION,
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
