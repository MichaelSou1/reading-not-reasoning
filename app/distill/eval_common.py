"""Shared evaluation vocabulary + a TEXT-AWARE grader (Spec §0, §0.5).

The headline numbers are at the ±2% scale, while letter-only MCQ grading lets
"letter-luck" cases (the answer explicitly cites option X's letter but its prose
describes a different option) score as correct. This module provides a grader that
demotes those cases — conservatively: a case is flipped to incorrect ONLY when the
cited letter matches gold yet the answer's content clearly endorses a *different*
option. The small MCQ parser lives here so the current numeric-eval path no longer
depends on the removed video-agent harness modules.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from app.eval_fingerprint import AGENT_CODE_VERSION, prompt_fingerprint

MCQ_OPTION_RE = re.compile(
    r"(?ms)^\s*([A-E])[\).]\s+(.+?)(?=^\s*[A-E][\).]\s+|\Z)"
)

# Spec §0: the five comparable methods.
METHODS = (
    "free_form",
    "self_reflect",
    "orch_reflect_blind",
    "orch_reflect_sighted",
    "agent_retrieval",
)
DEFAULT_N_FRAMES = 16


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
def parse_candidates(question: str) -> list[dict[str, str]]:
    """Parse Candidates blocks formatted as A) option text."""
    if "Candidates:" not in (question or ""):
        return []
    block = (question or "").split("Candidates:", 1)[1]
    block = re.split(
        r"\n\s*\n\s*(?:Answer the question|Reference key frames|请回答|回答问题)",
        block,
        maxsplit=1,
    )[0]
    out: list[dict[str, str]] = []
    for match in MCQ_OPTION_RE.finditer(block):
        text = _clean_option_text(match.group(2))
        if text:
            out.append({"label": match.group(1).upper(), "text": text})
    return out


def selected_candidate(answer: str, candidates: list[dict[str, str]]) -> dict[str, str] | None:
    """Return the MCQ candidate selected in an answer, if one is explicit."""
    if not candidates:
        return None
    labels = {item["label"].upper(): item for item in candidates}
    match = re.search(
        r"\b(?:answer|correct answer|选项|答案)\s*[:：]?\s*\**([A-E])\**\s*[\).]",
        answer or "",
        re.I,
    )
    if not match:
        match = re.search(r"\b([A-E])\s*[\).]\s+", answer or "")
    if match:
        return labels.get(match.group(1).upper())
    normalized_answer = normalize_text(answer)
    for item in candidates:
        option = normalize_text(item["text"])
        if option and option in normalized_answer:
            return item
    return None


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(value: Any) -> list[str]:
    normalized = normalize_text(value)
    return [
        token
        for token in normalized.split()
        if len(token) >= 3 and token not in {"the", "and", "for", "with", "from", "that", "this"}
    ]


def text_contains_option(text: str, option_text: str) -> bool:
    normalized_text = normalize_text(text)
    tokens = content_tokens(option_text)
    if not tokens:
        return False
    return all(token in normalized_text for token in tokens[:6])


def _clean_option_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().rstrip()


def _answer_matches(question: str, reference: str, answer: str) -> bool:
    candidates = parse_candidates(question)
    if candidates:
        selected = selected_candidate(answer, candidates)
        expected = selected_candidate(reference, candidates)
        if expected is None:
            expected_text = reference.strip().lower().rstrip(".")
            for candidate in candidates:
                if expected_text and expected_text in candidate["text"].lower().rstrip("."):
                    expected = candidate
                    break
        return bool(selected and expected and selected["label"] == expected["label"])
    return relaxed_match(answer, reference)


def _gold_candidate(gold: str, cands: list[dict[str, str]]) -> dict[str, str] | None:
    """Map a gold answer string to its candidate (bare letter → letter → text)."""
    g = str(gold).strip()
    # bare single letter, e.g. "B" or "B." or "(B)"
    bare = re.fullmatch(r"\(?([A-Ea-e])\)?\.?", g)
    if bare:
        lab = bare.group(1).upper()
        for cand in cands:
            if cand["label"].upper() == lab:
                return cand
    c = selected_candidate(gold, cands)
    if c is not None:
        return c
    gnorm = normalize_text(gold).rstrip(".")
    for cand in cands:
        cn = normalize_text(cand["text"]).rstrip(".")
        if gnorm and (gnorm in cn or cn in gnorm):
            return cand
    return None


def relaxed_match(pred: str, gold: str) -> bool:
    """Open-ended (e.g. ChartQA): 5% tolerance for numbers, normalized substring for text."""
    g = str(gold).strip()
    nums = re.findall(r"-?\d+\.?\d*", str(pred).replace(",", ""))
    try:
        gv = float(g.replace(",", "").replace("%", ""))
        return any(abs(float(p) - gv) <= abs(gv) * 0.05 + 1e-6 for p in nums)
    except ValueError:
        gn = normalize_text(g)
        return bool(gn) and gn in normalize_text(pred)


def grade_textaware(question: str, gold: str, answer: str) -> dict[str, Any]:
    """Returns {correct, letter_luck, mcq, ...}. Text-aware MCQ grading (Spec §0.5.1)."""
    cands = parse_candidates(question)
    if not cands:
        return {"correct": relaxed_match(answer, gold), "letter_luck": False, "mcq": False}

    gold_c = _gold_candidate(gold, cands)
    if gold_c is None:
        return {"correct": _answer_matches(question, gold, answer), "letter_luck": False,
                "mcq": True, "gold_resolved": False}
    sel = selected_candidate(answer, cands)
    letter_ok = bool(sel and sel["label"] == gold_c["label"])

    gold_text_present = text_contains_option(answer, gold_c["text"])
    other_text_present = any(
        c["label"] != gold_c["label"] and text_contains_option(answer, c["text"])
        for c in cands
    )
    # letter says gold, but the prose endorses a different option only.
    letter_luck = bool(letter_ok and not gold_text_present and other_text_present)
    correct = bool(letter_ok and not letter_luck)
    return {
        "correct": correct,
        "letter_luck": letter_luck,
        "letter_ok": bool(letter_ok),
        "gold_text_present": bool(gold_text_present),
        "mcq": True,
        "gold_resolved": True,
    }


# ---------------------------------------------------------------------------
# Per-run config fingerprint (Spec §0 determinism rule)
# ---------------------------------------------------------------------------
def config_fingerprint(
    *,
    dataset: str,
    split_hash: str,
    model_id: str,
    method: str,
    n_frames: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """A reproducible fingerprint dict for the result store. `fp` is a short hash of all of it."""
    payload = {
        "dataset": dataset,
        "split_hash": split_hash,
        "model_id": model_id,
        "method": method,
        "n_frames": n_frames,
        "decode": {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens},
        "seed": seed,
        "prompt_fp": prompt_fingerprint(),
        "code_version": AGENT_CODE_VERSION,
    }
    blob = repr(sorted(payload.items())).encode("utf-8")
    payload["fp"] = hashlib.sha1(blob).hexdigest()[:12]
    return payload


def case_set_hash(case_ids: list[str]) -> str:
    """Stable hash of a case-id set (for split_hash)."""
    blob = "\n".join(sorted(case_ids)).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]
