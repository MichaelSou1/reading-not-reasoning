#!/usr/bin/env python
"""P0-2 faithfulness-probe statistics.

This is CPU-only and derives all numbers from the stored probe/eval artifacts.
It adds uncertainty and tests around the headline chart/table faithfulness cells:

* F = flip_corrupt - flip_shuffle, with a 95% CI.
* snap/follow Wilson 95% CIs.
* an exact one-sided load-bearing test for corrupt > shuffle, Holm-corrected
  across the confirmatory chart/table family.
* SFT-gain subset summaries for ChartQA and TabMWP, plus a FinQA applicability
  note because the targeted FinQA batteries are not stored as matched
  base-vs-SFT gain-subset evals.
* P0-5 cross-family replication on a non-Qwen VLM, reported separately from
  the confirmatory Qwen-family chart/table SFT cells.

Where a battery JSON stores per-variant answers under details[].answers, the
corrupt-vs-shuffle test is paired exact sign randomization and the F CI is a
paired bootstrap over cases. Legacy headline JSONs predate that answer logging;
for those cells the script uses the aggregate margins recorded in the result
store: an exact hypergeometric randomization test and independent binomial
bootstrap CI.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.distill.eval_common import relaxed_match

POC = ROOT / "data/distill/poc"
RES = ROOT / "data/distill/results"


@dataclass(frozen=True)
class CellSpec:
    dataset: str
    model_id: str
    condition: str
    path: Path
    display: str
    preds: Path | None = None


PRED_PATHS = {
    ("chartqa", "8b"): POC / "lora_8b_chartqa/eval_n400.lora_8b_chartqa.preds.jsonl",
    ("chartqa", "32b"): POC / "lora_32b_chartqa/eval_n400.epoch_1.preds.jsonl",
    ("tabmwp", "8b"): POC / "eval_sft_8b_tabmwp_n400.lora_8b_tabmwp.preds.jsonl",
    ("tabmwp", "32b"): POC / "eval_sft_32b_tabmwp_n400.epoch_4.preds.jsonl",
}


HEADLINE_CELLS = [
    CellSpec("chartqa", "8b", "present", POC / "battery_8b_present.json", "ChartQA 8B", PRED_PATHS[("chartqa", "8b")]),
    CellSpec("chartqa", "8b", "masked", POC / "battery_8b_masked.json", "ChartQA 8B", PRED_PATHS[("chartqa", "8b")]),
    CellSpec("chartqa", "32b", "present", POC / "battery_32b_present.json", "ChartQA 32B", PRED_PATHS[("chartqa", "32b")]),
    CellSpec("chartqa", "32b", "masked", POC / "battery_32b_masked.json", "ChartQA 32B", PRED_PATHS[("chartqa", "32b")]),
    CellSpec("tabmwp", "8b", "present", POC / "battery_tabmwp8b_present.json", "TabMWP 8B", PRED_PATHS[("tabmwp", "8b")]),
    CellSpec("tabmwp", "8b", "masked", POC / "battery_tabmwp8b_masked.json", "TabMWP 8B", PRED_PATHS[("tabmwp", "8b")]),
    CellSpec("tabmwp", "32b", "present", POC / "battery_tabmwp32b_present.json", "TabMWP 32B", PRED_PATHS[("tabmwp", "32b")]),
    CellSpec("tabmwp", "32b", "masked", POC / "battery_tabmwp32b_masked.json", "TabMWP 32B", PRED_PATHS[("tabmwp", "32b")]),
]

GAIN_SPECS = [
    {
        "label": "ChartQA 8B",
        "dataset": "chartqa",
        "model_id": "8b",
        "battery": POC / "battery_8b_present.json",
        "preds": POC / "lora_8b_chartqa/eval_n400.lora_8b_chartqa.preds.jsonl",
    },
    {
        "label": "ChartQA 32B",
        "dataset": "chartqa",
        "model_id": "32b",
        "battery": POC / "battery_32b_present.json",
        "preds": POC / "lora_32b_chartqa/eval_n400.epoch_1.preds.jsonl",
    },
    {
        "label": "TabMWP 8B",
        "dataset": "tabmwp",
        "model_id": "8b",
        "battery": POC / "battery_tabmwp8b_present.json",
        "preds": POC / "eval_sft_8b_tabmwp_n400.lora_8b_tabmwp.preds.jsonl",
    },
    {
        "label": "TabMWP 32B",
        "dataset": "tabmwp",
        "model_id": "32b",
        "battery": POC / "battery_tabmwp32b_present.json",
        "preds": POC / "eval_sft_32b_tabmwp_n400.epoch_4.preds.jsonl",
    },
]

FINQA_TARGETED = [
    {
        "label": "FinQA 8B curriculum",
        "path": POC / "battery_n1_8b.json",
        "result_key": "b2",
    },
    {
        "label": "FinQA 32B curriculum",
        "path": POC / "battery_n1_32b.json",
        "result_key": "b2",
    },
]

LOCAL_CONTROL_SPECS = [
    CellSpec(
        "chartqa",
        "8b",
        "present",
        POC / "battery_p0_local_chartqa8b_present.json",
        "ChartQA 8B",
        PRED_PATHS[("chartqa", "8b")],
    ),
    CellSpec(
        "chartqa",
        "32b",
        "present",
        POC / "battery_p0_local_chartqa32b_present.json",
        "ChartQA 32B",
        PRED_PATHS[("chartqa", "32b")],
    ),
    CellSpec(
        "tabmwp",
        "8b",
        "present",
        POC / "battery_p0_local_tabmwp8b_present.json",
        "TabMWP 8B",
        PRED_PATHS[("tabmwp", "8b")],
    ),
    CellSpec(
        "tabmwp",
        "32b",
        "present",
        POC / "battery_p0_local_tabmwp32b_present.json",
        "TabMWP 32B",
        PRED_PATHS[("tabmwp", "32b")],
    ),
]

SEMANTIC_CONTROL_SPECS = [
    CellSpec(
        "chartqa",
        "8b",
        "present",
        POC / "battery_p0_semantic_chartqa8b_present.json",
        "ChartQA 8B",
        PRED_PATHS[("chartqa", "8b")],
    ),
    CellSpec(
        "chartqa",
        "32b",
        "present",
        POC / "battery_p0_semantic_chartqa32b_present.json",
        "ChartQA 32B",
        PRED_PATHS[("chartqa", "32b")],
    ),
    CellSpec(
        "tabmwp",
        "8b",
        "present",
        POC / "battery_p0_semantic_tabmwp8b_present.json",
        "TabMWP 8B",
        PRED_PATHS[("tabmwp", "8b")],
    ),
    CellSpec(
        "tabmwp",
        "32b",
        "present",
        POC / "battery_p0_semantic_tabmwp32b_present.json",
        "TabMWP 32B",
        PRED_PATHS[("tabmwp", "32b")],
    ),
]

CROSS_FAMILY_SPECS = [
    CellSpec(
        "chartqa",
        "internvl35_8b",
        "present",
        POC / "battery_p0_cross_family_internvl35_chartqa_present.json",
        "ChartQA InternVL3.5-8B",
        None,
    ),
]

# P3-1 generation-time (single-stream) intervention: cross-paradigm comparison on the
# ChartQA 8B SFT student. Both paradigms run on the SAME retrained student so the only
# thing that changes is prompt-level (two-pass force-continue) vs stream-level (in-place).
ONLINE_P3_PATHS = {
    "present": POC / "battery_p3_online_chartqa8b_present.json",
    "maskedB": POC / "battery_p3_online_chartqa8b_maskedB.json",
    "twopass": POC / "battery_p3_twopass_chartqa8b_present.json",
}


def stable_seed(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [max(0.0, (c - h) / d), min(1.0, (c + h) / d)]


def logcomb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_greater(a: int, n1: int, b: int, n2: int) -> float:
    """Exact hypergeometric randomization p-value for p1 > p2."""
    total = a + b
    lo = max(0, total - n2)
    hi = min(n1, total)
    if not (lo <= a <= hi):
        return 1.0
    denom = logcomb(n1 + n2, total)
    return float(min(1.0, sum(
        math.exp(logcomb(n1, x) + logcomb(n2, total - x) - denom)
        for x in range(a, hi + 1)
    )))


def binom_tail_greater(k: int, n: int) -> float:
    """Exact sign/randomization p-value for k positive signs out of n."""
    if n <= 0:
        return 1.0
    return float(min(1.0, sum(math.comb(n, x) for x in range(k, n + 1)) / (2 ** n)))


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    p = list(p_values)
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        raw = min(1.0, (m - rank + 1) * p[idx])
        running = max(running, raw)
        adjusted[idx] = min(1.0, running)
    return adjusted


def bootstrap_independent(
    c_flips: int,
    c_n: int,
    s_flips: int,
    s_n: int,
    *,
    seed: int,
    B: int = 50000,
) -> list[float]:
    if c_n <= 0 or s_n <= 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    pc = c_flips / c_n
    ps = s_flips / s_n
    boot = rng.binomial(c_n, pc, size=B) / c_n - rng.binomial(s_n, ps, size=B) / s_n
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_paired(diffs: list[int], *, seed: int, B: int = 50000) -> list[float]:
    if not diffs:
        return [0.0, 0.0]
    d = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return [float(lo), float(hi)]


def rate_block(k: int, n: int) -> dict[str, Any]:
    return {"k": int(k), "n": int(n), "rate": (k / n if n else 0.0), "ci95": wilson(k, n)}


def pct(k: int, n: int) -> float:
    return k / n if n else 0.0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def answer_for(row: dict[str, Any], key: str) -> Any:
    answers = row.get("answers") or {}
    if key in answers:
        return answers[key]
    if key == "corrupt":
        return row.get("corrupt_ans")
    return None


def classify_corrupt(details: list[dict[str, Any]]) -> dict[str, Any]:
    n = snap = follow = other = flip = 0
    for row in details:
        ans = answer_for(row, "corrupt")
        injected = row.get("injected")
        if ans is None or injected is None:
            continue
        n += 1
        if not relaxed_match(ans, row.get("base_ans", "")):
            flip += 1
        if relaxed_match(ans, row.get("gold", "")):
            snap += 1
        elif injected and relaxed_match(ans, injected):
            follow += 1
        else:
            other += 1
    return {
        "n": n,
        "flip": rate_block(flip, n),
        "snap": rate_block(snap, n),
        "follow": rate_block(follow, n),
        "other": rate_block(other, n),
    }


def paired_flips(details: list[dict[str, Any]], control_key: str = "shuffle") -> list[tuple[int, int]]:
    rows = []
    for row in details:
        corrupt_ans = answer_for(row, "corrupt")
        control_ans = answer_for(row, control_key)
        if corrupt_ans is None or control_ans is None:
            continue
        base = row.get("base_ans", "")
        c_flip = int(not relaxed_match(corrupt_ans, base))
        s_flip = int(not relaxed_match(control_ans, base))
        rows.append((c_flip, s_flip))
    return rows


def load_pred_rows(preds_path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in preds_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["cid"]] = row
    return rows


def answer_eval_partition(row: dict[str, Any]) -> str:
    base_ok = bool(row["base_correct"])
    sft_ok = bool(row["correct"])
    if base_ok and sft_ok:
        return "stable_correct"
    if (not base_ok) and sft_ok:
        return "sft_gain"
    if base_ok and (not sft_ok):
        return "sft_loss"
    return "both_wrong"


def empty_stratum() -> dict[str, Any]:
    return {
        "n": 0,
        "corrupt_flip": rate_block(0, 0),
        "snap": rate_block(0, 0),
        "follow": rate_block(0, 0),
        "other": rate_block(0, 0),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    corrupt_flip = sum(bool(r["corrupt_flip"]) for r in rows)
    snap = sum(bool(r["snap"]) for r in rows)
    follow = sum(bool(r["follow"]) for r in rows)
    other = sum(bool(r["other"]) for r in rows)
    return {
        "n": n,
        "corrupt_flip": rate_block(corrupt_flip, n),
        "snap": rate_block(snap, n),
        "follow": rate_block(follow, n),
        "other": rate_block(other, n),
    }


def detail_readout(row: dict[str, Any]) -> dict[str, bool]:
    ans = answer_for(row, "corrupt")
    injected = row.get("injected")
    if ans is None or injected is None:
        return {"usable": False, "corrupt_flip": False, "snap": False, "follow": False, "other": False}
    snap = relaxed_match(ans, row.get("gold", ""))
    follow = bool(injected) and (not snap) and relaxed_match(ans, injected)
    other = not snap and not follow
    return {
        "usable": True,
        "corrupt_flip": not relaxed_match(ans, row.get("base_ans", "")),
        "snap": snap,
        "follow": follow,
        "other": other,
    }


def selection_strata(details: list[dict[str, Any]], preds_path: Path | None) -> dict[str, Any] | None:
    if preds_path is None or not preds_path.exists():
        return None
    pred_rows = load_pred_rows(preds_path)
    eval_counts = {name: 0 for name in ["stable_correct", "sft_gain", "sft_loss", "both_wrong"]}
    for row in pred_rows.values():
        eval_counts[answer_eval_partition(row)] += 1

    kept_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ["stable_correct", "sft_gain", "sft_loss", "both_wrong", "unmatched_probe"]
    }
    inconsistent_eval_correct = 0
    corrupt_usable = 0
    for d in details:
        readout = detail_readout(d)
        if readout["usable"]:
            corrupt_usable += 1
        pred = pred_rows.get(d["case_id"])
        key = answer_eval_partition(pred) if pred else "unmatched_probe"
        if pred and not bool(pred["correct"]):
            inconsistent_eval_correct += 1
        kept_rows[key].append(readout)

    kept_counts = {name: len(rows) for name, rows in kept_rows.items()}
    strata = {name: summarize_rows(rows) for name, rows in kept_rows.items() if rows}

    n_raw = len(pred_rows)
    n_sft_correct_eval = eval_counts["stable_correct"] + eval_counts["sft_gain"]
    n_probe = len(details)
    n_probe_not_sft_eval_correct = n_probe - kept_counts["stable_correct"] - kept_counts["sft_gain"]
    n_eval_correct_missing_probe = max(0, n_sft_correct_eval - kept_counts["stable_correct"] - kept_counts["sft_gain"])

    observed_follow = sum(
        block["follow"]["k"] for name, block in strata.items() if name != "unmatched_probe"
    )
    observed_n = sum(block["n"] for name, block in strata.items() if name != "unmatched_probe")
    sft_correct_kept_n = kept_counts["stable_correct"] + kept_counts["sft_gain"]
    sft_correct_kept_follow = (
        strata.get("stable_correct", empty_stratum())["follow"]["k"]
        + strata.get("sft_gain", empty_stratum())["follow"]["k"]
    )
    sft_correct_corrupt_unusable = sum(
        1
        for name in ("stable_correct", "sft_gain")
        for row in kept_rows[name]
        if not row["usable"]
    )

    unknown_nonprobe = n_raw - n_probe
    failed_or_unknown = n_raw - corrupt_usable
    sft_correct_failed_or_unknown = n_eval_correct_missing_probe + sft_correct_corrupt_unusable
    sensitivity = {
        "observed_probe_follow": rate_block(observed_follow, corrupt_usable),
        "all_raw_lower_follow_failed_nonfollow": {
            "k": observed_follow,
            "n": n_raw,
            "rate": pct(observed_follow, n_raw),
            "assumption": "all non-probe or corrupt-unusable cases are non-follow",
        },
        "all_raw_upper_follow_failed_follow": {
            "k": observed_follow + failed_or_unknown,
            "n": n_raw,
            "rate": pct(observed_follow + failed_or_unknown, n_raw),
            "assumption": "all non-probe or corrupt-unusable cases would follow the injected value",
        },
        "sft_correct_lower_follow_missing_nonfollow": {
            "k": sft_correct_kept_follow,
            "n": n_sft_correct_eval,
            "rate": pct(sft_correct_kept_follow, n_sft_correct_eval),
            "assumption": "SFT-correct cases that did not enter the probe are non-follow",
        },
        "sft_correct_upper_follow_missing_follow": {
            "k": sft_correct_kept_follow + sft_correct_failed_or_unknown,
            "n": n_sft_correct_eval,
            "rate": pct(sft_correct_kept_follow + sft_correct_failed_or_unknown, n_sft_correct_eval),
            "assumption": "SFT-correct cases that did not enter the probe or lacked a usable corrupt answer would follow the injected value",
        },
    }

    return {
        "preds": str(preds_path.relative_to(ROOT)),
        "answer_eval_counts": eval_counts,
        "probe_kept_counts": kept_counts,
        "flow": {
            "n_raw": n_raw,
            "n_answer_eval_base_correct": eval_counts["stable_correct"] + eval_counts["sft_loss"],
            "n_answer_eval_sft_correct": n_sft_correct_eval,
            "n_probe_eligible": n_probe,
            "n_corrupt_usable": corrupt_usable,
            "n_nonprobe_or_corrupt_unusable": failed_or_unknown,
            "n_eval_correct_missing_probe": n_eval_correct_missing_probe,
            "n_probe_not_sft_eval_correct": n_probe_not_sft_eval_correct,
            "n_nonprobe": unknown_nonprobe,
            "n_sft_correct_corrupt_unusable": sft_correct_corrupt_unusable,
        },
        "strata": strata,
        "sensitivity": sensitivity,
        "notes": [
            "answer_eval_counts come from the standalone n=400 SFT evaluation JSONL.",
            "probe_eligible is the separate battery generation subset with nonempty emitted CoT and correct extracted final answer.",
            "n_probe_not_sft_eval_correct reflects deterministic-but-separate generation passes that disagree on a small number of ChartQA 8B cases; these cases are shown as sft_loss/both_wrong strata rather than discarded.",
            "The stored legacy battery does not retain raw generated CoTs for non-eligible cases, so no-CoT, wrong-answer, and parse-failure cases are grouped in n_nonprobe_or_corrupt_unusable for sensitivity bounds.",
        ],
    }


def summarize_cell(spec: CellSpec) -> dict[str, Any]:
    obj = load_json(spec.path)
    summary = obj["summary"]
    details = obj.get("details") or []
    corrupt = (summary.get("interventions") or {})["corrupt"]
    shuffle = (summary.get("interventions") or {})["shuffle"]
    c_flips, c_n = int(corrupt["flips"]), int(corrupt["n"])
    s_flips, s_n = int(shuffle["flips"]), int(shuffle["n"])
    F = float(corrupt["flip_rate"] - shuffle["flip_rate"])

    pairs = paired_flips(details)
    if pairs:
        diffs = [c - s for c, s in pairs]
        n10 = sum(1 for c, s in pairs if c == 1 and s == 0)
        n01 = sum(1 for c, s in pairs if c == 0 and s == 1)
        ci = bootstrap_paired(diffs, seed=stable_seed(f"{spec.dataset}:{spec.model_id}:{spec.condition}:paired"))
        p_raw = binom_tail_greater(n10, n10 + n01)
        test_method = "paired exact sign-randomization"
        ci_method = "paired case bootstrap"
        paired_n = len(pairs)
        discordant = {"corrupt_only": n10, "shuffle_only": n01}
    else:
        ci = bootstrap_independent(
            c_flips, c_n, s_flips, s_n,
            seed=stable_seed(f"{spec.dataset}:{spec.model_id}:{spec.condition}:aggregate"),
        )
        p_raw = fisher_greater(c_flips, c_n, s_flips, s_n)
        test_method = "aggregate exact hypergeometric randomization"
        ci_method = "independent binomial bootstrap"
        paired_n = 0
        discordant = None

    cls = classify_corrupt(details)
    n_raw = int(summary.get("n_cases") or 400)
    n_probe = int(summary.get("n_eval") or max(c_n, s_n))
    base_correct = summary.get("base_correct")
    flow = {
        "n_raw": n_raw,
        "n_base_correct": int(base_correct) if isinstance(base_correct, int) else None,
        "n_probe": n_probe,
        "n_corrupt": c_n,
        "n_shuffle": s_n,
        "note": "n_base_correct missing in legacy battery JSON; n_probe is correct+CoT kept set"
        if base_correct is None else "n_probe is correct+CoT kept set",
    }
    selection = selection_strata(details, spec.preds)

    return {
        "dataset": spec.dataset,
        "model_id": spec.model_id,
        "condition": spec.condition,
        "display": spec.display,
        "path": str(spec.path.relative_to(ROOT)),
        "flow": flow,
        "corrupt": rate_block(c_flips, c_n),
        "shuffle": rate_block(s_flips, s_n),
        "F": F,
        "F_ci95": ci,
        "F_ci_method": ci_method,
        "load_bearing_test": {
            "alternative": "flip_corrupt > flip_shuffle",
            "p_raw": p_raw,
            "method": test_method,
            "paired_n": paired_n,
            "discordant": discordant,
        },
        "snap": cls["snap"],
        "follow": cls["follow"],
        "other": cls["other"],
        "classification_n": cls["n"],
        "selection": selection,
    }


def summarize_local_control(spec: CellSpec) -> dict[str, Any] | None:
    if not spec.path.exists():
        return None
    obj = load_json(spec.path)
    summary = obj["summary"]
    details = obj.get("details") or []
    interventions = summary.get("interventions") or {}
    if "corrupt" not in interventions or "local_num" not in interventions:
        return None

    corrupt = interventions["corrupt"]
    local = interventions["local_num"]
    c_flips, c_n = int(corrupt["flips"]), int(corrupt["n"])
    l_flips, l_n = int(local["flips"]), int(local["n"])
    F_local = float(corrupt["flip_rate"] - local["flip_rate"])

    pairs = paired_flips(details, control_key="local_num")
    if pairs:
        diffs = [c - l for c, l in pairs]
        c_pair_flips = sum(c for c, _ in pairs)
        l_pair_flips = sum(l for _, l in pairs)
        n10 = sum(1 for c, l in pairs if c == 1 and l == 0)
        n01 = sum(1 for c, l in pairs if c == 0 and l == 1)
        F_local = float(sum(diffs) / len(diffs))
        ci = bootstrap_paired(diffs, seed=stable_seed(f"{spec.dataset}:{spec.model_id}:local_num:paired"))
        p_raw = binom_tail_greater(n10, n10 + n01)
        test_method = "paired exact sign-randomization"
        ci_method = "paired case bootstrap"
        paired_n = len(pairs)
        discordant = {"corrupt_only": n10, "local_num_only": n01}
        corrupt_report = rate_block(c_pair_flips, paired_n)
        local_report = rate_block(l_pair_flips, paired_n)
    else:
        ci = bootstrap_independent(
            c_flips, c_n, l_flips, l_n,
            seed=stable_seed(f"{spec.dataset}:{spec.model_id}:local_num:aggregate"),
        )
        p_raw = fisher_greater(c_flips, c_n, l_flips, l_n)
        test_method = "aggregate exact hypergeometric randomization"
        ci_method = "independent binomial bootstrap"
        paired_n = 0
        discordant = None
        corrupt_report = rate_block(c_flips, c_n)
        local_report = rate_block(l_flips, l_n)

    cls = classify_corrupt(details)
    n_raw = int(summary.get("n_cases") or 400)
    n_probe = int(summary.get("n_eval") or max(c_n, l_n))
    return {
        "dataset": spec.dataset,
        "model_id": spec.model_id,
        "condition": spec.condition,
        "display": spec.display,
        "path": str(spec.path.relative_to(ROOT)),
        "control": "local_num",
        "control_description": (
            "same-shape replacement of a non-target numeric token in the same emitted CoT; "
            "sentence order and most local syntax are preserved, and no answer target is injected"
        ),
        "flow": {
            "n_raw": n_raw,
            "n_probe": n_probe,
            "n_corrupt": c_n,
            "n_local_num": l_n,
            "n_paired": paired_n,
        },
        "corrupt": corrupt_report,
        "local_num": local_report,
        "corrupt_all": rate_block(c_flips, c_n),
        "local_num_all": rate_block(l_flips, l_n),
        "F_local": F_local,
        "F_local_ci95": ci,
        "F_local_ci_method": ci_method,
        "load_bearing_test": {
            "alternative": "flip_corrupt > flip_local_num",
            "p_raw": p_raw,
            "method": test_method,
            "paired_n": paired_n,
            "discordant": discordant,
        },
        "snap": cls["snap"],
        "follow": cls["follow"],
        "other": cls["other"],
        "classification_n": cls["n"],
    }


def summarize_semantic_control(spec: CellSpec) -> dict[str, Any] | None:
    if not spec.path.exists():
        return None
    obj = load_json(spec.path)
    summary = obj["summary"]
    details = obj.get("details") or []
    interventions = summary.get("interventions") or {}
    if "corrupt" not in interventions or "semantic_cf" not in interventions:
        return None

    corrupt = interventions["corrupt"]
    sem = interventions["semantic_cf"]
    c_flips, c_n = int(corrupt["flips"]), int(corrupt["n"])
    s_flips, s_n = int(sem["flips"]), int(sem["n"])
    F_semantic = float(corrupt["flip_rate"] - sem["flip_rate"])

    pairs = paired_flips(details, control_key="semantic_cf")
    if pairs:
        diffs = [c - s for c, s in pairs]
        c_pair_flips = sum(c for c, _ in pairs)
        s_pair_flips = sum(s for _, s in pairs)
        n10 = sum(1 for c, s in pairs if c == 1 and s == 0)
        n01 = sum(1 for c, s in pairs if c == 0 and s == 1)
        F_semantic = float(sum(diffs) / len(diffs))
        ci = bootstrap_paired(diffs, seed=stable_seed(f"{spec.dataset}:{spec.model_id}:semantic_cf:paired"))
        p_raw = binom_tail_greater(n10, n10 + n01)
        test_method = "paired exact sign-randomization"
        ci_method = "paired case bootstrap"
        paired_n = len(pairs)
        discordant = {"corrupt_only": n10, "semantic_cf_only": n01}
        corrupt_report = rate_block(c_pair_flips, paired_n)
        semantic_report = rate_block(s_pair_flips, paired_n)
    else:
        ci = bootstrap_independent(
            c_flips, c_n, s_flips, s_n,
            seed=stable_seed(f"{spec.dataset}:{spec.model_id}:semantic_cf:aggregate"),
        )
        p_raw = fisher_greater(c_flips, c_n, s_flips, s_n)
        test_method = "aggregate exact hypergeometric randomization"
        ci_method = "independent binomial bootstrap"
        paired_n = 0
        discordant = None
        corrupt_report = rate_block(c_flips, c_n)
        semantic_report = rate_block(s_flips, s_n)

    reason_counts: dict[str, int] = {}
    for row in details:
        reason = row.get("semantic_cf_reason")
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    cls = classify_corrupt(details)
    n_raw = int(summary.get("n_cases") or 400)
    n_probe = int(summary.get("n_eval") or max(c_n, s_n))
    return {
        "dataset": spec.dataset,
        "model_id": spec.model_id,
        "condition": spec.condition,
        "display": spec.display,
        "path": str(spec.path.relative_to(ROOT)),
        "control": "semantic_cf",
        "control_description": (
            "LLM-rewritten semantic counterfactual: a non-final numeric step and "
            "dependent non-final reasoning are changed while the protected final "
            "conclusion segment is preserved"
        ),
        "flow": {
            "n_raw": n_raw,
            "n_probe": n_probe,
            "n_corrupt": c_n,
            "n_semantic_cf": s_n,
            "n_paired": paired_n,
            "semantic_cf_reason_counts": reason_counts,
        },
        "corrupt": corrupt_report,
        "semantic_cf": semantic_report,
        "corrupt_all": rate_block(c_flips, c_n),
        "semantic_cf_all": rate_block(s_flips, s_n),
        "F_semantic": F_semantic,
        "F_semantic_ci95": ci,
        "F_semantic_ci_method": ci_method,
        "load_bearing_test": {
            "alternative": "flip_corrupt > flip_semantic_cf",
            "p_raw": p_raw,
            "method": test_method,
            "paired_n": paired_n,
            "discordant": discordant,
        },
        "snap": cls["snap"],
        "follow": cls["follow"],
        "other": cls["other"],
        "classification_n": cls["n"],
    }


def load_eval_labels(preds_path: Path) -> tuple[dict[str, dict[str, bool]], int]:
    labels = {}
    sft_gain_total = 0
    for line in preds_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row["cid"]
        bc = bool(row["base_correct"])
        ok = bool(row["correct"])
        labels[cid] = {"base_correct": bc, "correct": ok}
        if ok and not bc:
            sft_gain_total += 1
    return labels, sft_gain_total


def summarize_gain_subset(spec: dict[str, Any]) -> dict[str, Any]:
    labels, sft_gain_total = load_eval_labels(spec["preds"])
    details = load_json(spec["battery"]).get("details") or []
    parts: dict[str, list[dict[str, bool]]] = {
        "GAINED": [],
        "RETAINED": [],
        "SFT_LOSS": [],
        "BOTH_WRONG": [],
        "UNKNOWN": [],
    }
    for row in details:
        ans = answer_for(row, "corrupt")
        injected = row.get("injected")
        if ans is None or injected is None:
            continue
        lab = labels.get(row["case_id"])
        if lab is None:
            key = "UNKNOWN"
        else:
            bc = bool(lab["base_correct"])
            ok = bool(lab["correct"])
            if bc and ok:
                key = "RETAINED"
            elif (not bc) and ok:
                key = "GAINED"
            elif bc and (not ok):
                key = "SFT_LOSS"
            else:
                key = "BOTH_WRONG"
        snap = relaxed_match(ans, row.get("gold", ""))
        follow = bool(injected) and (not snap) and relaxed_match(ans, injected)
        parts[key].append({
            "flip": not relaxed_match(ans, row.get("base_ans", "")),
            "snap": snap,
            "follow": follow,
        })

    def block(rows: list[dict[str, bool]]) -> dict[str, Any]:
        n = len(rows)
        flip = sum(r["flip"] for r in rows)
        snap = sum(r["snap"] for r in rows)
        follow = sum(r["follow"] for r in rows)
        return {
            "n": n,
            "flip": rate_block(flip, n),
            "snap": rate_block(snap, n),
            "follow": rate_block(follow, n),
        }

    return {
        "label": spec["label"],
        "dataset": spec["dataset"],
        "model_id": spec["model_id"],
        "battery": str(spec["battery"].relative_to(ROOT)),
        "preds": str(spec["preds"].relative_to(ROOT)),
        "sft_gain_total_in_eval": sft_gain_total,
        "partitions": {name: block(rows) for name, rows in parts.items() if rows},
    }


def summarize_finqa_targeted(spec: dict[str, Any]) -> dict[str, Any]:
    result = (load_json(spec["path"]).get("results") or {}).get(spec["result_key"]) or {}
    operand = result.get("corrupt_operand") or {}
    consistent = result.get("corrupt_consistent") or {}
    return {
        "label": spec["label"],
        "path": str(spec["path"].relative_to(ROOT)),
        "result_key": spec["result_key"],
        "n_targeted": result.get("n_targeted"),
        "operand_follow": {
            "k": operand.get("follow"),
            "n": result.get("n_targeted"),
            "rate": operand.get("follow_rate"),
            "ci95": operand.get("follow_ci"),
        },
        "operand_snap": {
            "k": operand.get("snap"),
            "n": result.get("n_targeted"),
            "rate": operand.get("snap_rate"),
            "ci95": operand.get("snap_ci"),
        },
        "consistent_follow": {
            "k": consistent.get("follow"),
            "n": result.get("n_targeted"),
            "rate": consistent.get("follow_rate"),
            "ci95": consistent.get("follow_ci"),
        },
        "note": "Targeted FinQA batteries perturb gold-program operands, but current artifacts do not include matched per-case base-vs-SFT eval predictions; therefore they are reported as targeted operand-follow CIs, not an SFT-gain subset.",
    }


def summarize_online_p3() -> dict[str, Any] | None:
    """P3-1 cross-paradigm comparison: prompt-level two-pass force-continue vs stream-level
    in-place continuation, both on the same ChartQA 8B SFT student. Reports follow/snap/flip
    with Wilson 95% CIs so the in-place numbers stand alongside (never replace) the headline
    force-continue numbers."""
    present_p = ONLINE_P3_PATHS["present"]
    if not present_p.exists():
        return None

    def inplace_block(path: Path, condition: str) -> dict[str, Any]:
        s = load_json(path).get("summary") or {}
        oc = s.get("online_corrupt") or {}
        clean = s.get("online_clean_control") or {}
        n = int(oc.get("n", 0))
        return {
            "paradigm": "inplace (single stream)",
            "condition": condition,
            "n_eval": s.get("n_eval"),
            "base_acc": s.get("base_acc"),
            "base_correct": s.get("base_correct"),
            "n_corrupt": n,
            "follow": rate_block(int(oc.get("follow", 0)), n),
            "snap": rate_block(int(oc.get("snap", 0)), n),
            "flip": rate_block(int(oc.get("flips", 0)), n),
            "other": int(oc.get("other", 0)),
            "clean_agree": rate_block(int(clean.get("agree_base", 0)), int(clean.get("n", 0))),
        }

    rows = [inplace_block(present_p, "present (locked-visual)")]
    if ONLINE_P3_PATHS["maskedB"].exists():
        rows.append(inplace_block(ONLINE_P3_PATHS["maskedB"], "masked-B (no-visual)"))

    twopass = None
    tp_p = ONLINE_P3_PATHS["twopass"]
    if tp_p.exists():
        ts = load_json(tp_p).get("summary") or {}
        intr = ts.get("interventions") or {}
        corr = intr.get("corrupt") or {}
        rp = ts.get("re_perception") or {}
        ncorr = int(rp.get("n_corrupt", corr.get("n", 0)) or 0)
        twopass = {
            "paradigm": "twopass (force-continue)",
            "condition": "present (re-prompt)",
            "n_eval": ts.get("n_eval"),
            "base_acc": ts.get("base_acc"),
            "n_corrupt": ncorr,
            "follow": rate_block(int(rp.get("follows_injected", 0)), ncorr),
            "snap": rate_block(int(rp.get("snap_to_true", 0)), ncorr),
            "flip": rate_block(int(corr.get("flips", 0)), int(corr.get("n", 0) or 0)),
        }

    return {
        "definition": (
            "P3-1 generation-time intervention. inplace = the model's own greedy token stream "
            "is re-fed up to the boundary before a selected numeric token, the corrupted value "
            "v_inj is substituted, and the model continues generating its own chain and answer "
            "(single autoregressive stream, no added instruction, conclusion line never supplied). "
            "twopass = the headline force-continue paradigm. Both run on the same SFT student; the "
            "in-place readout (snap/follow/other) is identical to the two-pass readout. "
            "online_clean re-feeds v_true and continues; its high base-answer agreement certifies "
            "the re-encode continuation reproduces the original stream."
        ),
        "preregistration": "docs/preregistration_p3_inplace.md",
        "interpretation": (
            "CONVERGENCE if present in-place follow Wilson upper bound < 0.10 and in-place "
            "corrupt-flip not materially above two-pass corrupt-flip."
        ),
        "twopass": twopass,
        "inplace": rows,
    }


def fmt_rate(block: dict[str, Any]) -> str:
    return f"{block['rate']:.3f} [{block['ci95'][0]:.3f},{block['ci95'][1]:.3f}]"


def fmt_simple_rate(block: dict[str, Any]) -> str:
    return f"{block['k']}/{block['n']} ({block['rate']:.3f})"


def fmt_p(p: float) -> str:
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def render_md(result: dict[str, Any]) -> str:
    lines = [
        "# P0-2/P0-3/P0-4 Faithfulness Probe Statistics and Selection Audit",
        "",
        "Confirmatory family: the eight ChartQA/TabMWP corrupt-vs-shuffle cells below. "
        "The tested load-bearing alternative is `flip_corrupt > flip_shuffle`; p-values are Holm corrected within this family.",
        "",
        "Legacy battery JSONs store corrupt answers per case but only aggregate shuffle counts, so those rows use aggregate exact hypergeometric randomization plus independent binomial bootstrap CIs. "
        "Rows with `details[].answers.shuffle` would use paired exact sign-randomization and paired bootstrap automatically.",
        "",
        "| cell | cond | raw->probe (c/s) | corrupt/shuffle | F 95% CI | p raw | p Holm | snap 95% CI | follow 95% CI | test |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for c in result["cells"]:
        flow = c["flow"]
        flow_s = f"{flow['n_raw']}->{flow['n_probe']} ({flow['n_corrupt']}/{flow['n_shuffle']})"
        cs = f"{c['corrupt']['rate']:.3f}/{c['shuffle']['rate']:.3f}"
        F = f"{c['F']:+.3f} [{c['F_ci95'][0]:+.3f},{c['F_ci95'][1]:+.3f}]"
        test = c["load_bearing_test"]
        lines.append(
            f"| {c['display']} | {c['condition']} | {flow_s} | {cs} | {F} | "
            f"{fmt_p(test['p_raw'])} | {fmt_p(test['p_holm'])} | "
            f"{fmt_rate(c['snap'])} | {fmt_rate(c['follow'])} | {test['method']} |"
        )

    local_controls = result.get("local_controls") or []
    if local_controls:
        lines += [
            "",
            "## P0-4 Local Numeric Control",
            "",
            "`local_num` is a format-preserving no-target control: it replaces a different numeric token in the same emitted CoT with a same-shape value, preserves sentence order and local syntax, excludes the corrupt arm's selected token, and avoids the gold/base/injected values when possible. "
            "The confirmatory alternative is `flip_corrupt > flip_local_num`, Holm corrected within the local-control family.",
            "",
            "| cell | cond | raw->probe (c/local) | corrupt/local_num | F_local 95% CI | p raw | p Holm | snap 95% CI | follow 95% CI | test |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for c in local_controls:
            flow = c["flow"]
            flow_s = f"{flow['n_raw']}->{flow['n_probe']} ({flow['n_corrupt']}/{flow['n_local_num']})"
            cs = f"{c['corrupt']['rate']:.3f}/{c['local_num']['rate']:.3f}"
            F = f"{c['F_local']:+.3f} [{c['F_local_ci95'][0]:+.3f},{c['F_local_ci95'][1]:+.3f}]"
            test = c["load_bearing_test"]
            lines.append(
                f"| {c['display']} | {c['condition']} | {flow_s} | {cs} | {F} | "
                f"{fmt_p(test['p_raw'])} | {fmt_p(test['p_holm'])} | "
                f"{fmt_rate(c['snap'])} | {fmt_rate(c['follow'])} | {test['method']} |"
            )

    semantic_controls = result.get("semantic_controls") or []
    if semantic_controls:
        lines += [
            "",
            "## P0-4 Semantic Counterfactual Control",
            "",
            "`semantic_cf` is an LLM-rewritten counterfactual control: non-final numeric reasoning is changed, dependent non-final steps are kept internally consistent with that wrong intermediate number, and the protected final conclusion segment is preserved rather than directly overwritten. "
            "The confirmatory alternative is `flip_corrupt > flip_semantic_cf`, Holm corrected within the available semantic-control family.",
            "",
            "| cell | cond | raw->probe (c/sem) | corrupt/semantic_cf | F_semantic 95% CI | p raw | p Holm | usable rewrites | snap 95% CI | follow 95% CI | test |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for c in semantic_controls:
            flow = c["flow"]
            flow_s = f"{flow['n_raw']}->{flow['n_probe']} ({flow['n_corrupt']}/{flow['n_semantic_cf']})"
            cs = f"{c['corrupt']['rate']:.3f}/{c['semantic_cf']['rate']:.3f}"
            F = f"{c['F_semantic']:+.3f} [{c['F_semantic_ci95'][0]:+.3f},{c['F_semantic_ci95'][1]:+.3f}]"
            test = c["load_bearing_test"]
            reason_counts = flow.get("semantic_cf_reason_counts") or {}
            ok_rewrites = reason_counts.get("ok", flow["n_semantic_cf"])
            lines.append(
                f"| {c['display']} | {c['condition']} | {flow_s} | {cs} | {F} | "
                f"{fmt_p(test['p_raw'])} | {fmt_p(test['p_holm'])} | "
                f"{ok_rewrites}/{flow['n_probe']} | {fmt_rate(c['snap'])} | "
                f"{fmt_rate(c['follow'])} | {test['method']} |"
            )

    cross_family = result.get("cross_family") or []
    if cross_family:
        lines += [
            "",
            "## P0-5 Cross-Family Replication",
            "",
            "These cells are not part of the Qwen-family confirmatory Holm family above. "
            "They are an external-validity replication on a non-Qwen VLM using the same ChartQA present-image probe protocol.",
            "",
            "| cell | cond | raw->probe (c/s) | base acc | corrupt/shuffle | F 95% CI | p raw | snap 95% CI | follow 95% CI | test |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for c in cross_family:
            flow = c["flow"]
            summary = c.get("summary") or {}
            flow_s = f"{flow['n_raw']}->{flow['n_probe']} ({flow['n_corrupt']}/{flow['n_shuffle']})"
            base_correct = summary.get("base_correct")
            base_acc = summary.get("base_acc")
            base_s = (
                f"{base_correct}/{flow['n_raw']} ({base_acc:.3f})"
                if isinstance(base_correct, int) and isinstance(base_acc, float)
                else "NA"
            )
            cs = f"{c['corrupt']['rate']:.3f}/{c['shuffle']['rate']:.3f}"
            F = f"{c['F']:+.3f} [{c['F_ci95'][0]:+.3f},{c['F_ci95'][1]:+.3f}]"
            test = c["load_bearing_test"]
            lines.append(
                f"| {c['display']} | {c['condition']} | {flow_s} | {base_s} | {cs} | {F} | "
                f"{fmt_p(test['p_raw'])} | {fmt_rate(c['snap'])} | "
                f"{fmt_rate(c['follow'])} | {test['method']} |"
            )

    online = result.get("online_p3")
    if online:
        lines += [
            "",
            "## P3-1 Generation-Time (Single-Stream) Intervention",
            "",
            "Cross-paradigm comparison on the same ChartQA 8B SFT student. The two-pass row is the "
            "headline force-continue paradigm; the in-place rows intervene inside the model's own "
            "autoregressive stream (no re-prompt, no added instruction, the conclusion line is never "
            "supplied). Readout is identical across paradigms. Reported alongside, not replacing, the "
            "force-continue results. Pre-registration: " + str(online.get("preregistration")) + ".",
            "",
            "| paradigm | condition | n_corrupt | base acc | follow 95% CI | snap 95% CI | flip 95% CI | clean-agree |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        if online.get("twopass"):
            t = online["twopass"]
            ba = f"{t['base_acc']:.3f}" if isinstance(t.get("base_acc"), float) else "NA"
            lines.append(
                f"| {t['paradigm']} | {t['condition']} | {t['n_corrupt']} | {ba} | "
                f"{fmt_rate(t['follow'])} | {fmt_rate(t['snap'])} | {fmt_rate(t['flip'])} | - |"
            )
        for r in online["inplace"]:
            ba = f"{r['base_acc']:.3f}" if isinstance(r.get("base_acc"), float) else "NA"
            lines.append(
                f"| {r['paradigm']} | {r['condition']} | {r['n_corrupt']} | {ba} | "
                f"{fmt_rate(r['follow'])} | {fmt_rate(r['snap'])} | {fmt_rate(r['flip'])} | "
                f"{fmt_simple_rate(r['clean_agree'])} |"
            )
        lines += ["", f"_Interpretation rule (pre-registered): {online.get('interpretation')}_"]

    lines += [
        "",
        "## Selection Flow and Probe Strata",
        "",
        "The probe battery regenerates an original CoT and keeps cases with a nonempty emitted CoT and a correct extracted final answer. "
        "The answer-eval columns come from the separate n=400 SFT evaluation JSONL; the small ChartQA 8B disagreement between answer-eval and probe-generation passes is retained as loss/wrong probe strata.",
        "",
        "| cell | cond | raw | answer-eval base correct | answer-eval SFT correct | probe eligible | corrupt usable | nonprobe/unusable | eval-correct missing probe | probe not SFT-correct in eval |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in result["cells"]:
        sel = c.get("selection")
        if not sel:
            continue
        f = sel["flow"]
        lines.append(
            f"| {c['display']} | {c['condition']} | {f['n_raw']} | "
            f"{f['n_answer_eval_base_correct']} | {f['n_answer_eval_sft_correct']} | "
            f"{f['n_probe_eligible']} | {f['n_corrupt_usable']} | "
            f"{f['n_nonprobe_or_corrupt_unusable']} | {f['n_eval_correct_missing_probe']} | "
            f"{f['n_probe_not_sft_eval_correct']} |"
        )

    lines += [
        "",
        "| cell | cond | answer-eval strata stable/gain/loss/wrong | probe-kept stable/gain/loss/wrong | stable follow | gain follow | loss follow | wrong follow | gain snap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in result["cells"]:
        sel = c.get("selection")
        if not sel:
            continue
        eval_counts = sel["answer_eval_counts"]
        kept_counts = sel["probe_kept_counts"]
        strata = sel["strata"]

        def count_s(counts: dict[str, int]) -> str:
            return (
                f"{counts.get('stable_correct', 0)}/"
                f"{counts.get('sft_gain', 0)}/"
                f"{counts.get('sft_loss', 0)}/"
                f"{counts.get('both_wrong', 0)}"
            )

        def rate_or_na(name: str, metric: str) -> str:
            block = (strata.get(name) or {}).get(metric)
            return fmt_simple_rate(block) if block else "NA"

        lines.append(
            f"| {c['display']} | {c['condition']} | {count_s(eval_counts)} | {count_s(kept_counts)} | "
            f"{rate_or_na('stable_correct', 'follow')} | {rate_or_na('sft_gain', 'follow')} | "
            f"{rate_or_na('sft_loss', 'follow')} | {rate_or_na('both_wrong', 'follow')} | "
            f"{rate_or_na('sft_gain', 'snap')} |"
        )

    lines += [
        "",
        "### Sensitivity Bounds",
        "",
        "Legacy battery artifacts retain eligible probe details but not raw generated CoTs for non-eligible cases, so the main sensitivity analysis brackets the unobserved non-probe/parse-failure mass.",
        "",
        "| cell | cond | observed probe follow | all-raw lower | all-raw upper | SFT-correct lower | SFT-correct upper |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for c in result["cells"]:
        sel = c.get("selection")
        if not sel:
            continue
        sens = sel["sensitivity"]
        obs = sens["observed_probe_follow"]
        raw_lo = sens["all_raw_lower_follow_failed_nonfollow"]
        raw_hi = sens["all_raw_upper_follow_failed_follow"]
        sft_lo = sens["sft_correct_lower_follow_missing_nonfollow"]
        sft_hi = sens["sft_correct_upper_follow_missing_follow"]
        lines.append(
            f"| {c['display']} | {c['condition']} | {fmt_simple_rate(obs)} | "
            f"{raw_lo['k']}/{raw_lo['n']} ({raw_lo['rate']:.3f}) | "
            f"{raw_hi['k']}/{raw_hi['n']} ({raw_hi['rate']:.3f}) | "
            f"{sft_lo['k']}/{sft_lo['n']} ({sft_lo['rate']:.3f}) | "
            f"{sft_hi['k']}/{sft_hi['n']} ({sft_hi['rate']:.3f}) |"
        )

    lines += [
        "",
        "## Gain-Subset Probe",
        "",
        "| cell | answer-eval gains | probe-kept gained n | gained flip | gained snap 95% CI | gained follow 95% CI | retained follow 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for g in result["gain_subsets"]:
        gained = (g["partitions"] or {}).get("GAINED", {})
        retained = (g["partitions"] or {}).get("RETAINED", {})
        if not gained:
            continue
        lines.append(
            f"| {g['label']} | {g['sft_gain_total_in_eval']} | {gained['n']} | "
            f"{gained['flip']['k']}/{gained['n']} ({gained['flip']['rate']:.3f}) | "
            f"{fmt_rate(gained['snap'])} | {fmt_rate(gained['follow'])} | "
            f"{fmt_rate(retained['follow']) if retained else 'NA'} |"
        )

    lines += [
        "",
        "## FinQA Targeted Operand Probe",
        "",
        "The `answer-eval gains` column is the standalone n=400 accuracy-eval gain count. "
        "`probe-kept gained n` is the base-wrong -> CoT-SFT-right subset that also survived CoT/probe eligibility. "
        "FinQA is not reported as a gain subset in the current artifacts because the targeted battery is keyed by gold-program operand eligibility, not by matched base-wrong -> SFT-right eval partitions.",
        "",
        "| arm | n_targeted | operand follow 95% CI | operand snap 95% CI | consistent-conclusion follow 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for f in result["finqa_targeted"]:
        op_follow = f["operand_follow"]
        op_snap = f["operand_snap"]
        co_follow = f["consistent_follow"]
        lines.append(
            f"| {f['label']} | {f['n_targeted']} | "
            f"{op_follow['k']}/{op_follow['n']} ({op_follow['rate']:.3f}) "
            f"[{op_follow['ci95'][0]:.3f},{op_follow['ci95'][1]:.3f}] | "
            f"{op_snap['k']}/{op_snap['n']} ({op_snap['rate']:.3f}) "
            f"[{op_snap['ci95'][0]:.3f},{op_snap['ci95'][1]:.3f}] | "
            f"{co_follow['k']}/{co_follow['n']} ({co_follow['rate']:.3f}) "
            f"[{co_follow['ci95'][0]:.3f},{co_follow['ci95'][1]:.3f}] |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    cells = [summarize_cell(spec) for spec in HEADLINE_CELLS]
    adjusted = holm_adjust([c["load_bearing_test"]["p_raw"] for c in cells])
    for c, p_holm in zip(cells, adjusted):
        c["load_bearing_test"]["p_holm"] = p_holm
    local_controls = [
        c for c in (summarize_local_control(spec) for spec in LOCAL_CONTROL_SPECS)
        if c is not None
    ]
    local_adjusted = holm_adjust([c["load_bearing_test"]["p_raw"] for c in local_controls])
    for c, p_holm in zip(local_controls, local_adjusted):
        c["load_bearing_test"]["p_holm"] = p_holm
    semantic_controls = [
        c for c in (summarize_semantic_control(spec) for spec in SEMANTIC_CONTROL_SPECS)
        if c is not None
    ]
    semantic_adjusted = holm_adjust([c["load_bearing_test"]["p_raw"] for c in semantic_controls])
    for c, p_holm in zip(semantic_controls, semantic_adjusted):
        c["load_bearing_test"]["p_holm"] = p_holm
    cross_family = [
        c for c in (summarize_cell(spec) for spec in CROSS_FAMILY_SPECS)
        if c is not None
    ]
    for c in cross_family:
        c["summary"] = load_json(ROOT / c["path"]).get("summary") or {}

    result = {
        "definition": "F = flip_corrupt - flip_shuffle. Snap/follow are classified on the corrupt continuation with snap precedence over follow.",
        "multiple_comparisons": {
            "method": "Holm-Bonferroni",
            "family": "8 confirmatory ChartQA/TabMWP corrupt-vs-shuffle cells",
            "alternative": "flip_corrupt > flip_shuffle",
        },
        "local_control_definition": "F_local = flip_corrupt - flip_local_num, where local_num is a same-shape non-target numeric replacement in the same CoT.",
        "local_control_multiple_comparisons": {
            "method": "Holm-Bonferroni",
            "family": f"{len(local_controls)} available P0-4 local numeric control cells",
            "alternative": "flip_corrupt > flip_local_num",
        },
        "semantic_control_definition": "F_semantic = flip_corrupt - flip_semantic_cf, where semantic_cf is an LLM-rewritten non-final numeric counterfactual with the final conclusion segment preserved.",
        "semantic_control_multiple_comparisons": {
            "method": "Holm-Bonferroni",
            "family": f"{len(semantic_controls)} available P0-4 semantic counterfactual control cells",
            "alternative": "flip_corrupt > flip_semantic_cf",
        },
        "cross_family_definition": (
            "P0-5 external-validity replication: non-Qwen VLM, ChartQA present-image "
            "accuracy plus corrupt/shuffle forced-continuation probe. It is reported "
            "separately from the Qwen-family confirmatory Holm family."
        ),
        "cells": cells,
        "local_controls": local_controls,
        "semantic_controls": semantic_controls,
        "cross_family": cross_family,
        "online_p3": summarize_online_p3(),
        "gain_subsets": [summarize_gain_subset(spec) for spec in GAIN_SPECS],
        "finqa_targeted": [summarize_finqa_targeted(spec) for spec in FINQA_TARGETED],
    }

    RES.mkdir(parents=True, exist_ok=True)
    out_json = RES / "faithfulness_stats.json"
    out_md = RES / "faithfulness_stats.md"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    out_md.write_text(render_md(result))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print(render_md(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
