# P3-1 Pre-registration — generation-time (single-stream) intervention

Date: 2026-06-28
Author: project owner
Scope: `todo/0627.md` P3-1 (closer-to-causal-path robustness probe), design spec `todo/06.md`.

This file is written and committed **before** running the in-place / single-stream
probe so the convergence threshold is not chosen post-hoc (`todo/06.md` §6).

## Motivation

The Devil's-Advocate objection to the headline force-continue probe is structural:
the two-pass paradigm builds a **new prompt** ("Reasoning so far: <edited CoT> …
Given ONLY the reasoning above, state ANSWER:") and runs a **second, independent
forward pass with an added instruction**. The low follow-rate in present-image
cells therefore admits an alternative reading: "the model just re-read the image
on pass 2," not "the written chain is not load-bearing."

## Intervention under test (satisfies C1 of `todo/06.md`)

Single autoregressive stream, no added instruction, no conclusion line shown:

1. **Pass-Generate** — greedy generation of the model's own CoT+ANSWER from the
   normal prompt (image present or absent). Record the generated token stream.
2. **Online replacement / prefix-only continuation** — re-feed the model's *own*
   greedy tokens up to the token boundary just before a selected numeric token,
   then substitute the corrupted value `v_inj` (= same `2v+7` rule as the
   two-pass `corrupt`) and let the model **continue generating** the rest of the
   chain and its answer. The conclusion line is never supplied — the model writes
   it. Re-feeding the model's own greedy token ids is numerically equivalent to
   KV-cache continuation (same ids → same forward states), but uses `generate()`
   so multimodal RoPE is handled by the library rather than hand-rolled.

Readout is **identical** to the two-pass battery so the only thing that changes
is the paradigm:
- `snap`   = continuation answer matches gold (`c_true`)
- `follow` = continuation answer matches the injected value `v_inj`
- `other`  = neither
- `flip`   = continuation answer != base answer

Paradigm-validity control: **`online_clean`** re-feeds the model's own tokens up
to and including the *true* number `v_true` and continues. It must reproduce the
base answer at a high rate; otherwise the re-encode continuation is not faithful
to the original stream and the corrupt arm is uninterpretable.

## Conditions (no-visual / locked-visual)

- **present (locked-visual)** — image present in Pass-Generate and continuation.
- **masked-B (no-visual)** — image absent throughout (the model never sees it).
  Distinguishes re-reading visible evidence from using the text prefix: if follow
  stays near zero with the image present but rises when the image is removed, the
  answer is recovered from the still-visible source, not the corrupted text.

Substrate: ChartQA, Qwen3-VL-8B SFT student (headline cell, matches the P0-4 /
P0-5 artifacts). Decode greedy. Target selection: `Random(0)` over kept cases,
same selector as the two-pass `corrupt` arm.

## Pre-specified interpretation (decided before looking)

Let `U` = Wilson 95% upper bound on the present in-place follow-rate.

- **CONVERGENCE (expected, strongest):** `U < 0.10` and in-place corrupt-flip is
  not materially above the two-pass corrupt-flip → the written chain is not
  load-bearing under *both* prompt-level and stream-level intervention. The
  force-continue conclusion survives removal of the two-pass confound.
- **DIVERGENCE (would force reframing):** present in-place follow point estimate
  > 0.10 with a Wilson CI whose lower bound clearly exceeds the two-pass follow
  estimate → the chain *is* load-bearing in-stream and re-prompting destroyed the
  signal. This would be reported honestly as the more important finding and would
  require retitling / reframing.

Reporting rule: the in-place numbers are reported **alongside**, never replacing,
the force-continue results (P3-1 step 4). Sample target: all probe-eligible cases
(~321), accept ≥150 valid in-place interventions per cell.

## Outcome (recorded 2026-06-28, after the run)

The headline ChartQA 8B adapter weights had been pruned from disk, so the student
was retrained from the same `chartqa_cot_train.jsonl` recipe (QLoRA nf4) and BOTH
paradigms were run on that single fresh student (base accuracy .785 in both,
n_eval 314). Numbers from `data/distill/results/faithfulness_stats.md`:

| paradigm | follow [95% CI] | snap [95% CI] | flip [95% CI] | clean-agree |
|---|---|---|---|---|
| two-pass (force-continue), present | .036 [.020,.063] | .935 [.902,.957] | .098 [.069,.136] | — |
| in-place (single stream), present  | .094 [.067,.132] | .733 [.681,.779] | .293 [.245,.346] | 287/307 (.935) |
| in-place, masked-B (no-visual)     | .000 [.000,.143] | .478 [.292,.670] | .609 [.408,.778] | 12/23 (.522) |

**Verdict against the pre-registered threshold: NOT clean convergence.** The
present in-place follow upper bound (.132) exceeds .10 and in-place flip (.293) is
materially above two-pass (.098), so by the pre-specified rule this is the
divergence-leaning outcome and is reported as such, not as a confirmation.

**Honest reading.** The two-pass re-prompt *under*-estimates the written chain's
causal influence: removing the re-prompt confound raises both flip and follow.
But the core conclusion survives the stricter test — follow stays low (.094) and
snap still dominates (.733), and most of the extra movement is into "other"
(53/307 = .173), i.e. stream-level corruption derails generation more than
re-prompting does, yet the model rarely adopts the injected value as its answer.
The clean control (re-feed v_true and continue) reproduces the base answer 93.5%
of the time, certifying the re-encoded continuation tracks the original stream.

The no-visual cell keeps follow at 0/23 but is low-powered (base accuracy .07
without the chart) with a degraded clean control (.52), so it is reported only as
a secondary diagnostic consistent with the masked force-continue finding. This
satisfies P3-1's acceptance: a probe closer to the causal path exists and is
reported alongside force-continue, and it does so without overclaiming.
