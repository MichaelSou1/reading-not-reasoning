# Reading, Not Reasoning

**A causal faithfulness audit of chain-of-thought in distilled chart-and-table VLMs.**

Paper draft: [`docs/paper/paper.tex`](docs/paper/paper.tex) / [`docs/paper/paper.pdf`](docs/paper/paper.pdf)
Target: AAAI 2027 Main Technical Track

---

## Core finding

CoT SFT reliably improves accuracy on chart, table, natural-image counting, and rendered financial QA. Yet a causal probe battery shows the emitted written chain is **not load-bearing**: corrupting an on-path numeric intermediate flips the final answer no more often than a format-matched control, and the model almost never follows the injected value — it snaps back to the answer readable from the original visual evidence.

The information source behind each accuracy gain is task-dependent:

| Task | Dominant info source |
|---|---|
| ChartQA / TabMWP | Re-reading visible chart / table |
| TallyQA (natural counting) | Redundant enumeration in the rationale |
| FinQA curriculum (N1) | Copying the chain's own conclusion |

Even a shortcut-removing curriculum with gold-program-verified operands raises accuracy substantially (8B: .08 → .67; 32B: .12 → .68 on FinQA) without installing load-bearing intermediate computation (`operand-follow = 0/172` at 8B, `0/175` at 32B).

---

## Probes

| Intervention | What it tests |
|---|---|
| `CORRUPT` | Replace one on-path numeric intermediate with a wrong value |
| `SHUFFLE` | Same tokens, shuffled order — format control with no injected target |
| `paraphrase` / `filler` / `delete` / `truncate` | Robustness battery |
| `image-mask` | Remove visual input — identifies snap-to-memory vs re-read |

Readout: **snap** (returns to true value after CORRUPT) / **follow** (tracks injected wrong value) / **other**.
A chain is load-bearing only when CORRUPT flips substantially more than controls and follow is nontrivial.

---

## Models and tasks

| | |
|---|---|
| **Models** | Qwen3-VL 8B and 32B (dense), served locally via vLLM |
| **Tasks** | ChartQA (400), TabMWP (400), TallyQA-complex natural counting (400), FinQA (curriculum probe) |
| **SFT variants** | LoRA and dense/full-SFT (8B: embedding + first 3 layers frozen, ~79.7% trainable params) |
| **Statistics** | McNemar exact tests; Holm correction across 4 headline cells; all below *p* = .01 |

---

## Key experiments

| ID | Description | Status | Key result |
|---|---|---|---|
| WU-1 | Expand ChartQA to n=400, gate all cells | ✅ | 32B SFT +7 pp, *p*=.0001 |
| WU-2 / N2 | Full 6-intervention battery + re-read control | ✅ | snap ≈ .98, follow ≈ .02 |
| WU-3 | TabMWP — chart+table regime | ✅ | SFT +10.3 pp (8B), F≈0 |
| N1 | FinQA curriculum probe (H_fail_bypass) | ✅ | operand-follow 0/172 (8B), 0/175 (32B) |
| N3 | TallyQA natural-image pole probe | ✅ | F=0, follow 0/345; unified re-readability axis |
| P0-2 | Dense/full-SFT control — blocks LoRA-capacity alternative | ✅ | same F≈0/high-snap/low-follow signature |
| WU-5 | faithfulness ⊥ accuracy figure | ✅ | |
| WU-6 / N4 | Full paper reframe to reading-not-reasoning + regime narrative | ✅ | canonical `paper.tex` |

Detailed execution log: [`todo/0622.md`](todo/0622.md), [`todo/0626.md`](todo/0626.md)
Current submission TODO: [`todo/0627.md`](todo/0627.md)
Snapshots: [`docs/snapshots/`](docs/snapshots/)

---

## Repo Layout

```
app/distill/          # Numeric grading, method runners, statistics, result-store helpers
app/eval_distill/     # Lightweight result diagnostics and control-set builders
scripts/              # CLI entry points: battery_n400.py, probe_n400.py,
                      #   poc_sft*.py, audit_*.py, regen_tables.py, etc.
data/distill/         # Append-only result store (results.jsonl + per-run poc/*.json)
docs/paper/           # LaTeX source + figures
docs/snapshots/       # Per-milestone evidence snapshots
eval/                 # Benchmark question files (ChartQA, TabMWP, natcount)
```

Runtime directories (`data/`, `models/`) are not tracked in Git.

---

## Reproducing results

### 1. Environments

Two conda envs:

| Env | Purpose |
|---|---|
| `vllm-qwen` | Model serving only — do not install harness deps here |
| `mbe-up` | All non-serving work: probing, SFT, eval, paper scripts |

Create `mbe-up`:

```bash
conda create -n mbe-up python=3.11 -y
conda run -n mbe-up pip install torch --index-url https://download.pytorch.org/whl/cu124
conda run -n mbe-up pip install -r requirements-upgrade.txt
```

Smoke-test:

```bash
conda run -n mbe-up python -c "
import torch, transformers, peft, bitsandbytes
print(torch.__version__, torch.cuda.is_available(), transformers.__version__)
"
# Expected: 2.6.0+cu124 True 5.5.0
```

### 2. Serving

Start both model servers before running any gate or battery:

```bash
# From vllm-qwen env — see scripts/serve/ for the exact commands
# 32B @ :30001,  8B @ :30000 (or :30002)
```

### 3. Running a probe battery

```bash
# ChartQA 6-intervention battery (present condition)
conda run -n mbe-up python scripts/battery_n400.py \
  --task chartqa --condition present --model 8b

# FinQA curriculum (N1) targeted probe
conda run -n mbe-up python scripts/battery_n1_targeted.py \
  --arms b2 vanilla base --model 8b
```

Results are appended to `data/distill/results/results.jsonl` and per-run JSON files under `data/distill/poc/`.

### 4. Regenerating paper tables

```bash
conda run -n mbe-up python scripts/regen_tables.py
```

### 5. Full-SFT control audit

```bash
conda run -n mbe-up python scripts/audit_full_sft_8b_nonvideo.py --strict
# PASS = LoRA-capacity alternative explanation blocked
```

---

## Result store

All numeric results used in the paper come from the append-only store at `data/distill/results/results.jsonl` and `data/distill/poc/*.json`. The paper's claim is that all numbers regenerate from this store — see §Reproducibility in the paper.

Prediction and judge caches live under `data/distill/` per task subdirectory.

---

## Current paper status

Draft date: 2026-06-26. Current revision work is tracked in [`todo/0627.md`](todo/0627.md).

Completed foundation:

- P0-1 method transparency is complete in the current draft.
- Dense/full-SFT control blocks the LoRA-capacity alternative.
- F is measured on the same student checkpoint as accuracy.
- Figure 1 / Figure 2 text-figure consistency has been verified.
- TabMWP full 6-intervention battery (present + masked) is complete with `details[].answers`.

Remaining before submission: P0-2 statistics, P0-3 selection-flow analysis, P0-4 cleaner control or claim downgrade, P0-6 claim-boundary pass, P1 positioning/method-title edits, AAAI template, double-blind audit, and reproducibility checklist.
