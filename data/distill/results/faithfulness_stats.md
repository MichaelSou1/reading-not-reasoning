# P0-2/P0-3/P0-4 Faithfulness Probe Statistics and Selection Audit

Confirmatory family: the eight ChartQA/TabMWP corrupt-vs-shuffle cells below. The tested load-bearing alternative is `flip_corrupt > flip_shuffle`; p-values are Holm corrected within this family.

Legacy battery JSONs store corrupt answers per case but only aggregate shuffle counts, so those rows use aggregate exact hypergeometric randomization plus independent binomial bootstrap CIs. Rows with `details[].answers.shuffle` would use paired exact sign-randomization and paired bootstrap automatically.

| cell | cond | raw->probe (c/s) | corrupt/shuffle | F 95% CI | p raw | p Holm | snap 95% CI | follow 95% CI | test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ChartQA 8B | present | 400->321 (321/321) | 0.212/0.302 | -0.090 [-0.159,-0.022] | 0.997 | 1.000 | 0.816 [0.770,0.855] | 0.031 [0.017,0.056] | aggregate exact hypergeometric randomization |
| ChartQA 8B | masked | 400->321 (321/321) | 0.333/0.408 | -0.075 [-0.150,+0.000] | 0.980 | 1.000 | 0.710 [0.658,0.757] | 0.100 [0.072,0.137] | aggregate exact hypergeometric randomization |
| ChartQA 32B | present | 400->318 (316/318) | 0.051/0.047 | +0.003 [-0.031,+0.038] | 0.493 | 1.000 | 0.981 [0.959,0.991] | 0.016 [0.007,0.036] | aggregate exact hypergeometric randomization |
| ChartQA 32B | masked | 400->318 (316/318) | 0.120/0.129 | -0.009 [-0.059,+0.042] | 0.674 | 1.000 | 0.908 [0.871,0.935] | 0.057 [0.036,0.088] | aggregate exact hypergeometric randomization |
| TabMWP 8B | present | 400->385 (385/385) | 0.034/0.145 | -0.112 [-0.151,-0.073] | 1.000 | 1.000 | 0.971 [0.950,0.984] | 0.008 [0.003,0.023] | aggregate exact hypergeometric randomization |
| TabMWP 8B | masked | 400->385 (385/385) | 0.278/0.299 | -0.021 [-0.086,+0.044] | 0.763 | 1.000 | 0.722 [0.675,0.764] | 0.088 [0.064,0.121] | aggregate exact hypergeometric randomization |
| TabMWP 32B | present | 400->389 (389/389) | 0.033/0.072 | -0.039 [-0.069,-0.008] | 0.995 | 1.000 | 0.969 [0.947,0.982] | 0.015 [0.007,0.033] | aggregate exact hypergeometric randomization |
| TabMWP 32B | masked | 400->389 (389/389) | 0.075/0.103 | -0.028 [-0.069,+0.010] | 0.935 | 1.000 | 0.928 [0.898,0.950] | 0.039 [0.024,0.063] | aggregate exact hypergeometric randomization |

## P0-4 Local Numeric Control

`local_num` is a format-preserving no-target control: it replaces a different numeric token in the same emitted CoT with a same-shape value, preserves sentence order and local syntax, excludes the corrupt arm's selected token, and avoids the gold/base/injected values when possible. The confirmatory alternative is `flip_corrupt > flip_local_num`, Holm corrected within the local-control family.

| cell | cond | raw->probe (c/local) | corrupt/local_num | F_local 95% CI | p raw | p Holm | snap 95% CI | follow 95% CI | test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ChartQA 8B | present | 400->321 (321/294) | 0.235/0.224 | +0.010 [-0.034,+0.054] | 0.385 | 0.385 | 0.807 [0.760,0.846] | 0.040 [0.024,0.068] | paired exact sign-randomization |

## P0-4 Semantic Counterfactual Control

`semantic_cf` is an LLM-rewritten counterfactual control: non-final numeric reasoning is changed, dependent non-final steps are kept internally consistent with that wrong intermediate number, and the protected final conclusion segment is preserved rather than directly overwritten. The confirmatory alternative is `flip_corrupt > flip_semantic_cf`, Holm corrected within the available semantic-control family.

| cell | cond | raw->probe (c/sem) | corrupt/semantic_cf | F_semantic 95% CI | p raw | p Holm | usable rewrites | snap 95% CI | follow 95% CI | test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ChartQA 8B | present | 400->321 (321/288) | 0.236/0.434 | -0.198 [-0.257,-0.139] | 1.000 | 1.000 | 288/321 | 0.807 [0.760,0.846] | 0.040 [0.024,0.068] | paired exact sign-randomization |

## P0-5 Cross-Family Replication

These cells are not part of the Qwen-family confirmatory Holm family above. They are an external-validity replication on a non-Qwen VLM using the same ChartQA present-image probe protocol.

| cell | cond | raw->probe (c/s) | base acc | corrupt/shuffle | F 95% CI | p raw | snap 95% CI | follow 95% CI | test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ChartQA InternVL3.5-8B | present | 400->302 (302/302) | 302/400 (0.755) | 0.070/0.096 | -0.026 [-0.053,-0.003] | 0.994 | 0.990 [0.971,0.997] | 0.000 [0.000,0.013] | paired exact sign-randomization |

## P3-1 Generation-Time (Single-Stream) Intervention

Cross-paradigm comparison on the same ChartQA 8B SFT student. The two-pass row is the headline force-continue paradigm; the in-place rows intervene inside the model's own autoregressive stream (no re-prompt, no added instruction, the conclusion line is never supplied). Readout is identical across paradigms. Reported alongside, not replacing, the force-continue results. Pre-registration: docs/preregistration_p3_inplace.md.

| paradigm | condition | n_corrupt | base acc | follow 95% CI | snap 95% CI | flip 95% CI | clean-agree |
|---|---|---:|---:|---:|---:|---:|---:|
| twopass (force-continue) | present (re-prompt) | 307 | 0.785 | 0.036 [0.020,0.063] | 0.935 [0.902,0.957] | 0.098 [0.069,0.136] | - |
| inplace (single stream) | present (locked-visual) | 307 | 0.785 | 0.094 [0.067,0.132] | 0.733 [0.681,0.779] | 0.293 [0.245,0.346] | 287/307 (0.935) |
| inplace (single stream) | masked-B (no-visual) | 23 | 0.070 | 0.000 [0.000,0.143] | 0.478 [0.292,0.670] | 0.609 [0.408,0.778] | 12/23 (0.522) |

_Interpretation rule (pre-registered): CONVERGENCE if present in-place follow Wilson upper bound < 0.10 and in-place corrupt-flip not materially above two-pass corrupt-flip._

## Selection Flow and Probe Strata

The probe battery regenerates an original CoT and keeps cases with a nonempty emitted CoT and a correct extracted final answer. The answer-eval columns come from the separate n=400 SFT evaluation JSONL; the small ChartQA 8B disagreement between answer-eval and probe-generation passes is retained as loss/wrong probe strata.

| cell | cond | raw | answer-eval base correct | answer-eval SFT correct | probe eligible | corrupt usable | nonprobe/unusable | eval-correct missing probe | probe not SFT-correct in eval |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ChartQA 8B | present | 400 | 285 | 311 | 321 | 321 | 79 | 10 | 20 |
| ChartQA 8B | masked | 400 | 285 | 311 | 321 | 321 | 79 | 10 | 20 |
| ChartQA 32B | present | 400 | 290 | 318 | 318 | 316 | 84 | 0 | 0 |
| ChartQA 32B | masked | 400 | 290 | 318 | 318 | 316 | 84 | 0 | 0 |
| TabMWP 8B | present | 400 | 346 | 385 | 385 | 385 | 15 | 0 | 0 |
| TabMWP 8B | masked | 400 | 346 | 385 | 385 | 385 | 15 | 0 | 0 |
| TabMWP 32B | present | 400 | 334 | 389 | 389 | 389 | 11 | 0 | 0 |
| TabMWP 32B | masked | 400 | 334 | 389 | 389 | 389 | 11 | 0 | 0 |

| cell | cond | answer-eval strata stable/gain/loss/wrong | probe-kept stable/gain/loss/wrong | stable follow | gain follow | loss follow | wrong follow | gain snap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ChartQA 8B | present | 265/46/20/69 | 262/39/10/10 | 9/262 (0.034) | 0/39 (0.000) | 0/10 (0.000) | 1/10 (0.100) | 30/39 (0.769) |
| ChartQA 8B | masked | 265/46/20/69 | 262/39/10/10 | 30/262 (0.115) | 2/39 (0.051) | 0/10 (0.000) | 0/10 (0.000) | 30/39 (0.769) |
| ChartQA 32B | present | 281/37/9/73 | 281/37/0/0 | 5/281 (0.018) | 0/37 (0.000) | NA | NA | 37/37 (1.000) |
| ChartQA 32B | masked | 281/37/9/73 | 281/37/0/0 | 18/281 (0.064) | 0/37 (0.000) | NA | NA | 35/37 (0.946) |
| TabMWP 8B | present | 345/40/1/14 | 345/40/0/0 | 3/345 (0.009) | 0/40 (0.000) | NA | NA | 40/40 (1.000) |
| TabMWP 8B | masked | 345/40/1/14 | 345/40/0/0 | 32/345 (0.093) | 2/40 (0.050) | NA | NA | 36/40 (0.900) |
| TabMWP 32B | present | 332/57/2/9 | 332/57/0/0 | 6/332 (0.018) | 0/57 (0.000) | NA | NA | 56/57 (0.982) |
| TabMWP 32B | masked | 332/57/2/9 | 332/57/0/0 | 15/332 (0.045) | 0/57 (0.000) | NA | NA | 57/57 (1.000) |

### Sensitivity Bounds

Legacy battery artifacts retain eligible probe details but not raw generated CoTs for non-eligible cases, so the main sensitivity analysis brackets the unobserved non-probe/parse-failure mass.

| cell | cond | observed probe follow | all-raw lower | all-raw upper | SFT-correct lower | SFT-correct upper |
|---|---|---:|---:|---:|---:|---:|
| ChartQA 8B | present | 10/321 (0.031) | 10/400 (0.025) | 89/400 (0.223) | 9/311 (0.029) | 19/311 (0.061) |
| ChartQA 8B | masked | 32/321 (0.100) | 32/400 (0.080) | 111/400 (0.278) | 32/311 (0.103) | 42/311 (0.135) |
| ChartQA 32B | present | 5/316 (0.016) | 5/400 (0.013) | 89/400 (0.223) | 5/318 (0.016) | 7/318 (0.022) |
| ChartQA 32B | masked | 18/316 (0.057) | 18/400 (0.045) | 102/400 (0.255) | 18/318 (0.057) | 20/318 (0.063) |
| TabMWP 8B | present | 3/385 (0.008) | 3/400 (0.007) | 18/400 (0.045) | 3/385 (0.008) | 3/385 (0.008) |
| TabMWP 8B | masked | 34/385 (0.088) | 34/400 (0.085) | 49/400 (0.122) | 34/385 (0.088) | 34/385 (0.088) |
| TabMWP 32B | present | 6/389 (0.015) | 6/400 (0.015) | 17/400 (0.043) | 6/389 (0.015) | 6/389 (0.015) |
| TabMWP 32B | masked | 15/389 (0.039) | 15/400 (0.037) | 26/400 (0.065) | 15/389 (0.039) | 15/389 (0.039) |

## Gain-Subset Probe

| cell | answer-eval gains | probe-kept gained n | gained flip | gained snap 95% CI | gained follow 95% CI | retained follow 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| ChartQA 8B | 46 | 39 | 11/39 (0.282) | 0.769 [0.617,0.874] | 0.000 [0.000,0.090] | 0.034 [0.018,0.064] |
| ChartQA 32B | 37 | 37 | 5/37 (0.135) | 1.000 [0.906,1.000] | 0.000 [0.000,0.094] | 0.018 [0.008,0.041] |
| TabMWP 8B | 40 | 40 | 0/40 (0.000) | 1.000 [0.912,1.000] | 0.000 [0.000,0.088] | 0.009 [0.003,0.025] |
| TabMWP 32B | 57 | 57 | 1/57 (0.018) | 0.982 [0.907,0.997] | 0.000 [0.000,0.063] | 0.018 [0.008,0.039] |

## FinQA Targeted Operand Probe

The `answer-eval gains` column is the standalone n=400 accuracy-eval gain count. `probe-kept gained n` is the base-wrong -> CoT-SFT-right subset that also survived CoT/probe eligibility. FinQA is not reported as a gain subset in the current artifacts because the targeted battery is keyed by gold-program operand eligibility, not by matched base-wrong -> SFT-right eval partitions.

| arm | n_targeted | operand follow 95% CI | operand snap 95% CI | consistent-conclusion follow 95% CI |
|---|---:|---:|---:|---:|
| FinQA 8B curriculum | 172 | 0/172 (0.000) [0.000,0.022] | 172/172 (1.000) [0.978,1.000] | 165/172 (0.959) [0.918,0.980] |
| FinQA 32B curriculum | 175 | 0/175 (0.000) [0.000,0.021] | 175/175 (1.000) [0.979,1.000] | 170/175 (0.971) [0.935,0.988] |
