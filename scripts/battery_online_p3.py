#!/usr/bin/env python
"""P3-1 — generation-time (single-stream) intervention battery.

Addresses the Devil's-Advocate objection that the headline force-continue probe is a
TWO-PASS re-prompt (new prompt + added instruction + second independent forward), so a
low present-image follow-rate could mean "the model re-read the image on pass 2" rather
than "the written chain is not load-bearing."

This probe stays inside the model's OWN autoregressive stream (C1 of todo/06.md):

  Pass-Generate    greedy-decode the model's own CoT+ANSWER from the normal prompt.
  online_corrupt   re-feed the model's own greedy tokens up to the boundary just before a
                   selected numeric token, substitute v_inj (= 2v+7, same rule as the
                   two-pass `corrupt`), and let the model CONTINUE generating the rest of
                   the chain and its answer. No conclusion line is supplied; no added
                   instruction. Re-feeding the model's own greedy ids is numerically the
                   same as KV-cache continuation, but uses generate() so Qwen3-VL mRoPE is
                   handled by the library instead of hand-rolled.
  online_clean     paradigm-validity control: re-feed up to and INCLUDING v_true and
                   continue. Must reproduce base_ans at a high rate.

Readout matches the two-pass battery exactly (only the paradigm changes):
  snap   = continuation answer == gold (c_true)
  follow = continuation answer == injected v_inj
  other  = neither
  flip   = continuation answer != base_ans

Conditions: present (locked-visual; image throughout) and, with --mask-image,
masked-B (no-visual; image absent throughout). See docs/preregistration_p3_inplace.md.

Run in env `mbe-up` via the env python directly (NOT `conda run`):
  conda activate mbe-up && python scripts/battery_online_p3.py \
    --base /home/gpus/models/Qwen3-VL-8B-Instruct \
    --adapter data/distill/poc/lora_8b_chartqa --scale-tag 8b \
    --out data/distill/poc/battery_p3_online_chartqa8b_present.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re as _re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"; os.environ["no_proxy"] = "*"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.battery_n400 import (  # noqa: E402
    NUM_RE,
    _corrupt_replacement,
    extract_answer,
    relaxed_match,
)


def choose_target(base_cot: str, rng: random.Random):
    """Pick one numeric token to corrupt (same selector family as the two-pass corrupt arm:
    Random(0), uniform over regex numeric matches). Returns (v_true, v_inj, span) or Nones."""
    nums = list(NUM_RE.finditer(base_cot))
    if not nums:
        return None, None, None
    m = rng.choice(nums)
    v_inj = _corrupt_replacement(m.group(0))
    if v_inj is None:
        return None, None, None
    return m.group(0), v_inj, (m.start(), m.end())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--scale-tag", required=True)
    ap.add_argument("--quant", choices=["nf4", "none"], default="nf4")
    ap.add_argument("--dump", default="data/distill/chartqa/test_cases_400.jsonl")
    ap.add_argument("--img-dir", default="/home/gpus/mbe_data/chartqa_test_images")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--cont-new", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--mask-image", action="store_true",
                    help="masked-B: image absent throughout (Pass-Generate AND continuation).")
    args = ap.parse_args()
    rng = random.Random(0)

    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from peft import PeftModel

    USER_INSTR = "Solve step by step, end with 'ANSWER: <final>'.\n\nQuestion: "
    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    tok = processor.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id

    quant_cfg = None
    if args.quant == "nf4":
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_use_double_quant=True,
                                       bnb_4bit_compute_dtype=torch.bfloat16)
    use_adapter = bool(args.adapter) and str(args.adapter).lower() != "none"
    dev0 = torch.device("cuda:0")

    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base, quantization_config=quant_cfg, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    if use_adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval(); model.config.use_cache = True
    print(f"loaded {'base+adapter' if use_adapter else 'BASE-ONLY'} ({args.quant}) in {time.time()-t0:.0f}s",
          flush=True)

    prog = Path("data/distill/poc/logs/online_p3_progress.txt"); prog.parent.mkdir(parents=True, exist_ok=True)

    def prompt_enc(c):
        """Tokenized prompt for one case (with or without the image)."""
        txt = {"type": "text", "text": USER_INSTR + c["q"]}
        content = [txt] if args.mask_image else [{"type": "image", "image": c["img"]}, txt]
        return processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                             return_dict=True, add_generation_prompt=True,
                                             return_tensors="pt")

    @torch.no_grad()
    def gen_from_encs(encs, max_new, tag):
        """Left-pad a list of pre-tokenized encs (dicts with input_ids[1,L], optional pixel/grid/mmtt)
        and greedy-generate. Returns list of decoded continuations (new tokens only)."""
        res = [None] * len(encs)
        t_s = time.time()
        for i in range(0, len(encs), args.batch_size):
            chunk = encs[i:i + args.batch_size]
            B = len(chunk)
            maxlen = max(e["input_ids"].shape[1] for e in chunk)
            ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
            att = torch.zeros((B, maxlen), dtype=torch.long)
            has_mmtt = "mm_token_type_ids" in chunk[0]
            mmtt = torch.zeros((B, maxlen), dtype=torch.long) if has_mmtt else None
            pix, grid = [], []
            for b, e in enumerate(chunk):
                L = e["input_ids"].shape[1]
                ids[b, maxlen - L:] = e["input_ids"][0]
                att[b, maxlen - L:] = 1
                if has_mmtt:
                    mmtt[b, maxlen - L:] = e["mm_token_type_ids"][0]
                if "pixel_values" in e:
                    pix.append(e["pixel_values"])
                if "image_grid_thw" in e:
                    grid.append(e["image_grid_thw"])
            batch = {"input_ids": ids.to(dev0), "attention_mask": att.to(dev0)}
            if has_mmtt:
                batch["mm_token_type_ids"] = mmtt.to(dev0)
            if pix:
                batch["pixel_values"] = torch.cat(pix, dim=0).to(dev0)
            if grid:
                batch["image_grid_thw"] = torch.cat(grid, dim=0).to(dev0)
            g = model.generate(**batch, max_new_tokens=max_new, do_sample=False, pad_token_id=pad_id)
            for b in range(B):
                res[i + b] = tok.decode(g[b][maxlen:], skip_special_tokens=True)
            with open(prog, "a") as pf:
                pf.write(f"[{args.scale_tag} mask={args.mask_image}] {tag} "
                         f"{min(i+args.batch_size,len(encs))}/{len(encs)} elapsed={time.time()-t_s:.0f}s\n")
        return res

    @torch.no_grad()
    def gen_ids_from_encs(encs, max_new, tag):
        """Same as gen_from_encs but returns the generated token ids (1D long tensors, new tokens only)."""
        res = [None] * len(encs)
        t_s = time.time()
        for i in range(0, len(encs), args.batch_size):
            chunk = encs[i:i + args.batch_size]
            B = len(chunk)
            maxlen = max(e["input_ids"].shape[1] for e in chunk)
            ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
            att = torch.zeros((B, maxlen), dtype=torch.long)
            has_mmtt = "mm_token_type_ids" in chunk[0]
            mmtt = torch.zeros((B, maxlen), dtype=torch.long) if has_mmtt else None
            pix, grid = [], []
            for b, e in enumerate(chunk):
                L = e["input_ids"].shape[1]
                ids[b, maxlen - L:] = e["input_ids"][0]
                att[b, maxlen - L:] = 1
                if has_mmtt:
                    mmtt[b, maxlen - L:] = e["mm_token_type_ids"][0]
                if "pixel_values" in e:
                    pix.append(e["pixel_values"])
                if "image_grid_thw" in e:
                    grid.append(e["image_grid_thw"])
            batch = {"input_ids": ids.to(dev0), "attention_mask": att.to(dev0)}
            if has_mmtt:
                batch["mm_token_type_ids"] = mmtt.to(dev0)
            if pix:
                batch["pixel_values"] = torch.cat(pix, dim=0).to(dev0)
            if grid:
                batch["image_grid_thw"] = torch.cat(grid, dim=0).to(dev0)
            g = model.generate(**batch, max_new_tokens=max_new, do_sample=False, pad_token_id=pad_id)
            for b in range(B):
                res[i + b] = g[b][maxlen:].detach().cpu()
            with open(prog, "a") as pf:
                pf.write(f"[{args.scale_tag} mask={args.mask_image}] {tag} "
                         f"{min(i+args.batch_size,len(encs))}/{len(encs)} elapsed={time.time()-t_s:.0f}s\n")
        return res

    # ---- load cases ----
    rows = [json.loads(l) for l in open(args.dump) if l.strip()][:args.n]
    cases = []
    for r in rows:
        cid = r["case_id"]; idx = cid.rsplit("-", 1)[-1]
        prefix = cid.rsplit("-", 1)[0] or "chartqa"
        imgp = Path(args.img_dir) / f"{prefix}_{idx}.png"
        if imgp.exists():
            cases.append({"cid": cid, "q": str(r["question"]), "gold": str(r["gold"]),
                          "img": Image.open(imgp).convert("RGB")})
    print(f"online-p3 cases: {len(cases)} (mask_image={args.mask_image}, scale={args.scale_tag})", flush=True)

    # ---- Pass-Generate: model's own greedy CoT+ANSWER stream ----
    base_encs = [prompt_enc(c) for c in cases]
    gen_ids = gen_ids_from_encs(base_encs, args.max_new, "base")
    kept = []
    base_correct = 0
    for c, enc, g in zip(cases, base_encs, gen_ids):
        full = tok.decode(g, skip_special_tokens=True)
        base_ans = extract_answer(full)
        cot_raw = full.split("ANSWER:")[0]
        base_cot = cot_raw.strip()
        if relaxed_match(base_ans, c["gold"]):
            base_correct += 1
        if base_cot and relaxed_match(base_ans, c["gold"]):
            kept.append({**c, "enc": enc, "gen_ids": g, "full": full, "cot_raw": cot_raw,
                         "base_cot": base_cot, "base_ans": base_ans})
    base_acc = base_correct / len(cases) if cases else 0.0
    print(f"base free-form acc: {base_correct}/{len(cases)} = {base_acc:.3f}; probe eligible {len(kept)}",
          flush=True)

    # ---- locate target token boundary inside each kept stream ----
    def build_continuation_enc(k, include_true: bool):
        """Re-feed model's own tokens up to the boundary just before the chosen number, then append
        either v_inj (online_corrupt) or v_true (online_clean) as text and let generate() continue.

        Returns (enc_dict, meta) or (None, reason). enc_dict carries prompt image features +
        the model's own prefix token ids + the substituted-number text tokens.
        """
        v_true, v_inj, span = choose_target(k["base_cot"], rng)
        if v_true is None:
            return None, "no_numeric_target"
        # char offset of the number in the UNstripped decoded text (cot_raw)
        lead = len(k["cot_raw"]) - len(k["cot_raw"].lstrip())
        char_start = lead + span[0]
        g = k["gen_ids"]
        # largest prefix token count whose decoded text fits entirely before char_start
        # (incremental decode; robust to BPE merges via the leftover re-feed below)
        tok_start = 0
        prefix_text = ""
        for t in range(1, len(g) + 1):
            cand = tok.decode(g[:t], skip_special_tokens=True)
            if len(cand) <= char_start:
                tok_start = t
                prefix_text = cand
            else:
                break
        if tok_start == 0:
            return None, "boundary_at_start"
        leftover = k["full"][len(prefix_text):char_start]
        sub_val = v_true if include_true else v_inj
        extra_ids = tok(leftover + str(sub_val), add_special_tokens=False)["input_ids"]
        prefix_ids = g[:tok_start].to(torch.long)
        extra = torch.tensor(extra_ids, dtype=torch.long)
        prompt_ids = k["enc"]["input_ids"][0].to(torch.long)
        full_ids = torch.cat([prompt_ids, prefix_ids, extra]).unsqueeze(0)
        enc = {"input_ids": full_ids}
        if "pixel_values" in k["enc"]:
            enc["pixel_values"] = k["enc"]["pixel_values"]
        if "image_grid_thw" in k["enc"]:
            enc["image_grid_thw"] = k["enc"]["image_grid_thw"]
        if "mm_token_type_ids" in k["enc"]:
            prompt_mmtt = k["enc"]["mm_token_type_ids"][0].to(torch.long)
            tail_mmtt = torch.zeros(prefix_ids.numel() + extra.numel(), dtype=torch.long)
            enc["mm_token_type_ids"] = torch.cat([prompt_mmtt, tail_mmtt]).unsqueeze(0)
        meta = {"v_true": v_true, "v_inj": v_inj, "tok_start": tok_start, "span": span}
        return enc, meta

    corrupt_encs, corrupt_meta, corrupt_idx = [], [], []
    clean_encs, clean_idx = [], []
    skips = {}
    for i, k in enumerate(kept):
        enc, meta = build_continuation_enc(k, include_true=False)
        if enc is None:
            skips[meta] = skips.get(meta, 0) + 1
            continue
        corrupt_encs.append(enc); corrupt_meta.append(meta); corrupt_idx.append(i)
        # clean control reuses the SAME chosen number (re-seed not needed: deterministic per-case
        # rebuild — rng already advanced once for corrupt, so rebuild clean from stored meta span)
    # build clean control deterministically from the corrupt meta (same span, inject v_true)
    for i, meta in zip(corrupt_idx, corrupt_meta):
        k = kept[i]
        # reconstruct enc with v_true at the stored boundary
        v_true = meta["v_true"]
        span = meta["span"]
        lead = len(k["cot_raw"]) - len(k["cot_raw"].lstrip())
        char_start = lead + span[0]
        g = k["gen_ids"]
        tok_start = meta["tok_start"]
        prefix_text = tok.decode(g[:tok_start], skip_special_tokens=True)
        leftover = k["full"][len(prefix_text):char_start]
        extra_ids = tok(leftover + str(v_true), add_special_tokens=False)["input_ids"]
        prefix_ids = g[:tok_start].to(torch.long)
        extra = torch.tensor(extra_ids, dtype=torch.long)
        prompt_ids = k["enc"]["input_ids"][0].to(torch.long)
        full_ids = torch.cat([prompt_ids, prefix_ids, extra]).unsqueeze(0)
        enc = {"input_ids": full_ids}
        if "pixel_values" in k["enc"]:
            enc["pixel_values"] = k["enc"]["pixel_values"]
        if "image_grid_thw" in k["enc"]:
            enc["image_grid_thw"] = k["enc"]["image_grid_thw"]
        if "mm_token_type_ids" in k["enc"]:
            prompt_mmtt = k["enc"]["mm_token_type_ids"][0].to(torch.long)
            tail_mmtt = torch.zeros(prefix_ids.numel() + extra.numel(), dtype=torch.long)
            enc["mm_token_type_ids"] = torch.cat([prompt_mmtt, tail_mmtt]).unsqueeze(0)
        clean_encs.append(enc); clean_idx.append(i)

    print(f"continuation encs: corrupt={len(corrupt_encs)} clean={len(clean_encs)} skips={skips}", flush=True)

    # ---- run continuations ----
    corrupt_out = gen_from_encs(corrupt_encs, args.cont_new, "corrupt") if corrupt_encs else []
    clean_out = gen_from_encs(clean_encs, args.cont_new, "clean") if clean_encs else []

    # ---- score ----
    def readout(ans, k, v_inj):
        if relaxed_match(ans, k["gold"]):
            return "snap"
        if v_inj is not None and relaxed_match(ans, v_inj):
            return "follow"
        return "other"

    snap = follow = other = flips = acc = 0
    details = []
    for enc_i, (i, meta, out) in enumerate(zip(corrupt_idx, corrupt_meta, corrupt_out)):
        k = kept[i]
        ans = extract_answer(out)
        r = readout(ans, k, meta["v_inj"])
        snap += r == "snap"; follow += r == "follow"; other += r == "other"
        flip = not relaxed_match(ans, k["base_ans"])
        flips += flip
        acc += relaxed_match(ans, k["gold"])
        details.append({"case_id": k["cid"], "paradigm": "inplace",
                        "condition": "masked-B" if args.mask_image else "present",
                        "v_true": meta["v_true"], "v_inj": meta["v_inj"], "tok_start": meta["tok_start"],
                        "base_ans": k["base_ans"], "gold": k["gold"], "inj_ans": ans,
                        "readout": r, "flip": bool(flip)})
    n_corr = len(corrupt_idx)

    clean_agree = 0
    clean_details = []
    for i, out in zip(clean_idx, clean_out):
        k = kept[i]
        ans = extract_answer(out)
        agree = relaxed_match(ans, k["base_ans"])
        clean_agree += agree
        clean_details.append({"case_id": k["cid"], "clean_ans": ans, "base_ans": k["base_ans"],
                              "agree": bool(agree)})
    n_clean = len(clean_idx)

    summary = {
        "scale": args.scale_tag, "paradigm": "inplace",
        "condition": "masked-B" if args.mask_image else "present",
        "mask_image": bool(args.mask_image),
        "n_cases": len(cases), "base_correct": base_correct, "base_acc": base_acc,
        "n_eval": len(kept), "adapter": args.adapter if use_adapter else None,
        "online_corrupt": {
            "n": n_corr, "snap": snap, "follow": follow, "other": other, "flips": flips,
            "snap_rate": snap / n_corr if n_corr else 0.0,
            "follow_rate": follow / n_corr if n_corr else 0.0,
            "flip_rate": flips / n_corr if n_corr else 0.0,
            "acc_after": acc / n_corr if n_corr else 0.0,
        },
        "online_clean_control": {
            "n": n_clean, "agree_base": clean_agree,
            "agree_rate": clean_agree / n_clean if n_clean else 0.0,
        },
        "skips": {str(k): v for k, v in skips.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"summary": summary, "details": details, "clean_details": clean_details},
        ensure_ascii=False, indent=2))
    sha = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()[:12]
    print(f"\n=== P3-1 ONLINE [{args.scale_tag} {'masked-B' if args.mask_image else 'present'}] "
          f"n_eval={len(kept)} ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"out={args.out} sha256={sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
