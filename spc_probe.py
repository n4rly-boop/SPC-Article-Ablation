#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPC self-evolution probe on NATURAL errors (ProcessBench).
================================================================================
Reproduces the round-0 -> round-2 self-evolution gain of SPC
(Chen et al., "SPC: Evolving Self-Play Critic via Adversarial Games for
LLM Reasoning", NeurIPS 2025, arXiv:2504.19162) on the *natural* errors of
ProcessBench (Zheng et al., arXiv:2412.06559), AND decomposes that gain into:

    * erroneous-step accuracy  (recall on REAL model errors)
    * correct-step accuracy    (= 1 - false-positive rate on correct steps)

HYPOTHESIS BEING TESTED
-----------------------
SPC trains the critic via adversarial self-play, where the sneaky generator
*injects* an error into (almost) every probe. A critic trained on a steady diet
of "there is an error here" may shift its prior toward "Incorrect" and become
over-sensitive. If so, going Critic-0 -> Critic-2 should raise erroneous-step
recall while LOWERING correct-step accuracy on real, naturally-occurring steps.
That asymmetry is evidence of distribution shift / over-sensitivity and directly
motivates the "develop the paper" section of the AIRI report.

WHAT THIS SCRIPT DOES
---------------------
1. Loads ProcessBench (Qwen/ProcessBench) from the HuggingFace Hub.
2. Converts each item into per-step (problem, partial_solution, last_step, label)
   examples, matching SPC's binary per-step critic formulation.
3. Runs BOTH released critics (judge/SPC-Critic-0, judge/SPC-Critic-2) with SPC's
   exact prompt + greedy decoding, and parses <Answer>Correct/Incorrect</Answer>.
4. Prints a side-by-side comparison (overall + per subset) and saves raw outputs.

DESIGN NOTES (be honest about these in the report)
---------------------------------------------------
* The original eval uses vLLM on 2 GPUs. This script uses plain `transformers`
  on Apple-Silicon MPS, one example at a time. No CUDA, no vLLM, no training.
* SPC's own preprocessed ProcessBench eval files live on the authors' OneDrive.
  Here we reconstruct the per-step examples directly from raw ProcessBench, so
  ABSOLUTE numbers may differ slightly from the paper; the round-0 vs round-2
  COMPARISON is the point.
* The sneaky-generator checkpoint was NOT released, so we cannot reproduce SPC's
  own synthetic errors. See the comment block at the bottom for how to add a
  synthetic-error probe with a second model if you want that extension.

USAGE
-----
    pip install "torch>=2.4" transformers datasets accelerate huggingface_hub
    # quick smoke test of the data/metric logic, no model, no network:
    python spc_probe.py --selftest
    # tiny real run first (a few minutes), then scale up:
    python spc_probe.py --max-per-subset 5
    python spc_probe.py --max-per-subset 40          # fuller run
    python spc_probe.py --subsets gsm8k math         # subset of subsets

On a 24 GB Apple-Silicon Mac, one 7B critic in float16 (~15 GB) fits with room
for the KV cache; the script loads one model at a time and frees it before the
next. If you hit out-of-memory: close other apps, lower --max-new-tokens, or
reduce --max-per-subset. bitsandbytes 4-bit does NOT work on Mac; for 4-bit use
MLX/llama.cpp instead (out of scope here).
"""

import argparse
import gc
import json
import re
import sys
import time
from collections import defaultdict

# ----------------------------------------------------------------------------
# SPC critic prompt, copied verbatim from eval/infer_batch.py so the released
# checkpoints behave exactly as intended.
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful critic. Given a math Problem, a Partial Solution, the "
    "current Last Step of the solution, You need to provide a critique for the "
    "correctness of the Last Step.\n\n"
    "You need to response a step-by-step analysis:\n"
    "1. Analyzing the general thought of the Partial Solution. \n"
    "2. Critique. You should write a brief critique here. This part should also "
    "maintain logical coherence with the summary of the general thought of the "
    "Partial Solution.\n"
    "3. Conclusion. At the end of the response, output <Answer>Correct</Answer> "
    "or <Answer>Incorrect</Answer> to represent the correctness of the Last Step."
)

PROMPT_TEMPLATE = (
    "## Problem\n{problem}\n\n"
    "## Partial Solution\n{partial_solution}\n\n"
    "## Last Step\n{last_step}"
)

CHECKPOINTS = [
    ("Critic-0 (SFT)", "judge/SPC-Critic-0"),
    ("Critic-2 (RL)",  "judge/SPC-Critic-2"),
]

DEFAULT_SUBSETS = ["gsm8k", "math", "olympiadbench", "omnimath"]
STEP_SEP = "\n\n"

# human_label convention follows SPC: +1 = correct step, -1 = erroneous step.
CORRECT, ERRONEOUS, INVALID = 1, -1, 0


# ----------------------------------------------------------------------------
# 1. Data conversion: ProcessBench item -> per-step examples
# ----------------------------------------------------------------------------
def build_per_step_examples(dataset, subset, max_items):
    """Turn ProcessBench items into SPC-style per-step probes.

    ProcessBench item schema (HF Qwen/ProcessBench):
        problem: str
        steps:   list[str]
        label:   int   # 0-based index of the FIRST erroneous step, or -1 if all correct
        generator, final_answer_correct, id ...

    For an errored item (label = L >= 0):
        * erroneous probe: steps[:L] as context, steps[L] as last step  -> -1
        * correct  probe (if L >= 1): steps[:L-1] context, steps[L-1]   -> +1
    For an all-correct item (label = -1):
        * correct probe: steps[:-1] context, steps[-1] as last step     -> +1
    """
    examples = []
    n = 0
    for item in dataset:
        if max_items is not None and n >= max_items:
            break
        steps = item.get("steps") or []
        if not steps:
            continue
        label = item.get("label", -1)
        used = False

        if label == -1:
            # all steps correct -> use the final step as a (hard) correct probe
            ctx, last = steps[:-1], steps[-1]
            examples.append(_mk(item, subset, ctx, last, CORRECT))
            used = True
        elif 0 <= label < len(steps):
            # first erroneous step
            ctx, last = steps[:label], steps[label]
            examples.append(_mk(item, subset, ctx, last, ERRONEOUS))
            # a known-correct step just before the error (if any)
            if label >= 1:
                ctx_c, last_c = steps[:label - 1], steps[label - 1]
                examples.append(_mk(item, subset, ctx_c, last_c, CORRECT))
            used = True
        else:
            # malformed label -> skip
            continue

        if used:
            n += 1
    return examples


def _mk(item, subset, ctx_steps, last_step, human_label):
    partial = STEP_SEP.join(ctx_steps).strip()
    if partial == "":
        partial = "Let's solve this problem."  # SPC's behavior for empty context
    return {
        "id": str(item.get("id", "")),
        "data_type": subset,
        "problem": item["problem"],
        "partial_solution": partial,
        "last_step": last_step,
        "human_label": human_label,
    }


# ----------------------------------------------------------------------------
# 2. Answer parsing (tolerant of whitespace + case around SPC's <Answer>...</Answer>)
# ----------------------------------------------------------------------------
_ANSWER_RE = re.compile(r"<Answer>\s*(correct|incorrect)\s*</Answer>", re.IGNORECASE)


def parse_answer(response):
    """Return +1 (Correct), -1 (Incorrect), or 0 (no valid tag found)."""
    m = _ANSWER_RE.search(response or "")
    if not m:
        return INVALID
    return CORRECT if m.group(1).lower() == "correct" else ERRONEOUS


# ----------------------------------------------------------------------------
# 3. Metrics (mirrors filter_process_bench_critique + filter_critique)
# ----------------------------------------------------------------------------
def summarize(records):
    """records: list of dicts with keys data_type, human_label, pred."""
    total = len(records)
    valid = [r for r in records if r["pred"] != INVALID]
    valid_ratio = len(valid) / total if total else 0.0

    def acc(items):
        return (sum(1 for r in items if r["pred"] == r["human_label"]) / len(items)) if items else float("nan")

    correct_items = [r for r in valid if r["human_label"] == CORRECT]
    error_items = [r for r in valid if r["human_label"] == ERRONEOUS]
    c_acc = acc(correct_items)
    e_acc = acc(error_items)

    def hmean(a, b):
        if a != a or b != b or (a + b) == 0:  # NaN guard
            return float("nan")
        return 2 * a * b / (a + b)

    # per-subset combined accuracy (SPC's headline metric), then macro-average
    per_subset = {}
    by_type = defaultdict(list)
    for r in valid:
        by_type[r["data_type"]].append(r)
    for dt, items in by_type.items():
        per_subset[dt] = acc(items)
    macro = (sum(v for v in per_subset.values() if v == v) / len(per_subset)) if per_subset else float("nan")

    return {
        "total": total,
        "valid_ratio": valid_ratio,
        "n_correct": len(correct_items),
        "n_error": len(error_items),
        "correct_step_acc": c_acc,
        "erroneous_step_acc": e_acc,
        "average_acc": (c_acc + e_acc) / 2 if (c_acc == c_acc and e_acc == e_acc) else float("nan"),
        "harmonic_mean": hmean(c_acc, e_acc),
        "per_subset_combined_acc": per_subset,
        "macro_avg_combined_acc": macro,
    }


def _fmt(x):
    return "  nan" if x != x else f"{x:6.3f}"


def print_summary(name, s):
    print(f"\n=== {name} ===")
    print(f"  examples: {s['total']}  (correct: {s['n_correct']}, error: {s['n_error']})"
          f"   valid-tag ratio: {s['valid_ratio']:.3f}")
    print(f"  erroneous-step accuracy (recall on real errors): {_fmt(s['erroneous_step_acc'])}")
    print(f"  correct-step  accuracy (1 - false-positive rate): {_fmt(s['correct_step_acc'])}")
    print(f"  average / harmonic mean:                          {_fmt(s['average_acc'])} / {_fmt(s['harmonic_mean'])}")
    print(f"  macro-avg combined acc (SPC headline metric):     {_fmt(s['macro_avg_combined_acc'])}")
    if s["per_subset_combined_acc"]:
        print("  per-subset combined acc:")
        for dt, v in s["per_subset_combined_acc"].items():
            print(f"      {dt:>14}: {_fmt(v)}")


def print_comparison(results):
    """results: list of (name, summary) in checkpoint order."""
    if len(results) < 2:
        return
    (n0, s0), (n1, s1) = results[0], results[1]
    print("\n" + "=" * 70)
    print(f"SELF-EVOLUTION DELTA  ({n0}  ->  {n1})")
    print("=" * 70)

    def delta(key, label):
        a, b = s0[key], s1[key]
        d = b - a
        print(f"  {label:<46} {_fmt(a)} -> {_fmt(b)}   (Δ {d:+.3f})")

    delta("erroneous_step_acc", "erroneous-step recall (real errors)")
    delta("correct_step_acc", "correct-step accuracy (real correct steps)")
    delta("average_acc", "average accuracy")
    delta("harmonic_mean", "harmonic mean")
    print("-" * 70)
    de = s1["erroneous_step_acc"] - s0["erroneous_step_acc"]
    dc = s1["correct_step_acc"] - s0["correct_step_acc"]
    if de == de and dc == dc:
        if de > 0 and dc < 0:
            print("  READING: RL self-play improved error recall but HURT accuracy on")
            print("  correct steps -> over-sensitivity signature (supports the distribution-")
            print("  shift critique). Quantify the trade-off in the report.")
        elif de > 0 and dc >= 0:
            print("  READING: RL self-play improved error recall WITHOUT a correct-step")
            print("  penalty -> no over-sensitivity on this sample. Note this honestly.")
        else:
            print("  READING: unexpected pattern on this sample; inspect raw outputs and")
            print("  consider a larger --max-per-subset before drawing conclusions.")


# ----------------------------------------------------------------------------
# 4. Model loading + inference (transformers on MPS)
# ----------------------------------------------------------------------------
def load_critic(path, device, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  loading tokenizer + model: {path}  (dtype={dtype}, device={device})")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=getattr(torch, dtype), low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    return tok, model


def _ckpt_slug(name):
    # "Critic-0 (SFT)" -> "Critic-0"
    return name.split()[0]


def _per_ckpt_path(base, name):
    slug = _ckpt_slug(name)
    if base.endswith(".jsonl"):
        return base[:-6] + f".{slug}.jsonl"
    return base + f".{slug}.jsonl"


def _load_existing(path):
    """Read previously-written records; tolerate truncated trailing line from crash."""
    import os
    records, seen = [], set()
    if not os.path.exists(path):
        return records, seen
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r["id"], r["data_type"], r["human_label"])
            if key in seen:
                continue
            seen.add(key)
            records.append(r)
    return records, seen


def run_critic(tok, model, examples, device, max_new_tokens, write_rec=None, skip_keys=None):
    """Run greedy critic over examples. Per-probe write+flush via write_rec for crash safety.

    write_rec(rec): called immediately after each generation; should fsync/flush.
    skip_keys: set of (id, data_type, human_label) tuples already done; resumed from disk.
    """
    import torch
    skip_keys = skip_keys or set()
    todo = [e for e in examples if (e["id"], e["data_type"], e["human_label"]) not in skip_keys]
    total = len(todo)
    if skip_keys:
        print(f"    resume: {len(skip_keys)} probes already on disk, {total} remaining")
    records = []
    t0 = time.time()
    for i, ex in enumerate(todo, 1):
        user = PROMPT_TEMPLATE.format(
            problem=ex["problem"],
            partial_solution=ex["partial_solution"],
            last_step=ex["last_step"],
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy, matches SPC temperature=0.0
                pad_token_id=tok.eos_token_id,
            )
        resp = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_answer(resp)
        rec = {
            "id": ex["id"], "data_type": ex["data_type"],
            "human_label": ex["human_label"], "pred": pred,
            "response": resp,
        }
        records.append(rec)
        if write_rec is not None:
            write_rec(rec)  # flushes -> survives crash
        if i % 10 == 0 or i == total:
            rate = i / (time.time() - t0)
            print(f"    [{i}/{total}]  {rate:.2f} ex/s", end="\r", flush=True)
    print()
    return records


def free_model(model):
    import torch
    del model
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ----------------------------------------------------------------------------
# Self-test: verify conversion + parsing + metrics with NO model / NO network
# ----------------------------------------------------------------------------
def selftest():
    print("Running self-test of data/parse/metric logic (no model, no network)...")

    fake = [
        {"id": "a", "problem": "P1", "steps": ["s0", "s1(bad)", "s2"], "label": 1},   # error at idx 1
        {"id": "b", "problem": "P2", "steps": ["t0", "t1", "t2"], "label": -1},        # all correct
        {"id": "c", "problem": "P3", "steps": ["u0(bad)"], "label": 0},                # error at idx 0
        {"id": "d", "problem": "P4", "steps": [], "label": -1},                        # malformed, skipped
    ]
    ex = build_per_step_examples(fake, "unit", max_items=None)
    # a -> error(s1) + correct(s0); b -> correct(t2); c -> error(u0); d -> skipped
    labels = sorted([(e["id"], e["human_label"], e["last_step"]) for e in ex])
    expected = sorted([
        ("a", ERRONEOUS, "s1(bad)"), ("a", CORRECT, "s0"),
        ("b", CORRECT, "t2"),
        ("c", ERRONEOUS, "u0(bad)"),
    ])
    assert labels == expected, f"conversion mismatch:\n got {labels}\n exp {expected}"
    # empty-context handling for c (error at first step -> no preceding context)
    c_ex = [e for e in ex if e["id"] == "c"][0]
    assert c_ex["partial_solution"] == "Let's solve this problem.", c_ex["partial_solution"]

    # parsing
    assert parse_answer("blah <Answer>Correct</Answer>") == CORRECT
    assert parse_answer("...\n<Answer> Incorrect </Answer>") == ERRONEOUS
    assert parse_answer("no tag here") == INVALID
    assert parse_answer("<answer>incorrect</answer>") == ERRONEOUS  # case-insensitive

    # metrics: build a tiny record set with a known confusion pattern
    recs = [
        {"data_type": "x", "human_label": ERRONEOUS, "pred": ERRONEOUS},  # caught error
        {"data_type": "x", "human_label": ERRONEOUS, "pred": CORRECT},    # missed error
        {"data_type": "x", "human_label": CORRECT,   "pred": CORRECT},    # ok
        {"data_type": "x", "human_label": CORRECT,   "pred": ERRONEOUS},  # false positive
        {"data_type": "y", "human_label": CORRECT,   "pred": INVALID},    # invalid (dropped)
    ]
    s = summarize(recs)
    assert abs(s["erroneous_step_acc"] - 0.5) < 1e-9, s["erroneous_step_acc"]
    assert abs(s["correct_step_acc"] - 0.5) < 1e-9, s["correct_step_acc"]
    assert abs(s["valid_ratio"] - 0.8) < 1e-9, s["valid_ratio"]
    assert abs(s["harmonic_mean"] - 0.5) < 1e-9, s["harmonic_mean"]
    print_summary("self-test summary", s)
    print("\nSelf-test PASSED ✅")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SPC Critic-0 vs Critic-2 probe on ProcessBench.")
    ap.add_argument("--selftest", action="store_true", help="verify logic without model/network")
    ap.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS,
                    help=f"ProcessBench subsets (default: {DEFAULT_SUBSETS})")
    ap.add_argument("--max-per-subset", type=int, default=40,
                    help="max ProcessBench items per subset before per-step expansion")
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="generation budget for the critique (lower = faster/less memory)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                    help="if you get gibberish/NaNs in float16, try bfloat16")
    ap.add_argument("--device", default=None, help="mps / cpu / cuda (auto-detected if omitted)")
    ap.add_argument("--out", default="spc_probe_outputs.jsonl",
                    help="base path; per-checkpoint files derived (e.g. spc_probe_outputs.Critic-0.jsonl)")
    ap.add_argument("--no-resume", action="store_true",
                    help="overwrite per-checkpoint files instead of resuming from them")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    # heavy imports only for a real run
    import torch
    from datasets import load_dataset

    device = args.device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"device = {device}")

    # 1-2. Build the per-step probe set from ProcessBench
    print("Loading ProcessBench and building per-step probes...")
    all_examples = []
    for subset in args.subsets:
        ds = load_dataset("Qwen/ProcessBench", split=subset)
        ex = build_per_step_examples(ds, subset, args.max_per_subset)
        n_err = sum(1 for e in ex if e["human_label"] == ERRONEOUS)
        n_cor = sum(1 for e in ex if e["human_label"] == CORRECT)
        print(f"  {subset:>14}: {len(ex):4d} probes  (error: {n_err}, correct: {n_cor})")
        all_examples.extend(ex)
    print(f"  TOTAL: {len(all_examples)} probes")

    # show two converted examples so you can eyeball the label/indexing assumption
    print("\n--- sample converted probe (sanity-check the 0-based label assumption) ---")
    for e in all_examples[:2]:
        print(json.dumps({k: (v[:120] + "…" if isinstance(v, str) and len(v) > 120 else v)
                          for k, v in e.items()}, ensure_ascii=False, indent=2))
    print("---------------------------------------------------------------------------\n")

    # 3. Run each checkpoint with per-probe checkpointing + resume
    results = []
    out_paths = []
    for name, path in CHECKPOINTS:
        out_path = _per_ckpt_path(args.out, name)
        out_paths.append(out_path)
        existing, seen = ([], set()) if args.no_resume else _load_existing(out_path)
        print(f"\n>>> {name}  ({path})")
        print(f"    out: {out_path}  ({'overwrite' if args.no_resume else 'append'}, {len(existing)} prior probes)")
        if len(seen) >= len(all_examples):
            print("    all probes already done; skipping inference")
            s = summarize(existing)
            print_summary(name, s)
            results.append((name, s))
            continue
        mode = "w" if args.no_resume else "a"
        fout = open(out_path, mode, encoding="utf-8")

        def write_rec(rec, _fout=fout, _name=name):
            _fout.write(json.dumps({"checkpoint": _name, **rec}, ensure_ascii=False) + "\n")
            _fout.flush()  # persist after each probe -> crash-safe

        tok, model = load_critic(path, device, args.dtype)
        new_records = run_critic(tok, model, all_examples, device, args.max_new_tokens,
                                 write_rec=write_rec, skip_keys=seen)
        fout.close()
        s = summarize(existing + new_records)
        print_summary(name, s)
        results.append((name, s))
        free_model(model)
        del tok
        gc.collect()
    print(f"\nPer-checkpoint outputs: {out_paths}")

    # 4. Comparison
    print_comparison(results)


if __name__ == "__main__":
    main()


# ============================================================================
# OPTIONAL EXTENSION: synthetic-error probe (not included by default)
# ----------------------------------------------------------------------------
# SPC's sneaky-generator checkpoint was not released, so you cannot reproduce
# THEIR synthetic errors exactly. To still test "does the critic catch injected
# errors better than natural ones?", inject errors yourself:
#   1. Take ProcessBench all-correct items (label == -1).
#   2. For a known-correct step, ask a cheap local model (e.g. Qwen2.5-7B-Instruct
#      via MLX) to rewrite it into a SUBTLY wrong step, using a small fixed list
#      of error types (sign error, wrong formula, arithmetic slip, ...).
#   3. Label those as ERRONEOUS, run the same critics, and compare erroneous-step
#      accuracy on SYNTHETIC vs NATURAL errors.
# If synthetic >> natural, that is direct evidence of distribution shift toward
# the generator's error style. Keep the error-type list explicit in the report.
# ============================================================================
