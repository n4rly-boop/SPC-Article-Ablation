#!/usr/bin/env python3
"""SPC critic probe on Modal: vLLM-batched remote inference.

Builds the probe set locally (deterministic), ships it to a Modal GPU function
that runs vLLM batched generation on each SPC checkpoint, then writes
per-checkpoint jsonl and prints summary locally.

Run:
    modal run spc_probe_modal.py
    modal run spc_probe_modal.py --max-per-subset 80 --max-new-tokens 512
"""

import json

import modal

# Reuse data + metric logic from the local script
from spc_probe import (
    CHECKPOINTS,
    DEFAULT_SUBSETS,
    PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    _per_ckpt_path,
    build_per_step_examples,
    parse_answer,
    print_comparison,
    print_summary,
    summarize,
)

APP_NAME = "spc-probe"
VOLUME_NAME = "spc-hf-cache"

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm",
        "huggingface_hub",
        "hf_transfer",
    )
    .env({
        "HF_HOME": "/cache/hf",
        "HF_XET_HIGH_PERFORMANCE": "1",
        # avoid flashinfer JIT (needs nvcc, not in slim image); greedy doesnt need it
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
    .add_local_file("spc_probe.py", "/root/spc_probe.py")
)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=60 * 60,
    memory=32 * 1024,
)
def run_critic_remote(model_path: str, examples: list, max_new_tokens: int):
    """Load one critic via vLLM, batched-generate over all probes, return responses."""
    import os
    import time

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=model_path,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        download_dir="/cache/hf/hub",
    )
    tok = llm.get_tokenizer()
    print(f"[remote] loaded {model_path} in {time.time() - t0:.1f}s")

    prompts = []
    for ex in examples:
        user = PROMPT_TEMPLATE.format(
            problem=ex["problem"],
            partial_solution=ex["partial_solution"],
            last_step=ex["last_step"],
        )
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    dt = time.time() - t0
    print(f"[remote] generated {len(outputs)} probes in {dt:.1f}s ({len(outputs)/dt:.2f} probes/s)")

    hf_cache.commit()  # persist any newly-downloaded shards
    return [out.outputs[0].text for out in outputs]


@app.local_entrypoint()
def main(
    max_per_subset: int = 40,
    max_new_tokens: int = 512,
    subsets: str = ",".join(DEFAULT_SUBSETS),
    out: str = "spc_probe_outputs.jsonl",
):
    from datasets import load_dataset

    subset_list = [s.strip() for s in subsets.split(",") if s.strip()]
    print(f"building probes from {subset_list}  (max_per_subset={max_per_subset})")
    all_examples = []
    for subset in subset_list:
        ds = load_dataset("Qwen/ProcessBench", split=subset)
        ex = build_per_step_examples(ds, subset, max_per_subset)
        all_examples.extend(ex)
        print(f"  {subset:>14}: {len(ex):4d} probes")
    print(f"TOTAL: {len(all_examples)} probes")

    results = []
    out_paths = []
    for name, path in CHECKPOINTS:
        print(f"\n>>> {name}  ({path})")
        responses = run_critic_remote.remote(path, all_examples, max_new_tokens)
        out_path = _per_ckpt_path(out, name)
        out_paths.append(out_path)
        records = []
        with open(out_path, "w", encoding="utf-8") as fout:
            for ex, resp in zip(all_examples, responses):
                pred = parse_answer(resp)
                rec = {
                    "checkpoint": name,
                    "id": ex["id"],
                    "data_type": ex["data_type"],
                    "human_label": ex["human_label"],
                    "pred": pred,
                    "response": resp,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records.append(rec)
        print(f"  wrote {out_path}")
        s = summarize(records)
        print_summary(name, s)
        results.append((name, s))

    print(f"\nPer-checkpoint outputs: {out_paths}")
    print_comparison(results)
