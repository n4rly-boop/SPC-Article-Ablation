#!/usr/bin/env python3
"""Pull Critic-2 false positives from spc_probe_outputs.jsonl.

Reads the raw outputs, finds probes where human_label == +1 (correct step)
but Critic-2 predicted -1 (Incorrect). Bonus: prints cases where Critic-0
got it right and Critic-2 flipped to wrong (the over-sensitivity signal).
"""
import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+",
                    default=["spc_probe_outputs.Critic-0.jsonl",
                             "spc_probe_outputs.Critic-2.jsonl"],
                    help="one or more per-checkpoint jsonl files")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--flipped-only", action="store_true",
                    help="show only cases where C0=correct, C2=incorrect")
    args = ap.parse_args()

    by_key = defaultdict(dict)  # (id, data_type, human_label) -> {ckpt: rec}
    for path in args.inp:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate truncated tail
                key = (r["id"], r["data_type"], r["human_label"])
                by_key[key][r["checkpoint"]] = r

    fps = []
    flipped = []
    for key, by_ckpt in by_key.items():
        _, _, human = key
        if human != 1:
            continue
        c0 = next((v for k, v in by_ckpt.items() if k.startswith("Critic-0")), None)
        c2 = next((v for k, v in by_ckpt.items() if k.startswith("Critic-2")), None)
        if c2 is None or c2["pred"] != -1:
            continue
        fps.append(c2)
        if c0 is not None and c0["pred"] == 1:
            flipped.append((c0, c2))

    print(f"total false positives by Critic-2: {len(fps)}")
    print(f"  of which flipped from Critic-0 correct: {len(flipped)}")

    chosen = flipped if args.flipped_only else [(None, fp) for fp in fps]
    for i, item in enumerate(chosen[: args.n], 1):
        c0, c2 = item
        print("\n" + "=" * 70)
        print(f"[{i}] id={c2['id']}  subset={c2['data_type']}")
        if c0:
            print(f"  Critic-0 response (first 300 chars):\n    {c0['response'][:300]!r}")
        print(f"  Critic-2 response (first 600 chars):\n    {c2['response'][:600]!r}")


if __name__ == "__main__":
    main()
