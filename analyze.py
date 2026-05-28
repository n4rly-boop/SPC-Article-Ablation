#!/usr/bin/env python3
"""Cross-tab Critic-0 vs Critic-2 predictions, split by human label.

Locates the 4 disagreement cells per truth class:
  truth=Correct:  C0_right/C2_wrong, C0_wrong/C2_right, both_right, both_wrong
  truth=Error:    same four

Prints counts + a few examples per cell so the report can cite concrete cases.
"""
import argparse
import json
from collections import defaultdict


def load_records(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c0", default="spc_probe_outputs.Critic-0.jsonl")
    ap.add_argument("--c2", default="spc_probe_outputs.Critic-2.jsonl")
    ap.add_argument("--n", type=int, default=2, help="examples per cell")
    args = ap.parse_args()

    c0 = {(r["id"], r["data_type"], r["human_label"]): r for r in load_records(args.c0)}
    c2 = {(r["id"], r["data_type"], r["human_label"]): r for r in load_records(args.c2)}
    keys = sorted(set(c0) & set(c2))

    cells = defaultdict(list)  # (truth, c0_correct, c2_correct) -> [(key, c0, c2)]
    for k in keys:
        truth = k[2]
        r0, r2 = c0[k], c2[k]
        cells[(truth, r0["pred"] == truth, r2["pred"] == truth)].append((k, r0, r2))

    def fmt_truth(t):
        return "CORRECT-step" if t == 1 else "ERRONEOUS-step"

    print(f"Loaded {len(keys)} paired probes\n")
    print("=" * 70)
    print("DISAGREEMENT MATRIX (rows = truth class)")
    print("=" * 70)
    print(f"{'truth':<18} {'both right':>11} {'C0✓ C2✗':>10} {'C0✗ C2✓':>10} {'both wrong':>11}")
    for truth in (1, -1):
        bb = len(cells[(truth, True, True)])
        c0w = len(cells[(truth, True, False)])
        c2w = len(cells[(truth, False, True)])
        ww = len(cells[(truth, False, False)])
        print(f"{fmt_truth(truth):<18} {bb:>11} {c0w:>10} {c2w:>10} {ww:>11}")

    # the two cells that explain C2's net gain/loss
    for truth in (1, -1):
        for c0_ok, c2_ok, label in [
            (False, True, "C0 wrong, C2 right (C2 WINS)"),
            (True, False, "C0 right, C2 wrong (C2 REGRESSES)"),
        ]:
            bucket = cells[(truth, c0_ok, c2_ok)]
            if not bucket:
                continue
            print("\n" + "-" * 70)
            print(f"{fmt_truth(truth)}  |  {label}  |  n={len(bucket)}")
            print("-" * 70)
            for i, (k, r0, r2) in enumerate(bucket[: args.n], 1):
                print(f"\n[{i}] id={k[0]}  subset={k[1]}")
                print(f"  C0 pred={r0['pred']}  resp[:250]: {r0['response'][:250]!r}")
                print(f"  C2 pred={r2['pred']}  resp[:250]: {r2['response'][:250]!r}")


if __name__ == "__main__":
    main()
