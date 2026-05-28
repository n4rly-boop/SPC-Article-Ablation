# SPC: Evolving Self-Play Critic. Review and probe experiment

Paper: Chen et al., *SPC: Evolving Self-Play Critic via Adversarial Games for LLM Reasoning*, NeurIPS 2025, [arXiv:2504.19162](https://arxiv.org/abs/2504.19162).

## Method

SPC trains a per-step critic for math reasoning through adversarial self-play between two copies of Qwen2.5-7B-Instruct:

- **Sneaky generator** rewrites one step of a correct solution into a wrong step.
- **Critic** reads `(problem, partial_solution, last_step)` and outputs `<Answer>Correct|Incorrect</Answer>` after a short chain-of-thought.

Both models start from the same SFT checkpoint (round 0). Each round, the generator and critic play against each other; rewards come from outcome (did the generator fool the critic, did the critic catch the injection), and both get RL-finetuned. SPC releases checkpoints for round 0 (SFT only) and round 2 (after two rounds of self-play RL).

## Tasks

The critic is evaluated as a *per-step error detector* on **ProcessBench** (Zheng et al., [arXiv:2412.06559](https://arxiv.org/abs/2412.06559)), which contains real LLM-written math solutions with human labels for the first erroneous step. Subsets: `gsm8k`, `math`, `olympiadbench`, `omnimath`. Headline metric: macro-averaged accuracy across subsets, computed over per-item probes.

## Strengths

- Step-level training signal: no outcome supervision, no search over solutions.
- Self-play removes the need for human-annotated wrong-step data.
- Released checkpoints run on a single 24 GB GPU.

## Weaknesses

- The critic trains against a *synthetic* error distribution (sneaky generator) but is evaluated on a *natural* one (real LLM mistakes in ProcessBench). The two are not the same.
- The sneaky-generator checkpoint is not released, so the training-side distribution is not reproducible.
- Only two RL rounds shipped; behaviour past round 2 is not observable from the release.

## Hypothesis we set out to test

A critic trained against a sneaky generator that injects an error into almost every probe might shift its prior toward "Incorrect" and become **over-sensitive**. If so, Critic-0 → Critic-2 should raise recall on real errors but *lower* accuracy on real correct steps.

## Experiment

We run both released checkpoints (`judge/SPC-Critic-0`, `judge/SPC-Critic-2`) on the first 40 items of each ProcessBench subset, expanded into per-step probes following SPC's own formulation. Both critics see an identical probe set (paired design), use SPC's verbatim prompt, greedy decoding (`temperature=0`), and `max_new_tokens=512`. Inference runs via vLLM on one Modal A10G GPU. N = 296 probes per critic (135 correct, 160 erroneous; 1 invalid generation in Critic-0, 2 in Critic-2).

**Deviations from the paper's eval:**

- vLLM on a single A10G instead of 2×GPU.
- ProcessBench reconstructed from the raw Hugging Face dataset rather than SPC's preprocessed OneDrive files.
- N = 296 (40 items × 4 subsets) vs the full ~3400 items used in the paper.

### Headline numbers

| metric                             | Critic-0 (SFT) | Critic-2 (RL) | Δ      |
|------------------------------------|---------------:|--------------:|-------:|
| erroneous-step recall              | 0.731          | 0.711         | −0.020 |
| correct-step accuracy              | 0.674          | 0.859         | +0.185 |
| macro-avg combined acc (headline)  | 0.705          | **0.781**     | +0.076 |
| harmonic mean                      | 0.701          | 0.778         | +0.077 |

Paper reports Critic-2 ProcessBench ≈ 0.777. Our 0.781 reproduces within 0.4 pp on a 12× smaller sample.

Per-subset combined accuracy for Critic-2: `gsm8k 0.740`, `math 0.853`, `olympiadbench 0.762`, `omnimath 0.767`.

### Decomposition

The over-sensitivity hypothesis predicted recall ↑ and correct-step accuracy ↓. We see the opposite. Disagreement matrix on paired probes:

| truth class            | both right | C0 right, C2 wrong | C0 wrong, C2 right | both wrong |
|------------------------|-----------:|-------------------:|-------------------:|-----------:|
| correct step (n=136)   | 85         | 6                  | **31**             | 14         |
| erroneous step (n=160) | 91         | **26**             | 22                 | 21         |

Critic-2 flips Incorrect → Correct net +25 times on correct-step probes and net +4 times on erroneous-step probes. The RL critic is *more permissive*. Its gain comes from passing correct steps that Critic-0 flagged as wrong, at the cost of missing four extra real errors.

## Why this matters (development of the paper)

The paper reports the round-0 → round-2 gain as one accuracy number. The decomposition shows the gain is asymmetric: a small recall loss on real errors is overshadowed by a large drop in false positives on correct steps. Three follow-ups:

1. **Calibration trajectory across rounds.** Does Critic-2's permissiveness keep growing in later rounds and eventually swallow real errors, or does the trend reverse? Only rounds 0 and 2 ship; rounds 1, 3+ would answer.
2. **Error-type stratification.** The 22 erroneous-step probes Critic-2 newly caught and the 26 it newly missed likely correlate with error type. Splitting ProcessBench errors by category (arithmetic slip, wrong formula, sign error, missing step) would show whether the regression concentrates in one category.
3. **Adversarial difficulty drift.** As the critic relaxes, the generator's job gets easier; with outcome-based reward, the generator may drift toward injecting easier-to-spot errors. Comparing generator outputs across rounds would show whether injected errors got softer.

A controlled extension, since the sneaky-generator weights are not released: use a cheap local model to inject errors from a small fixed taxonomy into ProcessBench correct items, then re-run both critics on synthetic and natural variants of the same step. The recall gap between the two variants quantifies the train/test distribution shift.

## Reproducibility

- `spc_probe.py`: local transformers/MPS implementation with per-probe checkpointing and resume.
- `spc_probe_modal.py`: Modal + vLLM port used for the numbers above.
- `analyze.py`: disagreement-matrix and per-cell example extraction.
- `extract_fps.py`: false-positive lister for Critic-2.

Raw outputs: `spc_probe_outputs.Critic-0.jsonl`, `spc_probe_outputs.Critic-2.jsonl`. SPC prompt and per-step probe construction copied verbatim from the authors' `eval/infer_batch.py` and follow ProcessBench's 0-based label indexing.
