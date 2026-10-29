# neuro-reasoning

A procedurally generated RL environment of EEG BCI reasoning tasks: sampling and filtering, power
analysis, train/test split hygiene, epoch rejection, and preprocessing pipeline order. No LLM
judge. Every answer is computed by the generator and checked by exact match.

- **Environment ID:** `neuro-reasoning`
- **verifiers API:** v1 (`verifiers>=0.3.1`)
- **Task type:** single-turn, one final answer per item
- **Tags:** neuroscience, eeg, bci, science, reasoning, single-turn, eval, train

## Task families

| Family | Templates | Answer type | What it tests |
|---|---|---|---|
| `filter_design` | 7 | integer, number, yes/no | Nyquist limits with transition bands, decimation, aliasing, FFT resolution, FIR length and group delay |
| `power` | 7 | integer, number | sample size for two-group, paired, and Bonferroni designs; exclusion and rejection inflation; trial counts; session length; achieved power |
| `split_hygiene` | 7 | letter (a to d) | whether an evaluation design leaks: (a) temporal leakage (b) stimulus-level overlap (c) no problem (d) subject identity leakage |
| `epoch_rejection` | 7 | integer | how many epochs survive a stated rule applied to a table of amplitudes |
| `pipeline_order` | 7 | letter sequence | the one valid order of 4 to 7 preprocessing steps given stated failure reasons |

Every template chains at least two constraints, so that recalling a single fact is not enough.
Examples include the Nyquist limit applied to the filter's attenuation edge rather than its
passband edge, and the one bad channel that must be excluded before counting.

## Guarantees

- **Deterministic.** The same `seed` and `split` always give the same items. Each item records the
  sub-seed that rebuilds it.
- **Exact verification.** Numbers must match exactly, or within a stated tolerance where the
  question asks for rounding. Choices, yes/no, and orderings must match exactly. Replies are
  parsed only from a final `ANSWER: <value>` line.
- **Unambiguous.** Any sampled value near a rounding boundary is resampled. Every
  `pipeline_order` item is checked by brute force over all permutations and has exactly one valid
  order.
- **Balanced.** Items round-robin over families and templates, so any prefix is balanced. Items
  are unique on (template, parameters).

## Splits

| Split | Purpose |
|---|---|
| `train` | default parameter ranges |
| `heldout` | on one axis per family, values disjoint from `train`: high-gamma frequencies 55 to 120 Hz, small effect sizes 0.10 to 0.25, cohorts of 50 to 200 participants, strict thresholds 30 to 60 uV, and 6 to 7 pipeline steps |
| `anchor` | exact values from THINGS-EEG2 (Gifford et al., 2022) and textbook defaults, used for the contamination spot-check; a template may yield only a few distinct items |

The ranges are defined and locked in `neuro_reasoning/splits.py`.

## Quickstart

```bash
prime env install <owner>/neuro-reasoning
uv run eval neuro-reasoning -m <model> -n 175 \
  --env.agent.harness.id null --env.agent.max-turns 1 --env.agent.runtime.type subprocess
uv run eval neuro-reasoning -m <model> -n 175 --env.taskset.split heldout \
  --env.agent.harness.id null --env.agent.max-turns 1 --env.agent.runtime.type subprocess
```

Use the `null` harness, a plain chat loop. The default `bash` harness gives the model a shell,
which this environment does not need. `-n 175` gives five items per template.

## Taskset config

| Field | Type | Default | Description |
|---|---|---|---|
| `num_tasks` | int | `350` | items to generate |
| `seed` | int | `0` | generation seed |
| `split` | `train` \| `heldout` \| `anchor` | `train` | parameter ranges to sample from |
| `families` | list[str] \| null | `null` | restrict to these families |

## Metrics

| Metric | Meaning |
|---|---|
| `correct` (reward) | 1.0 if the final answer matches exactly, else 0.0 |
| `format_ok` (metric) | 1.0 if the reply contained an `ANSWER:` line |

Each trace also records the generator's parameters, template, split, and derivation in its task
data, so accuracy can be broken down by family and template.

## Baselines

Accuracy (%) on 175 train items (35 per family), temperature 0, 4-bit local weights.

| Model | filter | power | split | epoch | pipeline | All |
|---|---|---|---|---|---|---|
| constant-answer floor | 26 | 3 | 46 | 26 | 0 | 20 |
| Qwen2.5-0.5B-Instruct | 6 | 6 | 0 | 0 | 3 | 3 |
| Qwen2.5-1.5B-Instruct | 6 | 11 | 0 | 11 | 0 | 6 |
| Llama-3.2-3B-Instruct | 20 | 23 | 9 | 26 | 14 | 18 |

The "constant-answer floor" row always gives each template's most likely answer. Models up to 3B
score at or below it. Held-out accuracy for the 3B model is 10%. Full results, held-out tables, and
the contamination check are in `docs/RESULTS.md` in the source repository.

## Scope and limits

Questions state their formulas and conventions, such as the normal-approximation sample size,
the z-values, and the FIR design rules. They test whether a model can apply and chain constraints,
not whether it can recall a textbook table. The anchor values come from one dataset paper and
should be checked against the paper before the contamination result is reported.
