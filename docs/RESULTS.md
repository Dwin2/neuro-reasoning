# Baseline results

Three open instruct models from 0.5B to 3B parameters were run locally at 4-bit quantization
(MLX, Apple silicon). Each answered 175 train and 175 held-out items: 5 per template, 35 per
family. The 3B model also answered the anchor split for the contamination check. No frontier model
was tested. These results test the predictions in [PREREGISTRATION.md](PREREGISTRATION.md), which
was committed before any result was read.

**Setup.** Seed 0, temperature 0, at most 4096 output tokens, a single turn, and the `null`
harness. Every stored reply is re-graded with the final grader by `scripts/analyze.py`, and
errored or unparseable replies count as wrong. With 35 items per cell, a 95% interval spans about
plus or minus 15 points, so only large differences are interpretable.

## Summary

- **All three models are near or below a trivial floor.** Always giving each template's most
  likely answer scores 20% on train. The best model, Llama 3.2 3B, scores 18%. At this scale the
  environment has plenty of headroom: it measures failure well but cannot yet rank strong models.
- **Accuracy rises with size.** Overall train accuracy goes from 3% to 6% to 18%. Only the 3B model
  reliably ends its reply with the required `ANSWER:` line, at 99 to 100% versus 79 to 86%.
- **Longer pipelines are harder.** For the 3B model, pipeline accuracy falls from 20% at 4 steps
  to 0% at 7 steps.
- **Split hygiene reveals a bias, not a knowledge gap.** The 3B model answers "subject identity
  leakage" on 27 of 35 train items and never answers "no problem", although "no problem" is the most
  common correct answer.
- **Held-out is lower than train for the 3B model,** by 8 points overall, most on epoch rejection
  and pipeline order. The 95% intervals overlap.

## Predictions against results

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 1 | split_hygiene highest; pipeline_order and epoch_rejection lowest | split_hygiene is among the lowest for every model, because of a systematic "subject leakage" answer bias | **Not supported** |
| 2 | pipeline accuracy falls with the number of steps | 3B: 20, 10, 7, 0% at 4, 5, 6, 7 steps (14 to 21 items each); smaller models near 0 throughout | **Supported for 3B**, small samples |
| 3 | held-out below train for filter_design and power | 3B: -3 and -9 points, inside the noise | **Direction consistent, not significant** |
| 4 | larger models score higher, split_hygiene gap persists | overall 3, 6, 18%; the premised split_hygiene advantage never appears | **First half supported** |
| 5 | anchor no more than 10 points above train for every family | 3B power: +21 points (11 of 25 vs 8 of 35) | **Violated, but not by recall** (see below) |

The headline hypothesis, "models recall EEG terminology correctly but fail when constraints must
propagate across multiple parameters", is only partly supported. Constraint propagation fails
as predicted: pipeline accuracy drops with length, and every arithmetic family is low. But the
conceptual family is not easier. On split hygiene these models do not apply the concepts. They
default to the most familiar leakage label, (d), regardless of the design described.

## Contamination spot-check

The 3B model's power-family gain on the anchor split does not look like memorized answers.
- **All of the gain is in two arithmetic templates,** trial count and session length, where the
  anchor effect size of 0.5 makes the arithmetic exact, since (2 / 0.5)^2 = 16.
- **On the one item where recall would show,** the classic two-group sample size at d = 0.5,
  alpha = 0.05, and 80% power, the model answered the computed normal-approximation value, 63. It
  did not give the widely memorized t-test value, 64.
- **The comparison is confounded:** anchor values are round numbers, which also makes them easier
  to compute with. Several anchor templates have only one distinct item.

Anchor values come from THINGS-EEG2 (Gifford et al., 2022) and textbook defaults. The values marked
VERIFY in `splits.py` must be checked against the paper before this check is cited.

## Limitations

- **Small models only.** Nothing above 3B parameters was tested, so the results cannot show whether
  the failure patterns persist at scale. A frontier-model run is prepared in
  `configs/eval_frontier.toml` and needs Prime Inference funds.
- **Small cells.** 35 items per family and split gives wide intervals. Five items per template
  cannot support template-level claims.
- **4-bit quantization** may lower accuracy relative to full-precision weights.
- **Unbalanced labels.** Split-hygiene gold labels are skewed toward "no problem" (43% of train
  items). The constant-answer baseline below accounts for this.

## Tables


### Accuracy (%), train split, 35 items per family cell

| Model | filter_design | power | split_hygiene | epoch_rejection | pipeline_order | All (95% CI) | Format OK |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 6 | 6 | 0 | 0 | 3 | 3 (1 to 7) | 79 |
| Qwen2.5-1.5B-Instruct | 6 | 11 | 0 | 11 | 0 | 6 (3 to 10) | 83 |
| Llama-3.2-3B-Instruct | 20 | 23 | 9 | 26 | 14 | 18 (13 to 25) | 100 |

### Accuracy (%), heldout split, 35 items per family cell

| Model | filter_design | power | split_hygiene | epoch_rejection | pipeline_order | All (95% CI) | Format OK |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 3 | 0 | 0 | 0 | 0 | 1 (0 to 3) | 79 |
| Qwen2.5-1.5B-Instruct | 0 | 6 | 3 | 3 | 0 | 2 (1 to 6) | 86 |
| Llama-3.2-3B-Instruct | 17 | 14 | 9 | 9 | 3 | 10 (7 to 16) | 99 |

### Constant-answer baseline (%): always each template's most common answer, estimated from 300 fresh items

| Split | filter_design | power | split_hygiene | epoch_rejection | pipeline_order | All |
|---|---|---|---|---|---|---|
| train | 26 | 3 | 46 | 26 | 0 | 20 |
| heldout | 9 | 6 | 37 | 31 | 0 | 17 |

### Held-out minus train (percentage points)

| Model | filter_design | power | split_hygiene | epoch_rejection | pipeline_order | All |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | -3 | -6 | +0 | +0 | -3 | -2 |
| Qwen2.5-1.5B-Instruct | -6 | -6 | +3 | -9 | +0 | -3 |
| Llama-3.2-3B-Instruct | -3 | -9 | +0 | -17 | -11 | -8 |

### Pipeline accuracy (%) by number of steps (train and held-out pooled)

| Model | 4 steps | 5 steps | 6 steps | 7 steps |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 7 (n=15) | 0 (n=20) | 0 (n=14) | 0 (n=21) |
| Qwen2.5-1.5B-Instruct | 0 (n=15) | 0 (n=20) | 0 (n=14) | 0 (n=21) |
| Llama-3.2-3B-Instruct | 20 (n=15) | 10 (n=20) | 7 (n=14) | 0 (n=21) |

### Contamination spot-check: anchor minus train (percentage points)

| Model | Family | Train acc | Anchor acc | Anchor n | Difference |
|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct | filter_design | 20 | 20 | 30 | +0 |
| Llama-3.2-3B-Instruct | power | 23 | 44 | 25 | +21 |
| Llama-3.2-3B-Instruct | split_hygiene | 9 | 9 | 33 | +1 |
| Llama-3.2-3B-Instruct | epoch_rejection | 26 | 10 | 80 | -16 |
| Llama-3.2-3B-Instruct | pipeline_order | 14 | 11 | 80 | -3 |

## Reproduce

```bash
scripts/run_local_baselines.sh 175      # edit MODELS in the script to choose models
uv run python scripts/analyze.py        # writes results/*.csv and prints these tables
```
