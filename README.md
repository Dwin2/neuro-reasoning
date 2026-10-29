# neuro-reasoning workspace

A procedurally generated RL environment of EEG BCI reasoning tasks found at: https://app.primeintellect.ai/dashboard/environments/dwin2/neuro-reasoning. 

| Path | What it is |
|---|---|
| `environments/neuro_reasoning/` | the publishable package (its README is the Hub card) |
| `environments/neuro_reasoning/neuro_reasoning/splits.py` | locked held-out parameter ranges |
| `environments/neuro_reasoning/neuro_reasoning/families/` | the 35 templates, 7 per family |
| `tests/` | grader, template, and adapter tests (no model needed) |
| `configs/eval_local.toml` | eval config: null harness, subprocess runtime, no upload |
| `scripts/run_local_baselines.sh` | baselines on local open models via an MLX server |
| `scripts/analyze.py` | results tables with 95% Wilson intervals |
| `docs/PREREGISTRATION.md` | hypothesis, committed before results were read |
| `docs/RESULTS.md` | baseline results and writeup |
| `drafts/` | the original v0 draft, kept for reference |

## Commands

```bash
uv sync                                           # install workspace + editable environment
uv run pytest -q                                  # all tests
uv run validate neuro-reasoning --runtime.type subprocess -n 350    # SDK model-free check
uv run eval @ configs/eval_local.toml --dry-run   # resolve the eval config
uv run eval @ configs/eval_local.toml -m <model>  # eval via Prime Inference (needs wallet funds)
scripts/run_local_baselines.sh 175                # local baselines (Apple silicon)
uv run python scripts/analyze.py                  # tables into results/*.csv
prime env push neuro-reasoning --visibility PRIVATE   # publish (run from this directory)
```
