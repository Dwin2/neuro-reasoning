"""Aggregate every baseline run into results tables with 95% Wilson intervals.

usage: uv run python scripts/analyze.py [results/runs]
Writes results/summary.csv (model x split x family), results/templates.csv, and
results/pipeline_by_length.csv, and prints Markdown tables for docs/RESULTS.md.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from neuro_reasoning.grading import extract_answer, grade

FAMILIES = ["filter_design", "power", "split_hygiene", "epoch_rejection", "pipeline_order"]
Z = 1.96


def best_constant(family: str, template: str, split: str, n: int = 300) -> str:
    """Most common answer over n fresh items (seeds disjoint from any eval run's seeds)."""
    from collections import Counter

    from neuro_reasoning import families  # noqa: F401
    from neuro_reasoning.templates import REGISTRY, build

    fn = REGISTRY[family][template]
    return Counter(build(fn, 10**9 + i, split).answer for i in range(n)).most_common(1)[0][0]


def wilson(k: float, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rows(runs: Path):
    for f in sorted(runs.glob("*/traces.jsonl")):
        cfg = json.loads((f.parent / "configs" / "resolved" / "eval.json").read_text())
        model = cfg["model"].split("/")[-1].replace("-4bit", "")
        for line in f.open():
            if not line.strip():
                continue
            ep = json.loads(line)
            d = ep["task"]["data"]
            tr = (ep.get("traces") or [{}])[-1]
            calls = tr.get("calls") or []
            replies = [n["message"].get("content") or "" for n in tr.get("nodes") or []
                       if n["message"].get("role") == "assistant" and n.get("sampled")]
            # re-grade from the stored reply so every run is scored by the current grader
            pred = extract_answer(replies[-1]) if replies and ep.get("ok") else None
            score = grade(d["answer_type"], d["answer"], pred, tolerance=d.get("tolerance", 0.0))
            yield {
                "model": model, "split": d["split"], "family": d["family"],
                "template": d["template"], "n_steps": d["params"].get("n_steps"),
                "correct": score, "gold": d["answer"],
                "format_ok": float(pred is not None),
                "truncated": any(c.get("finish_reason") == "length" for c in calls),
                "errored": not ep.get("ok", False),
            }


def summarize(rs):
    n = len(rs)
    k = sum(r["correct"] for r in rs)
    lo, hi = wilson(k, n)
    return {"n": n, "accuracy": round(k / n, 4) if n else None, "ci_low": round(lo, 4),
            "ci_high": round(hi, 4), "format_ok": round(sum(r["format_ok"] for r in rs) / n, 4),
            "truncated": sum(r["truncated"] for r in rs), "errored": sum(r["errored"] for r in rs)}


def write(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def pct(x) -> str:
    return "" if x is None else f"{100 * x:.0f}"


def main() -> None:
    runs = Path(sys.argv[1] if len(sys.argv) > 1 else "results/runs")
    data = list(rows(runs))
    by = defaultdict(list)
    for r in data:
        by[(r["model"], r["split"], r["family"])].append(r)
        by[(r["model"], r["split"], "ALL")].append(r)
    summary = [{"model": m, "split": s, "family": f, **summarize(rs)} for (m, s, f), rs in sorted(by.items())]
    write(Path("results/summary.csv"), summary)

    tby = defaultdict(list)
    for r in data:
        tby[(r["model"], r["split"], r["family"], r["template"])].append(r)
    write(Path("results/templates.csv"),
          [{"model": m, "split": s, "family": f, "template": t, **summarize(rs)}
           for (m, s, f, t), rs in sorted(tby.items())])

    lby = defaultdict(list)
    for r in data:
        if r["family"] == "pipeline_order" and r["split"] in ("train", "heldout"):
            lby[(r["model"], r["n_steps"])].append(r)
    write(Path("results/pipeline_by_length.csv"),
          [{"model": m, "n_steps": n, **summarize(rs)} for (m, n), rs in sorted(lby.items())])

    models = sorted({r["model"] for r in data}, key=lambda m: min(
        (i for i, x in enumerate(["0.5B", "1.5B", "3B", "4B", "7B"]) if x in m), default=9))
    idx = {(s["model"], s["split"], s["family"]): s for s in summary}
    for split in ("train", "heldout"):
        print(f"\n### Accuracy (%), {split} split, 35 items per family cell\n")
        print("| Model | " + " | ".join(FAMILIES) + " | All (95% CI) | Format OK |")
        print("|---" * (len(FAMILIES) + 3) + "|")
        for m in models:
            if (m, split, "ALL") not in idx:
                continue
            a = idx[(m, split, "ALL")]
            cells = [pct(idx.get((m, split, f), {}).get("accuracy")) for f in FAMILIES]
            print(f"| {m} | " + " | ".join(cells) +
                  f" | {pct(a['accuracy'])} ({pct(a['ci_low'])} to {pct(a['ci_high'])}) | {pct(a['format_ok'])} |")

    # Constant-answer baseline: always give the most common gold answer of each template.
    from collections import Counter
    print("\n### Constant-answer baseline (%): always each template's most common answer, estimated from 300 fresh items\n")
    print("| Split | " + " | ".join(FAMILIES) + " | All |")
    print("|---" * (len(FAMILIES) + 2) + "|")
    ref_model = models[0]
    base_rows = []
    for split in ("train", "heldout"):
        items = [r for r in data if r["model"] == ref_model and r["split"] == split]
        if not items:
            continue
        cells, tot_k, tot_n = [], 0, 0
        for f in FAMILIES:
            fam = [r for r in items if r["family"] == f]
            k = 0
            for t in {r["template"] for r in fam}:
                guess = best_constant(f, t, split)
                k += sum(r["gold"] == guess for r in fam if r["template"] == t)
            cells.append(pct(k / len(fam)) if fam else "")
            tot_k, tot_n = tot_k + k, tot_n + len(fam)
            base_rows.append({"split": split, "family": f, "baseline_accuracy": round(k / len(fam), 4)})
        print(f"| {split} | " + " | ".join(cells) + f" | {pct(tot_k / tot_n)} |")
    write(Path("results/constant_baseline.csv"), base_rows)

    print("\n### Held-out minus train (percentage points)\n")
    print("| Model | " + " | ".join(FAMILIES) + " | All |")
    print("|---" * (len(FAMILIES) + 2) + "|")
    for m in models:
        cells = []
        for f in FAMILIES + ["ALL"]:
            t, h = idx.get((m, "train", f)), idx.get((m, "heldout", f))
            cells.append(f"{100 * (h['accuracy'] - t['accuracy']):+.0f}" if t and h else "")
        print(f"| {m} | " + " | ".join(cells) + " |")

    print("\n### Pipeline accuracy (%) by number of steps (train and held-out pooled)\n")
    steps = [4, 5, 6, 7]
    print("| Model | " + " | ".join(f"{s} steps" for s in steps) + " |")
    print("|---" * (len(steps) + 1) + "|")
    lidx = {(m, n): summarize(rs) for (m, n), rs in lby.items()}
    for m in models:
        print(f"| {m} | " + " | ".join(
            f"{pct(lidx[(m, s)]['accuracy'])} (n={lidx[(m, s)]['n']})" if (m, s) in lidx else ""
            for s in steps) + " |")

    anchor_models = sorted({m for (m, s, f) in idx if s == "anchor"})
    if anchor_models:
        print("\n### Contamination spot-check: anchor minus train (percentage points)\n")
        print("| Model | Family | Train acc | Anchor acc | Anchor n | Difference |")
        print("|---|---|---|---|---|---|")
        for m in anchor_models:
            for f in FAMILIES:
                t, a = idx.get((m, "train", f)), idx.get((m, "anchor", f))
                if t and a:
                    print(f"| {m} | {f} | {pct(t['accuracy'])} | {pct(a['accuracy'])} | {a['n']} | "
                          f"{100 * (a['accuracy'] - t['accuracy']):+.0f} |")


if __name__ == "__main__":
    main()
