"""Per-family and per-template accuracy from one or more eval runs.

usage: uv run python scripts/breakdown.py <run-dir-or-traces.jsonl> [...] [--templates] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path):
    f = path / "traces.jsonl" if path.is_dir() else path
    model = None
    cfg = f.parent / "configs" / "resolved" / "eval.json"
    if cfg.exists():
        model = json.loads(cfg.read_text())["model"]
    for line in f.open():
        if not line.strip():
            continue
        ep = json.loads(line)
        d = ep["task"]["data"]
        traces = ep.get("traces") or []
        tr = traces[-1] if traces else {}
        r = (tr.get("rewards") or {}).get("correct")
        fmt = (tr.get("metrics") or {}).get("format_ok")
        yield {
            "model": model or "?", "split": d["split"], "family": d["family"],
            "template": d["template"],
            "correct": float(r["score"]) if r else 0.0,  # errored rollouts count as wrong
            "format_ok": float(fmt) if fmt is not None else 0.0,
            "errored": not ep.get("ok", False),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--templates", action="store_true", help="also print per-template rows")
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    rows = [r for p in a.paths for r in load(p)]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["split"], r["family"], "*")].append(r)
        groups[(r["model"], r["split"], "*", "*")].append(r)
        if a.templates:
            groups[(r["model"], r["split"], r["family"], r["template"])].append(r)
    out = []
    for (model, split, fam, tpl), rs in sorted(groups.items()):
        n = len(rs)
        out.append({"model": model, "split": split, "family": fam, "template": tpl, "n": n,
                    "accuracy": round(sum(r["correct"] for r in rs) / n, 4),
                    "format_ok": round(sum(r["format_ok"] for r in rs) / n, 4),
                    "errored": sum(r["errored"] for r in rs)})
    print(f"{'model':44} {'split':8} {'family':16} {'template':32} {'n':>4} {'acc':>6} {'fmt':>6} {'err':>4}")
    for o in out:
        print(f"{o['model'][:44]:44} {o['split']:8} {o['family']:16} {o['template'][:32]:32} "
              f"{o['n']:4d} {o['accuracy']:6.3f} {o['format_ok']:6.3f} {o['errored']:4d}")
    if a.csv:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        with a.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)


if __name__ == "__main__":
    main()
