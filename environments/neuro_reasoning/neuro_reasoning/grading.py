"""Answer extraction and exact graders. Pure Python: no SDK, no model, no judge.

Every reply must end with a line ``ANSWER: <value>``. The last such line wins, and anything
after a closing ``</think>`` is the only text considered. A reply without the marker scores 0.
"""

from __future__ import annotations

import re

ANSWER_RE = re.compile(r"ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\.\d+")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")

ANSWER_TYPES = ("integer", "number", "boolean", "choice", "ordering")


def extract_answer(reply: str | None) -> str | None:
    """The value on the last ``ANSWER:`` line, or None when the reply has no marker."""
    if not reply:
        return None
    text = reply.rsplit("</think>", 1)[-1]
    found = ANSWER_RE.findall(text)
    if not found:
        return None
    value = found[-1].strip().strip("*`").strip()
    boxed = BOXED_RE.search(value)  # tolerate "ANSWER: \boxed{42}"
    return boxed.group(1).strip() if boxed else value


def parse_number(s: str) -> float | None:
    m = NUM_RE.search(s.replace("\u2212", "-"))
    return float(m.group().replace(",", "")) if m else None


def grade(answer_type: str, gold: str, pred: str | None, *, tolerance: float = 0.0) -> float:
    """1.0 when ``pred`` matches ``gold`` exactly under the answer type's rule, else 0.0."""
    if pred is None:
        return 0.0
    p = pred.strip().rstrip(".").lower()
    if len(p) >= 2 and p[0] in "<[{" and p[-1] in ">]}":  # "<d>" copied from the <value> placeholder
        p = p[1:-1].strip()
    if answer_type == "integer":
        x = parse_number(p)
        return float(x is not None and x == int(gold))
    if answer_type == "number":
        x = parse_number(p)
        return float(x is not None and abs(x - float(gold)) <= tolerance + 1e-9)
    if answer_type == "boolean":
        first = p.split()[0] if p.split() else ""
        word = re.sub(r"[^a-z]", "", first)
        if word in {"yes", "true"}:
            return float(gold == "yes")
        if word in {"no", "false"}:
            return float(gold == "no")
        return 0.0
    if answer_type == "choice":
        m = re.match(r"\(?([a-z])\)?(?:[.:)\s]|$)", p)
        return float(bool(m) and m.group(1) == gold)
    if answer_type == "ordering":
        tokens = [t for t in re.findall(r"[a-z0-9]+", p) if len(t) == 1]
        return float(tokens == gold.split(","))
    raise ValueError(f"unknown answer type {answer_type!r}")
