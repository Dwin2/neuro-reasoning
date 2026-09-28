"""Item dataclass, template registry, and deterministic generation. Pure Python."""

from __future__ import annotations

import itertools
import json
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

SPLITS = ("train", "heldout", "anchor")


class Resample(Exception):
    """Raised by a template when its sampled parameters give an ambiguous or degenerate item
    (e.g. a value that lands within rounding error of an integer boundary)."""


@dataclass(frozen=True)
class Item:
    family: str
    template: str
    split: str
    seed: int
    params: dict = field(hash=False)
    """Every sampled value, JSON-serializable. Drives analysis and the contamination check."""
    question: str
    answer: str
    """Canonical string the grader compares against."""
    answer_type: str
    """integer | number | boolean | choice | ordering"""
    tolerance: float = 0.0
    rationale: str = ""
    """One-line programmatic derivation. Never shown to the model, never scored."""


TemplateFn = Callable[[random.Random, str, int], Item]
REGISTRY: dict[str, dict[str, TemplateFn]] = {}


def template(family: str, name: str) -> Callable[[TemplateFn], TemplateFn]:
    def register(fn: TemplateFn) -> TemplateFn:
        if name in REGISTRY.setdefault(family, {}):
            raise ValueError(f"duplicate template {family}/{name}")
        REGISTRY[family][name] = fn
        return fn

    return register


def build(fn: TemplateFn, seed: int, split: str, max_tries: int = 200) -> Item:
    """Run one template deterministically, retrying on `Resample` with derived sub-seeds.
    The returned item's ``seed`` is the sub-seed that produced it, so it can be rebuilt."""
    for attempt in range(max_tries):
        sub_seed = seed if attempt == 0 else seed * 7919 + attempt
        try:
            return fn(random.Random(sub_seed), split, sub_seed)
        except Resample:
            continue
    raise RuntimeError(f"{fn.__name__} could not produce an item for seed {seed}")


def generate(
    n: int, seed: int, split: str, families: list[str] | None = None
) -> Iterator[Item]:
    """Deterministic, balanced round-robin over templates, de-duplicated on (template, params)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    per_family = [
        [(fam, name, fn) for name, fn in sorted(d.items())]
        for fam, d in sorted(REGISTRY.items())
        if not families or fam in families
    ]
    # interleave families (f1t1, f2t1, ..., f1t2, ...) so any prefix of the stream is balanced
    fns = [x for group in itertools.zip_longest(*per_family) for x in group if x is not None]
    if not fns:
        raise ValueError(f"no templates for families={families}")
    seen: set[tuple[str, str]] = set()
    exhausted: set[str] = set()
    tries = 1000 if split != "anchor" else 60
    for produced in range(n):
        _, name, fn = fns[produced % len(fns)]
        if name in exhausted:
            continue
        for attempt in range(tries):
            item = build(fn, seed * 1_000_003 + produced * 1000 + attempt, split)
            key = (name, json.dumps(item.params, sort_keys=True))
            if key not in seen:
                break
        else:
            # The anchor split pins parameters to literature values, so a template may have only
            # a handful of distinct items; `n` is then an upper bound. Other splits must fill `n`.
            if split == "anchor":
                exhausted.add(name)
                continue
            raise RuntimeError(f"{name}: parameter space too small for {n} unique items")
        seen.add(key)
        yield item


def near_integer(x: float, eps: float = 0.02) -> bool:
    """True when rounding ``x`` up or down would depend on arithmetic precision."""
    return abs(x - round(x)) < eps


def near_half(x: float, eps: float = 0.02) -> bool:
    """True when rounding ``x`` to the nearest integer is ambiguous."""
    return abs((x - int(x)) - 0.5) < eps
