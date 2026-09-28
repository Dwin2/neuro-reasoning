import itertools
import json
from collections import Counter

import pytest

from neuro_reasoning import families  # noqa: F401
from neuro_reasoning.families.pipeline_order import CONTEXTS
from neuro_reasoning.grading import grade
from neuro_reasoning.splits import AXIS, RANGES, check_locked
from neuro_reasoning.templates import REGISTRY, build, generate

ALL = [(fam, name) for fam, d in sorted(REGISTRY.items()) for name in sorted(d)]
WRONG = {"integer": "-987654", "number": "-987654", "boolean": "maybe", "choice": "z",
         "ordering": "z,y"}


def test_counts():
    assert set(REGISTRY) == set(AXIS)
    assert all(7 <= len(t) <= 8 for t in REGISTRY.values())


def test_splits_locked():
    check_locked()


@pytest.mark.parametrize("split", ["train", "heldout", "anchor"])
@pytest.mark.parametrize("fam,name", ALL)
def test_template(fam, name, split):
    fn = REGISTRY[fam][name]
    for s in range(40):
        it = build(fn, s, split)
        assert it.family == fam and it.template == name and it.split == split
        assert it.question.strip() and it.answer
        json.dumps(it.params)
        assert grade(it.answer_type, it.answer, it.answer, tolerance=it.tolerance) == 1.0
        assert grade(it.answer_type, it.answer, WRONG[it.answer_type], tolerance=it.tolerance) == 0.0
        assert build(fn, s, split) == it  # deterministic
        assert build(fn, it.seed, split) == it  # rebuildable from its recorded seed


@pytest.mark.parametrize("fam,name", ALL)
def test_heldout_axis_is_used(fam, name):
    """Every template must read the family's held-out axis, so heldout items really differ."""
    axis = AXIS[fam]
    held = set(RANGES[fam][axis]["heldout"])
    train = set(RANGES[fam][axis]["train"])
    for s in range(20):
        h = build(REGISTRY[fam][name], s, "heldout").params[axis]
        t = build(REGISTRY[fam][name], s, "train").params[axis]
        assert h in held and t in train


@pytest.mark.parametrize("fam,name", [x for x in ALL if x[0] in ("split_hygiene",)]
                         + [("filter_design", "anti_alias_cutoff_valid")])
def test_label_balance(fam, name):
    counts = Counter(build(REGISTRY[fam][name], s, "train").answer for s in range(300))
    assert len(counts) >= 2
    assert min(counts.values()) / 300 > 0.15, counts


@pytest.mark.parametrize("split", ["train", "heldout"])
def test_pipeline_orders_unique(split):
    """Independent re-check: brute-force every item's stated constraints."""
    for name in CONTEXTS:
        for s in range(15):
            it = build(REGISTRY["pipeline_order"][name], s, split)
            steps = it.params["steps"]
            facts = [(a, b) for a, b, _ in CONTEXTS[name]["facts"] if a in steps and b in steps]
            valid = [p for p in itertools.permutations(steps)
                     if all(p.index(a) < p.index(b) for a, b in facts)]
            assert len(valid) == 1
            letters = {st: chr(65 + i) for i, st in enumerate(it.params["shown"])}
            assert ",".join(letters[st].lower() for st in valid[0]) == it.answer


def test_generate_balanced_unique_deterministic():
    items = list(generate(350, seed=0, split="train"))
    assert len(items) == 350
    assert items == list(generate(350, seed=0, split="train"))
    per_template = Counter(it.template for it in items)
    assert set(per_template.values()) == {10}
    keys = {(it.template, json.dumps(it.params, sort_keys=True)) for it in items}
    assert len(keys) == 350


def test_train_heldout_disjoint_on_axis():
    train = list(generate(350, 0, "train"))
    held = list(generate(350, 0, "heldout"))
    for fam, axis in AXIS.items():
        tv = {it.params[axis] for it in train if it.family == fam}
        hv = {it.params[axis] for it in held if it.family == fam}
        assert tv.isdisjoint(hv)


def test_reference_values():
    """Hand-checked anchor items (THINGS-EEG2 values and the textbook d = 0.5 case)."""
    def one(fam, name):
        return build(REGISTRY[fam][name], 0, "anchor")
    it = one("filter_design", "min_sampling_rate")                          # (40+10)*2 = 100 Hz
    assert int(it.answer) == min(r for r in it.params["menu"] if r >= 100)
    assert one("filter_design", "max_decimation_factor").answer == "10"     # 1000 Hz -> 100 Hz
    assert one("filter_design", "fir_filter_length").answer == "331"        # 3.3*1000/10=330 -> odd
    assert one("filter_design", "anti_alias_cutoff_valid").answer == "yes"  # 50 <= 100/2
    assert one("power", "n_per_group_two_sample").answer == "63"            # not the t-test's 64
    assert one("power", "n_paired").answer == "32"                          # (2.802/0.5)^2=31.4
