"""LOCKED parameter ranges. Held-out axes fixed 2026-09-28, before any template was written.

Each family has exactly one held-out axis (``AXIS``). On that axis the ``train`` and ``heldout``
value sets are disjoint. Every other parameter uses the same values in both splits.

The ``anchor`` split feeds templates exact values from published EEG work (THINGS-EEG2,
Gifford et al. 2022, NeuroImage) or canonical textbook defaults. It exists only for the
contamination spot-check: compare accuracy on ``anchor`` against ``train``. Anchor values marked
VERIFY must be checked against the paper before the check is reported.
"""

from __future__ import annotations

import random


def _steps(start: float, stop: float, step: float) -> list[float]:
    n = int(round((stop - start) / step))
    return [round(start + i * step, 4) for i in range(n + 1)]


AXIS = {
    "filter_design": "f_max_hz",
    "power": "effect_size",
    "split_hygiene": "n_subjects",
    "epoch_rejection": "threshold_uv",
    "pipeline_order": "n_steps",
}

RANGES: dict[str, dict[str, dict[str, list]]] = {
    "filter_design": {
        # held-out axis: highest frequency of interest (high-gamma held out)
        "f_max_hz": {"train": list(range(8, 46)), "heldout": list(range(55, 121)), "anchor": [40]},
        "transition_hz": {"train": list(range(5, 31)), "anchor": [10]},
        "fs_hz": {"train": [250, 256, 500, 512, 1000, 1024, 2048], "anchor": [1000]},  # THINGS-EEG2 raw rate
        "target_fs_hz": {"train": [100, 128, 200, 250, 256], "anchor": [100]},  # THINGS-EEG2 downsampled rate
        "line_hz": {"train": [50, 60], "anchor": [50]},
    },
    "power": {
        # held-out axis: standardized effect size (small effects held out)
        "effect_size": {"train": _steps(0.30, 0.80, 0.05), "heldout": _steps(0.10, 0.25, 0.01), "anchor": [0.5]},
        "alpha": {"train": [0.05, 0.01], "anchor": [0.05]},
        "power": {"train": [0.80, 0.90, 0.95], "anchor": [0.80]},
        "rejection_rate": {"train": _steps(0.05, 0.30, 0.05), "anchor": [0.20]},
        "conditions": {"train": [2, 3, 4, 6], "anchor": [2]},
        "n_comparisons": {"train": list(range(2, 11)), "anchor": [2]},
        "soa_s": {"train": _steps(0.5, 2.0, 0.25), "anchor": [0.2]},  # THINGS-EEG2 RSVP SOA (VERIFY)
    },
    "split_hygiene": {
        # held-out axis: cohort size (large cohorts held out)
        "n_subjects": {"train": list(range(6, 31)), "heldout": list(range(50, 201)), "anchor": [10]},  # THINGS-EEG2 (VERIFY)
        "n_images": {"train": [200, 500, 1000, 1654, 2000, 5000], "anchor": [1654]},  # 16,540 train / 10 (VERIFY)
        "n_test_subjects": {"train": [1, 2, 3, 4], "anchor": [1]},
    },
    "epoch_rejection": {
        # held-out axis: peak-to-peak rejection threshold (strict thresholds held out)
        "threshold_uv": {"train": list(range(75, 151, 5)), "heldout": list(range(30, 61, 5)), "anchor": [100]},
        "n_epochs": {"train": list(range(8, 13)), "anchor": [10]},
        "n_channels": {"train": [3, 4], "anchor": [3]},
        "flat_uv": {"train": [1, 2, 3, 5], "anchor": [1]},
    },
    "pipeline_order": {
        # held-out axis: pipeline length (longer chains held out). No literature anchor exists.
        "n_steps": {"train": [4, 5], "heldout": [6, 7], "anchor": [4, 5]},
    },
}


def values(family: str, param: str, split: str) -> list:
    spec = RANGES[family][param]
    return spec.get(split) or spec["train"]


def draw(rng: random.Random, family: str, param: str, split: str):
    return rng.choice(values(family, param, split))


def check_locked() -> None:
    """The held-out axis must be disjoint across train and heldout; nothing else may differ."""
    for family, axis in AXIS.items():
        for param, spec in RANGES[family].items():
            train, heldout = set(spec["train"]), set(spec.get("heldout") or spec["train"])
            if param == axis:
                assert train.isdisjoint(heldout), f"{family}.{param} overlaps"
            else:
                assert train == heldout, f"{family}.{param} differs across splits"
