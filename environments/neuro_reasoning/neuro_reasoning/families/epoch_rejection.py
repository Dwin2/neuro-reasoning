"""Family 4: epoch rejection. Given a stated rule and a table of per-epoch values, count what
survives. The threshold is always given; the model never chooses one."""

from __future__ import annotations

from ..splits import draw
from ..templates import Item, Resample, template

F = "epoch_rejection"
CHANNELS = ["Fz", "Cz", "Pz", "Oz", "C3", "C4"]


def _cell(rng, t: int, p_art: float, flat: int | None = None, p_flat: float = 0.0) -> int:
    u = rng.random()
    if flat is not None and u < p_flat:
        return rng.randint(0, flat - 1)
    if u < p_flat + p_art:
        return rng.randint(t + 1, 3 * t)
    lo = max(int(0.2 * t), (flat or 0) + 1)
    return rng.randint(lo, t - 1)


def _grid(rng, split, p_art=0.12, flat=None, p_flat=0.0):
    t = draw(rng, F, "threshold_uv", split)
    n_ep = draw(rng, F, "n_epochs", split)
    n_ch = draw(rng, F, "n_channels", split)
    chans = CHANNELS[:n_ch]
    rows = [[_cell(rng, t, p_art, flat, p_flat) for _ in chans] for _ in range(n_ep)]
    return t, chans, rows


def _table(chans, rows, extra_col: tuple[str, list] | None = None) -> str:
    head = ["Epoch"] + ([extra_col[0]] if extra_col else []) + chans
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, row in enumerate(rows):
        cells = [str(i + 1)] + ([str(extra_col[1][i])] if extra_col else []) + [str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _check_range(survivors: int, n: int) -> None:
    if survivors in (0, n):
        raise Resample


def _item(name, split, seed, params, rows, q, answer, rationale) -> Item:
    return Item(F, name, split, seed, {**params, "table": rows}, q, str(answer), "integer",
                rationale=rationale)


INTRO = "Values are peak-to-peak amplitudes in microvolts (uV) per channel for each epoch."


@template(F, "any_channel_exceeds")
def any_channel_exceeds(rng, split, seed) -> Item:
    t, chans, rows = _grid(rng, split)
    keep = sum(all(v <= t for v in r) for r in rows)
    _check_range(keep, len(rows))
    q = (f"{INTRO}\n\n{_table(chans, rows)}\n\nRule: reject an epoch if the peak-to-peak amplitude on "
         f"any channel is greater than {t} uV.\n\nHow many epochs survive?")
    return _item("any_channel_exceeds", split, seed, {"threshold_uv": t}, rows, q, keep,
                 f"{keep} of {len(rows)} have all <= {t}")


@template(F, "exceeds_or_flat")
def exceeds_or_flat(rng, split, seed) -> Item:
    flat = draw(rng, F, "flat_uv", split)
    t, chans, rows = _grid(rng, split, p_art=0.1, flat=flat, p_flat=0.08)
    keep = sum(all(flat <= v <= t for v in r) for r in rows)
    _check_range(keep, len(rows))
    q = (f"{INTRO}\n\n{_table(chans, rows)}\n\nRule: reject an epoch if any channel's peak-to-peak "
         f"amplitude is greater than {t} uV (artifact) or less than {flat} uV (flat or disconnected "
         f"electrode).\n\nHow many epochs survive?")
    return _item("exceeds_or_flat", split, seed, {"threshold_uv": t, "flat_uv": flat}, rows, q, keep,
                 f"{keep} with all values in [{flat}, {t}]")


@template(F, "k_channels_exceed")
def k_channels_exceed(rng, split, seed) -> Item:
    t, chans, rows = _grid(rng, split, p_art=0.3)
    k = 2
    keep = sum(sum(v > t for v in r) < k for r in rows)
    _check_range(keep, len(rows))
    q = (f"{INTRO}\n\n{_table(chans, rows)}\n\nRule: reject an epoch only if {k} or more channels "
         f"have a peak-to-peak amplitude greater than {t} uV. A single noisy channel is tolerated."
         f"\n\nHow many epochs survive?")
    return _item("k_channels_exceed", split, seed, {"threshold_uv": t, "k": k}, rows, q, keep,
                 f"{keep} with fewer than {k} channels > {t}")


@template(F, "ignore_interpolated_channel")
def ignore_interpolated_channel(rng, split, seed) -> Item:
    t, chans, rows = _grid(rng, split, p_art=0.1)
    bad = rng.randrange(len(chans))
    for r in rows:  # the bad channel is noisy in most epochs
        if rng.random() < 0.7:
            r[bad] = rng.randint(t + 1, 4 * t)
    keep = sum(all(v <= t for j, v in enumerate(r) if j != bad) for r in rows)
    naive = sum(all(v <= t for v in r) for r in rows)
    _check_range(keep, len(rows))
    if keep == naive:
        raise Resample
    q = (f"{INTRO}\n\n{_table(chans, rows)}\n\nChannel {chans[bad]} was marked bad for the whole "
         f"recording and will be interpolated, so the rejection rule ignores it. Rule: reject an epoch "
         f"if any other channel's peak-to-peak amplitude is greater than {t} uV.\n\n"
         f"How many epochs survive?")
    return _item("ignore_interpolated_channel", split, seed,
                 {"threshold_uv": t, "bad_channel": chans[bad]}, rows, q, keep,
                 f"{keep} ignoring {chans[bad]} (naive {naive})")


@template(F, "repair_or_reject_interpolations")
def repair_or_reject_interpolations(rng, split, seed) -> Item:
    t, chans, rows = _grid(rng, split, p_art=0.25)
    k = 1
    counts = [sum(v > t for v in r) for r in rows]
    kept = [c for c in counts if c <= k]
    answer = sum(kept)
    if answer == 0 or len(kept) == len(rows):
        raise Resample
    q = (f"{INTRO}\n\n{_table(chans, rows)}\n\nRule, applied per epoch: a channel is bad in an epoch if "
         f"its peak-to-peak amplitude is greater than {t} uV. If an epoch has at most {k} bad channel, "
         f"the epoch is kept and its bad channel is repaired by interpolation. If it has more than {k}, "
         f"the whole epoch is rejected and nothing in it is interpolated.\n\n"
         f"How many channel interpolations are performed in total?")
    return _item("repair_or_reject_interpolations", split, seed, {"threshold_uv": t, "k": k}, rows, q,
                 answer, f"bad counts {counts} -> kept {len(kept)}, interpolations {answer}")


@template(F, "equalize_conditions")
def equalize_conditions(rng, split, seed) -> Item:
    t, chans, rows = _grid(rng, split, p_art=0.15)
    conds = [rng.choice("AB") for _ in rows]
    kept = {c: sum(1 for r, cc in zip(rows, conds) if cc == c and all(v <= t for v in r)) for c in "AB"}
    if min(kept.values()) == 0 or kept["A"] == kept["B"]:
        raise Resample
    answer = 2 * min(kept.values())
    q = (f"{INTRO} The Condition column gives each epoch's experimental condition.\n\n"
         f"{_table(chans, rows, ('Condition', conds))}\n\nStep 1: reject an epoch if any channel's "
         f"peak-to-peak amplitude is greater than {t} uV. Step 2: equalize the conditions by dropping "
         f"surviving epochs from the larger condition until both conditions have the same number.\n\n"
         f"How many epochs remain in total after step 2?")
    return _item("equalize_conditions", split, seed, {"threshold_uv": t, "conditions": conds}, rows, q,
                 answer, f"survivors per condition {kept} -> 2 x {min(kept.values())}")


@template(F, "min_max_to_ptp")
def min_max_to_ptp(rng, split, seed) -> Item:
    t, chans, ptp = _grid(rng, split, p_art=0.12)
    rows = []
    for r in ptp:
        cells = []
        for v in r:
            lo = -rng.randint(0, v)
            cells.append((lo, lo + v))
        rows.append(cells)
    keep = sum(all(v <= t for v in r) for r in ptp)
    _check_range(keep, len(rows))
    head = ["Epoch"] + [f"{c} min / max" for c in chans]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, r in enumerate(rows):
        lines.append("| " + " | ".join([str(i + 1)] + [f"{lo} / {hi}" for lo, hi in r]) + " |")
    q = ("The table gives the minimum and maximum voltage in microvolts (uV) on each channel within "
         "each epoch.\n\n" + "\n".join(lines) +
         f"\n\nRule: reject an epoch if the peak-to-peak amplitude (maximum minus minimum) on any "
         f"channel is greater than {t} uV.\n\nHow many epochs survive?")
    return _item("min_max_to_ptp", split, seed, {"threshold_uv": t}, rows, q, keep,
                 f"ptp {ptp} -> {keep}")
