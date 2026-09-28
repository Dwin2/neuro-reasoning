"""Family 1: filter design and sampling rate. Nyquist arithmetic with propagated constraints."""

from __future__ import annotations

import math
from fractions import Fraction

from ..splits import draw, values
from ..templates import Item, Resample, near_half, template

F = "filter_design"
RATES_HZ = [128, 250, 256, 500, 512, 1000, 1024, 2048]
AMP_RATES_HZ = [64, 100, 125, 128, 160, 200, 250, 256, 300, 400, 500, 512, 1000, 1024, 2000, 2048]


def _rate_menu(rng, naive_min, true_min, k: int = 5) -> list[int]:
    """A supported-rate list with at least one valid rate, one rate below the true minimum, and,
    when one exists, a trap rate that satisfies the naive requirement but not the true one."""
    valid = [r for r in AMP_RATES_HZ if r >= true_min]
    below = [r for r in AMP_RATES_HZ if r < true_min]
    trap = [r for r in below if r >= naive_min]
    if not valid or not below:
        raise Resample
    menu = {rng.choice(valid[:2])}
    menu.add(rng.choice(trap) if trap else rng.choice(below))
    menu.add(rng.choice(below))
    while len(menu) < k:
        menu.add(rng.choice(AMP_RATES_HZ))
    return sorted(menu)
LOW_RATES_HZ = [20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200, 250]
DECIMATION = [2, 4, 5, 8, 10]


def _fmt(x) -> str:
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else f"{float(x):g}"


def _mne_rules(fs: int, f_c: int) -> tuple[Fraction, int]:
    """Transition width and FIR length under the rules stated in the question text."""
    trans = min(max(Fraction(f_c, 4), Fraction(2)), Fraction(fs, 2) - f_c)
    if trans <= 0:
        raise Resample
    length = math.ceil(Fraction(33, 10) * fs / trans)
    if length % 2 == 0:
        length += 1
    return trans, length


FIR_RULES = (
    "Use these design rules: the transition bandwidth is min(max(0.25 x cutoff, 2 Hz), "
    "fs/2 - cutoff); the filter length in samples is ceil(3.3 x fs / transition bandwidth), "
    "and if that number is even, add 1 so the filter has an odd length."
)


@template(F, "min_sampling_rate")
def min_sampling_rate(rng, split, seed) -> Item:
    f_max = draw(rng, F, "f_max_hz", split)
    trans = draw(rng, F, "transition_hz", split)
    stop_edge = f_max + trans
    menu = _rate_menu(rng, 2 * f_max, 2 * stop_edge)
    answer = min(r for r in menu if r >= 2 * stop_edge)
    q = (
        f"You are configuring an EEG amplifier for a BCI experiment. The highest frequency you must "
        f"preserve is {f_max} Hz, so the analog anti-aliasing low-pass filter has its passband edge at "
        f"{f_max} Hz. Its transition band is {trans} Hz wide: it reaches full attenuation only at "
        f"{stop_edge} Hz. To prevent aliasing, the Nyquist frequency must be at or above the frequency "
        f"where full attenuation begins. The amplifier supports {', '.join(map(str, menu))} Hz.\n\n"
        f"What is the lowest supported sampling rate, in Hz, that satisfies this requirement?"
    )
    return Item(F, "min_sampling_rate", split, seed,
                {"f_max_hz": f_max, "transition_hz": trans, "menu": menu},
                q, str(answer), "integer",
                rationale=f"stop edge {stop_edge} Hz -> fs >= {2 * stop_edge} from {menu} -> {answer}")


@template(F, "anti_alias_cutoff_valid")
def anti_alias_cutoff_valid(rng, split, seed) -> Item:
    """Extends the author's first draft (drafts/neuro_reasoning_v0_draft.py): is a low-pass cutoff
    safe relative to Nyquist? Here the check must account for decimation and the transition band."""
    f_max = draw(rng, F, "f_max_hz", split)
    trans = draw(rng, F, "transition_hz", split)
    want_yes = rng.random() < 0.5
    combos = [(fs, d) for fs in values(F, "fs_hz", "train") for d in DECIMATION if fs % d == 0]
    if split == "anchor":
        combos = [(1000, 10)]
    ok = [(fs, d) for fs, d in combos if (f_max + trans <= fs / d / 2) == want_yes]
    if not ok:
        ok = combos
    fs, d = rng.choice(ok)
    new_fs = fs // d
    answer = "yes" if f_max + trans <= new_fs / 2 else "no"
    q = (
        f"An EEG recording sampled at {fs} Hz will be downsampled by an integer factor of {d}, to "
        f"{new_fs} Hz. Before downsampling you apply a digital low-pass anti-aliasing filter with its "
        f"cutoff (passband edge) at {f_max} Hz and a transition band {trans} Hz wide, so full "
        f"attenuation starts at {f_max + trans} Hz.\n\n"
        f"Does full attenuation start at or below the Nyquist frequency of the downsampled signal, so "
        f"that nothing above the new Nyquist frequency survives unattenuated? Answer yes or no."
    )
    return Item(F, "anti_alias_cutoff_valid", split, seed,
                {"f_max_hz": f_max, "transition_hz": trans, "fs_hz": fs, "decimation": d},
                q, answer, "boolean",
                rationale=f"stop {f_max + trans} vs new Nyquist {new_fs / 2} -> {answer}")


@template(F, "max_decimation_factor")
def max_decimation_factor(rng, split, seed) -> Item:
    fs = draw(rng, F, "fs_hz", split)
    f_max = draw(rng, F, "f_max_hz", split)
    trans = draw(rng, F, "transition_hz", split)
    stop = f_max + trans
    ok = [d for d in range(1, fs + 1) if fs % d == 0 and Fraction(fs, 2 * d) >= stop]
    if not ok:
        raise Resample
    answer = max(ok)
    q = (
        f"You recorded EEG at {fs} Hz and want to downsample it by an integer factor d so that the new "
        f"rate, {fs}/d Hz, is a whole number of Hz. You will preserve activity up to {f_max} Hz with an "
        f"anti-aliasing low-pass filter whose full attenuation begins {trans} Hz above that. The new "
        f"Nyquist frequency must be at or above the frequency where full attenuation begins.\n\n"
        f"What is the largest valid decimation factor d?"
    )
    return Item(F, "max_decimation_factor", split, seed,
                {"fs_hz": fs, "f_max_hz": f_max, "transition_hz": trans},
                q, str(answer), "integer",
                rationale=f"need fs/d >= {2 * stop}; divisors of {fs} -> {answer}")


@template(F, "aliased_frequency")
def aliased_frequency(rng, split, seed) -> Item:
    f = draw(rng, F, "f_max_hz", split)
    candidates = [fs for fs in LOW_RATES_HZ if fs / 2 < f < 2 * fs and f % fs != 0 and 2 * f % fs != 0]
    if not candidates:
        raise Resample
    fs = rng.choice(candidates)
    alias = abs(f - fs * round(f / fs))
    q = (
        f"An EEG channel contains a strong oscillatory component at {f} Hz. The data are downsampled "
        f"to {fs} Hz by keeping every n-th sample, with no anti-aliasing filter applied first.\n\n"
        f"At what frequency, in Hz, does this component appear in the downsampled data?"
    )
    return Item(F, "aliased_frequency", split, seed, {"f_max_hz": f, "fs_low_hz": fs},
                q, str(alias), "integer",
                rationale=f"|{f} - {fs}*round({f}/{fs})| = {alias}")


@template(F, "min_epoch_samples_for_resolution")
def min_epoch_samples_for_resolution(rng, split, seed) -> Item:
    f1 = draw(rng, F, "f_max_hz", split)
    trans = draw(rng, F, "transition_hz", split)
    delta = Fraction(rng.choice(["0.2", "0.25", "0.5", "1", "2"]))
    f2 = f1 + delta
    menu = _rate_menu(rng, 2 * f2, 2 * (f2 + trans))
    fs = min(r for r in menu if Fraction(r, 2) >= f2 + trans)
    n = math.ceil(fs / delta)
    q = (
        f"An SSVEP BCI flickers two targets at {_fmt(f1)} Hz and {_fmt(f2)} Hz. The amplifier supports "
        f"{', '.join(map(str, menu))} Hz, and you choose the lowest rate whose Nyquist frequency is "
        f"at or above {trans} Hz beyond the higher stimulation frequency (the filter's transition band). "
        f"You will separate the two targets with a plain FFT of each epoch, which needs a frequency "
        f"resolution (1 / epoch duration) no coarser than the spacing between the two frequencies.\n\n"
        f"What is the minimum epoch length, in samples, at the chosen sampling rate?"
    )
    return Item(F, "min_epoch_samples_for_resolution", split, seed,
                {"f_max_hz": f1, "delta_hz": _fmt(delta), "transition_hz": trans, "menu": menu},
                q, str(n), "integer",
                rationale=f"fs={fs}; T >= 1/{_fmt(delta)} s -> N = ceil({fs}/{_fmt(delta)}) = {n}")


@template(F, "fir_filter_length")
def fir_filter_length(rng, split, seed) -> Item:
    fs = draw(rng, F, "fs_hz", split)
    f_c = draw(rng, F, "f_max_hz", split)
    trans, length = _mne_rules(fs, f_c)
    q = (
        f"You design a windowed-sinc (Hamming) FIR low-pass filter for EEG sampled at {fs} Hz with a "
        f"cutoff of {f_c} Hz. {FIR_RULES}\n\nWhat is the filter length in samples?"
    )
    return Item(F, "fir_filter_length", split, seed, {"fs_hz": fs, "f_max_hz": f_c},
                q, str(length), "integer",
                rationale=f"trans={_fmt(trans)} Hz -> ceil(3.3*{fs}/{_fmt(trans)}) -> odd {length}")


@template(F, "fir_group_delay_ms")
def fir_group_delay_ms(rng, split, seed) -> Item:
    fs = draw(rng, F, "fs_hz", split)
    f_c = draw(rng, F, "f_max_hz", split)
    trans, length = _mne_rules(fs, f_c)
    delay_ms = Fraction(length - 1, 2) / fs * 1000
    if near_half(float(delay_ms) * 10, eps=0.01):
        raise Resample
    answer = round(float(delay_ms), 1)
    q = (
        f"You design a linear-phase FIR low-pass filter for EEG sampled at {fs} Hz with a cutoff of "
        f"{f_c} Hz. {FIR_RULES} You apply it causally (a single forward pass, not zero-phase), so every "
        f"component is delayed by the filter's group delay of (length - 1) / 2 samples.\n\n"
        f"By how many milliseconds are ERP latencies shifted? Round to one decimal place."
    )
    return Item(F, "fir_group_delay_ms", split, seed, {"fs_hz": fs, "f_max_hz": f_c},
                q, f"{answer:.1f}", "number", tolerance=0.05,
                rationale=f"length {length} -> {(length - 1) // 2} samples / {fs} Hz = {float(delay_ms):.3f} ms")
