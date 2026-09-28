"""Family 2: trial counts and power calculations (normal approximation, z-values supplied)."""

from __future__ import annotations

import math
from fractions import Fraction as Q
from statistics import NormalDist

from ..splits import draw
from ..templates import Item, Resample, near_half, near_integer, template

F = "power"
PHI = NormalDist()


def z(p: float) -> float:
    return round(PHI.inv_cdf(p), 3)


def ceil_safe(x: float) -> int:
    """ceil(), refusing values whose ceiling would depend on rounding of the inputs."""
    if near_integer(x):
        raise Resample
    return math.ceil(x)


def exact(x) -> Q:
    return Q(str(x))


def _common(rng, split):
    d = draw(rng, F, "effect_size", split)
    alpha = draw(rng, F, "alpha", split)
    power = draw(rng, F, "power", split)
    return d, alpha, power, z(1 - alpha / 2), z(power)


def _zline(alpha, power, za, zb) -> str:
    return (f"Use the normal approximation with z(1 - alpha/2) = {za:.3f} for alpha = {alpha} "
            f"(two-sided) and z(power) = {zb:.3f} for power = {power:.2f}.")


@template(F, "n_per_group_two_sample")
def n_per_group_two_sample(rng, split, seed) -> Item:
    d, alpha, power, za, zb = _common(rng, split)
    n = ceil_safe(2 * ((za + zb) / d) ** 2)
    q = (
        f"You compare mean P300 amplitude between two independent groups (patients vs. controls). "
        f"The expected standardized effect size is Cohen's d = {d}. {_zline(alpha, power, za, zb)} "
        f"For two independent groups, n per group = 2 x ((z(1 - alpha/2) + z(power)) / d)^2, rounded up.\n\n"
        f"How many participants do you need per group?"
    )
    return Item(F, "n_per_group_two_sample", split, seed,
                {"effect_size": d, "alpha": alpha, "power": power}, q, str(n), "integer",
                rationale=f"2*(({za}+{zb})/{d})^2 = {2 * ((za + zb) / d) ** 2:.3f} -> {n}")


@template(F, "n_paired")
def n_paired(rng, split, seed) -> Item:
    d, alpha, power, za, zb = _common(rng, split)
    n = ceil_safe(((za + zb) / d) ** 2)
    q = (
        f"In a within-subject BCI study, every participant does both a motor-imagery and a rest "
        f"condition, and you test the paired difference in mu-band power. The expected effect size of "
        f"the paired difference is d_z = {d}. {_zline(alpha, power, za, zb)} For a paired design, "
        f"n = ((z(1 - alpha/2) + z(power)) / d_z)^2, rounded up. Unlike a two-group design, a paired "
        f"design has no factor of 2.\n\nHow many participants do you need?"
    )
    return Item(F, "n_paired", split, seed, {"effect_size": d, "alpha": alpha, "power": power},
                q, str(n), "integer",
                rationale=f"(({za}+{zb})/{d})^2 = {((za + zb) / d) ** 2:.3f} -> {n}")


@template(F, "recruit_with_exclusion")
def recruit_with_exclusion(rng, split, seed) -> Item:
    d, alpha, power, za, zb = _common(rng, split)
    e = draw(rng, F, "rejection_rate", split)
    n = ceil_safe(((za + zb) / d) ** 2)
    recruit = math.ceil(n / (1 - exact(e)))
    q = (
        f"A paired-design EEG study expects an effect size of d_z = {d}. {_zline(alpha, power, za, zb)} "
        f"The required analyzable sample is n = ((z(1 - alpha/2) + z(power)) / d_z)^2, rounded up. From "
        f"pilot data you expect to exclude {round(e * 100)}% of recruited participants for excessive "
        f"artifacts, so recruit = n / (1 - exclusion rate), rounded up.\n\n"
        f"How many participants must you recruit?"
    )
    return Item(F, "recruit_with_exclusion", split, seed,
                {"effect_size": d, "alpha": alpha, "power": power, "exclusion_rate": e},
                q, str(recruit), "integer", rationale=f"n={n} -> {n}/(1-{e}) -> {recruit}")


@template(F, "recorded_trials_per_session")
def recorded_trials_per_session(rng, split, seed) -> Item:
    d = draw(rng, F, "effect_size", split)
    r = draw(rng, F, "rejection_rate", split)
    c = draw(rng, F, "conditions", split)
    target = rng.choice([1.5, 2.0, 3.0, 4.0, 5.0])
    m = math.ceil((exact(target) / exact(d)) ** 2)
    per_cond = math.ceil(m / (1 - exact(r)))
    total = c * per_cond
    q = (
        f"A single-trial ERP difference has a standardized effect size of {d}. Averaging m trials "
        f"increases the effect size by a factor of sqrt(m), and you need the trial-averaged effect size "
        f"to reach at least {target}, so m = (target / single-trial effect)^2, rounded up, clean trials "
        f"per condition. You expect to reject {round(r * 100)}% of recorded trials, so trials recorded per "
        f"condition = m / (1 - rejection rate), rounded up. The experiment has {c} conditions.\n\n"
        f"How many trials in total must you record per participant?"
    )
    return Item(F, "recorded_trials_per_session", split, seed,
                {"effect_size": d, "rejection_rate": r, "conditions": c, "target": target},
                q, str(total), "integer",
                rationale=f"m={m}, per condition {per_cond}, x{c} = {total}")


@template(F, "session_minutes")
def session_minutes(rng, split, seed) -> Item:
    d = draw(rng, F, "effect_size", split)
    r = draw(rng, F, "rejection_rate", split)
    c = draw(rng, F, "conditions", split)
    soa = draw(rng, F, "soa_s", split)
    target = rng.choice([2.0, 3.0, 4.0, 5.0])
    m = math.ceil((exact(target) / exact(d)) ** 2)
    per_cond = math.ceil(m / (1 - exact(r)))
    minutes = math.ceil(c * per_cond * exact(soa) / 60)
    q = (
        f"You need a trial-averaged ERP effect size of {target} from a single-trial effect size of {d}. "
        f"Averaging m trials multiplies the effect size by sqrt(m), so m = (target / single-trial "
        f"effect)^2 clean trials per condition, rounded up. You expect to reject {round(r * 100)}% of "
        f"recorded trials, so recorded per condition = m / (1 - rejection rate), rounded up. There are "
        f"{c} conditions and one trial starts every {soa} s.\n\n"
        f"How many minutes of recording does each participant need, rounded up to a whole minute?"
    )
    return Item(F, "session_minutes", split, seed,
                {"effect_size": d, "rejection_rate": r, "conditions": c, "soa_s": soa, "target": target},
                q, str(minutes), "integer",
                rationale=f"{c}x{per_cond} trials x {soa}s = {float(c * per_cond * exact(soa)):.1f}s -> {minutes} min")


@template(F, "bonferroni_n_per_group")
def bonferroni_n_per_group(rng, split, seed) -> Item:
    d = draw(rng, F, "effect_size", split)
    k = draw(rng, F, "n_comparisons", split)
    power = draw(rng, F, "power", split)
    alpha_fw = 0.05
    table = sorted({alpha_fw, alpha_fw / k, alpha_fw / (2 * k), 0.01}, reverse=True)
    zs = {a: z(1 - a / 2) for a in table}
    zb = z(power)
    n = ceil_safe(2 * ((zs[alpha_fw / k] + zb) / d) ** 2)
    rows = "\n".join(f"- per-test alpha = {a:.6g}: z(1 - alpha/2) = {zs[a]:.3f}" for a in table)
    q = (
        f"You test a two-group difference at each of {k} electrode clusters and Bonferroni-correct for "
        f"the {k} comparisons, keeping the family-wise alpha at {alpha_fw} (two-sided tests). The "
        f"expected Cohen's d is {d}, and you want power {power:.2f} per test (z(power) = {zb:.3f}). "
        f"Two-sided z-values:\n{rows}\n"
        f"For two independent groups, n per group = 2 x ((z(1 - alpha/2) + z(power)) / d)^2, rounded up, "
        f"using the per-test alpha.\n\nHow many participants do you need per group?"
    )
    return Item(F, "bonferroni_n_per_group", split, seed,
                {"effect_size": d, "n_comparisons": k, "power": power}, q, str(n), "integer",
                rationale=f"alpha/k={alpha_fw / k:.6g}, z={zs[alpha_fw / k]} -> {n}")


@template(F, "achieved_power_percent")
def achieved_power_percent(rng, split, seed) -> Item:
    d = draw(rng, F, "effect_size", split)
    alpha = draw(rng, F, "alpha", split)
    za = z(1 - alpha / 2)
    n = rng.choice([10, 12, 15, 16, 20, 24, 25, 30, 40, 50, 64, 80, 100, 150, 200])
    pct = 100 * PHI.cdf(d * math.sqrt(n / 2) - za)
    if near_half(pct, eps=0.05) or pct > 99.5 or pct < 5:
        raise Resample
    answer = round(pct)
    q = (
        f"A two-group EEG study has {n} participants per group and an expected Cohen's d of {d}. Using "
        f"the normal approximation, power = Phi(d x sqrt(n / 2) - z(1 - alpha/2)), where Phi is the "
        f"standard normal CDF, z(1 - alpha/2) = {za:.3f} for alpha = {alpha} two-sided, and the "
        f"opposite tail is ignored.\n\n"
        f"What is the achieved power, as a percentage rounded to the nearest whole number?"
    )
    return Item(F, "achieved_power_percent", split, seed,
                {"effect_size": d, "alpha": alpha, "n_per_group": n}, q, str(answer), "number",
                tolerance=1.0, rationale=f"Phi({d * math.sqrt(n / 2) - za:.4f}) = {pct:.2f}%")
