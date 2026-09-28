"""Family 5: preprocessing pipeline ordering. Exactly one valid order, proven by brute force.

Each context lists steps (a label: description) and facts. A fact (x, y, why) means step x must come
before step y, and ``why`` says what goes wrong if y is done first. Only genuine order dependencies
are listed; steps that commute (e.g. re-referencing and high-pass filtering, both linear) have no
fact. For each item the generator picks a step subset, keeps the facts among those steps, and
accepts the subset only if exactly one permutation satisfies every fact."""

from __future__ import annotations

import itertools
import random
import string
from functools import cache

from ..splits import draw
from ..templates import Item, Resample, template

F = "pipeline_order"

CONTEXTS: dict[str, dict] = {
    "erp_p300": {
        "title": "an oddball ERP study measuring the P300",
        "steps": {
            "interp": "Interpolate the channels marked bad",
            "reref": "Re-reference to the average of all channels",
            "hp": "High-pass filter the continuous recording at 0.1 Hz",
            "epoch": "Cut epochs from -200 to 800 ms around stimulus onset",
            "baseline": "Baseline-correct each epoch with its -200 to 0 ms mean",
            "reject": "Reject epochs in which any sample exceeds +/-100 uV (absolute amplitude)",
            "average": "Average the surviving epochs within each condition",
            "diff": "Compute the rare-minus-frequent difference wave",
            "measure": "Measure the difference wave's mean amplitude from 300 to 500 ms",
        },
        "facts": [
            ("interp", "reref", "an average reference computed while a bad channel is present spreads that channel's noise into every channel"),
            ("interp", "reject", "the uninterpolated bad channel would push nearly every epoch over the threshold"),
            ("reref", "reject", "re-referencing afterwards changes every amplitude, so epochs judged clean may no longer be clean"),
            ("hp", "epoch", "a 1-s epoch is too short for a 0.1 Hz filter, which would mostly produce edge artifacts"),
            ("epoch", "baseline", "there is no pre-stimulus window to subtract until the data are cut around stimulus onset"),
            ("baseline", "reject", "an absolute threshold is meaningless while each epoch still carries its own voltage offset"),
            ("reject", "average", "artifact epochs cannot be removed once they have been averaged in"),
            ("average", "diff", "the difference wave is defined between the condition averages"),
            ("diff", "measure", "the measurement is taken on the difference wave, which does not exist yet"),
        ],
    },
    "ica_ocular": {
        "title": "removing eye-blink artifacts with ICA",
        "steps": {
            "copy_hp": "Make a copy of the continuous data and high-pass filter the copy at 1 Hz",
            "fit": "Fit ICA on the 1 Hz-filtered copy",
            "identify": "Identify the components that correlate with the EOG channel",
            "apply": "Remove the EOG-related ICA components from the original 0.1 Hz-filtered continuous data",
            "epoch": "Cut epochs around each stimulus",
            "reject": "Reject epochs whose peak-to-peak amplitude still exceeds 150 uV",
            "average": "Average the surviving epochs within each condition",
            "grand": "Compute the grand average across participants",
        },
        "facts": [
            ("copy_hp", "fit", "slow drifts in unfiltered data make the ICA decomposition unstable, and the filtered copy does not exist yet"),
            ("fit", "identify", "there are no components to inspect before ICA has been fit"),
            ("identify", "apply", "you do not yet know which components to remove"),
            ("apply", "reject", "epochs containing blinks that ICA would have repaired are thrown away"),
            ("epoch", "reject", "epoch rejection needs epochs"),
            ("reject", "average", "artifact epochs cannot be removed once they have been averaged in"),
            ("epoch", "average", "averaging within a condition needs epochs"),
            ("average", "grand", "the grand average is computed from each participant's condition averages"),
        ],
    },
    "decoding_normalization": {
        "title": "a cross-subject decoding analysis",
        "steps": {
            "epoch": "Cut labeled epochs from every participant's recording",
            "split": "Split participants into a training group and a test group",
            "stats": "Compute channel-wise mean and standard deviation on the training group only",
            "zscore": "Z-score the training and test epochs using the training-group statistics",
            "train": "Train the classifier on the training group's epochs",
            "predict": "Predict labels for the test group's epochs",
            "score": "Compute balanced accuracy on the test group",
            "pvalue": "Compute a p-value for the observed balanced accuracy against a label-permutation null",
        },
        "facts": [
            ("split", "stats", "statistics computed before the split would include test participants, leaking them into training"),
            ("epoch", "stats", "the statistics are computed over epochs"),
            ("stats", "zscore", "there are no statistics to normalize with yet"),
            ("zscore", "train", "a classifier trained on unnormalized data cannot be applied to normalized test data"),
            ("train", "predict", "there is no trained model to predict with"),
            ("predict", "score", "accuracy requires predictions"),
            ("score", "pvalue", "the p-value compares the observed accuracy, which is not yet known, to the null"),
        ],
    },
    "ssvep_spectrum": {
        "title": "an SSVEP frequency-recognition analysis",
        "steps": {
            "hp": "High-pass filter the continuous recording at 1 Hz",
            "epoch": "Cut a 4-s epoch for each stimulation trial",
            "crop": "Drop the first 0.5 s of each epoch to remove the onset transient",
            "window": "Apply a Hann taper to each cropped epoch",
            "fft": "Zero-pad each epoch to 10 s and compute its FFT",
            "snr": "Compute the SNR at each stimulation frequency relative to neighboring bins",
            "classify": "Classify each trial as the stimulation frequency with the highest SNR",
            "itr": "Compute the information transfer rate from classification accuracy",
        },
        "facts": [
            ("hp", "epoch", "filtering short epochs creates edge artifacts that distort the spectrum"),
            ("epoch", "crop", "the onset transient is defined relative to each trial's start, which requires epochs"),
            ("crop", "window", "cropping after tapering cuts off one end of the taper and leaves a sharp edge"),
            ("window", "fft", "the taper must be applied in the time domain before the transform"),
            ("fft", "snr", "SNR is read from the spectrum"),
            ("snr", "classify", "classification uses the SNR values"),
            ("classify", "itr", "the information transfer rate depends on the accuracy"),
        ],
    },
    "source_localization": {
        "title": "an EEG source-localization analysis",
        "steps": {
            "coreg": "Co-register the digitized electrode positions with the head model",
            "forward": "Compute the forward (lead-field) model",
            "avgref": "Apply an average reference to the epoched data",
            "noisecov": "Estimate the noise covariance from the pre-stimulus baseline",
            "inverse": "Compute the inverse operator",
            "apply": "Apply the inverse operator to the evoked response",
            "roi": "Extract the time course of a region of interest",
            "peak": "Find the peak latency in the region's time course",
            "stats": "Compare peak latencies between conditions",
        },
        "facts": [
            ("coreg", "forward", "the forward model needs to know where each electrode sits on the head"),
            ("forward", "inverse", "the inverse operator is built from the forward model"),
            ("avgref", "noisecov", "a noise covariance estimated in a different reference does not match the data"),
            ("noisecov", "inverse", "the inverse operator is built from the noise covariance"),
            ("inverse", "apply", "there is no inverse operator to apply yet"),
            ("apply", "roi", "region time courses exist only in source space"),
            ("roi", "peak", "the peak is found in the region's time course"),
            ("peak", "stats", "the comparison needs each condition's peak latency"),
        ],
    },
    "online_bci": {
        "title": "setting up an online motor-imagery BCI session",
        "steps": {
            "calib": "Record a calibration block of cued motor imagery",
            "bandpass": "Band-pass filter the calibration data at 8 to 30 Hz",
            "csp": "Fit CSP spatial filters on the calibration data",
            "lda": "Train an LDA classifier on the CSP log-variance features",
            "feedback": "Start the online feedback block with the trained model",
            "classify": "Classify each incoming 1-s window in real time",
            "log": "Compute the online accuracy for the block",
            "criterion": "Decide whether accuracy meets the 70% criterion for continuing",
        },
        "facts": [
            ("calib", "bandpass", "there is no calibration data to filter yet"),
            ("bandpass", "csp", "CSP fit on unfiltered data captures broadband drifts instead of mu and beta rhythms"),
            ("csp", "lda", "the LDA features are computed with the CSP filters"),
            ("lda", "feedback", "online feedback needs a trained classifier"),
            ("feedback", "classify", "there is no incoming data to classify until the online block starts"),
            ("classify", "log", "the online accuracy is computed from the real-time classifications"),
            ("log", "criterion", "the decision depends on the online accuracy"),
        ],
    },
    "nested_cv": {
        "title": "tuning and evaluating a cross-subject decoder with nested cross-validation",
        "steps": {
            "outer": "Split participants into outer training and outer test folds",
            "inner": "Split the outer-training participants into inner training and validation folds",
            "select": "Select hyperparameters by inner-validation accuracy",
            "refit": "Refit the model with those hyperparameters on all outer-training participants",
            "evaluate": "Evaluate the refit model once on the outer-test participants",
            "repeat": "Repeat for every outer fold and collect the outer-test accuracies",
            "report": "Report the mean and standard error across outer folds",
        },
        "facts": [
            ("outer", "inner", "inner folds drawn before the outer split can include outer-test participants"),
            ("inner", "select", "there are no validation folds to select with yet"),
            ("select", "refit", "the hyperparameters to refit with have not been chosen"),
            ("refit", "evaluate", "there is no final model to evaluate"),
            ("evaluate", "repeat", "each fold's outer-test accuracy is needed"),
            ("repeat", "report", "the summary needs every fold's accuracy"),
        ],
    },
}


def _count_orders(steps: tuple[str, ...], facts: list[tuple[str, str, str]]) -> tuple[int, tuple]:
    count, first = 0, ()
    for perm in itertools.permutations(steps):
        pos = {s: i for i, s in enumerate(perm)}
        if all(pos[a] < pos[b] for a, b, _ in facts):
            count += 1
            first = first or perm
            if count > 1:
                break
    return count, first


@cache
def unique_subsets(context: str, n: int) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    """Every n-step subset whose induced facts admit exactly one order, with that order."""
    ctx = CONTEXTS[context]
    out = []
    for subset in itertools.combinations(ctx["steps"], n):
        facts = [f for f in ctx["facts"] if f[0] in subset and f[1] in subset]
        count, order = _count_orders(subset, facts)
        if count == 1:
            out.append((subset, order))
    return tuple(out)


def _make(context: str):
    ctx = CONTEXTS[context]

    def fn(rng, split, seed) -> Item:
        n = draw(rng, F, "n_steps", split)
        options = unique_subsets(context, n)
        if not options:
            raise Resample
        subset, order = rng.choice(options)
        shuffler = random.Random(f"{context}:{rng.random()}")  # context-salted: no shared letter patterns
        shown = list(subset)
        shuffler.shuffle(shown)
        label = {s: string.ascii_uppercase[i] for i, s in enumerate(shown)}
        facts = [f for f in ctx["facts"] if f[0] in subset and f[1] in subset]
        shuffler.shuffle(facts)
        step_lines = "\n".join(f"{label[s]}. {ctx['steps'][s]}" for s in shown)
        fact_lines = "\n".join(
            f"- Doing step {label[b]} before step {label[a]} fails: {why}." for a, b, why in facts
        )
        q = (
            f"You are ordering the steps of {ctx['title']}. The steps, in arbitrary order:\n{step_lines}\n\n"
            f"Constraints:\n{fact_lines}\n\n"
            f"Exactly one order satisfies every constraint. Give the step letters in the order they "
            f"are performed, separated by commas."
        )
        answer = ",".join(label[s].lower() for s in order)
        return Item(F, context, split, seed,
                    {"n_steps": n, "steps": list(subset), "shown": shown}, q, answer, "ordering",
                    rationale=" -> ".join(order))

    fn.__name__ = context
    return fn


for _name in CONTEXTS:
    template(F, _name)(_make(_name))
