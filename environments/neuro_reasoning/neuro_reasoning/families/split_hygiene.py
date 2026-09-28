"""Family 3: cross-subject split hygiene. Fixed answer set, exactly one flaw (or none) per item.

Every scenario states the claim the split must support; whether an overlap is a flaw depends on
that claim (the same participant in train and test is fine for a within-subject claim)."""

from __future__ import annotations

from ..splits import draw
from ..templates import Item, template

F = "split_hygiene"
OPTIONS = (
    "(a) temporal leakage: some training and test samples contain the same time points of the "
    "same recording (for example, overlapping windows or epochs placed on both sides of the split)\n"
    "(b) stimulus-level overlap: the same stimuli appear in both the training and the test data\n"
    "(c) no problem: the split supports the stated claim\n"
    "(d) subject identity leakage: data from the same participant appear in both the training and "
    "the test data, although the claim is about unseen participants"
)
LETTER = {"temporal": "a", "stimulus": "b", "none": "c", "subject": "d"}


def _item(name, split, seed, params, claim, design, label) -> Item:
    q = (
        f"Claim: {claim}\n\nEvaluation design: {design}\n\n"
        f"Which single description fits this design?\n{OPTIONS}\n\nAnswer with the letter."
    )
    return Item(F, name, split, seed, {**params, "label": label}, q, LETTER[label], "choice",
                rationale=label)


def _ppl(k: int) -> str:
    return "participant is" if k == 1 else f"{k} participants are"


def _pct(rng) -> int:
    return rng.choice([10, 15, 20, 25, 30])


@template(F, "cross_subject_novel_images")
def cross_subject_novel_images(rng, split, seed) -> Item:
    n = draw(rng, F, "n_subjects", split)
    m = draw(rng, F, "n_images", split)
    k = min(draw(rng, F, "n_test_subjects", split), n - 1)
    p = _pct(rng)
    label = rng.choice(["none", "subject", "stimulus"])
    claim = (f"an EEG image decoder generalizes to participants it has never seen and to images it "
             f"has never seen.")
    head = f"{n} participants each viewed the same {m} natural images once each. "
    design = {
        "none": head + f"The last {_ppl(k)} held out as the test set. The {m} images are "
                f"split into disjoint training ({100 - p}%) and test ({p}%) sets. The model is trained "
                f"on training-image trials from the other {n - k} participants and evaluated on "
                f"test-image trials from the held-out participants.",
        "subject": head + f"The {m} images are split into disjoint training ({100 - p}%) and test ({p}%) "
                   f"sets. Trials from all {n} participants are pooled; the model is trained on every "
                   f"participant's training-image trials and evaluated on every participant's "
                   f"test-image trials.",
        "stimulus": head + f"The last {_ppl(k)} held out as the test set. The model is "
                    f"trained on all {m} images from the other {n - k} participants and evaluated on the "
                    f"held-out participants' trials for a random {p}% of those same {m} images.",
    }[label]
    return _item("cross_subject_novel_images", split, seed, {"n_subjects": n, "n_images": m, "k": k},
                 claim, design, label)


@template(F, "within_subject_future_windows")
def within_subject_future_windows(rng, split, seed) -> Item:
    n = draw(rng, F, "n_subjects", split)
    w = rng.choice([1, 2, 4])
    label = rng.choice(["temporal", "none"])
    claim = ("a motor-imagery decoder, trained on one participant's data, classifies that same "
             "participant's future data (a within-subject claim).")
    head = f"Each of {n} participants performed 30 minutes of continuous motor imagery. "
    design = {
        "temporal": head + f"Each recording is cut into {w}-s windows with a {w / 2:g}-s step (50% "
                    f"overlap), and the windows are randomly shuffled into 80% training and 20% test "
                    f"per participant.",
        "none": head + f"Each recording is cut into non-overlapping {w}-s windows. Per participant, the "
                f"first 24 minutes are used for training, a {2 * w}-s gap is discarded, and the "
                f"remaining minutes are used for testing.",
    }[label]
    return _item("within_subject_future_windows", split, seed, {"n_subjects": n, "window_s": w},
                 claim, design, label)


@template(F, "calibration_transfer")
def calibration_transfer(rng, split, seed) -> Item:
    n = draw(rng, F, "n_subjects", split)
    label = rng.choice(["none", "temporal", "subject"])
    claim = ("a decoder pretrained on other people needs only a short calibration from a new user; "
             "the new user's data are never seen during pretraining.")
    head = (f"The dataset has {n} participants with 6 motor-imagery blocks each. For each target "
            f"participant in turn: ")
    design = {
        "none": head + f"pretrain on the other {n - 1} participants, fine-tune on the target's first "
                f"block, and test on the target's blocks 2 to 6. All windows are non-overlapping.",
        "temporal": head + f"pretrain on the other {n - 1} participants. Cut all 6 of the target's "
                    f"blocks into 2-s windows with a 0.5-s step, fine-tune on a random 15% of these "
                    f"windows, and test on the rest.",
        "subject": head + f"pretrain one model on all {n} participants' data, fine-tune on the target's "
                   f"first block, and test on the target's blocks 2 to 6. All windows are "
                   f"non-overlapping.",
    }[label]
    return _item("calibration_transfer", split, seed, {"n_subjects": n}, claim, design, label)


@template(F, "repetition_split")
def repetition_split(rng, split, seed) -> Item:
    n = draw(rng, F, "n_subjects", split)
    m = draw(rng, F, "n_images", split)
    reps = rng.choice([4, 8, 10, 20])
    label = rng.choice(["stimulus", "none", "temporal"])
    claim = ("within each participant, an image decoder identifies images it was never trained on "
             "(a within-subject claim).")
    head = (f"Each of {n} participants viewed {m} images, each shown {reps} times in a rapid serial "
            f"visual presentation stream. ")
    design = {
        "stimulus": head + f"Repetitions 1 to {reps - 1} of every image are used for training and "
                    f"repetition {reps} of every image for testing.",
        "none": head + f"The {m} images are split into disjoint training and test sets. One image "
                f"starts every 1.5 s and epochs span 0 to 800 ms, so epochs never overlap.",
        "temporal": head + f"The {m} images are split into disjoint training and test sets. One image "
                    f"starts every 200 ms but each epoch spans -200 to 800 ms, so neighboring epochs "
                    f"share most of their samples, and the stream interleaves training and test images.",
    }[label]
    return _item("repetition_split", split, seed,
                 {"n_subjects": n, "n_images": m, "reps": reps}, claim, design, label)


@template(F, "ssvep_windows")
def ssvep_windows(rng, split, seed) -> Item:
    n = draw(rng, F, "n_subjects", split)
    t = rng.choice([4, 5, 6])
    label = rng.choice(["temporal", "none"])
    claim = ("an SSVEP classifier recognizes the target on new trials from the same participant "
             "(a within-subject claim).")
    head = f"{n} participants each completed 40 SSVEP trials of {t} s. "
    design = {
        "temporal": head + "Every trial is cut into 1-s windows with a 0.5-s step, and all windows are "
                    "randomly split into 80% training and 20% test per participant.",
        "none": head + "Every trial is cut into 1-s windows with a 0.5-s step. Trials, not windows, are "
                "split into 80% training and 20% test, so all windows of a trial fall on the same side.",
    }[label]
    return _item("ssvep_windows", split, seed, {"n_subjects": n, "trial_s": t}, claim, design, label)


@template(F, "merged_datasets")
def merged_datasets(rng, split, seed) -> Item:
    n = draw(rng, F, "n_subjects", split)
    n_b = max(2, n // 3)
    overlap = max(1, n_b // 3)
    label = rng.choice(["subject", "none"])
    claim = "a model trained on dataset A generalizes to the unseen participants of dataset B."
    design = {
        "subject": f"Dataset A has {n} participants and dataset B, from the same lab, has {n_b}. Each "
                   f"dataset assigns its own participant codes. The lab's records show {overlap} people "
                   f"took part in both studies. The model is trained on all of A and tested on all of B.",
        "none": f"Dataset A has {n} participants and dataset B, from the same lab, has {n_b}. Each "
                f"dataset assigns its own participant codes. The lab's records confirm that no one took "
                f"part in both studies. The model is trained on all of A and tested on all of B.",
    }[label]
    return _item("merged_datasets", split, seed, {"n_subjects": n, "n_b": n_b}, claim, design, label)


@template(F, "cross_session")
def cross_session(rng, split, seed) -> Item:
    n = draw(rng, F, "n_subjects", split)
    m = draw(rng, F, "n_images", split)
    label = rng.choice(["none", "stimulus"])
    claim = ("an image decoder trained on a participant's first session identifies new images in "
             "that participant's second session, recorded a week later (same participants in both).")
    head = f"{n} participants each completed two sessions a week apart. "
    design = {
        "none": head + f"Session 1 showed {m} images and session 2 showed {m // 5} different images. "
                f"The model is trained on session 1 and tested on session 2.",
        "stimulus": head + f"Both sessions showed the same {m} images. The model is trained on session 1 "
                    f"and tested on session 2.",
    }[label]
    return _item("cross_session", split, seed, {"n_subjects": n, "n_images": m}, claim, design, label)
