"""verifiers v1 adapter: wraps the pure-Python generator (templates, graders) as a Taskset."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

import verifiers.v1 as vf

from neuro_reasoning import families  # noqa: F401  (registers every template)
from neuro_reasoning.grading import extract_answer, grade
from neuro_reasoning.templates import generate

SYSTEM_PROMPT = (
    "You are answering a quantitative question about EEG brain-computer interface experimental "
    "design. Reason through the constraints, then end your reply with one final line in exactly "
    "this form:\n"
    "ANSWER: <value>\n"
    "Numbers: digits only, no units. Yes/no questions: yes or no. Multiple choice: the letter only. "
    "Orderings: the step letters in order, separated by commas."
)
FORMAT_REMINDER = "\n\nEnd your reply with a final line of the form: ANSWER: <value>"


class NeuroReasoningData(vf.TaskData):
    family: str
    template: str
    split: str
    seed: int
    params: dict[str, Any]
    answer: str
    answer_type: str
    tolerance: float = 0.0
    rationale: str = ""


class NeuroReasoningTask(vf.Task[NeuroReasoningData]):
    @property
    def key(self) -> str:
        d = self.data
        return f"{d.split}/{d.family}/{d.template}/{d.seed}"

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        pred = extract_answer(trace.last_reply)
        trace.info["parsed_answer"] = pred
        return grade(self.data.answer_type, self.data.answer, pred, tolerance=self.data.tolerance)

    @vf.metric
    async def format_ok(self, trace: vf.Trace) -> float:
        return float(extract_answer(trace.last_reply) is not None)

    async def validate(self, runtime: vf.Runtime) -> bool:
        gold = grade(self.data.answer_type, self.data.answer, self.data.answer,
                     tolerance=self.data.tolerance)
        wrong = grade(self.data.answer_type, self.data.answer, None)
        return gold == 1.0 and wrong == 0.0


class NeuroReasoningConfig(vf.TasksetConfig):
    num_tasks: int = 350
    """Items to generate (balanced round-robin over the selected templates)."""
    seed: int = 0
    split: Literal["train", "heldout", "anchor"] = "train"
    """train | heldout (disjoint values on each family's held-out axis) | anchor (literature values)."""
    families: list[str] | None = None
    """Restrict to these families; None = all five."""


class NeuroReasoningTaskset(vf.Taskset[NeuroReasoningTask, NeuroReasoningConfig]):
    def load(self) -> Iterator[NeuroReasoningTask]:
        c = self.config
        for i, it in enumerate(generate(c.num_tasks, c.seed, c.split, c.families)):
            yield NeuroReasoningTask(
                NeuroReasoningData(
                    idx=i, prompt=it.question + FORMAT_REMINDER, system_prompt=SYSTEM_PROMPT,
                    family=it.family, template=it.template, split=it.split, seed=it.seed,
                    params=it.params, answer=it.answer, answer_type=it.answer_type,
                    tolerance=it.tolerance, rationale=it.rationale,
                ),
                c.task,
            )
