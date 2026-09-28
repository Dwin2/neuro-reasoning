import asyncio

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.utils.loaders import taskset_class

from neuro_reasoning.taskset import NeuroReasoningConfig, NeuroReasoningTaskset


def make_trace(task, reply: str) -> vf.Trace:
    return vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data, key=task.key, hash=task.hash),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[MessageNode(message=vf.UserMessage(content=task.data.prompt_text)),
               MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True)],
    )


def test_loader_resolves():
    assert taskset_class("neuro-reasoning") is NeuroReasoningTaskset


def test_scores_offline_every_template():
    tasks = list(NeuroReasoningTaskset(NeuroReasoningConfig(num_tasks=35)))
    assert len({t.data.template for t in tasks}) == 35
    assert len({t.key for t in tasks}) == 35
    for task in tasks:
        good = make_trace(task, f"Reasoning...\nANSWER: {task.data.answer}")
        asyncio.run(task.score(good))
        assert good.rewards["correct"].score == 1.0
        assert good.info["parsed_answer"] == task.data.answer
        bad = make_trace(task, "Reasoning...\nANSWER: zz")
        asyncio.run(task.score(bad))
        assert bad.rewards["correct"].score == 0.0 and bad.metrics["format_ok"] == 1.0
        missing = make_trace(task, "I think it is the first option.")
        asyncio.run(task.score(missing))
        assert missing.rewards["correct"].score == 0.0 and missing.metrics["format_ok"] == 0.0


def test_splits_via_config():
    for split in ("train", "heldout", "anchor"):
        tasks = list(NeuroReasoningTaskset(NeuroReasoningConfig(num_tasks=10, split=split)))
        assert all(t.data.split == split for t in tasks)
    fam = list(NeuroReasoningTaskset(NeuroReasoningConfig(num_tasks=14, families=["power"])))
    assert {t.data.family for t in fam} == {"power"}
