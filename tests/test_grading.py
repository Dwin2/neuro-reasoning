import pytest

from neuro_reasoning.grading import extract_answer, grade


@pytest.mark.parametrize("reply,expected", [
    ("work...\nANSWER: 42", "42"),
    ("ANSWER: 1\nmore thought\nANSWER: 2", "2"),
    ("<think>ANSWER: 9</think>\nFinal.\nANSWER: 7", "7"),
    ("<think>ANSWER: 9</think>\nno marker after thinking", None),
    ("answer: yes", "yes"),
    ("**ANSWER: c**", "c"),
    ("ANSWER: \\boxed{256}", "256"),
    ("no marker at all", None),
    ("", None),
])
def test_extract(reply, expected):
    assert extract_answer(reply) == expected


@pytest.mark.parametrize("atype,gold,pred,tol,score", [
    ("integer", "256", "256", 0, 1), ("integer", "256", "256 Hz", 0, 1),
    ("integer", "1000", "1,000", 0, 1), ("integer", "256", "257", 0, 0),
    ("integer", "63", "63.0", 0, 1), ("integer", "63", "63.4", 0, 0),
    ("number", "2.5", "2.54", 0.05, 1), ("number", "2.5", "2.56", 0.05, 0),
    ("number", "80", "81", 1.0, 1), ("number", "80", "82", 1.0, 0),
    ("boolean", "yes", "Yes.", 0, 1), ("boolean", "no", "no, because", 0, 1),
    ("boolean", "yes", "no", 0, 0), ("boolean", "yes", "maybe", 0, 0),
    ("choice", "d", "d", 0, 1), ("choice", "d", "(d) subject identity leakage", 0, 1),
    ("choice", "d", "D.", 0, 1), ("choice", "d", "<d>", 0, 1), ("integer", "42", "<42>", 0, 1), ("choice", "d", "b", 0, 0), ("choice", "a", "all of them", 0, 0),
    ("ordering", "c,a,b", "C, A, B", 0, 1), ("ordering", "c,a,b", "C -> A -> B", 0, 1),
    ("ordering", "c,a,b", "C,A,B (because filtering first)", 0, 1),
    ("ordering", "c,a,b", "A, C, B", 0, 0), ("ordering", "c,a,b", "C, A", 0, 0),
])
def test_grade(atype, gold, pred, tol, score):
    assert grade(atype, gold, pred, tolerance=tol) == score


def test_missing_prediction_scores_zero():
    for atype in ("integer", "number", "boolean", "choice", "ordering"):
        assert grade(atype, "1", None) == 0.0
