import asyncio
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from openrlhf.algorithms.bifocalrl.legacy import math as grader
from openrlhf.algorithms.bifocalrl.legacy import timeout
from openrlhf.algorithms.bifocalrl.legacy.math_extraction import simplify_expression_string
from openrlhf.algorithms.bifocalrl.legacy.timeout import _timeout_decorator
from openrlhf.algorithms.bifocalrl.voting import is_math_equivalent, weighted_majority_voting


@_timeout_decorator(timeout_seconds=5)
def _process_probe(delay=0, pid_file=None, fail=False):
    if pid_file:
        Path(pid_file).write_text(str(os.getpid()))
    if fail:
        raise ValueError("probe failure")
    time.sleep(delay)
    return os.getpid()


@pytest.mark.parametrize("answer,truth,expected", [
    ("sqrt(2)/2", "1/sqrt(2)", True),
    ("x+x", "2*x", True),
    ("sqrt(2)/2", "sqrt(3)/2", False),
])
def test_symbolic_grading_from_working_thread(answer, truth, expected, monkeypatch):
    def reject_signal(*args):
        raise AssertionError("grading must not register a signal handler in the caller")

    monkeypatch.setattr(signal, "signal", reject_signal)
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(grader.grade, answer, truth).result() is expected
        assert executor.submit(is_math_equivalent, answer, truth).result() is expected


def test_voting_simplifies_equivalent_answers_from_async_thread():
    async def vote():
        return await asyncio.to_thread(weighted_majority_voting, ["x+x", "2*x", "3*x"], [1, 1, 1])

    result = asyncio.run(vote())
    assert result.selected_answer == "x+x"
    assert result.answer_clusters == {"x+x": 2, "3*x": 1}
    assert simplify_expression_string(expression_string="x+x") == "2*x"


def test_timeout_terminates_worker_and_next_call_recovers(tmp_path, caplog):
    # Load math modules before timing the actual operation.
    assert grader.are_equal_under_sympy("x+x", "2*x")
    with ThreadPoolExecutor(max_workers=1) as executor:
        original_pid = executor.submit(_process_probe).result()
        assert original_pid != os.getpid()
        assert executor.submit(_process_probe).result() == original_pid
        pid_file = tmp_path / "worker.pid"
        started = time.monotonic()
        assert executor.submit(_process_probe, 30, str(pid_file)).result() is False
        assert time.monotonic() - started < 9
        expired_pid = int(pid_file.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(expired_pid, 0)
        replacement_pid = executor.submit(_process_probe).result()
        assert type(replacement_pid) is int and replacement_pid != expired_pid
    assert "_process_probe exceeded its 5 second timeout" in caplog.text


def test_worker_errors_propagate_without_becoming_incorrect_answers():
    with pytest.raises(ValueError, match="probe failure"):
        _process_probe(fail=True)
    assert _process_probe() != os.getpid()


def test_timeout_fallbacks_preserve_boolean_and_string_results(monkeypatch, caplog):
    def expire(*args, **kwargs):
        raise TimeoutError("task exceeded its deadline")

    monkeypatch.setattr(timeout._POOL, "schedule", expire)
    assert grader.are_equal_under_sympy("x+x", "2*x") is False
    assert grader.is_latex_equal("x+x", "2*x") is False
    assert simplify_expression_string(expression_string="  x+x  ") == "x+x"
    assert caplog.text.count("second timeout") == 3


def test_math_verify_fallback_from_working_thread():
    pytest.importorskip("math_verify")
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(grader.is_latex_equal, r"\frac{1}{2}", "0.5").result() is True
        assert executor.submit(grader.is_latex_equal, "1", "2").result() is False
