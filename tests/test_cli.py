"""Tests for readable terminal trace formatting."""

import main
import pytest


def test_trace_events_use_labeled_steps_and_indent_multiline_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep live trace events readable without printing raw JSON."""
    main._print_trace_event({"event": "repair_started", "case": "arithmetic"})
    main._print_trace_event(
        {
            "event": "model_response",
            "valid": True,
            "tool": "run_tests",
            "summary": "Reproduce failure",
        }
    )
    main._print_trace_event(
        {
            "event": "tool_result",
            "tool": "run_tests",
            "passed": False,
            "returncode": 1,
            "output": "line one\nline two\n",
        }
    )
    main._print_trace_event(
        {
            "event": "repair_finished",
            "passed": False,
            "reason": "Decision budget exhausted.",
        }
    )

    output = capsys.readouterr().out
    assert "[START] arithmetic" in output
    assert "[ACTION] RUN_TESTS - Reproduce failure" in output
    assert "[RESULT] run_tests" in output
    assert "  line one\n  line two" in output
    assert "[FINISHED] FAILED - Decision budget exhausted." in output
    assert "TRACE {" not in output
