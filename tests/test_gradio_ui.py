"""Tests for provider-specific Gradio service configuration."""

from unittest.mock import Mock

import pytest

from adapters.gradio_ui import _format_trace_event, _selected_service
from core.repair_service import RepairService


def test_qwen_enables_langfuse_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compose the Qwen service with Langfuse enabled and research as selected."""
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.setenv(name, "configured")
    factory = Mock(return_value=Mock(spec=RepairService))
    callback = Mock()

    _selected_service("qwen", True, None, factory, callback)

    factory.assert_called_once_with(
        "ollama",
        model="qwen2.5-coder:14b",
        research_enabled=True,
        observe_enabled=True,
        trace_callback=callback,
    )


def test_qwen_requires_langfuse_environment_before_service_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Show a configuration error rather than silently skipping Qwen telemetry."""
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    factory = Mock()

    with pytest.raises(ValueError, match="Langfuse observability requires"):
        _selected_service("qwen", False, None, factory)

    factory.assert_not_called()


def test_non_qwen_providers_do_not_enable_langfuse() -> None:
    """Leave observability off for providers other than the Qwen option."""
    factory = Mock(return_value=Mock(spec=RepairService))

    _selected_service("ollama", False, None, factory)

    factory.assert_called_once_with("ollama", trace_callback=None)


def test_ui_trace_uses_cli_labels_and_indented_multiline_result() -> None:
    """Render the CLI-style labels and indent tool output in the UI transcript."""
    assert _format_trace_event(
        {"event": "repair_started", "case": "arithmetic"}
    ) == "[START] arithmetic"
    assert _format_trace_event(
        {
            "event": "model_response",
            "valid": True,
            "tool": "run_tests",
            "summary": "Reproduce failure",
        }
    ) == "[ACTION] RUN_TESTS - Reproduce failure"
    assert _format_trace_event(
        {
            "event": "tool_result",
            "tool": "run_tests",
            "passed": False,
            "returncode": 1,
            "output": "first line\nsecond line\n",
        }
    ) == (
        "[RESULT] run_tests\n"
        "  status: failed\n"
        "  return code: 1\n"
        "  first line\n"
        "  second line"
    )
    assert _format_trace_event(
        {
            "event": "repair_finished",
            "passed": True,
            "reason": "Tests passed after the repair.",
        }
    ) == "[FINISHED] PASS - Tests passed after the repair."
