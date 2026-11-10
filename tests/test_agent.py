"""Tests for the bounded repair agent using scripted replies and fake tools."""

import json
from pathlib import Path
from typing import Mapping, Sequence

from adapters.models.scripted import ScriptedClient
from core.models import TestResult as StructuredTestResult
from core.repair_agent import RepairAgent


def action(tool: str, **fields: str) -> str:
    """Encode a scripted tool action as JSON."""
    return json.dumps({"tool": tool, **fields})


class FakeTools:
    """In-memory repair tools that never execute project code."""

    def __init__(self, test_results: Sequence[StructuredTestResult] = ()) -> None:
        self.workspace = Path("/workspace")
        self.changed_files: list[str] = []
        self.test_results = list(test_results)
        self.calls: list[tuple[object, ...]] = []

    def read_file(self, path: str) -> str:
        self.calls.append(("read_file", path))
        return "source"

    def edit_file(self, path: str, content: str) -> None:
        self.calls.append(("edit_file", path, content))
        if path not in self.changed_files:
            self.changed_files.append(path)

    def run_tests(self) -> StructuredTestResult:
        self.calls.append(("run_tests",))
        return self.test_results.pop(0)


class CapturingScriptedClient(ScriptedClient):
    """Scripted client that keeps copies of the conversation it receives."""

    def __init__(self, replies: Sequence[str]) -> None:
        super().__init__(replies)
        self.conversations: list[list[Mapping[str, str]]] = []

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        self.conversations.append(list(messages))
        return super().chat(messages)


def test_successful_read_failure_edit_and_passing_test() -> None:
    """Require the full evidence sequence and succeed after a verified edit."""
    client = CapturingScriptedClient(
        [
            action("read_file", path="test_calculator.py", summary="Read test"),
            action("run_tests", summary="Reproduce failure"),
            action("edit_file", path="calculator.py", content="def add(a, b): return a + b"),
            action("run_tests", summary="Verify repair"),
        ]
    )
    tools = FakeTools(
        [
            StructuredTestResult(False, "1 failed", 1),
            StructuredTestResult(True, "1 passed", 0),
        ]
    )

    result = RepairAgent(client, tools).run(
        "arithmetic",
        ["calculator.py", "test_calculator.py"],
    )

    assert result.passed is True
    assert result.changed_files == ["calculator.py"]
    assert result.test_result == StructuredTestResult(True, "1 passed", 0)
    assert [call[0] for call in tools.calls] == [
        "read_file",
        "run_tests",
        "edit_file",
        "run_tests",
    ]
    assert [event["event"] for event in result.trace] == [
        "repair_started",
        "model_response",
        "tool_result",
        "model_response",
        "tool_result",
        "model_response",
        "tool_result",
        "model_response",
        "tool_result",
        "repair_finished",
    ]
    first_messages = client.conversations[0]
    assert first_messages[0]["role"] == "system"
    assert "arithmetic" in first_messages[1]["content"]
    assert "calculator.py" in first_messages[1]["content"]
    assert client.conversations[1][-2]["role"] == "assistant"
    assert "source" in client.conversations[1][-1]["content"]


def test_trace_callback_receives_the_stored_trace_in_order() -> None:
    """Publish each event as it is stored without changing the trace."""
    events: list[Mapping[str, object]] = []
    client = ScriptedClient(
        [
            action("read_file", path="calculator.py", summary="Inspect source"),
            action("run_tests", summary="Check baseline"),
        ]
    )
    tools = FakeTools([StructuredTestResult(True, "1 passed", 0)])

    result = RepairAgent(client, tools, trace_callback=events.append).run(
        "arithmetic",
        ["calculator.py"],
    )

    assert events == result.trace
    assert [event["event"] for event in events] == [
        "repair_started",
        "model_response",
        "tool_result",
        "model_response",
        "tool_result",
        "repair_finished",
    ]


def test_invalid_json_fails_without_calling_a_tool() -> None:
    """Reject malformed replies before dispatching any tool."""
    tools = FakeTools()
    result = RepairAgent(ScriptedClient(["not JSON"]), tools).run("arithmetic", [])

    assert result.passed is False
    assert tools.calls == []
    assert result.trace[-1]["event"] == "repair_finished"
    assert "Invalid model reply" in result.trace[-1]["reason"]


def test_edit_before_evidence_is_rejected_and_observed() -> None:
    """Do not edit early; report the policy error before asking again."""
    client = CapturingScriptedClient(
        [
            action("edit_file", path="calculator.py", content="broken"),
            action("read_file", path="calculator.py"),
        ]
    )
    tools = FakeTools()

    result = RepairAgent(client, tools).run("arithmetic", ["calculator.py"])

    assert result.passed is False
    assert not any(call[0] == "edit_file" for call in tools.calls)
    assert any(event.get("tool") == "policy_error" for event in result.trace)
    assert "Policy error" in client.conversations[1][-1]["content"]


def test_eight_decisions_exhaust_the_budget() -> None:
    """Stop after eight actions even if the client has more replies."""
    read_reply = action("read_file", path="calculator.py")
    client = ScriptedClient([read_reply] * 9)
    tools = FakeTools()

    result = RepairAgent(client, tools).run("arithmetic", ["calculator.py"])

    assert result.passed is False
    assert len(tools.calls) == 8
    assert result.trace[-1]["reason"] == "Decision budget exhausted after 8 decisions."


def test_passing_baseline_is_not_reported_as_a_repair() -> None:
    """A passing test run before any edit means there is no reproduced bug."""
    tools = FakeTools([StructuredTestResult(True, "2 passed", 0)])
    result = RepairAgent(
        ScriptedClient([action("run_tests")]),
        tools,
    ).run("arithmetic", [])

    assert result.passed is False
    assert result.test_result == StructuredTestResult(True, "2 passed", 0)
    assert "No reproducible bug" in result.trace[-1]["reason"]
    assert tools.calls == [("run_tests",)]


def test_client_error_returns_a_failed_repair_result() -> None:
    """Return a finished result when the scripted reply cards run out."""
    result = RepairAgent(ScriptedClient([]), FakeTools()).run("arithmetic", [])

    assert result.passed is False
    assert result.trace[-1]["event"] == "repair_finished"
    assert "Client error" in result.trace[-1]["reason"]


def test_tool_error_is_observed_and_agent_continues() -> None:
    """Expose tool failures to the model, which may then choose another action."""
    client = CapturingScriptedClient(
        [
            action("read_file", path="missing.py"),
            action("run_tests"),
        ]
    )
    tools = FakeTools([StructuredTestResult(True, "1 passed", 0)])

    def fail_read(path: str) -> str:
        tools.calls.append(("read_file", path))
        raise FileNotFoundError(path)

    tools.read_file = fail_read
    result = RepairAgent(client, tools).run("arithmetic", [])

    assert result.passed is False
    assert len(client.conversations) == 2
    assert "Tool error" in client.conversations[1][-1]["content"]
    assert any("Tool error" in str(event.get("error", "")) for event in result.trace)
