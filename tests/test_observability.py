"""Tests for sanitized, best-effort Langfuse trajectory recording."""

import json
from pathlib import Path
from typing import Mapping, Sequence

from adapters.models.scripted import ScriptedClient
from adapters.observability.langfuse import LangfuseTrajectoryRecorder
from core.models import TestResult as StructuredTestResult
from core.ports import TraceRecorder
from core.repair_agent import RepairAgent
import main
import pytest


def _action(tool: str, **fields: str) -> str:
    """Encode a scripted action as JSON."""
    return json.dumps({"tool": tool, **fields})


class FakeTools:
    """In-memory repair tools that never execute project code."""

    workspace = Path("/workspace")

    def __init__(self) -> None:
        self.changed_files: list[str] = []
        self.test_results = [
            StructuredTestResult(False, "private test output", 1),
            StructuredTestResult(True, "private passing output", 0),
        ]

    def read_file(self, path: str) -> str:
        return "private source content"

    def edit_file(self, path: str, content: str) -> None:
        self.changed_files.append(path)

    def run_tests(self) -> StructuredTestResult:
        return self.test_results.pop(0)


class FakeObservation:
    """Record lifecycle calls made on an observation."""

    def __init__(self, events: list[tuple[object, ...]], name: str) -> None:
        self.events = events
        self.name = name
        self.trace_id = "trace-id"
        self.id = f"{name}-id"

    def update(self, **kwargs: object) -> None:
        self.events.append((self.name, "update", kwargs))

    def end(self) -> None:
        self.events.append((self.name, "end"))


class FakeLangfuse:
    """Capture Langfuse client calls without contacting Cloud."""

    def __init__(self, *, fail_start: bool = False) -> None:
        self.events: list[tuple[object, ...]] = []
        self.fail_start = fail_start

    def start_observation(self, **kwargs: object) -> FakeObservation:
        if self.fail_start:
            raise RuntimeError("recording unavailable")
        name = str(kwargs["name"])
        self.events.append(("start", name, kwargs))
        return FakeObservation(self.events, name)

    def flush(self) -> None:
        self.events.append(("flush",))


class FakeRecorder:
    """Recorder that fails every write to verify repair isolation."""

    def record(self, event: str, payload: Mapping[str, object]) -> None:
        raise RuntimeError("telemetry unavailable")


class CapturingClient(ScriptedClient):
    """Scripted model that retains replies for telemetry privacy checks."""

    def __init__(self, replies: Sequence[str]) -> None:
        super().__init__(replies)
        self.replies = list(replies)


def _successful_repair(recorder: TraceRecorder) -> object:
    """Run a successful repair without external services."""
    return RepairAgent(
        CapturingClient(
            [
                _action("read_file", path="test.py", summary="Read local test"),
                _action("run_tests", summary="Reproduce failure"),
                _action(
                    "edit_file",
                    path="source.py",
                    content="replacement secret content",
                    summary="Apply focused fix",
                ),
                _action("run_tests", summary="Verify repair"),
            ]
        ),
        FakeTools(),
        recorder=recorder,
    ).run("arithmetic", ["source.py", "test.py"])


def test_langfuse_records_ordered_children_then_closes_and_flushes() -> None:
    """Record one root, event children, and flush only after repair completion."""
    client = FakeLangfuse()
    recorder = LangfuseTrajectoryRecorder("pk", "sk", "https://cloud.example", client=client)

    result = _successful_repair(recorder)

    assert result.passed is True
    names = [event[1] for event in client.events if event[0] == "start"]
    assert names == [
        "repair",
        "model_response",
        "tool_result",
        "model_response",
        "tool_result",
        "model_response",
        "tool_result",
        "model_response",
        "tool_result",
    ]
    assert client.events[-3][0:2] == ("repair", "update")
    assert client.events[-2] == ("repair", "end")
    assert client.events[-1] == ("flush",)

    serialized = repr(client.events)
    for private_value in (
        "private source content",
        "replacement secret content",
        "private test output",
        "private passing output",
        "test.py",
    ):
        assert private_value not in serialized
    for event in client.events:
        if event[0] == "start" and event[1] != "repair":
            payload = event[2]["input"]
            assert set(payload) <= {
                "case_name",
                "step_number",
                "requested_tool",
                "public_summary",
                "passed",
                "return_code",
                "changed_relative_file_names",
                "final_outcome",
            }


def test_recorder_failure_does_not_change_repair_result() -> None:
    """Ignore recorder errors so observability cannot stop a repair."""
    result = _successful_repair(FakeRecorder())

    assert result.passed is True
    assert result.changed_files == ["source.py"]


def test_langfuse_adapter_failure_is_best_effort_and_flushes() -> None:
    """A child-observation creation failure must not escape record()."""
    client = FakeLangfuse(fail_start=True)
    recorder = LangfuseTrajectoryRecorder("pk", "sk", "https://cloud.example", client=client)

    recorder.record("repair_started", {"case_name": "arithmetic"})
    recorder.record("model_response", {"step_number": 1})
    recorder.record("repair_finished", {"passed": False, "final_outcome": "failed"})

    assert client.events == [("flush",)]


def test_observe_requires_all_cloud_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject missing Cloud credentials before composing or starting a repair."""
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    called = False

    def fail_if_composed(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("repair service must not be composed")

    monkeypatch.setattr(main, "create_repair_service", fail_if_composed)
    status = main.main(["repair", "evals/arithmetic", "--observe"])

    assert status == 2
    assert called is False
    assert "LANGFUSE_BASE_URL" in capsys.readouterr().err
