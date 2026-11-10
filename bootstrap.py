"""Connect the repair agent to the scripted model and Docker-backed tools."""

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable

from adapters.models.ollama import OllamaClient
from adapters.models.scripted import ScriptedClient
from adapters.sandbox import runner
from adapters.sandbox.repair_tools import RepairTools
from core.models import TestResult
from core.ports import ChatClient, RepairToolsPort, ResearchPort
from core.repair_agent import RepairAgent
from core.repair_service import RepairService

TraceCallback = Callable[[Mapping[str, object]], None]


def _emit(callback: TraceCallback | None, event: dict[str, object]) -> None:
    """Send one live trace event to an optional observer."""
    if callback is not None:
        callback(event)


class _TracingClient:
    """Forward chat calls and report their visible response events."""

    def __init__(
        self,
        client: ChatClient,
        trace_callback: TraceCallback | None,
    ) -> None:
        self._client = client
        self._trace_callback = trace_callback

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Return the next reply and emit a response event immediately."""
        try:
            reply = self._client.chat(messages)
        except Exception as error:
            _emit(
                self._trace_callback,
                {"event": "model_response", "error": f"Client error: {error}"},
            )
            raise
        _emit(self._trace_callback, {"event": "model_response", "valid": True})
        return reply


class _TracingTools:
    """Forward permitted tool calls and report results as they complete."""

    def __init__(
        self,
        tools: RepairTools,
        trace_callback: TraceCallback | None,
    ) -> None:
        self._tools = tools
        self._trace_callback = trace_callback

    @property
    def workspace(self) -> Path:
        return self._tools.workspace

    @property
    def changed_files(self) -> list[str]:
        return self._tools.changed_files

    def read_file(self, path: str) -> str:
        """Read a workspace file and emit its visible result."""
        try:
            output = self._tools.read_file(path)
        except Exception as error:
            _emit(
                self._trace_callback,
                {"event": "tool_result", "tool": "read_file", "error": f"Tool error: {error}"},
            )
            raise
        _emit(
            self._trace_callback,
            {"event": "tool_result", "tool": "read_file", "output": output},
        )
        return output

    def edit_file(self, path: str, content: str) -> None:
        """Edit a workspace file and emit its visible result."""
        try:
            self._tools.edit_file(path, content)
        except Exception as error:
            _emit(
                self._trace_callback,
                {"event": "tool_result", "tool": "edit_file", "error": f"Tool error: {error}"},
            )
            raise
        _emit(
            self._trace_callback,
            {"event": "tool_result", "tool": "edit_file", "output": f"Edited {path}."},
        )

    def run_tests(self) -> TestResult:
        """Run sandboxed tests and emit their structured result."""
        try:
            result = self._tools.run_tests()
        except Exception as error:
            _emit(
                self._trace_callback,
                {"event": "tool_result", "tool": "run_tests", "error": f"Tool error: {error}"},
            )
            raise
        _emit(
            self._trace_callback,
            {
                "event": "tool_result",
                "tool": "run_tests",
                "passed": result.passed,
                "returncode": result.returncode,
                "output": result.output,
            },
        )
        return result


def _reply(tool: str, **fields: str) -> str:
    """Encode one scripted tool action as JSON."""
    return json.dumps({"tool": tool, **fields})


_SCRIPTED_REPLIES = {
    "arithmetic": (
        _reply("read_file", path="test_calculator.py", summary="Read the tests"),
        _reply("run_tests", summary="Reproduce the failure"),
        _reply(
            "edit_file",
            path="calculator.py",
            content=(
                'def add(left, right):\n'
                '    """Return the sum of two numbers."""\n'
                "    return left + right\n"
            ),
            summary="Correct addition",
        ),
        _reply("run_tests", summary="Verify the repair"),
    ),
    "balanced_brackets": (
        _reply("read_file", path="test_brackets.py", summary="Read the tests"),
        _reply("run_tests", summary="Reproduce the failure"),
        _reply(
            "edit_file",
            path="brackets.py",
            content=(
                'def is_balanced(text):\n'
                '    """Return whether parentheses, brackets, and braces are properly nested."""\n'
                '    opening = "([{"\n'
                '    pairs = {")": "(", "]": "[", "}": "{"}\n'
                "    stack = []\n"
                "    for character in text:\n"
                "        if character in opening:\n"
                "            stack.append(character)\n"
                "        elif character in pairs:\n"
                "            if not stack or stack.pop() != pairs[character]:\n"
                "                return False\n"
                "    return not stack\n"
            ),
            summary="Match bracket types",
        ),
        _reply("run_tests", summary="Verify the repair"),
    ),
    "median": (
        _reply("read_file", path="test_statistics_tools.py", summary="Read the tests"),
        _reply("run_tests", summary="Reproduce the failure"),
        _reply(
            "edit_file",
            path="statistics_tools.py",
            content=(
                "def median(values):\n"
                '    """Return the median value without changing the input sequence."""\n'
                "    ordered = sorted(values)\n"
                "    middle = len(ordered) // 2\n"
                "    if len(ordered) % 2:\n"
                "        return ordered[middle]\n"
                "    return (ordered[middle - 1] + ordered[middle]) / 2\n"
            ),
            summary="Handle even-length input",
        ),
        _reply("run_tests", summary="Verify the repair"),
    ),
    "multi_function": (
        _reply("read_file", path="test_text_tools.py", summary="Read the tests"),
        _reply("run_tests", summary="Reproduce the failure"),
        _reply(
            "edit_file",
            path="text_tools.py",
            content=(
                "def normalize_name(name):\n"
                '    """Trim surrounding whitespace and lowercase a supplied name."""\n'
                "    return name.strip().lower()\n"
                "\n"
                "\n"
                "def greeting(name):\n"
                '    """Create a greeting from a normalized name."""\n'
                '    return f"Hello, {normalize_name(name).title()}!"\n'
                "\n"
                "\n"
                "def initials(name):\n"
                '    """Return uppercase initials for the whitespace-separated parts of a name."""\n'
                '    return "".join(part[0].upper() for part in name.split())\n'
            ),
            summary="Fix greeting capitalization",
        ),
        _reply("run_tests", summary="Verify the repair"),
    ),
    "unique_words": (
        _reply("read_file", path="test_word_tools.py", summary="Read the tests"),
        _reply("run_tests", summary="Reproduce the failure"),
        _reply(
            "edit_file",
            path="word_tools.py",
            content=(
                "def unique_words(text):\n"
                '    """Return lowercase words once, in the order first seen."""\n'
                "    return list(dict.fromkeys(text.lower().split()))\n"
            ),
            summary="Preserve first appearance",
        ),
        _reply("run_tests", summary="Verify the repair"),
    ),
}


def create_repair_service(
    provider: str = "scripted",
    trace_callback: TraceCallback | None = None,
    model: str | None = None,
    temperature: float | None = None,
    research_enabled: bool = False,
) -> RepairService:
    """Build the repair service with a scripted or local Ollama client."""
    if provider not in {"scripted", "ollama"}:
        raise ValueError(f"unsupported provider: {provider}")
    if provider == "scripted" and (model is not None or temperature is not None):
        raise ValueError("--model and --temperature are only supported with --provider ollama")
    research_factory: Callable[[], ResearchPort] | None = None
    if research_enabled:
        from adapters.research.github_research import GitHubResearch

        research_token = os.environ.get("GITHUB_TOKEN")
        research_factory = lambda: GitHubResearch(token=research_token)

    def client_for_case(case: str) -> ChatClient:
        if provider == "ollama":
            client: ChatClient = OllamaClient(model=model, temperature=temperature)
        else:
            try:
                replies = _SCRIPTED_REPLIES[case]
            except KeyError as error:
                raise ValueError(f"no scripted replies for evaluation: {case}") from error
            client = ScriptedClient(replies)
        return _TracingClient(client, trace_callback)

    def tools_for_workspace(workspace: Path) -> RepairToolsPort:
        return _TracingTools(RepairTools(workspace), trace_callback)

    def create_agent(client: ChatClient, tools: RepairToolsPort) -> RepairAgent:
        research = research_factory() if research_factory is not None else None
        return RepairAgent(client, tools, research=research)

    return RepairService(
        workspace_factory=runner.create_workspace,
        tools_factory=tools_for_workspace,
        client_factory=client_for_case,
        agent_factory=create_agent,
        trace_callback=trace_callback,
    )
