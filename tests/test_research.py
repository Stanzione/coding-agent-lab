"""Tests for optional, bounded GitHub research without network access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence
from unittest.mock import patch

import pytest

import bootstrap
from adapters.models.scripted import ScriptedClient
from adapters.research.github_research import GitHubResearch
from core.models import TestResult as StructuredTestResult
from core.prompts import system_message
from core.repair_agent import RepairAgent


_SOURCE_URL = "https://github.com/example/project/blob/main/module.py"


class FakeResponse:
    """Provide a JSON response compatible with urllib's response context."""

    def __init__(self, value: dict[str, object]) -> None:
        self._body = json.dumps(value).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _item(
    *,
    private: bool = False,
    path: str = "module.py",
    snippet: str = "public code",
) -> dict[str, object]:
    """Create one GitHub code-search item."""
    return {
        "repository": {"full_name": "example/project", "private": private},
        "path": path,
        "html_url": _SOURCE_URL,
        "text_matches": [{"fragment": snippet}],
        "content": "full remote file must not be returned",
    }


class FakeTools:
    """In-memory repair tools that never execute project code."""

    workspace = Path("/workspace")
    changed_files: list[str] = []

    def __init__(self, test_results: Sequence[StructuredTestResult] = ()) -> None:
        self.test_results = list(test_results)
        self.calls: list[str] = []

    def read_file(self, path: str) -> str:
        self.calls.append("read_file")
        return "local code"

    def edit_file(self, path: str, content: str) -> None:
        self.calls.append("edit_file")

    def run_tests(self) -> StructuredTestResult:
        self.calls.append("run_tests")
        return self.test_results.pop(0)


class FakeResearch:
    """Count optional research calls and return a public source record."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def search_github(self, query: str) -> dict[str, object]:
        self.queries.append(query)
        return {
            "query": query,
            "retrieved_at": "2026-01-01T00:00:00+00:00",
            "sources": [
                {
                    "repository": "example/project",
                    "path": "module.py",
                    "url": _SOURCE_URL,
                    "snippet": "public code",
                }
            ],
        }


class CapturingClient(ScriptedClient):
    """Capture each conversation passed to a scripted client."""

    def __init__(self, replies: Sequence[str]) -> None:
        super().__init__(replies)
        self.conversations: list[list[Mapping[str, str]]] = []

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        self.conversations.append(list(messages))
        return super().chat(messages)


def _action(tool: str, **fields: str) -> str:
    """Encode an action as JSON."""
    return json.dumps({"tool": tool, **fields})


def test_query_validation_happens_before_http_request() -> None:
    """Reject empty, multiline, and overlong queries without requesting."""
    client = GitHubResearch(token="secret")
    with patch("adapters.research.github_research.urlopen") as urlopen:
        for query in ("", "  ", "two\nlines", "x" * 181, "contains secret"):
            with pytest.raises(ValueError):
                client.search_github(query)
    urlopen.assert_not_called()


def test_search_uses_get_and_caps_public_results_and_snippets() -> None:
    """Use only the code-search GET endpoint and emit at most three public sources."""
    response = FakeResponse(
        {
            "items": [
                _item(private=True, path="private.py"),
                _item(path="one.py", snippet="do-not-return-this" + "a" * 700),
                _item(path="two.py"),
                _item(path="three.py"),
                _item(path="four.py"),
            ]
        }
    )
    with patch("adapters.research.github_research.urlopen", return_value=response) as urlopen:
        evidence = GitHubResearch(token="do-not-return-this").search_github(
            "Python median even length"
        )

    request = urlopen.call_args.args[0]
    assert request.method == "GET"
    assert request.full_url.startswith("https://api.github.com/search/code?")
    assert request.get_header("Authorization")
    sources = evidence["sources"]
    assert isinstance(sources, list)
    assert len(sources) == 3
    assert all(source["repository"] == "example/project" for source in sources)
    assert all(len(source["snippet"]) <= 500 for source in sources)
    assert "[redacted]" in sources[0]["snippet"]
    assert sources[0]["url"] == _SOURCE_URL
    assert "private.py" not in repr(evidence)
    assert "full remote file" not in repr(evidence)
    assert "do-not-return-this" not in repr(evidence)


def test_request_and_result_budgets_are_enforced() -> None:
    """Cap a repair at three GET requests and three returned sources."""
    empty_response = FakeResponse({"items": []})
    client = GitHubResearch()
    with patch(
        "adapters.research.github_research.urlopen",
        return_value=empty_response,
    ) as urlopen:
        for index in range(3):
            client.search_github(f"query {index}")
        with pytest.raises(RuntimeError, match="request budget"):
            client.search_github("query 4")
    assert urlopen.call_count == 3

    response = FakeResponse({"items": [_item(path=f"{index}.py") for index in range(4)]})
    results_client = GitHubResearch()
    with patch("adapters.research.github_research.urlopen", return_value=response) as urlopen:
        evidence = results_client.search_github("public code")
        assert len(evidence["sources"]) == 3
        with pytest.raises(RuntimeError, match="result budget"):
            results_client.search_github("another query")
    assert urlopen.call_count == 1


def test_disabled_and_early_research_do_not_call_the_research_port() -> None:
    """Reject disabled research and enforce local evidence before searching."""
    disabled_tools = FakeTools()
    disabled_research = FakeResearch()
    disabled_result = RepairAgent(
        ScriptedClient([_action("search_github", query="Python median")]),
        disabled_tools,
    ).run("median", [])
    assert not disabled_result.passed
    assert disabled_research.queries == []
    assert disabled_tools.calls == []
    assert "Invalid model reply" in str(disabled_result.trace[-1]["reason"])

    early_research = FakeResearch()
    early_client = CapturingClient(
        [
            _action("search_github", query="Python median"),
            _action("read_file", path="test_statistics_tools.py"),
            _action("run_tests"),
        ]
    )
    early_tools = FakeTools([StructuredTestResult(False, "failure", 1)])
    early_result = RepairAgent(
        early_client,
        early_tools,
        research=early_research,
    ).run("median", ["statistics_tools.py", "test_statistics_tools.py"])

    assert early_research.queries == []
    assert any(event.get("tool") == "policy_error" for event in early_result.trace)
    assert "Policy error" in early_client.conversations[1][-1]["content"]


def test_allowed_research_source_is_visible_after_local_evidence() -> None:
    """Preserve the GitHub source URL in the trace and next model observation."""
    research = FakeResearch()
    client = CapturingClient(
        [
            _action("read_file", path="test_statistics_tools.py"),
            _action("run_tests"),
            _action("search_github", query="Python median even length"),
            _action("read_file", path="statistics_tools.py"),
        ]
    )
    tools = FakeTools([StructuredTestResult(False, "one failed", 1)])
    result = RepairAgent(client, tools, research=research).run(
        "median",
        ["statistics_tools.py", "test_statistics_tools.py"],
    )

    assert research.queries == ["Python median even length"]
    research_event = next(
        event
        for event in result.trace
        if event.get("event") == "tool_result" and event.get("tool") == "search_github"
    )
    assert _SOURCE_URL in json.dumps(research_event)
    assert _SOURCE_URL in client.conversations[3][-1]["content"]


def test_bootstrap_constructs_research_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture the token at composition and avoid constructing disabled research."""
    class CapturedService:
        def __init__(self, **kwargs: object) -> None:
            self.agent_factory = kwargs["agent_factory"]

    monkeypatch.setattr(bootstrap, "RepairService", CapturedService)
    with patch("adapters.research.github_research.GitHubResearch") as research_class:
        bootstrap.create_repair_service("scripted")
        research_class.assert_not_called()

        monkeypatch.setenv("GITHUB_TOKEN", "composition-token")
        service = bootstrap.create_repair_service(
            "scripted",
            research_enabled=True,
        )
        monkeypatch.setenv("GITHUB_TOKEN", "later-token")
        agent = service.agent_factory(ScriptedClient([]), FakeTools())

    research_class.assert_called_once_with(token="composition-token")
    assert agent.research is research_class.return_value
    assert "composition-token" not in system_message(research_enabled=True)["content"]


def test_research_flag_is_opt_in_for_repair_command() -> None:
    """Expose the extension only as an explicit repair command flag."""
    from main import _parser

    default_args = _parser().parse_args(["repair", "evals/arithmetic"])
    research_args = _parser().parse_args(
        ["repair", "evals/arithmetic", "--research"]
    )
    assert default_args.research is False
    assert research_args.research is True
