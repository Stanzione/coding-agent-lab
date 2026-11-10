"""Protocols for the repair agent's external capabilities."""

from typing import Mapping, Protocol, Sequence

from core.models import TestResult


class ChatClient(Protocol):
    """Send conversation messages to a chat model."""

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str: ...


class RepairToolsPort(Protocol):
    """Read project files, edit them, and run tests."""

    def read_file(self, path: str) -> str: ...

    def edit_file(self, path: str, content: str) -> None: ...

    def run_tests(self) -> TestResult: ...


class ResearchPort(Protocol):
    """Provide optional, bounded GitHub search evidence."""

    def search_github(self, query: str) -> dict[str, object]: ...


class TraceRecorder(Protocol):
    """Record an event and its structured payload."""

    def record(self, event: str, payload: Mapping[str, object]) -> None: ...
