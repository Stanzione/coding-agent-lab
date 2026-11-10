"""Select an evaluation, prepare its workspace, and run the repair agent."""

from collections.abc import Callable, Mapping
from pathlib import Path, PureWindowsPath

from core.models import RepairResult
from core.ports import ChatClient, RepairToolsPort
from core.repair_agent import RepairAgent

WorkspaceFactory = Callable[[str], Path]
ToolsFactory = Callable[[Path], RepairToolsPort]
AgentFactory = Callable[[ChatClient, RepairToolsPort], RepairAgent]
TraceCallback = Callable[[Mapping[str, object]], None]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RepairService:
    """Run one repair against a disposable copy of a known evaluation."""

    def __init__(
        self,
        workspace_factory: WorkspaceFactory,
        tools_factory: ToolsFactory,
        client_factory: Callable[[str], ChatClient],
        agent_factory: AgentFactory = RepairAgent,
        trace_callback: TraceCallback | None = None,
    ) -> None:
        self._workspace_factory = workspace_factory
        self._tools_factory = tools_factory
        self._client_factory = client_factory
        self._agent_factory = agent_factory
        self._trace_callback = trace_callback

    def repair(self, selected_project: str | Path) -> RepairResult:
        """Validate a selected evaluation and return its repair result."""
        requested = Path(selected_project)
        if (
            requested.is_absolute()
            or PureWindowsPath(str(selected_project)).is_absolute()
            or len(requested.parts) != 2
            or requested.parts[0] != "evals"
            or any(part in {".", ".."} for part in requested.parts)
        ):
            raise ValueError("project must be a relative path of the form evals/<case>")

        evals_dir = (_PROJECT_ROOT / "evals").resolve(strict=True)
        case_name = requested.parts[1]
        known_cases = {
            entry.name for entry in evals_dir.iterdir() if entry.is_dir()
        }
        if case_name not in known_cases:
            raise ValueError(f"unknown evaluation project: {case_name}")

        source = (evals_dir / case_name).resolve(strict=True)
        if source.parent != evals_dir or not source.is_dir():
            raise ValueError("project must be a directory directly under evals")

        workspace = Path(self._workspace_factory(case_name)).resolve(strict=True)
        if not workspace.is_dir():
            raise NotADirectoryError(f"workspace is not a directory: {workspace}")
        if self._trace_callback is not None:
            self._trace_callback(
                {
                    "event": "repair_started",
                    "case": case_name,
                    "workspace": str(workspace),
                }
            )

        files = sorted(
            path.relative_to(workspace).as_posix()
            for path in workspace.rglob("*")
            if path.is_file()
        )
        tools = self._tools_factory(workspace)
        client = self._client_factory(case_name)
        agent = self._agent_factory(client, tools)
        result = agent.run(case_name, files)
        if self._trace_callback is not None:
            self._trace_callback(result.trace[-1])
        return result
