"""Gradio presentation layer for the existing repair service."""

import json
import os
import queue
import tempfile
import threading
import zipfile
import textwrap
from collections.abc import Callable, Mapping
from pathlib import Path

import gradio as gr

from core.models import RepairResult
from core.repair_service import RepairService

_CASES = (
    "arithmetic",
    "balanced_brackets",
    "median",
    "multi_function",
    "unique_words",
)
_PROVIDERS = (
    ("scripted", "scripted"),
    ("ollama (modelo configurado)", "ollama"),
    ("qwen2.5-coder:14b", "qwen"),
)
_LANGFUSE_ENVIRONMENT = (
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
)


def _selected_service(
    provider: str,
    research_enabled: bool,
    service: RepairService | None,
    service_factory: Callable[..., RepairService],
    trace_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> RepairService:
    """Create the selected provider service and enable Qwen observability."""
    if provider == "scripted" and service is not None:
        return service
    if provider == "qwen":
        missing = [name for name in _LANGFUSE_ENVIRONMENT if not os.environ.get(name)]
        if missing:
            raise ValueError(
                "Qwen Langfuse observability requires environment variables: "
                f"{', '.join(missing)}"
            )
        return service_factory(
            "ollama",
            model="qwen2.5-coder:14b",
            research_enabled=research_enabled,
            observe_enabled=True,
            trace_callback=trace_callback,
        )
    if provider == "ollama":
        return service_factory("ollama", trace_callback=trace_callback)
    if provider == "scripted":
        return service_factory("scripted", trace_callback=trace_callback)
    raise ValueError(f"Unknown provider: {provider}")


def _format_trace_event(event: Mapping[str, object]) -> str:
    """Format one repair event as a concise labeled UI step."""
    event_type = event.get("event")
    if event_type == "repair_started":
        return f"[START] {event.get('case', 'repair')}"
    if event_type == "model_response":
        if event.get("valid") is False:
            return f"[ACTION] Invalid model response: {event.get('error', '')}"
        if "error" in event:
            return f"[ACTION] Model error: {event['error']}"
        tool = str(event.get("tool", "unknown")).upper()
        summary = event.get("summary")
        suffix = f" - {summary}" if summary else ""
        return f"[ACTION] {tool}{suffix}"
    if event_type == "tool_result":
        lines = [f"[RESULT] {event.get('tool', 'tool')}"]
        if "passed" in event:
            lines.append(f"  status: {'passed' if event['passed'] else 'failed'}")
        if "returncode" in event:
            lines.append(f"  return code: {event['returncode']}")
        if "error" in event:
            output = str(event["error"])
        elif "output" in event:
            value = event["output"]
            output = (
                json.dumps(value, ensure_ascii=False, indent=2)
                if isinstance(value, (dict, list))
                else str(value)
            )
        else:
            output = ""
        if output:
            lines.append(textwrap.indent(output.rstrip("\n"), "  "))
        return "\n".join(lines)
    if event_type == "repair_finished":
        status = "PASS" if event.get("passed") else "FAILED"
        return f"[FINISHED] {status} - {event.get('reason', '')}"
    return ""


def get_download_directory() -> Path:
    """Create and return the directory containing downloadable workspaces."""
    home = Path.home().resolve()
    download_dir = home / ".cache" / "coding-agent-lab" / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    resolved_dir = download_dir.resolve()
    resolved_dir.relative_to(home)
    return resolved_dir


def _workspace_zip(workspace: Path, case_name: str) -> str:
    """Create a downloadable ZIP of the disposable workspace."""
    download_dir = get_download_directory()
    descriptor, archive_name = tempfile.mkstemp(
        prefix=f"{case_name}-",
        suffix=".zip",
        dir=download_dir,
    )
    os.close(descriptor)
    archive_path = Path(archive_name)
    root = workspace.resolve(strict=True)

    try:
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_symlink() or not path.is_file():
                    continue
                archive.write(path, path.relative_to(root).as_posix())
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return str(archive_path)


def create_demo(
    service: RepairService | None = None,
    service_factory: Callable[..., RepairService] | None = None,
) -> gr.Blocks:
    """Build the case, provider, and repair-result page."""
    if service_factory is None:
        from bootstrap import create_repair_service

        service_factory = create_repair_service

    def repair(
        case_name: str,
        provider: str,
        research_enabled: bool,
    ):
        """Stream labeled trace events and return the final repair artifacts."""
        if case_name not in _CASES:
            yield f"Unknown evaluation case: {case_name}", "", "", None
            return
        events: queue.Queue[
            tuple[str, Mapping[str, object] | RepairResult | Exception | None]
        ] = queue.Queue()

        def publish(event: Mapping[str, object]) -> None:
            """Forward an agent trace event to the Gradio generator."""
            events.put(("event", dict(event)))

        try:
            repair_service = _selected_service(
                provider,
                provider == "qwen" and research_enabled,
                service,
                service_factory,
                publish,
            )
        except ValueError as error:
            yield f"Repair failed: {error}", "", "", None
            return

        def run_repair() -> None:
            """Run repair off-thread so trace events can stream to the page."""
            try:
                events.put(("result", repair_service.repair(f"evals/{case_name}")))
            except Exception as error:
                events.put(("error", error))
            finally:
                events.put(("finished", None))

        worker = threading.Thread(target=run_repair, daemon=True)
        worker.start()
        trace_lines: list[str] = []
        result: RepairResult | None = None
        while True:
            kind, payload = events.get()
            if kind == "event" and isinstance(payload, Mapping):
                formatted = _format_trace_event(payload)
                if formatted:
                    trace_lines.append(formatted)
                    yield "\n\n".join(trace_lines), "", "", None
            elif kind == "result" and isinstance(payload, RepairResult):
                result = payload
            elif kind == "error" and isinstance(payload, Exception):
                yield f"Repair failed: {payload}", "", "", None
            elif kind == "finished":
                break

        if result is None:
            return

        if not trace_lines:
            trace_lines = [
                line
                for event in result.trace
                if (line := _format_trace_event(event))
            ]
        trace = "\n\n".join(trace_lines)
        changed_text: list[str] = []
        workspace = Path(result.workspace)
        for relative_path in result.changed_files:
            changed_path = (workspace / relative_path).resolve(strict=True)
            try:
                changed_path.relative_to(workspace.resolve(strict=True))
            except ValueError as error:
                raise ValueError(f"changed file escapes workspace: {relative_path}") from error
            changed_text.append(
                f"--- {relative_path} ---\n"
                f"{changed_path.read_text(encoding='utf-8')}"
            )
        changed_output = "Changed files:"
        if result.changed_files:
            changed_output += "\n" + "\n".join(
                f"- {relative_path}" for relative_path in result.changed_files
            )
        if changed_text:
            changed_output += "\n\n" + "\n\n".join(changed_text)

        test_result = result.test_result
        if test_result is None:
            test_output = "Final sandboxed test result: NOT RUN"
        else:
            test_output = (
                "Final sandboxed test result: "
                f"{'PASSED' if test_result.passed else 'FAILED'} "
                f"(return code {test_result.returncode})"
            )
            if test_result.output:
                test_output += "\n" + textwrap.indent(
                    test_result.output.rstrip("\n"),
                    "  ",
                )
        archive = _workspace_zip(workspace, case_name)
        yield trace, changed_output, test_output, archive

    with gr.Blocks(title="Coding Agent Lab") as demo:
        gr.Markdown("# Coding Agent Lab\nRun a scripted repair on a supplied evaluation.")
        case_selector = gr.Dropdown(
            choices=list(_CASES),
            value="arithmetic",
            label="Evaluation case",
        )
        provider_selector = gr.Dropdown(
            choices=list(_PROVIDERS),
            value="scripted",
            label="AI provider/model",
        )
        research_toggle = gr.Checkbox(
            label="Enable optional GitHub code search (Qwen only)",
            value=False,
            interactive=False,
        )
        provider_selector.change(
            fn=lambda provider: gr.update(
                interactive=provider == "qwen",
                value=False,
            ),
            inputs=provider_selector,
            outputs=research_toggle,
        )
        start_button = gr.Button("Start repair")
        trace_output = gr.Textbox(label="Visible trace", lines=14, interactive=False)
        changed_output = gr.Textbox(
            label="Changed relative files and contents",
            lines=14,
            interactive=False,
        )
        test_output = gr.Textbox(label="Final sandboxed test output", lines=8, interactive=False)
        download_output = gr.File(label="Download repaired workspace ZIP", interactive=False)
        start_button.click(
            fn=repair,
            inputs=[case_selector, provider_selector, research_toggle],
            outputs=[trace_output, changed_output, test_output, download_output],
        )
    return demo
