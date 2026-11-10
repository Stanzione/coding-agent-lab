"""Gradio presentation layer for the existing repair service."""

import json
import os
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import gradio as gr

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

    def selected_service(provider: str) -> RepairService:
        """Create a service using the provider selected in the page."""
        if provider == "scripted" and service is not None:
            return service
        if provider == "qwen":
            return service_factory("ollama", model="qwen2.5-coder:14b")
        if provider == "ollama":
            return service_factory("ollama")
        if provider == "scripted":
            return service_factory("scripted")
        raise ValueError(f"Unknown provider: {provider}")

    def repair(case_name: str, provider: str) -> tuple[str, str, str, str | None]:
        """Run the selected case through the injected repair service."""
        if case_name not in _CASES:
            return f"Unknown evaluation case: {case_name}", "", "", None
        try:
            repair_service = selected_service(provider)
        except ValueError as error:
            return f"Repair failed: {error}", "", "", None
        try:
            result = repair_service.repair(f"evals/{case_name}")
        except (OSError, ValueError) as error:
            return f"Repair failed: {error}", "", "", None

        trace = "\n".join(
            json.dumps(event, ensure_ascii=False)
            for event in result.trace
        )
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

        test_result = result.test_result
        if test_result is None:
            test_output = "Tests were not run."
        else:
            test_output = (
                f"passed: {test_result.passed}\n"
                f"returncode: {test_result.returncode}\n"
                f"{test_result.output}"
            )
        archive = _workspace_zip(workspace, case_name)
        return trace, "\n\n".join(changed_text), test_output, archive

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
            inputs=[case_selector, provider_selector],
            outputs=[trace_output, changed_output, test_output, download_output],
        )
    return demo
