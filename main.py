"""Command-line entry point for scripted evaluation repairs."""

import argparse
import json
import os
import sys
import textwrap
from collections.abc import Mapping

from bootstrap import create_repair_service


def _print_trace_event(event: Mapping[str, object]) -> None:
    """Print one structured repair event in a readable terminal format."""
    event_type = event.get("event")
    if event_type == "repair_started":
        print(f"[START] {event.get('case', 'repair')}", flush=True)
    elif event_type == "model_response":
        if event.get("valid") is False:
            print(f"[ACTION] Invalid model response: {event.get('error', '')}", flush=True)
        elif "error" in event:
            print(f"[ACTION] Model error: {event['error']}", flush=True)
        else:
            tool = str(event.get("tool", "unknown")).upper()
            summary = event.get("summary")
            suffix = f" - {summary}" if summary else ""
            print(f"[ACTION] {tool}{suffix}", flush=True)
    elif event_type == "tool_result":
        tool = str(event.get("tool", "tool"))
        print(f"[RESULT] {tool}", flush=True)
        if "passed" in event:
            print(
                f"  status: {'passed' if event['passed'] else 'failed'}",
                flush=True,
            )
        if "returncode" in event:
            print(f"  return code: {event['returncode']}", flush=True)
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
            print(textwrap.indent(output.rstrip("\n"), "  "), flush=True)
    elif event_type == "repair_finished":
        status = "PASS" if event.get("passed") else "FAILED"
        print(f"[FINISHED] {status} - {event.get('reason', '')}", flush=True)


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Repair a supplied evaluation project.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    repair_parser = subparsers.add_parser("repair", help="repair one evals/<case> project")
    repair_parser.add_argument("project", help="selected project, such as evals/arithmetic")
    repair_parser.add_argument(
        "--provider",
        choices=("scripted", "ollama"),
        default="scripted",
        help="chat client to use",
    )
    repair_parser.add_argument(
        "--model",
        help="Ollama model name (default: OLLAMA_MODEL or qwen2.5-coder:14b)",
    )
    repair_parser.add_argument(
        "--temperature",
        type=float,
        help="Ollama sampling temperature",
    )
    repair_parser.add_argument(
        "--research",
        action="store_true",
        help="enable optional bounded public GitHub code-search evidence",
    )
    repair_parser.add_argument(
        "--observe",
        action="store_true",
        help="record a sanitized repair trajectory in Langfuse Cloud",
    )
    ui_parser = subparsers.add_parser("ui", help="launch the Gradio repair page")
    ui_parser.add_argument(
        "--provider",
        choices=("scripted",),
        default="scripted",
        help="chat client to use",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested repair and print live progress and its final test result."""
    args = _parser().parse_args(argv)
    if args.command == "ui":
        from adapters.gradio_ui import create_demo

        create_demo().launch(
            allowed_paths=["/home/joaostanzione/.cache/coding-agent-lab"]
        )
        return 0

    if args.observe:
        missing = [
            name
            for name in (
                "LANGFUSE_PUBLIC_KEY",
                "LANGFUSE_SECRET_KEY",
                "LANGFUSE_BASE_URL",
            )
            if not os.environ.get(name)
        ]
        if missing:
            print(
                "Cannot enable Langfuse observability; missing environment "
                f"variables: {', '.join(missing)}",
                file=sys.stderr,
            )
            return 2

    try:
        service = create_repair_service(
            args.provider,
            trace_callback=_print_trace_event,
            model=args.model,
            temperature=args.temperature,
            research_enabled=args.research,
            observe_enabled=args.observe,
        )
        result = service.repair(args.project)
    except (OSError, ValueError) as error:
        print(f"Repair failed: {error}", file=sys.stderr)
        return 2

    print("Changed files:")
    for path in result.changed_files:
        print(f"- {path}")

    if result.test_result is None:
        print("Final sandboxed test result: NOT RUN")
    else:
        test = result.test_result
        print(
            "Final sandboxed test result: "
            f"{'PASSED' if test.passed else 'FAILED'} "
            f"(return code {test.returncode})"
        )
        if test.output:
            print(textwrap.indent(test.output.rstrip("\n"), "  "))

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
