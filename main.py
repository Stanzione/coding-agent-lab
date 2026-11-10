"""Command-line entry point for scripted evaluation repairs."""

import argparse
import json
import sys

from bootstrap import create_repair_service


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
    ui_parser = subparsers.add_parser("ui", help="launch the Gradio repair page")
    ui_parser.add_argument(
        "--provider",
        choices=("scripted",),
        default="scripted",
        help="chat client to use",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested repair and print its trace and test result."""
    args = _parser().parse_args(argv)
    if args.command == "ui":
        from adapters.gradio_ui import create_demo

        create_demo().launch(
            allowed_paths=["/home/joaostanzione/.cache/coding-agent-lab"]
        )
        return 0

    try:
        service = create_repair_service(
            args.provider,
            trace_callback=lambda event: print(
                f"TRACE {json.dumps(event, ensure_ascii=False)}",
                flush=True,
            ),
            model=args.model,
            temperature=args.temperature,
        )
        result = service.repair(args.project)
    except (OSError, ValueError) as error:
        print(f"Repair failed: {error}", file=sys.stderr)
        return 2

    print("Changed files:")
    for path in result.changed_files:
        print(f"- {path}")

    if result.test_result is None:
        print("Final sandboxed test result: not run")
    else:
        test = result.test_result
        print(
            "Final sandboxed test result: "
            f"passed={test.passed}, returncode={test.returncode}"
        )
        if test.output:
            print(test.output, end="" if test.output.endswith("\n") else "\n")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
