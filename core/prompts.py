"""Build the messages used by the repair agent."""

from collections.abc import Sequence


def system_message(research_enabled: bool = False) -> dict[str, str]:
    """Return the repair agent's system instruction."""
    research_instruction = ""
    if research_enabled:
        research_instruction = (
            "\nOptional GitHub research may provide supporting evidence, but it is "
            "not an authority and must not override local code or test results. "
            "Use it only after reading local code and observing a failing test. "
            'Example: {"tool":"search_github","query":"Python median even length",'
            '"summary":"Find a public reference"}'
        )
    return {
        "role": "system",
        "content": (
            "You are a teaching-sized code repair assistant. Inspect local code, "
            "reproduce a failing test before editing, make a focused edit, and "
            "verify the repair with tests. Reply with one JSON object only, with "
            "no Markdown or prose. Allowed actions are illustrated by these "
            "examples:\n"
            '{"tool":"read_file","path":"test_calculator.py","summary":"Read the test"}\n'
            '{"tool":"run_tests","summary":"Reproduce the failure"}\n'
            '{"tool":"edit_file","path":"calculator.py","content":"..."}\n'
            "The optional summary is a short label shown in the trace, not "
            "private reasoning."
            f"{research_instruction}"
        ),
    }


def case_request(case_name: str, files: Sequence[str]) -> dict[str, str]:
    """Return a user request naming the selected case and allowed files."""
    file_list = "\n".join(f"- {path}" for path in files) or "- (none supplied)"
    return {
        "role": "user",
        "content": (
            f"Repair the selected case: {case_name}\n"
            "The supplied relative files are:\n"
            f"{file_list}\n"
            "Choose one allowed action as a JSON object."
        ),
    }


def tool_result_message(tool: str, result: str) -> dict[str, str]:
    """Return a user message containing the outcome of one tool action."""
    return {
        "role": "user",
        "content": f"Tool result ({tool}):\n{result}",
    }
