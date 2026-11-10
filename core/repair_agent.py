"""Control a bounded, evidence-based repair conversation."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn

from core.models import ModelAction, RepairResult, TestResult
from core.ports import ChatClient, RepairToolsPort, ResearchPort
from core.prompts import case_request, system_message, tool_result_message

_DECISION_BUDGET = 8
_TOOL_FIELDS = {
    "read_file": ({"tool", "path", "summary"}, {"tool", "path"}),
    "run_tests": ({"tool", "summary"}, {"tool"}),
    "edit_file": (
        {"tool", "path", "content", "summary"},
        {"tool", "path", "content"},
    ),
    "search_github": ({"tool", "query", "summary"}, {"tool", "query"}),
}


def _reject_constant(value: str) -> NoReturn:
    """Reject non-standard JSON values such as NaN and Infinity."""
    raise ValueError(f"invalid JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting duplicate keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_action(reply: str, research_enabled: bool = False) -> ModelAction:
    """Validate one JSON tool action and convert it to a model."""
    value = json.loads(
        reply,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("reply must be one JSON object")

    tool_fields = _TOOL_FIELDS
    if not research_enabled:
        tool_fields = {
            name: fields for name, fields in _TOOL_FIELDS.items()
            if name != "search_github"
        }

    tool = value.get("tool")
    if not isinstance(tool, str) or tool not in tool_fields:
        allowed = "read_file, run_tests, and edit_file"
        if research_enabled:
            allowed += ", and search_github"
        raise ValueError(f"tool must be {allowed}")

    allowed_fields, required_fields = tool_fields[tool]
    fields = set(value)
    if fields - allowed_fields:
        raise ValueError(f"unexpected fields: {', '.join(sorted(fields - allowed_fields))}")
    if required_fields - fields:
        raise ValueError(f"missing fields: {', '.join(sorted(required_fields - fields))}")

    for field in ("path", "content", "query"):
        if field in required_fields and not isinstance(value.get(field), str):
            raise ValueError(f"{field} must be a string")
    summary = value.get("summary")
    if "summary" in value and not isinstance(summary, str):
        raise ValueError("summary must be a string")

    arguments = {key: value[key] for key in required_fields if key != "tool"}
    return ModelAction(tool=tool, arguments=arguments, summary=summary)


class RepairAgent:
    """Ask for permitted actions and enforce evidence before repairs."""

    def __init__(
        self,
        client: ChatClient,
        tools: RepairToolsPort,
        research: ResearchPort | None = None,
    ) -> None:
        self.client = client
        self.tools = tools
        self.research = research

    def run(self, case_name: str, files: Sequence[str]) -> RepairResult:
        """Run at most eight model decisions and return a visible trace."""
        workspace = Path(getattr(self.tools, "workspace", Path(".")))
        trace: list[dict[str, object]] = [
            {
                "event": "repair_started",
                "case": case_name,
                "workspace": str(workspace),
            }
        ]
        messages = [
            system_message(research_enabled=self.research is not None),
            case_request(case_name, files),
        ]
        changed_files: list[str] = []
        test_result: TestResult | None = None
        has_read = False
        has_failing_baseline = False
        has_edit = False

        def finish(passed: bool, reason: str) -> RepairResult:
            trace.append(
                {
                    "event": "repair_finished",
                    "passed": passed,
                    "reason": reason,
                }
            )
            tool_changed_files = getattr(self.tools, "changed_files", changed_files)
            return RepairResult(
                passed=passed,
                trace=trace,
                changed_files=list(tool_changed_files),
                workspace=workspace,
                test_result=test_result,
            )

        for _ in range(_DECISION_BUDGET):
            try:
                reply = self.client.chat(messages)
            except Exception as error:
                message = f"Client error: {error}"
                trace.append({"event": "model_response", "error": message})
                return finish(False, message)

            try:
                action = _parse_action(
                    reply,
                    research_enabled=self.research is not None,
                )
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                message = f"Invalid model reply: {error}"
                trace.append({"event": "model_response", "valid": False, "error": message})
                return finish(False, message)

            messages.append({"role": "assistant", "content": reply})
            response_event: dict[str, object] = {
                "event": "model_response",
                "valid": True,
                "tool": action.tool,
            }
            if action.summary is not None:
                response_event["summary"] = action.summary
            trace.append(response_event)

            if action.tool == "edit_file" and not (has_read and has_failing_baseline):
                observation = (
                    "Policy error: edit_file requires a successful read_file and a "
                    "failing run_tests observation before the edit."
                )
                trace.append(
                    {
                        "event": "tool_result",
                        "tool": "policy_error",
                        "error": observation,
                    }
                )
                messages.append(tool_result_message("policy_error", observation))
                continue
            if action.tool == "search_github" and not (
                has_read and has_failing_baseline
            ):
                observation = (
                    "Policy error: search_github requires a successful read_file and a "
                    "failing run_tests observation before research."
                )
                trace.append(
                    {
                        "event": "tool_result",
                        "tool": "policy_error",
                        "error": observation,
                    }
                )
                messages.append(tool_result_message("policy_error", observation))
                continue

            try:
                if action.tool == "read_file":
                    output = self.tools.read_file(action.arguments["path"])
                    if not isinstance(output, str):
                        raise TypeError("read_file must return text")
                    has_read = True
                    trace.append(
                        {"event": "tool_result", "tool": action.tool, "output": output}
                    )
                    messages.append(tool_result_message(action.tool, output))
                elif action.tool == "run_tests":
                    result = self.tools.run_tests()
                    if not isinstance(result, TestResult):
                        raise TypeError("run_tests must return a TestResult")
                    test_result = result
                    observation = (
                        f"passed: {result.passed}\n"
                        f"returncode: {result.returncode}\n"
                        f"output:\n{result.output}"
                    )
                    trace.append(
                        {
                            "event": "tool_result",
                            "tool": action.tool,
                            "passed": result.passed,
                            "returncode": result.returncode,
                            "output": result.output,
                        }
                    )
                    messages.append(tool_result_message(action.tool, observation))
                    if result.passed and not has_edit:
                        return finish(
                            False,
                            "No reproducible bug: the baseline tests passed before any edit.",
                        )
                    if not result.passed and not has_edit and has_read:
                        has_failing_baseline = True
                    if result.passed and has_edit and has_failing_baseline:
                        return finish(True, "Tests passed after the repair.")
                elif action.tool == "search_github":
                    if self.research is None:
                        raise RuntimeError("GitHub research is not enabled")
                    evidence = self.research.search_github(action.arguments["query"])
                    observation = json.dumps(evidence, ensure_ascii=False)
                    trace.append(
                        {
                            "event": "tool_result",
                            "tool": action.tool,
                            "output": evidence,
                        }
                    )
                    messages.append(tool_result_message(action.tool, observation))
                else:
                    self.tools.edit_file(
                        action.arguments["path"],
                        action.arguments["content"],
                    )
                    has_edit = True
                    tool_changed_files = getattr(self.tools, "changed_files", None)
                    if tool_changed_files is not None:
                        changed_files = list(tool_changed_files)
                    else:
                        path = action.arguments["path"]
                        if path not in changed_files:
                            changed_files.append(path)
                    output = f"Edited {action.arguments['path']}."
                    trace.append(
                        {"event": "tool_result", "tool": action.tool, "output": output}
                    )
                    messages.append(tool_result_message(action.tool, output))
            except Exception as error:
                observation = f"Tool error: {error}"
                trace.append(
                    {
                        "event": "tool_result",
                        "tool": action.tool,
                        "error": observation,
                    }
                )
                messages.append(tool_result_message(action.tool, observation))

        return finish(False, f"Decision budget exhausted after {_DECISION_BUDGET} decisions.")
