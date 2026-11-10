"""Best-effort Langfuse Cloud trajectory recorder."""

import logging
from collections.abc import Mapping
from typing import Any

from langfuse import Langfuse

_LOGGER = logging.getLogger(__name__)
_PUBLIC_FIELDS = {
    "case_name",
    "step_number",
    "requested_tool",
    "public_summary",
    "passed",
    "return_code",
    "changed_relative_file_names",
    "final_outcome",
}


def _public_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Keep only the deliberately small public telemetry fields."""
    result: dict[str, object] = {}
    for key in _PUBLIC_FIELDS:
        value = payload.get(key)
        if key == "changed_relative_file_names" and isinstance(value, list):
            result[key] = [
                name for name in value
                if isinstance(name, str) and not name.startswith(("/", "\\"))
            ]
        elif key == "public_summary" and isinstance(value, str):
            result[key] = value[:120]
        elif isinstance(value, (str, int, float, bool)) or value is None:
            if key in payload:
                result[key] = value
    return result


class LangfuseTrajectoryRecorder:
    """Record a repair root and closed child observations without blocking it."""

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        base_url: str,
        *,
        client: Any | None = None,
    ) -> None:
        self._client = client
        self._root: Any | None = None
        if self._client is None:
            try:
                self._client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    base_url=base_url,
                )
            except Exception:
                _LOGGER.warning("Could not initialize Langfuse recorder")

    def record(self, event: str, payload: Mapping[str, object]) -> None:
        """Record one safe event, closing and flushing after the repair."""
        if self._client is None:
            return
        safe_payload = _public_payload(payload)
        try:
            if event == "repair_started":
                self._root = self._client.start_observation(
                    name="repair",
                    as_type="agent",
                    input=safe_payload,
                )
                return

            if event == "repair_finished":
                if self._root is not None:
                    try:
                        self._root.update(output=safe_payload)
                    except Exception:
                        _LOGGER.warning("Could not update Langfuse repair observation")
                    try:
                        self._root.end()
                    except Exception:
                        _LOGGER.warning("Could not close Langfuse repair observation")
                try:
                    self._client.flush()
                except Exception:
                    _LOGGER.warning("Could not flush Langfuse observations")
                return

            if self._root is None:
                return
            child = self._client.start_observation(
                trace_context={
                    "trace_id": self._root.trace_id,
                    "parent_span_id": self._root.id,
                },
                name=event,
                as_type="span",
                input=safe_payload,
            )
            child.end()
        except Exception:
            _LOGGER.warning("Could not record Langfuse event %s", event)
