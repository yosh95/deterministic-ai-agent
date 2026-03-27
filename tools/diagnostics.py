import typing
from pathlib import Path

import yaml  # type: ignore[import-untyped]


def _load_tool_data(key: str) -> dict[str, typing.Any]:
    path = Path("config/tools_data.yaml")
    if not path.exists():
        return {}
    try:
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            return {}
        result = data.get(key, {})
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def run_diagnostics(device_id: str) -> dict[str, typing.Any]:
    """
    Simulated tool to run diagnostics on a specific machine.
    """
    diagnostics_data = _load_tool_data("diagnostics")
    result = diagnostics_data.get(
        device_id, {"health": "Unknown", "last_check": "N/A", "details": "No data available."}
    )
    return {"tool": "run_diagnostics", "device": device_id, "result": result}
