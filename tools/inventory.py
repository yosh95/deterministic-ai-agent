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


def check_inventory(item_name: str) -> dict[str, typing.Any]:
    """
    Simulated tool to check inventory of a specific item.
    """
    inventory_data = _load_tool_data("inventory")
    result = inventory_data.get(item_name, {"stock": "Unknown", "status": "Not Found"})
    return {"tool": "check_inventory", "item": item_name, "result": result}
