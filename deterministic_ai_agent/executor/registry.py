from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Optional

# Import tool implementations (Stubs)
from tools.diagnostics import run_diagnostics
from tools.inventory import check_inventory


class IntentID(IntEnum):
    DIAGNOSTICS = 0
    INVENTORY = 1
    LOG_EVENT = 2


@dataclass
class ToolSpec:
    name: str
    fn: Callable[..., dict[str, Any]]
    param_key: Optional[str] = None


# Default tool registry
TOOL_REGISTRY: dict[IntentID, ToolSpec] = {
    IntentID.DIAGNOSTICS: ToolSpec(
        name="run_diagnostics", fn=run_diagnostics, param_key="device_id"
    ),
    IntentID.INVENTORY: ToolSpec(name="check_inventory", fn=check_inventory, param_key="item_name"),
    IntentID.LOG_EVENT: ToolSpec(
        name="log_event",
        fn=lambda payload: {"tool": "log_event", "data": payload, "result": "Logged"},
    ),
}
