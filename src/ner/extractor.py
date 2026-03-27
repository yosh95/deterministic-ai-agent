import re
from typing import Any
from pathlib import Path

import yaml  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Default pattern definitions (will be augmented by config if available)
# ---------------------------------------------------------------------------

# Generic sensor labels: Sensor XYZ, sensor xyz (case-insensitive)
_SENSOR_LABEL_PATTERN = re.compile(r"\bsensor\s+([A-Za-z0-9_\-]+)\b", re.IGNORECASE)

# Production / process line numbers: "line 3", "line3", "ライン3"
_LINE_PATTERN = re.compile(r"\b(?:line|ライン)\s*(\d+)\b", re.IGNORECASE)

# Date and Time patterns to be stripped before numeric extraction
# e.g., 2026-03-27, 2026/03/27, 14:32:00, 14:32
_DATETIME_PATTERN = re.compile(
    r"\b(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}:\d{2}(?::\d{2})?)\b"
)

# Numeric readings with optional units: "85.5", "85.5°C", "85.5 bar"
_NUMERIC_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:°C|bar|rpm|kPa|V|A|Hz)?\b"
)

# Common fault keywords
_FAULT_KEYWORDS = re.compile(
    r"\b(overheat|over.?heat|overvoltage|over.?voltage|overcurrent|over.?current"
    r"|vibration|calibration\s+drift|pressure\s+drop|manual.?stop|shutdown)\b",
    re.IGNORECASE,
)


class NERExtractor:
    """
    Rule-based Named Entity Recogniser for OT / industrial log messages.

    Extracts structured parameters from raw text so that the Executor can call
    tool functions with concrete arguments — without relying on an LLM.

    Design constraints:
      - Pure Python + stdlib (re): no external model, no network access.
      - Returns an empty dict {} for inputs that contain no recognisable entities,
        which signals the Executor to escalate rather than guess.
    """

    def __init__(self, config_path: str = "config/devices.yaml"):
        self.devices: list[str] = []
        self.prefixes: list[str] = []
        self._load_config(config_path)

        # Dynamic device pattern based on config
        # e.g., (Conveyor_A|Motor_B|...)
        self.device_pattern = self._build_device_pattern()

    def _load_config(self, config_path: str) -> None:
        """Load device list from YAML config."""
        path = Path(config_path)
        if path.exists():
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.devices = data.get("devices", [])
                self.prefixes = data.get("prefixes", [])
            except Exception:
                # Fallback to empty lists if YAML is invalid
                pass

    def _build_device_pattern(self) -> re.Pattern[str]:
        """Construct a compiled regex for known devices and prefixes."""
        patterns = []
        if self.devices:
            # Escape each device name for regex safety
            patterns.append("|".join(re.escape(d) for d in self.devices))
        if self.prefixes:
            # Match prefixes followed by alphanumeric characters
            patterns.append("|".join(f"{re.escape(p)}[A-Za-z0-9_]+" for p in self.prefixes))

        if not patterns:
            # Default fallback pattern if config is empty
            return re.compile(r"\b(Unknown_Device)\b")

        return re.compile(rf"\b({'|'.join(patterns)})\b", re.IGNORECASE)

    def extract(self, text: str) -> dict[str, Any]:
        """
        Parse *text* and return a flat dict of extracted entities.

        Keys (all optional — only present when found):
          device_id  : str   — primary device identifier (e.g. "Conveyor_A")
          item_name  : str   — alias for device_id used by inventory tool
          sensor     : str   — sensor label when no structured device ID is found
          line_id    : int   — production line number
          value      : float — first numeric reading found
          fault      : str   — normalised fault keyword
        """
        params: dict[str, Any] = {}

        # 1. Structured device ID (highest priority)
        device_match = self.device_pattern.search(text)
        if device_match:
            device_id = device_match.group(1)
            params["device_id"] = device_id
            params["item_name"] = device_id  # inventory tool uses "item_name"

        # 2. Fallback: generic sensor label
        if "device_id" not in params:
            sensor_match = _SENSOR_LABEL_PATTERN.search(text)
            if sensor_match:
                params["sensor"] = sensor_match.group(1).upper()

        # 3. Production line number
        line_match = _LINE_PATTERN.search(text)
        if line_match:
            params["line_id"] = int(line_match.group(1))

        # 4. Numeric reading (first occurrence)
        # Avoid picking up line numbers or dates as sensor values.
        text_for_numeric = _DATETIME_PATTERN.sub(" ", text)
        text_for_numeric = _LINE_PATTERN.sub(" ", text_for_numeric)

        numeric_match = _NUMERIC_PATTERN.search(text_for_numeric)
        if numeric_match:
            try:
                params["value"] = float(numeric_match.group(1))
            except ValueError:
                pass

        # 5. Fault keyword
        fault_match = _FAULT_KEYWORDS.search(text)
        if fault_match:
            # Normalise whitespace/hyphens for downstream consumers
            params["fault"] = re.sub(r"[\s\-]+", "_", fault_match.group(1).lower())

        return params
