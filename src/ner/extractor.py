import re
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
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
_DATETIME_PATTERN = re.compile(r"\b(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}:\d{2}(?::\d{2})?)\b")

# Numeric readings with optional units: "85.5", "85.5°C", "85.5 bar"
_NUMERIC_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:°C|bar|rpm|kPa|V|A|Hz)?\b")

# Common fault keywords
_FAULT_KEYWORDS = re.compile(
    r"\b(overheat|over.?heat|overvoltage|over.?voltage|overcurrent|over.?current"
    r"|vibration|calibration\s+drift|pressure\s+drop|manual.?stop|shutdown)\b",
    re.IGNORECASE,
)


class NERExtractor:
    """
    Hybrid Named Entity Recogniser for OT / industrial log messages.

    Design:
      1. Regex-based (High Precision): Uses strict patterns and config-defined lists.
      2. Semantic Fallback (High Recall): If Regex fails, uses sentence embeddings
         to find the closest matching device from the config list.

    Returns an empty dict {} for inputs that contain no recognisable entities,
    signaling the Executor to escalate rather than guess.
    """

    def __init__(
        self,
        config_path: str = "config/devices.yaml",
        encoder: Optional[Any] = None,
        semantic_threshold: float = 0.88,
    ):
        """
        Args:
            config_path: Path to the device/prefix configuration YAML.
            encoder: Optional EmbeddingEncoder instance for semantic fallback.
            semantic_threshold: Cosine similarity threshold for semantic matching.
        """
        self.devices: list[str] = []
        self.prefixes: list[str] = []
        self._load_config(config_path)

        # Dynamic device pattern based on config
        self.device_pattern = self._build_device_pattern()

        # Semantic matching setup
        self.encoder = encoder
        self.semantic_threshold = semantic_threshold
        self.device_embeddings: Optional[torch.Tensor] = None

        if self.encoder and self.devices:
            # Pre-compute embeddings for all known devices
            # We wrap device names in a simple context for better E5 embedding quality
            device_texts = [f"industrial device: {d}" for d in self.devices]
            self.device_embeddings = torch.stack([self.encoder.encode(t) for t in device_texts])
            # Normalize for cosine similarity (matrix multiplication)
            self.device_embeddings = F.normalize(self.device_embeddings, p=2, dim=1)

    def _load_config(self, config_path: str) -> None:
        """Load device list from YAML config."""
        path = Path(config_path)
        if path.exists():
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.devices = data.get("devices", [])
                self.prefixes = data.get("prefixes", [])
            except Exception:
                pass

    def _build_device_pattern(self) -> re.Pattern[str]:
        """Construct a compiled regex for known devices and prefixes."""
        patterns = []
        if self.devices:
            patterns.append("|".join(re.escape(d) for d in self.devices))
        if self.prefixes:
            patterns.append("|".join(f"{re.escape(p)}[A-Za-z0-9_]+" for p in self.prefixes))

        if not patterns:
            return re.compile(r"\b(Unknown_Device)\b")

        return re.compile(rf"\b({'|'.join(patterns)})\b", re.IGNORECASE)

    def extract(self, text: str) -> dict[str, Any]:
        """
        Parse *text* and return a flat dict of extracted entities.
        """
        params: dict[str, Any] = {}

        # --- Tier 1: Regex Matching (Strict) ---
        device_match = self.device_pattern.search(text)
        if device_match:
            device_id = device_match.group(1)
            params["device_id"] = device_id
            params["item_name"] = device_id
            params["extraction_method"] = "regex"

        # --- Tier 2: Semantic Fallback (Fuzzy) ---
        # If no device found via regex and encoder is available, try semantic similarity.
        if "device_id" not in params and self.encoder and self.device_embeddings is not None:
            semantic_match = self._semantic_match(text)
            if semantic_match:
                params["device_id"] = semantic_match
                params["item_name"] = semantic_match
                params["extraction_method"] = "semantic"

        # 3. Fallback: generic sensor label (only if still no device)
        if "device_id" not in params:
            sensor_match = _SENSOR_LABEL_PATTERN.search(text)
            if sensor_match:
                params["sensor"] = sensor_match.group(1).upper()

        # 4. Production line number
        line_match = _LINE_PATTERN.search(text)
        if line_match:
            params["line_id"] = int(line_match.group(1))

        # 5. Numeric reading (first occurrence)
        text_for_numeric = _DATETIME_PATTERN.sub(" ", text)
        text_for_numeric = _LINE_PATTERN.sub(" ", text_for_numeric)
        numeric_match = _NUMERIC_PATTERN.search(text_for_numeric)
        if numeric_match:
            try:
                params["value"] = float(numeric_match.group(1))
            except ValueError:
                pass

        # 6. Fault keyword
        fault_match = _FAULT_KEYWORDS.search(text)
        if fault_match:
            params["fault"] = re.sub(r"[\s\-]+", "_", fault_match.group(1).lower())

        return params

    def _semantic_match(self, text: str) -> Optional[str]:
        """Find the best device match based on cosine similarity."""
        if self.device_embeddings is None or not self.encoder:
            return None

        # Encode input text (using the same context prefix for E5 consistency)
        input_vec = self.encoder.encode(f"industrial device: {text}")
        input_vec = F.normalize(input_vec.unsqueeze(0), p=2, dim=1)

        # Compute cosine similarities
        similarities = torch.mm(input_vec, self.device_embeddings.t()).squeeze(0)
        max_score, max_idx = torch.max(similarities, dim=0)

        if max_score.item() >= self.semantic_threshold:
            return self.devices[int(max_idx.item())]

        return None
