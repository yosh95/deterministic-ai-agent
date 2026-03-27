import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Default pattern definitions
# ---------------------------------------------------------------------------

_SENSOR_LABEL_PATTERN = re.compile(r"\bsensor\s+([A-Za-z0-9_\-]+)\b", re.IGNORECASE)
_LINE_PATTERN = re.compile(r"\b(?:line|ライン)\s*(\d+)\b", re.IGNORECASE)
_DATETIME_PATTERN = re.compile(r"\b(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}:\d{2}(?::\d{2})?)\b")
_NUMERIC_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:°C|bar|rpm|kPa|V|A|Hz)?\b")
_FAULT_KEYWORDS = re.compile(
    r"\b(overheat|over.?heat|overvoltage|over.?voltage|overcurrent|over.?current"
    r"|vibration|calibration\s+drift|pressure\s+drop|manual.?stop|shutdown)\b",
    re.IGNORECASE,
)


class NERExtractor:
    """
    Hybrid Named Entity Recogniser for OT / industrial log messages.
    Optimized to NOT load torch unless necessary.
    """

    def __init__(
        self,
        config_path: str = "config/devices.yaml",
        encoder: Optional[Any] = None,
        semantic_threshold: float = 0.88,
    ):
        self.devices: list[str] = []
        self.prefixes: list[str] = []
        self._load_config(config_path)
        self.device_pattern = self._build_device_pattern()

        self.encoder = encoder
        self.semantic_threshold = semantic_threshold
        self.device_embeddings: Optional[Any] = None

        if self.encoder and self.devices:
            device_texts = [f"industrial device: {d}" for d in self.devices]
            # Check the type of output from encoder without importing torch globally
            sample_vec = self.encoder.encode(device_texts[0])

            # Use string representation to detect torch.Tensor without importing torch
            if "torch.Tensor" in str(type(sample_vec)):
                import torch
                import torch.nn.functional as F

                self.device_embeddings = torch.stack([self.encoder.encode(t) for t in device_texts])
                self.device_embeddings = F.normalize(self.device_embeddings, p=2, dim=1)
            else:
                # Numpy path (ONNX)
                vectors = [self.encoder.encode(t) for t in device_texts]
                self.device_embeddings = np.vstack([v.reshape(1, -1) for v in vectors])
                norms = np.linalg.norm(self.device_embeddings, axis=1, keepdims=True)
                self.device_embeddings = self.device_embeddings / np.maximum(norms, 1e-9)

    def _load_config(self, config_path: str) -> None:
        path = Path(config_path)
        if path.exists():
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.devices = data.get("devices", [])
                self.prefixes = data.get("prefixes", [])
            except Exception:
                pass

    def _build_device_pattern(self) -> re.Pattern[str]:
        patterns = []
        if self.devices:
            patterns.append("|".join(re.escape(d) for d in self.devices))
        if self.prefixes:
            patterns.append("|".join(f"{re.escape(p)}[A-Za-z0-9_]+" for p in self.prefixes))
        if not patterns:
            return re.compile(r"\b(Unknown_Device)\b")
        return re.compile(rf"\b({'|'.join(patterns)})\b", re.IGNORECASE)

    def extract(self, text: str) -> dict[str, Any]:
        params: dict[str, Any] = {}
        device_match = self.device_pattern.search(text)
        if device_match:
            device_id = device_match.group(1)
            params["device_id"] = device_id
            params["item_name"] = device_id
            params["extraction_method"] = "regex"

        if "device_id" not in params and self.encoder and self.device_embeddings is not None:
            semantic_match = self._semantic_match(text)
            if semantic_match:
                params["device_id"] = semantic_match
                params["item_name"] = semantic_match
                params["extraction_method"] = "semantic"

        if "device_id" not in params:
            sensor_match = _SENSOR_LABEL_PATTERN.search(text)
            if sensor_match:
                params["sensor"] = sensor_match.group(1).upper()

        line_match = _LINE_PATTERN.search(text)
        if line_match:
            params["line_id"] = int(line_match.group(1))

        text_for_numeric = _DATETIME_PATTERN.sub(" ", text)
        text_for_numeric = _LINE_PATTERN.sub(" ", text_for_numeric)
        numeric_match = _NUMERIC_PATTERN.search(text_for_numeric)
        if numeric_match:
            try:
                params["value"] = float(numeric_match.group(1))
            except ValueError:
                pass

        fault_match = _FAULT_KEYWORDS.search(text)
        if fault_match:
            params["fault"] = re.sub(r"[\s\-]+", "_", fault_match.group(1).lower())

        return params

    def _semantic_match(self, text: str) -> Optional[str]:
        if self.device_embeddings is None or not self.encoder:
            return None

        input_vec = self.encoder.encode(f"industrial device: {text}")

        if "torch.Tensor" in str(type(input_vec)):
            import torch
            import torch.nn.functional as F

            input_vec = F.normalize(input_vec.unsqueeze(0), p=2, dim=1)
            similarities = torch.mm(input_vec, self.device_embeddings.t()).squeeze(0)
            max_score, max_idx = torch.max(similarities, dim=0)
            score = float(max_score.item())
            idx = int(max_idx.item())
        else:
            # Numpy logic (ONNX path)
            input_vec = input_vec.reshape(1, -1)
            norm = np.linalg.norm(input_vec, axis=1, keepdims=True)
            input_vec = input_vec / np.maximum(norm, 1e-9)
            similarities = np.dot(input_vec, self.device_embeddings.T).flatten()
            idx = int(np.argmax(similarities))
            score = float(similarities[idx])

        if score >= self.semantic_threshold:
            return self.devices[idx]

        return None
