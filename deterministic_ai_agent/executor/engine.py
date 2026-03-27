import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import yaml  # type: ignore[import-untyped]

if TYPE_CHECKING:
    import torch
    from numpy.typing import NDArray

from deterministic_ai_agent.executor.registry import TOOL_REGISTRY, IntentID, ToolSpec
from deterministic_ai_agent.ner.extractor import NERExtractor

# ---------------------------------------------------------------------------
# Structured Logging
# ---------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_entry.update(record.extra)  # type: ignore
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logger(name: str, structured: bool = True, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    handler = logging.StreamHandler()
    if structured:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Protocols for flexibility (No hard dependency on torch in inference)
# ---------------------------------------------------------------------------


class EncoderProtocol(Protocol):
    def encode(self, text: str) -> Any: ...

    @property
    def embedding_dimension(self) -> int: ...


class ClassifierProtocol(Protocol):
    def predict_with_confidence(self, x: "torch.Tensor | NDArray[Any]") -> tuple[int, float]: ...

    def get_ood_score(self, x: "torch.Tensor | NDArray[Any]") -> float: ...


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    input: str
    action_id: int
    confidence: float
    ood_score: float
    params: dict[str, Any]
    result: dict[str, Any]


class AgentEngine:
    """
    Core engine for the deterministic AI agent.
    Optimized for zero-torch loading in production inference.
    """

    def __init__(
        self,
        encoder: EncoderProtocol,
        adapter: ClassifierProtocol,
        registry: dict[IntentID, ToolSpec] = TOOL_REGISTRY,
        config_path: str = "config/agent_settings.yaml",
    ):
        self.encoder = encoder
        self.adapter = adapter
        self.registry = registry
        self.ner = NERExtractor(encoder=self.encoder)

        # Load external settings
        settings = self._load_settings(config_path)
        self.confidence_threshold = settings["engine"]["thresholds"]["confidence"]
        self.ood_threshold = settings["engine"]["thresholds"]["ood"]

        # Setup structured logger
        self.logger = setup_logger(
            __name__,
            structured=settings["engine"]["logging"]["structured"],
            level=settings["engine"]["logging"]["level"],
        )
        self.session_history: list[StepRecord] = []

    def _load_settings(self, path_str: str) -> dict[str, Any]:
        path = Path(path_str)
        defaults = {
            "engine": {
                "thresholds": {"confidence": 0.70, "ood": 0.88},
                "logging": {"structured": True, "level": "INFO"},
            }
        }
        if not path.exists():
            return defaults
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def run_step(self, input_data: str) -> dict[str, Any]:
        start_time = time.perf_counter()
        vector = self.encoder.encode(input_data)
        action_id, confidence = self.adapter.predict_with_confidence(vector)
        ood_score = self.adapter.get_ood_score(vector)

        params: dict[str, Any] = {}
        result: dict[str, Any]

        if ood_score < self.ood_threshold:
            result = {
                "success": False,
                "reason": "out_of_distribution",
                "message": "Action refused. Input outside industrial operational domain.",
            }
        elif confidence < self.confidence_threshold:
            result = {
                "success": False,
                "reason": "low_confidence",
                "message": "Action refused. Human operator escalation required.",
            }
        else:
            params = self.ner.extract(input_data)
            result = self._execute_tool(action_id, params)

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Structured log entry
        self.logger.info(
            "Step execution completed",
            extra={
                "extra": {
                    "input": input_data,
                    "action_id": action_id,
                    "confidence": round(confidence, 4),
                    "ood_score": round(ood_score, 4),
                    "params": params,
                    "success": result.get("success", False),
                    "latency_ms": round(duration_ms, 2),
                }
            },
        )

        self.session_history.append(
            StepRecord(
                input=input_data,
                action_id=action_id,
                confidence=confidence,
                ood_score=ood_score,
                params=params,
                result=result,
            )
        )
        return result

    def train(self, data_path: str | Path, epochs: int = 50, learning_rate: float = 1e-3) -> None:
        """Training requires torch. Import it locally here."""
        import torch

        records = json.loads(Path(data_path).read_text(encoding="utf-8"))
        texts = [r["input"] for r in records]
        labels_list = [r["intent_id"] for r in records]

        vectors = torch.stack([self.encoder.encode(t) for t in texts])  # type: ignore
        labels = torch.tensor(labels_list, dtype=torch.long)

        optimizer = torch.optim.Adam(self.adapter.parameters(), lr=learning_rate)  # type: ignore

        for epoch in range(1, epochs + 1):
            loss = self.adapter.train_one_epoch(vectors, labels, optimizer)  # type: ignore
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch:>3}/{epochs}  loss={loss:.6f}")

        self.adapter.update_centroids(vectors, labels)  # type: ignore

    def _execute_tool(self, action_id: int, params: dict[str, Any]) -> dict[str, Any]:
        try:
            intent = IntentID(action_id)
        except ValueError:
            return {"success": False, "message": f"Unknown action ID: {action_id}"}
        spec = self.registry.get(intent)
        if spec is None:
            return {"success": False, "message": "No tool registered."}

        result: dict[str, Any]
        if spec.param_key is None:
            result = spec.fn(params)
        else:
            arg = params.get(spec.param_key)
            if arg is None:
                return {
                    "success": False,
                    "message": f"Missing required parameter: {spec.param_key}",
                }
            result = spec.fn(arg)

        # Ensure 'success' field is present
        if "success" not in result:
            result["success"] = True
        return result
