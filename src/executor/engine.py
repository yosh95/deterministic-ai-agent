import json
import logging
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

if TYPE_CHECKING:
    import torch
    from numpy.typing import NDArray

# Use local imports for NER to check if it still loads torch
from src.ner.extractor import NERExtractor
from tools.diagnostics import run_diagnostics
from tools.inventory import check_inventory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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
# Tool registry
# ---------------------------------------------------------------------------


class IntentID(IntEnum):
    DIAGNOSTICS = 0
    INVENTORY = 1
    LOG_EVENT = 2


@dataclass
class ToolSpec:
    name: str
    fn: Callable[..., dict[str, Any]]
    param_key: Optional[str] = None


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

CONFIDENCE_THRESHOLD = 0.70
OOD_THRESHOLD = 0.88


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
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        ood_threshold: float = OOD_THRESHOLD,
    ):
        self.encoder = encoder
        self.adapter = adapter
        self.ner = NERExtractor(encoder=self.encoder)
        self.confidence_threshold = confidence_threshold
        self.ood_threshold = ood_threshold
        self.session_history: list[StepRecord] = []

    def run_step(self, input_data: str) -> dict[str, Any]:
        logger.info(f"Input: '{input_data}'")
        vector = self.encoder.encode(input_data)
        action_id, confidence = self.adapter.predict_with_confidence(vector)
        ood_score = self.adapter.get_ood_score(vector)

        logger.info(f"Action ID: {action_id} | Conf: {confidence:.4f} | OOD Score: {ood_score:.4f}")

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
                logger.info(f"Epoch {epoch:>3}/{epochs}  loss={loss:.6f}")

        self.adapter.update_centroids(vectors, labels)  # type: ignore

    def _execute_tool(self, action_id: int, params: dict[str, Any]) -> dict[str, Any]:
        try:
            intent = IntentID(action_id)
        except ValueError:
            return {"success": False, "message": f"Unknown action ID: {action_id}"}
        spec = TOOL_REGISTRY.get(intent)
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
