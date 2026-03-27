import json
import logging
import typing
from typing import Any
from dataclasses import dataclass
from pathlib import Path

import torch

from src.adapter.classifier import IntentAdapter
from src.encoder.model import EmbeddingEncoder
from src.ner.extractor import NERExtractor
from tools.diagnostics import run_diagnostics
from tools.inventory import check_inventory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
# Each entry defines the callable and the parameter key it expects from NER.
# Adding a new tool requires only a new entry here — no changes to engine logic.


@dataclass
class ToolSpec:
    name: str
    fn: typing.Callable[..., dict[str, Any]]
    param_key: str  # key in the NER params dict to pass as the first argument


TOOL_REGISTRY: dict[int, ToolSpec] = {
    0: ToolSpec(name="run_diagnostics", fn=run_diagnostics, param_key="device_id"),
    1: ToolSpec(name="check_inventory", fn=check_inventory, param_key="item_name"),
    2: ToolSpec(
        name="log_event",
        fn=lambda payload: {"tool": "log_event", "data": payload, "result": "Logged"},
        param_key="",  # log_event receives the full params dict
    ),
}

# Minimum softmax confidence to execute a tool.
# Below this threshold the engine refuses to act and escalates to a human operator.
CONFIDENCE_THRESHOLD = 0.70

# Minimum cosine similarity to training centroids (OOD Detection - L3).
# Values close to 1.0 mean 'In-Distribution'.
# For E5-small, In-Distribution typically > 0.90, while OOD is < 0.86.
OOD_THRESHOLD = 0.88


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    input: str
    action_id: int
    confidence: float
    ood_score: float
    params: dict[str, typing.Any]
    result: dict[str, typing.Any]


class AgentEngine:
    """
    Core engine for the deterministic OT agent pipeline.

    Pipeline per step:
      1. Encode  — EmbeddingEncoder converts raw text to a vector.
      2. Classify — IntentAdapter maps the vector to an action ID + confidence.
      3. OOD Check — L3 Similarity Gate. If similarity < OOD_THRESHOLD, refuse.
      4. Guard   — L2 Confidence Gate. If confidence < CONFIDENCE_THRESHOLD, refuse.
      5. Extract — NERExtractor pulls structured parameters from the raw text.
      6. Execute — The registered tool function is called with those parameters.

    No LLM, no text generation, no stochastic sampling.
    """

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        adapter: IntentAdapter,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        ood_threshold: float = OOD_THRESHOLD,
    ):
        self.encoder = encoder
        self.adapter = adapter
        self.ner = NERExtractor()
        self.confidence_threshold = confidence_threshold
        self.ood_threshold = ood_threshold
        self.session_history: list[StepRecord] = []

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_step(self, input_data: str) -> dict[str, typing.Any]:
        """Execute a single perception-decision-action cycle."""
        logger.info(f"Input: '{input_data}'")

        # Step 1: Encode
        vector = self.encoder.encode(input_data)

        # Step 2: Classify with confidence (L2)
        action_id, confidence = self.adapter.predict_with_confidence(vector)
        # Step 3: OOD Score (L3)
        ood_score = self.adapter.get_ood_score(vector)

        logger.info(
            f"Action ID: {action_id} | "
            f"Conf: {confidence:.4f} | "
            f"OOD Score (Sim): {ood_score:.4f}"
        )

        # Step 4: OOD Guard (L3 safety gate)
        if ood_score < self.ood_threshold:
            logger.warning(
                f"OOD Score {ood_score:.4f} below threshold {self.ood_threshold}. "
                "Input recognized as Out-of-Distribution. Refusing execution."
            )
            result: dict[str, typing.Any] = {
                "success": False,
                "reason": "out_of_distribution",
                "action_id": action_id,
                "confidence": confidence,
                "ood_score": ood_score,
                "message": "Action refused. Input outside industrial operational domain.",
            }
            self._record(input_data, action_id, confidence, ood_score, {}, result)
            return result

        # Step 5: Confidence guard (L2 safety gate)
        if confidence < self.confidence_threshold:
            logger.warning(
                f"Confidence {confidence:.4f} below threshold {self.confidence_threshold}. "
                "Refusing to execute — escalate to human operator."
            )
            result = {
                "success": False,
                "reason": "low_confidence",
                "action_id": action_id,
                "confidence": confidence,
                "ood_score": ood_score,
                "message": "Action refused. Human operator escalation required.",
            }
            self._record(input_data, action_id, confidence, ood_score, {}, result)
            return result

        # Step 6: NER parameter extraction
        params = self.ner.extract(input_data)
        logger.info(f"Extracted params: {params}")

        # Step 7: Tool execution
        result = self._execute_tool(action_id, params)
        logger.info(f"Result: {result}")

        self._record(input_data, action_id, confidence, ood_score, params, result)
        return result

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        data_path: str | Path,
        epochs: int = 50,
        learning_rate: float = 1e-3,
    ) -> None:
        """
        Train the IntentAdapter from a labelled JSON dataset.

        Expected JSON format (list of objects):
          [{"input": "...", "intent_id": 0, "parameters": {...}}, ...]

        The encoder weights are frozen; only the adapter is updated.
        """
        records = json.loads(Path(data_path).read_text(encoding="utf-8"))
        logger.info(f"Training on {len(records)} samples for {epochs} epochs.")

        # Pre-compute all embeddings (encoder is frozen, so this is safe to do once)
        texts = [r["input"] for r in records]
        labels_list = [r["intent_id"] for r in records]

        vectors = torch.stack([self.encoder.encode(t) for t in texts])  # (N, dim)
        labels = torch.tensor(labels_list, dtype=torch.long)

        optimizer = torch.optim.Adam(self.adapter.parameters(), lr=learning_rate)

        for epoch in range(1, epochs + 1):
            loss = self.adapter.train_one_epoch(vectors, labels, optimizer)
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:>3}/{epochs}  loss={loss:.6f}")

        # Update centroids for L3 OOD detection
        self.adapter.update_centroids(vectors, labels)
        logger.info("Training complete. Centroids updated.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_tool(self, action_id: int, params: dict[str, typing.Any]) -> dict[str, typing.Any]:
        spec = TOOL_REGISTRY.get(action_id)
        if spec is None:
            return {"success": False, "message": f"Unknown action ID: {action_id}"}

        # log_event receives the whole params dict; other tools receive a single value
        if spec.param_key == "":
            return spec.fn(params)

        arg = params.get(spec.param_key)
        if arg is None:
            return {
                "success": False,
                "message": (
                    f"Tool '{spec.name}' requires parameter '{spec.param_key}' "
                    "but NER could not extract it from the input."
                ),
            }
        return spec.fn(arg)

    def _record(
        self,
        input_data: str,
        action_id: int,
        confidence: float,
        ood_score: float,
        params: dict[str, typing.Any],
        result: dict[str, typing.Any],
    ) -> None:
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


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_PATH = Path("data/sample_data.json")

    encoder = EmbeddingEncoder()
    adapter = IntentAdapter(input_dim=encoder.embedding_dimension, num_classes=3)
    engine = AgentEngine(encoder, adapter)

    # Train the adapter on labelled sample data before running inference
    engine.train(DATA_PATH, epochs=50)

    scenarios = [
        "Critical failure detected on production line 3. High motor temperature.",
        "Requesting stock status for Motor_B before maintenance.",
        "General system status report for Conveyor_A.",
        "What's the weather like today in Tokyo?",  # OOD
        "Tell me a joke about robots.",  # OOD
    ]

    print("\n--- Starting Simulation ---\n")
    for i, text in enumerate(scenarios, 1):
        print(f"Step {i}: {text}")
        engine.run_step(text)
        print("-" * 50)

    print(f"\nSimulation complete. Steps recorded: {len(engine.session_history)}")
