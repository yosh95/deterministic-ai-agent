from unittest.mock import MagicMock, patch

import torch

from src.adapter.classifier import IntentAdapter
from src.executor.engine import CONFIDENCE_THRESHOLD, AgentEngine, IntentID

INPUT_DIM = 384
NUM_CLASSES = 3


def _make_engine(
    confidence: float = 0.95, action_id: int = IntentID.DIAGNOSTICS, ood_score: float = 1.0
) -> AgentEngine:
    """Return an AgentEngine with a mocked encoder and a stubbed adapter."""
    encoder = MagicMock()
    encoder.encode.return_value = torch.randn(INPUT_DIM)

    adapter = IntentAdapter(INPUT_DIM, NUM_CLASSES)
    engine = AgentEngine(encoder, adapter, confidence_threshold=CONFIDENCE_THRESHOLD)

    # Patch predict_with_confidence via patch.object to satisfy mypy (no method-assign).
    # The patch is applied permanently on this instance for the test's lifetime.
    patch.object(
        adapter,
        "predict_with_confidence",
        return_value=(int(action_id), confidence),
    ).start()

    # Patch get_ood_score to return a high value (In-Domain) by default
    patch.object(
        adapter,
        "get_ood_score",
        return_value=ood_score,
    ).start()

    return engine


# ---------------------------------------------------------------------------
# Normal execution
# ---------------------------------------------------------------------------


def test_run_step_returns_dict():
    engine = _make_engine(action_id=IntentID.DIAGNOSTICS, confidence=0.95)
    result = engine.run_step("Conveyor_A has stopped. Run diagnostics.")
    assert isinstance(result, dict)


def test_run_step_records_history():
    engine = _make_engine(action_id=IntentID.INVENTORY, confidence=0.95)
    engine.run_step("Check inventory for Motor_B.")
    assert len(engine.session_history) == 1


def test_run_step_multiple_calls_accumulate_history():
    engine = _make_engine(action_id=IntentID.LOG_EVENT, confidence=0.95)
    engine.run_step("Log event A.")
    engine.run_step("Log event B.")
    assert len(engine.session_history) == 2


def test_run_step_calls_diagnostics_tool():
    engine = _make_engine(action_id=IntentID.DIAGNOSTICS, confidence=0.95)
    result = engine.run_step("Conveyor_A vibration detected. Diagnostic check needed.")
    assert result.get("tool") == "run_diagnostics"


def test_run_step_calls_inventory_tool():
    engine = _make_engine(action_id=IntentID.INVENTORY, confidence=0.95)
    result = engine.run_step("Check inventory for Motor_B spare parts.")
    assert result.get("tool") == "check_inventory"


# ---------------------------------------------------------------------------
# L2: Confidence guard
# ---------------------------------------------------------------------------


def test_low_confidence_returns_refusal():
    """Below CONFIDENCE_THRESHOLD the engine must refuse to execute."""
    engine = _make_engine(confidence=0.30, action_id=IntentID.DIAGNOSTICS)
    result = engine.run_step("Some ambiguous input.")
    assert result["success"] is False
    assert result["reason"] == "low_confidence"


def test_low_confidence_still_records_history():
    """Refused steps must still be recorded for audit purposes."""
    engine = _make_engine(confidence=0.30, action_id=IntentID.DIAGNOSTICS)
    engine.run_step("Some ambiguous input.")
    assert len(engine.session_history) == 1


def test_high_confidence_does_not_refuse():
    engine = _make_engine(confidence=0.99, action_id=IntentID.DIAGNOSTICS)
    result = engine.run_step("Conveyor_A stopped. Run diagnostics.")
    assert result.get("success") is not False or result.get("tool") is not None


# ---------------------------------------------------------------------------
# L3: OOD guard
# ---------------------------------------------------------------------------


def test_ood_input_returns_refusal():
    """Inputs with low similarity must be rejected as Out-of-Distribution."""
    engine = _make_engine(ood_score=0.1)  # Very low similarity
    result = engine.run_step("What is the meaning of life?")
    assert result["success"] is False
    assert result["reason"] == "out_of_distribution"


# ---------------------------------------------------------------------------
# TOOL_REGISTRY coverage
# ---------------------------------------------------------------------------


def test_unknown_action_id_returns_failure():
    engine = _make_engine(confidence=0.99, action_id=99)
    result = engine.run_step("Conveyor_A diagnostics please.")
    assert result["success"] is False
    assert "Unknown action ID" in result["message"] or "No tool registered" in result["message"]
