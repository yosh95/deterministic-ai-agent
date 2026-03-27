import tempfile

import torch

from deterministic_ai_agent.adapter.classifier import IntentAdapter

INPUT_DIM = 384
NUM_CLASSES = 5


def _make_adapter() -> IntentAdapter:
    return IntentAdapter(INPUT_DIM, NUM_CLASSES)


def _dummy(batch: int = 1) -> torch.Tensor:
    return torch.randn(batch, INPUT_DIM)


# ---------------------------------------------------------------------------
# Forward / shape
# ---------------------------------------------------------------------------


def test_adapter_output_shape():
    adapter = _make_adapter()
    logits = adapter(_dummy())
    assert logits.shape == (1, NUM_CLASSES)


# ---------------------------------------------------------------------------
# L1: Deterministic prediction
# ---------------------------------------------------------------------------


def test_adapter_prediction_type_and_range():
    adapter = _make_adapter()
    pred = adapter.predict(_dummy())
    assert isinstance(pred, int)
    assert 0 <= pred < NUM_CLASSES


def test_adapter_prediction_is_deterministic():
    """Same input must always yield the same output (L1 guarantee)."""
    adapter = _make_adapter()
    x = _dummy()
    results = {adapter.predict(x) for _ in range(10)}
    assert len(results) == 1


# ---------------------------------------------------------------------------
# L2: Confidence-aware prediction
# ---------------------------------------------------------------------------


def test_predict_with_confidence_types():
    adapter = _make_adapter()
    intent_id, confidence = adapter.predict_with_confidence(_dummy())
    assert isinstance(intent_id, int)
    assert isinstance(confidence, float)


def test_predict_with_confidence_range():
    adapter = _make_adapter()
    _, confidence = adapter.predict_with_confidence(_dummy())
    assert 0.0 <= confidence <= 1.0


def test_predict_with_confidence_agrees_with_predict():
    """predict() and predict_with_confidence() must return the same intent ID."""
    adapter = _make_adapter()
    x = _dummy()
    assert adapter.predict(x) == adapter.predict_with_confidence(x)[0]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_train_one_epoch_reduces_loss():
    """Loss should decrease over multiple epochs on a tiny dataset."""
    adapter = _make_adapter()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-2)

    vectors = torch.randn(8, INPUT_DIM)
    labels = torch.randint(0, NUM_CLASSES, (8,))

    first_loss = adapter.train_one_epoch(vectors, labels, optimizer)
    for _ in range(99):
        last_loss = adapter.train_one_epoch(vectors, labels, optimizer)

    assert last_loss < first_loss


def test_train_switches_back_to_eval_after_predict():
    """After predict() the model must be in eval mode (Dropout disabled)."""
    adapter = _make_adapter()
    adapter.predict(_dummy())
    assert not adapter.training


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def test_save_and_load_preserves_predictions():
    adapter = _make_adapter()
    x = _dummy()
    original_pred = adapter.predict(x)

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        adapter.save(f.name)
        reloaded = _make_adapter()
        reloaded.load(f.name)

    assert reloaded.predict(x) == original_pred
