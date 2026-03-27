import pytest

from deterministic_ai_agent.encoder.model import EmbeddingEncoder
from deterministic_ai_agent.ner.extractor import NERExtractor


@pytest.fixture(scope="module")
def encoder():
    return EmbeddingEncoder()


def test_regex_extraction(encoder):
    # Standard exact match defined in config/devices.yaml
    extractor = NERExtractor(encoder=encoder)
    text = "Check status of Conveyor_A immediately."
    params = extractor.extract(text)

    assert params["device_id"] == "Conveyor_A"
    assert params["extraction_method"] == "regex"


def test_semantic_fallback_extraction(encoder):
    # Misspelled or abbreviated name not in regex list
    # "Conv-A" is not in devices.yaml (only "Conveyor_A")
    extractor = NERExtractor(encoder=encoder, semantic_threshold=0.8)
    text = "Is Conv-A running normally?"
    params = extractor.extract(text)

    # Should fallback to semantic and find Conveyor_A
    assert params["device_id"] == "Conveyor_A"
    assert params["extraction_method"] == "semantic"


def test_semantic_threshold_rejection(encoder):
    # Something completely unrelated
    extractor = NERExtractor(encoder=encoder, semantic_threshold=0.95)
    text = "The weather is nice today."
    params = extractor.extract(text)

    # Should not find any device_id
    assert "device_id" not in params


if __name__ == "__main__":
    # Manual run for debugging
    enc = EmbeddingEncoder()
    ext = NERExtractor(encoder=enc, semantic_threshold=0.8)

    test_cases = [
        "Conveyor_A is broken",
        "Conv-A is broken",
        "Motor B overheat",
        "The light is red",
    ]

    for t in test_cases:
        p = ext.extract(t)
        print(f"Input: {t} -> Result: {p.get('device_id')} (via {p.get('extraction_method')})")
