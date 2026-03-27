import os
import sys

# Set the project root to sys.path
sys.path.insert(0, os.getcwd())

from deterministic_ai_agent.adapter.onnx_classifier import OnnxIntentClassifier
from deterministic_ai_agent.encoder.onnx_model import OnnxEmbeddingEncoder
from deterministic_ai_agent.executor.engine import AgentEngine
from deterministic_ai_agent.executor.registry import IntentID


def test_v2():
    print("--- Testing V2 Deterministic Agent (ONNX Backend) ---")

    # 1. Initialize with V2 models
    encoder = OnnxEmbeddingEncoder(
        model_path="models/onnx/encoder/model.onnx",
        tokenizer_path="models/onnx/encoder/tokenizer.json",
    )
    classifier = OnnxIntentClassifier(
        model_path="models/onnx/adapter_v2.onnx", metadata_path="models/onnx/metadata_v2.json"
    )

    engine = AgentEngine(encoder, classifier)

    # 2. Test Cases
    test_queries = [
        # In-Distribution (OT domain)
        "Critical: Motor_A reports overheat (Code E404). Request diagnostics.",
        "Check stock for Sensor_Z spare parts.",
        "Log: Conveyor_1 manual stop by operator Tanaka.",
        "Robotic_Arm_03 の通信エラー。診断を開始してください。",  # Japanese
        # Low Confidence / Ambiguous (Should trigger human escalation or low confidence logic)
        "I think maybe Motor_A is acting weird but I'm not sure.",
        # Out-of-Distribution (OOD)
        "What is the capital of France?",
        "Tell me a joke about robots.",
        "How do I cook pasta?",
    ]

    print(f"{'Query':<60} | {'Intent':<12} | {'Conf':<6} | {'OOD':<6} | {'Action'}")
    print("-" * 110)

    for query in test_queries:
        result = engine.run_step(query)

        # Access internals for reporting
        vector = encoder.encode(query)
        action_id, conf = classifier.predict_with_confidence(vector)
        ood_score = classifier.get_ood_score(vector)

        status = "EXECUTED" if "success" in result and result.get("success") else "REFUSED"
        reason = result.get("reason", "N/A")

        intent_name = IntentID(action_id).name

        output = (
            f"{query[:58]:<60} | {intent_name:<12} | "
            f"{conf:.4f} | {ood_score:.4f} | {status} ({reason})"
        )
        print(output)


if __name__ == "__main__":
    test_v2()
