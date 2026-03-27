import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


from src.adapter.classifier import IntentAdapter
from src.encoder.model import EmbeddingEncoder
from src.executor.engine import AgentEngine, IntentID


def export_models():
    print("--- Exporting Models to ONNX ---")

    # 1. Setup paths
    models_dir = Path("models/onnx")
    models_dir.mkdir(parents=True, exist_ok=True)

    # 2. Skip Encoder for now if it fails due to filesystem limits
    print("Training and Exporting IntentAdapter (Custom Classifier)...")
    model_id = "intfloat/multilingual-e5-small"
    encoder = EmbeddingEncoder(model_id)
    adapter = IntentAdapter(input_dim=encoder.embedding_dimension, num_classes=len(IntentID))
    engine = AgentEngine(encoder, adapter)

    # Train on sample data
    data_path = Path("data/sample_data.json")
    engine.train(data_path, epochs=10)

    # Export adapter to ONNX
    adapter_onnx_path = models_dir / "adapter.onnx"
    adapter.export_to_onnx(str(adapter_onnx_path), encoder.embedding_dimension)

    # Export metadata (centroids)
    metadata = {
        "num_classes": adapter.num_classes,
        "centroids": adapter.centroids.detach().cpu().numpy().tolist(),
    }
    with open(models_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Adapter export complete. Files in {models_dir}")


if __name__ == "__main__":
    export_models()
