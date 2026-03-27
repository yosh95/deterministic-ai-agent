import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
except ImportError:
    print("Error: Missing training dependencies. Run 'pip install .[train]' first.")
    sys.exit(1)

from deterministic_ai_agent.adapter.classifier import IntentAdapter
from deterministic_ai_agent.encoder.model import EmbeddingEncoder
from deterministic_ai_agent.executor.engine import AgentEngine
from deterministic_ai_agent.executor.registry import IntentID
from tools.generate_ot_data import generate_ot_data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_production_models():
    """
    Consolidated script to generate data, train the adapter, and export all components to ONNX.
    Ensures the 'models/' directory is fully populated for production inference.
    """
    models_dir = Path("models")
    onnx_dir = models_dir / "onnx"
    encoder_dir = onnx_dir / "encoder"

    onnx_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Domain-Specific Training Data
    logger.info("Step 1: Generating OT-domain training data...")
    data = generate_ot_data(num_samples=500)
    data_path = Path("data/ot_training_data.json")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f"Generated {len(data)} samples in {data_path}")

    # 2. Initialize and Train the Adapter
    logger.info("Step 2: Training the Intent Adapter...")
    model_id = "intfloat/multilingual-e5-small"
    encoder_pt = EmbeddingEncoder(model_id)
    adapter_pt = IntentAdapter(input_dim=encoder_pt.embedding_dimension, num_classes=len(IntentID))
    engine = AgentEngine(encoder_pt, adapter_pt)

    # Train for a sufficient number of epochs to ensure good centroids/weights
    engine.train(data_path, epochs=30)

    # Save the PyTorch version (for future fine-tuning)
    pt_adapter_path = models_dir / "production_adapter.pt"
    adapter_pt.save(str(pt_adapter_path))
    logger.info(f"Saved PyTorch adapter to {pt_adapter_path}")

    # 3. Export Encoder to ONNX (using Optimum)
    logger.info("Step 3: Exporting Encoder (BERT-based) to ONNX...")
    try:
        model_enc = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_enc.save_pretrained(encoder_dir)
        tokenizer.save_pretrained(encoder_dir)
        logger.info(f"Encoder exported successfully to {encoder_dir}")
    except Exception as e:
        logger.error(f"Failed to export encoder: {e}")
        # Note: In some environments (like restricted CI), this might fail.
        # But for the recommended build process, it should work.

    # 4. Export Adapter to ONNX and Save Metadata
    logger.info("Step 4: Exporting Intent Adapter to ONNX...")
    adapter_onnx_path = onnx_dir / "production_adapter.onnx"
    adapter_pt.export_to_onnx(str(adapter_onnx_path), encoder_pt.embedding_dimension)

    metadata = {
        "num_classes": adapter_pt.num_classes,
        "centroids": adapter_pt.centroids.detach().cpu().numpy().tolist(),
    }
    metadata_path = onnx_dir / "production_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Adapter exported successfully to {adapter_onnx_path}")
    logger.info(f"Metadata (centroids) saved to {metadata_path}")

    logger.info("--- Production Model Build Complete ---")


if __name__ == "__main__":
    build_production_models()
