from pathlib import Path

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer


def export_encoder():
    model_id = "intfloat/multilingual-e5-small"
    save_dir = Path("models/onnx/encoder")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting Encoder {model_id} to {save_dir}...")
    # This is the memory-intensive part
    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Encoder export successful.")


if __name__ == "__main__":
    export_encoder()
