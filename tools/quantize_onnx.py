import os
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_model(model_input, model_output):
    print(f"Quantizing {model_input} to {model_output}...")
    quantize_dynamic(
        model_input=model_input, model_output=model_output, weight_type=QuantType.QInt8
    )

    input_size = os.path.getsize(model_input) / (1024 * 1024)
    output_size = os.path.getsize(model_output) / (1024 * 1024)
    print(f"Original size: {input_size:.2f} MB")
    print(f"Quantized size: {output_size:.2f} MB")
    print(f"Reduction: {(1 - output_size / input_size) * 100:.2f}%")


def main():
    # 1. Quantize Encoder
    encoder_path = Path("models/onnx/encoder/model.onnx")
    quantized_encoder_path = Path("models/onnx/encoder/model_quantized.onnx")

    if encoder_path.exists():
        quantize_model(str(encoder_path), str(quantized_encoder_path))
    else:
        print(f"Encoder not found at {encoder_path}")

    # 2. Quantize Adapter
    adapter_path = Path("models/onnx/adapter.onnx")
    quantized_adapter_path = Path("models/onnx/adapter_quantized.onnx")

    if adapter_path.exists():
        quantize_model(str(adapter_path), str(quantized_adapter_path))
    else:
        print(f"Adapter not found at {adapter_path}")


if __name__ == "__main__":
    main()
