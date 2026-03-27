import os

import onnx
from onnx import shape_inference
from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    encoder_path = "models/onnx/encoder/model.onnx"
    quantized_path = "models/onnx/encoder/model_quantized.onnx"

    print(f"Loading {encoder_path}...")
    model = onnx.load(encoder_path)

    # Try shape inference
    print("Inferring shapes...")
    try:
        model = shape_inference.infer_shapes(model)
        print("Shape inference successful.")
    except Exception as e:
        print(f"Shape inference failed: {e}. Proceeding anyway...")

    # Save a temporary inferred model to check
    # inferred_path = "models/onnx/encoder/model_manual_inferred.onnx"
    # onnx.save(model, inferred_path)

    # Quantize
    print(f"Quantizing to {quantized_path}...")
    # Note: quantize_dynamic typically expects file paths, not model objects.

    # Let's use a temporary file for quantization that we manage ourselves
    temp_model_path = "models/onnx/encoder/temp_for_quant.onnx"
    onnx.save(model, temp_model_path)

    try:
        quantize_dynamic(
            model_input=temp_model_path,
            model_output=quantized_path,
            weight_type=QuantType.QInt8,
            # Skip shape inference within quantize_dynamic if possible
            # Actually onnxruntime's quantize_dynamic always tries to run shape inference.
        )
        print("Quantization successful.")
    except Exception as e:
        print(f"Quantization failed: {e}")
    finally:
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)


if __name__ == "__main__":
    main()
