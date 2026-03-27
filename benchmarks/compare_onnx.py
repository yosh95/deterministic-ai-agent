import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import psutil

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Current (PyTorch) versions
from deterministic_ai_agent.adapter.classifier import IntentAdapter

# ONNX (Optimized) versions
from deterministic_ai_agent.adapter.onnx_classifier import OnnxIntentClassifier
from deterministic_ai_agent.encoder.model import EmbeddingEncoder
from deterministic_ai_agent.encoder.onnx_model import OnnxEmbeddingEncoder
from deterministic_ai_agent.executor.engine import AgentEngine
from deterministic_ai_agent.executor.registry import IntentID


def get_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def measure_inference(engine: Any, inputs: list[str], iterations: int = 50):
    latencies = []
    for text in inputs:
        # Warm-up
        for _ in range(5):
            engine.run_step(text)

        # Actual measurement
        for _ in range(iterations):
            t0 = time.perf_counter()
            engine.run_step(text)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms
    return latencies


def run_comparison():
    print("=" * 60)
    print("   AI Agent Performance Comparison: PyTorch vs. FULL ONNX")
    print("=" * 60)

    test_inputs = [
        "Check inventory for Sensor_A",
        "Stop production line 3 motor overheat",
        "What's the weather in Tokyo?",
    ]

    # 1. PyTorch Baseline
    print("\n[1/2] Benchmarking PyTorch Baseline...")
    mem_start_pt = get_memory_mb()
    t_start_pt = time.perf_counter()

    import logging

    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

    encoder_pt = EmbeddingEncoder()
    adapter_pt = IntentAdapter(input_dim=encoder_pt.embedding_dimension, num_classes=len(IntentID))
    engine_pt = AgentEngine(encoder_pt, adapter_pt)
    engine_pt.train("data/sample_data.json", epochs=5)

    t_end_pt = time.perf_counter()
    mem_end_pt = get_memory_mb()

    init_pt = t_end_pt - t_start_pt
    mem_pt = mem_end_pt - mem_start_pt
    latencies_pt = measure_inference(engine_pt, test_inputs, iterations=20)

    print(f"  Init Time: {init_pt:5.2f} s")
    print(f"  RAM Usage: {mem_pt:5.2f} MB")
    print(f"  Avg Latency: {mean(latencies_pt):5.2f} ms")

    # 2. FULL ONNX (Actual Measurement)
    onnx_path = Path("models/onnx")
    print("\n[2/2] Benchmarking FULL ONNX Optimized...")

    # Clean memory as much as possible before starting ONNX (not perfect in Python)
    del engine_pt, adapter_pt, encoder_pt
    import gc

    gc.collect()

    mem_start_ox = get_memory_mb()
    t_start_ox = time.perf_counter()

    encoder_ox = OnnxEmbeddingEncoder(
        model_path=onnx_path / "encoder/model.onnx",
        tokenizer_path=onnx_path / "encoder/tokenizer.json",
    )
    adapter_ox = OnnxIntentClassifier(
        model_path=onnx_path / "adapter.onnx", metadata_path=onnx_path / "metadata.json"
    )
    engine_ox = AgentEngine(encoder_ox, adapter_ox)

    t_end_ox = time.perf_counter()
    mem_end_ox = get_memory_mb()

    init_ox = t_end_ox - t_start_ox
    mem_ox = mem_end_ox - mem_start_ox
    latencies_ox = measure_inference(engine_ox, test_inputs, iterations=20)

    print(f"  Init Time: {init_ox:5.2f} s")
    print(f"  RAM Usage: {mem_ox:5.2f} MB")
    print(f"  Avg Latency: {mean(latencies_ox):5.2f} ms")

    # Summary
    print("\n" + "-" * 60)
    print("  ONNX ACTUAL IMPROVEMENT SUMMARY")
    print("-" * 60)
    print(f"  Cold Start Speedup : {init_pt / init_ox:4.1f}x FASTER")
    print(f"  Memory Efficiency  : {mem_pt / mem_ox:4.1f}x LIGHTER")
    print(f"  Inference Speedup  : {mean(latencies_pt) / mean(latencies_ox):4.1f}x FASTER")
    print("-" * 60)

    print("\n" + "=" * 60)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Disable engine logs
    import logging

    logging.getLogger("deterministic_ai_agent.executor.engine").setLevel(logging.ERROR)

    run_comparison()
