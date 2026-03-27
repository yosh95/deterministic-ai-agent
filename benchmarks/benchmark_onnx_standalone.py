import os
import sys
import time
from pathlib import Path
from statistics import mean

import psutil

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ONLY import ONNX related classes.
# Due to our refactor, these should NOT trigger torch loading.
from deterministic_ai_agent.adapter.onnx_classifier import OnnxIntentClassifier
from deterministic_ai_agent.encoder.onnx_model import OnnxEmbeddingEncoder
from deterministic_ai_agent.executor.engine import AgentEngine


def get_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_standalone_onnx_benchmark():
    print("=" * 60)
    print("   STANDALONE ONNX BENCHMARK (ZERO TORCH)")
    print("=" * 60)

    # Check if torch is loaded in this process
    is_torch_loaded = "torch" in sys.modules
    print(f"PyTorch Loaded: {is_torch_loaded}")

    # 1. Initialization
    print(f"\n[{time.strftime('%H:%M:%S')}] Initializing FULL ONNX Agent...")
    start_init = time.perf_counter()
    mem_before = get_memory_mb()

    onnx_path = Path("models/onnx")
    if not (onnx_path / "adapter.onnx").exists():
        print("Error: ONNX models not found. Please export them first.")
        return

    encoder = OnnxEmbeddingEncoder(
        model_path=onnx_path / "encoder/model.onnx",
        tokenizer_path=onnx_path / "encoder/tokenizer.json",
    )
    adapter = OnnxIntentClassifier(
        model_path=onnx_path / "adapter.onnx", metadata_path=onnx_path / "metadata.json"
    )
    engine = AgentEngine(encoder, adapter)

    end_init = time.perf_counter()
    mem_after = get_memory_mb()

    init_time = end_init - start_init
    mem_usage = mem_after - mem_before

    print(f"Initialization Time : {init_time:.2f} s")
    print(f"Memory Footprint    : {mem_usage:.2f} MB")
    print(f"Process Total RAM   : {mem_after:.2f} MB")

    # Verify again that torch was not accidentally pulled in
    is_torch_loaded_after = "torch" in sys.modules
    print(f"PyTorch Loaded (After Init): {is_torch_loaded_after}")

    # 2. Inference Benchmark
    test_inputs = [
        "Check inventory for Sensor_A",
        "Stop production line 3 motor overheat",
        "What's the weather in Tokyo?",
    ]

    print(f"\n[{time.strftime('%H:%M:%S')}] Running Inference (50 iterations per input)...")
    all_latencies = []

    for text in test_inputs:
        latencies = []
        # Warm-up
        for _ in range(5):
            engine.run_step(text)

        # Measurement
        for _ in range(50):
            t0 = time.perf_counter()
            engine.run_step(text)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        avg_l = mean(latencies)
        all_latencies.extend(latencies)
        print(f"Input: {text[:25]:<25} | Avg: {avg_l:6.2f} ms")

    total_avg = mean(all_latencies)
    print("-" * 60)
    print(f"Overall Average Latency : {total_avg:.2f} ms")
    print(f"Estimated Throughput    : {1000 / total_avg:.2f} actions/sec")
    print("=" * 60)


if __name__ == "__main__":
    # Disable logs to focus on speed
    import logging

    logging.getLogger("deterministic_ai_agent.executor.engine").setLevel(logging.ERROR)

    run_standalone_onnx_benchmark()
