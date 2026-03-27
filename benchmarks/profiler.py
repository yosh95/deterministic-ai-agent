import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import psutil

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deterministic_ai_agent.adapter.classifier import IntentAdapter
from deterministic_ai_agent.encoder.model import EmbeddingEncoder
from deterministic_ai_agent.executor.engine import AgentEngine


def get_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_benchmark():
    print("=" * 60)
    print("   Deterministic AI Agent - Edge Performance Profiler")
    print("=" * 60)

    # 1. Cold Start / Initialization
    print(f"[{time.strftime('%H:%M:%S')}] Initializing components...")
    start_init = time.perf_counter()

    mem_before = get_memory_mb()
    encoder = EmbeddingEncoder()
    adapter = IntentAdapter(input_dim=encoder.embedding_dimension, num_classes=3)
    engine = AgentEngine(encoder, adapter)

    # Train briefly to populate centroids (necessary for OOD check)
    data_path = Path("data/sample_data.json")
    engine.train(data_path, epochs=10)

    end_init = time.perf_counter()
    mem_after = get_memory_mb()

    init_time = end_init - start_init
    mem_usage = mem_after - mem_before

    print(f"Cold Start Time    : {init_time:.2f} seconds")
    print(f"Memory Footprint   : {mem_usage:.2f} MB (Total process: {mem_after:.2f} MB)")
    print(f"PyTorch Device     : {encoder.device}")
    print("-" * 60)

    # 2. Latency & Throughput
    test_inputs = [
        "Stop Motor_B",  # Short
        "Critical vibration alarm detected on Conveyor_A. Run diagnostics.",  # Medium
        "Checking inventory for Sensor_C. Urgent maintenance required for line 3.",  # Long
        "What is the weather today?",  # OOD Case
    ]

    print(f"[{time.strftime('%H:%M:%S')}] Starting Inference Benchmark (50 iterations per case)...")

    results = {}
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
            latencies.append((t1 - t0) * 1000)  # ms

        avg_l = mean(latencies)
        std_l = stdev(latencies)
        results[text] = {"avg": avg_l, "std": std_l}
        all_latencies.extend(latencies)
        print(f"Input: {text[:30]:<30} | Avg Latency: {avg_l:6.2f} ms | Std: {std_l:4.2f} ms")

    # 3. Overall Statistics
    total_avg = mean(all_latencies)
    throughput = 1000 / total_avg  # Actions per second

    print("-" * 60)
    print(f"Overall Average Latency : {total_avg:.2f} ms")
    print(f"Estimated Throughput    : {throughput:.2f} actions/sec")
    print("Safety Gate Logic       : L1(Argmax) + L2(Softmax) + L3(OOD Sim)")
    print("=" * 60)

    # Save summary
    summary = {
        "init_time_sec": init_time,
        "memory_usage_mb": mem_usage,
        "total_memory_mb": mem_after,
        "avg_latency_ms": total_avg,
        "throughput_aps": throughput,
        "device": str(encoder.device),
    }
    with open("benchmarks/results.json", "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    # Disable logging to focus on speed
    import logging

    logging.getLogger("deterministic_ai_agent.executor.engine").setLevel(logging.ERROR)

    run_benchmark()
