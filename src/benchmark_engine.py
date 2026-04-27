"""
Benchmark Engine - Core execution loop with CUDA-synchronized timing.
Runs warm-up iterations (discarded) then measured iterations with precise timing.
"""

import torch
import time
import numpy as np


def run_benchmark(model, inputs, max_new_tokens: int = 128,
                  warmup_runs: int = 5, measured_runs: int = 30):
    latencies = []
    ttft_values = []

    # Warm-up (results discarded — JIT compilation + GPU thermal stability)
    for i in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()

    # Reset peak memory tracking after warm-up
    torch.cuda.reset_peak_memory_stats()

    # Measured runs
    for i in range(measured_runs):
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        torch.cuda.synchronize()
        t_end = time.perf_counter()

        latency_ms = (t_end - t_start) * 1000
        latencies.append(latency_ms)

        # Tokens generated
        num_generated = outputs.shape[1] - inputs['input_ids'].shape[1]

    # TTFT measurement (single prefill pass)
    ttft_ms = measure_ttft(model, inputs)

    # Memory
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

    return {
        'latencies': latencies,
        'ttft_ms': ttft_ms,
        'num_generated_tokens': num_generated,
        'batch_size': inputs['input_ids'].shape[0],
        'peak_memory_mb': peak_mem_mb,
    }


def measure_ttft(model, inputs):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    with torch.no_grad():
        _ = model(**inputs)

    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event)  # milliseconds
