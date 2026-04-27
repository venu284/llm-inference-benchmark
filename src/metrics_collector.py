"""
Metrics Collector - Aggregates raw benchmark data into summary statistics.
Computes mean, median, P99, std and writes structured CSV output.
"""

import csv
import os
import numpy as np


def compute_metrics(raw_results: dict, framework: str, seq_length: int,
                    max_new_tokens: int, compile_meta: dict, gpu_summary: dict):
    latencies = np.array(raw_results['latencies'])
    batch_size = raw_results['batch_size']
    num_tokens = raw_results['num_generated_tokens']
    total_tokens = num_tokens * batch_size

    return {
        'framework': framework,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'max_new_tokens': max_new_tokens,
        'ttft_ms': round(raw_results['ttft_ms'], 2),
        'per_token_latency_ms': round(np.mean(latencies) / num_tokens, 2),
        'latency_mean_ms': round(np.mean(latencies), 2),
        'latency_median_ms': round(np.median(latencies), 2),
        'latency_p99_ms': round(np.percentile(latencies, 99), 2),
        'latency_std_ms': round(np.std(latencies), 2),
        'throughput_tok_per_sec': round(total_tokens / (np.mean(latencies) / 1000), 1),
        'peak_memory_mb': round(raw_results['peak_memory_mb'], 1),
        'gpu_util_mean_pct': round(gpu_summary.get('gpu_util_mean_pct', 0), 1),
        'power_draw_mean_w': round(gpu_summary.get('power_draw_mean_w', 0), 1),
        'compile_time_sec': round(compile_meta.get('compile_time_sec', 0), 2),
        'num_graph_breaks': compile_meta.get('num_graph_breaks', 0),
    }


def write_csv(metrics_list: list, output_path: str):
    if not metrics_list:
        return

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fieldnames = list(metrics_list[0].keys())

    file_exists = os.path.isfile(output_path)
    mode = 'a' if file_exists else 'w'

    with open(output_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in metrics_list:
            writer.writerow(row)

    print(f"Results written to {output_path} ({len(metrics_list)} rows)")
