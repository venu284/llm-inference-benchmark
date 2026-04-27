#!/usr/bin/env python3
"""
Main Benchmark Orchestrator
CSCI 8970 - LLM Inference Benchmarking
Authors: Venu Dattathreya Vemuru, Varshith Peddineni

Reads experiment_config.yaml and runs the full benchmark matrix:
  frameworks × batch_sizes × seq_lengths
Outputs results to CSV and prints progress.
"""

import os
import sys
import time
import yaml
import argparse
import gc
import torch

from src.model_loader import load_model, load_tokenizer
from src.prompt_generator import generate_prompts
from src.benchmark_engine import run_benchmark
from src.metrics_collector import compute_metrics, write_csv
from src.gpu_monitor import GPUMonitor


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def clear_gpu():
    """Aggressively free GPU memory between framework runs."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    time.sleep(2)  # Let CUDA fully release


def run_single_config(model, tokenizer, framework_name, batch_size, seq_length,
                      config, compile_meta, gpu_monitor):
    """Run benchmark for a single (framework, batch_size, seq_length) config."""
    max_new_tokens = config['model']['max_new_tokens']
    warmup_runs = config['experiment']['warmup_runs']
    measured_runs = config['experiment']['measured_runs']
    seed = config['experiment']['seed']

    # Generate prompts
    inputs = generate_prompts(tokenizer, batch_size, seq_length, seed=seed)

    # Start GPU monitoring
    gpu_monitor.start()

    # Run benchmark
    raw_results = run_benchmark(
        model, inputs,
        max_new_tokens=max_new_tokens,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs
    )

    # Stop GPU monitoring and get summary
    gpu_monitor.stop()
    gpu_summary = gpu_monitor.get_summary()

    # Compute metrics
    metrics = compute_metrics(
        raw_results=raw_results,
        framework=framework_name,
        seq_length=seq_length,
        max_new_tokens=max_new_tokens,
        compile_meta=compile_meta,
        gpu_summary=gpu_summary
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description='LLM Inference Benchmark Runner')
    parser.add_argument('--config', type=str, default='config/experiment_config.yaml',
                        help='Path to experiment config YAML')
    parser.add_argument('--frameworks', type=str, nargs='+', default=None,
                        help='Run only specific frameworks (e.g., eager compile_default)')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=None,
                        help='Override batch sizes')
    parser.add_argument('--seq-lengths', type=int, nargs='+', default=None,
                        help='Override sequence lengths')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print experiment matrix without running')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    model_name = config['model']['name']
    precision = config['model']['precision']

    # Determine experiment matrix
    frameworks = args.frameworks or [fw['name'] for fw in config['frameworks']]
    batch_sizes = args.batch_sizes or config['experiment']['batch_sizes']
    seq_lengths = args.seq_lengths or config['experiment']['seq_lengths']

    total_configs = len(frameworks) * len(batch_sizes) * len(seq_lengths)
    warmup = config['experiment']['warmup_runs']
    measured = config['experiment']['measured_runs']
    total_runs = total_configs * (warmup + measured)

    print("=" * 70)
    print("  LLM Inference Benchmark")
    print("  CSCI 8970 | Venu & Varshith | Spring 2026")
    print("=" * 70)
    print(f"\n  Model:       {model_name}")
    print(f"  Precision:   {precision}")
    print(f"  Frameworks:  {frameworks}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Seq lengths: {seq_lengths}")
    print(f"  Warmup/Measured: {warmup}/{measured}")
    print(f"  Total configs:   {total_configs}")
    print(f"  Total inference runs: {total_runs}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Experiment matrix:")
        for fw in frameworks:
            for bs in batch_sizes:
                for sl in seq_lengths:
                    print(f"  {fw} | batch={bs} | seq={sl}")
        print(f"\nTotal: {total_configs} configs, {total_runs} runs")
        return

    # Output path
    results_dir = config['output']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, config['output']['csv_filename'])

    # Load tokenizer once
    print(f"\nLoading tokenizer for {model_name}...")
    tokenizer = load_tokenizer(model_name)

    # GPU monitor
    poll_ms = config['monitoring']['gpu_poll_interval_ms']
    gpu_monitor = GPUMonitor(poll_interval_ms=poll_ms)

    all_metrics = []
    config_num = 0

    for framework_name in frameworks:
        print(f"\n{'='*70}")
        print(f"  FRAMEWORK: {framework_name}")
        print(f"{'='*70}")

        # Load model for this framework
        try:
            model, compile_meta = load_model(
                framework_name, model_name, precision=precision
            )
        except Exception as e:
            print(f"  [ERROR] Failed to load {framework_name}: {e}")
            print(f"  Skipping all configs for {framework_name}")
            continue

        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                config_num += 1
                print(f"\n  [{config_num}/{total_configs}] "
                      f"{framework_name} | batch={batch_size} | seq={seq_length}")

                try:
                    metrics = run_single_config(
                        model, tokenizer, framework_name,
                        batch_size, seq_length,
                        config, compile_meta, gpu_monitor
                    )
                    all_metrics.append(metrics)

                    # Print key results
                    print(f"    Latency: {metrics['latency_mean_ms']:.1f}ms "
                          f"(P99: {metrics['latency_p99_ms']:.1f}ms)")
                    print(f"    Throughput: {metrics['throughput_tok_per_sec']:.1f} tok/s")
                    print(f"    Peak Memory: {metrics['peak_memory_mb']:.0f} MB")
                    print(f"    TTFT: {metrics['ttft_ms']:.2f} ms")

                except torch.cuda.OutOfMemoryError:
                    print(f"    [OOM] Skipped — GPU out of memory")
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    print(f"    [ERROR] {e}")
                    continue

        # Unload model before loading next framework
        del model
        clear_gpu()
        print(f"\n  Unloaded {framework_name}, freed GPU memory.")

    # Write all results
    if all_metrics:
        write_csv(all_metrics, csv_path)
        print(f"\n{'='*70}")
        print(f"  BENCHMARK COMPLETE")
        print(f"  Results: {csv_path} ({len(all_metrics)} rows)")
        print(f"{'='*70}")
    else:
        print("\n  [WARN] No results collected!")


if __name__ == "__main__":
    main()
