# LLM Inference Benchmarking: PyTorch Eager vs torch.compile vs TensorRT

**CSCI 8970 — Advanced Topics in Machine Learning Systems | Spring 2026**  
**University of Georgia | Instructor: Prof. Wei Niu**  
**Team: Venu Dattathreya Vemuru & Varshith Peddineni**

## Overview

This project benchmarks large language model inference across three ML frameworks on consumer-grade GPU hardware (NVIDIA RTX 3090). We measure latency, throughput, GPU memory, compilation overhead, and GPU utilization to provide data-driven framework selection guidance.

## Frameworks Compared

| Framework | Optimization Level | Effort |
|-----------|-------------------|--------|
| PyTorch Eager | None (baseline) | 0 lines |
| torch.compile | JIT graph compilation (TorchDynamo + TorchInductor) | 1 line |
| TensorRT | Hardware-specific kernel auto-tuning | Setup + build |

## Project Structure

```
llm-inference-bench/
├── config/
│   └── experiment_config.yaml    # All experiment parameters
├── src/
│   ├── model_loader.py           # Framework-specific model loading
│   ├── prompt_generator.py       # Standardized prompt creation
│   ├── benchmark_engine.py       # Core benchmarking with CUDA sync
│   ├── metrics_collector.py      # Statistics computation + CSV output
│   ├── gpu_monitor.py            # Background nvidia-smi polling
│   ├── verify_environment.py     # Environment verification script
│   └── validate_model.py         # Model loading validation
├── analysis/
│   ├── analyze_results.py        # Aggregate analysis
│   └── plot_charts.py            # Visualization generation
├── docs/
│   └── literature_review.md      # 14-paper literature review
├── results/                      # Raw CSV output
├── figures/                      # Generated charts
├── requirements.txt              # Python dependencies
└── README.md
```

## Quick Start

```bash
# 1. Verify environment
python src/verify_environment.py

# 2. Validate model loads on GPU
python src/validate_model.py

# 3. Run benchmarks (coming soon)
# python run_benchmark.py --config config/experiment_config.yaml
```

## Experiment Matrix

- **Model:** Mistral-7B (FP16, ~14 GB VRAM)
- **GPU:** NVIDIA RTX 3090 (24 GB)
- **Batch sizes:** 1, 4, 8
- **Sequence lengths:** 256, 512, 1024 tokens
- **Iterations:** 5 warm-up + 30 measured per config
- **Total:** 27 configs × 35 runs = 945 inference runs

## Metrics Collected (16 columns)

| Group | Columns |
|-------|---------|
| Config | framework, batch_size, seq_length, max_new_tokens |
| Latency | ttft_ms, per_token_latency_ms, latency_mean_ms, latency_p99_ms |
| System | throughput_tok/s, peak_memory_mb, gpu_util_mean_%, power_draw_w |
| Overhead | compile_time_s, latency_median_ms, latency_std_ms, num_graph_breaks |
