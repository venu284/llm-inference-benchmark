# Literature Review: Benchmarking LLM Inference Across ML Frameworks
### CSCI 8970 — Advanced Topics in Machine Learning Systems | Spring 2026
### Authors: Venu Dattathreya Vemuru, Varshith Peddineni

---

## 1. LLM Inference Benchmarking (Directly Related Work)

### 1.1 LLM-Inference-Bench (Chitty-Venkata et al., SC Workshops 2024)
- **Paper:** "LLM-Inference-Bench: Inference Benchmarking of Large Language Models on AI Accelerators"
- **Venue:** SC '24 Workshops (PMBS24)
- **Key Contribution:** Comprehensive benchmarking suite evaluating LLM inference across GPUs (NVIDIA A100, H100, AMD) and AI accelerators (Intel Habana, SambaNova). Tests LLaMA, Mistral, and Qwen families at 7B and 70B scales.
- **Key Findings:** GQA models (Mistral-7B, LLaMA-3-8B) are ~1.9-2.79x faster than LLaMA-2-7B on H100/A100. H100 shows more significant speedup over A100 at larger batch sizes.
- **Relevance to Our Work:** Most directly comparable project, but focuses on hardware comparison across datacenter GPUs. We focus on framework comparison on consumer GPUs (RTX 3090).
- **URL:** https://arxiv.org/abs/2411.00136

### 1.2 Bench360 (Stuhlmann et al., 2025)
- **Paper:** "Bench360: Benchmarking Local LLM Inference from 360 Degrees"
- **Key Contribution:** Benchmarks mid-size models (Mistral, Gemma-2, Qwen2.5) across L4, A10, A30 GPUs under a 24GB VRAM budget. Measures latency-quality and energy-quality Pareto frontiers.
- **Key Findings:** Gemma-2 variants dominate both latency-quality and energy-quality frontiers. Startup time and TTFT vary significantly across engines.
- **Relevance:** The 24GB VRAM budget constraint directly parallels our RTX 3090 setup.
- **URL:** https://arxiv.org/abs/2511.16682

### 1.3 Accelerating Deep Learning Inference: A Comparative Analysis (2025)
- **Paper:** "Accelerating Deep Learning Inference: A Comparative Analysis of Modern Acceleration Frameworks"
- **Venue:** MDPI Electronics
- **Key Contribution:** Compares PyTorch, ONNX Runtime, TensorRT, TVM, and JAX on Jetson AGX Orin for image classification models.
- **Key Findings:** TensorRT consistently fastest, PyTorch follows closely with competitive latency. ONNX Runtime lags in throughput.
- **Relevance:** Closest methodology to ours but targets edge hardware, not consumer desktop GPUs.
- **URL:** https://www.mdpi.com/2079-9292/14/15/2977

### 1.4 TokenPowerBench (2025)
- **Paper:** "TokenPowerBench: Benchmarking the Power Consumption of LLM Inference"
- **Key Contribution:** First benchmark for quantifying power consumption and energy cost of LLM inference with phase-aware telemetry (prefill vs decode).
- **Relevance:** Methodology for GPU power measurement via nvidia-smi; could extend our work with energy metrics.
- **URL:** https://arxiv.org/abs/2512.03024

---

## 2. Framework Foundations (Papers Behind What We Test)

### 2.1 PyTorch 2: TorchDynamo and TorchInductor (Ansel et al., ASPLOS 2024)
- **Paper:** "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation"
- **Venue:** ASPLOS 2024
- **Key Contribution:** Introduces TorchDynamo (Python-level JIT graph capture via bytecode modification) and TorchInductor (backend compiler generating Triton/C++ code). Implements torch.compile().
- **Key Findings:** 2.27x inference and 1.41x training speedup on A100 across 180+ real-world models. Outperforms 6 other compilers.
- **Relevance:** The foundational paper for torch.compile, one of our three test frameworks.
- **URL:** https://dl.acm.org/doi/10.1145/3620665.3640366

### 2.2 Collabora torch.compile vs TensorRT Study (December 2024)
- **Source:** Collabora Engineering Blog
- **Key Contribution:** Direct comparison of torch.compile vs TensorRT on LLaMA-7B, LLaMA-3-8B, Mistral, Phi-3, Phi-2 across 3090, 4090, H100 GPUs.
- **Key Findings:** torch.compile demonstrated similar or better performance than TensorRT for these models. Concluded there is little reason to use TensorRT unless tightly coupled with NVIDIA's ecosystem.
- **Relevance:** Our project validates or challenges this finding with more detailed configuration sweeps on RTX 3090.
- **URL:** https://www.collabora.com/news-and-blog/blog/2024/12/19/faster-inference-torch.compile-vs-tensorrt/

### 2.3 NVIDIA TensorRT and Torch-TensorRT
- **Source:** NVIDIA Developer Documentation
- **Key Contribution:** TensorRT performs graph-level and kernel-level optimizations including layer fusion, precision calibration (FP16/INT8), kernel auto-tuning per GPU, and memory optimization. Torch-TensorRT provides PyTorch integration with up to 6x speedup.
- **Relevance:** The dedicated inference optimizer in our comparison. Understanding its optimization passes is essential for interpreting results.
- **URL:** https://developer.nvidia.com/tensorrt

### 2.4 Exploring TensorRT for Real-Time Inference (Zhou et al., ICESS 2022)
- **Paper:** "Exploring TensorRT to Improve Real-Time Inference for Deep Learning"
- **Key Contribution:** Compares TensorRT conversion workflows (ONNX-to-TRT, Torch-TensorRT, ONNX Runtime with TRT). Finds ONNX-to-TRT has best overall performance.
- **Relevance:** Justifies our Torch-TensorRT pathway choice and provides baseline expectations for speedup ratios (1.6-2x for ResNet variants).
- **URL:** https://userweb.cs.txstate.edu/~k_y47/webpage/pubs/icess22.pdf

---

## 3. Key Optimization Techniques (Understanding What Frameworks Do)

### 3.1 FlashAttention (Dao et al., NeurIPS 2022)
- **Paper:** "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
- **Key Contribution:** IO-aware attention algorithm using tiling to reduce HBM reads/writes. Achieves linear memory complexity (vs quadratic), up to 7.6x speedup on GPT-2.
- **Key Findings:** Standard attention wastes bandwidth by materializing the full N×N attention matrix to HBM. FlashAttention fuses all attention ops into a single GPU kernel.
- **Relevance:** All frameworks in our benchmark benefit from FlashAttention. Understanding it explains why attention may or may not be the bottleneck.
- **URL:** https://arxiv.org/abs/2205.14135

### 3.2 FlashAttention-2 (Dao, ICLR 2024)
- **Paper:** "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
- **Key Contribution:** Reduces non-matmul FLOPs, parallelizes across thread blocks, optimizes warp-level work distribution. 2x faster than FlashAttention, reaching 50-73% theoretical FLOPs/s on A100.
- **Relevance:** The attention backend used by Mistral-7B in our tests.
- **URL:** https://arxiv.org/abs/2307.08691

### 3.3 vLLM / PagedAttention (Kwon et al., SOSP 2023)
- **Paper:** "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- **Key Contribution:** PagedAttention stores KV-cache in non-contiguous memory blocks (inspired by OS virtual memory paging). vLLM achieves near-zero KV-cache waste, 2-4x throughput improvement over FasterTransformer/Orca.
- **Key Findings:** Existing systems waste 60-80% of KV-cache memory due to fragmentation. PagedAttention reduces waste to under 4%.
- **Relevance:** Essential context for understanding GPU memory measurements. KV-cache management is a major factor in peak memory consumption.
- **URL:** https://arxiv.org/abs/2309.06180

---

## 4. Surveys (Background Context)

### 4.1 A Survey on Efficient Inference for Large Language Models (Zhou et al., 2024)
- **Key Contribution:** Comprehensive survey analyzing causes of inefficient LLM inference (large model size, quadratic attention, auto-regressive decoding) and organizing solutions into data-level, model-level, and system-level optimization.
- **Relevance:** Frames our project within the broader efficient inference landscape.
- **URL:** https://arxiv.org/abs/2404.14294

### 4.2 A Survey on Inference Engines for LLMs (Park et al., 2025)
- **Key Contribution:** Evaluates 25 open-source and commercial inference engines across ease-of-use, deployment, scalability, and optimization techniques.
- **Relevance:** Maps how engines handle quantization, caching, and parallelization — contextualizes our framework-level comparison.
- **URL:** https://arxiv.org/abs/2505.01658

### 4.3 A Comparative Survey of PyTorch vs TensorFlow (2025)
- **Key Contribution:** Reviews usability, performance, and deployment trade-offs. Consensus: both frameworks are highly optimized; wins vary by model and settings since both use same low-level libraries (cuDNN).
- **Relevance:** Motivates why careful benchmarking is needed rather than relying on assumptions.
- **URL:** https://arxiv.org/abs/2508.04035

---

## Summary

| Category | Count | Key Papers |
|----------|-------|------------|
| Directly Related Benchmarking | 4 | LLM-Inference-Bench, Bench360, Comparative Analysis, TokenPowerBench |
| Framework Foundations | 4 | PyTorch 2 ASPLOS, Collabora Study, TensorRT Docs, Zhou et al. |
| Optimization Techniques | 3 | FlashAttention, FlashAttention-2, vLLM/PagedAttention |
| Surveys | 3 | Zhou et al. Survey, Park et al. Engines Survey, PyTorch vs TF Survey |
| **Total** | **14** | |
