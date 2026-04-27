#!/usr/bin/env python3
"""
Environment Setup Verification Script
CSCI 8970 - LLM Inference Benchmarking
Verifies all dependencies, GPU access, and framework availability.
"""

import sys
import subprocess
import importlib

def check(name, fn):
    try:
        result = fn()
        print(f"  [PASS] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False

def main():
    print("=" * 65)
    print("  LLM Inference Benchmark - Environment Verification")
    print("  CSCI 8970 | Venu & Varshith | Spring 2026")
    print("=" * 65)
    
    passed, total = 0, 0
    
    # --- Python ---
    print("\n[1/6] Python Environment")
    total += 1
    if check("Python version", lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"):
        passed += 1
    
    # --- PyTorch ---
    print("\n[2/6] PyTorch & CUDA")
    total += 3
    if check("PyTorch version", lambda: __import__('torch').__version__):
        passed += 1
    if check("CUDA available", lambda: f"{__import__('torch').cuda.is_available()} (device count: {__import__('torch').cuda.device_count()})"):
        passed += 1
    if check("GPU name", lambda: __import__('torch').cuda.get_device_name(0)):
        passed += 1
    
    # --- GPU Details ---
    print("\n[3/6] GPU Details")
    total += 3
    import torch
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        if check("GPU memory", lambda: f"{gpu_mem:.1f} GB"):
            passed += 1
        if check("CUDA version", lambda: torch.version.cuda):
            passed += 1
        if check("cuDNN version", lambda: str(torch.backends.cudnn.version())):
            passed += 1
    
    # --- HuggingFace ---
    print("\n[4/6] HuggingFace Transformers")
    total += 2
    if check("transformers version", lambda: __import__('transformers').__version__):
        passed += 1
    if check("AutoModelForCausalLM", lambda: "Available" if hasattr(__import__('transformers'), 'AutoModelForCausalLM') else "Missing"):
        passed += 1
    
    # --- torch.compile ---
    print("\n[5/6] torch.compile Availability")
    total += 2
    if check("torch.compile", lambda: "Available" if hasattr(torch, 'compile') else "Not available (need PyTorch 2.0+)"):
        passed += 1
    if check("torch._dynamo", lambda: f"Available (TorchDynamo)" if importlib.util.find_spec('torch._dynamo') else "Missing"):
        passed += 1
    
    # --- nvidia-smi ---
    print("\n[6/6] nvidia-smi (GPU Monitoring)")
    total += 1
    def check_nvidia_smi():
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version,temperature.gpu',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    if check("nvidia-smi query", check_nvidia_smi):
        passed += 1
    
    # --- Summary ---
    print("\n" + "=" * 65)
    print(f"  RESULT: {passed}/{total} checks passed")
    if passed == total:
        print("  STATUS: Environment is READY for benchmarking")
    else:
        print(f"  STATUS: {total - passed} issue(s) need attention")
    print("=" * 65)

if __name__ == "__main__":
    main()
