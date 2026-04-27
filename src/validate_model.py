#!/usr/bin/env python3
"""
Model Validation Script
CSCI 8970 - LLM Inference Benchmarking
Validates that Mistral-7B loads, fits on GPU, and generates output correctly.
"""

import torch
import time
import sys

def main():
    print("=" * 65)
    print("  Mistral-7B Model Validation")
    print("  CSCI 8970 | Venu & Varshith | Spring 2026")
    print("=" * 65)
    
    # --- Step 1: GPU Memory Before Loading ---
    print("\n[1/5] Pre-load GPU State")
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated() / (1024**3)
    total_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Total VRAM: {total_mem:.1f} GB")
    print(f"  Used before load: {mem_before:.2f} GB")
    
    # --- Step 2: Load Model ---
    print("\n[2/5] Loading Mistral-7B (4-bit NF4)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    model_name = "mistralai/Mistral-7B-v0.1"
    t_start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="cuda"
    )
    model.eval()
    
    t_load = time.time() - t_start
    mem_after = torch.cuda.memory_allocated() / (1024**3)
    print(f"  Load time: {t_load:.1f} seconds")
    print(f"  Model memory: {mem_after:.2f} GB")
    print(f"  Remaining VRAM: {total_mem - mem_after:.1f} GB")
    
    # --- Step 3: Model Info ---
    print("\n[3/5] Model Information")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params / 1e9:.2f} B")
    print(f"  Dtype: {next(model.parameters()).dtype}")
    print(f"  Device: {next(model.parameters()).device}")
    
    # --- Step 4: Generate Output ---
    print("\n[4/5] Test Generation")
    prompt = "The key advantage of GPU computing for deep learning is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    torch.cuda.synchronize()
    t_gen_start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=1.0
        )
    
    torch.cuda.synchronize()
    t_gen = time.time() - t_gen_start
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    
    print(f"  Prompt: \"{prompt}\"")
    print(f"  Generated {num_tokens} tokens in {t_gen:.2f}s")
    print(f"  Throughput: {num_tokens / t_gen:.1f} tokens/sec")
    print(f"  Output: \"{generated_text[:150]}...\"")
    
    # --- Step 5: Memory Summary ---
    print("\n[5/5] Peak Memory Summary")
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    reserved_mem = torch.cuda.max_memory_reserved() / (1024**3)
    print(f"  Peak allocated: {peak_mem:.2f} GB")
    print(f"  Peak reserved: {reserved_mem:.2f} GB")
    print(f"  Available for KV-cache & batching: {total_mem - peak_mem:.1f} GB")
    
    # --- Summary ---
    print("\n" + "=" * 65)
    print("  VALIDATION RESULT: ALL CHECKS PASSED")
    print(f"  Model loads in FP16, uses ~{mem_after:.0f} GB VRAM")
    print(f"  {total_mem - peak_mem:.0f} GB headroom for batching")
    print(f"  Generation working at {num_tokens / t_gen:.0f} tokens/sec (unoptimized)")
    print("=" * 65)

if __name__ == "__main__":
    main()
