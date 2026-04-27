"""
Prompt Generator - Creates standardized test inputs.
Fixed seed ensures identical prompts across all framework runs.
"""

import torch
from transformers import AutoTokenizer


def generate_prompts(tokenizer, batch_size: int, seq_length: int, seed: int = 42):
    torch.manual_seed(seed)
    
    base_prompts = [
        "The key advantage of GPU computing for deep learning is",
        "In recent years, large language models have transformed",
        "Efficient memory management is critical because",
        "The transformer architecture revolutionized NLP by introducing",
        "When comparing inference frameworks, the most important metrics are",
        "Parallel computing enables faster model training through",
        "The trade-off between model accuracy and inference latency",
        "Modern compiler optimizations for neural networks include",
    ]
    
    prompts = [base_prompts[i % len(base_prompts)] for i in range(batch_size)]
    
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_length,
    ).to("cuda")
    
    return inputs
