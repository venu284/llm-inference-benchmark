"""
Model Loader - Framework-specific model preparation
Loads Mistral-7B and prepares it for Eager, torch.compile, or TensorRT execution.
Supports FP16 and 4-bit (NF4) quantization for memory-constrained GPUs.
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_eager(model_name: str, precision: str = "fp16"):
    if precision == "4bit":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant_config, device_map="cuda"
        )
    else:
        dtype = torch.float16 if precision == "fp16" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="cuda"
        )
    model.eval()
    return model, {"compile_time_sec": 0.0, "num_graph_breaks": 0}


def load_compile(model_name: str, mode: str = "default", precision: str = "fp16"):
    model, _ = load_eager(model_name, precision)
    
    t_start = time.time()
    compiled_model = torch.compile(model, mode=mode)
    compile_time = time.time() - t_start
    
    # Count graph breaks
    try:
        import torch._dynamo as dynamo
        explanation = dynamo.explain(compiled_model)
        graph_breaks = explanation.graph_break_count if hasattr(explanation, 'graph_break_count') else 0
    except Exception:
        graph_breaks = -1  # Could not determine
    
    return compiled_model, {
        "compile_time_sec": compile_time,
        "num_graph_breaks": graph_breaks
    }


def load_tensorrt(model_name: str, precision: str = "fp16"):
    model, _ = load_eager(model_name, precision)
    
    try:
        import torch_tensorrt
        
        enabled_precisions = {torch.float16} if precision == "fp16" else {torch.float32}
        
        t_start = time.time()
        trt_model = torch_tensorrt.compile(
            model,
            ir="dynamo",
            enabled_precisions=enabled_precisions,
            min_block_size=1,
        )
        build_time = time.time() - t_start
        
        return trt_model, {
            "compile_time_sec": build_time,
            "num_graph_breaks": 0
        }
    except ImportError:
        print("[WARN] torch_tensorrt not installed. Returning eager model as fallback.")
        return model, {"compile_time_sec": -1.0, "num_graph_breaks": -1}
    except Exception as e:
        print(f"[WARN] TensorRT conversion failed: {e}. Returning eager model.")
        return model, {"compile_time_sec": -1.0, "num_graph_breaks": -1}


def load_model(framework: str, model_name: str, **kwargs):
    precision = kwargs.get("precision", "fp16")
    loaders = {
        "eager": lambda: load_eager(model_name, precision),
        "compile_default": lambda: load_compile(model_name, "default", precision),
        "compile_reduce_overhead": lambda: load_compile(model_name, "reduce-overhead", precision),
        "compile_max_autotune": lambda: load_compile(model_name, "max-autotune", precision),
        "tensorrt": lambda: load_tensorrt(model_name, precision),
    }
    
    if framework not in loaders:
        raise ValueError(f"Unknown framework: {framework}. Options: {list(loaders.keys())}")
    
    print(f"Loading model with framework: {framework}")
    model, meta = loaders[framework]()
    print(f"  Compile time: {meta['compile_time_sec']:.2f}s | Graph breaks: {meta['num_graph_breaks']}")
    return model, meta
