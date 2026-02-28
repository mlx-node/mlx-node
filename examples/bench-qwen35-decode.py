#!/usr/bin/env python3
"""Benchmark Qwen3.5-27B decode speed with Python mlx-lm."""
import time
from pathlib import Path
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL_PATH = str(Path(__file__).parent.parent / ".cache" / "models" / "qwen3.5-27b-mlx-bf16")

def main():
    print(f"Loading model from: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)
    print("Model loaded\n")

    sampler = make_sampler(temp=0.7, top_p=0.9)

    prompt = "Count from 1 to 100, one number per line."
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer.encode(formatted)
    print(f"Prompt tokens: {len(prompt_tokens)}")

    # Warmup
    print("Warming up...")
    _ = generate(model, tokenizer, prompt=formatted, max_tokens=5, sampler=sampler, verbose=False)

    # Benchmark: 100 tokens
    print("\nGenerating 100 tokens...")
    start = time.time()
    response = generate(model, tokenizer, prompt=formatted, max_tokens=100, sampler=sampler, verbose=False)
    elapsed = time.time() - start

    generated_text = response[len(formatted):]
    num_tokens = len(tokenizer.encode(generated_text))
    tps = num_tokens / elapsed

    print(f"Tokens: {num_tokens}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Speed: {tps:.1f} tok/s")
    print(f"Text: {generated_text[:200]}...")

if __name__ == "__main__":
    main()
