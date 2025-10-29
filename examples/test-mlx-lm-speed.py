#!/usr/bin/env python3
"""
Test MLX-LM Model Generation Speed

Benchmark script to test token generation speed with mlx-lm for comparison
with the Node.js implementation. Uses the same model and prompts.

Note: Both implementations use similar parameters (temperature, top_p, etc.).
      MLX-LM uses a sampler object while Node.js passes parameters directly.

Installation:
    pip install mlx-lm

Setup:
    # Convert HuggingFace model to MLX format (if not already done)
    python -m mlx_lm.convert \
        --hf-path Qwen/Qwen3-0.6B-Instruct \
        --mlx-path .cache/models/qwen3-0.6b-mlx-bf16 \
        --dtype float32

Usage:
    python examples/test-mlx-lm-speed.py

    # Or make it executable and run directly
    chmod +x examples/test-mlx-lm-speed.py
    ./examples/test-mlx-lm-speed.py
"""

import time
from pathlib import Path
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# Get the project root (parent of node/ directory where script runs)
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / ".cache" / "models" / "qwen3-0.6b-mlx-bf16"


def main():
    print("╔════════════════════════════════════════════════════════╗")
    print("║   Testing MLX-LM Generation Speed (Python)             ║")
    print("╚════════════════════════════════════════════════════════╝\n")

    print(f"Loading model from: {MODEL_PATH}")
    print("(This may take a moment...)\n")

    # Load model and tokenizer
    model, tokenizer = load(str(MODEL_PATH))

    print("✓ Model and tokenizer loaded\n")

    # Create sampler with generation parameters
    # Note: MLX-LM uses a sampler object instead of passing parameters directly
    sampler = make_sampler(
        temp=0.7,  # temperature
        top_p=0.9,
        min_p=0.0,  # minimum probability threshold
        min_tokens_to_keep=1,
        top_k=50,
    )

    # Test prompts (same as TypeScript example)
    prompts = [
        "Hello! How are you today?",
        "What is the capital of France?",
        "Write a haiku about coding:",
        "Explain what machine learning is in one sentence:",
    ]

    for prompt in prompts:
        print("─" * 60)
        print(f'Prompt: "{prompt}"')
        print("─" * 60)

        # Format as chat message
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate with timing
        start_time = time.time()
        response = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=2048,
            sampler=sampler,  # Pass sampler with temp, top_p, top_k
            verbose=False,  # Disable per-token printing
        )
        duration = (time.time() - start_time) * 1000  # Convert to ms

        # Count tokens in response
        # Remove the prompt from the response to count only generated tokens
        generated_text = response[len(formatted_prompt) :]
        num_tokens = len(tokenizer.encode(generated_text))

        # Calculate tokens/second
        tokens_per_second = (num_tokens / duration) * 1000

        print(
            f"\nGenerated ({num_tokens} tokens, {duration:.0f}ms, {tokens_per_second:.2f} tokens/s):"
        )
        print(generated_text)
        print("")

    print("╔════════════════════════════════════════════════════════╗")
    print("║   Test Complete                                        ║")
    print("╚════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n❌ Test failed!")
        print(f"Error: {e}")
        import traceback

        print("\nStack trace:")
        traceback.print_exc()
        exit(1)
