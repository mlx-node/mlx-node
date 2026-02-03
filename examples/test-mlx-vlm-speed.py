#!/usr/bin/env python3
"""
Test MLX-VLM Model Generation Speed

Benchmark script to test PaddleOCR-VL generation speed with mlx-vlm for comparison
with the Node.js implementation. Tests both text-only and image+text modes.

Installation:
    pip install mlx-vlm

Setup:
    # Convert model first: oxnode scripts/convert-model.ts --model-type paddleocr-vl
    # Or use HuggingFace model: --model PaddlePaddle/PaddleOCR-VL-1.5

Usage:
    # Run with default model path
    python examples/test-mlx-vlm-speed.py

    # Run with custom model path
    python examples/test-mlx-vlm-speed.py --model /path/to/model

    # Run with custom image
    python examples/test-mlx-vlm-speed.py --image /path/to/image.png
"""

import argparse
import time
from pathlib import Path

from mlx_vlm import load, generate


# Defaults
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = str(PROJECT_ROOT / ".cache" / "models" / "PaddleOCR-VL-1.5-mlx")
DEFAULT_IMAGE = str(PROJECT_ROOT / "examples" / "ocr.png")

TEXT_PROMPTS = [
    "What is the capital of France?",
    "Write a detailed paragraph about the history of machine learning, covering key milestones and breakthroughs.",
    "Explain the difference between supervised and unsupervised learning.",
]

IMAGE_PROMPT = "Read all the text in this image."


def format_paddleocr_prompt(text, has_image=False):
    """Format prompt using PaddleOCR-VL's chat template.

    PaddleOCR-VL expects:
      <|begin_of_sentence|>User: [<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>]text
      Assistant:
    """
    prompt = "<|begin_of_sentence|>User: "
    if has_image:
        prompt += "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>"
    prompt += text + "\nAssistant:\n"
    return prompt

NUM_WARMUP = 1
NUM_RUNS = 5


def benchmark_text_only(model, processor, prompts, max_tokens=128):
    """Benchmark text-only generation (no images)."""
    print("\n" + "=" * 60)
    print("  TEXT-ONLY GENERATION")
    print("=" * 60)

    all_gen_tps = []

    for raw_prompt in prompts:
        print(f'\nPrompt: "{raw_prompt[:80]}{"..." if len(raw_prompt) > 80 else ""}"')
        prompt = format_paddleocr_prompt(raw_prompt)

        # Warmup
        for _ in range(NUM_WARMUP):
            generate(model, processor, prompt, max_tokens=max_tokens,
                     temperature=0.0, verbose=False)

        # Benchmark runs
        run_tps = []
        for i in range(NUM_RUNS):
            result = generate(model, processor, prompt, max_tokens=max_tokens,
                              temperature=0.0, verbose=False)

            gen_tokens = result.generation_tokens
            gen_tps = result.generation_tps
            prompt_tps = result.prompt_tps
            run_tps.append(gen_tps)

            if i == 0:
                text = result.text[:120]
                print(f"  Output: {text}{'...' if len(result.text) > 120 else ''}")

            print(f"  Run {i+1}: {gen_tokens} tokens, "
                  f"prefill={prompt_tps:.1f} tok/s, "
                  f"decode={gen_tps:.1f} tok/s")

        avg_tps = sum(run_tps) / len(run_tps)
        all_gen_tps.extend(run_tps)
        print(f"  Average decode: {avg_tps:.1f} tok/s")

    overall_avg = sum(all_gen_tps) / len(all_gen_tps) if all_gen_tps else 0
    print(f"\n  Overall average decode: {overall_avg:.1f} tok/s")
    return overall_avg


def benchmark_image(model, processor, image_path, max_tokens=256):
    """Benchmark image+text generation."""
    print("\n" + "=" * 60)
    print("  IMAGE + TEXT GENERATION")
    print("=" * 60)

    if not Path(image_path).exists():
        print(f"\n  Image not found: {image_path}")
        print("  Skipping image benchmark.")
        return 0.0

    prompt = format_paddleocr_prompt(IMAGE_PROMPT, has_image=True)
    print(f'\nImage: {image_path}')
    print(f'Prompt: "{IMAGE_PROMPT}"')

    # Warmup
    for _ in range(NUM_WARMUP):
        generate(model, processor, prompt, [image_path],
                 max_tokens=max_tokens, temperature=0.0, verbose=False)

    # Benchmark runs
    run_tps = []
    run_total = []
    for i in range(NUM_RUNS):
        t0 = time.perf_counter()
        result = generate(model, processor, prompt, [image_path],
                          max_tokens=max_tokens, temperature=0.0, verbose=False)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        gen_tokens = result.generation_tokens
        gen_tps = result.generation_tps
        prompt_tokens = result.prompt_tokens
        prompt_tps = result.prompt_tps
        total_tps = gen_tokens / elapsed

        run_tps.append(gen_tps)
        run_total.append(total_tps)

        if i == 0:
            text = result.text[:150]
            print(f"  Output: {text}{'...' if len(result.text) > 150 else ''}")

        print(f"  Run {i+1}: {gen_tokens} gen tokens (prefill {prompt_tokens} tokens), "
              f"prefill={prompt_tps:.1f} tok/s, "
              f"decode={gen_tps:.1f} tok/s, "
              f"total={total_tps:.1f} tok/s ({elapsed:.3f}s)")

    avg_decode = sum(run_tps) / len(run_tps) if run_tps else 0
    avg_total = sum(run_total) / len(run_total) if run_total else 0
    print(f"\n  Average decode: {avg_decode:.1f} tok/s")
    print(f"  Average total (including vision+prefill): {avg_total:.1f} tok/s")
    return avg_decode


def main():
    parser = argparse.ArgumentParser(description="Benchmark mlx-vlm generation speed")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model path (default: {DEFAULT_MODEL})")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE,
                        help=f"Image path for OCR benchmark (default: {DEFAULT_IMAGE})")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max tokens for text generation (default: 128)")
    parser.add_argument("--max-image-tokens", type=int, default=256,
                        help="Max tokens for image generation (default: 256)")
    args = parser.parse_args()

    print("+" + "-" * 58 + "+")
    print("|   MLX-VLM Speed Benchmark (Python)                       |")
    print("+" + "-" * 58 + "+")
    print(f"\nModel: {args.model}")

    print("\nLoading model...")
    t0 = time.perf_counter()
    model, processor = load(args.model)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Text-only benchmark
    text_tps = benchmark_text_only(model, processor, TEXT_PROMPTS,
                                   max_tokens=args.max_tokens)

    # Image benchmark
    image_tps = benchmark_image(model, processor, args.image,
                                max_tokens=args.max_image_tokens)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Text-only decode:   {text_tps:.1f} tok/s")
    if image_tps > 0:
        print(f"  Image decode:       {image_tps:.1f} tok/s")
    print(f"  Model load time:    {load_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
