#!/usr/bin/env python3
# K2-Horizon coding-agent workload benchmark (mlx-vlm reference path).
# Same workload JSON as the node harness: persistent PromptCacheState gives
# delta prefill across turns — the reference equivalent of ChatSession reuse.

import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.environ.get("MLX_VLM_ROOT", str(Path(__file__).resolve().parents[1] / "mlx-vlm")))

import mlx.core as mx  # noqa: E402
from mlx_vlm import load  # noqa: E402
from mlx_vlm.generate.common import PromptCacheState  # noqa: E402
from mlx_vlm.generate.dispatch import stream_generate  # noqa: E402
from mlx_vlm.models.k2_horizon.config import ModelConfig  # noqa: E402
from mlx_vlm.utils import prepare_inputs  # noqa: E402

MODEL = os.environ["K2_MXFP8"]
WORKLOAD = os.environ["K2_WORKLOAD"]
REFERENCE = os.environ["K2_REFERENCE_INPUTS"]
MAX_TOKENS = int(os.environ.get("K2_MAX_TOKENS", "200"))
REPETITIONS = int(os.environ.get("K2_REPETITIONS", "1"))
if not 1 <= MAX_TOKENS <= 200 or REPETITIONS < 1:
    raise ValueError("Require K2_MAX_TOKENS in 1..200 and positive K2_REPETITIONS")

checkpoint_config = json.loads((Path(MODEL) / "config.json").read_text())
reference_config = ModelConfig.from_dict(checkpoint_config)
if getattr(reference_config, "layernorm_num_groups", 1) != checkpoint_config.get("layernorm_num_groups", 1):
    raise ValueError("The reference drops K2 grouped RMSNorm configuration; an inference comparison would be invalid")

workload_bytes = Path(WORKLOAD).read_bytes()
workload = json.loads(workload_bytes)
frozen = [json.loads(line) for line in Path(REFERENCE).read_text().splitlines() if line]
source = frozen[0]
metadata = {
    "kind": "metadata",
    "runtime": "mlx-vlm",
    "pid": os.getpid(),
    "model": MODEL,
    "workloadSha256": hashlib.sha256(workload_bytes).hexdigest(),
    "configSha256": hashlib.sha256((Path(MODEL) / "config.json").read_bytes()).hexdigest(),
    "tokenizerSha256": hashlib.sha256((Path(MODEL) / "tokenizer.json").read_bytes()).hexdigest(),
    "maxTokens": MAX_TOKENS,
    "repetitions": REPETITIONS,
    "effort": "low",
    "temperature": 0,
    "inputMode": "frozen rendered prompts, prepared without an extra BOS",
}
for key in ("workloadSha256", "configSha256", "tokenizerSha256", "maxTokens", "effort", "temperature"):
    if metadata[key] != source[key]:
        raise ValueError(f"Frozen capture differs in {key}")
cases = {r["turn"]: r for r in frozen if r.get("kind") == "turn" and r["repetition"] == 0}
if set(cases) != set(range(len(workload["turns"]))):
    raise ValueError("Frozen capture must contain every workload turn exactly once in repetition zero")

capture = None
if os.environ.get("K2_CAPTURE"):
    descriptor = os.open(os.environ["K2_CAPTURE"], os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    capture = os.fdopen(descriptor, "w")


def record(value):
    if capture is not None:
        capture.write(json.dumps(value) + "\n")
        capture.flush()


try:
    print(json.dumps(metadata), flush=True)
    record(metadata)
    model, processor = load(MODEL)
    print(json.dumps({"loaded": MODEL}), flush=True)
    prepared = {}
    for index, case in cases.items():
        inputs = prepare_inputs(processor, prompts=case["rendered"], add_special_tokens=False)
        ids = inputs["input_ids"].reshape(-1).tolist()
        if ids != case["promptTokenIds"]:
            raise ValueError(f"Exact prepared prompt IDs differ at turn {index}")
        prepared[index] = inputs

    for repetition in range(REPETITIONS):
        cache_state = PromptCacheState()
        for index, turn in enumerate(workload["turns"]):
            case = cases[index]
            inputs = prepared[index]
            t0 = time.perf_counter()
            text = ""
            last = None
            for chunk in stream_generate(
                model,
                processor,
                case["rendered"],
                input_ids=inputs["input_ids"],
                mask=inputs["attention_mask"],
                max_tokens=MAX_TOKENS,
                temperature=0.0,
                skip_special_tokens=True,
                prompt_cache_state=cache_state,
            ):
                text += chunk.text
                last = chunk
            wall_ms = (time.perf_counter() - t0) * 1000
            if last is None:
                raise RuntimeError("Reference generation produced no terminal result")
            num_tokens = last.generation_tokens
            cached_tokens = last.cached_tokens
            ttft_seconds = last.prompt_tokens / last.prompt_tps
            output_matches = text == case["result"]["rawText"] and num_tokens == case["result"]["numTokens"]
            result = {
                "turn": index,
                "repetition": repetition,
                "phase": "first-pass" if repetition == 0 else "steady-state",
                "promptChars": len(turn),
                "renderedPromptTokens": len(case["promptTokenIds"]),
                "cachedTokens": cached_tokens,
                "numTokens": num_tokens,
                "finishReason": last.finish_reason,
                "ttftMs": ttft_seconds * 1000,
                "prefillTps": (last.prompt_tokens - cached_tokens) / ttft_seconds,
                "decodeTps": last.generation_tps * (num_tokens - 1) / num_tokens if num_tokens > 1 else 0,
                "wallMs": round(wall_ms),
                "inputIdsMatch": True,
                "outputMatchesNative": output_matches,
                "text": text[:50],
            }
            record({"kind": "turn", **result, "rawText": text, "tokenIds": last.token_ids})
            print(json.dumps(result), flush=True)
        mx.synchronize()
finally:
    if capture is not None:
        capture.close()
