#!/usr/bin/env python3
# K2-Horizon coding-agent workload benchmark (mlx-vlm reference path).
# Same workload JSON as the node harness: persistent PromptCacheState gives
# delta prefill across turns — the reference equivalent of ChatSession reuse.

import json
import os
import sys
import time

sys.path.insert(0, "/Users/brooklyn/workspace/github/mlx-node/mlx-vlm")

import mlx.core as mx  # noqa: E402
from mlx_vlm import load  # noqa: E402
from mlx_vlm.generate.common import PromptCacheState  # noqa: E402
from mlx_vlm.generate.dispatch import stream_generate  # noqa: E402
from mlx_vlm.prompt_utils import apply_chat_template  # noqa: E402

MODEL = os.environ.get("K2_MXFP8", "/tmp/k2-out/k2-horizon-7b-mxfp8")
WORKLOAD = os.environ.get("K2_WORKLOAD", "/tmp/k2-workload.json")
MAX_TOKENS = int(os.environ.get("K2_MAX_TOKENS", "200"))

workload = json.load(open(WORKLOAD))

model, processor = load(MODEL)
print(json.dumps({"loaded": MODEL}), flush=True)

cache_state = PromptCacheState()


def msg(role, text):
    return {"role": role, "content": [{"type": "text", "text": text}]}


history = [msg("system", workload["system"])]

def asst_msg(text):
    # K2 template requires a thinking field on assistant history. Generation
    # continues inside the injected `<ifm|think_faster>` (effort=low); split the
    # close tag back into the structured field.
    close = "</ifm|think_faster>"
    if close in text:
        think, _, content = text.partition(close)
        return {"role": "assistant", "content": content, "think_faster": think}
    return {"role": "assistant", "content": "", "think_faster": text}


for i, turn in enumerate(workload["turns"]):
    history.append(msg("user", turn))
    prompt = apply_chat_template(
        processor,
        model.config,
        history,
        num_images=0,
        reasoning_effort="low",
    )
    t0 = time.perf_counter()
    text = ""
    stats = {}
    for chunk in stream_generate(
        model,
        processor,
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        prompt_cache_state=cache_state,
    ):
        text += chunk.text
        stats = {
            "prompt_tps": getattr(chunk, "prompt_tps", None),
            "generation_tps": getattr(chunk, "generation_tps", None),
            "prompt_tokens": getattr(chunk, "prompt_tokens", None),
            "generation_tokens": getattr(chunk, "generation_tokens", None),
            "peak_memory": getattr(chunk, "peak_memory", None),
        }
    wall_ms = (time.perf_counter() - t0) * 1000
    history.append(asst_msg(text))
    print(
        json.dumps(
            {
                "turn": i,
                "promptChars": len(turn),
                "numTokens": stats.get("generation_tokens"),
                "promptTokens": stats.get("prompt_tokens"),
                "prefillTps": stats.get("prompt_tps"),
                "decodeTps": stats.get("generation_tps"),
                "wallMs": round(wall_ms),
                "text": text[:50],
            }
        ),
        flush=True,
    )
