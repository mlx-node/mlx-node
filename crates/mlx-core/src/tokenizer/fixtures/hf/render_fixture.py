#!/usr/bin/env python3
"""Render a checkpoint's own `chat_template.jinja` through HuggingFace transformers
and write the bytes out as a Rust test fixture.

These fixtures are GROUND TRUTH for `tokenizer.rs`'s Jinja renderer: the
definition of a correct prompt is what HF's own renderer produces from the same
template, not what ours happens to emit. Checked in next to the fixtures so the
expectation is reproducible rather than a mystery blob.

    uv run --with 'transformers==5.5.0' \
      crates/mlx-core/src/tokenizer/fixtures/hf/render_fixture.py \
      --cache ~/.cache/models --out crates/mlx-core/src/tokenizer/fixtures/hf

Generated against: transformers 5.5.0, Python 3.14, jinja2 3.1.x.
The `--check` flag re-renders and diffs instead of writing, which is how you
confirm a transformers upgrade did not move the goalposts.

## Why `_compile_jinja_template` rather than `apply_chat_template`

`apply_chat_template` needs a real tokenizer (and for these checkpoints, a
28 MB `tokenizer.json` plus a trust-remote-code dance). The Jinja environment is
the part under test, and `_compile_jinja_template` IS the environment
`apply_chat_template` renders in — same `ImmutableSandboxedEnvironment`, same
`trim_blocks=True, lstrip_blocks=True`, same `tojson` / `strftime_now` /
`raise_exception` globals. Calling it directly keeps the fixture about the
template and the filters.

## The context and message shapes are copied from the Rust side

Deliberately duplicated rather than inferred, because a fixture built from a
guess about our own inputs would prove nothing:

  * context keys — `tokenizer.rs` `render_chat_template_jinja2_with_content_order`
  * message dicts — `tokenizer.rs` `serialize_message_for_jinja_with_policy`
    (note `tool_calls[]` carries BOTH the flat `name`/`arguments` and the wrapped
    `function.{name,arguments}`, and a `tool` message carries the `name`
    resolved from the call it answers)
  * tool dicts — `tokenizer.rs`'s `tools_value` construction, which re-parses
    `FunctionParameters::properties` from its JSON string into a real object
  * the probe conversation itself — `separator_probe_tool` /
    `separator_probe_messages` in `tokenizer.rs`'s test module

If any of those move, this script has to move with them, and the Rust test that
consumes the fixtures is what will tell you.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Pinned so a template that prints today's date does not make the fixtures rot
# overnight. Must equal `PROBE_DATE` / `BOS_PROBE` / `EOS_PROBE` in tokenizer.rs.
PROBE_DATE = "2026-08-10"
BOS_PROBE = "<|begin_of_text|>"
EOS_PROBE = "<|endoftext|>"

# `separator_probe_tool()`. Two properties and two `required` entries so all three
# of `PythonDefaultFormatter`'s overrides are load-bearing: `, ` between keys,
# `: ` after a key, `, ` between array elements.
PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "wx.forecast",
            "description": "Get a forecast.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city", "days"],
            },
        },
    }
]

_ARGS = {"city": "Paris", "opts": {"deep": [1, 2]}}

# `separator_probe_messages()`, already through
# `serialize_message_for_jinja_with_policy`.
PROBE_MESSAGES = [
    {"role": "user", "content": "weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "name": "wx.forecast",
                "arguments": _ARGS,
                "function": {"name": "wx.forecast", "arguments": _ARGS},
            }
        ],
    },
    {
        "role": "tool",
        "content": "18C, clear",
        # Resolved by `serialize_messages_for_jinja` from the call's `id`.
        "name": "wx.forecast",
        "tool_call_id": "call_1",
    },
    {"role": "assistant", "content": "18C and clear."},
]

PROBE_CONTEXT = {
    "messages": PROBE_MESSAGES,
    "tools": PROBE_TOOLS,
    "add_generation_prompt": True,
    "enable_thinking": True,
    "preserve_thinking": True,
    "keep_past_thinking": True,
    "bos_token": BOS_PROBE,
    "eos_token": EOS_PROBE,
    # `build_render_context`: present only when `RenderContextOptions` set it.
    # `reasoning_strength` stays ABSENT, which is not the same as empty — see
    # `RenderContextOptions::reasoning_strength`.
    "current_date": PROBE_DATE,
}


def render(template: str) -> str:
    from transformers.utils import chat_template_utils as ctu

    compiled = ctu._compile_jinja_template(template)
    return compiled.render(**PROBE_CONTEXT)


def slug(name: str) -> str:
    """Family label for the fixture filename. Only cosmetic — the Rust side keys
    on the template's sha256, so a wrong guess here costs nothing but a rename."""
    n = name.lower()
    for family in (
        "muse-glimmer",
        "nemotron-3.5-lightning",
        "lfm2.5-1.2b-thinking",
        "lfm2.5-2.6b",
        "lfm2.5-8b-a1b",
        "ornith-1.0-9b",
        "ornith-1.0-35b",
        "agentworld",
        "agents-a1-4b",
        "agents-a1",
        "qwen3.5",
        "qwen3.6",
    ):
        if family in n:
            return family
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path, help="dir of checkpoint dirs")
    ap.add_argument("--out", required=True, type=Path, help="fixture output dir")
    ap.add_argument(
        "--check",
        action="store_true",
        help="re-render and report drift instead of writing",
    )
    args = ap.parse_args()

    # One fixture per DISTINCT template: the cache holds several quant variants
    # per family and they all ship the same bytes.
    by_hash: dict[str, tuple[str, str]] = {}
    for path in sorted(args.cache.glob("*/chat_template.jinja")):
        template = path.read_text()
        if "tojson" not in template:
            continue
        digest = hashlib.sha256(template.encode()).hexdigest()[:12]
        by_hash.setdefault(digest, (path.parent.name, template))

    args.out.mkdir(parents=True, exist_ok=True)
    drift, wrote, failed = [], [], []
    for digest, (name, template) in sorted(by_hash.items()):
        target = args.out / f"{slug(name)}-{digest}.txt"
        try:
            out = render(template)
        except Exception as e:  # noqa: BLE001 — a template that HF itself cannot
            # render with this context is a fact about the fixture set, not a
            # crash: record it and keep going so one bad template does not hide
            # the other nine.
            failed.append(f"{target.name}: {type(e).__name__}: {e}")
            continue
        if args.check:
            if not target.exists():
                drift.append(f"{target.name}: MISSING")
            elif target.read_text() != out:
                drift.append(f"{target.name}: {len(target.read_text())}B -> {len(out)}B")
            continue
        target.write_text(out)
        wrote.append(f"{target.name} ({len(out)}B)")

    for line in wrote:
        print(f"wrote {line}")
    for line in failed:
        print(f"UNRENDERABLE {line}", file=sys.stderr)
    for line in drift:
        print(f"DRIFT {line}", file=sys.stderr)
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
