"""Golden renders of the Hy4-flash chat template.

    uv run --no-project --with jinja2 python -I render.py \
        --template <chat_template.gguf.jinja> [--hf-template <hf/chat_template.jinja>] --out <dir>

The template files are read as text and rendered in jinja2's ImmutableSandboxedEnvironment
configured like transformers' `_compile_jinja_template` (trim_blocks, lstrip_blocks,
loopcontrols, the json.dumps `tojson`, `raise_exception`, `strftime_now`). No transformers import
and no downloaded Python is executed.

Writes <out>/<case>.json (the exact render kwargs) and <out>/<case>.txt (the render), or
<out>/<case>.err when the template raises. With --hf-template it also writes <out>/hf/<case>.txt
for every case the HF template accepts and <out>/hf/COMPARE.txt (HF render with the suffix
':opensource' swapped for ':6124c78e' vs the GGUF render).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from jinja2.exceptions import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

GGUF_SUFFIX = ":6124c78e"
HF_SUFFIX = ":opensource"


def make_env() -> ImmutableSandboxedEnvironment:
    def raise_exception(message):
        raise TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(fmt):
        return datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询城市天气 (weather lookup)",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"},
                "days": {"type": "integer"},
            },
            "required": ["city"],
        },
        "strict": True,
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
    },
}

DEFERRED_TOOL = {
    "type": "function",
    "function": {
        "name": "hidden_tool",
        "description": "Loaded later.",
        "parameters": {"type": "object", "properties": {}},
        "defer_loading": True,
    },
}


def call(call_id, name, arguments):
    entry = {"name": name, "arguments": arguments, "function": {"name": name, "arguments": arguments}}
    if call_id is not None:
        entry = {"id": call_id, **entry}
    return entry


def tool_result(call_id, name, content):
    msg = {"role": "tool", "content": content, "name": name}
    if call_id is not None:
        msg["tool_call_id"] = call_id
    return msg


U1 = {"role": "user", "content": "Hello, who are you?"}
SYS = {"role": "system", "content": "You are a concise assistant. 请用中文回答。"}
A1 = {
    "role": "assistant",
    "content": "I am Hy4, an assistant.",
    "reasoning_content": "The user asks who I am. Answer briefly.",
}
U2 = {"role": "user", "content": "What is 17 * 23?"}
A2 = {"role": "assistant", "content": "17 * 23 = 391.", "reasoning_content": "17*23 = 17*20 + 17*3 = 340 + 51 = 391."}

TOOL_TURNS = [
    {"role": "user", "content": "Weather in 北京 for 3 days, and search 'Hy4 release'."},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Two independent calls.",
        "tool_calls": [
            call("call_a", "get_weather", {"city": "北京", "days": 3}),
            call("call_b", "web_search", {"query": "Hy4 release", "top_k": 5, "filters": {"lang": ["en", "zh"]}}),
        ],
    },
    tool_result("call_b", "web_search", "Hy4 was released in 2026."),
    tool_result("call_a", "get_weather", '{"temp_c": [3, 5, 4], "sky": "晴"}'),
]

FINAL = {"role": "assistant", "content": "北京: 3-5°C, sunny. Hy4 shipped in 2026.", "reasoning_content": "Summarize both."}


def ctx(messages, add_generation_prompt, tools=None, **extra):
    out = {"messages": messages, "add_generation_prompt": add_generation_prompt}
    if tools is not None:
        out["tools"] = tools
    out.update(extra)
    return out


CASES = {
    "single_user_agp": ctx([U1], True),
    "single_user_no_agp": ctx([U1], False),
    "system_user_agp": ctx([SYS, U1], True),
    "content_parts_agp": ctx(
        [{"role": "user", "content": [{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}]}], True
    ),
    "multi_turn_reasoning_agp": ctx([SYS, U1, A1, U2], True),
    "multi_turn_reasoning_preserved_agp": ctx([SYS, U1, A1, U2], True, preserved_thinking=True),
    "multi_turn_last_assistant_no_agp": ctx([U1, A1, U2, A2], False),
    "multi_turn_last_assistant_agp": ctx([U1, A1, U2, A2], True),
    "reasoning_field_precedence": ctx(
        [U1, {"role": "assistant", "content": "Hi.", "reasoning": "from reasoning", "reasoning_content": "from reasoning_content"}],
        False,
    ),
    "effort_high_explicit_agp": ctx([U1], True, reasoning_effort="high"),
    "effort_max_agp": ctx([U1], True, reasoning_effort="max"),
    "effort_low_agp": ctx([U1], True, reasoning_effort="low"),
    "effort_no_think_agp": ctx([U1], True, reasoning_effort="no_think"),
    "effort_no_think_no_agp": ctx([U1], False, reasoning_effort="no_think"),
    "effort_no_think_history_agp": ctx([U1, A1, U2], True, reasoning_effort="no_think", preserved_thinking=True),
    "effort_invalid": ctx([U1], True, reasoning_effort="medium"),
    "raw_last_assistant_prefill": ctx([U1, {"role": "assistant", "content": "Sure, the answer"}], False, raw_last_assistant=True),
    "tools_only_agp": ctx([U1], True, tools=[WEATHER_TOOL, SEARCH_TOOL, DEFERRED_TOOL]),
    "tools_system_agp": ctx([SYS, U1], True, tools=[WEATHER_TOOL, SEARCH_TOOL]),
    "tools_call_result_agp": ctx(list(TOOL_TURNS), True, tools=[WEATHER_TOOL, SEARCH_TOOL]),
    "tools_call_result_final_no_agp": ctx(list(TOOL_TURNS) + [FINAL], False, tools=[WEATHER_TOOL, SEARCH_TOOL]),
    "tools_call_result_not_preserved_agp": ctx(
        list(TOOL_TURNS) + [FINAL, U2], True, tools=[WEATHER_TOOL, SEARCH_TOOL], preserved_thinking=False
    ),
    "tools_no_ids_agp": ctx(
        [
            TOOL_TURNS[0],
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call(None, "get_weather", {"city": "北京"}), call(None, "web_search", {"query": "Hy4"})],
            },
            tool_result(None, "web_search", "result B"),
            tool_result(None, "get_weather", "result A"),
        ],
        True,
        tools=[WEATHER_TOOL, SEARCH_TOOL],
    ),
}


def render(env, template_text, kwargs):
    try:
        return env.from_string(template_text).render(**kwargs), None
    except TemplateError as e:
        return None, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, type=Path)
    ap.add_argument("--hf-template", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    env = make_env()
    template_text = args.template.read_text(encoding="utf-8")
    hf_text = args.hf_template.read_text(encoding="utf-8") if args.hf_template else None
    args.out.mkdir(parents=True, exist_ok=True)
    if hf_text is not None:
        (args.out / "hf").mkdir(exist_ok=True)

    compare = []
    for name, kwargs in CASES.items():
        (args.out / f"{name}.json").write_text(json.dumps(kwargs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        text, err = render(env, template_text, kwargs)
        for stale in (args.out / f"{name}.txt", args.out / f"{name}.err"):
            stale.unlink(missing_ok=True)
        if err is not None:
            (args.out / f"{name}.err").write_text(err + "\n", encoding="utf-8")
            print(f"{name}: ERROR {err}")
        else:
            (args.out / f"{name}.txt").write_text(text, encoding="utf-8")
            print(f"{name}: {len(text.encode('utf-8'))} B")
        if hf_text is None:
            continue
        hf_out, hf_err = render(env, hf_text, kwargs)
        if hf_err is not None:
            compare.append(f"{name}\tHF-ERROR\t{hf_err}")
            continue
        (args.out / "hf" / f"{name}.txt").write_text(hf_out, encoding="utf-8")
        if err is not None:
            compare.append(f"{name}\tGGUF-ERROR-HF-OK")
            continue
        same = hf_out.replace(HF_SUFFIX, GGUF_SUFFIX) == text
        compare.append(f"{name}\t{'SAME' if same else 'DIFF'}")
    if hf_text is not None:
        (args.out / "hf" / "COMPARE.txt").write_text("\n".join(compare) + "\n", encoding="utf-8")
        print("\n".join(compare))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
