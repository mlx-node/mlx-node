# vLLM Reasoning / Thinking Control Flow Spec for `mlx-node`

This document is a **feature-parity spec** for implementing vLLM-style reasoning / thinking control flow in `mlx-node`, with special attention to the **Qwen3 / Qwen3.5** family because that is where the serving-time behavior is the most nuanced. It covers the request path from **OpenAI-compatible ingress** through **chat template rendering**, **reasoning parser selection**, **streaming and non-streaming response assembly**, **tool-calling interaction**, and **thinking-token budget enforcement**. It does **not** attempt to spec unrelated internals such as scheduler fairness, KV-cache allocation, or kernel-level execution. The current vLLM docs also note a naming transition: the structured field is now called **`reasoning`**, while older integrations may still refer to **`reasoning_content`**. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/features/reasoning_outputs/))

---

## 1. Scope and target behavior

`mlx-node` should aim to replicate the behavior of vLLM’s **reasoning-aware online serving path**, especially the semantics exposed through the **chat completions** API. In vLLM’s public docs, reasoning outputs are documented primarily on the **online chat completion endpoint**: the non-streaming response carries `message.reasoning`, and streaming responses carry `delta.reasoning`. Tool parsing is explicitly documented to operate on the **final content channel**, not on the reasoning channel. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/features/reasoning_outputs/))

For Qwen3/Qwen3.5, reasoning control is primarily implemented through three cooperating layers:

1. **Chat template kwargs**, especially `enable_thinking`.
2. **A model-specific reasoning parser**, selected with `--reasoning-parser qwen3`.
3. **An optional thinking budget**, enforced by `thinking_token_budget` plus `--reasoning-config`. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/protocol/))

The core design principle is:

> **Prompt rendering decides what mode the model enters; parsing decides how generated text is split into `reasoning` vs `content`; budget enforcement decides when a reasoning block must be terminated.**

---

## 2. Public API surface that matters

A vLLM-compatible `mlx-node` implementation should expose the following request-level controls on the chat-completions path:

- `chat_template_kwargs`
- `include_reasoning`
- `thinking_token_budget`
- `reasoning_effort`
  plus normal generation parameters such as max tokens, temperature, top-p, stop sequences, and tool-calling options. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/protocol/))

### 2.1 `chat_template_kwargs`

This is the main hook for model-family-specific behavior. For Qwen3/Qwen3.5, the important key is:

```json
{
  "enable_thinking": false
}
```

In vLLM serving, this can be provided per request via `extra_body.chat_template_kwargs`, or globally at server startup with `--default-chat-template-kwargs '{"enable_thinking": false}'`. Request-level kwargs override server defaults. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/protocol/))

### 2.2 `include_reasoning`

This controls whether structured reasoning is emitted back to the client. In vLLM’s serving flow, when `include_reasoning` is false, the serving path initializes `reasoning_ended = True` and later suppresses reasoning in the assembled response. Separately, the request validator maps `reasoning_effort == "none"` to `include_reasoning = False`. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/serving/))

### 2.3 `thinking_token_budget`

This is a **hard serving-time control** for reasoning length. vLLM passes it through `SamplingParams`, and the request path rejects it unless `reasoning_config` is configured. The public docs state that once the parser sees the model has entered the thinking section, reaching the configured budget causes vLLM to force the model to emit the configured `think_end_str`. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/))

### 2.4 `reasoning_effort`

In current vLLM protocol code, `reasoning_effort` is fed into chat rendering parameters, and `reasoning_effort == "none"` disables structured reasoning output. It is therefore best treated as a **template/rendering concern** plus a **response policy flag**, not as a substitute for `thinking_token_budget`. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/protocol/))

---

## 3. Server bootstrap control flow

At startup, the vLLM server wires together four distinct configuration surfaces:

1. **Reasoning parser selection** via `--reasoning-parser`.
2. **Server-wide template defaults** via `--default-chat-template-kwargs`.
3. **Reasoning boundary strings** via `--reasoning-config`.
4. **Optional tool parser / tool-calling configuration**. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/cli_args/))

### 3.1 Reasoning parser registry

vLLM keeps a central `ReasoningParserManager` registry that can register parsers eagerly or lazily and resolve them by name at runtime. If the configured parser name is unknown, the manager throws a `KeyError`. This means `mlx-node` should also separate **parser registration** from **parser instantiation**, rather than hard-coding Qwen behavior directly into the serving loop. ([docs.vllm.ai](https://docs.vllm.ai/en/stable/api/vllm/reasoning/basic_parsers/))

### 3.2 Parser selection for Qwen

For Qwen3/Qwen3.5, vLLM uses the `qwen3` reasoning parser. Qwen’s own deployment docs note that compatibility between `enable_thinking=False` and proper reasoning parsing was fixed in **vLLM 0.9.0**; earlier **0.8.5-era** guidance warned that disabling thinking and parsing reasoning content were not compatible. For parity, `mlx-node` should target the newer behavior, not the old limitation. ([qwen.readthedocs.io](https://qwen.readthedocs.io/en/latest/deployment/vllm.html))

### 3.3 Reasoning config bootstrap

`ReasoningConfig` carries at least:

- `think_start_str`
- `think_end_str`

At initialization time, vLLM tokenizes these strings with the model tokenizer and stores their token-ID sequences. If tokenization fails, initialization raises an error. This is important: the budget mechanism is **not** based on plain string matching at serving time; it is based on **token-sequence-aware state tracking**. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/config/reasoning/))

---

## 4. End-to-end request lifecycle

The full reasoning-aware request path in vLLM is:

1. Request arrives at the **OpenAI-compatible chat completion endpoint**.
2. The server validates model and basic request shape.
3. Chat messages are rendered into a prompt using the selected chat template plus merged template kwargs.
4. If a reasoning parser is configured, the server instantiates it with the **same template kwargs** used for rendering.
5. The server computes sampling params, including `thinking_token_budget`.
6. The server determines whether the **prompt already ends the reasoning section**.
7. The request is handed to the engine.
8. Generated text is split into `reasoning` and `content` in streaming or non-streaming mode.
9. Tool parsing, if enabled, only sees **content**, never reasoning.
10. The final response is assembled with `message.reasoning` and `message.content`, or with streaming deltas. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/serving/))

---

## 5. Prompt rendering control flow

### 5.1 Entry point

The chat completion path calls `render_chat_request`, which validates the target model and delegates to the rendering layer. vLLM’s serving renderer then:

- preprocesses the request,
- merges default chat-template kwargs,
- adds tool information if needed,
- builds tokenization params,
- builds chat params,
- and asynchronously renders the final prompt. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/serving/))

### 5.2 Merge semantics

The server merges template kwargs with this precedence:

```text
effective_chat_template_kwargs = server_defaults overridden by request_kwargs
```

In code terms, vLLM’s helper uses `default | request`, meaning **request-level keys win**. `mlx-node` should match this exactly, since most users expect `extra_body.chat_template_kwargs` to override server defaults on a per-request basis. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/engine/serving/))

### 5.3 Why parser instantiation must use the same kwargs

vLLM instantiates the reasoning parser with the same merged `chat_template_kwargs` used for rendering. This is not incidental. For Qwen3/Qwen3.5, the parser needs to know whether the prompt was rendered with `enable_thinking=False`, because that changes how the prompt is seeded and therefore how the first generated tokens must be interpreted. `mlx-node` should treat this as a **hard invariant**:

> **The parser configuration must be derived from the exact same effective template kwargs that were used to render the prompt.** ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/serving/))

---

## 6. Qwen3 / Qwen3.5 template semantics

This is the most important model-family-specific behavior for parity.

### 6.1 When thinking is enabled

Qwen3.5’s official chat template emits a generation prefix that begins with:

```text
<think>\n
```

That means the model is being prompted to start generating inside the reasoning section. ([huggingface.co](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/chat_template.jinja))

### 6.2 When thinking is disabled

When `enable_thinking` is false, Qwen3.5’s template emits:

```text
<think>\n\n</think>\n\n
```

before the assistant response content. This means the prompt already contains a **closed reasoning block**. The consequence is subtle but crucial: in generation, the model may not emit `<think>` at all, and the first generated tokens should generally be interpreted as **final content**, not reasoning. vLLM’s Qwen parser has explicit logic for this case. ([huggingface.co](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/chat_template.jinja))

### 6.3 Why Qwen3.5 differs from older Qwen3 behavior

vLLM’s Qwen parser docs state that **starting with Qwen3.5**, the chat template puts `<think>` into the prompt, so generation commonly only encounters `</think>`. The parser still strips an emitted `<think>` if it appears, for backward compatibility with older pre-2507 templates. `mlx-node` should therefore support both patterns:

- **new style**: prompt already contains `<think>`, generation may only produce `</think>`;
- **old style**: generation may still emit `<think>` explicitly. ([docs.vllm.ai](https://docs.vllm.ai/en/stable/api/vllm/reasoning/qwen3_reasoning_parser/))

### 6.4 History serialization rule

Qwen’s model card recommends **not feeding prior thinking content back into conversation history**. The provided chat template implements this best practice by omitting older reasoning blocks from prior assistant turns, while still allowing reasoning serialization in the most recent turn context where needed. `mlx-node` should replicate that selective history behavior rather than blindly replaying all prior reasoning text. ([huggingface.co](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/README.md?utm_source=chatgpt.com))

---

## 7. Reasoning parser architecture

### 7.1 Base parser contract

vLLM’s `ReasoningParser` is an abstract class initialized with a tokenizer and required to implement, among other things, `is_reasoning_end(input_ids)`. `BaseThinkingReasoningParser` extends this with token validation for the configured think-start and think-end tokens and stores their token IDs. ([docs.vllm.ai](https://docs.vllm.ai/en/stable/api/vllm/reasoning/basic_parsers/))

### 7.2 `is_reasoning_end(input_ids)`

In the base implementation, `is_reasoning_end` scans backward through input token IDs. If it sees an end token before a start token, it returns true; if it sees a start token first, it returns false. In practice, this asks:

> “At the end of the current prompt or prefix, are we logically **outside** the reasoning section?” ([docs.vllm.ai](https://docs.vllm.ai/en/stable/api/vllm/reasoning/))

This function is used in two places:

1. Before generation begins, to determine whether the **prompt already ended reasoning**.
2. During tool-calling / streaming logic, to decide when it is safe to interpret new tokens as content or tool calls. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/serving/))

### 7.3 `extract_content_ids`

The base parser also exposes `extract_content_ids`, which returns token IDs after the first detected reasoning-end token. This is especially important in the **streaming + tool-calling** path, where tool parsing must not see reasoning tokens. `mlx-node` should expose an equivalent token-level utility. ([docs.vllm.ai](https://docs.vllm.ai/en/stable/api/vllm/reasoning/))

---

## 8. Qwen-specific parser behavior

### 8.1 Constructor behavior

The Qwen parser reads `chat_template_kwargs` from the request object at construction time and stores:

```python
self.thinking_enabled = chat_kwargs.get("enable_thinking", True)
```

This is the parser’s switch for all later special-casing. ([docs.vllm.ai](https://docs.vllm.ai/en/stable/api/vllm/reasoning/qwen3_reasoning_parser/))

### 8.2 Non-streaming extraction logic

In non-streaming mode, the Qwen parser behaves as follows:

1. If output starts with `<think>`, strip it.
2. If no `</think>` is found:
   - if `thinking_enabled == False`, treat the entire output as **content**;
   - if `thinking_enabled == True`, treat the entire output as **reasoning** because the answer is truncated before reasoning ended.
3. If `</think>` is found, split into:
   - reasoning = text before the end token,
   - content = text after the end token. ([docs.vllm.ai](https://docs.vllm.ai/en/stable/api/vllm/reasoning/qwen3_reasoning_parser/))

This distinction is essential. In `mlx-node`, the absence of `</think>` must not always mean “all reasoning”; it depends on whether the template told the model to think in the first place. ([docs.vllm.ai](https://docs.vllm.ai/en/stable/api/vllm/reasoning/qwen3_reasoning_parser/))

### 8.3 Streaming extraction logic

In streaming mode, the Qwen parser:

1. strips `<think>` if present in the current delta;
2. checks whether `</think>` occurs in the current delta;
3. if yes, emits a delta split between reasoning and content;
4. if the end token already occurred in previous token IDs, treats subsequent deltas as content;
5. otherwise treats the delta as reasoning. ([docs.vllm.ai](https://docs.vllm.ai/en/stable/api/vllm/reasoning/qwen3_reasoning_parser/))

This means `mlx-node` streaming must preserve **cross-chunk parser state**, not just inspect one delta at a time. The parser decision depends on:

- previous token IDs,
- the current delta text,
- and whether the prompt was already outside reasoning. ([docs.vllm.ai](https://docs.vllm.ai/en/stable/api/vllm/reasoning/qwen3_reasoning_parser/))

---

## 9. Request preparation before generation

After rendering, vLLM computes generation parameters. On the chat completion path, `create_chat_completion` does the following:

1. get tokenizer,
2. merge `chat_template_kwargs`,
3. instantiate the reasoning parser if configured,
4. compute sampling params or beam params,
5. determine initial `reasoning_ended`,
6. call the engine. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/serving/))

### 9.1 Initial `reasoning_ended`

The initialization logic is:

- if `include_reasoning == False`, set `reasoning_ended = True`;
- else if a reasoning parser exists, compute `reasoning_ended = parser.is_reasoning_end(prompt_token_ids)`;
- else leave it unset / none. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/serving/))

This is a key design choice: the engine and stream assembler are told **up front** whether generation starts inside or outside a reasoning block.

### 9.2 Beam-search caveat

In the current serving code path, ordinary sampling goes through `SamplingParams` and passes reasoning-aware state into `engine_client.generate`. Beam search uses a separate path with `BeamSearchParams`. Since `thinking_token_budget` is part of `SamplingParams`, parity work in `mlx-node` should treat the budget controller as part of the **sampling/generation path**, not as a beam-search primitive. That is an implementation inference from the public code paths and API surfaces. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/serving/))

---

## 10. Streaming response control flow

This is where most of the subtle logic lives.

### 10.1 State initialized per streamed choice

The streaming generator maintains arrays such as:

- `previous_texts`
- `all_previous_token_ids`
- `added_content_delta_arr`
- `reasoning_end_arr`
- `prompt_is_reasoning_end_arr` ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/serving/))

`mlx-node` should maintain equivalent per-choice state objects rather than one global parser state.

### 10.2 One-time prompt inspection

On the first chunk for each choice, if prompt token IDs are available and a parser exists, vLLM computes:

```text
prompt_is_reasoning_end = parser.is_reasoning_end(prompt_token_ids)
```

This is especially important for Qwen with `enable_thinking=False`, because the prompt may already contain a closed think block. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/serving/))

### 10.3 Fast path when prompt already ended reasoning

If `prompt_is_reasoning_end` is true, vLLM may bypass the reasoning parser and route `delta_text` directly to `DeltaMessage(content=...)`. The Qwen parser docs explicitly mention this path for the case where the prompt includes `<think>\n\n</think>\n\n`. This is the mechanism that makes Qwen3.5 “no-thinking mode” work correctly under streaming. ([docs.vllm.ai](https://docs.vllm.ai/en/stable/api/vllm/reasoning/qwen3_reasoning_parser/))

### 10.4 Normal reasoning-to-content transition

If the prompt starts inside reasoning, the stream path repeatedly calls `extract_reasoning_streaming(...)`. Once the parser detects the reasoning end token:

- the current delta may be split into reasoning and content,
- `reasoning_end_arr[i]` flips true,
- future deltas are content-only. ([docs.vllm.ai](https://docs.vllm.ai/en/stable/api/vllm/reasoning/qwen3_reasoning_parser/))

### 10.5 Streaming response schema

vLLM’s public docs state that reasoning deltas are emitted as `delta.reasoning`, while answer text is emitted as normal `delta.content`. `mlx-node` should match that schema if it is targeting vLLM/OpenAI-compatible clients expecting the current field naming. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/features/reasoning_outputs/))

---

## 11. Non-streaming response control flow

The non-streaming path is simpler.

After generation completes, vLLM:

1. runs `parser.extract_reasoning(output.text, request=request)`,
2. optionally nulls reasoning if `include_reasoning == False`,
3. runs tool parsing on **content only**,
4. builds the final `ChatMessage(role, reasoning, content, tool_calls)`. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/serving/))

This means `mlx-node` non-streaming should always do post-generation splitting in this order:

```text
raw model text
→ reasoning/content split
→ reasoning suppression if requested
→ tool parsing from content only
→ final response assembly
```

That ordering matters. Tool parsing must not inspect text that was classified as reasoning. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/serving/))

---

## 12. Tool-calling interaction

This is a common source of bugs in parity implementations.

### 12.1 Core rule

vLLM’s docs explicitly state:

> tool calling only parses functions from the **content** field, not the reasoning field. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/features/reasoning_outputs/))

That rule holds in both streaming and non-streaming paths.

### 12.2 Auto-tool mode

In the streaming path with auto tool choice and reasoning enabled, vLLM waits until reasoning has ended before feeding token IDs into the tool parser. When the transition happens, it uses `extract_content_ids(...)` so the tool parser only sees the content portion. `mlx-node` should implement the same delayed handoff. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/serving/))

### 12.3 Named tool choice / required tool choice

vLLM also special-cases named-tool and required-tool flows to avoid spuriously classifying the first streamed chunk as reasoning when the prompt already ended the reasoning block. This is another reason `prompt_is_reasoning_end` must be tracked per choice. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/serving/))

### 12.4 Validation boundary

The rendering/preprocessing layer validates tool-related constraints before generation, and later stages only run tool extraction if tool parsing is configured and `tool_choice != "none"`. `mlx-node` should keep this separation: **validate early, parse late**. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/serve/render/serving/?utm_source=chatgpt.com))

---

## 13. Thinking-token budget control flow

This is the serving-time mechanism that stops overthinking.

### 13.1 Public behavior

The vLLM docs describe the budget like this: if `thinking_token_budget` is set, once generation enters the thinking section and the budget is reached, vLLM forces the model to produce `think_end_str`. The feature requires:

- `--reasoning-parser`
- `--reasoning-config`
- per-request `thinking_token_budget` ([docs.vllm.ai](https://docs.vllm.ai/en/latest/features/reasoning_outputs/))

### 13.2 Validation

The input processor rejects a request that sets `thinking_token_budget` when `reasoning_config` is missing. `mlx-node` should enforce the same validation and fail early. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/v1/engine/input_processor/))

### 13.3 Internal state machine

vLLM’s built-in `ThinkingTokenBudgetLogitsProcessor` keeps per-request state including:

- whether generation is currently inside thinking (`in_think`),
- whether it is currently forcing the end token sequence (`in_end`),
- current think token count (`think_count`),
- current progress within the end-token sequence (`end_count`),
- prompt token IDs,
- output token IDs,
- previous output length,
- countdown fields for cheaper re-checks. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/v1/sample/logits_processor/))

### 13.4 Initialization from prompt

When a request is added, the processor scans the prompt for the **last** think-start and think-end token sequences. If the last start comes after the last end, the request begins in thinking mode. For Qwen3.5 this matters because the prompt may already contain `<think>` or even a closed `<think>...</think>` block depending on `enable_thinking`. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/v1/sample/logits_processor/))

### 13.5 Budget counting

As generation progresses, the processor scans newly appended output tokens for start/end marker sequences. While `in_think` is true, it increments `think_count`. If `think_count >= thinking_token_budget`, the state flips into `in_end = True`, which means the processor will start forcing the end-token sequence. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/v1/sample/logits_processor/))

### 13.6 Forced end-token emission

When `in_end` is true, the logits processor overrides the next token choice by assigning a huge logit to the appropriate next token in `think_end_token_ids`. It continues until the full end-token sequence has been emitted, then clears `in_end`. This is not advisory prompting; it is an explicit decoding-time intervention. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/v1/sample/logits_processor/))

### 13.7 Implication for `mlx-node`

For parity, `mlx-node` should implement the thinking budget as a **token-level decoding controller**, not as a post-hoc truncation step. Post-hoc truncation would not match vLLM’s behavior in streaming, because clients would already have seen the extra reasoning tokens. The budget controller must act **during generation**. That conclusion follows directly from the logits-processor design in vLLM. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/api/vllm/v1/sample/logits_processor/))

---

## 14. `mlx-node` implementation architecture

### 14.1 Existing infrastructure (already implemented)

The following components are already in place and will be reused:

**Template controller** — `Qwen3Tokenizer::apply_chat_template_sync()` (`crates/mlx-core/src/tokenizer.rs`, line 1078) renders the Jinja2 chat template with `enable_thinking` in the template context. When `enable_thinking=true` (default), the template emits `<think>\n` as the generation prefix. When `false`, it emits `<think>\n\n</think>\n\n`. The `reasoning_content` field on `ChatMessage` is serialized into the template context for multi-turn history.

**Think-end token detection** — `Qwen3Tokenizer::detect_think_end()` (`tokenizer.rs`, line 1128) scans the tokenizer vocabulary at load time for `</think>` or `</longcat_think>` and stores the token ID as `think_end_id` and the string as `think_end_str`. These are available via accessor methods.

**Post-generation parsing** — `tools::split_at_think_end()` (`crates/mlx-core/src/tools/mod.rs`, line 564) splits generated text at the `</think>` boundary using both token-level and text-level detection. It already enforces tool parsing isolation: tools are parsed only from the content portion after `</think>`, never from the thinking portion. The helper `tools::has_think_end_token()` (line 554) checks whether the `think_end_id` token was generated.

**Chat finalization** — `chat_common::finalize_chat_result()` (`crates/mlx-core/src/models/qwen3_5/chat_common.rs`, line 127) decodes tokens, calls `split_at_think_end`, and builds the `ChatResult` with `thinking: Option<String>`.

### 14.2 New component: `ReasoningTracker`

**Location:** `crates/mlx-core/src/models/qwen3_5/chat_common.rs`

Unlike vLLM's `ReasoningParserManager` registry + `BaseThinkingReasoningParser` class hierarchy, `mlx-node` uses a lightweight `ReasoningTracker` struct. This is justified because:

1. `mlx-node` currently only targets Qwen3/Qwen3.5 models, not a multi-model serving fleet.
2. The Qwen `</think>` token is a single token (not a multi-token sequence), making token-level tracking trivial.
3. The existing `split_at_think_end()` already handles all the text-level parsing for non-streaming finalization.

The `ReasoningTracker` operates at the **token level** during the decode loop:

```rust
pub(crate) struct ReasoningTracker {
    /// Whether the model is currently generating reasoning tokens.
    in_thinking: bool,
    /// Count of tokens generated while in thinking mode.
    thinking_token_count: i32,
    /// Maximum thinking tokens before forcing </think>. None = unlimited.
    budget: Option<i32>,
    /// Token ID for </think> (from tokenizer vocabulary).
    think_end_id: Option<u32>,
    /// Whether the budget has been exhausted and we must force </think> next.
    force_think_end: bool,
}
```

**Initialization:** `starts_in_thinking` is `true` when `enable_thinking != Some(false)` AND `think_end_id.is_some()`. This matches the Qwen3.5 template behavior: when thinking is enabled, the template injects `<think>\n` in the prompt and the model starts generating inside a reasoning block. When thinking is disabled, the template injects `<think>\n\n</think>\n\n` and the model starts in content mode.

**Per-token operation:** `observe_token(token_id) -> bool` returns whether the token is reasoning content. When it sees `think_end_id`, it transitions `in_thinking` to `false`. While `in_thinking`, it increments the token counter and sets `force_think_end` when the budget is reached.

**Budget enforcement:** `should_force_think_end() -> bool` is checked before building the next decode step's graph. When true, the decode loop short-circuits the normal forward+sample path and directly produces the `think_end_id` token.

### 14.3 Streaming reasoning tags

vLLM uses a `stream_choice()` function that calls `parser.extract_reasoning_streaming()` with accumulated text and token IDs. `mlx-node` takes a simpler approach: since `think_end_id` is a single token, each streaming delta chunk is tagged with `is_reasoning: bool` based on the `ReasoningTracker` state at the time of emission. No text-level parsing is needed during streaming.

This maps to vLLM's `delta.reasoning` / `delta.content` distinction: consumers check `isReasoning` on each delta chunk to route text to the appropriate display channel.

### 14.4 Tool parsing isolation (already implemented)

`tools::split_at_think_end()` first separates thinking content from response content at the `</think>` boundary, then calls `parse_tool_calls()` only on the response portion. This satisfies the vLLM requirement that tool extraction never sees reasoning text. No additional changes are needed.

### 14.5 Thinking budget controller

vLLM implements budget enforcement as a `ThinkingTokenBudgetLogitsProcessor` that overrides logits. `mlx-node` has no logits processor infrastructure and does not need one. Instead, budget enforcement is a **token-level override in the decode loop**: when the `ReasoningTracker` signals budget exhaustion, the next decode step skips the forward+sample pipeline and directly produces the `think_end_id` token as an `MxArray`.

**Pipelining consideration:** The Qwen3.5 decode loop is pipelined — step N+1's graph is submitted before step N's result is extracted. When the budget is reached at step N (detected via `observe_token`), step N+1's graph has already been submitted. The forced `think_end_id` takes effect at step N+2: the `should_force_think_end()` check in the "build next step" block produces the forced token instead of running forward+sample. This means the actual thinking token count may be `budget + 1`. This 1-token lag matches vLLM's behavior, where the logits processor also operates with a scan-and-apply delay.

For the **compiled C++ path**, the forced token must still go through `eval_token_and_compiled_caches()` so the compiled KV caches update correctly. For the **Rust fallback path**, it goes through `MxArray::async_eval_arrays()`.

---

## 15. Normative behavior for `mlx-node`

### 15.1 Prompt / tracker consistency

`mlx-node` **MUST** initialize the `ReasoningTracker` using the same effective `enable_thinking` value used to render the prompt. The tracker’s `starts_in_thinking` flag must match the template’s behavior: `true` when the template injected `<think>\n`, `false` when it injected the closed `<think>\n\n</think>\n\n` block.

### 15.2 Qwen no-thinking mode

For Qwen3.5 with `enable_thinking=false`, `mlx-node` **MUST** render the closed think block in the prompt and **MUST** classify all generated text as content (tracker starts with `in_thinking=false`). This is the equivalent of vLLM’s `prompt_is_reasoning_end=true` fast path.

### 15.3 Streaming schema

In streaming mode, `mlx-node` **MUST** tag each delta chunk with `is_reasoning: bool` to distinguish reasoning from content. The existing `thinking: Option<String>` field on the final chunk continues to carry the accumulated reasoning text. Consumers check `isReasoning` to route delta text.

### 15.4 Tool parsing isolation (already satisfied)

`mlx-node` **MUST NOT** parse tool calls from reasoning text. This is already enforced by `split_at_think_end()` which separates at the `</think>` boundary before parsing tools. No changes needed.

### 15.5 Budget enforcement

`mlx-node` **MUST** enforce `thinking_token_budget` during generation by forcing the `think_end_id` token in the decode loop, not by post-generation truncation. The budget controller is the `ReasoningTracker` struct integrated into the decode loop.

### 15.6 History handling for Qwen (already supported)

`mlx-node` **SHOULD** follow Qwen’s official best practice and avoid replaying old reasoning content in history. The `ChatMessage.reasoning_content` field and the Jinja2 template already handle this — the template selectively includes or omits reasoning from prior turns.

### 15.7 Old-template compatibility (already handled)

`mlx-node` **SHOULD** tolerate older Qwen templates that emit `<think>` in generated output. The existing `parse_thinking()` and `split_at_think_end()` functions already handle both patterns: explicit `<think>...</think>` pairs and the implicit-prefix case where only `</think>` appears in the generated text.

---

## 16. `mlx-node` implementation pseudocode

### 16.1 Request entry point (`chat()` / `chat_stream()`)

```rust
fn chat(messages, config) {
    // 1. Resolve reasoning_effort → enable_thinking
    let enable_thinking = match config.reasoning_effort.as_deref() {
        Some("low") | Some("none") => Some(false),
        Some("medium") | Some("high") => Some(true),
        _ => config.enable_thinking,
    };

    // 2. Render prompt via Jinja2 template
    let tokens = tokenizer.apply_chat_template_sync(
        &messages, Some(true), tool_defs, enable_thinking,
    );

    // 3. Initialize reasoning tracker
    let starts_in_thinking = enable_thinking.unwrap_or(true)
        && think_end_id.is_some();
    let mut tracker = ReasoningTracker::new(
        starts_in_thinking,
        config.thinking_token_budget,
        think_end_id,
    );

    // 4. Prefill + decode loop (see 16.2)
    // 5. Finalize with include_reasoning suppression
}
```

### 16.2 Decode loop with reasoning tracking and budget enforcement

```rust
// Pipelined decode loop (compiled or Rust variant)
for step in 0..max_new_tokens {
    // Build step N+1’s graph — with budget check
    let next_y = if step + 1 < max_new_tokens {
        if tracker.should_force_think_end() {
            // Budget exhausted: force </think> token
            let forced = MxArray::from_int32(&[think_end_id as i32], &[1]);
            eval_token_and_compiled_caches(&forced); // or async_eval for Rust path
            Some(forced)
        } else {
            // Normal path: forward → penalties → sample → eval
            let logits = forward_compiled(&y.reshape(&[1, 1]), &embedding_weight);
            let logits = apply_all_penalties(logits, &token_history, &params);
            let next_token = sample(&logits, sampling_config);
            eval_token_and_compiled_caches(&next_token);
            Some(next_token)
        }
    } else {
        None
    };

    // Wait for step N
    y.eval();
    let token_id = y.item_at_int32(0) as u32;
    generated_tokens.push(token_id);
    token_history.push(token_id);

    // Track reasoning state
    let is_reasoning = tracker.observe_token(token_id);

    // [streaming only] Emit tagged delta chunk
    callback(ChatStreamChunk {
        text: step_decode_stream(token_id),
        done: false,
        is_reasoning: Some(is_reasoning),
        ..
    });

    // Stop conditions
    if token_id == eos_id { break; }
    if check_repetition_cutoff(..) { break; }

    y = next_y.unwrap_or_else(|| break);
}
```

### 16.3 `ReasoningTracker` state machine

```rust
impl ReasoningTracker {
    fn observe_token(&mut self, token_id: u32) -> bool {
        if !self.in_thinking { return false; }

        if self.think_end_id == Some(token_id) {
            self.in_thinking = false;
            self.force_think_end = false;
            return true; // </think> itself is part of reasoning
        }

        self.thinking_token_count += 1;
        if let Some(budget) = self.budget {
            if self.thinking_token_count >= budget {
                self.force_think_end = true;
            }
        }
        true
    }

    fn should_force_think_end(&self) -> bool {
        self.force_think_end && self.think_end_id.is_some()
    }
}
```

### 16.4 Finalization with `include_reasoning` suppression

```rust
fn finalize_chat_result(.., include_reasoning: bool) -> ChatResult {
    let text = tokenizer.decode_sync(&generated_tokens, true);
    let (clean_text, tool_calls, thinking) = split_at_think_end(&text, think_tag);
    let thinking = if include_reasoning { thinking } else { None };
    ChatResult { text: clean_text, tool_calls, thinking, .. }
}
```

---

## 17. Edge cases `mlx-node` must test

### 17.1 Qwen3.5 with thinking disabled

`ReasoningTracker` starts with `in_thinking=false`. First streamed chunk must have `is_reasoning: false`. All deltas are content.

### 17.2 Qwen3.5 with thinking enabled, truncated generation

No `</think>` appears before EOS / max_tokens. `ReasoningTracker` stays in `in_thinking=true` throughout. Non-streaming `finalize_chat_result` treats all text as thinking (existing `split_at_think_end` behavior when `think_end_tag` is not found and thinking was enabled).

### 17.3 Qwen3.5 with thinking disabled, no end token in output

`ReasoningTracker` started in content mode. All output is content. `thinking` field is `None`.

### 17.4 Old Qwen templates that emit `<think>` in generated text

The `ReasoningTracker` operates at the token level and only looks for `think_end_id`. An emitted `<think>` token in the generated text does not affect tracker state. The post-generation `parse_thinking()` already handles stripping `<think>` from the text.

### 17.5 `</think>` token in streaming

When `think_end_id` is generated, `observe_token()` transitions `in_thinking` to `false`. The `</think>` token’s delta text itself is tagged `is_reasoning: true` (it is part of the reasoning block). The very next delta is `is_reasoning: false`.

### 17.6 Auto tool choice after reasoning

`split_at_think_end()` already ensures tool parsing only sees content after `</think>`. No changes needed.

### 17.7 `include_reasoning=false`

`finalize_chat_result()` sets `thinking = None` when `include_reasoning` is false. During streaming, delta chunks still carry `is_reasoning` tags (consumers can filter), but the final chunk’s `thinking` field is `None`.

### 17.8 `thinking_token_budget=0`

`ReasoningTracker` starts with `thinking_token_count=0` and `budget=Some(0)`. On the very first thinking token, `observe_token()` increments to 1 which exceeds budget 0, setting `force_think_end=true`. The next decode step forces `think_end_id`. The model emits at most 1 thinking token before `</think>`.

### 17.9 `thinking_token_budget` without `think_end_id`

If the tokenizer does not have a `</think>` token, `ReasoningTracker::new()` sets `think_end_id=None` and `should_force_think_end()` returns `false` regardless of budget. The budget is silently ignored. This is safe because models without `</think>` in their vocabulary cannot enter thinking mode.

---

## 18. Compatibility notes

### 18.1 `enable_thinking` is the canonical control

The main vLLM docs and Qwen docs consistently use `chat_template_kwargs.enable_thinking=false` to disable thinking, and the official Qwen3.5 chat template keys off `enable_thinking`. A vLLM recipe page uses `{"reasoning": false}` instead, but `mlx-node` follows the official template and uses `enable_thinking`.

### 18.2 Field naming: `thinking` (not `reasoning`)

vLLM uses `message.reasoning` / `delta.reasoning` as the canonical field name. `mlx-node` uses `thinking` on `ChatResult` and `ChatStreamChunk` to match the existing API. The streaming delta uses `is_reasoning: bool` as a tagging flag (not a separate text field). For input messages, `ChatMessage.reasoning_content` is used for multi-turn history, matching the Jinja2 template variable name.

### 18.3 No server-level defaults

vLLM supports server-level `--default-chat-template-kwargs` and request-level overrides. `mlx-node` is a library, not a server — all kwargs are per-call via `ChatConfig`. The merge semantics are not needed.

---

## 19. Implementation order

### Phase 1: Data structures (no behavioral change)

1. Add `thinking_token_budget`, `include_reasoning`, `reasoning_effort` to `ChatConfig`
2. Add `is_reasoning` to `ChatStreamChunk`
3. Add fields to `ChatParams`, update `extract_chat_params()`
4. Create `ReasoningTracker` struct in `chat_common.rs`
5. Build and verify compilation

### Phase 2: `reasoning_effort` resolution

6. Add `reasoning_effort → enable_thinking` resolution before `apply_chat_template_sync()` in `chat()` and `chat_stream()` (×4 locations: dense + MoE)

### Phase 3: Streaming reasoning tags

7. Create `ReasoningTracker` before each decode loop (×8 loops)
8. Call `observe_token()` after token extraction in all loops
9. Set `is_reasoning` on streaming delta chunks (×4 streaming loops)
10. Update TypeScript `ChatStreamDelta` type and `_createChatStream()`

### Phase 4: Budget enforcement

11. Add `should_force_think_end()` check in "build next_y" block (×8 loops)
12. Force `think_end_id` token when budget exhausted

### Phase 5: `include_reasoning` suppression

13. Update `finalize_chat_result()` to accept and apply `include_reasoning`
14. Apply suppression in streaming finalization blocks

### Phase 6: Tests

15. Unit tests for `ReasoningTracker` state machine
16. Streaming test for `isReasoning` delta tagging
17. Integration test: thinking budget enforcement

---

## 20. Files to modify

| File | Changes |
|------|---------|
| `crates/mlx-core/src/models/qwen3_5/model.rs` | `ChatConfig` +3 fields, `ChatStreamChunk` +1 field, `reasoning_effort` resolution (×2), `ReasoningTracker` in 4 decode loops, budget enforcement in 4 loops, `is_reasoning` in 2 streaming loops |
| `crates/mlx-core/src/models/qwen3_5/chat_common.rs` | `ReasoningTracker` struct + unit tests, `ChatParams` +2 fields, `extract_chat_params()`, `finalize_chat_result()` + `include_reasoning` |
| `crates/mlx-core/src/models/qwen3_5_moe/model.rs` | Same decode loop changes as dense: `reasoning_effort` (×2), tracker in 4 loops, budget in 4 loops, `is_reasoning` in 2 streaming loops, `finalize_chat_result` call |
| `packages/lm/src/stream.ts` | `ChatStreamDelta` + `isReasoning`, `_createChatStream()` propagation |

Files NOT modified (already correct):
- `crates/mlx-core/src/tools/mod.rs` — tool parsing isolation already works
- `crates/mlx-core/src/tokenizer.rs` — `think_end_id` detection, `enable_thinking` already wired
- `crates/mlx-core/src/sampling.rs` — no logits processor needed

---

## 21. Summary

The `mlx-node` reasoning control flow relies on three cooperating mechanisms:

- **Template controls mode.** `enable_thinking` (resolved from `reasoning_effort` if set) changes the prompt shape via the Jinja2 chat template.
- **`ReasoningTracker` controls structure.** Token-level tracking of `think_end_id` determines reasoning vs content during decoding. Each streaming delta is tagged `is_reasoning`.
- **Budget controller acts during decoding.** `thinking_token_budget` forces `think_end_id` inline in the decode loop when the budget is exhausted. This is not post-hoc truncation.
- **Tool parser sees content only.** `split_at_think_end()` separates at the `</think>` boundary before calling `parse_tool_calls()`. Already implemented.
- **`include_reasoning` suppresses output.** The `thinking` field is set to `None` in `finalize_chat_result()` when suppression is requested.
