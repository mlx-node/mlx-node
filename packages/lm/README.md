# @mlx-node/lm

High-level language model inference for Node.js on Apple Silicon. Supports Qwen3, Qwen3.5 (Dense and MoE), LFM2, and Gemma4 with streaming, multi-turn chat sessions, tool calling, and profiling — all running locally on Metal GPU.

## Requirements

- macOS with Apple Silicon (M1 or later)
- Node.js 18+

## Installation

```bash
npm install @mlx-node/lm
```

## Quick Start

Multi-turn chat runs through `ChatSession`, which owns the server-side KV cache and hides the session bookkeeping behind `send()` / `sendStream()`. The `loadSession()` convenience wrapper loads the model and constructs the session in one step:

```typescript
import { loadSession } from '@mlx-node/lm';

const session = await loadSession('./models/Qwen3-0.6B');

const result = await session.send('What is the capital of France?');
console.log(result.text);

// Follow-ups reuse the live KV cache — no prompt replay.
const followUp = await session.send('And its population?');
console.log(followUp.text);
```

## Streaming

Every generative model wrapper supports token-by-token streaming via `session.sendStream()`, which yields an `AsyncGenerator<ChatStreamEvent>`:

```typescript
import { loadSession } from '@mlx-node/lm';

const session = await loadSession('./models/Qwen3.5-0.8B');

for await (const event of session.sendStream('Write a haiku about TypeScript.')) {
  if (!event.done) {
    process.stdout.write(event.text);
  } else {
    console.log(`\n${event.numTokens} tokens, finish: ${event.finishReason}`);
  }
}
```

Breaking out of the loop automatically cancels generation. The session tracks its turn state so the next `send()` / `sendStream()` continues the same conversation against the live cache.

## Tool Calling

OpenAI-compatible function calling with `createToolDefinition`. Tool-result turns feed back through the same session via `sendToolResult()`, which dispatches a native `chatSessionContinueTool` against the live KV cache:

```typescript
import { loadSession, createToolDefinition } from '@mlx-node/lm';

const session = await loadSession('./models/Qwen3-0.6B');

const tools = [
  createToolDefinition(
    'get_weather',
    'Get weather for a city',
    {
      city: { type: 'string', description: 'City name' },
    },
    ['city'],
  ),
];

const result = await session.send('What is the weather in Tokyo?', { config: { tools } });

// The chat-session API only supports exactly one tool call per assistant turn:
// each `sendToolResult` dispatch immediately re-opens the assistant turn, so
// feeding a second result for the same turn would interleave a new assistant
// reply between the two results. `ChatSession` enforces this at runtime — a
// subsequent `sendToolResult*` after a multi-call turn throws with a clear
// error — and the caller must refuse multi-call turns up front. Tighten the
// prompt or tool spec so the model emits at most one call per turn.
const okCalls = result.toolCalls?.filter((tc) => tc.status === 'ok') ?? [];
if (okCalls.length > 1) {
  throw new Error(
    `ChatSession only supports one tool call per assistant turn; ` +
      `model emitted ${okCalls.length}. Tighten the prompt or tool spec.`,
  );
}
const call = okCalls[0];
if (call) {
  const toolOutput = JSON.stringify(await executeMyTool(call));
  const followUp = await session.sendToolResult(call.id, toolOutput, { config: { tools } });
  console.log(followUp.text);
}
```

## Model Loading

`loadModel()` auto-detects the model architecture from `config.json`. Use `loadSession()` when you want an ergonomic `ChatSession` handle in one step, or load a concrete model class and construct `new ChatSession(model)` when you need a reference to both the model and the session (e.g. for `generate()` calls, training, or model metadata):

```typescript
import { loadSession, ChatSession, Qwen35Model, Qwen35MoeModel } from '@mlx-node/lm';

// Convenience: auto-detect architecture and wrap in a ChatSession.
const session = await loadSession('./models/Qwen3-0.6B', { system: 'Be concise.' });

// Or load a specific architecture directly — every generative model wrapper
// structurally satisfies ChatSession's SessionCapableModel bound.
const dense = await Qwen35Model.load('./models/Qwen3.5-0.8B');
const moe = await Qwen35MoeModel.load('./models/Qwen3.5-35B-A3B');
const denseSession = new ChatSession(dense);
const moeSession = new ChatSession(moe);
```

`loadSession()` rejects embedding models (`HarrierModel`) and the native `QianfanOCRModel` — for the VLM case, import `QianfanOCRModel` from `@mlx-node/vlm` and wrap it with `new ChatSession(...)` directly.

`ChatSession` accepts an options bag with `{ system?, defaultConfig? }`. The system prompt is injected on the first turn and never re-sent. Per-call config passed to `send()` / `sendStream()` shallow-merges on top of `defaultConfig`. Call `session.reset()` to wipe the KV cache and start a fresh conversation.

### Pre-defined Configs

```typescript
import { QWEN3_CONFIGS, QWEN35_CONFIGS, getQwen3Config, getQwen35Config } from '@mlx-node/lm';

// Available Qwen3 configs: 'qwen3-0.6b', 'qwen3-1.7b', 'qwen3-7b'
const config = getQwen3Config('qwen3-0.6b');

// Available Qwen3.5 configs: 'qwen3.5-0.6b'
const config35 = getQwen35Config('qwen3.5-0.6b');
```

## Profiling

Track per-generation timing, memory usage, and TTFT:

```typescript
import { enableProfiling, disableProfiling } from '@mlx-node/lm';

enableProfiling();

// ... run inference ...

disableProfiling(); // writes mlx-profile-{timestamp}.json
```

Or set `MLX_PROFILE_DECODE=1` to auto-enable and write a report on exit.

## API Reference

### Classes

| Class            | Description                                                                       |
| ---------------- | --------------------------------------------------------------------------------- |
| `loadModel()`    | Auto-detect and load any supported model from disk                                |
| `loadSession()`  | `loadModel()` + `new ChatSession(model)` in one step                              |
| `ChatSession<M>` | Multi-turn chat wrapper — `send()`, `sendStream()`, `sendToolResult()`, `reset()` |
| `Qwen3Model`     | Qwen3 inference — `generate()`, paged attention, speculative decoding             |
| `Qwen35Model`    | Qwen3.5 Dense — `generate()` with compiled C++ forward                            |
| `Qwen35MoeModel` | Qwen3.5 MoE — `generate()` with compiled C++ forward and expert routing           |
| `Gemma4Model`    | Gemma4 inference — `generate()`                                                   |
| `Lfm2Model`      | LFM2.5 hybrid conv+attention inference — `generate()`                             |

### Streaming Types

```typescript
// Intermediate token
interface ChatStreamDelta {
  text: string;
  done: false;
}

// Final result
interface ChatStreamFinal {
  text: string;
  done: true;
  finishReason: string;
  toolCalls?: ToolCallResult[];
  thinking?: string;
  numTokens: number;
  rawText: string;
  performance?: PerformanceMetrics;
}

type ChatStreamEvent = ChatStreamDelta | ChatStreamFinal;
```

### Tool Types

```typescript
interface ToolDefinition {
  type: 'function';
  function: FunctionDefinition;
}

interface FunctionDefinition {
  name: string;
  description?: string;
  parameters?: FunctionParameters;
}

function createToolDefinition(
  name: string,
  description?: string,
  properties?: Record<string, FunctionParameterProperty>,
  required?: string[],
): ToolDefinition;
```

### Functions

| Function                 | Description                                          |
| ------------------------ | ---------------------------------------------------- |
| `createToolDefinition()` | Create an OpenAI-compatible tool definition          |
| `detectModelType()`      | Read `config.json` and return the `model_type` field |
| `enableProfiling()`      | Start profiling with auto-report on exit             |
| `disableProfiling()`     | Stop profiling and write JSON report                 |

## Supported Models

Every generative model wrapper exposes the same `ChatSession<M>` surface — `send()`, `sendStream()`, and `sendToolResult()` all work against any of the models below.

| Model         | `generate()` | `ChatSession` | Training | Notes                                 |
| ------------- | :----------: | :-----------: | :------: | ------------------------------------- |
| Qwen3         |     Yes      |      Yes      | GRPO/SFT | Paged attention, speculative decoding |
| Qwen3.5 Dense |     Yes      |      Yes      | GRPO/SFT | Compiled C++ forward, VLM variant     |
| Qwen3.5 MoE   |     Yes      |      Yes      | GRPO/SFT | Compiled C++ forward, expert routing  |
| Gemma4        |     Yes      |      Yes      |    No    | Streaming chat via session            |
| LFM2.5        |     Yes      |      Yes      |    No    | Hybrid conv + attention architecture  |

## Performance

- Almost the same with the python `mlx-lm` package
- Metal GPU acceleration on all Apple Silicon
- Compiled forward passes via `mlx::core::compile` for graph caching

## License

[MIT](https://github.com/mlx-node/mlx-node/blob/main/LICENSE)
