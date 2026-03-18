# @mlx-node/lm

High-level language model inference for Node.js on Apple Silicon. Supports Qwen3 and Qwen3.5 (Dense and MoE) with streaming, tool calling, and profiling — all running locally on Metal GPU.

## Requirements

- macOS with Apple Silicon (M1 or later)
- Node.js 18+

## Installation

```bash
npm install @mlx-node/lm
```

## Quick Start

```typescript
import { loadModel } from '@mlx-node/lm';

const model = await loadModel('./models/Qwen3-0.6B');

const result = model.chat([{ role: 'user', content: 'What is the capital of France?' }]);

console.log(result.text);
```

## Streaming

Qwen3.5 models support token-by-token streaming via `AsyncGenerator`:

```typescript
import { loadModel } from '@mlx-node/lm';

const model = await loadModel('./models/Qwen3.5-0.6B');

for await (const event of model.chatStream(messages, config)) {
  if (!event.done) {
    process.stdout.write(event.text);
  } else {
    console.log(`\n${event.numTokens} tokens, finish: ${event.finishReason}`);
  }
}
```

Breaking out of the loop automatically cancels generation.

## Tool Calling

OpenAI-compatible function calling with `createToolDefinition`:

```typescript
import { loadModel, createToolDefinition, formatToolResponse } from '@mlx-node/lm';

const model = await loadModel('./models/Qwen3-0.6B');

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

const result = model.chat([{ role: 'user', content: 'What is the weather in Tokyo?' }], { tools });

// If the model calls a tool, execute it and continue
if (result.toolCalls?.length) {
  const toolResult = executeMyTool(result.toolCalls[0]);
  const followUp = model.chat(
    [
      ...messages,
      { role: 'assistant', content: result.rawText },
      { role: 'tool', content: formatToolResponse(toolResult) },
    ],
    { tools },
  );
}
```

## Model Loading

`loadModel()` auto-detects the model architecture from `config.json`:

```typescript
import { loadModel, Qwen35Model, Qwen35MoeModel } from '@mlx-node/lm';

// Auto-detect (reads config.json model_type field)
const model = await loadModel('./models/Qwen3-0.6B');

// Or load a specific architecture directly
const dense = await Qwen35Model.load('./models/Qwen3.5-0.8B');
const moe = await Qwen35MoeModel.load('./models/Qwen3.5-35B-A3B');
```

### Pre-defined Configs

```typescript
import { QWEN3_CONFIGS, QWEN35_CONFIGS, getQwen3Config, getQwen35Config } from '@mlx-node/lm';

// Available Qwen3 configs: 'qwen3-0.6b', 'qwen3-1.7b', 'qwen3-7b'
const config = getQwen3Config('qwen3-0.6b');

// Available Qwen3.5 configs: 'qwen3.5-0.8b'
const config35 = getQwen35Config('qwen3.5-0.8b');
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

| Class            | Description                                                                      |
| ---------------- | -------------------------------------------------------------------------------- |
| `loadModel()`    | Auto-detect and load any supported model from disk                               |
| `Qwen3Model`     | Qwen3 inference — `generate()`, `chat()`, paged attention, speculative decoding  |
| `Qwen35Model`    | Qwen3.5 Dense — `generate()`, `chat()`, `chatStream()` with compiled C++ forward |
| `Qwen35MoeModel` | Qwen3.5 MoE — same API as Dense with expert routing                              |

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

function formatToolResponse(content: string): string;
```

### Functions

| Function                 | Description                                          |
| ------------------------ | ---------------------------------------------------- |
| `createToolDefinition()` | Create an OpenAI-compatible tool definition          |
| `formatToolResponse()`   | Wrap tool output in `<tool_response>` tags           |
| `detectModelType()`      | Read `config.json` and return the `model_type` field |
| `enableProfiling()`      | Start profiling with auto-report on exit             |
| `disableProfiling()`     | Stop profiling and write JSON report                 |

## Supported Models

| Model         | `chat()` | `chatStream()` | Training | Notes                                 |
| ------------- | :------: | :------------: | :------: | ------------------------------------- |
| Qwen3         |   Yes    |       No       | GRPO/SFT | Paged attention, speculative decoding |
| Qwen3.5 Dense |   Yes    |      Yes       | GRPO/SFT | Compiled C++ forward, VLM variant     |
| Qwen3.5 MoE   |   Yes    |      Yes       | GRPO/SFT | Compiled C++ forward, expert routing  |

## Performance

- Almost the same with the python `mlx-lm` package
- Metal GPU acceleration on all Apple Silicon
- Compiled forward passes via `mlx::core::compile` for graph caching

## License

[MIT](https://github.com/mlx-node/mlx-node/blob/main/LICENSE)
