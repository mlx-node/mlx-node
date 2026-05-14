/**
 * Tool calling utilities
 *
 * Provides types and helpers for working with tool/function calling
 * through the `ChatSession` API. Tools are passed via `ChatConfig.tools`
 * on `session.send()`, and a tool result is fed back through
 * `session.sendToolResult()`.
 *
 * **Single tool call per assistant turn.** Each `sendToolResult(...)`
 * call appends one tool message and immediately re-opens the assistant
 * turn, so the session API only supports assistant turns that emit
 * exactly one tool call. If the model emits multiple tool calls in a
 * single turn, the caller must treat that as an unsupported state:
 * throw, surface an error to the user, or tighten the system prompt /
 * tool spec so the model produces at most one call per turn. Do **not**
 * loop `sendToolResult` across the remaining calls — later results
 * would be interleaved with a new assistant reply from the first, and
 * the conversation state would become inconsistent (especially for
 * stateful tools whose effects must land in order).
 *
 * @example
 * ```typescript
 * import { createToolDefinition, loadSession } from '@mlx-node/lm';
 *
 * const weatherTool = createToolDefinition(
 *   'get_weather',
 *   'Get weather for a location',
 *   { location: { type: 'string', description: 'City name' } },
 *   ['location'],
 * );
 *
 * const session = await loadSession('./my-model');
 *
 * const result = await session.send('What is the weather in Tokyo?', {
 *   config: { tools: [weatherTool] },
 * });
 *
 * const okCalls = result.toolCalls.filter((c) => c.status === 'ok');
 * if (okCalls.length > 1) {
 *   throw new Error(
 *     `ChatSession only supports one tool call per assistant turn; ` +
 *       `model emitted ${okCalls.length}. Tighten the prompt or tool spec.`,
 *   );
 * }
 * const call = okCalls[0];
 * if (call) {
 *   const toolOutput = await executeMyTool(call.name, call.arguments);
 *   const followUp = await session.sendToolResult(call.id, JSON.stringify(toolOutput), {
 *     config: { tools: [weatherTool] },
 *   });
 *   console.log(followUp.text);
 * }
 * ```
 *
 * @module tools
 */

export * from './types.js';
