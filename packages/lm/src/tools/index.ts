/**
 * Tool calling utilities for Qwen3
 *
 * Provides types and helpers for working with tool/function calling in the chat() API.
 *
 * @example
 * ```typescript
 * import { createToolDefinition } from '@mlx-node/lm';
 *
 * const weatherTool = createToolDefinition(
 *   'get_weather',
 *   'Get weather for a location',
 *   { location: { type: 'string', description: 'City name' } },
 *   ['location']
 * );
 *
 * const result = await model.chat(messages, { tools: [weatherTool] });
 *
 * for (const call of result.toolCalls) {
 *   if (call.status === 'ok') {
 *     const toolResult = await executeMyTool(call.name, call.arguments);
 *     // Use role: 'tool' — the Jinja2 template wraps content in <tool_response> tags
 *     messages.push({ role: 'tool', content: JSON.stringify(toolResult) });
 *   }
 * }
 * ```
 *
 * @module tools
 */

export * from './types.js';
