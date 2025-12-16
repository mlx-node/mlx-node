/**
 * Tool calling utilities for Qwen3
 *
 * Provides types and helpers for working with tool/function calling in the chat() API.
 *
 * @example
 * ```typescript
 * import { createToolDefinition, formatToolResponse } from '@mlx-node/lm';
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
 *     // Continue conversation with tool result
 *     messages.push({ role: 'user', content: formatToolResponse(toolResult) });
 *   }
 * }
 * ```
 *
 * @module tools
 */

export * from './types';
