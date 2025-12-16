/**
 * OpenAI-compatible tool calling types for Qwen3
 *
 * These types match the OpenAI function calling API format and can be used
 * with applyChatTemplate() when tools are provided.
 *
 * @remarks
 * **Important**: Due to NAPI-RS limitations with recursive generic types,
 * `FunctionParameters.properties` must be passed as a JSON string to the Rust layer.
 * Use the {@link createToolDefinition} helper to automatically handle this conversion.
 *
 * @example
 * ```typescript
 * // Recommended: Use the helper function
 * const tool = createToolDefinition('get_weather', 'Get weather info', {
 *   location: { type: 'string', description: 'City name' },
 *   units: { type: 'string', enum: ['celsius', 'fahrenheit'] }
 * }, ['location']);
 *
 * // Manual approach (if needed)
 * const manualTool: ToolDefinition = {
 *   type: 'function',
 *   function: {
 *     name: 'get_weather',
 *     parameters: {
 *       type: 'object',
 *       properties: JSON.stringify({ location: { type: 'string' } }),
 *       required: ['location']
 *     }
 *   }
 * };
 * ```
 */

/**
 * Tool type - currently only "function" is supported
 */
export type ToolType = 'function';

/**
 * Function parameter property definition (JSON Schema subset)
 *
 * This type is used for the developer-friendly API in {@link createToolDefinition}.
 * It represents the structure of JSON Schema properties.
 */
export interface FunctionParameterProperty {
  type: 'string' | 'number' | 'boolean' | 'integer' | 'array' | 'object';
  description?: string;
  enum?: string[];
  items?: FunctionParameterProperty;
  properties?: Record<string, FunctionParameterProperty>;
  required?: string[];
}

/**
 * Function parameters schema (JSON Schema subset)
 *
 * @remarks
 * **NAPI Limitation**: The `properties` field must be a JSON string, not an object.
 * This is because NAPI-RS cannot expose recursive generic types like
 * `Record<string, FunctionParameterProperty>` directly to Rust.
 *
 * Use {@link createToolDefinition} to avoid manual JSON.stringify() calls.
 */
export interface FunctionParameters {
  /** Type of the parameters object (always "object") */
  type: 'object';
  /**
   * JSON string of property definitions.
   *
   * @remarks
   * Must be a JSON-stringified object, e.g.: `JSON.stringify({ name: { type: 'string' } })`
   * Use {@link createToolDefinition} helper to avoid manual stringification.
   */
  properties?: string;
  /** List of required parameter names */
  required?: string[];
}

/**
 * Function definition for tool calling
 */
export interface FunctionDefinition {
  /** Name of the function */
  name: string;
  /** Description of what the function does */
  description?: string;
  /** JSON Schema for the function parameters */
  parameters?: FunctionParameters;
}

/**
 * OpenAI-compatible tool definition
 */
export interface ToolDefinition {
  /** Tool type (currently only "function" is supported) */
  type: ToolType;
  /** Function definition */
  function: FunctionDefinition;
}

/**
 * Tool call made by an assistant
 */
export interface ToolCall {
  /** Optional unique identifier for the tool call */
  id?: string;
  /** Name of the tool/function to call */
  name: string;
  /** JSON string of arguments to pass to the tool */
  arguments: string;
}

/**
 * Chat message roles
 */
export type ChatRole = 'system' | 'user' | 'assistant' | 'tool';

/**
 * Chat message with tool calling support
 *
 * This extends the basic ChatMessage to support tool calls and responses.
 */
export interface ChatMessageWithTools {
  /** Message role */
  role: ChatRole;
  /** Message content */
  content: string;
  /** Tool calls made by the assistant (for assistant messages) */
  toolCalls?: ToolCall[];
  /** Tool call ID this message is responding to (for tool messages) */
  toolCallId?: string;
  /** Reasoning content for thinking mode (used with <think> tags) */
  reasoningContent?: string;
}

/**
 * Options for applying chat template with tools
 */
export interface ApplyChatTemplateOptions {
  /** Whether to add generation prompt at end (default: true) */
  addGenerationPrompt?: boolean;
  /** Array of tool definitions for function calling */
  tools?: ToolDefinition[];
  /**
   * Control thinking mode behavior.
   *
   * @remarks
   * **Counter-intuitive semantics** (from Qwen3's Jinja2 template):
   * - `undefined` or `true`: Model thinks naturally (no tags added)
   * - `false`: Adds empty `<think>\n\n</think>\n\n` tags to **disable** thinking
   *
   * The default is `false` for tool use, which disables thinking to avoid
   * verbose reasoning during tool calls.
   *
   * @example
   * ```typescript
   * // Allow model to think (default behavior without tools)
   * { enableThinking: true }
   *
   * // Disable thinking (adds empty <think></think> tags)
   * { enableThinking: false }
   * ```
   */
  enableThinking?: boolean;
}

/**
 * Create a tool definition with automatic JSON stringification of properties.
 *
 * This helper handles the NAPI-RS limitation where `properties` must be a JSON string.
 *
 * @param name - The function name
 * @param description - Description of what the function does
 * @param properties - Object defining the function parameters (will be JSON stringified)
 * @param required - Array of required parameter names
 * @returns A properly formatted ToolDefinition ready for use with model.chat()
 *
 * @example
 * ```typescript
 * const weatherTool = createToolDefinition(
 *   'get_weather',
 *   'Get weather information for a location',
 *   {
 *     location: { type: 'string', description: 'City name' },
 *     units: { type: 'string', enum: ['celsius', 'fahrenheit'] }
 *   },
 *   ['location']
 * );
 *
 * const result = await model.chat(messages, { tools: [weatherTool] });
 * ```
 */
export function createToolDefinition(
  name: string,
  description?: string,
  properties?: Record<string, FunctionParameterProperty>,
  required?: string[],
): ToolDefinition {
  return {
    type: 'function',
    function: {
      name,
      description,
      parameters: properties
        ? {
            type: 'object',
            properties: JSON.stringify(properties),
            required,
          }
        : undefined,
    },
  };
}

/**
 * Format a tool response for inclusion in a message
 *
 * Creates a properly formatted tool response string that can be used
 * in tool messages when continuing a conversation after a tool call.
 *
 * @param content - The response content (will be JSON stringified if object)
 * @returns Formatted tool response string wrapped in `<tool_response>` tags
 *
 * @example
 * ```typescript
 * // After executing a tool call from model.chat()
 * const toolResult = await executeMyTool(call.arguments);
 * const responseMessage = {
 *   role: 'user',
 *   content: formatToolResponse(toolResult)
 * };
 * const finalResult = await model.chat([...messages, responseMessage]);
 * ```
 */
export function formatToolResponse(content: unknown): string {
  const contentStr = typeof content === 'string' ? content : JSON.stringify(content);
  return `<tool_response>
${contentStr}
</tool_response>`;
}
