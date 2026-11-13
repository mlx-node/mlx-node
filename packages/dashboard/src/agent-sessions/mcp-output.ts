/** The text of a successful MCP tool result, as the agent reports it in `tool_call_update.rawOutput`. */

export interface McpOutput {
  server: string;
  tool: string;
  text: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * `{type:"MCP", tool_name, server_name, output:{OkayOutput:"<text>"}}` → its parts. Any other value, another output
 * variant, an error variant or a missing field is null.
 */
export function mcpOutputText(rawOutput: unknown): McpOutput | null {
  if (!isRecord(rawOutput) || rawOutput.type !== 'MCP') return null;
  const server = rawOutput.server_name;
  const tool = rawOutput.tool_name;
  const output = rawOutput.output;
  if (typeof server !== 'string' || typeof tool !== 'string' || !isRecord(output)) return null;
  const text = output.OkayOutput;
  return typeof text === 'string' ? { server, tool, text } : null;
}
