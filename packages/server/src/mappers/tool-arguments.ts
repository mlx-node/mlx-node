/**
 * A local model asked for the agent's `use_tool` `{"tool_name": "…", "tool_input": {"url": "…"}}` often writes
 * `{"tool_name": "…", "url": "…"}` (the Qwen `<parameter=…>` format makes every argument a flat pair), and the agent
 * refuses that form before the call can run. When a call misses exactly one required property and that property is an
 * object, the top-level keys the schema does not declare move inside it.
 */

import type { ToolCallResult } from '@mlx-node/core';

import type { AnthropicToolDefinition } from '../types-anthropic.js';

const UNION_KEYS = ['oneOf', 'anyOf', 'allOf', 'not', '$ref', 'patternProperties'];

/**
 * The agent's `use_tool` accepts `{tool_name, tool_input_file}` (it reads the arguments from that file) but leaves the
 * key out of the schema it sends on a Messages backend: a call naming it is already a form the agent runs.
 */
const UNDECLARED_FORM_KEYS = ['tool_input_file'];

function plainObject(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/** An own, enumerable property: a key named `__proto__` must not set the prototype. */
function put(target: Record<string, unknown>, key: string, value: unknown): void {
  Object.defineProperty(target, key, { value, enumerable: true, writable: true, configurable: true });
}

/** The arguments with the stray top-level keys moved into the one missing object property, or null to keep them. */
export function nestStrayArguments(args: Record<string, unknown>, schema: unknown): Record<string, unknown> | null {
  const root = plainObject(schema);
  if (root === null || root.type !== 'object') return null;
  if (UNION_KEYS.some((key) => Object.hasOwn(root, key))) return null;
  if (root.additionalProperties !== undefined && root.additionalProperties !== false) return null;
  const properties = plainObject(root.properties);
  if (properties === null || !Array.isArray(root.required)) return null;
  const missing = root.required.filter(
    (name): name is string => typeof name === 'string' && !Object.hasOwn(args, name),
  );
  if (missing.length !== 1) return null;
  const target = missing[0]!;
  const targetSchema = plainObject(properties[target]);
  if (targetSchema === null || targetSchema.type !== 'object') return null;
  const stray = Object.keys(args).filter((key) => !Object.hasOwn(properties, key));
  if (stray.length === 0 || stray.some((key) => UNDECLARED_FORM_KEYS.includes(key))) return null;
  if (targetSchema.additionalProperties === false) {
    const allowed = plainObject(targetSchema.properties);
    if (allowed === null || !stray.every((key) => Object.hasOwn(allowed, key))) return null;
  }
  const kept: Record<string, unknown> = {};
  const nested: Record<string, unknown> = {};
  for (const key of Object.keys(args)) put(stray.includes(key) ? nested : kept, key, args[key]);
  put(kept, target, nested);
  return kept;
}

/**
 * `nestStrayArguments` for every parsed call of a Messages request, by the schema of the one tool with that name.
 * Text arguments are left as they are: a key written twice in a text is read differently by different readers.
 */
export function withNestedArguments(
  calls: ToolCallResult[],
  tools: readonly AnthropicToolDefinition[] | undefined,
): ToolCallResult[] {
  if (!Array.isArray(tools)) return calls;
  return calls.map((call) => {
    if (call.status !== 'ok') return call;
    const args = plainObject(call.arguments);
    if (args === null) return call;
    const named = tools.filter((tool) => plainObject(tool)?.name === call.name);
    if (named.length !== 1) return call;
    const repaired = nestStrayArguments(args, named[0]!.input_schema);
    return repaired === null ? call : { ...call, arguments: repaired };
  });
}
