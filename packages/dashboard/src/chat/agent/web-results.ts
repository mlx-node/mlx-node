/** What the Chat host takes from a web tool result: search addresses for the ledger, and key failures. */

import { mcpOutputText } from '../../agent-sessions/mcp-output.js';
import { parseWebResultText, type WebResult } from '../web/result-text.js';
import type { AddressLedger } from './address-ledger.js';
import { WEB_SERVER_NAME } from './profiles.js';

/**
 * The parsed result of a `tool_call_update`, in either call shape: only a successful MCP result of the `mlx_web`
 * server whose text parses whole. Anything else (another server, a failed call, text that does not parse) is null.
 */
export function readWebToolResult(update: Record<string, unknown>): WebResult | null {
  if (update.sessionUpdate !== 'tool_call_update' || update.rawOutput === undefined) return null;
  const output = mcpOutputText(update.rawOutput);
  if (output === null || output.server !== WEB_SERVER_NAME) return null;
  return parseWebResultText(output.text);
}

export interface WebResultEffects {
  ledger: AddressLedger;
  /** A `WEB_ERROR key-rejected` or `no-credit` marks the provider's key check. */
  onKeyCheck?: (provider: 'exa' | 'deepseek' | 'brave' | 'firecrawl', check: 'rejected' | 'no-credit') => void;
}

export function applyWebResult(result: WebResult, effects: WebResultEffects): void {
  if (result.kind === 'search') {
    effects.ledger.addSearchResults(result.results.map((r) => r.url));
    return;
  }
  if (result.kind === 'error' && result.provider !== 'none') {
    if (result.code === 'key-rejected') effects.onKeyCheck?.(result.provider, 'rejected');
    else if (result.code === 'no-credit') effects.onKeyCheck?.(result.provider, 'no-credit');
  }
}
