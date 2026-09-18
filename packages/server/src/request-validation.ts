import type { ChatMessage } from '@mlx-node/core';

/**
 * Upper bound for a client-supplied output-token budget. The native
 * `ChatConfig.max_new_tokens` is `Option<i32>`, and NAPI's
 * `napi_get_value_int32` silently truncates a JS integer above `i32::MAX`
 * to a NEGATIVE value — which the core clamp then turns into 0 (a silent
 * empty completion). Reject anything above this bound at the edge so an
 * over-large budget 400s instead of producing nothing. Shared with
 * `/v1/messages` (`messages.ts`).
 */
export const MAX_OUTPUT_TOKENS = 2147483647; // i32::MAX — native ChatConfig.max_new_tokens is i32

/**
 * Reorder tool messages in `messages[startOffset, blockEnd)` to match
 * `expectedOrder`. Replay correctness for a multi-call fan-out depends
 * on POSITION — several native backends drop the id on the wire and
 * pair results to calls by sibling index, so a reordered submission
 * would silently bind results to the wrong calls even after the
 * id-set gate passes.
 *
 * `blockEnd` MUST be sized to a single contiguous tool block; the
 * full-history walker computes one per fan-out. No-op when any
 * precondition fails.
 */
export function canonicalizeToolMessageOrder(
  messages: ChatMessage[],
  startOffset: number,
  blockEnd: number,
  expectedOrder: readonly string[],
): void {
  const toolPositions: number[] = [];
  const byId = new Map<string, ChatMessage>();
  for (let i = startOffset; i < blockEnd; i++) {
    const m = messages[i]!;
    if (m.role === 'tool' && typeof m.toolCallId === 'string' && m.toolCallId.length > 0) {
      toolPositions.push(i);
      byId.set(m.toolCallId, m);
    }
  }
  if (toolPositions.length !== expectedOrder.length) return;
  for (const id of expectedOrder) {
    if (!byId.has(id)) return;
  }
  let alreadyOrdered = true;
  for (let k = 0; k < toolPositions.length; k++) {
    if (messages[toolPositions[k]!]!.toolCallId !== expectedOrder[k]) {
      alreadyOrdered = false;
      break;
    }
  }
  if (alreadyOrdered) return;
  for (let k = 0; k < toolPositions.length; k++) {
    messages[toolPositions[k]!] = byId.get(expectedOrder[k]!)!;
  }
}

/**
 * Walk the full `messages` history, validate each assistant fan-out's
 * tool-result block, and canonicalize each block to sibling order in
 * place. Invoked on stateless cold-start histories and on the
 * Anthropic `/v1/messages` endpoint (both feed caller-supplied tool
 * order straight into `primeHistory()` without the continuation gate).
 *
 * Validation rejects: orphan tool messages, unknown `toolCallId`s,
 * missing/duplicate resolutions, and a trailing unresolved fan-out in
 * a stateless history. Returns `null` on success or a human-readable
 * error string (sent as 400 `invalid_request_error`).
 *
 * @param apiSurface controls error-string vocabulary (`openai` default
 *   uses `function_call_output` / `call_id`; `anthropic` uses
 *   `tool_result` / `tool_use_id`). Validation logic is identical.
 */
export function validateAndCanonicalizeHistoryToolOrder(
  messages: ChatMessage[],
  apiSurface: 'openai' | 'anthropic' = 'openai',
): string | null {
  const vocab =
    apiSurface === 'anthropic'
      ? {
          toolResult: 'tool_result',
          toolCallId: 'tool_use_id',
          fanOut: 'assistant turn with tool_use blocks',
        }
      : {
          toolResult: 'function_call_output',
          toolCallId: 'call_id',
          fanOut: 'assistant fan-out',
        };

  let i = 0;
  while (i < messages.length) {
    const m = messages[i]!;
    if (m.role === 'tool') {
      return (
        `tool message at index ${i} (${vocab.toolCallId} "${m.toolCallId ?? ''}") is not preceded by an ` +
        `${vocab.fanOut}. Every ${vocab.toolResult} must immediately follow the assistant turn whose ` +
        `tool calls include its ${vocab.toolCallId}.`
      );
    }
    if (m.role !== 'assistant' || !m.toolCalls || m.toolCalls.length === 0) {
      i++;
      continue;
    }

    // Assistant fan-out. Collect declared sibling ids.
    const declaredIds: string[] = [];
    const declaredSet = new Set<string>();
    for (const tc of m.toolCalls) {
      const id = typeof tc.id === 'string' ? tc.id : null;
      if (id === null || id.length === 0) {
        return (
          `${vocab.fanOut} at index ${i} declares a tool call with no id, which cannot be paired ` +
          `with its ${vocab.toolResult} positionally.`
        );
      }
      if (declaredSet.has(id)) {
        return (
          `${vocab.fanOut} at index ${i} declares duplicate ${vocab.toolCallId} "${id}". Each sibling ` +
          `call must have a unique ${vocab.toolCallId}.`
        );
      }
      declaredIds.push(id);
      declaredSet.add(id);
    }

    // Read the contiguous tool block following the fan-out.
    const blockStart = i + 1;
    let blockEnd = blockStart;
    const seenInBlock = new Set<string>();
    while (blockEnd < messages.length && messages[blockEnd]!.role === 'tool') {
      const tool = messages[blockEnd]!;
      const id = typeof tool.toolCallId === 'string' ? tool.toolCallId : null;
      if (id === null || id.length === 0) {
        return (
          `tool message at index ${blockEnd} is missing ${vocab.toolCallId}. Every ${vocab.toolResult} ` +
          `in an ${vocab.fanOut}'s resolution block must carry the ${vocab.toolCallId} it resolves.`
        );
      }
      if (!declaredSet.has(id)) {
        return (
          `tool message at index ${blockEnd} references ${vocab.toolCallId} "${id}", which is not ` +
          `declared by the preceding ${vocab.fanOut} at index ${i}. Submitting a ${vocab.toolResult} ` +
          `for an undeclared ${vocab.toolCallId} would silently bind output to the wrong sibling.`
        );
      }
      if (seenInBlock.has(id)) {
        return (
          `duplicate tool message for ${vocab.toolCallId} "${id}" inside the ${vocab.fanOut}'s ` +
          `resolution block (index ${blockEnd}). Each outstanding sibling must be resolved exactly once.`
        );
      }
      seenInBlock.add(id);
      blockEnd++;
    }

    const blockLength = blockEnd - blockStart;
    if (blockLength === 0) {
      // Trailing unresolved fan-out is rejected — a stateless history
      // has nothing for the model to continue from. Mid-history the
      // next non-tool turn orphans the fan-out.
      if (blockEnd === messages.length) {
        return (
          `${vocab.fanOut} at index ${i} is the trailing turn of the history but has no ` +
          `${vocab.toolResult} resolutions. A stateless cold-start history cannot end on an ` +
          `unresolved tool-call fan-out because there is nothing for the model to continue from.`
        );
      }
      return (
        `${vocab.fanOut} at index ${i} declares ${declaredIds.length} tool call${declaredIds.length === 1 ? '' : 's'} ` +
        `but the next message at index ${blockEnd} is a ${messages[blockEnd]!.role} turn. Every fan-out ` +
        `must be fully resolved by ${vocab.toolResult} messages before the next assistant/user/system turn.`
      );
    }
    if (blockLength < declaredIds.length) {
      const missing = declaredIds.filter((id) => !seenInBlock.has(id));
      return (
        `${vocab.fanOut} at index ${i} has unresolved sibling tool calls: ${missing.join(', ')}. ` +
        `Every declared tool call must be answered by a ${vocab.toolResult} before the next turn.`
      );
    }
    // blockLength > declaredIds.length is impossible (every id is in
    // declaredSet and seenInBlock dedupes).

    canonicalizeToolMessageOrder(messages, blockStart, blockEnd, declaredIds);
    i = blockEnd;
  }

  return null;
}
