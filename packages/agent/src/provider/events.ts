/**
 * `TurnEmitter` — maps one native `ChatStreamEvent` turn onto pi's
 * `AssistantMessageEvent` protocol.
 *
 * Correctness contract (spike-proven):
 *   - pi's agent loop takes the final message from `stream.result()` and
 *     reads `stopReason` off THAT message — the `done` event's `reason`
 *     field is discarded. Both are still emitted protocol-correct.
 *   - Aborted native streams end with NO final event, so the caller must
 *     invoke {@link TurnEmitter.onAborted} to synthesize the terminal
 *     AssistantMessage (stopReason 'aborted', accumulated deltas intact).
 *   - Every method is throw-safe: the pi StreamFn contract does not allow
 *     the emitter to throw, so internal failures route to {@link onError}.
 */

import type {
  Api,
  AssistantMessage,
  AssistantMessageEventStream,
  Model,
  TextContent,
  ThinkingContent,
  ToolCall,
  Usage,
} from '@earendil-works/pi-ai';
import type { ChatStreamDelta, ChatStreamFinal, ToolCallResult } from '@mlx-node/lm';

import { coerceErrorMessage } from './error-coercion.js';
import { ToolCallTagBuffer } from './tool-call-buffer.js';

/**
 * All-zero usage. Shared with the stream adapter's TurnEmitter-independent
 * failsafe terminal, so it must stay trivially non-throwing.
 */
export function emptyUsage(): Usage {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

/**
 * pi auto-compaction thresholds on `usage.totalTokens` vs
 * `model.contextWindow`, so these numbers are load-bearing:
 * input excludes the cache-served prefix, totalTokens is the full
 * prompt + completion. All costs are 0 for local inference.
 */
function usageFromFinal(final: ChatStreamFinal): Usage {
  const cachedTokens = final.cachedTokens ?? 0;
  return {
    input: Math.max(0, final.promptTokens - cachedTokens),
    output: final.numTokens,
    cacheRead: cachedTokens,
    cacheWrite: 0,
    reasoning: final.reasoningTokens,
    totalTokens: final.promptTokens + final.numTokens,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

/**
 * Native `ToolCallResult` → pi `ToolCall`.
 *
 * Non-ok results (invalid_json / missing_name / parse_error) become a
 * deliberately-invalid pi ToolCall so pi's own tool validation fails it
 * and feeds an error tool result back to the model — pi owns that retry
 * loop (spike-proven).
 */
function toPiToolCall(call: ToolCallResult): ToolCall {
  if (call.status === 'ok') {
    const args =
      typeof call.arguments === 'object' && call.arguments !== null ? (call.arguments as Record<string, unknown>) : {};
    return { type: 'toolCall', id: call.id, name: call.name, arguments: args };
  }
  return {
    type: 'toolCall',
    id: call.id,
    name: call.name || 'malformed_tool_call',
    arguments: { raw: call.rawContent, error: call.error },
  };
}

export class TurnEmitter {
  private readonly stream: AssistantMessageEventStream;
  private readonly partial: AssistantMessage;
  private readonly textBuffer = new ToolCallTagBuffer();
  /**
   * Leading whitespace-only text parked before any text block exists, so
   * a `"\n\n"` emitted right before `<tool_call>` markup never ratifies a
   * whitespace-only text content block (mirrors the server endpoints).
   * Joined onto the first non-whitespace text; dropped at terminal time.
   */
  private pendingLeadingWhitespace = '';
  private openBlock: TextContent | ThinkingContent | null = null;
  private finished = false;

  constructor(stream: AssistantMessageEventStream, model: Model<Api>) {
    this.stream = stream;
    this.partial = {
      role: 'assistant',
      content: [],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: emptyUsage(),
      stopReason: 'stop',
      timestamp: Date.now(),
    };
    stream.push({ type: 'start', partial: this.partial });
  }

  onDelta(delta: ChatStreamDelta): void {
    if (this.finished) return;
    try {
      if (delta.isReasoning === true) {
        this.appendThinking(delta.text);
      } else {
        // Text routes through the tag buffer: partial structural markup
        // (`<tool_call>` etc.) must never leak into pi-visible text.
        const { safeText, tagFound, cleanPrefix } = this.textBuffer.push(delta.text);
        if (tagFound) {
          this.appendVisibleText(cleanPrefix);
        } else if (safeText) {
          this.appendVisibleText(safeText);
        }
      }
    } catch (err) {
      this.onError(err);
    }
  }

  onFinal(final: ChatStreamFinal): void {
    if (this.finished) return;
    try {
      this.partial.usage = usageFromFinal(final);
      if (final.finishReason === 'error') {
        this.finishWithError('error', 'model reported finishReason=error');
        return;
      }

      // Release any held-back non-tag suffix, then close the open block.
      this.appendVisibleText(this.textBuffer.flush());
      this.closeOpenBlock();

      let sawOkToolCall = false;
      for (const call of final.toolCalls) {
        if (call.status === 'ok') sawOkToolCall = true;
        const toolCall = toPiToolCall(call);
        this.partial.content.push(toolCall);
        const contentIndex = this.partial.content.length - 1;
        this.stream.push({ type: 'toolcall_start', contentIndex, partial: this.partial });
        this.stream.push({
          type: 'toolcall_delta',
          contentIndex,
          delta: JSON.stringify(toolCall.arguments),
          partial: this.partial,
        });
        this.stream.push({ type: 'toolcall_end', contentIndex, toolCall, partial: this.partial });
      }

      const reason = sawOkToolCall ? 'toolUse' : final.finishReason === 'length' ? 'length' : 'stop';
      this.partial.stopReason = reason;
      this.finished = true;
      this.stream.push({ type: 'done', reason, message: this.partial });
      this.stream.end();
    } catch (err) {
      this.onError(err);
    }
  }

  /**
   * Synthesize the terminal message for an aborted native stream (which
   * ends with no final event). Mirrors pi's provider abort pattern:
   * `{type:'error', reason:'aborted', error: <partial message>}` with all
   * accumulated text/thinking preserved on the message.
   */
  onAborted(): void {
    if (this.finished) return;
    this.finishWithError('aborted', 'Request was aborted');
  }

  /**
   * Terminal for internal/adapter failures. `err` is untrusted: coercion
   * is fully guarded (shared `coerceErrorMessage`), so even a revoked
   * Proxy or a poisoned `message` getter cannot throw out of here.
   */
  onError(err: unknown): void {
    if (this.finished) return;
    this.finishWithError('error', coerceErrorMessage(err));
  }

  /**
   * Shared terminal path for every non-`done` ending (abort, native
   * finishReason=error, internal failure). Recovers any held-back buffer
   * residue and closes the open text/thinking block BEFORE emitting the
   * terminal event — a stream must never end `text_start/text_delta/error`
   * with no `text_end` (pi's reference providers balance all blocks
   * before terminals).
   */
  private finishWithError(reason: 'aborted' | 'error', message: string): void {
    this.finished = true;
    try {
      try {
        this.appendVisibleText(this.textBuffer.flush());
      } catch {
        // Preserving buffered residue is best-effort; the block close and
        // terminal event below must still go out.
      }
      this.closeOpenBlock();
      this.partial.stopReason = reason;
      this.partial.errorMessage = message;
      this.stream.push({ type: 'error', reason, error: this.partial });
      this.stream.end();
    } catch {
      // The StreamFn contract forbids throwing; a push/end failure here
      // has no further recovery surface.
    }
  }

  private appendThinking(text: string): void {
    let block = this.openBlock;
    if (block?.type !== 'thinking') {
      this.closeOpenBlock();
      block = { type: 'thinking', thinking: '' };
      this.partial.content.push(block);
      this.openBlock = block;
      this.stream.push({ type: 'thinking_start', contentIndex: this.blockIndex(), partial: this.partial });
    }
    block.thinking += text;
    this.stream.push({
      type: 'thinking_delta',
      contentIndex: this.blockIndex(),
      delta: text,
      partial: this.partial,
    });
  }

  private appendVisibleText(text: string): void {
    if (this.openBlock?.type === 'text') {
      if (!text) return;
      this.openBlock.text += text;
      this.stream.push({ type: 'text_delta', contentIndex: this.blockIndex(), delta: text, partial: this.partial });
      return;
    }
    // No text block open yet: whitespace-only text stays parked so it can
    // front real text later or be dropped silently at tag/terminal time.
    const combined = this.pendingLeadingWhitespace + text;
    if (combined.trim().length === 0) {
      this.pendingLeadingWhitespace = combined;
      return;
    }
    this.pendingLeadingWhitespace = '';
    this.closeOpenBlock();
    const block: TextContent = { type: 'text', text: '' };
    this.partial.content.push(block);
    this.openBlock = block;
    this.stream.push({ type: 'text_start', contentIndex: this.blockIndex(), partial: this.partial });
    block.text += combined;
    this.stream.push({
      type: 'text_delta',
      contentIndex: this.blockIndex(),
      delta: combined,
      partial: this.partial,
    });
  }

  private closeOpenBlock(): void {
    const block = this.openBlock;
    if (!block) return;
    this.openBlock = null;
    if (block.type === 'text') {
      this.stream.push({
        type: 'text_end',
        contentIndex: this.partial.content.indexOf(block),
        content: block.text,
        partial: this.partial,
      });
    } else {
      this.stream.push({
        type: 'thinking_end',
        contentIndex: this.partial.content.indexOf(block),
        content: block.thinking,
        partial: this.partial,
      });
    }
  }

  private blockIndex(): number {
    return this.partial.content.length - 1;
  }
}
