/**
 * The MCP surface of the `mlx_web` server, hand-written: `initialize`, `notifications/initialized`, `tools/list`,
 * `tools/call` and `ping`. Anything else with an id is answered `-32601`. Frames are newline-delimited JSON; a frame
 * over 1 MiB or one that is not JSON is dropped and reported with one fixed stderr line.
 */

export const MAX_FRAME_BYTES = 1024 * 1024;
export const SERVER_NAME = 'mlx_web';
const DEFAULT_PROTOCOL_VERSION = '2025-06-18';

export interface McpToolHost {
  definitions(): readonly unknown[];
  call(name: string, args: unknown): Promise<string>;
}

type JsonRpcId = string | number;

function isId(value: unknown): value is JsonRpcId {
  return typeof value === 'string' || (typeof value === 'number' && Number.isFinite(value));
}

function result(id: JsonRpcId, value: unknown): Record<string, unknown> {
  return { jsonrpc: '2.0', id, result: value };
}

function error(id: JsonRpcId | null, code: number, message: string): Record<string, unknown> {
  return { jsonrpc: '2.0', id, error: { code, message } };
}

/** The reply to one parsed frame, or null when none is due (a notification, a reply to nothing). */
export async function answerFrame(frame: unknown, tools: McpToolHost): Promise<Record<string, unknown> | null> {
  if (typeof frame !== 'object' || frame === null || Array.isArray(frame))
    return error(null, -32600, 'Invalid request');
  const message = frame as { id?: unknown; method?: unknown; params?: unknown };
  if (typeof message.method !== 'string') return null;
  if (message.id === undefined) return null;
  if (!isId(message.id)) return error(null, -32600, 'Invalid request');
  const id = message.id;
  const params =
    typeof message.params === 'object' && message.params !== null ? (message.params as Record<string, unknown>) : {};
  switch (message.method) {
    case 'initialize':
      return result(id, {
        protocolVersion: typeof params.protocolVersion === 'string' ? params.protocolVersion : DEFAULT_PROTOCOL_VERSION,
        capabilities: { tools: { listChanged: false } },
        serverInfo: { name: SERVER_NAME, version: '1' },
      });
    case 'ping':
      return result(id, {});
    case 'tools/list':
      return result(id, { tools: tools.definitions() });
    case 'tools/call': {
      const name = typeof params.name === 'string' ? params.name : '';
      const text = await tools.call(name, params.arguments);
      return result(id, { content: [{ type: 'text', text }], isError: false });
    }
    default:
      return error(id, -32601, 'Method not found');
  }
}

export interface FrameSink {
  frame(text: string): void;
  tooLarge(): void;
}

/** Bytes → newline-terminated frames. A frame over the cap is skipped up to its newline and reported once. */
export class FrameReader {
  private parts: Buffer[] = [];
  private size = 0;
  private skipping = false;
  private readonly sink: FrameSink;
  private readonly maxBytes: number;

  constructor(sink: FrameSink, maxBytes = MAX_FRAME_BYTES) {
    this.sink = sink;
    this.maxBytes = maxBytes;
  }

  push(chunk: Buffer): void {
    let at = 0;
    while (at < chunk.length) {
      const nl = chunk.indexOf(0x0a, at);
      const end = nl === -1 ? chunk.length : nl;
      if (!this.skipping) {
        const piece = chunk.subarray(at, end);
        if (this.size + piece.length > this.maxBytes) {
          this.parts = [];
          this.size = 0;
          this.skipping = true;
          this.sink.tooLarge();
        } else {
          this.parts.push(piece);
          this.size += piece.length;
        }
      }
      if (nl === -1) return;
      if (!this.skipping) {
        const text = Buffer.concat(this.parts, this.size).toString('utf8');
        this.parts = [];
        this.size = 0;
        if (text.trim() !== '') this.sink.frame(text);
      }
      this.skipping = false;
      at = nl + 1;
    }
  }
}
