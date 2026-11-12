/** ACP framing: one JSON object per `\n`-terminated UTF-8 line. */

/** A line longer than this is a protocol error. */
export const MAX_LINE_BYTES = 32 * 1024 * 1024;

export class FramingError extends Error {
  constructor() {
    super('ACP line too long');
    this.name = 'FramingError';
  }
}

/**
 * Bytes → lines. Splits on 0x0A over Buffer chunks, so a multi-byte character split across chunks is decoded whole.
 * After a {@link FramingError} the splitter is dead: every later push throws again.
 */
export class LineSplitter {
  private parts: Buffer[] = [];
  private size = 0;
  private failed = false;
  private readonly onLine: (line: string) => void;
  private readonly maxLineBytes: number;

  constructor(onLine: (line: string) => void, maxLineBytes = MAX_LINE_BYTES) {
    this.onLine = onLine;
    this.maxLineBytes = maxLineBytes;
  }

  push(chunk: Buffer): void {
    if (this.failed) throw new FramingError();
    let start = 0;
    while (start < chunk.length) {
      const nl = chunk.indexOf(0x0a, start);
      if (nl === -1) {
        this.hold(chunk.subarray(start));
        return;
      }
      this.hold(chunk.subarray(start, nl));
      const line = this.parts.length === 1 ? this.parts[0]! : Buffer.concat(this.parts, this.size);
      this.parts = [];
      this.size = 0;
      start = nl + 1;
      this.onLine(line.toString('utf8'));
    }
  }

  /** Bytes after the last newline are a torn write and are dropped. */
  end(): void {
    this.parts = [];
    this.size = 0;
  }

  private hold(part: Buffer): void {
    if (part.length === 0) return;
    this.size += part.length;
    if (this.size > this.maxLineBytes) {
      this.failed = true;
      this.parts = [];
      this.size = 0;
      throw new FramingError();
    }
    this.parts.push(part);
  }
}

/** One outgoing frame. */
export function encodeFrame(message: unknown): string {
  return `${JSON.stringify(message)}\n`;
}
