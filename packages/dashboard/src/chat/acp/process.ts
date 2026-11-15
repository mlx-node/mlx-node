/**
 * The agent child process: stdout is protocol only, stderr stays in memory (never logged, never written, never sent
 * to the page), stdin writes queue behind backpressure, and stop runs a fixed ladder.
 */

import { spawn, type ChildProcess } from 'node:child_process';

import { AcpClient, type AcpClientHandlers, type ProtocolErrorReason } from './client.js';
import { FramingError, LineSplitter } from './framing.js';

/** A queued stdin backlog over this many bytes is a protocol error. */
export const MAX_WRITE_QUEUE_BYTES = 8 * 1024 * 1024;
export const STDERR_LINES = 200;
export const STDERR_LINE_CHARS = 2_000;
/** Stop ladder: stdin EOF, then SIGTERM after this long, then SIGKILL after {@link STOP_TERM_MS}. */
export const STOP_EOF_MS = 1_500;
export const STOP_TERM_MS = 1_000;

export interface AgentCommand {
  command: string;
  args: readonly string[];
  env: Record<string, string>;
  cwd: string;
}

export interface AgentExit {
  code: number | null;
  signal: NodeJS.Signals | null;
  /** True when the exit followed {@link AgentProcess.stop} or {@link AgentProcess.kill}. */
  asked: boolean;
  /** The spawn error code (`ENOENT`, …) when the process never started. */
  errorCode: string | null;
  /** The process was killed because it broke the protocol. */
  protocolError: boolean;
}

export type SpawnAgent = (command: AgentCommand) => ChildProcess;

export const spawnAgent: SpawnAgent = (command) =>
  spawn(command.command, [...command.args], {
    cwd: command.cwd,
    env: command.env,
    stdio: ['pipe', 'pipe', 'pipe'],
    detached: false,
  });

/** Keeps the last {@link STDERR_LINES} lines, each cut at {@link STDERR_LINE_CHARS}. */
class StderrRing {
  private readonly lines: string[] = [];
  private partial = '';

  push(text: string): void {
    let start = 0;
    for (;;) {
      const nl = text.indexOf('\n', start);
      if (nl === -1) {
        this.partial = (this.partial + text.slice(start)).slice(0, STDERR_LINE_CHARS);
        return;
      }
      this.add(this.partial + text.slice(start, nl));
      this.partial = '';
      start = nl + 1;
    }
  }

  snapshot(): string[] {
    return this.partial === '' ? [...this.lines] : [...this.lines, this.partial];
  }

  private add(line: string): void {
    this.lines.push(line.slice(0, STDERR_LINE_CHARS));
    if (this.lines.length > STDERR_LINES) this.lines.shift();
  }
}

export interface AgentProcessHandlers extends Omit<AcpClientHandlers, 'onProtocolError'> {
  onProtocolError?: (reason: ProtocolErrorReason) => void;
  onExit?: (exit: AgentExit) => void;
}

export class AgentProcess {
  readonly client: AcpClient;
  readonly exited: Promise<AgentExit>;
  private readonly child: ChildProcess;
  private readonly stderr = new StderrRing();
  private readonly queue: string[] = [];
  private queuedBytes = 0;
  private draining = false;
  private asked = false;
  private done = false;
  private protocolFailed = false;

  constructor(command: AgentCommand, handlers: AgentProcessHandlers = {}, spawnFn: SpawnAgent = spawnAgent) {
    const protocolError = (reason: ProtocolErrorReason): void => {
      if (this.protocolFailed || this.done) return;
      this.protocolFailed = true;
      handlers.onProtocolError?.(reason);
      // Nothing the peer sends after the verdict is read: the exit may come late on a loaded machine.
      this.client.close();
      void this.kill();
    };
    this.client = new AcpClient(
      { write: (frame) => this.write(frame, protocolError) },
      { onNotification: handlers.onNotification, onRequest: handlers.onRequest, onProtocolError: protocolError },
    );
    this.child = spawnFn(command);
    const splitter = new LineSplitter((line) => this.client.receive(line));
    this.child.stdout?.on('data', (chunk: Buffer) => {
      try {
        splitter.push(chunk);
      } catch (error) {
        protocolError(error instanceof FramingError ? 'framing' : 'unreadable');
      }
    });
    this.child.stdout?.on('end', () => splitter.end());
    this.child.stderr?.setEncoding('utf8');
    this.child.stderr?.on('data', (text: string) => this.stderr.push(text));
    this.child.stdin?.on('error', () => {});
    this.child.stdin?.on('drain', () => this.flush());
    this.exited = new Promise<AgentExit>((resolve) => {
      let settled = false;
      let errorCode: string | null = null;
      const finish = (code: number | null, signal: NodeJS.Signals | null): void => {
        if (settled) return;
        settled = true;
        this.done = true;
        this.client.close();
        const exit = { code, signal, asked: this.asked, errorCode, protocolError: this.protocolFailed };
        handlers.onExit?.(exit);
        resolve(exit);
      };
      // `close`, not `exit`: the last stdout lines are read before the client is closed.
      this.child.once('close', finish);
      this.child.once('error', (error: NodeJS.ErrnoException) => {
        errorCode = typeof error.code === 'string' ? error.code : null;
        finish(null, null);
      });
    });
  }

  get pid(): number | undefined {
    return this.child.pid;
  }

  get running(): boolean {
    return !this.done;
  }

  /** The peer broke the protocol: the client is closed and the process is on its way out. */
  get brokeProtocol(): boolean {
    return this.protocolFailed;
  }

  /** The kept stderr lines, for tests and diagnostics in memory only. */
  stderrTail(): string[] {
    return this.stderr.snapshot();
  }

  /** stdin EOF, wait, SIGTERM, wait, SIGKILL. Resolves on exit. */
  async stop(eofMs = STOP_EOF_MS, termMs = STOP_TERM_MS): Promise<AgentExit> {
    this.asked = true;
    if (!this.done) {
      this.child.stdin?.end();
      if (!(await this.exitWithin(eofMs))) {
        this.signal('SIGTERM');
        if (!(await this.exitWithin(termMs))) this.signal('SIGKILL');
      }
    }
    return this.exited;
  }

  /** The ladder from SIGTERM. `protocolError` marks the exit as caused by a peer that broke the protocol. */
  async kill(termMs = STOP_TERM_MS, protocolError = false): Promise<AgentExit> {
    this.asked = true;
    if (protocolError) this.protocolFailed = true;
    if (!this.done) {
      this.signal('SIGTERM');
      if (!(await this.exitWithin(termMs))) this.signal('SIGKILL');
    }
    return this.exited;
  }

  private signal(name: NodeJS.Signals): void {
    try {
      this.child.kill(name);
    } catch {
      // Already gone.
    }
  }

  private exitWithin(ms: number): Promise<boolean> {
    if (this.done) return Promise.resolve(true);
    return new Promise((resolve) => {
      const timer = setTimeout(() => resolve(false), ms);
      void this.exited.then(() => {
        clearTimeout(timer);
        resolve(true);
      });
    });
  }

  private write(frame: string, onOverflow: (reason: ProtocolErrorReason) => void): void {
    const stdin = this.child.stdin;
    if (this.done || stdin === null || stdin.destroyed || !stdin.writable) throw new Error('agent stdin closed');
    if (this.draining || this.queue.length > 0) {
      this.queue.push(frame);
      this.queuedBytes += Buffer.byteLength(frame);
      if (this.queuedBytes > MAX_WRITE_QUEUE_BYTES) onOverflow('write-queue');
      return;
    }
    if (!stdin.write(frame)) this.draining = true;
  }

  private flush(): void {
    this.draining = false;
    const stdin = this.child.stdin;
    if (stdin === null) return;
    while (this.queue.length > 0) {
      const frame = this.queue.shift()!;
      this.queuedBytes -= Buffer.byteLength(frame);
      if (!stdin.write(frame)) {
        this.draining = true;
        return;
      }
    }
  }
}
