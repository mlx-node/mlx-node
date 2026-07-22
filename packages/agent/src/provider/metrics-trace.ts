/**
 * `MetricsTrace` — always-on per-turn inference telemetry, appended as JSON
 * Lines to `$HOME/.mlx-node/metrics/traces/<YYYY-MM-DD>-<pid>.jsonl`.
 *
 * This is a durable sink that complements the transient in-memory
 * {@link ./performance-status.ts} WeakMap (which only feeds the live TUI
 * footer). One record is written per successful inference turn so the
 * dashboard can correlate throughput, cache reuse, and cold-tier deltas back
 * to the pi session that produced them via `mlxTraceId`.
 *
 * Contract:
 *   - Default-on; the `MLX_AGENT_METRICS` env var set to `0` / `false` / `off`
 *     (case-insensitive) is the only kill switch.
 *   - `record()` NEVER throws: every field is allowlisted (no free text ever
 *     lands on disk) and all fs work is wrapped — telemetry must never break
 *     an inference turn.
 */

import { appendFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';

import { metricsTraceDir } from '../paths.js';

/**
 * One inference turn's telemetry. Every field is a number, a small
 * enumerated string (`finishReason`), or an identifier — never model output
 * text, tool arguments, or prompt content.
 */
export interface MetricsTraceRecord {
  /** Schema version. */
  v: 1;
  /** Join key minted by `TurnEmitter`; also stamped on the pi message as `mlxTraceId`. */
  traceId: string;
  /** Wall-clock write time (ms since epoch). */
  ts: number;
  /** pi per-request session id (parent vs each subagent differ). */
  sessionId?: string;
  /** Root pi session id the turn was SUBMITTED under (snapshotted at submit). */
  rootSessionId?: string;
  /** Root pi session JSONL file path the turn was SUBMITTED under (snapshotted at submit). */
  rootSessionFile?: string;
  /** Model id served this turn (`mlx/<dir-name>`'s `<dir-name>`). */
  model: string;
  /** Turn duration (ms) bracketing resident selection + prefill + decode. */
  durationMs: number;
  /**
   * Queue + cold-load wait (ms) before native work began this turn — the gap
   * between turn submission and the serialized inference callback firing.
   * Subtract from `durationMs` to isolate execution-only latency.
   */
  queueMs?: number;
  /** `true` when the model was already warm/resident, `false` on a cold load/swap. */
  resident?: boolean;
  /** Native finish reason (`stop` / `length` / `tool_calls` / …). */
  finishReason: string;
  promptTokens: number;
  cachedTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  ttftMs?: number;
  prefillTps?: number;
  decodeTps?: number;
  mtpCycles?: number;
  mtpMeanAccepted?: number;
  /** Cold-tier restore hits accrued this turn (synchronous counter — exact). */
  coldHits?: number;
  /** Cold-tier lookup misses accrued this turn (synchronous counter — exact). */
  coldMisses?: number;
  /**
   * Cold-tier bytes committed this turn. Advances on an async writer thread,
   * so this delta is APPROXIMATE — it may attribute a prior turn's flush to
   * the turn that observes it.
   */
  coldBytesWritten?: number;
  /** Cold-tier bytes read back on validated hits this turn (synchronous — exact). */
  coldBytesRestored?: number;
}

type MetricsTraceInput = Omit<MetricsTraceRecord, 'v'>;

function envDisabled(): boolean {
  const raw = process.env.MLX_AGENT_METRICS;
  if (raw === undefined) return false;
  const normalized = raw.trim().toLowerCase();
  return normalized === '0' || normalized === 'false' || normalized === 'off';
}

/** A finite number, or `undefined` if the value is absent / non-finite. */
function finite(value: number | undefined): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

export class MetricsTrace {
  readonly enabled: boolean;
  private readonly dir: string;
  private readonly now: () => number;

  constructor(opts?: { dir?: string; now?: () => number }) {
    this.enabled = !envDisabled();
    this.dir = opts?.dir ?? metricsTraceDir();
    this.now = opts?.now ?? Date.now;
  }

  /** `<dir>/<YYYY-MM-DD>-<pid>.jsonl` — UTC date so rotation is timezone-stable. */
  currentFile(): string {
    const date = new Date(this.now()).toISOString().slice(0, 10);
    return join(this.dir, `${date}-${process.pid}.jsonl`);
  }

  /**
   * Append one allowlisted JSON line. Never throws: a broken sink must not
   * surface into the inference path. Excess input properties are dropped — the
   * record is rebuilt field by field so free text can never reach disk.
   */
  record(rec: MetricsTraceInput): void {
    if (!this.enabled) return;
    try {
      const out: MetricsTraceRecord = {
        v: 1,
        traceId: rec.traceId,
        ts: rec.ts,
        model: rec.model,
        durationMs: rec.durationMs,
        finishReason: rec.finishReason,
        promptTokens: rec.promptTokens,
        cachedTokens: rec.cachedTokens,
        outputTokens: rec.outputTokens,
        reasoningTokens: rec.reasoningTokens,
      };
      if (rec.sessionId !== undefined) out.sessionId = rec.sessionId;
      if (rec.rootSessionId !== undefined) out.rootSessionId = rec.rootSessionId;
      if (rec.rootSessionFile !== undefined) out.rootSessionFile = rec.rootSessionFile;
      const queueMs = finite(rec.queueMs);
      if (queueMs !== undefined) out.queueMs = queueMs;
      if (typeof rec.resident === 'boolean') out.resident = rec.resident;
      const ttftMs = finite(rec.ttftMs);
      if (ttftMs !== undefined) out.ttftMs = ttftMs;
      const prefillTps = finite(rec.prefillTps);
      if (prefillTps !== undefined) out.prefillTps = prefillTps;
      const decodeTps = finite(rec.decodeTps);
      if (decodeTps !== undefined) out.decodeTps = decodeTps;
      const mtpCycles = finite(rec.mtpCycles);
      if (mtpCycles !== undefined) out.mtpCycles = mtpCycles;
      const mtpMeanAccepted = finite(rec.mtpMeanAccepted);
      if (mtpMeanAccepted !== undefined) out.mtpMeanAccepted = mtpMeanAccepted;
      const coldHits = finite(rec.coldHits);
      if (coldHits !== undefined) out.coldHits = coldHits;
      const coldMisses = finite(rec.coldMisses);
      if (coldMisses !== undefined) out.coldMisses = coldMisses;
      const coldBytesWritten = finite(rec.coldBytesWritten);
      if (coldBytesWritten !== undefined) out.coldBytesWritten = coldBytesWritten;
      const coldBytesRestored = finite(rec.coldBytesRestored);
      if (coldBytesRestored !== undefined) out.coldBytesRestored = coldBytesRestored;

      const file = this.currentFile();
      mkdirSync(dirname(file), { recursive: true });
      appendFileSync(file, `${JSON.stringify(out)}\n`);
    } catch {
      // Telemetry is best-effort; a broken sink must never break inference.
    }
  }
}
