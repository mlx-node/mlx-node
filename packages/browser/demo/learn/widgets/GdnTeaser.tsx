import * as React from 'react';

/**
 * Chapter 9 supplement — a per-layer teaser for the hybrid attention recipe.
 *
 * One slider sweeps the context length. Two stacked bars contrast a single
 * layer's memory cost as that context grows:
 *   • A GatedDeltaNet (linear) layer folds history into a FIXED-SIZE recurrent
 *     state — its bar never moves, no matter how long the context.
 *   • A full-attention layer keeps a KV cache that GROWS linearly with seq_len.
 *
 * That contrast is the whole reason Qwen3.5-0.8B is a 3:1 hybrid (18 GDN : 6
 * full): only 6 of 24 layers pay the growing-cache cost. Chapter 12 digs into
 * the recipe. This is deliberately NOT KvGrowthCurve's whole-model log-log
 * plot — it is the simpler single-layer fixed-box-vs-growing-bar picture.
 *
 * All numbers are formula-derived from the same architecture constants that
 * KvGrowthCurve uses (kept in sync so the two widgets never drift). No model,
 * no worker, no network.
 */

// Architecture constants — kept identical to KvGrowthCurve.tsx so the two
// widgets report the same bytes for the same context length.
const NUM_LAYERS = 24;
const NUM_KV_HEADS = 2;
const HEAD_DIM = 256;
const BYTES_PER_FLOAT = 2; // bf16
const FULL_LAYER_INTERVAL = 4;
const NUM_FULL_LAYERS = NUM_LAYERS / FULL_LAYER_INTERVAL; // 6
const NUM_LINEAR_LAYERS = NUM_LAYERS - NUM_FULL_LAYERS; // 18

// GatedDeltaNet recurrent state — fixed regardless of sequence length.
const LINEAR_NUM_HEADS = 16;
const LINEAR_HEAD_DIM = 128;
const LINEAR_STATE_BYTES = LINEAR_NUM_HEADS * LINEAR_HEAD_DIM * LINEAR_HEAD_DIM * BYTES_PER_FLOAT;

// One full-attention layer's KV cache grows by this many bytes per token:
// 2 (K&V) × kv_heads × head_dim × bf16.
const FULL_KV_BYTES_PER_TOKEN = 2 * NUM_KV_HEADS * HEAD_DIM * BYTES_PER_FLOAT;

function fullKvBytes(seqLen: number): number {
  return seqLen * FULL_KV_BYTES_PER_TOKEN;
}

// Discrete sequence-length stops. Top end matches Qwen3.5's max_position.
const SEQ_STOPS = [256, 1024, 4096, 16384, 65536, 262144] as const;
const MAX_STOP_INDEX = SEQ_STOPS.length - 1;
// Default to a middle stop so both bars are legible on first paint.
const DEFAULT_STOP_INDEX = 3;
// The growing full-attn KV cache spans ~1000× (512 KB at 256 tokens → 512 MB at
// max context), far too wide for a linear bar to show both the tiny fixed state
// and the growth. So both bars are LOG-scaled. The log floor sits a little below
// the fixed state, so the constant GatedDeltaNet bar renders as a short-but-
// visible fixed reference while the full-attn bar grows above it.
const MAX_FULL_KV_BYTES = fullKvBytes(SEQ_STOPS[MAX_STOP_INDEX]);
const LOG_FLOOR = Math.log2(LINEAR_STATE_BYTES) - 2;
const LOG_CEIL = Math.log2(MAX_FULL_KV_BYTES);
function logPct(bytes: number): number {
  const p = ((Math.log2(bytes) - LOG_FLOOR) / (LOG_CEIL - LOG_FLOOR)) * 100;
  return Math.max(0, Math.min(100, p));
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '—';
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(n % 1_000 === 0 ? 0 : 1)}k`;
  return `${n}`;
}

function formatRatio(ratio: number): string {
  if (!Number.isFinite(ratio) || ratio <= 0) return '—';
  if (ratio >= 100) return ratio.toFixed(0);
  if (ratio >= 10) return ratio.toFixed(1);
  return ratio.toFixed(2);
}

export function GdnTeaser() {
  const [stopIndex, setStopIndex] = React.useState(DEFAULT_STOP_INDEX);
  const seqLen = SEQ_STOPS[stopIndex];

  const fullBytes = fullKvBytes(seqLen);
  const ratio = fullBytes / LINEAR_STATE_BYTES;

  // Both bars share one log scale (LOG_FLOOR…LOG_CEIL). The GatedDeltaNet bar is
  // constant; the full-attn bar grows with seq_len. At 256 tokens they are equal
  // (a full-attn layer is then also 512 KB) and the log scale shows that honestly
  // — no minimum-width floor that would fake them into looking equal at every stop.
  const gdnPct = logPct(LINEAR_STATE_BYTES);
  const fullPct = logPct(fullBytes);

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Per-layer memory · fixed state vs growing cache
        </div>
        <div className="text-[11px] text-muted-foreground">
          head_dim {HEAD_DIM} · {NUM_KV_HEADS} KV heads · bf16
        </div>
      </div>

      {/* Sequence-length slider over discrete stops. */}
      <div className="space-y-1">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <label htmlFor="gdn-teaser-seq" className="uppercase tracking-wider">
            sequence length (tokens)
          </label>
          <span className="font-mono text-foreground/80">
            {formatTokens(seqLen)} ({seqLen.toLocaleString()})
          </span>
        </div>
        <input
          id="gdn-teaser-seq"
          type="range"
          min={0}
          max={MAX_STOP_INDEX}
          step={1}
          value={stopIndex}
          onChange={(e) => setStopIndex(Number.parseInt(e.target.value, 10))}
          className="w-full accent-primary"
          aria-valuetext={`${seqLen.toLocaleString()} tokens`}
        />
        <div className="flex justify-between text-[10px] text-muted-foreground">
          {SEQ_STOPS.map((stop) => (
            <span key={stop} className="font-mono">
              {formatTokens(stop)}
            </span>
          ))}
        </div>
      </div>

      {/* Two contrasting bars: fixed GDN state vs growing full-attn KV cache. */}
      <div
        className="space-y-3"
        role="img"
        aria-label={`At ${seqLen.toLocaleString()} tokens, one full-attention layer's KV cache is ${formatBytes(
          fullBytes,
        )} and grows with context, while a GatedDeltaNet layer holds a fixed ${formatBytes(
          LINEAR_STATE_BYTES,
        )} recurrent state — a ${formatRatio(ratio)} times difference.`}
      >
        <div className="space-y-1">
          <div className="flex items-baseline justify-between text-[12px]">
            <span className="font-mono text-foreground/90">GatedDeltaNet layer — fixed-size state</span>
            <span className="font-mono text-foreground/80">{formatBytes(LINEAR_STATE_BYTES)}</span>
          </div>
          <div className="relative h-6 w-full overflow-hidden rounded-sm bg-muted/60">
            <div
              className="absolute inset-y-0 left-0 ring-1 ring-primary/50"
              style={{ width: `${gdnPct.toFixed(2)}%`, backgroundColor: '#22c55e' }}
            />
          </div>
          <div className="text-[11px] text-muted-foreground">
            Recurrent state — independent of sequence length (does not move as you drag the slider).
          </div>
        </div>

        <div className="space-y-1">
          <div className="flex items-baseline justify-between text-[12px]">
            <span className="font-mono text-foreground/90">Full-attention layer — KV cache</span>
            <span className="font-mono text-foreground/80">{formatBytes(fullBytes)}</span>
          </div>
          <div className="relative h-6 w-full overflow-hidden rounded-sm bg-muted/60">
            <div
              className="absolute inset-y-0 left-0 transition-all"
              style={{ width: `${fullPct.toFixed(2)}%`, backgroundColor: '#f59e0b' }}
            />
          </div>
          <div className="text-[11px] text-muted-foreground">
            <span className="font-mono">
              {seqLen.toLocaleString()} × {FULL_KV_BYTES_PER_TOKEN}
            </span>{' '}
            bytes/token — grows linearly with context.
          </div>
        </div>
      </div>

      <p className="text-[10px] text-muted-foreground">
        Bars use a log scale (512 KB → 512 MB) so the fixed state and the growing cache both fit; the byte counts above
        each bar are exact and grow linearly with tokens. At 256 tokens the two are equal.
      </p>

      {/* Readout comparing the two at the current seq_len. */}
      <div className="rounded-md border border-primary/40 bg-primary/5 px-3 py-2 text-[12px] text-foreground/90">
        At <span className="font-mono">{formatTokens(seqLen)}</span> tokens — full-attn KV:{' '}
        <span className="font-mono">{formatBytes(fullBytes)}</span> (grows with context) vs GDN state:{' '}
        <span className="font-mono">{formatBytes(LINEAR_STATE_BYTES)}</span> (constant). The full-attention layer needs{' '}
        <span className="font-mono">{formatRatio(ratio)}×</span> the memory of one GatedDeltaNet layer.
      </div>

      <p className="text-[12px] text-foreground/85">
        Only <strong>6 of 24</strong> layers pay this growing-cache cost; the other{' '}
        <strong>18 are GatedDeltaNet</strong> with constant memory. That 3:1 split ({NUM_LINEAR_LAYERS} linear :{' '}
        {NUM_FULL_LAYERS} full) is why a long context stays affordable. Chapter 12 (KV cache &amp; hybrid attention)
        digs into how this hybrid recipe keeps total memory in check.
      </p>

      <p className="text-[10px] text-muted-foreground">
        Illustrative — bytes are computed from Qwen3.5-0.8B&apos;s exact dimensions (head_dim 256, 2 KV heads) and the
        linear layer&apos;s fixed state size; not live output from the model.
      </p>
    </div>
  );
}
