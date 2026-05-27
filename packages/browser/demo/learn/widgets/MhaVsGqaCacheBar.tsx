import * as React from "react";

/**
 * A horizontal bar chart comparing whole-model KV cache memory between a
 * hypothetical MHA Qwen3.5-0.8B and the actual GQA layout, at a fixed 8k
 * context length. All numbers derived from the model's config below.
 */

const NUM_LAYERS = 24;
const NUM_HEADS = 8;
const NUM_KV_HEADS = 2;
const HEAD_DIM = 256;
const BYTES_PER_FLOAT = 2; // bf16
const SEQ_LEN = 8192;

function cacheBytes(kvHeads: number): number {
  return NUM_LAYERS * 2 * kvHeads * HEAD_DIM * SEQ_LEN * BYTES_PER_FLOAT;
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

const MHA_BYTES = cacheBytes(NUM_HEADS);
const GQA_BYTES = cacheBytes(NUM_KV_HEADS);
const RATIO = GQA_BYTES > 0 ? MHA_BYTES / GQA_BYTES : 0;

type Row = {
  label: string;
  detail: string;
  bytes: number;
  color: string;
  accent: boolean;
};

const ROWS: Row[] = [
  {
    label: "MHA hypothetical",
    detail: `num_kv_heads = ${NUM_HEADS} (same as query heads)`,
    bytes: MHA_BYTES,
    color: "#94a3b8",
    accent: false,
  },
  {
    label: "GQA actual",
    detail: `num_kv_heads = ${NUM_KV_HEADS} (Qwen3.5-0.8B config)`,
    bytes: GQA_BYTES,
    color: "#22c55e",
    accent: true,
  },
];

export function MhaVsGqaCacheBar() {
  const max = Math.max(...ROWS.map((r) => r.bytes), 1);
  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          KV cache · whole model · seq_len = {SEQ_LEN.toLocaleString()}
        </div>
        <div className="text-[11px] text-muted-foreground">
          24 layers · head_dim = {HEAD_DIM} · bf16
        </div>
      </div>

      <div className="space-y-2">
        {ROWS.map((r, i) => {
          const pct = (r.bytes / max) * 100;
          return (
            <div key={`row-${i}`} className="space-y-1">
              <div className="flex items-baseline justify-between text-[12px]">
                <span className="font-mono text-foreground/90">{r.label}</span>
                <span className="font-mono text-foreground/80">
                  {formatBytes(r.bytes)}
                </span>
              </div>
              <div
                className="relative h-6 w-full overflow-hidden rounded-sm bg-muted/60"
                role="img"
                aria-label={`${r.label}: ${formatBytes(r.bytes)}`}
              >
                <div
                  className={[
                    "absolute inset-y-0 left-0 transition-all",
                    r.accent ? "ring-1 ring-primary/50" : "",
                  ].join(" ")}
                  style={{
                    width: `${pct.toFixed(1)}%`,
                    backgroundColor: r.color,
                  }}
                />
              </div>
              <div className="text-[11px] text-muted-foreground">{r.detail}</div>
            </div>
          );
        })}
      </div>

      <div className="rounded-md border border-primary/40 bg-primary/5 px-3 py-2 text-[12px] text-foreground/90">
        GQA is <span className="font-mono">{RATIO.toFixed(1)}×</span> smaller
        than the equivalent MHA layout — that&apos;s the whole reason it ships
        in every modern open model. Hidden behind every fast LLM inference is
        grouped-query attention.
      </div>
    </div>
  );
}
