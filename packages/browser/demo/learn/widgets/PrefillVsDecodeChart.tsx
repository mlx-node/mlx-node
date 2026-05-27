import * as React from "react";

/**
 * Two-bar comparison showing the time per phase for a hypothetical 1024
 * prompt-token + 256 decode-token generation. Visually contrasts prefill
 * (one big parallel matmul) with decode (many small sequential matmuls).
 */

const PROMPT_TOKENS = 1024;
const DECODE_TOKENS = 256;
const PREFILL_MS = 700;
const DECODE_PER_TOKEN_MS = 200;
const DECODE_TOTAL_MS = DECODE_PER_TOKEN_MS * DECODE_TOKENS;

function formatMs(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)} s`;
  return `${ms.toFixed(0)} ms`;
}

export function PrefillVsDecodeChart() {
  const total = PREFILL_MS + DECODE_TOTAL_MS;
  const prefillPct = (PREFILL_MS / total) * 100;
  const decodePct = (DECODE_TOTAL_MS / total) * 100;

  // Render the decode phase as a stack of equal-width segments to visually
  // sell "N small matmuls". Cap the visible segment count so a 256-token
  // run doesn't produce 256 unreadable hairlines.
  const visibleSegments = Math.min(DECODE_TOKENS, 32);

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Prefill vs decode · {PROMPT_TOKENS} prompt + {DECODE_TOKENS} decode
          tokens
        </div>
        <div className="text-[11px] text-muted-foreground">
          Illustrative ballpark numbers
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex items-baseline justify-between text-[12px]">
          <span className="font-mono text-foreground/90">Prefill</span>
          <span className="font-mono text-foreground/80">
            ~{formatMs(PREFILL_MS)} (one big matmul)
          </span>
        </div>
        <div
          className="relative h-8 w-full overflow-hidden rounded-sm bg-muted/60"
          role="img"
          aria-label={`prefill bar: ${formatMs(PREFILL_MS)} (${prefillPct.toFixed(1)}% of total)`}
        >
          <div
            className="absolute inset-y-0 left-0 bg-sky-500"
            style={{ width: `${prefillPct.toFixed(2)}%` }}
          />
          <div className="absolute inset-0 flex items-center justify-end pr-2 font-mono text-[10px] text-foreground/70">
            {prefillPct.toFixed(1)}%
          </div>
        </div>

        <div className="flex items-baseline justify-between pt-2 text-[12px]">
          <span className="font-mono text-foreground/90">Decode</span>
          <span className="font-mono text-foreground/80">
            ~{formatMs(DECODE_PER_TOKEN_MS)} × {DECODE_TOKENS} ={" "}
            {formatMs(DECODE_TOTAL_MS)}
          </span>
        </div>
        <div
          className="relative h-8 w-full overflow-hidden rounded-sm bg-muted/60"
          role="img"
          aria-label={`decode bar: ${formatMs(DECODE_TOTAL_MS)} (${decodePct.toFixed(1)}% of total) split into ${DECODE_TOKENS} per-token matmuls`}
        >
          <div
            className="absolute inset-y-0 left-0 flex"
            style={{ width: `${decodePct.toFixed(2)}%` }}
          >
            {Array.from({ length: visibleSegments }).map((_, i) => (
              <div
                key={`seg-${i}`}
                className="h-full flex-1 border-r border-background/70 bg-amber-500 last:border-r-0"
              />
            ))}
          </div>
          <div className="absolute inset-0 flex items-center justify-end pr-2 font-mono text-[10px] text-foreground/70">
            {decodePct.toFixed(1)}%
          </div>
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">
        Prefill is one parallel matmul; decode is N small sequential matmuls.
        That&apos;s why your first token comes fast and the rest stream.
      </p>

      <div className="grid grid-cols-2 gap-2 text-[11px]">
        <div className="rounded-md border border-border/60 bg-muted/20 p-2">
          <div className="text-muted-foreground">Time to first token</div>
          <div className="mt-1 font-mono text-foreground/90">
            ~{formatMs(PREFILL_MS + DECODE_PER_TOKEN_MS)}
          </div>
        </div>
        <div className="rounded-md border border-border/60 bg-muted/20 p-2">
          <div className="text-muted-foreground">Total wall time</div>
          <div className="mt-1 font-mono text-foreground/90">
            ~{formatMs(total)}
          </div>
        </div>
      </div>
    </div>
  );
}
