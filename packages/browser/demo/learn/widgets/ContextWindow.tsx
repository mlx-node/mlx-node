import * as React from 'react';

/**
 * Chapter 16 (architecture) — the finite context window. A slider sets how long
 * the conversation has grown; the bar splits into the most-recent WINDOW tokens
 * the model can still see (bright) and everything older that has scrolled out of
 * view (dim). Drag past the window size to watch the model "forget" the start.
 */

const WINDOW = 262_144; // Qwen3.5-0.8B max_position_embeddings
const SLIDER_MAX = WINDOW * 2;

const fmt = (n: number) => Math.round(n).toLocaleString();

export function ContextWindow() {
  const [len, setLen] = React.useState(120_000);

  const inWindow = Math.min(len, WINDOW);
  const forgotten = Math.max(0, len - WINDOW);
  const forgottenPct = (forgotten / SLIDER_MAX) * 100;
  const inWindowPct = (inWindow / SLIDER_MAX) * 100;

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">The context window</div>
        <div className="font-mono text-[11px] text-muted-foreground">window = {fmt(WINDOW)} tokens</div>
      </div>

      <label className="block text-[11px] text-muted-foreground">
        Conversation so far: <span className="font-mono text-foreground/90">{fmt(len)} tokens</span>
        <input
          type="range"
          min={0}
          max={SLIDER_MAX}
          step={1000}
          value={len}
          onChange={(e) => setLen(Number(e.target.value))}
          className="mt-1 w-full accent-primary"
          aria-label="Conversation length in tokens"
        />
      </label>

      <div
        className="flex h-7 w-full overflow-hidden rounded-md border border-border bg-muted/30"
        role="img"
        aria-label={`${fmt(forgotten)} tokens scrolled out of the context window; ${fmt(inWindow)} tokens still visible.`}
      >
        <div
          className="h-full bg-muted-foreground/25"
          style={{ width: `${forgottenPct}%`, transition: 'width 120ms ease' }}
          title="Scrolled out of context"
        />
        <div
          className="h-full bg-primary/30"
          style={{ width: `${inWindowPct}%`, transition: 'width 120ms ease' }}
          title="In context"
        />
      </div>

      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
        <span className="text-foreground/85">
          <span className="mr-1 inline-block h-2 w-2 rounded-sm bg-primary/40 align-middle" />
          In context: <span className="font-mono">{fmt(inWindow)}</span>
        </span>
        <span className={forgotten > 0 ? 'text-foreground/85' : 'text-muted-foreground'}>
          <span className="mr-1 inline-block h-2 w-2 rounded-sm bg-muted-foreground/40 align-middle" />
          Forgotten: <span className="font-mono">{fmt(forgotten)}</span>
        </span>
      </div>

      <p className="text-[12px] text-foreground/85">
        The model can only attend to tokens inside the window — it's a hard ceiling, not a sliding memory. To carry a
        conversation past <span className="font-mono">{fmt(WINDOW)}</span> tokens you have to drop the oldest ones to
        make room, and anything dropped is gone unless it's re-supplied in a later prompt. There is no long-term store;
        the window is the model's <em>entire</em> working memory.
      </p>
    </div>
  );
}
