import * as React from "react";

/**
 * Reusable top-K probability bar chart used by chapter 9 (Sampling) and the
 * Free Chat inspector drawer. Renders one row per candidate token with a
 * proportional bar; the row matching `sampledTokenId` is highlighted with the
 * primary accent. Rows with `prob === 0` (truncated by top-p) are dimmed.
 */
export type TopKBarsProps = {
  /** Token ids, in the same order as `probs` and `texts`. */
  ids: number[];
  /** Probability per id. Should sum to ~1 (or 0 if every entry was -Inf). */
  probs: number[];
  /** Decoded text per id. May contain leading whitespace markers. */
  texts: string[];
  /** Id of the token the model actually sampled; row is highlighted. */
  sampledTokenId: number;
};

export function renderTokenDisplay(raw: string): string {
  if (raw === "") return "∅";
  let text = raw.replace(/^Ġ/, " ").replace(/Ġ/g, " ");
  text = text.replace(/\n/g, "↵").replace(/\t/g, "→");
  if (text.startsWith(" ")) text = "·" + text.slice(1);
  if (text.length > 18) text = text.slice(0, 17) + "…";
  return text;
}

export function TopKBars({ ids, probs, texts, sampledTokenId }: TopKBarsProps) {
  const maxProb = Math.max(0.0001, ...probs);
  return (
    <div
      role="list"
      aria-label="Top-K token probabilities"
      className="space-y-1 rounded-md border border-border bg-background p-3"
    >
      {ids.map((id, i) => {
        const prob = probs[i] ?? 0;
        const text = texts[i] ?? "";
        const display = renderTokenDisplay(text);
        const isSampled = id === sampledTokenId;
        const pct = (prob / maxProb) * 100;
        const truncated = prob === 0;
        return (
          <div
            key={`${id}-${i}`}
            role="listitem"
            className={[
              "grid grid-cols-[8rem_minmax(0,1fr)_4rem] items-center gap-2 rounded px-1.5 py-1 text-[12px]",
              isSampled ? "bg-primary/10" : "",
              truncated ? "opacity-40" : "",
            ].join(" ")}
            title={`id ${id} · ${JSON.stringify(text)}`}
          >
            <span
              className={[
                "truncate font-mono",
                isSampled ? "text-primary font-semibold" : "text-foreground/80",
              ].join(" ")}
            >
              {display}
            </span>
            <div className="relative h-4 w-full overflow-hidden rounded bg-muted">
              <div
                className={[
                  "h-full",
                  isSampled ? "bg-primary" : "bg-foreground/40",
                ].join(" ")}
                style={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
              />
            </div>
            <span className="text-right font-mono text-muted-foreground">
              {prob >= 0.0005 ? prob.toFixed(3) : prob > 0 ? "<0.001" : "—"}
            </span>
          </div>
        );
      })}
    </div>
  );
}
