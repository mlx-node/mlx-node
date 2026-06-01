import * as React from 'react';

import { SegmentedToggle } from '../scaffolding/SegmentedToggle';

/**
 * Chapter 15 supplement — the staggering scale gulf between pretraining data
 * and post-training (SFT) data.
 *
 * Pretraining sees trillions of *tokens* of generic text; SFT only needs a few
 * thousand to a few million curated *examples* to reshape behavior. The two
 * quantities are measured in different units — tokens vs conversations — so the
 * comparison is an order-of-magnitude scale story, NOT a literal apples-to-apples
 * bar. The toggle makes that explicit:
 *   • LINEAR: width ∝ raw count. Pretraining fills the track; the SFT bar is a
 *     sub-pixel sliver (≈0.00003% of the pretraining width) — so it's floored to
 *     a hairline min-width and disclosed, because the whole point is that SFT
 *     *vanishes* at linear scale.
 *   • LOG: width ∝ log10(count). Both bars become visible and the
 *     multiple-orders-of-magnitude gap reads clearly.
 *
 * Numbers are illustrative orders of magnitude (real corpora vary widely). The
 * bar idiom mirrors SwiGluVsPlain.
 */

type Scale = 'linear' | 'log';

// Illustrative, order-of-magnitude representative values (real corpora vary).
const PRETRAIN_TOKENS = 3_000_000_000_000; // ≈ 3T tokens
const SFT_EXAMPLES = 1_000_000; // ≈ 1M examples

// On the linear scale the SFT bar is sub-pixel, so floor it to a hairline so it
// stays visible — and disclose that we did so.
const LINEAR_FLOOR_PCT = 0.5;

const PRETRAIN_BG = 'oklch(0.7 0.13 220 / 0.32)';
const SFT_BG = 'oklch(0.7 0.13 60 / 0.4)';

const EASE = 'cubic-bezier(0.4, 0, 0.2, 1)';

function Bar({
  label,
  amount,
  widthPct,
  bg,
  ariaLabel,
  note,
}: {
  label: string;
  amount: string;
  widthPct: number;
  bg: string;
  ariaLabel: string;
  note?: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between text-[11px]">
        <span className="text-foreground/85">{label}</span>
        <span className="font-mono text-muted-foreground">{amount}</span>
      </div>
      <div
        className="flex h-9 w-full overflow-hidden rounded-md border border-border bg-muted/40"
        role="img"
        aria-label={ariaLabel}
      >
        <div
          className="flex h-full min-w-0 items-center overflow-hidden rounded-[3px] px-2"
          style={{ width: `${widthPct}%`, backgroundColor: bg, transition: `width 550ms ${EASE}` }}
        >
          {widthPct > 12 ? <span className="truncate font-mono text-[10px] text-foreground/90">{amount}</span> : null}
        </div>
      </div>
      {note ? <p className="text-[10px] text-muted-foreground">{note}</p> : null}
    </div>
  );
}

export function DataScaleBar() {
  const [scale, setScale] = React.useState<Scale>('linear');
  const log = scale === 'log';

  // LINEAR: width ∝ raw count, pretraining pinned to the full track. SFT is
  // sub-pixel (≈0.00003%), so floor it to a hairline and disclose the floor.
  const sftRawPct = (SFT_EXAMPLES / PRETRAIN_TOKENS) * 100; // ≈ 0.0000333
  const sftLinearPct = Math.max(sftRawPct, LINEAR_FLOOR_PCT);

  // LOG: width ∝ log10(count), pretraining pinned to the full track.
  const pretrainLog = Math.log10(PRETRAIN_TOKENS); // ≈ 12.48
  const sftLog = Math.log10(SFT_EXAMPLES); // 6
  const sftLogPct = (sftLog / pretrainLog) * 100; // ≈ 48.1

  const pretrainPct = 100;
  const sftPct = log ? sftLogPct : sftLinearPct;

  // Order-of-magnitude ratio (tokens vs examples → illustrative of scale only).
  const ordersOfMagnitude = Math.round(Math.log10(PRETRAIN_TOKENS / SFT_EXAMPLES)); // 6

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">Pretraining vs SFT — data scale</div>
        <SegmentedToggle
          value={scale}
          onChange={setScale}
          options={[
            { value: 'linear', label: 'Linear' },
            { value: 'log', label: 'Log' },
          ]}
        />
      </div>

      <Bar
        label="Pretraining (next-token prediction)"
        amount="≈ 3T tokens"
        widthPct={pretrainPct}
        bg={PRETRAIN_BG}
        ariaLabel="Pretraining: approximately 3 trillion tokens, filling the full width of the track."
      />

      <Bar
        label="Instruction tuning (SFT)"
        amount="≈ 1M examples"
        widthPct={sftPct}
        bg={SFT_BG}
        ariaLabel={
          log
            ? 'Instruction tuning: approximately 1 million examples, about 48 percent of the track on a log scale — six orders of magnitude smaller than pretraining.'
            : 'Instruction tuning: approximately 1 million examples, a sub-pixel sliver next to pretraining, floored to a hairline so it stays visible.'
        }
        note={
          log
            ? undefined
            : 'SFT bar floored to remain visible on the linear scale — its true width here is ≈ 0.00003% of the pretraining bar.'
        }
      />

      <div className="rounded-md border border-border bg-muted/20 p-2 text-[12px] text-foreground/85">
        <span className="font-mono">≈ 3T tokens</span> vs <span className="font-mono">≈ 1M examples</span> — about{' '}
        <span className="font-mono">10^{ordersOfMagnitude}×</span> apart in raw count. (Illustrative of scale only:
        those are different units — <em>tokens</em> of generic text vs curated <em>examples</em> — so this is not a
        strict apples-to-apples ratio.)
      </div>

      <p className="text-[12px] text-foreground/85">
        Pretraining digests <strong>trillions of tokens</strong> to build the model's knowledge; SFT needs only{' '}
        <strong>thousands to millions of curated examples</strong> to reshape behavior into a helpful assistant. On the{' '}
        <strong>linear</strong> scale the SFT data all but vanishes — that's the point. Switch to <strong>log</strong>{' '}
        to see the gap as the multiple-orders-of-magnitude story it really is. SFT is <em>tiny next to pretraining</em>,
        which is why post-training is comparatively cheap and fast.
      </p>

      <p className="text-[10px] text-muted-foreground">
        Illustrative orders of magnitude — real corpora vary widely. Tokens (pretraining) and examples (SFT) are
        different units, so the ratio reads as scale, not a literal apples-to-apples comparison.
      </p>
    </div>
  );
}
