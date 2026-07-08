import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Chapter 10 (lm-head) supplement — the tied-vs-untied parameter ledger.
 *
 * Static (no animation, no worker, no model): a two-column block that draws
 * the second copy of the embedding table that weight tying *doesn't* pay for.
 * Every number is verified against the Qwen3.5-0.8B checkpoint:
 *
 *   tied embedding   248,320 × 1,024 = 254,279,680  (254.3M)  ← bought once
 *   model total                       852,985,920   (29.8% is that one table)
 *   untied second copy                254,279,680             ← the hypothetical
 *   untied embed subtotal             508,559,360   (508.6M)
 *   untied model total                1,107,265,600 (~1.107B)
 *
 * GPT-3 footnote (history, dense transformer): GPT-3 also TIED its embeddings
 * (it inherited GPT-2's architecture, which computes logits from the transposed
 * token-embedding table). The doubled figure below is the hypothetical cost of
 * an *untied* table that size, not GPT-3's actual ledger:
 *   each embed/unembed copy  50,257 × 12,288 = 617,558,016  (617.6M)
 *   a hypothetical untied pair              = 1,235,116,032 (~1.24B on tables)
 */

const fmtFull = (n: number) => n.toLocaleString('en-US');

// ── Verified counts ──────────────────────────────────────────────────────────
const EMBED = 254_279_680; // 248,320 × 1,024
const TOTAL = 852_985_920; // checkpoint total
const UNTIED_EMBED = EMBED * 2; // 508,559,360
const UNTIED_TOTAL = TOTAL + EMBED; // 1,107,265,600
const EMBED_SHARE = ((EMBED / TOTAL) * 100).toFixed(1); // 29.8

const GPT3_COPY = 617_558_016; // 50,257 × 12,288
const GPT3_TWICE = GPT3_COPY * 2; // 1,235,116,032

const COPY = {
  en: {
    tiedHead: 'Tied (what Qwen3.5-0.8B ships)',
    untiedHead: 'Untied (hypothetical)',
    embedRow: 'embedding table',
    secondCopyRow: 'second copy (LM head)',
    embedSubtotal: 'embedding subtotal',
    modelTotal: 'model total',
    once: 'bought once, used twice',
    shareNote: `That one ${fmtFull(EMBED)}-float table is ${EMBED_SHARE}% of the whole model.`,
    footnote: (
      <>
        Scale sets the stakes. GPT-3&rsquo;s table was{' '}
        <span className="font-mono">50,257 × 12,288 = {fmtFull(GPT3_COPY)}</span> (≈617.6M), about 2.4× the size of this
        one — and GPT-3 tied its embeddings too. Leaving a table that big untied would have meant a second copy and ≈
        {fmtFull(GPT3_TWICE)} (~1.24B) parameters on lookup tables alone. The wider the vocabulary, the more tying saves,
        which is why it is standard from 0.8B up to 175B.
      </>
    ),
    barTiedLabel: 'tied',
    barUntiedLabel: 'untied',
  },
  zh: {
    tiedHead: '共享（Qwen3.5-0.8B 实际采用）',
    untiedHead: '不共享（假设）',
    embedRow: '嵌入表',
    secondCopyRow: '第二份拷贝（LM head）',
    embedSubtotal: '嵌入小计',
    modelTotal: '模型总计',
    once: '只买一份，用两次',
    shareNote: `这一张 ${fmtFull(EMBED)} 个浮点数的表，占了整个模型的 ${EMBED_SHARE}%。`,
    footnote: (
      <>
        规模决定利害。GPT-3 的这张表是{' '}
        <span className="font-mono">50,257 × 12,288 = {fmtFull(GPT3_COPY)}</span>（≈617.6M）——约是本模型这张表的 2.4
        倍——而且 GPT-3 同样共享了嵌入。若让这么大的一张表不共享，就要再添一份拷贝、约 {fmtFull(GPT3_TWICE)}（~1.24B）
        个参数全花在查表上。词表越宽，共享省得越多，所以从 0.8B 到 175B 它都是标准做法。
      </>
    ),
    barTiedLabel: '共享',
    barUntiedLabel: '不共享',
  },
} as const;

// ── Two-bar SVG geometry (totals, to scale) ──────────────────────────────────
const VB_W = 520;
const VB_H = 96;
const LABEL_W = 78;
const BAR_X = LABEL_W;
const BAR_W = VB_W - LABEL_W - 96;
const BAR_H = 24;
const TIED_Y = 14;
const UNTIED_Y = 54;
const SCALE = BAR_W / UNTIED_TOTAL; // untied total spans the full bar

export function TiedUntiedLedger() {
  const copy = COPY[useLocale()];

  const tiedW = TOTAL * SCALE;
  const tiedEmbedW = EMBED * SCALE;
  const untiedW = UNTIED_TOTAL * SCALE;
  const untiedEmbedW = UNTIED_EMBED * SCALE;

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Two totals to scale: tied vs untied. The lit portion is the embedding. */}
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-hidden="true">
        {/* tied row */}
        <text x={0} y={TIED_Y + BAR_H / 2 + 4} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
          {copy.barTiedLabel}
        </text>
        <rect
          x={BAR_X}
          y={TIED_Y}
          width={Math.max(tiedW, 1)}
          height={BAR_H}
          style={{ fill: 'var(--primary)', fillOpacity: 0.18, stroke: 'var(--border)' }}
          strokeWidth={1}
        />
        <rect
          x={BAR_X}
          y={TIED_Y}
          width={Math.max(tiedEmbedW, 1)}
          height={BAR_H}
          style={{ fill: 'var(--primary)', fillOpacity: 0.7, stroke: 'var(--border)' }}
          strokeWidth={1}
        />
        <text
          x={BAR_X + tiedW + 6}
          y={TIED_Y + BAR_H / 2 + 4}
          style={{ fill: 'var(--muted-foreground)' }}
          className="font-mono text-[10px]"
        >
          853M
        </text>

        {/* untied row */}
        <text x={0} y={UNTIED_Y + BAR_H / 2 + 4} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
          {copy.barUntiedLabel}
        </text>
        <rect
          x={BAR_X}
          y={UNTIED_Y}
          width={Math.max(untiedW, 1)}
          height={BAR_H}
          style={{ fill: 'var(--primary)', fillOpacity: 0.18, stroke: 'var(--border)' }}
          strokeWidth={1}
        />
        <rect
          x={BAR_X}
          y={UNTIED_Y}
          width={Math.max(untiedEmbedW, 1)}
          height={BAR_H}
          style={{ fill: 'var(--primary)', fillOpacity: 0.7, stroke: 'var(--border)' }}
          strokeWidth={1}
        />
        {/* divider marking where the duplicated copy begins */}
        <line
          x1={BAR_X + tiedEmbedW}
          y1={UNTIED_Y - 2}
          x2={BAR_X + tiedEmbedW}
          y2={UNTIED_Y + BAR_H + 2}
          style={{ stroke: 'var(--foreground)', strokeOpacity: 0.5 }}
          strokeWidth={1}
          strokeDasharray="2 2"
        />
        <text
          x={BAR_X + untiedW + 6}
          y={UNTIED_Y + BAR_H / 2 + 4}
          style={{ fill: 'var(--muted-foreground)' }}
          className="font-mono text-[10px]"
        >
          ~1.107B
        </text>
      </svg>

      {/* Side-by-side numeric ledger */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {/* Tied column */}
        <div className="rounded-md border border-border bg-card p-2.5">
          <div className="mb-1.5 text-xs font-semibold text-foreground">{copy.tiedHead}</div>
          <table className="w-full border-collapse text-[11px]">
            <tbody>
              <tr className="border-b border-border/40">
                <td className="py-1 pr-2 text-muted-foreground">{copy.embedRow}</td>
                <td className="py-1 text-right font-mono whitespace-nowrap">{fmtFull(EMBED)}</td>
              </tr>
              <tr className="border-b border-border/40">
                <td className="py-1 pr-2 text-muted-foreground">{copy.secondCopyRow}</td>
                <td className="py-1 text-right font-mono text-muted-foreground/70">{copy.once}</td>
              </tr>
              <tr className="font-medium">
                <td className="py-1 pr-2">{copy.modelTotal}</td>
                <td className="py-1 text-right font-mono whitespace-nowrap">{fmtFull(TOTAL)}</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Untied column */}
        <div className="rounded-md border border-dashed border-border bg-card p-2.5">
          <div className="mb-1.5 text-xs font-semibold text-muted-foreground">{copy.untiedHead}</div>
          <table className="w-full border-collapse text-[11px]">
            <tbody>
              <tr className="border-b border-border/40">
                <td className="py-1 pr-2 text-muted-foreground">{copy.embedRow}</td>
                <td className="py-1 text-right font-mono whitespace-nowrap">{fmtFull(EMBED)}</td>
              </tr>
              <tr className="border-b border-border/40">
                <td className="py-1 pr-2 text-muted-foreground">{copy.secondCopyRow}</td>
                <td className="py-1 text-right font-mono whitespace-nowrap">+{fmtFull(EMBED)}</td>
              </tr>
              <tr className="border-b border-border/40">
                <td className="py-1 pr-2 text-muted-foreground">{copy.embedSubtotal}</td>
                <td className="py-1 text-right font-mono whitespace-nowrap">{fmtFull(UNTIED_EMBED)}</td>
              </tr>
              <tr className="font-medium">
                <td className="py-1 pr-2">{copy.modelTotal}</td>
                <td className="py-1 text-right font-mono whitespace-nowrap">{fmtFull(UNTIED_TOTAL)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <p className="font-mono text-[11px] text-muted-foreground">{copy.shareNote}</p>
      <p className="text-[10px] text-muted-foreground/70">{copy.footnote}</p>
    </div>
  );
}
