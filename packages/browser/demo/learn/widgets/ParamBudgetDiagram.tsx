import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Chapter 14 (architecture) supplement — where the ~0.8B parameters live.
 *
 * Static (no animation, no worker, no model): a segmented horizontal bar plus
 * a table, computed from the REAL tensor shapes in the Qwen3.5-0.8B
 * checkpoint's safetensors header. Every number below was verified against the
 * actual file: the per-segment sums add up to the checkpoint's exact total of
 * 852,985,920 parameters.
 *
 * Per-layer arithmetic (hidden 1024, SwiGLU intermediate 3584, vocab 248,320):
 *   embedding   248,320 × 1,024                                  = 254,279,680
 *   FFN /layer  3 × (1,024 × 3,584)                              =  11,010,048  × 24
 *   GQA /layer  q(2048→×2 gate)·k(512)·v(512)·o + 2 norms        =   7,340,544  ×  6
 *               (q_proj is 1024×4096 because attn_output_gate
 *               doubles it: 8 heads × 256 = 2048 for Q, 2048 gate)
 *   GDN /layer  qkv(1024×6144) + z(1024×2048) + out(2048×1024)
 *               + conv(6144×4) + a,b(2×16×1024) + tiny rest      =  10,543,264  × 18
 *   norms       24 layers × 2 RMSNorms × 1024 + final 1024       =      50,176
 *   vision      12-layer ViT encoder (unused in this text demo)  = 100,592,896
 */

type SegmentId = 'ffn' | 'embedding' | 'gdn' | 'attn' | 'norms' | 'vision';

type Segment = {
  id: SegmentId;
  /** Exact parameter count from the safetensors header. */
  params: number;
  /** Bar paint: fillOpacity on --primary, or the greyed vision segment. */
  fillOpacity: number;
  grey?: boolean;
};

// Numbers and paint only — the per-segment labels and written-out arithmetic
// live in COPY so they can localize. Every count is byte-identical per locale.
const SEGMENTS: Segment[] = [
  { id: 'ffn', params: 264_241_152, fillOpacity: 0.85 },
  { id: 'embedding', params: 254_279_680, fillOpacity: 0.6 },
  { id: 'gdn', params: 189_778_752, fillOpacity: 0.38 },
  { id: 'attn', params: 44_043_264, fillOpacity: 0.2 },
  { id: 'norms', params: 50_176, fillOpacity: 0.1 },
  { id: 'vision', params: 100_592_896, fillOpacity: 0.3, grey: true },
];

const TOTAL = SEGMENTS.reduce((a, s) => a + s.params, 0); // 852,985,920
const TEXT_TOTAL = TOTAL - SEGMENTS.find((s) => s.id === 'vision')!.params; // 752,393,024

const fmtM = (n: number) => (n / 1e6).toFixed(n >= 1e6 ? 1 : 2) + 'M';
const fmtFull = (n: number) => n.toLocaleString();
const pct = (n: number) => ((n / TOTAL) * 100).toFixed(1) + '%';

const COPY = {
  en: {
    header: <>The parameter budget — where the &ldquo;0.8B&rdquo; lives</>,
    totalReadout: (total: string) => `total ${total} ≈ 853M`,
    svgAria: `Segmented bar of Qwen3.5-0.8B's 852,985,920 parameters: SwiGLU FFNs 264.2 million, the tied embedding 254.3 million, Gated DeltaNet mixers 189.8 million, full attention 44.0 million, RMSNorms 0.05 million, and a vision encoder of 100.6 million that the text demo never runs.`,
    toScale: 'one bar = every parameter in the checkpoint, to scale',
    barLabels: {
      ffn: 'FFN 264M',
      embedding: 'embed 254M',
      gdn: 'GDN 190M',
      attn: 'attn 44M',
      norms: null,
      vision: 'vision 101M',
    } as Record<SegmentId, string | null>,
    textDecoder: (total: string) => `text decoder ${total} ≈ 752M`,
    plusVision: '+ vision 100.6M = 853M',
    thBlock: 'block',
    thMath: 'the arithmetic',
    thParams: 'params',
    thShare: 'share',
    labels: {
      ffn: 'SwiGLU FFNs × 24',
      embedding: 'Embedding (tied LM head)',
      gdn: 'Gated DeltaNet mixers × 18',
      attn: 'Full attention (GQA) × 6',
      norms: 'RMSNorms',
      vision: 'Vision encoder (off the text path)',
    } as Record<SegmentId, string>,
    math: {
      ffn: '3 × (1,024 × 3,584) = 11,010,048 per layer × 24',
      embedding: '248,320 × 1,024 — counted once: the LM head reuses it',
      gdn: 'qkv 6,291,456 + z 2,097,152 + out 2,097,152 + conv/β/decay 57,504 = 10,543,264 per layer × 18',
      attn: 'q 4,194,304 (gate doubles it) + k 524,288 + v 524,288 + o 2,097,152 + norms 512 = 7,340,544 per layer × 6',
      norms: '24 layers × 2 × 1,024 + final 1,024',
      vision: '12-layer ViT — in the checkpoint, never run in this text-only demo',
    } as Record<SegmentId, string>,
    totalRow: 'total',
    footnote: (
      <>
        Counts read straight from the checkpoint&apos;s tensor shapes — they sum to exactly {fmtFull(TOTAL)}. Two things
        to notice: weight tying means the 248,320 × 1,024 embedding table is bought once but used twice (token → vector
        at the bottom, vector → logits at the top), and the dense SwiGLU FFNs are the single biggest block — bigger than
        both sequence mixers combined ({fmtM(264_241_152)} vs {fmtM(233_822_016)} for GDN + attention).
      </>
    ),
  },
  zh: {
    header: <>参数预算——“0.8B”住在哪里</>,
    totalReadout: (total: string) => `总计 ${total} ≈ 853M`,
    svgAria: `把 Qwen3.5-0.8B 的 852,985,920 个参数画成一根分段条：SwiGLU FFN 2.642 亿，权重共享的嵌入 2.543 亿，Gated DeltaNet 混合器 1.898 亿，全量注意力 4,400 万，RMSNorm 5 万，还有一个文本演示从不运行的 1.006 亿参数视觉编码器。`,
    toScale: '一根条 = checkpoint 中的全部参数，按比例绘制',
    barLabels: {
      ffn: 'FFN 264M',
      embedding: '嵌入 254M',
      gdn: 'GDN 190M',
      attn: '注意力 44M',
      norms: null,
      vision: '视觉 101M',
    } as Record<SegmentId, string | null>,
    textDecoder: (total: string) => `文本解码器 ${total} ≈ 752M`,
    plusVision: '+ 视觉 100.6M = 853M',
    thBlock: '模块',
    thMath: '算式',
    thParams: '参数量',
    thShare: '占比',
    labels: {
      ffn: 'SwiGLU FFN ×24',
      embedding: '嵌入（权重共享 LM head）',
      gdn: 'Gated DeltaNet 混合器 ×18',
      attn: '全量注意力（GQA）×6',
      norms: 'RMSNorm',
      vision: '视觉编码器（不在文本路径上）',
    } as Record<SegmentId, string>,
    math: {
      ffn: '3 × (1,024 × 3,584) = 每层 11,010,048 × 24',
      embedding: '248,320 × 1,024——只计一次：LM head 复用它',
      gdn: 'qkv 6,291,456 + z 2,097,152 + out 2,097,152 + conv/β/衰减 57,504 = 每层 10,543,264 × 18',
      attn: 'q 4,194,304（gate 使其翻倍）+ k 524,288 + v 524,288 + o 2,097,152 + norm 512 = 每层 7,340,544 × 6',
      norms: '24 层 × 2 × 1,024 + 最终 1,024',
      vision: '12 层 ViT——在 checkpoint 里，但这个纯文本演示从不运行它',
    } as Record<SegmentId, string>,
    totalRow: '总计',
    footnote: (
      <>
        所有数字都直接读自 checkpoint 的张量形状——加起来恰好是 {fmtFull(TOTAL)}。两件值得注意的事：权重共享意味着那张
        248,320 × 1,024 的嵌入表只花一份钱却用了两次（底部 token → 向量，顶部向量 → logits）；而稠密 SwiGLU FFN
        是最大的单一块——比两种序列混合器加起来还大（{fmtM(264_241_152)} vs {fmtM(233_822_016)}，GDN + 注意力）。
      </>
    ),
  },
} as const;

// ── SVG bar geometry ─────────────────────────────────────────────────────────
const VB_W = 760;
const BAR_X = 8;
const BAR_W = VB_W - 16;
const BAR_Y = 26;
const BAR_H = 38;
const VB_H = BAR_Y + BAR_H + 22;

export function ParamBudgetDiagram() {
  const copy = COPY[useLocale()];
  // Segment x-offsets, proportional to the exact parameter counts.
  let acc = 0;
  const placed = SEGMENTS.map((s) => {
    const x = BAR_X + (acc / TOTAL) * BAR_W;
    const w = (s.params / TOTAL) * BAR_W;
    acc += s.params;
    return { ...s, x, w };
  });

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <div className="font-mono text-[11px] text-muted-foreground">{copy.totalReadout(fmtFull(TOTAL))}</div>
      </div>

      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.svgAria}>
        <text x={BAR_X} y={16} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
          {copy.toScale}
        </text>
        {placed.map((s) => (
          <rect
            key={s.id}
            x={s.x}
            y={BAR_Y}
            width={Math.max(s.w, 1)}
            height={BAR_H}
            style={{
              fill: s.grey ? 'var(--muted-foreground)' : 'var(--primary)',
              fillOpacity: s.fillOpacity,
              stroke: 'var(--border)',
            }}
            strokeWidth={1}
          />
        ))}
        {/* In-bar labels for the four big segments */}
        {placed
          .filter((s) => s.w > 40)
          .map((s) => (
            <text
              key={`lbl-${s.id}`}
              x={s.x + s.w / 2}
              y={BAR_Y + BAR_H / 2 + 4}
              textAnchor="middle"
              style={{ fill: 'var(--foreground)' }}
              className="font-mono text-[11px] font-semibold"
            >
              {copy.barLabels[s.id]}
            </text>
          ))}
        <text x={BAR_X} y={BAR_Y + BAR_H + 16} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
          {copy.textDecoder(fmtFull(TEXT_TOTAL))}
        </text>
        <text
          x={BAR_X + BAR_W}
          y={BAR_Y + BAR_H + 16}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[10px]"
        >
          {copy.plusVision}
        </text>
      </svg>

      {/* The arithmetic, segment by segment */}
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-[11px]">
          <thead>
            <tr className="border-b border-border text-left text-muted-foreground">
              <th className="py-1 pr-2 font-medium">{copy.thBlock}</th>
              <th className="py-1 pr-2 font-medium">{copy.thMath}</th>
              <th className="py-1 pr-2 text-right font-medium">{copy.thParams}</th>
              <th className="py-1 text-right font-medium">{copy.thShare}</th>
            </tr>
          </thead>
          <tbody>
            {SEGMENTS.map((s) => (
              <tr key={s.id} className={['border-b border-border/40', s.grey ? 'text-muted-foreground' : ''].join(' ')}>
                <td className="py-1 pr-2 whitespace-nowrap">
                  <span
                    className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm align-middle"
                    style={{
                      backgroundColor: s.grey ? 'var(--muted-foreground)' : 'var(--primary)',
                      opacity: s.fillOpacity,
                    }}
                  />
                  {copy.labels[s.id]}
                </td>
                <td className="py-1 pr-2 font-mono text-[10px] text-muted-foreground">{copy.math[s.id]}</td>
                <td className="py-1 pr-2 text-right font-mono whitespace-nowrap">{fmtM(s.params)}</td>
                <td className="py-1 text-right font-mono text-muted-foreground">{pct(s.params)}</td>
              </tr>
            ))}
            <tr className="font-medium">
              <td className="py-1 pr-2">{copy.totalRow}</td>
              <td className="py-1 pr-2 font-mono text-[10px] text-muted-foreground">
                264.2 + 254.3 + 189.8 + 44.0 + 0.05 + 100.6
              </td>
              <td className="py-1 pr-2 text-right font-mono whitespace-nowrap">{fmtM(TOTAL)}</td>
              <td className="py-1 text-right font-mono text-muted-foreground">100%</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p className="text-[10px] text-muted-foreground/70">{copy.footnote}</p>
    </div>
  );
}
