import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * GgufQuantLadder — the GGUF "Q-type" ladder you actually scroll through on
 * Hugging Face / Ollama / LM Studio, ordered by bits-per-weight. Each rung is a
 * clickable bar whose LENGTH = the type's PURE bits/weight; click → a detail
 * panel with the exact struct arithmetic, a one-line "what it is," and (for the
 * k-quants) the popular download *mix* and which tensors it bumps higher.
 *
 * Self-contained and STATIC: NO worker, NO model, NO WASM, NO network, NO
 * animation, NO Date.now / Math.random — pure React + Tailwind/SVG, so it
 * server-renders for crawlers (createRoot replaces it on boot). The only state
 * is the selected rung; the first render is deterministic server == client, so
 * it needs no prefers-reduced-motion handling.
 *
 * EVERY NUMBER IS SOURCE-VERIFIED — do not edit without a citation. Two kinds of
 * "bpw" exist and are NOT interchangeable; this widget shows the first:
 *   1. PURE-TYPE bpw = struct_bytes × 8 / weights_per_block — a property of the
 *      format. EXACT. This is the bar length and the headline number.
 *   2. EFFECTIVE-FILE bpw = file_bits / weights for a real model — ALWAYS higher
 *      (the _S/_M/_L mixes bump some tensors, and token_embd/output stay high).
 *      Model-dependent → shown only as the "≈ X on an 8B" mix note, never a bar.
 *
 * Pure-type bpw, derived from llama.cpp ggml/src/ggml-common.h (QK_K = 256,
 * ggml_half = 2 B, K_SCALE_SIZE = 12):
 *   Q2_K  scales[16]+qs[64]+d(2)+dmin(2)            =  84 B → 84×8/256  = 2.625
 *   Q3_K  hmask[32]+qs[64]+scales[12]+d(2)          = 110 B → 110×8/256 = 3.4375
 *   Q4_K  d(2)+dmin(2)+scales[12]+qs[128]           = 144 B → 144×8/256 = 4.5
 *   Q5_K  d(2)+dmin(2)+scales[12]+qh[32]+qs[128]    = 176 B → 176×8/256 = 5.5
 *   Q6_K  ql[128]+qh[64]+scales[16]+d(2)            = 210 B → 210×8/256 = 6.5625
 *   Q8_0  d(2)+qs[32]   (block of 32, not 256)      =  34 B → 34×8/32   = 8.5
 * (Q2_K is 2.625, NOT the 2.5625 from the original k-quants PR #1684 — the dmin
 * super-block min was added to the struct afterward. Use the live struct.)
 *
 * IQ ("I-quant") pure-type bpw are the published values in llama.cpp /
 * huggingface.js quant-descriptions.ts: IQ1_S 1.56, IQ2_XXS 2.06, IQ3_XXS 3.06,
 * IQ4_XS 4.25. I-quants store indices into a fixed codebook/lattice and are
 * meant to be paired with an importance matrix; below ~3 bpw they need one.
 *
 * The _S/_M/_L mixes (definitional, k-quants PR #1684): _S = uniform, _M / _L
 * bump attention.wv, attention.wo and feed_forward.w2 (the value + down-projection
 * tensors) to a higher type. Effective 8B file bpw (arXiv 2601.14277 sweep):
 * Q3_K_M ≈ 4.0, Q4_K_M ≈ 4.9 — both ABOVE their pure types, as expected.
 *
 * bf16 (16 bpw) is what THIS browser tab ships for both weights and the KV cache
 * — the top of the ladder, drawn as an off-scale cap, not a proportional bar.
 */

type Tier = 'floor' | 'low' | 'sweet' | 'high';
type Rung = {
  id: string;
  name: string; // English glossary — never translated
  bpw: number; // PURE-TYPE bits/weight
  approx?: boolean; // render "≈" (IQ types have no clean struct, published value)
  tier: Tier;
  deriv?: string; // struct arithmetic for k-quants (locale-independent)
};

// Ascending by pure-type bpw. Curated rungs span the real ladder: the IQ floor,
// the round k-quants, and the near-lossless top.
const RUNGS: Rung[] = [
  { id: 'iq1s', name: 'IQ1_S', bpw: 1.56, approx: true, tier: 'floor' },
  { id: 'iq2xxs', name: 'IQ2_XXS', bpw: 2.06, approx: true, tier: 'low' },
  { id: 'q2k', name: 'Q2_K', bpw: 2.625, tier: 'low', deriv: '84 B ÷ 256 w × 8' },
  { id: 'iq3xxs', name: 'IQ3_XXS', bpw: 3.06, approx: true, tier: 'low' },
  { id: 'q3k', name: 'Q3_K', bpw: 3.4375, tier: 'low', deriv: '110 B ÷ 256 w × 8' },
  { id: 'iq4xs', name: 'IQ4_XS', bpw: 4.25, approx: true, tier: 'sweet' },
  { id: 'q4k', name: 'Q4_K', bpw: 4.5, tier: 'sweet', deriv: '144 B ÷ 256 w × 8' },
  { id: 'q5k', name: 'Q5_K', bpw: 5.5, tier: 'sweet', deriv: '176 B ÷ 256 w × 8' },
  { id: 'q6k', name: 'Q6_K', bpw: 6.5625, tier: 'high', deriv: '210 B ÷ 256 w × 8' },
  { id: 'q80', name: 'Q8_0', bpw: 8.5, tier: 'high', deriv: '34 B ÷ 32 w × 8' },
];

const DEMO_NAME = 'bf16'; // the tab ships this — the off-scale cap
const DEMO_BPW = 16;
const DEFAULT_SEL = 'q4k'; // the universally-recommended default download

const MAX_BPW = 8.5; // Q8_0 fills the track; bf16 is drawn off-scale above

// Tier → bar fill. Only EMERALD / RED literal hexes are allowed; the rest are tokens.
const EMERALD = '#10b981'; // the 4–5.5 bpw sweet spot people actually download
const RED = '#ef4444'; // the sub-2-bit floor: only sane for giant MoEs
const TIER_FILL: Record<Tier, string> = {
  floor: RED,
  low: 'var(--muted-foreground)',
  sweet: EMERALD,
  high: 'var(--primary)',
};

// ── Per-locale copy. zh translates conceptual prose only; ALL glossary terms stay
// English (every Q-type / IQ name, bf16, attn_v, ffn_down, imatrix, bits/weight,
// MoE) and zh uses full-width punctuation （，、。：）. Numbers/units never change. ──
const COPY = {
  en: {
    heading: 'The GGUF Q-type ladder — pure bits per weight',
    youAreHere: 'bf16 = 16 bpw · the tab ships this (off the chart, ~2× the widest rung)',
    scaleNote: 'bar length = bits/weight (pure type) · click a rung',
    legendFloor: '1-bit floor',
    legendSweet: '4-bit sweet spot',
    rowAria: (n: string, b: string) => `Select ${n} — ${b} bits per weight`,
    svgAria: (n: string) =>
      `A ladder of GGUF quantization types ordered by pure bits-per-weight, from IQ1_S near 1.5 up to Q8_0 at 8.5; bf16 at 16 sits off the top. Showing ${n}.`,
    pureBpw: 'bits/weight (pure type)',
    deriv: 'struct',
    download: 'you download',
    facts: {
      iq1s: {
        what: 'The 1-bit-ish floor. An I-quant (codebook + importance matrix) — only sane for huge MoEs, and unusable without an imatrix.',
        mix: '',
      },
      iq2xxs: { what: 'A 2-bit I-quant: weights become indices into a fixed codebook, picked with an importance matrix.', mix: '' },
      q2k: { what: 'The smallest round k-quant. A super-block of 256 weights with a two-level scale — a block format, just like MX.', mix: 'usually as Q2_K_XL (Unsloth) — even Q2_K bumps attn_v & ffn_down to Q4_K' },
      iq3xxs: { what: 'A 3-bit I-quant — noticeably better quality than Q3_K at the same size, thanks to the codebook + imatrix.', mix: '' },
      q3k: { what: '3-bit k-quant super-block. Aggressive but coherent for bigger models.', mix: 'Q3_K_M ≈ 4.0 bpw on an 8B: attn_v, attn_o & ffn_down bumped to Q4_K' },
      iq4xs: { what: 'A 4-bit I-quant — among the best quality-per-byte rungs on the whole ladder.', mix: '' },
      q4k: { what: 'The default pick. 4-bit k-quant — the knee of the curve where quality loss becomes hard to notice.', mix: 'Q4_K_M ≈ 4.9 bpw on an 8B: half of attn_v & ffn_down bumped to Q6_K' },
      q5k: { what: '5-bit k-quant. A little more headroom than Q4_K for a little more disk.', mix: 'Q5_K_M: half of attn_v & ffn_down bumped to Q6_K (same recipe as Q4_K_M)' },
      q6k: { what: '6-bit k-quant. Near-lossless — the KL-divergence to the original is tiny.', mix: 'shipped pure (no mix) — already close enough to the source' },
      q80: { what: '8-bit, plus one scale per block of 32. Effectively lossless; the safe baseline.', mix: 'shipped pure — the reference everyone benchmarks against' },
    } as Record<string, { what: string; mix: string }>,
  },
  zh: {
    heading: 'GGUF Q-type 阶梯 —— 纯类型的 bits/weight',
    youAreHere: 'bf16 = 16 bpw · 这个标签页用的就是它（在图外，约为最宽那一档的 2 倍）',
    scaleNote: '条的长度 = bits/weight（纯类型）· 点一档看细节',
    legendFloor: '1-bit 地板',
    legendSweet: '4-bit 甜点区',
    rowAria: (n: string, b: string) => `选择 ${n} —— ${b} bits/weight`,
    svgAria: (n: string) =>
      `一张 GGUF 量化类型的阶梯，按纯类型 bits/weight 排序，从约 1.5 的 IQ1_S 到 8.5 的 Q8_0；bf16 在 16，落在顶部之外。当前显示 ${n}。`,
    pureBpw: 'bits/weight（纯类型）',
    deriv: 'struct',
    download: '你下载到的是',
    facts: {
      iq1s: {
        what: '差不多 1-bit 的地板。一个 I-quant（codebook + importance matrix）—— 只有超大的 MoE 才说得过去，而且没有 imatrix 根本没法用。',
        mix: '',
      },
      iq2xxs: { what: '一个 2-bit 的 I-quant：权重变成一个固定 codebook 的索引，由 importance matrix 来挑。', mix: '' },
      q2k: { what: '最小的 round k-quant。256 个权重一个 super-block，带两级 scale —— 和 MX 一样是个 block format。', mix: '通常是 Q2_K_XL（Unsloth）—— 连 Q2_K 都会把 attn_v、ffn_down 抬到 Q4_K' },
      iq3xxs: { what: '一个 3-bit 的 I-quant —— 同样大小下，质量明显好过 Q3_K，靠的就是 codebook + imatrix。', mix: '' },
      q3k: { what: '3-bit 的 k-quant super-block。对大模型来说激进但还连贯。', mix: 'Q3_K_M 在 8B 上 ≈ 4.0 bpw：attn_v、attn_o、ffn_down 抬到 Q4_K' },
      iq4xs: { what: '一个 4-bit 的 I-quant —— 整条阶梯上质量/字节比最好的几档之一。', mix: '' },
      q4k: { what: '默认之选。4-bit 的 k-quant —— 曲线的拐点，质量损失开始变得难以察觉。', mix: 'Q4_K_M 在 8B 上 ≈ 4.9 bpw：一半的 attn_v 与 ffn_down 抬到 Q6_K' },
      q5k: { what: '5-bit 的 k-quant。比 Q4_K 多一点余量，多花一点磁盘。', mix: 'Q5_K_M：一半的 attn_v 与 ffn_down 抬到 Q6_K（和 Q4_K_M 同一套）' },
      q6k: { what: '6-bit 的 k-quant。近乎无损 —— 与原模型的 KL-divergence 已经很小。', mix: '直接发纯类型（不混）—— 已经够接近源模型' },
      q80: { what: '8-bit，外加每 32 个权重一个 scale。基本无损；最稳的基准。', mix: '直接发纯类型 —— 大家拿来对照的参考' },
    } as Record<string, { what: string; mix: string }>,
  },
} as const;

// ── SVG geometry. EN labels are wider than zh, so every box is sized for EN. ──
const VB_W = 560;
const ROW_H = 24;
const ROW_GAP = 5;
const TOP = 40; // room for the bf16 cap + scale note above row 0
const BAR_H = 16;
const NAME_X = 8; // left edge for the Q-type name
const TRACK_X = 78; // bars start clear of the widest EN name ("IQ2_XXS")
const PAD_R = 14;
const TRACK_W = VB_W - TRACK_X - PAD_R; // 468
const VB_H = TOP + RUNGS.length * (ROW_H + ROW_GAP) - ROW_GAP + 6;

function rowY(i: number) {
  return TOP + i * (ROW_H + ROW_GAP);
}
function bpwToX(bpw: number) {
  return TRACK_X + (Math.min(bpw, MAX_BPW) / MAX_BPW) * TRACK_W;
}
function fmtBpw(r: { bpw: number; approx?: boolean }) {
  return (r.approx ? '≈ ' : '') + r.bpw;
}

export function GgufQuantLadder() {
  const copy = COPY[useLocale()];
  const [sel, setSel] = React.useState<string>(DEFAULT_SEL);
  const selRung = RUNGS.find((r) => r.id === sel)!;
  const fact = copy.facts[sel];

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header + bf16 anchor */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>
        <span className="inline-flex items-center gap-1.5 rounded border border-foreground/40 bg-foreground/5 px-2 py-0.5 text-[11px] font-medium text-foreground/80">
          🔒 {copy.youAreHere}
        </span>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: RED }} />
          {copy.legendFloor}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: EMERALD }} />
          {copy.legendSweet}
        </span>
        <span>{copy.scaleNote}</span>
      </div>

      {/* The ladder */}
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.svgAria(selRung.name)}>
        {/* faint integer-bpw gridlines + labels */}
        {[1, 2, 3, 4, 5, 6, 7, 8].map((g) => {
          const x = bpwToX(g);
          return (
            <g key={g}>
              <line
                x1={x}
                y1={TOP - 4}
                x2={x}
                y2={VB_H - 6}
                style={{ stroke: 'var(--border)' }}
                strokeWidth={1}
                strokeDasharray="2 3"
              />
              <text x={x} y={TOP - 8} textAnchor="middle" style={{ fill: 'var(--muted-foreground)' }} className="text-[9px]">
                {g}
              </text>
            </g>
          );
        })}

        {/* bf16 off-scale cap: a capped bar to the right edge with a break mark */}
        <text x={NAME_X} y={16} style={{ fill: 'var(--foreground)' }} className="font-mono text-[11px] font-semibold">
          {DEMO_NAME}
        </text>
        <rect x={TRACK_X} y={8} width={TRACK_W} height={9} rx={2} style={{ fill: 'var(--foreground)', opacity: 0.18 }} />
        <text x={TRACK_X + TRACK_W} y={16} textAnchor="end" style={{ fill: 'var(--muted-foreground)' }} className="text-[9px]">
          {DEMO_BPW} →
        </text>

        {RUNGS.map((r, i) => {
          const y = rowY(i);
          const barY = y + (ROW_H - BAR_H) / 2;
          const barW = bpwToX(r.bpw) - TRACK_X;
          const isSel = r.id === sel;
          const labelInside = barW > 30;

          return (
            <g
              key={r.id}
              onClick={() => setSel(r.id)}
              style={{ cursor: 'pointer' }}
              role="button"
              aria-pressed={isSel}
              aria-label={copy.rowAria(r.name, fmtBpw(r))}
            >
              {/* selection highlight spans the full row */}
              <rect
                x={2}
                y={y}
                width={VB_W - 4}
                height={ROW_H}
                rx={4}
                style={{ fill: isSel ? 'var(--primary)' : 'transparent', opacity: isSel ? 0.1 : 1 }}
              />
              {/* Q-type name (left, never over a bar) */}
              <text
                x={NAME_X}
                y={y + ROW_H / 2 + 4}
                style={{ fill: isSel ? 'var(--foreground)' : 'var(--muted-foreground)' }}
                className={['font-mono text-[11px]', isSel ? 'font-semibold' : 'font-medium'].join(' ')}
              >
                {r.name}
              </text>
              {/* the bpw bar */}
              <rect
                x={TRACK_X}
                y={barY}
                width={Math.max(2, barW)}
                height={BAR_H}
                rx={3}
                style={{ fill: TIER_FILL[r.tier], opacity: isSel ? 1 : 0.85 }}
              />
              {/* bpw value: inside the bar if it fits, else just past its end */}
              <text
                x={labelInside ? TRACK_X + barW - 5 : TRACK_X + barW + 5}
                y={barY + BAR_H / 2 + 3.5}
                textAnchor={labelInside ? 'end' : 'start'}
                style={{ fill: labelInside ? 'var(--background)' : 'var(--muted-foreground)' }}
                className="font-mono text-[10px] font-semibold"
              >
                {fmtBpw(r)}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Detail panel */}
      <div className="rounded border border-border bg-card p-3 text-[13px]">
        <div className="mb-1 flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="font-mono text-sm font-semibold text-foreground">{selRung.name}</span>
          <span className="text-muted-foreground">
            {fmtBpw(selRung)} {copy.pureBpw}
          </span>
          {selRung.deriv ? (
            <span className="font-mono text-[11px] text-muted-foreground">
              {copy.deriv}: {selRung.deriv} = {selRung.bpw}
            </span>
          ) : null}
        </div>
        <p className="text-foreground/90">{fact.what}</p>
        {fact.mix ? (
          <p className="mt-1 text-[12px] text-muted-foreground">
            <span className="font-medium text-foreground/70">{copy.download}:</span> {fact.mix}
          </p>
        ) : null}
      </div>
    </div>
  );
}
