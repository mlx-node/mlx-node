import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * FormatZooLadder — the whole numeric-format zoo at a glance. Every format is
 * one row whose bar is segmented sign | exponent | mantissa, drawn to a COMMON
 * bit-width scale (1 bit = a constant pixel width) so you SEE the exponent field
 * grow and shrink independently of total width. E4M3 and E5M2 are both 8 bits,
 * yet their exponent and mantissa fields visibly swap. Integers render as a
 * single flat "uniform · N levels" bar — no fields, no exponent, no dynamic
 * range. Click a row → a detail panel with the exact ground truth.
 *
 * Self-contained and STATIC: NO worker, NO model, NO WASM, NO network, NO
 * animation — pure React + Tailwind/SVG, so it server-renders for crawlers
 * (createRoot replaces it on boot). The first render is deterministic and
 * identical server vs client; the only state is the selected row, so it needs
 * no prefers-reduced-motion handling.
 *
 * EVERY NUMBER IS SOURCE-VERIFIED — do not edit without a citation:
 *   - IEEE-754: fp32 1/8/23, fp16 1/5/10 (IEEE 754-2019). bf16 1/8/7 is the
 *     Google Brain truncated-fp32 layout (same 8-bit exponent as fp32).
 *   - tf32 1/8/10 = 19 stored bits, held in a 32-bit register; NVIDIA Ampere
 *     Tensor-Core compute mode (FP32 range, FP16 precision), NOT a storage type.
 *   - fp8 E4M3 1/4/3 max 448 and E5M2 1/5/2 max 57344 follow the OCP / NVIDIA /
 *     Intel / ARM "FP8 Formats for Deep Learning" convention (arXiv 2209.05433).
 *     E4M3 has NO infinities and a single NaN; it reclaims the inf/NaN bit
 *     patterns a naive IEEE 1.4.3 minifloat would spend (that naive layout maxes
 *     at 240) to reach 448. E5M2 stays IEEE-like with inf + NaN.
 *   - fp6 E3M2 1/3/2 max 28 and E2M3 1/2/3 max 7.5 are the OCP Microscaling
 *     (MXFP6) element formats; no inf, no NaN.
 *   - fp4 E2M1 1/2/1 is the OCP Microscaling element: exactly 8 magnitudes
 *     {0, 0.5, 1, 1.5, 2, 3, 4, 6}, max 6; only ever used as a block format
 *     (MXFP4 / NVFP4, with a shared per-block scale).
 *   - int8 / int4 are uniform signed integer grids: 256 levels (−128..127) and
 *     16 levels (−8..7). No exponent → no dynamic range, just a uniform step.
 *   - MXFP8 / MXFP4 (OCP Microscaling v1.0): block size 32, ONE shared E8M0
 *     scale — 8 exponent bits, 0 mantissa, a pure power of two 2^-127…2^127
 *     with 0xFF = NaN. Effective storage = element + 8/32 → 8.25 / 4.25 b/elem.
 *   - NVFP4 (NVIDIA Blackwell, non-OCP): block size 16, per-block FP8 E4M3
 *     scale + ONE per-tensor FP32 scale (two-level). 4 + 8/16 = 4.50 b/elem.
 *     These three render in a separate "block formats" group: the element's
 *     sign|exp|mant cells on the SAME common bit scale, plus a dashed
 *     shared-scale box — dashed because those 8 bits are paid once per block,
 *     not once per element (the ÷32 / ÷16 label says exactly that).
 *
 * HONESTY — speed/quality are NOT shown here. The course source gives no
 * perplexity numbers; this widget renders only EXACT structural facts (bit
 * fields, max value, level count, dynamic range read straight off the exponent).
 */

type FmtId =
  | 'fp32'
  | 'tf32'
  | 'fp16'
  | 'bf16'
  | 'fp8e4m3'
  | 'fp8e5m2'
  | 'fp6e3m2'
  | 'fp6e2m3'
  | 'fp4e2m1'
  | 'int8'
  | 'int4'
  | 'mxfp8'
  | 'nvfp4'
  | 'mxfp4';

const DEMO_FMT: FmtId = 'bf16'; // the in-browser demo ships this — "you are here"

// Tailwind palette hexes for SVG fills (only EMERALD/RED literals are allowed).
const EMERALD = '#10b981'; // good / "you are here"

// Float bit-field layout. sign + exp + mant === bits.
type FloatRow = {
  kind: 'float';
  id: FmtId;
  name: string; // glossary English — never translated
  bits: number;
  sign: number;
  exp: number;
  mant: number;
};
type IntRow = {
  kind: 'int';
  id: FmtId;
  name: string;
  bits: number;
  levels: number;
};
// Block (microscaling) formats: an element layout PLUS a shared per-block
// scale. The element's sign/exp/mant cells draw on the common bit scale; the
// scale draws as a dashed box because its bits are paid once per block.
type BlockRow = {
  kind: 'block';
  id: FmtId;
  name: string;
  /** element field split — same semantics as FloatRow */
  sign: number;
  exp: number;
  mant: number;
  elemBits: number; // sign + exp + mant
  elemName: string; // glossary English — never translated
  scaleBits: number; // bits in the per-block scale (8 for E8M0 and E4M3)
  scaleName: 'E8M0' | 'E4M3'; // per-block scale kind
  block: number; // elements sharing one scale (32 OCP MX, 16 NVFP4)
  effBits: number; // elemBits + scaleBits / block
  /** NVFP4 only: one extra per-tensor FP32 scale above every block */
  tensorScale: boolean;
};
type Row = FloatRow | IntRow;

// Rows sorted by total bits DESC (floats before ints at an equal width, so the
// E4M3 ↔ E5M2 and E3M2 ↔ E2M3 swaps sit adjacent). Every field is exact.
const ROWS: Row[] = [
  { kind: 'float', id: 'fp32', name: 'fp32', bits: 32, sign: 1, exp: 8, mant: 23 },
  { kind: 'float', id: 'tf32', name: 'tf32', bits: 19, sign: 1, exp: 8, mant: 10 },
  { kind: 'float', id: 'fp16', name: 'fp16', bits: 16, sign: 1, exp: 5, mant: 10 },
  { kind: 'float', id: 'bf16', name: 'bf16', bits: 16, sign: 1, exp: 8, mant: 7 },
  { kind: 'float', id: 'fp8e4m3', name: 'fp8 E4M3', bits: 8, sign: 1, exp: 4, mant: 3 },
  { kind: 'float', id: 'fp8e5m2', name: 'fp8 E5M2', bits: 8, sign: 1, exp: 5, mant: 2 },
  { kind: 'int', id: 'int8', name: 'int8', bits: 8, levels: 256 },
  { kind: 'float', id: 'fp6e3m2', name: 'fp6 E3M2', bits: 6, sign: 1, exp: 3, mant: 2 },
  { kind: 'float', id: 'fp6e2m3', name: 'fp6 E2M3', bits: 6, sign: 1, exp: 2, mant: 3 },
  { kind: 'float', id: 'fp4e2m1', name: 'fp4 E2M1', bits: 4, sign: 1, exp: 2, mant: 1 },
  { kind: 'int', id: 'int4', name: 'int4', bits: 4, levels: 16 },
];

// Block formats in their own group below the scalar ladder, sorted by
// effective bits/elem DESC. Every number verified (OCP MX v1.0; NVIDIA
// Blackwell docs) — same ground truth as MxBlockAnatomy: block 32 + E8M0 for
// OCP MX, block 16 + E4M3 (+ per-tensor FP32) for NVFP4.
const BLOCK_ROWS: BlockRow[] = [
  // MXFP8: fp8 E4M3 element (the OCP MX fp8 element; E5M2 also allowed) — 8 + 8/32 = 8.25
  {
    kind: 'block',
    id: 'mxfp8',
    name: 'mxfp8',
    sign: 1,
    exp: 4,
    mant: 3,
    elemBits: 8,
    elemName: 'fp8 E4M3',
    scaleBits: 8,
    scaleName: 'E8M0',
    block: 32,
    effBits: 8 + 8 / 32,
    tensorScale: false,
  },
  // NVFP4: fp4 E2M1 element, tighter 16-block, real E4M3 scale + fp32 tensor scale — 4 + 8/16 = 4.50
  {
    kind: 'block',
    id: 'nvfp4',
    name: 'nvfp4',
    sign: 1,
    exp: 2,
    mant: 1,
    elemBits: 4,
    elemName: 'fp4 E2M1',
    scaleBits: 8,
    scaleName: 'E4M3',
    block: 16,
    effBits: 4 + 8 / 16,
    tensorScale: true,
  },
  // MXFP4: fp4 E2M1 element, OCP 32-block, E8M0 scale — 4 + 8/32 = 4.25
  {
    kind: 'block',
    id: 'mxfp4',
    name: 'mxfp4',
    sign: 1,
    exp: 2,
    mant: 1,
    elemBits: 4,
    elemName: 'fp4 E2M1',
    scaleBits: 8,
    scaleName: 'E8M0',
    block: 32,
    effBits: 4 + 8 / 32,
    tensorScale: false,
  },
];

const MAX_BITS = 32; // fp32 — the widest bar = the full track

// Segment fills: sign = foreground, exponent = primary, mantissa = muted.
const SEG = {
  sign: 'var(--foreground)',
  exp: 'var(--primary)',
  mant: 'var(--muted-foreground)',
} as const;

// ── Per-locale copy. en is the original English; zh translates prose only and
// keeps ALL glossary terms English (every format name, sign/exponent/mantissa,
// dynamic range, scale, block, MXFP8/MXFP4/NVFP4, Tensor Core, Ampere, bits, etc.) and
// uses full-width punctuation （，、。：）. Numbers/units never translate. ───────
const COPY = {
  en: {
    heading: 'The format zoo — one common bit scale',
    youAreHere: 'you are here · the demo ships bf16',
    legendSign: 'sign',
    legendExp: 'exponent',
    legendMant: 'mantissa',
    legendInt: 'uniform integer grid',
    legendScale: 'shared per-block scale',
    scaleNote: '1 cell = 1 bit · all bars share one scale (fp32 = full width)',
    blockDivider: 'block formats — element bits + a shared per-block scale',
    rowAria: (n: string, b: number) => `Select ${n} — ${b} bits`,
    rowAriaBlock: (r: BlockRow) =>
      `Select ${r.name} — ${r.elemBits}-bit ${r.elemName} element plus one ${r.scaleBits}-bit ${r.scaleName} scale shared by ${r.block} elements${
        r.tensorScale ? ', plus one fp32 per-tensor scale above every block' : ''
      }`,
    tensorScaleTag: '× fp32 per-tensor', // glossary — English in both locales
    // detail panel
    panelTitle: (n: string) => n,
    totalBits: 'total bits',
    effHeader: (r: BlockRow) =>
      `${r.elemBits} + ${r.scaleBits}/${r.block} = ${r.effBits.toFixed(2)} bits/elem effective`,
    fieldSplit: 'field split',
    splitFloat: (s: number, e: number, m: number) => `sign ${s} · exponent ${e} · mantissa ${m}`,
    splitInt: (b: number) => `uniform ${b}-bit · no exponent`,
    splitBlock: (r: BlockRow) => `${r.block} × ${r.elemName} + one ${r.scaleName} scale`,
    maxVal: 'max value',
    dynRange: 'dynamic range',
    dynRangeFloat: (e: number) => `${e}-bit exponent`,
    dynRangeNone: 'none — uniform step',
    dynRangeBlock: (r: BlockRow) =>
      r.scaleName === 'E8M0' ? 'E8M0 block scale · 2^−127…2^127' : 'E4M3 block scale × fp32 tensor scale',
    levelsLabel: 'levels / precision',
    typicalUse: 'typical use',
    // per-format facts (exact). digits/notes verified above.
    facts: {
      fp32: {
        max: '≈ 3.40e38',
        levels: '~7.2 decimal digits',
        use: 'baseline / master weights',
        extra: '',
      },
      tf32: {
        max: '≈ 3.4e38',
        levels: 'FP32 range + FP16 precision',
        use: 'Ampere Tensor-Core compute mode — not a storage type (19 bits held in 32)',
        extra: '',
      },
      fp16: {
        max: '65504',
        levels: '~3.3 decimal digits',
        use: 'mixed-precision — needs loss scaling',
        extra: '',
      },
      bf16: {
        max: '≈ 3.39e38',
        levels: '~2-3 decimal digits',
        use: 'dominant training precision (you are here)',
        extra: '',
      },
      fp8e4m3: {
        max: '448',
        levels: 'no infinities · 1 NaN',
        use: 'weights / activations · MXFP8 block element',
        extra:
          'The 448 max is the ML (OCP) convention: a naive IEEE 1.4.3 minifloat maxes at 240, but E4M3 reclaims the inf/NaN bit patterns to reach 448.',
      },
      fp8e5m2: {
        max: '57344',
        levels: 'IEEE-like · inf + NaN',
        use: 'gradients (range-leaning)',
        extra: '',
      },
      fp6e3m2: {
        max: '28',
        levels: 'no inf / no NaN',
        use: 'MXFP6 element (range-leaning)',
        extra: '',
      },
      fp6e2m3: {
        max: '7.5',
        levels: 'no inf / no NaN',
        use: 'MXFP6 element (precision-leaning)',
        extra: '',
      },
      fp4e2m1: {
        max: '6',
        levels: 'values {0, 0.5, 1, 1.5, 2, 3, 4, 6}',
        use: 'only ever a block format (MXFP4 / NVFP4)',
        extra: '',
      },
      int8: {
        max: '127',
        levels: '256 levels · signed −128..127',
        use: 'no exponent → no dynamic range',
        extra: '',
      },
      int4: {
        max: '7',
        levels: '16 levels · signed −8..7',
        use: 'uniform grid',
        extra: '',
      },
      mxfp8: {
        max: 'element max 448',
        levels: 'element: no infinities · 1 NaN (E4M3)',
        use: 'near-lossless block fp8 (OCP MX, block 32)',
        extra:
          'OCP MX v1.0 fixes block = 32 and one shared E8M0 scale (8 exponent bits, 0 mantissa — a pure power of two, 0xFF = NaN) for every MX width. fp8 already has range, so MXFP8 is the gentle end of the recipe.',
      },
      nvfp4: {
        max: 'element max 6 · block scale ≤ 448',
        levels: 'element: 8 magnitudes {0…6}',
        use: 'NVIDIA Blackwell fp4 (block 16, two-level scaling)',
        extra:
          'Non-OCP: the block scale is a real FP8 E4M3 float (it has a mantissa, so it lands between powers of two — finer than E8M0), and one per-tensor FP32 scale sits above every block.',
      },
      mxfp4: {
        max: 'element max 6',
        levels: 'element: 8 magnitudes {0…6}',
        use: 'block fp4 weights (OCP MX, block 32)',
        extra:
          'Same fp4 E2M1 element as the scalar fp4 row — the shared E8M0 scale borrows back the dynamic range a lone 4-bit float lacks.',
      },
    } satisfies Record<FmtId, { max: string; levels: string; use: string; extra: string }>,
    svgAria: (n: string) =>
      `A ladder of numeric formats sorted by total bit width, every bar drawn to one common scale where one bit is a fixed pixel width. Each floating-point row is segmented into sign, exponent, and mantissa fields; integer rows are a single uniform grid. A separate block-formats group at the bottom (MXFP8, NVFP4, MXFP4) shows the element fields plus a dashed shared per-block scale box. Currently showing the ${n} detail.`,
  },
  zh: {
    heading: '格式动物园——共用一把 bit 标尺',
    youAreHere: '你在这里 · demo 用的是 bf16',
    legendSign: 'sign',
    legendExp: 'exponent',
    legendMant: 'mantissa',
    legendInt: '均匀整数格点',
    legendScale: '共享 per-block scale',
    scaleNote: '1 格 = 1 bit · 所有条共用一把标尺（fp32 = 满宽）',
    blockDivider: 'block formats——element bits + 共享 per-block scale',
    rowAria: (n: string, b: number) => `选择 ${n}——${b} bits`,
    rowAriaBlock: (r: BlockRow) =>
      `选择 ${r.name}——${r.elemBits}-bit ${r.elemName} element，外加每 ${r.block} 个元素共享 1 个 ${r.scaleBits}-bit ${r.scaleName} scale${
        r.tensorScale ? '，每个 block 之上还有 1 个 fp32 per-tensor scale' : ''
      }`,
    tensorScaleTag: '× fp32 per-tensor',
    panelTitle: (n: string) => n,
    totalBits: 'total bits',
    effHeader: (r: BlockRow) => `${r.elemBits} + ${r.scaleBits}/${r.block} = ${r.effBits.toFixed(2)} bits/elem 等效`,
    fieldSplit: '字段拆分',
    splitFloat: (s: number, e: number, m: number) => `sign ${s} · exponent ${e} · mantissa ${m}`,
    splitInt: (b: number) => `均匀 ${b}-bit · 无 exponent`,
    splitBlock: (r: BlockRow) => `${r.block} × ${r.elemName} + 1 个 ${r.scaleName} scale`,
    maxVal: 'max value',
    dynRange: 'dynamic range',
    dynRangeFloat: (e: number) => `${e}-bit exponent`,
    dynRangeNone: '无——均匀步长',
    dynRangeBlock: (r: BlockRow) =>
      r.scaleName === 'E8M0' ? 'E8M0 block scale · 2^−127…2^127' : 'E4M3 block scale × fp32 tensor scale',
    levelsLabel: 'levels / precision',
    typicalUse: '典型用途',
    facts: {
      fp32: {
        max: '≈ 3.40e38',
        levels: '~7.2 十进制位',
        use: '基准 / master weights',
        extra: '',
      },
      tf32: {
        max: '≈ 3.4e38',
        levels: 'FP32 range + FP16 precision',
        use: 'Ampere Tensor-Core 计算模式——不是存储类型（19 bits 装在 32 里）',
        extra: '',
      },
      fp16: {
        max: '65504',
        levels: '~3.3 十进制位',
        use: '混合精度——需要 loss scaling',
        extra: '',
      },
      bf16: {
        max: '≈ 3.39e38',
        levels: '~2-3 十进制位',
        use: '主流训练精度（你在这里）',
        extra: '',
      },
      fp8e4m3: {
        max: '448',
        levels: '没有 infinities · 1 个 NaN',
        use: 'weights / activations · MXFP8 block element',
        extra:
          '448 这个上限是 ML（OCP）约定：一个朴素的 IEEE 1.4.3 minifloat 上限只到 240，而 E4M3 把 inf/NaN 的位模式回收过来，达到 448。',
      },
      fp8e5m2: {
        max: '57344',
        levels: 'IEEE-like · inf + NaN',
        use: 'gradients（偏 range）',
        extra: '',
      },
      fp6e3m2: {
        max: '28',
        levels: '无 inf / 无 NaN',
        use: 'MXFP6 元素（偏 range）',
        extra: '',
      },
      fp6e2m3: {
        max: '7.5',
        levels: '无 inf / 无 NaN',
        use: 'MXFP6 元素（偏 precision）',
        extra: '',
      },
      fp4e2m1: {
        max: '6',
        levels: 'values {0, 0.5, 1, 1.5, 2, 3, 4, 6}',
        use: '只会作为 block format（MXFP4 / NVFP4）',
        extra: '',
      },
      int8: {
        max: '127',
        levels: '256 个档位 · signed −128..127',
        use: '无 exponent → 无 dynamic range',
        extra: '',
      },
      int4: {
        max: '7',
        levels: '16 个档位 · signed −8..7',
        use: '均匀格点',
        extra: '',
      },
      mxfp8: {
        max: 'element max 448',
        levels: 'element：无 infinities · 1 个 NaN（E4M3）',
        use: '近乎无损的 block fp8（OCP MX，block 32）',
        extra:
          'OCP MX v1.0 对所有 MX 宽度固定 block = 32、共享 1 个 E8M0 scale（8 个 exponent bit、0 个 mantissa——纯 2 的幂，0xFF = NaN）。fp8 本身就有 range，所以 MXFP8 是这套配方里最温和的一端。',
      },
      nvfp4: {
        max: 'element max 6 · block scale ≤ 448',
        levels: 'element：8 个幅值 {0…6}',
        use: 'NVIDIA Blackwell fp4（block 16，两级 scaling）',
        extra:
          '非 OCP：block scale 是真正的 FP8 E4M3 浮点（带 mantissa，能落在 2 的幂之间——比 E8M0 更细），每个 block 之上还有 1 个 per-tensor FP32 scale。',
      },
      mxfp4: {
        max: 'element max 6',
        levels: 'element：8 个幅值 {0…6}',
        use: 'block fp4 weights（OCP MX，block 32）',
        extra:
          '和标量 fp4 行用的是同一个 fp4 E2M1 element——共享的 E8M0 scale 把单个 4-bit 浮点缺失的 dynamic range 借了回来。',
      },
    } satisfies Record<FmtId, { max: string; levels: string; use: string; extra: string }>,
    svgAria: (n: string) =>
      `一张按 total bit 宽度排序的数值格式阶梯，所有条都画在同一把标尺上，1 bit 对应固定像素宽度。每个浮点行被拆成 sign、exponent、mantissa 字段；整数行是单一的均匀格点。底部单独的 block formats 组（MXFP8、NVFP4、MXFP4）展示 element 字段外加一个虚线的共享 per-block scale 框。当前显示 ${n} 的细节。`,
  },
} as const;

// ── SVG geometry. EN labels are wider than zh, so every box is sized for EN. ──
// One column of rows. Each row: a left-anchored format name, then a bit-scaled
// bar starting at TRACK_X. fp32 fills the track; narrower formats are shorter.
// The name sits LEFT of the track (never over the bar); per-field bit counts sit
// INSIDE wide-enough segments, else are dropped (the panel carries the split).
const VB_W = 560;
const ROW_H = 26;
const ROW_GAP = 6;
const TOP = 26; // room for the scale note above row 0
const BAR_H = 18;
const NAME_X = 8; // left edge for the format name
const TRACK_X = 92; // bars start here, clear of the widest EN name ("fp8 E5M2")
const PAD_R = 12;
const TRACK_W = VB_W - TRACK_X - PAD_R; // 456
// Block-format group geometry: a labeled divider band separates the scalar
// ladder from the three block rows; the scale box sits a small gap after the
// element cells so it never reads as part of one element's bits.
const SCALAR_BOTTOM = TOP + ROWS.length * (ROW_H + ROW_GAP) - ROW_GAP;
const DIVIDER_H = 26; // divider band: a rule + the group label
const BLOCK_TOP = SCALAR_BOTTOM + DIVIDER_H;
const BLOCK_GAP = 7; // px gap between element cells and the shared-scale box
const VB_H = BLOCK_TOP + BLOCK_ROWS.length * (ROW_H + ROW_GAP) - ROW_GAP + 4;

function rowY(i: number) {
  return TOP + i * (ROW_H + ROW_GAP);
}

function blockRowY(j: number) {
  return BLOCK_TOP + j * (ROW_H + ROW_GAP);
}

// sign | exponent | mantissa cells starting at x0, on the common bit scale.
// Shared by scalar float rows and the element part of block rows.
function FloatCells({
  sign,
  exp,
  mant,
  x0,
  barY,
  isSel,
}: {
  sign: number;
  exp: number;
  mant: number;
  x0: number;
  barY: number;
  isSel: boolean;
}) {
  const cells: Array<{ kind: keyof typeof SEG; bits: number }> = [
    { kind: 'sign', bits: sign },
    { kind: 'exp', bits: exp },
    { kind: 'mant', bits: mant },
  ];
  let cursor = x0;
  return (
    <>
      {cells.map((c) => {
        const w = (c.bits / MAX_BITS) * TRACK_W;
        const x = cursor;
        cursor += w;
        const showNum = w > 16; // only label segments wide enough to hold a number
        return (
          <g key={c.kind}>
            <rect
              x={x}
              y={barY}
              width={Math.max(1.5, w - 1)}
              height={BAR_H}
              rx={2}
              style={{ fill: SEG[c.kind], opacity: c.kind === 'mant' ? 0.4 : isSel ? 0.95 : 0.8 }}
            />
            {showNum ? (
              <text
                x={x + w / 2}
                y={barY + BAR_H / 2 + 3.5}
                textAnchor="middle"
                style={{ fill: c.kind === 'mant' ? 'var(--foreground)' : 'var(--background)' }}
                className="text-[9px] font-semibold"
              >
                {c.bits}
              </text>
            ) : null}
          </g>
        );
      })}
    </>
  );
}

export function FormatZooLadder() {
  const copy = COPY[useLocale()];
  const [sel, setSel] = React.useState<FmtId>(DEMO_FMT);
  const selRow = [...ROWS, ...BLOCK_ROWS].find((r) => r.id === sel)!;
  const f = copy.facts[sel];

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>
        <span className="inline-flex items-center gap-1.5 rounded border border-foreground/40 bg-foreground/5 px-2 py-0.5 text-[11px] font-medium text-foreground/80">
          🔒 {copy.youAreHere}
        </span>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: 'var(--foreground)' }} />
          {copy.legendSign}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: 'var(--primary)' }} />
          {copy.legendExp}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-2.5 rounded-sm"
            style={{ background: 'var(--muted-foreground)', opacity: 0.4 }}
          />
          {copy.legendMant}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-2.5 rounded-sm border border-border"
            style={{ background: 'var(--muted)' }}
          />
          {copy.legendInt}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-2.5 rounded-sm border border-dashed"
            style={{
              borderColor: 'var(--primary)',
              background: 'color-mix(in srgb, var(--primary) 18%, transparent)',
            }}
          />
          {copy.legendScale}
        </span>
      </div>

      {/* The ladder */}
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.svgAria(selRow.name)}>
        {/* scale note, top-left, above row 0 */}
        <text x={NAME_X} y={14} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
          {copy.scaleNote}
        </text>

        {ROWS.map((r, i) => {
          const y = rowY(i);
          const barY = y + (ROW_H - BAR_H) / 2;
          const barW = (r.bits / MAX_BITS) * TRACK_W;
          const isSel = r.id === sel;
          const isDemo = r.id === DEMO_FMT;

          return (
            <g key={r.id}>
              {/* clickable row: selection highlight spans the full row */}
              <rect
                x={2}
                y={y}
                width={VB_W - 4}
                height={ROW_H}
                rx={4}
                style={{
                  fill: isSel ? 'var(--primary)' : 'transparent',
                  opacity: isSel ? 0.1 : 1,
                }}
              />

              {/* format name (left, never over a bar) */}
              <text
                x={NAME_X}
                y={y + ROW_H / 2 + 4}
                style={{ fill: isSel ? 'var(--foreground)' : 'var(--muted-foreground)' }}
                className={['font-mono text-[11px]', isSel ? 'font-semibold' : 'font-medium'].join(' ')}
              >
                {r.name}
              </text>

              {/* you-are-here dot on the bf16 row name */}
              {isDemo ? <circle cx={NAME_X - 4 + 78} cy={y + ROW_H / 2} r={2.5} style={{ fill: EMERALD }} /> : null}

              {/* the bit-scaled bar */}
              {r.kind === 'float' ? (
                <FloatCells sign={r.sign} exp={r.exp} mant={r.mant} x0={TRACK_X} barY={barY} isSel={isSel} />
              ) : (
                <>
                  {/* integer: one flat uniform bar, no fields */}
                  <rect
                    x={TRACK_X}
                    y={barY}
                    width={Math.max(2, barW)}
                    height={BAR_H}
                    rx={2}
                    style={{ fill: 'var(--muted)', stroke: 'var(--border)' }}
                    strokeWidth={1}
                  />
                  <text
                    x={TRACK_X + Math.max(2, barW) / 2}
                    y={barY + BAR_H / 2 + 3.5}
                    textAnchor="middle"
                    style={{ fill: 'var(--foreground)' }}
                    className="text-[9px] font-semibold"
                  >
                    {r.levels}
                  </text>
                </>
              )}

              {/* transparent hit target on top of everything for the whole row */}
              <rect
                x={2}
                y={y}
                width={VB_W - 4}
                height={ROW_H}
                fill="transparent"
                style={{ cursor: 'pointer' }}
                onClick={() => setSel(r.id)}
                tabIndex={0}
                role="button"
                aria-pressed={isSel}
                aria-label={copy.rowAria(r.name, r.bits)}
                className="focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    setSel(r.id);
                  }
                }}
              />
            </g>
          );
        })}

        {/* ── Block-formats group: a labeled divider, then element + dashed shared scale ── */}
        <line
          x1={2}
          x2={VB_W - 2}
          y1={SCALAR_BOTTOM + 8}
          y2={SCALAR_BOTTOM + 8}
          style={{ stroke: 'var(--border)' }}
          strokeWidth={1}
        />
        <text x={NAME_X} y={SCALAR_BOTTOM + 21} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
          {copy.blockDivider}
        </text>

        {BLOCK_ROWS.map((r, j) => {
          const y = blockRowY(j);
          const barY = y + (ROW_H - BAR_H) / 2;
          const isSel = r.id === sel;
          const elemW = (r.elemBits / MAX_BITS) * TRACK_W;
          const scaleX = TRACK_X + elemW + BLOCK_GAP;
          const scaleW = (r.scaleBits / MAX_BITS) * TRACK_W;

          return (
            <g key={r.id}>
              {/* clickable row: selection highlight spans the full row */}
              <rect
                x={2}
                y={y}
                width={VB_W - 4}
                height={ROW_H}
                rx={4}
                style={{
                  fill: isSel ? 'var(--primary)' : 'transparent',
                  opacity: isSel ? 0.1 : 1,
                }}
              />

              {/* format name (left, never over a bar) */}
              <text
                x={NAME_X}
                y={y + ROW_H / 2 + 4}
                style={{ fill: isSel ? 'var(--foreground)' : 'var(--muted-foreground)' }}
                className={['font-mono text-[11px]', isSel ? 'font-semibold' : 'font-medium'].join(' ')}
              >
                {r.name}
              </text>

              {/* element cells — same common bit scale as every scalar row */}
              <FloatCells sign={r.sign} exp={r.exp} mant={r.mant} x0={TRACK_X} barY={barY} isSel={isSel} />

              {/* the shared per-block scale: dashed because these bits are paid
                  once per block (÷32 / ÷16), not once per element */}
              <rect
                x={scaleX}
                y={barY}
                width={scaleW - 1}
                height={BAR_H}
                rx={2}
                style={{
                  fill: 'var(--primary)',
                  fillOpacity: isSel ? 0.22 : 0.15,
                  stroke: 'var(--primary)',
                  strokeDasharray: '3 2.5',
                }}
                strokeWidth={1}
              />
              <text
                x={scaleX + scaleW / 2}
                y={barY + BAR_H / 2 + 3.5}
                textAnchor="middle"
                style={{ fill: 'var(--foreground)' }}
                className="text-[9px] font-semibold"
              >
                {`${r.scaleName} ÷${r.block}`}
              </text>
              {r.tensorScale ? (
                <text
                  x={scaleX + scaleW + 7}
                  y={barY + BAR_H / 2 + 3.5}
                  style={{ fill: 'var(--muted-foreground)' }}
                  className="text-[9px]"
                >
                  {copy.tensorScaleTag}
                </text>
              ) : null}

              {/* transparent hit target on top of everything for the whole row */}
              <rect
                x={2}
                y={y}
                width={VB_W - 4}
                height={ROW_H}
                fill="transparent"
                style={{ cursor: 'pointer' }}
                onClick={() => setSel(r.id)}
                tabIndex={0}
                role="button"
                aria-pressed={isSel}
                aria-label={copy.rowAriaBlock(r)}
                className="focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    setSel(r.id);
                  }
                }}
              />
            </g>
          );
        })}
      </svg>

      {/* Detail panel for the selected row */}
      <div className="rounded border border-border bg-card p-3 text-[12px]">
        <div className="mb-2 flex items-baseline gap-2">
          <span className="font-mono text-[14px] font-semibold text-foreground">{copy.panelTitle(selRow.name)}</span>
          <span className="text-[11px] text-muted-foreground">
            {selRow.kind === 'block' ? copy.effHeader(selRow) : `${selRow.bits} ${copy.totalBits}`}
          </span>
        </div>
        <dl className="grid grid-cols-1 gap-x-6 gap-y-1.5 sm:grid-cols-2">
          <div className="flex justify-between gap-3">
            <dt className="text-muted-foreground">{copy.fieldSplit}</dt>
            <dd className="text-right font-mono text-foreground">
              {selRow.kind === 'float'
                ? copy.splitFloat(selRow.sign, selRow.exp, selRow.mant)
                : selRow.kind === 'block'
                  ? copy.splitBlock(selRow)
                  : copy.splitInt(selRow.bits)}
            </dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="text-muted-foreground">{copy.maxVal}</dt>
            <dd className="text-right font-mono text-foreground">{f.max}</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="text-muted-foreground">{copy.dynRange}</dt>
            <dd className="text-right font-mono text-foreground">
              {selRow.kind === 'float'
                ? copy.dynRangeFloat(selRow.exp)
                : selRow.kind === 'block'
                  ? copy.dynRangeBlock(selRow)
                  : copy.dynRangeNone}
            </dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="text-muted-foreground">{copy.levelsLabel}</dt>
            <dd className="text-right font-mono text-foreground">{f.levels}</dd>
          </div>
        </dl>
        <p className="mt-2 border-t border-border/60 pt-2 text-muted-foreground">
          <span className="text-foreground/70">{copy.typicalUse}:</span> {f.use}
        </p>
        {f.extra ? <p className="mt-1.5 text-[11px] text-muted-foreground">{f.extra}</p> : null}
      </div>
    </div>
  );
}
