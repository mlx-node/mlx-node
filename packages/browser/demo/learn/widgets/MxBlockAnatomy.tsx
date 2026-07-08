import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * MxBlockAnatomy — the microscaling (MX) family: a shared per-block scale.
 *
 * A narrow element (fp4 E2M1 = 1 sign + 2 exponent + 1 mantissa) has almost no
 * dynamic range on its own — it carries only a relative value. A SHARED block
 * scale borrows the range back: every element is reconstructed as
 * (scale × element), so the scale supplies the magnitude and the element code
 * supplies the fine shape. That is microscaling (MX).
 *
 * This widget toggles between the THREE block formats the course names:
 *   • MXFP8 (OCP Microscaling) — 32 fp8 E4M3 elements share ONE E8M0 scale.
 *     fp8 already has range (max 448), so MXFP8 is the gentle, near-lossless end
 *     of the recipe — not borrowing range, just standardizing a block scale.
 *     Effective storage = 8 + 8/32 = 8.25 bits/element.
 *   • MXFP4 (OCP Microscaling) — 32 fp4 E2M1 elements share ONE E8M0 scale.
 *     E8M0 = 8 exponent bits, 0 mantissa → a pure power of two, 2^-127 … 2^127,
 *     with 0xFF reserved as NaN. Reconstruction v = X · P_i (one shared X).
 *     Effective storage = 4 + 8/32 = 4.25 bits/element.
 *   • NVFP4 (NVIDIA, Blackwell) — tighter 16-element blocks, and the per-block
 *     scale is a real FP8 E4M3 float (it has a 3-bit mantissa, so it lands
 *     between powers of two — finer than E8M0). A single per-tensor FP32 global
 *     scale sits above every block (two-level scaling):
 *       value = fp32_tensor_scale × e4m3_block_scale × element.
 *     Effective storage = 4 + 8/16 = 4.50 bits/element.
 *
 * OCP MX v1.0 fixes block size = 32 and the scale = E8M0 for ALL MX widths
 * (MXFP8 / MXFP6 / MXFP4); only the element type shrinks. NVFP4 is NVIDIA's
 * non-OCP variant (block 16, E4M3 block scale + FP32 per-tensor).
 *
 * Self-contained and STATIC: NO worker, NO model, NO WASM, NO network, NO
 * animation — pure React + Tailwind/SVG, so it server-renders for crawlers
 * (createRoot replaces it on boot). The first render is deterministic and
 * identical server vs client; the only state is the selected format, so it needs
 * no prefers-reduced-motion handling.
 *
 * EVERY NUMBER IS SOURCE-VERIFIED (OCP Microscaling Formats v1.0; NVIDIA NVFP4 /
 * Blackwell docs): block sizes (32 vs 16), the E8M0 layout and 2^-127..2^127
 * range with 0xFF=NaN, the E4M3 / FP32 scale roles, and the effective
 * bits/element (derived: element_bits + scale_bits/block_size). None invented.
 * The only qualitative claim — that smaller blocks + a float scale fit local
 * data more tightly (less error) — is shown WITHOUT a precise error number,
 * since the course source gives none.
 */

type Mode = 'mxfp8' | 'mxfp4' | 'nvfp4';

// Toggle order: the two E8M0 single-scale MX formats, then NVIDIA's two-level NV.
const MODES: Mode[] = ['mxfp8', 'mxfp4', 'nvfp4'];

// EMERALD = the "good / borrowed-back range" accent; the lone literal-hex pair
// the design system allows alongside RED. NVFP4's extra scale uses it too.
const EMERALD = '#10b981';

// ── Per-format ground truth (verified — do not edit numbers without a source). ─
type ModeInfo = {
  block: number; // elements per block
  scaleBits: number; // bits in the per-block scale
  elemBits: number; // 8 (fp8 E4M3) or 4 (fp4 E2M1)
  /** effective storage = element bits + scaleBits / block */
  effBits: number;
  elem: string; // element format name (glossary — English in both locales)
  elemFields: string; // bit split (glossary — English in both locales)
  scale: 'e8m0' | 'e4m3'; // per-block scale kind
};
const INFO: Record<Mode, ModeInfo> = {
  // MXFP8: 32-elem block, E8M0 scale (8 bits) → 8 + 8/32 = 8.25
  mxfp8: { block: 32, scaleBits: 8, elemBits: 8, effBits: 8 + 8 / 32, elem: 'fp8 E4M3', elemFields: '1 sign · 4 exponent · 3 mantissa', scale: 'e8m0' },
  // MXFP4: 32-elem block, E8M0 scale (8 bits) → 4 + 8/32 = 4.25
  mxfp4: { block: 32, scaleBits: 8, elemBits: 4, effBits: 4 + 8 / 32, elem: 'fp4 E2M1', elemFields: '1 sign · 2 exponent · 1 mantissa', scale: 'e8m0' },
  // NVFP4: 16-elem block, E4M3 scale (8 bits) + per-tensor FP32 → 4 + 8/16 = 4.50
  nvfp4: { block: 16, scaleBits: 8, elemBits: 4, effBits: 4 + 8 / 16, elem: 'fp4 E2M1', elemFields: '1 sign · 2 exponent · 1 mantissa', scale: 'e4m3' },
};

// ── Per-locale copy. en strings are the original English; zh translates the
// prose but KEEPS the glossary terms in English (fp4, fp6, fp8, E2M1, E8M0,
// E4M3, fp32, MXFP8, MXFP4, NVFP4, scale, scale factor, block, per-block,
// per-tensor, per-group, microscaling / MX, dynamic range, mantissa, exponent,
// sign, bits/elem, NaN, Blackwell) and uses full-width punctuation. ────────────
const COPY = {
  en: {
    heading: 'Block formats: a shared scale per block (the MX family)',
    modeLabel: 'format',
    pickAria: (m: string) => `Show the ${m} block layout`,
    // section labels
    elemHeading: (n: number, elem: string, fields: string) => `${n} × ${elem} elements (${fields})`,
    blockScaleHeading: 'per-block scale',
    tensorScaleHeading: 'per-tensor scale (one for the whole tensor)',
    // scale-box body text
    e8m0Title: 'E8M0 scale',
    e8m0Body: '8 exponent · 0 mantissa → a pure power of two',
    e8m0Range: '2^-127 … 2^127 · 0xFF = NaN',
    e4m3Title: 'FP8 E4M3 block scale',
    e4m3Body: '1 sign · 4 exponent · 3 mantissa → a real float',
    e4m3Note: 'has a mantissa → lands between powers of two (finer)',
    fp32Title: 'FP32 per-tensor scale',
    fp32Body: 'one global float scale, shared by every block',
    // reconstruction formula
    formulaMx: 'value = element × 2^scale',
    formulaMxSub: 'one shared scale  ·  v = X · Pᵢ',
    formulaNv: 'value = fp32_tensor_scale × e4m3_block_scale × element',
    formulaNvSub: 'two-level scaling',
    sharedLabel: 'shared by the whole block',
    // effective storage
    effHeading: 'effective storage',
    effValue: (b: number) => `${b.toFixed(2)} bits/elem`,
    eff: {
      mxfp8: '8 + 8/32 = 8.25 — one 8-bit E8M0 scale over 32 fp8 values (the near-lossless end)',
      mxfp4: '4 + 8/32 = 4.25 — one 8-bit scale spread over 32 fp4 elements',
      nvfp4: '4 + 8/16 = 4.50 — one 8-bit scale over 16 elements (smaller block, a touch more overhead)',
    },
    // captions
    why: (
      <>
        Why a 4-bit element can never stand alone: fp4 E2M1 has just{' '}
        <strong>1 sign · 2 exponent · 1 mantissa</strong> bit, so on its own it has almost no{' '}
        <strong>dynamic range</strong> — the code carries only a <em>relative</em> value. The shared{' '}
        <strong>scale</strong> supplies the magnitude, so the block as a whole spans a wide range while every element
        stays 4 bits. That is what <strong>microscaling (MX)</strong> means.
      </>
    ),
    whyFp8: (
      <>
        <strong>MXFP8</strong> is the gentle end of the same recipe. <code>fp8</code> E4M3 already carries real{' '}
        <strong>dynamic range</strong> (up to 448), so it is not <em>borrowing</em> range — the shared{' '}
        <span className="font-mono">E8M0</span> scale just re-centres each block of <strong>32</strong> values, staying
        near-lossless. <strong>Microscaling (MX)</strong> applies one rule across fp8, fp6 and fp4 — only the element shrinks.
      </>
    ),
    tradeoff: (
      <>
        One recipe, three widths. <strong>MXFP8</strong> wraps 32 <code>fp8</code> values in one power-of-two{' '}
        <span className="font-mono">E8M0</span> scale (≈ <span className="font-mono">8.25</span> bits/elem, near-lossless).{' '}
        <strong>MXFP4</strong> uses the same 32-element <span className="font-mono">E8M0</span> block on <code>fp4</code>{' '}
        (≈ <span className="font-mono">4.25</span>), where the shared scale is what makes 4-bit usable at all.{' '}
        <strong>NVFP4</strong> tightens the block to <strong>16</strong> elements, swaps in a richer float{' '}
        <span className="font-mono">E4M3</span> block scale, and adds a per-tensor <span className="font-mono">FP32</span>{' '}
        scale on top (≈ <span className="font-mono">4.50</span>). Smaller blocks and a float scale fit local data more
        tightly — <em>less rounding error</em> — for a touch more overhead. The error difference is real but qualitative;
        the course gives no measured number, so none is drawn here.
      </>
    ),
    aria: {
      mxfp8:
        'MXFP8 block: 32 fp8 E4M3 elements share one E8M0 power-of-two scale; effective storage 8.25 bits per element.',
      mxfp4:
        'MXFP4 block: 32 fp4 E2M1 elements all share one E8M0 power-of-two scale; effective storage 4.25 bits per element.',
      nvfp4:
        'NVFP4 block: 16 fp4 E2M1 elements share one FP8 E4M3 float block scale, with a single FP32 per-tensor scale above them; effective storage 4.50 bits per element.',
    },
  },
  zh: {
    heading: 'Block 格式：每个 block 共享一个 scale（MX 家族）',
    modeLabel: '格式',
    pickAria: (m: string) => `显示 ${m} 的 block 布局`,
    elemHeading: (n: number, elem: string, fields: string) => `${n} 个 ${elem} 元素（${fields}）`,
    blockScaleHeading: 'per-block scale',
    tensorScaleHeading: 'per-tensor scale（整个 tensor 共用一个）',
    e8m0Title: 'E8M0 scale',
    e8m0Body: '8 exponent · 0 mantissa → 纯粹的 2 的幂',
    e8m0Range: '2^-127 … 2^127 · 0xFF = NaN',
    e4m3Title: 'FP8 E4M3 block scale',
    e4m3Body: '1 sign · 4 exponent · 3 mantissa → 真正的浮点',
    e4m3Note: '有 mantissa → 落在 2 的幂之间（更细）',
    fp32Title: 'FP32 per-tensor scale',
    fp32Body: '一个全局浮点 scale，所有 block 共用',
    formulaMx: 'value = element × 2^scale',
    formulaMxSub: '一个共享 scale  ·  v = X · Pᵢ',
    formulaNv: 'value = fp32_tensor_scale × e4m3_block_scale × element',
    formulaNvSub: '两级 scaling',
    sharedLabel: '整个 block 共享',
    effHeading: '有效存储',
    effValue: (b: number) => `${b.toFixed(2)} bits/elem`,
    eff: {
      mxfp8: '8 + 8/32 = 8.25——一个 8-bit 的 E8M0 scale 摊到 32 个 fp8 值上（near-lossless 的那一端）',
      mxfp4: '4 + 8/32 = 4.25——一个 8-bit 的 scale 摊到 32 个 fp4 元素上',
      nvfp4: '4 + 8/16 = 4.50——一个 8-bit 的 scale 摊到 16 个元素上（block 更小，开销略高）',
    },
    why: (
      <>
        为什么 4-bit 的元素不能单独存在：fp4 E2M1 只有{' '}
        <strong>1 sign · 2 exponent · 1 mantissa</strong> 位，单独看几乎没有 <strong>dynamic range</strong>——这段
        编码只携带一个<em>相对</em>值。共享的 <strong>scale</strong> 提供量级，于是整个 block 能覆盖很宽的范围，而每个
        元素仍然只占 4 bit。这就是 <strong>microscaling（MX）</strong> 的含义。
      </>
    ),
    whyFp8: (
      <>
        <strong>MXFP8</strong> 是同一套 recipe 里最温和的一端。<code>fp8</code> E4M3 本身就带着真正的{' '}
        <strong>dynamic range</strong>（最大到 448），所以它并不是在<em>借</em>范围——共享的{' '}
        <span className="font-mono">E8M0</span> scale 只是把每个 <strong>32</strong> 值的 block 重新居中，几乎无损。{' '}
        <strong>microscaling（MX）</strong> 用同一条规则覆盖 fp8、fp6 和 fp4——变的只是元素的宽度。
      </>
    ),
    tradeoff: (
      <>
        一套 recipe，三种宽度。<strong>MXFP8</strong> 把 32 个 <code>fp8</code> 值装进一个 2-的-幂的{' '}
        <span className="font-mono">E8M0</span> scale（≈ <span className="font-mono">8.25</span> bits/elem，near-lossless）。{' '}
        <strong>MXFP4</strong> 在 <code>fp4</code> 上用同样的 32-元素 <span className="font-mono">E8M0</span> block（≈{' '}
        <span className="font-mono">4.25</span>），这里共享 scale 正是让 4-bit 能用起来的关键。<strong>NVFP4</strong> 把 block
        收紧到 <strong>16</strong> 个元素，换上更丰富的浮点 <span className="font-mono">E4M3</span> block scale，再在上面叠一个
        per-tensor 的 <span className="font-mono">FP32</span> scale（≈ <span className="font-mono">4.50</span>）。更小的 block
        加浮点 scale 能更贴合局部数据——<em>舍入误差更小</em>——代价是开销略高。误差差异真实但只是定性的；课程没有给出测量数字，所以这里也不画。
      </>
    ),
    aria: {
      mxfp8: 'MXFP8 block：32 个 fp8 E4M3 元素共享一个 E8M0 的 2 的幂 scale；有效存储为每元素 8.25 bit。',
      mxfp4: 'MXFP4 block：32 个 fp4 E2M1 元素共享一个 E8M0 的 2 的幂 scale；有效存储为每元素 4.25 bit。',
      nvfp4:
        'NVFP4 block：16 个 fp4 E2M1 元素共享一个 FP8 E4M3 浮点 block scale，上方还有一个 FP32 的 per-tensor scale；有效存储为每元素 4.50 bit。',
    },
  },
} as const;

// ── SVG geometry. EN labels are wider than zh, so the viewBox is sized for EN.
// Cells are laid out 16-per-row (32-elem blocks = 2 rows of 16, NVFP4 = 1 row of
// 16) so the cell grid never grows wider than the scale boxes below it. Wide
// section labels sit ABOVE their content, never wedged beside it. ──────────────
const VB_W = 580;
const VB_H = 330;
const PAD_L = 14;
const PAD_R = 14;
const TRACK_X = PAD_L;
const TRACK_W = VB_W - PAD_L - PAD_R; // 552

// cell grid
const COLS = 16;
const CELL_GAP = 4;
const CELL_W = (TRACK_W - CELL_GAP * (COLS - 1)) / COLS; // ≈ 30.75
const CELL_H = 22;
const GRID_LABEL_Y = 16;
const GRID_Y = 24; // top of first cell row

// scale boxes
const BOX_H = 50;
const FORMULA_Y = 318; // baseline for the reconstruction formula

function Cell({ x, y, fp8 }: { x: number; y: number; fp8?: boolean }) {
  return (
    <g>
      <rect x={x} y={y} width={CELL_W} height={CELL_H} rx={3} style={{ fill: 'var(--card)', stroke: 'var(--border)' }} strokeWidth={1} />
      {/* a tiny sign|exp|mant tick at a uniform 3px/bit so the cell honestly reads as its
          bit split: fp4 1+2+1 → 3/6/3 px · fp8 1+4+3 → 3/12/9 px */}
      <rect x={x + 3} y={y + CELL_H - 6} width={3} height={3} style={{ fill: 'var(--foreground)' }} />
      <rect x={x + 7} y={y + CELL_H - 6} width={fp8 ? 12 : 6} height={3} style={{ fill: 'var(--primary)' }} />
      <rect x={x + (fp8 ? 20 : 14)} y={y + CELL_H - 6} width={fp8 ? 9 : 3} height={3} style={{ fill: 'var(--muted-foreground)' }} />
    </g>
  );
}

export function MxBlockAnatomy() {
  const copy = COPY[useLocale()];
  const [mode, setMode] = React.useState<Mode>('mxfp4');
  const info = INFO[mode];
  // singleScale = the OCP MX path (mxfp8 / mxfp4): one E8M0 box + simple formula.
  // !singleScale = NVFP4: FP32 per-tensor box + E4M3 per-block box + two-level formula.
  const singleScale = info.scale === 'e8m0';
  const isFp8 = info.elemBits === 8;

  // cell rows: 32-elem block → 2 rows of 16; 16-elem block → 1 row of 16.
  const rows = info.block / COLS; // 2 or 1
  const gridBottom = GRID_Y + rows * (CELL_H + CELL_GAP) - CELL_GAP;

  // The arrows from the grid converge on a single scale box centered below.
  const blockBoxW = 232;
  const blockBoxX = TRACK_X + (TRACK_W - blockBoxW) / 2;
  // NVFP4 has a per-tensor FP32 box above the per-block box; the MX path just the one.
  const tensorBoxY = gridBottom + 16;
  const tensorBoxH = 40;
  const blockBoxY = singleScale ? gridBottom + 30 : tensorBoxY + tensorBoxH + 18;
  const convergeX = blockBoxX + blockBoxW / 2;

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + format toggle */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>
        <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
          <span className="hidden sm:inline">{copy.modeLabel}:</span>
          <div className="flex overflow-hidden rounded border border-border">
            {MODES.map((m, i) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                aria-pressed={mode === m}
                aria-label={copy.pickAria(m)}
                className={[
                  'px-2.5 py-1 font-mono text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                  i > 0 ? 'border-l border-border' : '',
                ].join(' ')}
                style={{
                  background: mode === m ? 'var(--primary)' : 'transparent',
                  color: mode === m ? 'var(--background)' : 'var(--muted-foreground)',
                }}
              >
                {m}
              </button>
            ))}
          </div>
        </div>
      </div>

      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.aria[mode]}>
        {/* ── element grid label + cells ── */}
        <text x={TRACK_X} y={GRID_LABEL_Y} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px] font-medium">
          {copy.elemHeading(info.block, info.elem, info.elemFields)}
        </text>
        {Array.from({ length: info.block }).map((_, i) => {
          const r = Math.floor(i / COLS);
          const c = i % COLS;
          return <Cell key={i} x={TRACK_X + c * (CELL_W + CELL_GAP)} y={GRID_Y + r * (CELL_H + CELL_GAP)} fp8={isFp8} />;
        })}

        {/* ── convergence arrow: every cell shares ONE scale ── */}
        {/* a single funnel line from the grid's bottom-center down to the scale box */}
        <line
          x1={convergeX}
          y1={gridBottom}
          x2={convergeX}
          y2={(singleScale ? blockBoxY : tensorBoxY) - 6}
          style={{ stroke: 'var(--muted-foreground)', opacity: 0.7 }}
          strokeWidth={1.5}
          markerEnd="url(#mxArrow)"
        />
        <text x={convergeX + 8} y={gridBottom + 14} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
          {copy.sharedLabel}
        </text>
        <defs>
          <marker id="mxArrow" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" style={{ fill: 'var(--muted-foreground)' }} />
          </marker>
        </defs>

        {/* ── NVFP4 only: per-tensor FP32 scale box (above the per-block box) ── */}
        {!singleScale ? (
          <>
            <text x={TRACK_X} y={tensorBoxY - 4} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px] font-medium">
              {copy.tensorScaleHeading}
            </text>
            <rect
              x={blockBoxX}
              y={tensorBoxY}
              width={blockBoxW}
              height={tensorBoxH}
              rx={5}
              style={{ fill: 'var(--card)', stroke: EMERALD }}
              strokeWidth={1.5}
            />
            <text x={blockBoxX + 12} y={tensorBoxY + 17} style={{ fill: 'var(--foreground)' }} className="font-mono text-[11px] font-semibold">
              {copy.fp32Title}
            </text>
            <text x={blockBoxX + 12} y={tensorBoxY + 32} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
              {copy.fp32Body}
            </text>
            {/* connector from tensor box down to the per-block box */}
            <line
              x1={convergeX}
              y1={tensorBoxY + tensorBoxH}
              x2={convergeX}
              y2={blockBoxY - 6}
              style={{ stroke: 'var(--muted-foreground)', opacity: 0.7 }}
              strokeWidth={1.5}
              markerEnd="url(#mxArrow)"
            />
          </>
        ) : null}

        {/* ── per-block scale box (E8M0 for the MX path, E4M3 for NVFP4) ── */}
        <text x={TRACK_X} y={blockBoxY - 4} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px] font-medium">
          {copy.blockScaleHeading}
        </text>
        <rect
          x={blockBoxX}
          y={blockBoxY}
          width={blockBoxW}
          height={BOX_H}
          rx={5}
          style={{ fill: 'var(--card)', stroke: 'var(--primary)' }}
          strokeWidth={1.5}
        />
        <text x={blockBoxX + 12} y={blockBoxY + 17} style={{ fill: 'var(--foreground)' }} className="font-mono text-[11px] font-semibold">
          {singleScale ? copy.e8m0Title : copy.e4m3Title}
        </text>
        <text x={blockBoxX + 12} y={blockBoxY + 31} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
          {singleScale ? copy.e8m0Body : copy.e4m3Body}
        </text>
        <text x={blockBoxX + 12} y={blockBoxY + 44} style={{ fill: 'var(--muted-foreground)' }} className="font-mono text-[10px]">
          {singleScale ? copy.e8m0Range : copy.e4m3Note}
        </text>

        {/* ── effective-storage readout (top-right, derived & exact) ── */}
        <text
          x={TRACK_X + TRACK_W}
          y={GRID_LABEL_Y}
          textAnchor="end"
          style={{ fill: 'var(--foreground)' }}
          className="font-mono text-[11px] font-semibold"
        >
          {copy.effValue(info.effBits)}
        </text>

        {/* ── reconstruction formula (bottom, centered) ── */}
        <text x={VB_W / 2} y={FORMULA_Y - 14} textAnchor="middle" style={{ fill: 'var(--foreground)' }} className="font-mono text-[11px] font-semibold">
          {singleScale ? copy.formulaMx : copy.formulaNv}
        </text>
        <text x={VB_W / 2} y={FORMULA_Y} textAnchor="middle" style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
          {singleScale ? copy.formulaMxSub : copy.formulaNvSub}
        </text>
      </svg>

      {/* effective-storage line */}
      <p className="text-[12px] text-muted-foreground">
        <span className="text-xs uppercase tracking-wider text-muted-foreground">{copy.effHeading}: </span>
        <span className="font-mono text-foreground">{copy.effValue(info.effBits)}</span> — {copy.eff[mode]}
      </p>

      {/* why this element needs a shared scale (fp4) / why MXFP8 is the gentle end (fp8) */}
      <p className="border-t border-border/60 pt-3 text-[12px] text-muted-foreground">{isFp8 ? copy.whyFp8 : copy.why}</p>

      {/* MXFP8 / MXFP4 / NVFP4 family trade-off */}
      <p className="text-[12px] text-muted-foreground">{copy.tradeoff}</p>
    </div>
  );
}
