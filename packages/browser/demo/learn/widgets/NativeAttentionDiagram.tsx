import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * NativeAttentionDiagram — chapter 4 deep-dive "Flash Attention", the
 * "Does your in-browser Qwen use it?" section.
 *
 * Self-contained: NO worker, NO model, NO WASM — pure SVG + React state, so it
 * server-renders for crawlers (createRoot replaces it on boot) and is robust
 * while the model downloads.
 *
 * It makes the NATIVE WebGPU implementation visible: the SDPA dispatch is a
 * runtime ROUTER. Every attention call carries two shapes — Tq (query rows)
 * and B·H (how many GPU workgroups the fused kernel would launch). An occupancy
 * gate decides which kernel actually runs:
 *
 *   gate true  ⟺  Tq == 1 AND B·H < 32   → decomposed matmul→softmax→matmul
 *   otherwise                            → fused FlashAttention-2 tile kernel
 *
 * This is the exact condition in mlx/backend/webgpu/sdpa.cpp (the Tq==1
 * occupancy gate, b*h<32). Toggle Prefill vs Decode and watch the signal flow
 * through the gate to the kernel that really runs:
 *   • Prefill (Tq>1)            → gate false → fused flash kernel.
 *   • Decode  (Tq=1, B·H=8<32)  → gate true  → decomposed path (~90% faster
 *                                  here, because it fans out far more workgroups).
 * Honors prefers-reduced-motion (no traveling signal; static highlight).
 */

const HEADS = 8; // Qwen3.5-0.8B query heads on the full-attention layers (B=1 → B·H=8).
const GATE_BH = 32; // occupancy threshold in sdpa.cpp: B·H < 32 routes Tq==1 to decomposed.
const PREFILL_TQ = 64; // illustrative prompt length (any Tq > 1 takes the fused path).

type Mode = 'prefill' | 'decode';

// ── Per-locale copy. COPY.en strings are the original English, verbatim.
// Code-ish pipeline labels (kernel names, mono gate condition, step chips)
// deliberately stay English in both locales. ─────────────────────────────────
const COPY = {
  en: {
    title: 'The native dispatch — which kernel actually runs',
    modePrefill: 'Prefill · digest prompt',
    modeDecode: 'Decode · one token',
    ariaLabel:
      'Diagram of the WebGPU SDPA dispatch routing an attention call through an occupancy gate to either the fused FlashAttention-2 kernel during prefill or the decomposed matmul-softmax-matmul path during single-token decode.',
    attentionCall: 'attention call',
    queryRows: 'query rows',
    headsBatch: 'heads × batch',
    occupancyGate: 'occupancy gate',
    fusedLine1: 'one dispatch · tiles Q,K,V',
    fusedLine2: 'online softmax m, ℓ, o in SRAM',
    runsInPrefill: 'runs in PREFILL',
    decompTitle: 'Decomposed path',
    fansOut: 'fans out 100+ workgroups',
    runsInDecode: 'runs in DECODE',
    captionDecode: (
      <>
        <strong>Decode</strong> generates one token, so <span className="font-mono">Tq = 1</span>. The fused kernel would
        launch only <span className="font-mono">B·H = {HEADS}</span> workgroups and leave most of the chip idle — so the
        gate is <span className="font-mono text-primary">true</span> and routes decode to the decomposed path, which fans
        out far more work and measured <strong>~90% faster here</strong>.
      </>
    ),
    captionPrefill: (
      <>
        <strong>Prefill</strong> digests the whole prompt at once, so <span className="font-mono">Tq &gt; 1</span>. The
        gate is <span className="font-mono text-primary">false</span> and the real <strong>fused flash-attention
        kernel</strong> runs — exactly the online-softmax recurrence above, never writing the N×N matrix to HBM. With
        many query rows the tile kernel launches <span className="font-mono">⌈Tq/16⌉·B·H</span> workgroups — far more
        than decode&apos;s {HEADS}, so the GPU stays busy.
      </>
    ),
    gateLabel: 'gate:',
    layersNote: 'Only 6 of 24 layers reach this gate — the other 18 are GatedDeltaNet (linear).',
    debugNote: (
      <>
        Default route shown. A debug switch (<span className="font-mono">?sdpa_fallback=1</span>) forces every call — prefill
        and decode — through the decomposed path, ahead of this gate.
      </>
    ),
  },
  zh: {
    title: '原生分派——真正运行的是哪个内核',
    modePrefill: 'Prefill · 消化提示词',
    modeDecode: 'Decode · 单个 token',
    ariaLabel:
      'WebGPU SDPA 分派示意图：注意力调用经过占用率闸门，prefill 阶段路由到融合的 FlashAttention-2 内核，逐 token 解码则路由到分解的 matmul-softmax-matmul 路径。',
    attentionCall: '注意力调用',
    queryRows: '行 query',
    headsBatch: '个头 × batch',
    occupancyGate: '占用率闸门',
    fusedLine1: '一次分发 · 给 Q,K,V 分块',
    fusedLine2: '在 SRAM 里在线 softmax m, ℓ, o',
    runsInPrefill: '在 PREFILL 运行',
    decompTitle: '分解路径',
    fansOut: '撒出 100+ 个 workgroup',
    runsInDecode: '在 DECODE 运行',
    captionDecode: (
      <>
        <strong>解码</strong>一次只生成一个 token，所以 <span className="font-mono">Tq = 1</span>。融合内核只会启动{' '}
        <span className="font-mono">B·H = {HEADS}</span> 个 workgroup，让芯片大部分闲置——于是闸门为{' '}
        <span className="font-mono text-primary">true</span>
        ，把解码路由到分解路径；它能撒出多得多的并行工作，在这里实测<strong>快约 90%</strong>。
      </>
    ),
    captionPrefill: (
      <>
        <strong>Prefill</strong> 一口气消化整条提示词，所以 <span className="font-mono">Tq &gt; 1</span>。闸门为{' '}
        <span className="font-mono text-primary">false</span>，真正的<strong>融合 flash-attention 内核</strong>
        运行——正是上文的在线 softmax 递推，从不把 N×N 矩阵写进 HBM。query 行一多，分块内核会启动{' '}
        <span className="font-mono">⌈Tq/16⌉·B·H</span> 个 workgroup——远多于解码的 {HEADS} 个，GPU 始终保持忙碌。
      </>
    ),
    gateLabel: '闸门：',
    layersNote: '24 层中只有 6 层会到达这个闸门——其余 18 层是 GatedDeltaNet（线性）。',
    debugNote: (
      <>
        图中为默认路由。调试开关（<span className="font-mono">?sdpa_fallback=1</span>
        ）会在此闸门之前，把每一次调用——prefill 与解码——强制走分解路径。
      </>
    ),
  },
} as const;

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = React.useState<boolean>(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });
  React.useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return;
    const mql = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, []);
  return reduced;
}

// ── SVG geometry ────────────────────────────────────────────────────────────
const VB_W = 760;
const VB_H = 300;
const MID_Y = 150;

const IN = { x: 20, y: 112, w: 150, h: 76 };
const GATE = { x: 252, y: 120, w: 140, h: 60 };
const FUSED = { x: 524, y: 34, w: 216, h: 104 };
const DECOMP = { x: 524, y: 168, w: 216, h: 114 };

const IN_EXIT: Pt = [IN.x + IN.w, MID_Y];
const GATE_IN: Pt = [GATE.x, MID_Y];
const GATE_OUT: Pt = [GATE.x + GATE.w, MID_Y];
const FUSED_IN: Pt = [FUSED.x, FUSED.y + FUSED.h / 2];
const DECOMP_IN: Pt = [DECOMP.x, DECOMP.y + DECOMP.h / 2];

type Pt = [number, number];

/** Point a fraction `t` (0..1) along a polyline, by cumulative length. */
function pointAt(pts: Pt[], t: number): Pt {
  const segs: number[] = [];
  let total = 0;
  for (let i = 0; i < pts.length - 1; i++) {
    const len = Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]);
    segs.push(len);
    total += len;
  }
  let d = t * total;
  for (let i = 0; i < segs.length; i++) {
    if (d <= segs[i] || i === segs.length - 1) {
      const r = segs[i] === 0 ? 0 : d / segs[i];
      return [pts[i][0] + (pts[i + 1][0] - pts[i][0]) * r, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * r];
    }
    d -= segs[i];
  }
  return pts[pts.length - 1];
}

export function NativeAttentionDiagram() {
  const copy = COPY[useLocale()];
  const reducedMotion = usePrefersReducedMotion();
  const [mode, setMode] = React.useState<Mode>('decode'); // decode is the surprising case the prose centers on
  const [t, setT] = React.useState(0);

  React.useEffect(() => {
    if (reducedMotion) return;
    const id = window.setInterval(() => setT((x) => (x + 0.016) % 1), 45);
    return () => window.clearInterval(id);
  }, [reducedMotion]);

  const isDecode = mode === 'decode';
  const tq = isDecode ? 1 : PREFILL_TQ;
  const gateTrue = tq === 1 && HEADS < GATE_BH; // B=1 → B·H=HEADS
  const active: 'fused' | 'decomp' = gateTrue ? 'decomp' : 'fused';
  const activeIn = active === 'fused' ? FUSED_IN : DECOMP_IN;

  // The signal route: input → gate → the kernel that actually runs.
  const route: Pt[] = [IN_EXIT, GATE_IN, GATE_OUT, activeIn];
  const sig = pointAt(route, t);
  // Hide the dot while it is "inside" the gate box, so it reads as enter→route→exit.
  const sigHidden = reducedMotion || (sig[0] > GATE.x - 2 && sig[0] < GATE.x + GATE.w + 2);

  const gateEval = isDecode ? `Tq = 1  and  B·H = ${HEADS} < ${GATE_BH}  →  true` : `Tq = ${PREFILL_TQ} ≠ 1  →  false`;

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + mode toggle */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <div className="inline-flex overflow-hidden rounded border border-border">
          {(['prefill', 'decode'] as Mode[]).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              aria-pressed={m === mode}
              className={[
                'px-2.5 py-1 text-[11px] font-medium transition-colors',
                m === mode ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground',
              ].join(' ')}
            >
              {m === 'prefill' ? copy.modePrefill : copy.modeDecode}
            </button>
          ))}
        </div>
      </div>

      {/* The diagram */}
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className="w-full"
        role="img"
        aria-label={copy.ariaLabel}
      >
        {/* ── Connectors (drawn first, under the boxes) ── */}
        <Connector from={IN_EXIT} to={GATE_IN} active />
        <Connector from={GATE_OUT} to={FUSED_IN} active={active === 'fused'} />
        <Connector from={GATE_OUT} to={DECOMP_IN} active={active === 'decomp'} />

        {/* ── Input: the SDPA call and its shapes ── */}
        <rect
          x={IN.x}
          y={IN.y}
          width={IN.w}
          height={IN.h}
          rx={10}
          style={{ fill: 'var(--card)', stroke: 'var(--border)' }}
          strokeWidth={1.5}
        />
        <text x={IN.x + 14} y={IN.y + 24} style={{ fill: 'var(--foreground)' }} className="text-[13px] font-semibold">
          {copy.attentionCall}
        </text>
        <text x={IN.x + 14} y={IN.y + 45} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
          <tspan className="font-mono" style={{ fill: 'var(--foreground)' }}>
            Tq = {tq}
          </tspan>{' '}
          {copy.queryRows}
        </text>
        <text x={IN.x + 14} y={IN.y + 63} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
          <tspan className="font-mono" style={{ fill: 'var(--foreground)' }}>
            B·H = {HEADS}
          </tspan>{' '}
          {copy.headsBatch}
        </text>

        {/* ── The occupancy gate ── */}
        <rect
          x={GATE.x}
          y={GATE.y}
          width={GATE.w}
          height={GATE.h}
          rx={10}
          style={{ fill: 'var(--muted)', stroke: 'var(--primary)' }}
          strokeWidth={1.5}
        />
        <text
          x={GATE.x + GATE.w / 2}
          y={GATE.y + 25}
          textAnchor="middle"
          style={{ fill: 'var(--foreground)' }}
          className="text-[12px] font-semibold"
        >
          {copy.occupancyGate}
        </text>
        <text
          x={GATE.x + GATE.w / 2}
          y={GATE.y + 44}
          textAnchor="middle"
          style={{ fill: 'var(--muted-foreground)' }}
          className="font-mono text-[10px]"
        >
          Tq==1 &amp; B·H&lt;{GATE_BH} ?
        </text>

        {/* ── Fused FlashAttention-2 kernel (taken when gate is false) ── */}
        <KernelBox box={FUSED} active={active === 'fused'}>
          <text x={FUSED.x + 14} y={FUSED.y + 25} style={{ fill: 'var(--foreground)' }} className="text-[13px] font-semibold">
            Fused FlashAttention-2
          </text>
          <text x={FUSED.x + 14} y={FUSED.y + 45} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
            {copy.fusedLine1}
          </text>
          <text x={FUSED.x + 14} y={FUSED.y + 61} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
            {copy.fusedLine2}
          </text>
          <Pill x={FUSED.x + 14} y={FUSED.y + 74} label={copy.runsInPrefill} on={active === 'fused'} />
        </KernelBox>

        {/* ── Decomposed path (taken when gate is true) ── */}
        <KernelBox box={DECOMP} active={active === 'decomp'}>
          <text x={DECOMP.x + 14} y={DECOMP.y + 25} style={{ fill: 'var(--foreground)' }} className="text-[13px] font-semibold">
            {copy.decompTitle}
          </text>
          <g transform={`translate(${DECOMP.x + 14}, ${DECOMP.y + 38})`}>
            <Step x={0} label="matmul QKᵀ" on={active === 'decomp'} />
            <Arrow x={62} />
            <Step x={74} label="softmax" on={active === 'decomp'} />
            <Arrow x={124} />
            <Step x={136} label="matmul ·V" on={active === 'decomp'} />
          </g>
          <text x={DECOMP.x + 14} y={DECOMP.y + 76} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
            {copy.fansOut}
          </text>
          <Pill x={DECOMP.x + 14} y={DECOMP.y + 86} label={copy.runsInDecode} on={active === 'decomp'} />
        </KernelBox>

        {/* ── The traveling signal: input → gate → the kernel that runs ── */}
        {!sigHidden ? (
          <circle cx={sig[0]} cy={sig[1]} r={5} style={{ fill: 'var(--primary)' }}>
          </circle>
        ) : null}
      </svg>

      {/* Caption — what the current mode does */}
      <p className="min-h-[3.25rem] text-[13px] text-foreground/90">
        {isDecode ? copy.captionDecode : copy.captionPrefill}
      </p>

      {/* Gate evaluation + grounding */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border/60 pt-3">
        <div className="text-[11px] text-muted-foreground">
          {copy.gateLabel} <span className="font-mono text-foreground/90">{gateEval}</span> →{' '}
          <span className="font-mono text-primary">{active === 'decomp' ? 'decomposed' : 'fused'}</span>
        </div>
        <div className="text-[10px] text-muted-foreground/70">{copy.layersNote}</div>
      </div>
      <p className="text-[10px] text-muted-foreground/70">{copy.debugNote}</p>
    </div>
  );
}

/** A connector line with a small right-pointing arrowhead at its destination. */
function Connector({ from, to, active }: { from: Pt; to: Pt; active: boolean }) {
  return (
    <g style={{ opacity: active ? 1 : 0.35 }}>
      <line
        x1={from[0]}
        y1={from[1]}
        x2={to[0]}
        y2={to[1]}
        style={{ stroke: active ? 'var(--primary)' : 'var(--border)' }}
        strokeWidth={active ? 2 : 1.5}
        strokeDasharray={active ? undefined : '5 4'}
      />
      <polygon
        points={`${to[0]},${to[1]} ${to[0] - 9},${to[1] - 5} ${to[0] - 9},${to[1] + 5}`}
        style={{ fill: active ? 'var(--primary)' : 'var(--border)' }}
      />
    </g>
  );
}

/** A kernel container box — highlighted when it is the path that runs. */
function KernelBox({
  box,
  active,
  children,
}: {
  box: { x: number; y: number; w: number; h: number };
  active: boolean;
  children: React.ReactNode;
}) {
  return (
    <g style={{ opacity: active ? 1 : 0.5 }}>
      <rect
        x={box.x}
        y={box.y}
        width={box.w}
        height={box.h}
        rx={10}
        style={{
          fill: active ? 'var(--primary)' : 'var(--card)',
          fillOpacity: active ? 0.08 : 1,
          stroke: active ? 'var(--primary)' : 'var(--border)',
        }}
        strokeWidth={active ? 2 : 1.5}
      />
      {children}
    </g>
  );
}

/** A small status pill (PREFILL / DECODE tag). */
function Pill({ x, y, label, on }: { x: number; y: number; label: string; on: boolean }) {
  const w = label.length * 6.4 + 16;
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={17}
        rx={8.5}
        style={{ fill: on ? 'var(--primary)' : 'var(--muted)', stroke: on ? 'var(--primary)' : 'var(--border)' }}
        strokeWidth={1}
      />
      <text
        x={x + w / 2}
        y={y + 12}
        textAnchor="middle"
        style={{ fill: on ? 'var(--bg)' : 'var(--muted-foreground)' }}
        className="text-[9px] font-semibold tracking-wide"
      >
        {label}
      </text>
    </g>
  );
}

/** One mini step chip inside the decomposed path. */
function Step({ x, label, on }: { x: number; label: string; on: boolean }) {
  return (
    <g>
      <rect
        x={x}
        y={0}
        width={58}
        height={20}
        rx={5}
        style={{
          fill: on ? 'var(--primary)' : 'var(--muted)',
          fillOpacity: on ? 0.16 : 1,
          stroke: on ? 'var(--primary)' : 'var(--border)',
        }}
        strokeWidth={1}
      />
      <text
        x={x + 29}
        y={14}
        textAnchor="middle"
        style={{ fill: on ? 'var(--primary)' : 'var(--muted-foreground)' }}
        className="text-[9px] font-medium"
      >
        {label}
      </text>
    </g>
  );
}

/** A tiny → between decomposed steps. */
function Arrow({ x }: { x: number }) {
  return (
    <text x={x} y={14} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
      →
    </text>
  );
}
