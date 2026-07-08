import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * LocalEdgeTradeoffs — the "Local & edge inference" sub-chapter: why run a SMALL
 * model on your OWN device at all? It names the cloud-datacenter ↔ local/edge
 * trade and shows what model sizes can run WHERE. The whole course demo IS edge
 * inference (Qwen3.5-0.8B, bf16, batch 1, in this browser via WebGPU), so the
 * "you are here" marker is locked at the small end.
 *
 * Self-contained and STATIC: NO worker, NO model, NO WASM, NO network — pure
 * React + Tailwind/SVG, so it server-renders for crawlers (createRoot replaces it
 * on boot) and is robust while the model downloads. The first render is
 * deterministic and identical server vs client; there is no time-dependent state
 * and no autoplay, so it needs no prefers-reduced-motion handling. The size
 * control is a pure stepped selector (no playback).
 *
 * Sources (verbatim where quoted): Inference Engineering (Kiely, Baseten 2026),
 * Ch3 "Local Inference" pp.89–91 — the four advantages (latency / independence /
 * privacy / cost), the four weaknesses (hardware / thermal / fragmented support /
 * battery), the RTX 5090 vs M3 Ultra capacity-vs-speed table (p.90), the model-
 * size→where-it-runs guidance (phones ≤1–2B; quantized 100B+ on high-end desktop
 * via Ollama/llama.cpp; Chromebooks/low-end run none; cloud = frontier 100s of B
 * to T), the "both working together" synthesis (p.91), and the WebLLM browser nod.
 *
 * LAYOUT SAFETY: the dense text comparison and the hardware table are HTML
 * (auto-layout) — only the model-size BAND is SVG, and it carries no text wider
 * than its band (tier labels live in HTML below it). EN labels are wider than zh,
 * so the one SVG was sized for EN.
 */

// Emerald accent: the edge side / where the edge "wins". Cloud stays neutral.
const EMERALD = '#10b981';

// ── Model-size steps (billions of params). Each maps to the SMALLEST tier that
// can host it. "you are here" is locked at the 0.8B step (this browser tab). ──
type Tier = 'phone' | 'laptopBf16' | 'laptopQuant' | 'cloud';

type SizeStep = {
  /** label shown on the size buttons */
  label: string;
  /** billions of parameters (for the log band position) */
  params: number;
  /** smallest tier that can realistically host it */
  tier: Tier;
};

const SIZES: SizeStep[] = [
  { label: '0.8B', params: 0.8, tier: 'phone' },
  { label: '3B', params: 3, tier: 'laptopBf16' },
  { label: '8B', params: 8, tier: 'laptopBf16' },
  { label: '70B', params: 70, tier: 'laptopQuant' },
  { label: '400B', params: 400, tier: 'cloud' },
];

// index 0 (0.8B) is the locked "you are here" marker — this very browser tab.
const YOU_ARE_HERE_IDX = 0;

// Comparison-axis identifiers (rows of the cloud ↔ edge table).
type Axis = 'latency' | 'privacy' | 'cost' | 'capability' | 'availability' | 'hardware';
const AXES: Axis[] = ['latency', 'privacy', 'cost', 'capability', 'availability', 'hardware'];

// Which side "wins" each axis. Neither side wins all of them — that IS the trade.
const WINNER: Record<Axis, 'cloud' | 'edge'> = {
  latency: 'edge',
  privacy: 'edge',
  cost: 'edge',
  capability: 'cloud',
  availability: 'edge',
  hardware: 'cloud',
};

// ── Per-locale copy. zh keeps the glossary terms in ENGLISH (cloud, datacenter,
// edge, local inference, on-device, latency, privacy, throughput, batch, WebGPU,
// GPU, VRAM, unified memory, bandwidth, quantized, bf16, MoE, token, Ollama,
// llama.cpp, WebLLM, H100, RTX 5090, M3 Ultra) with full-width punctuation; only
// the surrounding prose translates. Mirrors BatchingModesDiagram's zh tone. ──
const COPY = {
  en: {
    heading: 'Local & edge inference — the trade, made concrete',
    colCloud: 'Cloud datacenter',
    colEdge: 'Local / edge',
    colEdgeTag: 'you are here',
    winnerNote: 'Neither side wins every row — that is the trade.',
    axisName: {
      latency: 'latency',
      privacy: 'privacy',
      cost: 'cost',
      capability: 'model size / capability',
      availability: 'availability',
      hardware: 'hardware',
    } as Record<Axis, string>,
    cloudVal: {
      latency: 'network round-trip',
      privacy: 'data leaves your device',
      cost: 'per-token API · expensive GPUs',
      capability: 'frontier · 100B+ to trillions',
      availability: 'online-only',
      hardware: 'datacenter H100',
    } as Record<Axis, string>,
    edgeVal: {
      latency: 'on-device · zero round-trip',
      privacy: 'data never leaves the device',
      cost: 'free for the developer · $0',
      capability: 'small ≤~2B · quantized 100B+',
      availability: 'works offline',
      hardware: 'your VRAM / unified memory',
    } as Record<Axis, string>,
    winsLabel: 'wins',
    advHeading: 'Why edge wins (the book’s four)',
    weakHeading: 'What edge pays for (the book’s four)',
    advantages: [
      { k: 'zero network latency', v: 'saving tens or even hundreds of milliseconds' },
      { k: 'independence', v: 'no internet, server, or downtime to depend on' },
      { k: 'privacy', v: 'the end user’s data never leaves their device' },
      { k: 'cost', v: 'edge inference is free for the developer' },
    ],
    weaknesses: [
      { k: 'hardware', v: 'a fraction of the speed and power of datacenter GPUs' },
      { k: 'thermal constraints', v: 'phones and laptops throttle under sustained load' },
      { k: 'fragmented support', v: 'endless hardware × software combinations' },
      { k: 'battery life', v: 'inference drains laptop and phone batteries' },
    ],
    sizeHeading: 'Where can it run?',
    sizeHint: 'Step through model sizes — the smallest tier that can host it lights up.',
    sizeLabel: 'model size',
    bandAria: (label: string, tier: string) =>
      `Model-size band: at ${label} parameters, the smallest hosting tier is ${tier}.`,
    youAreHere: 'you are here',
    youAreHereDetail: '0.8B bf16 · this browser · WebGPU',
    tierName: {
      phone: 'phone',
      laptopBf16: 'your laptop · bf16',
      laptopQuant: 'your laptop · quantized',
      cloud: 'cloud only · frontier',
    } as Record<Tier, string>,
    tierNote: {
      phone: 'Phones struggle past one or two billion parameters — this is the phone sweet spot.',
      laptopBf16: 'Fits in your laptop’s memory at bf16, no quantization needed.',
      laptopQuant:
        'Aggressively quantized — quantization is the bridge that puts a 100B+ model on a high-end desktop (Ollama / llama.cpp). MoE helps too: fewer active params per request.',
      cloud:
        'Too big for any edge device — a frontier model lives on datacenter GPUs. Low-end laptops and Chromebooks run no meaningful local inference at all.',
    } as Record<Tier, string>,
    quantBridge: 'quantization unlocks big-on-small',
    hwHeading: 'Capacity vs speed (book p.90)',
    hwCols: { device: 'device', mem: 'memory', bw: 'bandwidth', price: 'approx. price' },
    hwRows: [
      { device: 'NVIDIA RTX 5090', mem: '32 GB', bw: '1,792 GB/s', price: '~$5,000', note: 'less memory, faster' },
      { device: 'Apple M3 Ultra', mem: '512 GB unified', bw: '819 GB/s', price: '~$10,000', note: 'far more capacity, slower' },
    ],
    hwCaption: 'Apple’s unified memory buys far more capacity at a slower speed; the 5090 buys less memory but more bandwidth.',
    synthHeading: 'The synthesis: it’s both',
    synthesis: (
      <>
        The future of inference isn’t local <em>or</em> cloud — it’s <strong>both working together</strong>: small models
        and quick queries on end-user devices, more demanding workloads on datacenter GPUs. Browser libraries like{' '}
        <span className="font-mono">WebLLM</span> and other cross-platform standards bring this mainstream — this course
        demo is exactly that.
      </>
    ),
    honestyHeading: 'You are here — this tab is edge inference',
    honesty: (
      <>
        This browser tab runs <strong>Qwen3.5-0.8B</strong> in <strong>bf16</strong>, <strong>batch 1</strong>, on{' '}
        <strong>WebGPU</strong>. It buys you <span style={{ color: EMERALD }}>privacy</span> (your text never leaves the
        tab), <span style={{ color: EMERALD }}>zero network latency</span>, and <span style={{ color: EMERALD }}>$0
        cost</span> — and pays for it in capability (a 0.8B model, not a frontier one) and speed (your GPU, not an H100).
        That is the trade, made concrete.
      </>
    ),
    ariaLabel:
      'Comparison of cloud-datacenter versus local/edge inference across six axes (latency, privacy, cost, model size, availability, hardware), with a stepped model-size control from 0.8B to 400B that lights the smallest tier able to host each size — phone, your laptop in bf16, your laptop quantized, or cloud-only — and a locked "you are here: 0.8B bf16 in this browser via WebGPU" marker at the small end.',
  },
  zh: {
    heading: 'local & edge inference——把这笔 trade 讲具体',
    colCloud: 'cloud datacenter',
    colEdge: 'local / edge',
    colEdgeTag: '你在这里',
    winnerNote: '没有哪一侧赢下每一行——这正是 trade。',
    axisName: {
      latency: 'latency',
      privacy: 'privacy',
      cost: 'cost',
      capability: 'model size / 能力',
      availability: '可用性',
      hardware: 'hardware',
    } as Record<Axis, string>,
    cloudVal: {
      latency: '一次网络往返',
      privacy: '数据离开你的设备',
      cost: '按 token 计费的 API · 昂贵的 GPU',
      capability: 'frontier · 100B+ 到万亿级',
      availability: '只能联网用',
      hardware: 'datacenter H100',
    } as Record<Axis, string>,
    edgeVal: {
      latency: 'on-device · 零往返',
      privacy: '数据永不离开设备',
      cost: '对开发者免费 · $0',
      capability: '小模型 ≤~2B · quantized 100B+',
      availability: '可离线运行',
      hardware: '你的 VRAM / unified memory',
    } as Record<Axis, string>,
    winsLabel: '胜',
    advHeading: 'edge 为何取胜（书里的四点）',
    weakHeading: 'edge 要付出什么（书里的四点）',
    advantages: [
      { k: 'zero network latency', v: '省下几十甚至上百毫秒' },
      { k: 'independence', v: '不依赖 internet、server，也不怕 downtime' },
      { k: 'privacy', v: '终端用户的数据永不离开其设备' },
      { k: 'cost', v: 'edge inference 对开发者免费' },
    ],
    weaknesses: [
      { k: 'hardware', v: '只有 datacenter GPU 速度与算力的一小部分' },
      { k: 'thermal constraints', v: '手机、笔电在持续负载下会降频' },
      { k: 'fragmented support', v: '无穷无尽的 hardware × software 组合' },
      { k: 'battery life', v: '推理会耗尽笔电和手机的电池' },
    ],
    sizeHeading: '它能跑在哪里？',
    sizeHint: '逐档切换模型大小——能容纳它的最小一档会亮起。',
    sizeLabel: 'model size',
    bandAria: (label: string, tier: string) =>
      `model size 区间：在 ${label} 参数时，能容纳它的最小一档是 ${tier}。`,
    youAreHere: '你在这里',
    youAreHereDetail: '0.8B bf16 · 这个浏览器 · WebGPU',
    tierName: {
      phone: 'phone',
      laptopBf16: '你的笔电 · bf16',
      laptopQuant: '你的笔电 · quantized',
      cloud: '只能上 cloud · frontier',
    } as Record<Tier, string>,
    tierNote: {
      phone: '手机超过一两 billion 参数就吃力——这是 phone 的甜区。',
      laptopBf16: '以 bf16 就能塞进你笔电的内存，无需 quantization。',
      laptopQuant:
        '激进 quantized——quantization 正是把 100B+ 模型搬上高端桌面的桥梁（Ollama / llama.cpp）。MoE 也有帮助：每次请求激活的参数更少。',
      cloud:
        '对任何 edge 设备都太大了——frontier 模型只能住在 datacenter GPU 上。低端笔电和 Chromebook 根本跑不动任何有意义的 local inference。',
    } as Record<Tier, string>,
    quantBridge: 'quantization 让大模型跑上小设备',
    hwHeading: 'capacity vs speed（书 p.90）',
    hwCols: { device: '设备', mem: 'memory', bw: 'bandwidth', price: '约价格' },
    hwRows: [
      { device: 'NVIDIA RTX 5090', mem: '32 GB', bw: '1,792 GB/s', price: '~$5,000', note: '内存更少、更快' },
      { device: 'Apple M3 Ultra', mem: '512 GB unified', bw: '819 GB/s', price: '~$10,000', note: '容量大得多、更慢' },
    ],
    hwCaption: 'Apple 的 unified memory 以更慢的速度换来大得多的容量；5090 则用更少的内存换来更高的 bandwidth。',
    synthHeading: '综合：答案是「两者皆是」',
    synthesis: (
      <>
        inference 的未来不是 local <em>或</em> cloud，而是 <strong>两者协同</strong>：小模型与快查询跑在终端设备上，更吃
        资源的工作负载交给 datacenter GPU。像 <span className="font-mono">WebLLM</span> 这样的浏览器库和其它跨平台标准
        把这一切带向主流——本课程的 demo 正是如此。
      </>
    ),
    honestyHeading: '你在这里——这个标签页就是 edge inference',
    honesty: (
      <>
        这个浏览器标签页以 <strong>bf16</strong>、<strong>batch 1</strong> 在 <strong>WebGPU</strong> 上运行{' '}
        <strong>Qwen3.5-0.8B</strong>。它为你换来 <span style={{ color: EMERALD }}>privacy</span>（你的文字永不离开标签
        页）、<span style={{ color: EMERALD }}>zero network latency</span> 和 <span style={{ color: EMERALD }}>$0
        成本</span>——代价是能力（一个 0.8B 模型，而非 frontier 模型）与速度（你的 GPU，而非 H100）。这就是这笔 trade，讲
        得很具体。
      </>
    ),
    ariaLabel:
      'cloud datacenter 与 local/edge inference 在六个维度（latency、privacy、cost、model size、可用性、hardware）上的对比，外加一个从 0.8B 到 400B 的逐档 model size 控件——它会点亮能容纳每个大小的最小一档：phone、你的笔电（bf16）、你的笔电（quantized）或只能上 cloud——以及一个锁定的「你在这里：0.8B bf16，在这个浏览器里通过 WebGPU 运行」标记，固定在最小一端。',
  },
} as const;

// ── Size-band SVG geometry. Only graphical content (no wide text in the band);
// the four tier names live in HTML below, so nothing here can overflow. The band
// is a log scale from 0.5B to 600B so the five steps spread out readably. ──
const VB_W = 720;
const VB_H = 70;
const BAND_X = 16;
const BAND_W = VB_W - BAND_X * 2;
const BAND_Y = 30;
const BAND_H = 16;
const P_MIN = 0.5;
const P_MAX = 600;
const LP_MIN = Math.log10(P_MIN);
const LP_MAX = Math.log10(P_MAX);

function bandX(params: number): number {
  const t = (Math.log10(params) - LP_MIN) / (LP_MAX - LP_MIN);
  return BAND_X + t * BAND_W;
}

// Tier color band boundaries (in params): phone ≤2 → laptop bf16 ≤~12 → laptop
// quantized ≤~120 → cloud beyond. Drawn as four colored segments behind the dots.
const TIER_BOUNDS: { tier: Tier; from: number; to: number }[] = [
  { tier: 'phone', from: P_MIN, to: 2 },
  { tier: 'laptopBf16', from: 2, to: 12 },
  { tier: 'laptopQuant', from: 12, to: 120 },
  { tier: 'cloud', from: 120, to: P_MAX },
];

function tierFill(tier: Tier, active: boolean): { fill: string; opacity: number } {
  if (tier === 'cloud') {
    // cloud stays neutral/primary
    return { fill: 'var(--primary)', opacity: active ? 0.28 : 0.08 };
  }
  // the edge-hostable tiers use emerald
  return { fill: EMERALD, opacity: active ? 0.3 : 0.08 };
}

function SizeBand({ idx }: { idx: number }) {
  const copy = COPY[useLocale()];
  const cur = SIZES[idx]!;
  const here = SIZES[YOU_ARE_HERE_IDX]!;

  return (
    <svg
      viewBox={`0 0 ${VB_W} ${VB_H}`}
      className="w-full"
      role="img"
      aria-label={copy.bandAria(cur.label, copy.tierName[cur.tier])}
    >
      {/* tier segments behind the dots; the active tier is brightened */}
      {TIER_BOUNDS.map((seg) => {
        const x0 = bandX(seg.from);
        const x1 = bandX(seg.to);
        const active = seg.tier === cur.tier;
        const st = tierFill(seg.tier, active);
        return (
          <rect
            key={seg.tier}
            x={x0}
            y={BAND_Y}
            width={Math.max(1, x1 - x0)}
            height={BAND_H}
            style={{ fill: st.fill, opacity: st.opacity }}
            stroke={active ? (seg.tier === 'cloud' ? 'var(--primary)' : EMERALD) : 'var(--border)'}
            strokeWidth={active ? 1.5 : 1}
            rx={3}
          />
        );
      })}

      {/* the size steps as dots along the band; the current one is filled */}
      {SIZES.map((s, i) => {
        const x = bandX(s.params);
        const isCur = i === idx;
        const onEdge = s.tier !== 'cloud';
        const color = onEdge ? EMERALD : 'var(--primary)';
        return (
          <g key={s.label}>
            <circle
              cx={x}
              cy={BAND_Y + BAND_H / 2}
              r={isCur ? 6 : 4}
              fill={isCur ? color : 'var(--background)'}
              stroke={color}
              strokeWidth={isCur ? 2 : 1.5}
            />
            {/* size label sits ABOVE its dot, centered — never wedged between boxes */}
            <text
              x={x}
              y={BAND_Y - 8}
              textAnchor="middle"
              style={{ fill: isCur ? 'var(--foreground)' : 'var(--muted-foreground)' }}
              className={isCur ? 'font-mono text-[11px] font-semibold' : 'font-mono text-[10px]'}
            >
              {s.label}
            </text>
          </g>
        );
      })}

      {/* locked "you are here" pin at the 0.8B step (this browser tab) */}
      <g>
        <line
          x1={bandX(here.params)}
          y1={BAND_Y + BAND_H + 2}
          x2={bandX(here.params)}
          y2={BAND_Y + BAND_H + 18}
          style={{ stroke: 'var(--foreground)' }}
          strokeWidth={1.5}
        />
        <text
          x={bandX(here.params)}
          y={VB_H - 4}
          textAnchor="start"
          style={{ fill: 'var(--foreground)' }}
          className="text-[9px] font-semibold"
        >
          🔒 {copy.youAreHere}
        </text>
      </g>
    </svg>
  );
}

export function LocalEdgeTradeoffs() {
  const copy = COPY[useLocale()];
  const [idx, setIdx] = React.useState(YOU_ARE_HERE_IDX);
  const cur = SIZES[idx]!;

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header */}
      <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>

      {/* ── Two-column comparison (HTML auto-layout = overflow-safe). ── */}
      <div className="overflow-hidden rounded-md border border-border/70">
        {/* column headers */}
        <div className="grid grid-cols-[7.5rem_1fr_1fr] items-stretch border-b border-border/70 text-[11px] font-semibold">
          <div className="bg-card/40 px-2 py-1.5" />
          <div className="border-l border-border/70 bg-card/40 px-2 py-1.5 text-muted-foreground">{copy.colCloud}</div>
          <div
            className="border-l px-2 py-1.5"
            style={{ borderColor: 'rgba(16,185,129,0.4)', background: 'rgba(16,185,129,0.07)', color: EMERALD }}
          >
            {copy.colEdge}
            <span className="ml-1 rounded border border-current/40 px-1 text-[9px] font-medium">{copy.colEdgeTag}</span>
          </div>
        </div>

        {/* axis rows */}
        {AXES.map((axis, i) => {
          const w = WINNER[axis];
          return (
            <div
              key={axis}
              className={[
                'grid grid-cols-[7.5rem_1fr_1fr] items-start text-[11px]',
                i > 0 ? 'border-t border-border/50' : '',
              ].join(' ')}
            >
              <div className="px-2 py-1.5 font-medium text-muted-foreground">{copy.axisName[axis]}</div>
              <div
                className="border-l border-border/70 px-2 py-1.5"
                style={w === 'cloud' ? { background: 'var(--muted)' } : undefined}
              >
                <span className="text-foreground/90">{copy.cloudVal[axis]}</span>
                {w === 'cloud' ? (
                  <span className="ml-1 inline-block align-middle text-[10px] font-semibold text-foreground/70">
                    ✓ {copy.winsLabel}
                  </span>
                ) : null}
              </div>
              <div
                className="border-l px-2 py-1.5"
                style={{
                  borderColor: 'rgba(16,185,129,0.4)',
                  background: w === 'edge' ? 'rgba(16,185,129,0.08)' : undefined,
                }}
              >
                <span className="text-foreground/90">{copy.edgeVal[axis]}</span>
                {w === 'edge' ? (
                  <span className="ml-1 inline-block align-middle text-[10px] font-semibold" style={{ color: EMERALD }}>
                    ✓ {copy.winsLabel}
                  </span>
                ) : null}
              </div>
            </div>
          );
        })}
      </div>
      <p className="text-[11px] italic text-muted-foreground/80">{copy.winnerNote}</p>

      {/* ── Advantages / weaknesses (the book's two lists of four). ── */}
      <div className="grid gap-2 sm:grid-cols-2">
        <div className="rounded-md border p-2.5" style={{ borderColor: 'rgba(16,185,129,0.4)', background: 'rgba(16,185,129,0.05)' }}>
          <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider" style={{ color: EMERALD }}>
            {copy.advHeading}
          </div>
          <ul className="space-y-1">
            {copy.advantages.map((a) => (
              <li key={a.k} className="text-[12px] text-foreground/90">
                <span className="font-semibold" style={{ color: EMERALD }}>
                  {a.k}
                </span>{' '}
                <span className="text-muted-foreground">— {a.v}</span>
              </li>
            ))}
          </ul>
        </div>
        <div className="rounded-md border border-border/70 bg-card/40 p-2.5">
          <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            {copy.weakHeading}
          </div>
          <ul className="space-y-1">
            {copy.weaknesses.map((w) => (
              <li key={w.k} className="text-[12px] text-foreground/90">
                <span className="font-semibold text-foreground/80">{w.k}</span>{' '}
                <span className="text-muted-foreground">— {w.v}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* ── "Where can it run?" stepped size control + band. ── */}
      <div className="space-y-2 rounded-md border border-border/70 bg-card/40 p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            {copy.sizeHeading}
          </div>
          <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
            <span className="hidden sm:inline">{copy.sizeLabel}:</span>
            <div className="flex overflow-hidden rounded border border-border">
              {SIZES.map((s, i) => (
                <button
                  key={s.label}
                  type="button"
                  onClick={() => setIdx(i)}
                  aria-pressed={idx === i}
                  aria-label={`${copy.sizeLabel} ${s.label} — ${copy.tierName[s.tier]}`}
                  className={[
                    'px-2 py-1 font-mono text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                    i > 0 ? 'border-l border-border' : '',
                  ].join(' ')}
                  style={{
                    background: idx === i ? 'var(--primary)' : 'transparent',
                    color: idx === i ? 'var(--background)' : 'var(--muted-foreground)',
                  }}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <p className="text-[11px] text-muted-foreground/80">{copy.sizeHint}</p>

        <SizeBand idx={idx} />

        {/* active tier readout */}
        <div className="flex flex-wrap items-center gap-2">
          <span
            className="inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-[11px] font-semibold"
            style={
              cur.tier === 'cloud'
                ? { borderColor: 'var(--primary)', color: 'var(--primary)', background: 'rgba(99,102,241,0.08)' }
                : { borderColor: EMERALD, color: EMERALD, background: 'rgba(16,185,129,0.08)' }
            }
          >
            {cur.label} → {copy.tierName[cur.tier]}
          </span>
          {cur.tier === 'laptopQuant' ? (
            <span className="rounded border border-border bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
              {copy.quantBridge}
            </span>
          ) : null}
        </div>
        <p className="min-h-[3rem] text-[12px] text-foreground/90">{copy.tierNote[cur.tier]}</p>

        {/* locked you-are-here chip */}
        <div className="flex items-center gap-2 text-[11px]">
          <span className="inline-flex items-center gap-1.5 rounded border border-foreground/30 bg-foreground/5 px-2 py-0.5 font-medium text-foreground/80">
            🔒 {copy.youAreHere}: {copy.youAreHereDetail}
          </span>
        </div>
      </div>

      {/* ── Capacity-vs-speed hardware table (book p.90). HTML for safety. ── */}
      <div className="rounded-md border border-border/70 bg-card/40 p-3">
        <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          {copy.hwHeading}
        </div>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[11px]">
            <thead>
              <tr className="text-left text-muted-foreground">
                <th className="py-1 pr-3 font-medium">{copy.hwCols.device}</th>
                <th className="py-1 pr-3 font-medium">{copy.hwCols.mem}</th>
                <th className="py-1 pr-3 font-medium">{copy.hwCols.bw}</th>
                <th className="py-1 pr-3 font-medium">{copy.hwCols.price}</th>
                <th className="py-1 font-medium" />
              </tr>
            </thead>
            <tbody className="font-mono">
              {copy.hwRows.map((r) => (
                <tr key={r.device} className="border-t border-border/50">
                  <td className="py-1 pr-3 text-foreground/90">{r.device}</td>
                  <td className="py-1 pr-3 text-foreground/90">{r.mem}</td>
                  <td className="py-1 pr-3 text-foreground/90">{r.bw}</td>
                  <td className="py-1 pr-3 text-foreground/90">{r.price}</td>
                  <td className="py-1 font-sans text-[10px] text-muted-foreground">{r.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-1.5 text-[11px] text-muted-foreground">{copy.hwCaption}</p>
      </div>

      {/* ── Synthesis: it's both. ── */}
      <div className="rounded-md border border-border/70 bg-card/40 p-3">
        <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          {copy.synthHeading}
        </div>
        <p className="text-[13px] text-foreground/90">{copy.synthesis}</p>
      </div>

      {/* ── Honesty: you are here. ── */}
      <div className="rounded-md border border-emerald-500/40 bg-emerald-500/5 p-3">
        <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-emerald-700 dark:text-emerald-300">
          {copy.honestyHeading}
        </div>
        <p className="text-[13px] text-foreground/90">{copy.honesty}</p>
      </div>

      {/* hidden full-diagram description for the whole widget */}
      <span className="sr-only">{copy.ariaLabel}</span>
    </div>
  );
}
