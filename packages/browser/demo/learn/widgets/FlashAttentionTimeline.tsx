import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * FlashAttentionTimeline — chapter 4 deep-dive "Flash Attention", the
 * "A short history" section.
 *
 * Self-contained: NO worker, NO model, NO WASM — pure React + Tailwind/SVG, so
 * it server-renders for crawlers (createRoot replaces it on boot) and is robust
 * while the model downloads.
 *
 * It walks the lineage that produced Flash Attention: the two precursor ideas
 * (online softmax; memory-efficient attention) and the four numbered releases
 * (FlashAttention v1/v2/v3/v4). Click a node — or step / auto-advance — to read
 * that milestone's date, authors, paper, the one key idea, and its headline number.
 *
 * Every fact below is grounded in the primary arXiv source for that milestone
 * (verified, not paraphrased from memory):
 *   - Online softmax ............ arXiv 1805.02867 (Milakov & Gimelshein, 2018)
 *   - Memory-efficient attention  arXiv 2112.05682 (Rabe & Staats, 2021)
 *   - FlashAttention (v1) ....... arXiv 2205.14135 (Dao et al., 2022, NeurIPS)
 *   - FlashAttention-2 .......... arXiv 2307.08691 (Tri Dao, 2023)
 *   - FlashAttention-3 .......... arXiv 2407.08608 (Shah, Dao et al., 2024)
 *   - FlashAttention-4 .......... arXiv 2603.05451 (Zadouri, Dao et al., 2026)
 *
 * Honors prefers-reduced-motion: no auto-advance, fully static, the namesake
 * "FlashAttention (v1)" node selected.
 */

type Kind = 'precursor' | 'release';

type Milestone = {
  short: string; // node label on the rail (paper-name shorthand — stays English)
  name: string; // detail-panel title (paper name — stays English)
  year: string; // node year label
  tag: string; // 'precursor' | 'v1' | 'v2' | 'v3' | 'v4'
  kind: Kind;
  authors: string;
  paper: string;
  arxiv: string;
};

// Locale-independent facts: names, years, authors, paper titles, arXiv ids.
// The per-locale prose (date, key idea, impact) lives in COPY, keyed by arXiv id.
const MILESTONES: Milestone[] = [
  {
    short: 'Online softmax',
    name: 'Online softmax',
    year: '2018',
    tag: 'precursor',
    kind: 'precursor',
    authors: 'Milakov & Gimelshein · NVIDIA',
    paper: 'Online normalizer calculation for softmax',
    arxiv: '1805.02867',
  },
  {
    short: 'Memory-efficient attention',
    name: 'Memory-efficient attention',
    year: '2021',
    tag: 'precursor',
    kind: 'precursor',
    authors: 'Rabe & Staats · Google',
    paper: 'Self-attention Does Not Need O(n²) Memory',
    arxiv: '2112.05682',
  },
  {
    short: 'FlashAttention',
    name: 'FlashAttention',
    year: '2022',
    tag: 'v1',
    kind: 'release',
    authors: 'Dao, Fu, Ermon, Rudra, Ré',
    paper: 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness',
    arxiv: '2205.14135',
  },
  {
    short: 'FlashAttention-2',
    name: 'FlashAttention-2',
    year: '2023',
    tag: 'v2',
    kind: 'release',
    authors: 'Tri Dao',
    paper: 'FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning',
    arxiv: '2307.08691',
  },
  {
    short: 'FlashAttention-3',
    name: 'FlashAttention-3',
    year: '2024',
    tag: 'v3',
    kind: 'release',
    authors: 'Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao',
    paper: 'FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision',
    arxiv: '2407.08608',
  },
  {
    short: 'FlashAttention-4',
    name: 'FlashAttention-4',
    year: '2026',
    tag: 'v4',
    kind: 'release',
    authors: 'Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao',
    paper: 'FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling',
    arxiv: '2603.05451',
  },
];

type MilestoneText = {
  date: string; // detail-panel date
  idea: string; // the one key idea
  impact: string; // headline number / significance
};

// Per-locale copy. English strings are moved verbatim from the original data/JSX.
const COPY = {
  en: {
    heading: 'The Flash Attention family tree',
    pause: '❚❚ Pause',
    play: '▶ Play',
    prevAria: 'Previous milestone',
    nextAria: 'Next milestone',
    railAria: 'Flash Attention version timeline',
    legend: 'dashed = precursor ideas · solid = the numbered FlashAttention releases',
    precursorBadge: 'precursor',
    impactLabel: 'impact',
    milestones: {
      '1805.02867': {
        date: 'May 2018',
        idea: 'Track a running max and a running sum of exponentials in one pass, rescaling the sum whenever a larger value appears — fusing what used to be two separate passes (find the max, then sum it) into one.',
        impact: 'Up to 1.3× faster softmax — but the real legacy is that running-rescale trick, reused by everything below.',
      },
      '2112.05682': {
        date: 'Dec 2021',
        idea: 'Apply that online softmax over chunks of keys and values: attention needs only O(1) memory in the sequence length — O(log n) for full self-attention, and O(√n) in their practical accelerator implementation. Compute stays O(n²).',
        impact: '59× less memory at 16K tokens (inference) — proved you never have to store the full N×N matrix.',
      },
      '2205.14135': {
        date: 'May 2022 · NeurIPS 2022',
        idea: 'Make it IO-aware: tile Q/K/V into on-chip SRAM, fuse the whole operation into one kernel, and recompute in the backward pass — so the N×N matrix is never written to slow HBM. Exact, not an approximation (plus a block-sparse variant).',
        impact: '15% faster BERT-large (vs the MLPerf 1.1 record), 3× GPT-2; first exact attention to reach 16K-token context (Path-X).',
      },
      '2307.08691': {
        date: 'Jul 2023',
        idea: 'Same algorithm, better GPU mechanics: cut the non-matmul FLOPs, parallelize across the sequence dimension (not just batch·heads), and partition work between warps to avoid shared-memory traffic.',
        impact: '~2× faster than FlashAttention; up to 225 TFLOP/s on an A100 — 50–73% of its FP16 peak.',
      },
      '2407.08608': {
        date: 'Jul 2024',
        idea: "Exploit NVIDIA Hopper's asynchrony — warp-specialization over the Tensor Cores and TMA — to overlap the matmuls with softmax (pingpong scheduling), plus FP8 via block quantization and incoherent processing.",
        impact: 'FP16 1.5–2× over FA2 → 740 TFLOP/s (75% of an H100); FP8 reaches ≈ 1.2 PFLOP/s.',
      },
      '2603.05451': {
        date: 'Mar 2026',
        idea: "Co-design the algorithm with the kernel pipeline for Blackwell's asymmetric scaling — Tensor-Core throughput roughly doubled while the other units didn't — so the schedule hides everything that isn't matmul.",
        impact: 'Up to 1,613 TFLOP/s bf16 on a B200 (71% utilization); up to 1.3× over cuDNN 9.13 and 2.7× over Triton.',
      },
    } as Record<string, MilestoneText>,
  },
  zh: {
    heading: 'Flash Attention 家族谱系',
    pause: '❚❚ 暂停',
    play: '▶ 播放',
    prevAria: '上一个里程碑',
    nextAria: '下一个里程碑',
    railAria: 'Flash Attention 版本时间线',
    legend: '虚线 = 先驱想法 · 实线 = 带编号的 FlashAttention 正式发布',
    precursorBadge: '先驱',
    impactLabel: '影响',
    milestones: {
      '1805.02867': {
        date: '2018 年 5 月',
        idea: '在一遍扫描中同时维护滚动最大值和指数的滚动和，每当出现更大的值就重新缩放这个和——把过去的两遍（先找最大值，再求和）融合成一遍。',
        impact: 'softmax 最高快 1.3×——但真正的遗产是那个滚动重缩放技巧，被下面的每一项工作复用。',
      },
      '2112.05682': {
        date: '2021 年 12 月',
        idea: '把这个在线 softmax 应用到 key 和 value 的分块上：注意力对序列长度只需要 O(1) 的内存——完整自注意力为 O(log n)，他们在加速器上的实际实现为 O(√n)。计算量仍是 O(n²)。',
        impact: '16K token 时内存少 59×（推理）——证明了你永远不必存下完整的 N×N 矩阵。',
      },
      '2205.14135': {
        date: '2022 年 5 月 · NeurIPS 2022',
        idea: '让它感知 IO：把 Q/K/V 切块装进片上 SRAM，把整个运算融合成一个内核，并在反向传播时重算——N×N 矩阵从不写入慢速 HBM。精确，而非近似（另有块稀疏变体）。',
        impact: 'BERT-large 快 15%（对比 MLPerf 1.1 纪录），GPT-2 快 3×；第一个达到 16K token 上下文（Path-X）的精确注意力。',
      },
      '2307.08691': {
        date: '2023 年 7 月',
        idea: '同样的算法，更好的 GPU 机制：削减非矩阵乘法的 FLOPs，沿序列维度并行（而不只是 batch·heads），并在 warp 之间重新划分工作以避免共享内存流量。',
        impact: '比 FlashAttention 快约 2×；在 A100 上最高 225 TFLOP/s——达到其 FP16 峰值的 50–73%。',
      },
      '2407.08608': {
        date: '2024 年 7 月',
        idea: '利用 NVIDIA Hopper 的异步能力——对 Tensor Core 与 TMA 做 warp 分工——让矩阵乘法与 softmax 重叠（乒乓调度），再加上经由块量化与不相干处理实现的 FP8。',
        impact: 'FP16 比 FA2 快 1.5–2× → 740 TFLOP/s（H100 的 75%）；FP8 达到 ≈ 1.2 PFLOP/s。',
      },
      '2603.05451': {
        date: '2026 年 3 月',
        idea: '为 Blackwell 的不对称扩展把算法与内核流水线协同设计——Tensor Core 吞吐大约翻倍而其他单元没有——让调度把一切非矩阵乘法的工作都藏起来。',
        impact: '在 B200 上 bf16 最高 1,613 TFLOP/s（利用率 71%）；最高比 cuDNN 9.13 快 1.3×、比 Triton 快 2.7×。',
      },
    } as Record<string, MilestoneText>,
  },
} as const;
const N = MILESTONES.length;
const DEFAULT_SEL = 2; // the namesake "FlashAttention (v1)" node
const FRAME_MS = 2800;

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

export function FlashAttentionTimeline() {
  const reducedMotion = usePrefersReducedMotion();
  const copy = COPY[useLocale()];
  const [sel, setSel] = React.useState(DEFAULT_SEL);
  const [playing, setPlaying] = React.useState(!reducedMotion);

  React.useEffect(() => {
    if (!playing || reducedMotion) return;
    const t = window.setInterval(() => setSel((s) => (s + 1) % N), FRAME_MS);
    return () => window.clearInterval(t);
  }, [playing, reducedMotion]);

  // Selecting a node by hand stops the auto-advance so reading isn't interrupted.
  const select = (i: number) => {
    setPlaying(false);
    setSel(i);
  };
  const step = (dir: number) => {
    setPlaying(false);
    setSel((s) => Math.min(N - 1, Math.max(0, s + dir)));
  };

  const m = MILESTONES[sel];
  const mt = copy.milestones[m.arxiv];
  // The progress fill spans from the first dot centre (10%) to the selected dot.
  const fillPct = (sel / (N - 1)) * 80;

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + controls */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>
        <div className="flex items-center gap-1">
          {!reducedMotion ? (
            <button
              type="button"
              onClick={() => setPlaying((p) => !p)}
              aria-pressed={playing}
              className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
            >
              {playing ? copy.pause : copy.play}
            </button>
          ) : null}
          <button
            type="button"
            onClick={() => step(-1)}
            disabled={sel === 0}
            aria-label={copy.prevAria}
            className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground disabled:opacity-30"
          >
            ‹
          </button>
          <button
            type="button"
            onClick={() => step(1)}
            disabled={sel === N - 1}
            aria-label={copy.nextAria}
            className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground disabled:opacity-30"
          >
            ›
          </button>
        </div>
      </div>

      {/* The lineage rail */}
      <div className="relative px-1 pt-2" role="group" aria-label={copy.railAria}>
        {/* track + progress fill, centred on the dot row */}
        <div className="absolute top-[14px] h-0.5 rounded" style={{ left: '10%', right: '10%', background: 'var(--border)' }} />
        <div
          className="absolute top-[14px] h-0.5 rounded transition-[width] duration-500"
          style={{ left: '10%', width: `${fillPct}%`, background: 'var(--primary)' }}
        />
        <div className="relative grid grid-cols-6">
          {MILESTONES.map((node, i) => {
            const active = i === sel;
            const passed = i < sel;
            const isPre = node.kind === 'precursor';
            return (
              <button
                key={node.short}
                type="button"
                onClick={() => select(i)}
                aria-pressed={active}
                aria-label={`${node.name}, ${copy.milestones[node.arxiv].date}`}
                className="flex flex-col items-center gap-1.5 rounded px-1 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              >
                <span
                  className="h-3 w-3 rounded-full transition-all duration-300"
                  style={{
                    background: active ? 'var(--primary)' : passed ? 'var(--primary)' : 'var(--card)',
                    opacity: !active && passed ? 0.55 : 1,
                    border: `1.5px ${isPre ? 'dashed' : 'solid'} ${active || passed ? 'var(--primary)' : 'var(--border)'}`,
                    transform: active ? 'scale(1.35)' : 'scale(1)',
                  }}
                />
                <span className="font-mono text-[10px] text-muted-foreground">{node.year}</span>
                <span
                  className="max-w-[88px] text-center text-[10px] leading-tight"
                  style={{ color: active ? 'var(--foreground)' : 'var(--muted-foreground)' }}
                >
                  {node.short}
                </span>
              </button>
            );
          })}
        </div>
        <div className="mt-1 text-center text-[10px] text-muted-foreground/70">
          {copy.legend}
        </div>
      </div>

      {/* Detail panel for the selected milestone */}
      <div className="space-y-2 rounded-md border border-border bg-card/40 p-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div className="text-[15px] font-semibold text-foreground">
            {m.name}
            {m.kind === 'release' ? (
              <span className="ml-2 rounded bg-primary/15 px-1.5 py-0.5 align-middle font-mono text-[10px] text-primary">
                {m.tag}
              </span>
            ) : (
              <span className="ml-2 rounded bg-muted px-1.5 py-0.5 align-middle text-[10px] text-muted-foreground">
                {copy.precursorBadge}
              </span>
            )}
          </div>
          <div className="font-mono text-[11px] text-muted-foreground">{mt.date}</div>
        </div>
        <div className="text-[11px] text-muted-foreground">{m.authors}</div>
        <p className="text-[13px] text-foreground/90">{mt.idea}</p>
        <div className="flex items-baseline gap-2 rounded bg-primary/10 px-2 py-1.5">
          <span className="text-[9px] font-semibold uppercase tracking-wide text-primary">{copy.impactLabel}</span>
          <span className="text-[12px] text-foreground/90">{mt.impact}</span>
        </div>
        <a
          href={`https://arxiv.org/abs/${m.arxiv}`}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-block text-[11px] text-muted-foreground transition-colors hover:text-foreground"
        >
          {m.paper} · arXiv {m.arxiv} ↗
        </a>
      </div>
    </div>
  );
}
