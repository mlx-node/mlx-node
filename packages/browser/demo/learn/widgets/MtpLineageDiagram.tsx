import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * MtpLineageDiagram — the Multi-Token-Prediction (MTP) lineage, three milestones
 * side by side, left→right chronological.
 *
 * Self-contained and STATIC: NO worker, NO model, NO WASM, NO animation — pure
 * React + Tailwind/SVG, so it server-renders for crawlers (createRoot replaces
 * it on boot) and is robust while the model downloads. The first render is
 * deterministic and identical server vs client; there is no time-dependent
 * state, so it also needs no prefers-reduced-motion handling.
 *
 * Each column is a small schematic of how that work attaches extra prediction
 * to a shared trunk, plus the headline facts. Every fact is grounded in the
 * primary source for that milestone (verified, not paraphrased from memory):
 *   - Meta 2024 ........ arXiv 2404.19737 (Gloeckle, Idrissi, Rozière, Lopez,
 *                        Hayou, Synnaeve) — n=4 parallel independent heads.
 *   - DeepSeek-V3 2024 . arXiv 2412.19437 (DeepSeek-AI) — one sequential MTP
 *                        module, depth D=1, keeps the causal chain.
 *   - Qwen3.5 / Qwen3-Next 2025 — same single sequential module (depth 1); the
 *                        model this course ships. Config keys are verbatim from
 *                        the checkpoint config.
 *
 * The arc of the story (also the caption): the field moved from Meta's parallel
 * independent heads to one sequential module that keeps the causal chain
 * (DeepSeek → Qwen). The Qwen column is highlighted as "the model in this
 * course".
 */

type ColId = 'meta' | 'deepseek' | 'qwen';

const COL_IDS: ColId[] = ['meta', 'deepseek', 'qwen'];

// ── SVG geometry ─────────────────────────────────────────────────────────────
// One viewBox per column schematic (each rendered in its own <svg>), so the
// three panels flow responsively (stacking on narrow screens) yet keep crisp
// internal proportions.
const VB_W = 240;
const VB_H = 150;

// Locale-independent facts: column titles, the org/citation line, arXiv ids.
// Per-locale prose (schematic labels, bullets, caption) lives in COPY.
type ColMeta = {
  title: string; // paper-name shorthand — stays English in both locales
  arxiv: string; // arXiv id — verbatim in both locales
  highlight: boolean; // the Qwen column, "the model in this course"
};

const COLS: Record<ColId, ColMeta> = {
  meta: { title: 'Meta 2024', arxiv: '2404.19737', highlight: false },
  deepseek: { title: 'DeepSeek-V3 2024', arxiv: '2412.19437', highlight: false },
  qwen: { title: 'Qwen3.5 / Qwen3-Next 2025', arxiv: '', highlight: true },
};

// Per-locale copy. zh keeps the English glossary terms (MTP, token, embedding,
// LM head / output head, HumanEval, MBPP, softmax, depth, Qwen3.5, Qwen3-Next,
// DeepSeek-V3) and the config keys / arXiv ids verbatim; only the surrounding
// prose translates, with full-width punctuation in zh.
const COPY = {
  en: {
    heading: 'The MTP lineage — three milestones',
    courseBadge: 'the model in this course',
    // Schematic labels (drawn inside the SVG).
    trunkLabel: 'shared trunk',
    hiddenLabel: 'final hidden',
    mainLabel: 'main model',
    moduleLabel: 'MTP module',
    causalLabel: 'keeps causal chain',
    depthLabel: 'depth 1',
    headLabels: ['t+1', 't+2', 't+3', 't+4'],
    sequentialTag: 'sequential',
    parallelTag: 'parallel',
    // Per-column bullet lists (order preserved).
    bullets: {
      meta: [
        'n = 4 parallel heads',
        'independent, no causal chain between them',
        'training: auxiliary task',
        'inference: self-speculative, up to 3× faster',
        '+12% HumanEval / +17% MBPP at 13B',
      ],
      deepseek: [
        'D = 1 (one extra token)',
        'sequential — keeps the causal chain',
        'shares embedding + output head',
        'train loss λ·(1/D)·ΣLᵏ, λ: 0.3→0.1',
        'inference: discard, or speculative-decode',
      ],
      qwen: [
        'one shared sequential module, depth 1',
        'mtp_num_hidden_layers: 1',
        'shares the tied embedding (mtp_use_dedicated_embeddings: false)',
        'boosts pretraining efficiency AND inference speed',
        "runs in vLLM / SGLang / this project's native engine",
      ],
    } as Record<ColId, string[]>,
    caption:
      'The field moved from parallel independent heads (Meta) to one sequential module that keeps the causal chain (DeepSeek → Qwen).',
  },
  zh: {
    heading: 'MTP 谱系——三个里程碑',
    courseBadge: '本课程使用的模型',
    trunkLabel: '共享主干',
    hiddenLabel: '最终 hidden',
    mainLabel: '主模型',
    moduleLabel: 'MTP module',
    causalLabel: '保持因果链',
    depthLabel: 'depth 1',
    headLabels: ['t+1', 't+2', 't+3', 't+4'],
    sequentialTag: '串联',
    parallelTag: '并行',
    bullets: {
      meta: [
        'n = 4 个并行 head',
        '彼此独立，相互之间没有因果链',
        '训练：作为辅助任务',
        '推理：自我推测式（self-speculative），最高快 3×',
        '在 13B 上 +12% HumanEval / +17% MBPP',
      ],
      deepseek: [
        'D = 1（一个额外 token）',
        '串联——保持因果链',
        '共享 embedding 与 output head',
        '训练损失 λ·(1/D)·ΣLᵏ，λ：0.3→0.1',
        '推理：丢弃，或用于 speculative decoding',
      ],
      qwen: [
        '一个共享的串联 module，depth 为 1',
        'mtp_num_hidden_layers: 1',
        '共享绑定的 embedding（mtp_use_dedicated_embeddings: false）',
        '同时提升预训练效率与推理速度',
        '可在 vLLM / SGLang / 本项目的原生引擎中运行',
      ],
    } as Record<ColId, string[]>,
    caption:
      '这个领域从并行的独立 head（Meta）转向一个保持因果链的串联 module（DeepSeek → Qwen）。',
  },
} as const;

/** Meta column — one shared trunk fanning out to 4 parallel independent heads. */
function MetaSchematic({ accent }: { accent: string }) {
  const copy = COPY[useLocale()];
  // Nudged right of the viewBox left edge so the wider "shared trunk" label,
  // centred on this x, does not clip (it would start at x≈-3 from x=30).
  const hiddenX = 42;
  const hiddenY = VB_H / 2;
  const headXs = [120, 160, 160, 200];
  const headYs = [26, 62, 98, 124];
  // Lay the four heads out in a fan so each branch reads as independent.
  const heads = [
    { x: 200, y: 26 },
    { x: 200, y: 62 },
    { x: 200, y: 98 },
    { x: 200, y: 124 },
  ];
  return (
    <g>
      {/* shared trunk → final hidden */}
      <rect x={hiddenX - 18} y={hiddenY - 18} width={36} height={36} rx={6} style={{ fill: 'var(--muted)', stroke: 'var(--border)' }} strokeWidth={1.5} />
      <text x={hiddenX} y={hiddenY - 24} textAnchor="middle" style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
        {copy.trunkLabel}
      </text>
      {/* four independent branches off the SAME final hidden */}
      {heads.map((h, i) => (
        <g key={i}>
          <path
            d={`M ${hiddenX + 18} ${hiddenY} C ${headXs[i]} ${hiddenY} ${headXs[i]} ${headYs[i]} ${h.x - 26} ${h.y}`}
            fill="none"
            style={{ stroke: accent, opacity: 0.65 }}
            strokeWidth={1.5}
          />
          <rect x={h.x - 24} y={h.y - 12} width={36} height={24} rx={5} style={{ fill: accent, fillOpacity: 0.14, stroke: accent }} strokeWidth={1.5} />
          <text x={h.x - 6} y={h.y + 4} textAnchor="middle" style={{ fill: 'var(--foreground)' }} className="font-mono text-[11px]">
            {copy.headLabels[i]}
          </text>
        </g>
      ))}
    </g>
  );
}

/** A single sequential MTP module chained after the main model (DeepSeek/Qwen). */
function SequentialSchematic({ accent }: { accent: string }) {
  const copy = COPY[useLocale()];
  const midY = VB_H / 2;
  return (
    <g>
      {/* main model */}
      <rect x={16} y={midY - 24} width={70} height={48} rx={7} style={{ fill: 'var(--muted)', stroke: 'var(--border)' }} strokeWidth={1.5} />
      <text x={51} y={midY + 4} textAnchor="middle" style={{ fill: 'var(--foreground)' }} className="text-[11px] font-medium">
        {copy.mainLabel}
      </text>
      {/* sequential arrow → keeps the causal chain */}
      <line x1={88} y1={midY} x2={142} y2={midY} style={{ stroke: accent }} strokeWidth={2} />
      <polygon points={`152,${midY} 141,${midY - 5} 141,${midY + 5}`} style={{ fill: accent }} />
      {/* Label sits ABOVE the two boxes (box top is at midY-24): the English
          "keeps causal chain" is wider than the inter-box gap, so keeping it on
          the arrow line would overlap the MTP-module box. */}
      <text x={120} y={midY - 35} textAnchor="middle" style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
        {copy.causalLabel}
      </text>
      {/* the one MTP module */}
      <rect x={154} y={midY - 24} width={74} height={48} rx={7} style={{ fill: accent, fillOpacity: 0.12, stroke: accent }} strokeWidth={1.5} />
      <text x={191} y={midY - 2} textAnchor="middle" style={{ fill: 'var(--foreground)' }} className="text-[11px] font-medium">
        {copy.moduleLabel}
      </text>
      <text x={191} y={midY + 14} textAnchor="middle" style={{ fill: accent }} className="font-mono text-[11px]">
        {copy.depthLabel}
      </text>
    </g>
  );
}

/** One column: title + tag, the schematic, then the bullet list. */
function Column({ id }: { id: ColId }) {
  const copy = COPY[useLocale()];
  const meta = COLS[id];
  const accent = meta.highlight ? 'var(--primary)' : 'var(--muted-foreground)';
  const isMeta = id === 'meta';
  return (
    <div
      className={[
        'flex flex-1 flex-col gap-2 rounded-md border p-3',
        meta.highlight ? 'border-primary/60 bg-primary/5' : 'border-border bg-card/40',
      ].join(' ')}
    >
      {/* column header */}
      <div className="flex flex-wrap items-baseline justify-between gap-1.5">
        <span className="text-[13px] font-semibold text-foreground">{meta.title}</span>
        <span
          className={[
            'rounded px-1.5 py-0.5 text-[11px] font-medium',
            id === 'meta'
              ? 'bg-muted text-muted-foreground'
              : meta.highlight
                ? 'bg-primary/15 text-primary'
                : 'bg-muted text-muted-foreground',
          ].join(' ')}
        >
          {id === 'meta' ? copy.parallelTag : copy.sequentialTag}
        </span>
      </div>
      {meta.highlight ? (
        <div className="text-[11px] font-medium text-primary">{copy.courseBadge}</div>
      ) : null}

      {/* The schematic is a redundant illustration of the bullets below, which
          carry every fact as text — so it is decorative for screen readers. */}
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" aria-hidden="true" focusable="false">
        {isMeta ? <MetaSchematic accent={accent} /> : <SequentialSchematic accent={accent} />}
      </svg>

      {/* bullets */}
      <ul className="space-y-1.5">
        {copy.bullets[id].map((b, i) => (
          <li key={i} className="flex gap-2 text-[11px] leading-snug text-foreground/90">
            <span aria-hidden className="mt-[2px] select-none" style={{ color: accent }}>
              ·
            </span>
            <span>{b}</span>
          </li>
        ))}
      </ul>

      {/* citation */}
      {meta.arxiv ? (
        <a
          href={`https://arxiv.org/abs/${meta.arxiv}`}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-auto inline-block font-mono text-[11px] text-muted-foreground transition-colors hover:text-foreground"
        >
          arXiv {meta.arxiv} ↗
        </a>
      ) : null}
    </div>
  );
}

export function MtpLineageDiagram() {
  const copy = COPY[useLocale()];
  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header */}
      <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>

      {/* Three columns, left→right chronological. Stack on narrow viewports. */}
      <div className="flex flex-col gap-3 md:flex-row md:items-stretch">
        {COL_IDS.map((id) => (
          <Column key={id} id={id} />
        ))}
      </div>

      {/* Caption under all three */}
      <p className="border-t border-border/60 pt-3 text-[13px] text-foreground/90">{copy.caption}</p>
    </div>
  );
}
