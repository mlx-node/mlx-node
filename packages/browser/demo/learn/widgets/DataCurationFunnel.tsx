import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import { SegmentedToggle } from '../scaffolding/SegmentedToggle';

/**
 * DataCurationFunnel — sub-chapter 13.1 (Training), "Pretraining data curation".
 *
 * STATIC, SSR-safe: plain div bars (same idiom as DataScaleBar / ScaleLadder),
 * no SVG needed, no worker/model/WASM dependency, no randomness. The only
 * state is the Pipeline/Mixture view toggle.
 *
 * Two views, each grounded in a distinct primary source:
 *
 *   PIPELINE — the FineWeb curation pipeline (Penedo et al. 2024,
 *   arXiv:2406.17557): raw Common Crawl (96 snapshots, the real entry point)
 *   through URL filtering -> extraction/language-ID -> quality heuristics ->
 *   MinHash dedup -> the final 15T-token dataset, with the FineWeb-Edu
 *   classifier-filtered branch (1.3T tokens). ONLY the two endpoint counts
 *   (96 snapshots in, 15T / 1.3T tokens out) are real cited numbers; the
 *   intermediate bar widths are an illustrative visual taper (FineWeb does not
 *   publish an exact token count after every individual stage) and are
 *   labeled as such. Below the funnel, the near-duplicate rates from Lee et
 *   al. 2021 (arXiv:2107.06499) are real MEASURED numbers (C4 3.04%, RealNews
 *   13.63%, LM1B 4.86%, Wiki-40B 0.39%) — shown separately from the
 *   illustrative funnel so the two kinds of number are never confused.
 *
 *   MIXTURE — LLaMA's published pretraining data mixture (Touvron et al.
 *   2023, arXiv:2302.13971): CommonCrawl 67.0% / C4 15.0% / GitHub 4.5% /
 *   Wikipedia 4.5% / Books 4.5% / ArXiv 2.5% / StackExchange 2.0%, with each
 *   source's epoch count. One concrete, real, cited mixture — explicitly NOT
 *   universal and NOT Qwen's own mixture. The callout underneath states
 *   exactly what the Qwen3 Technical Report (arXiv:2505.09388) discloses
 *   (token counts per of 3 stages, category labels) and what it does not
 *   (no dedup method, no filtering method, no source percentages) plus the
 *   Stanford CRFM FMTI's independent 0-disclosure score for Alibaba/Qwen on
 *   data-acquisition methods and composition.
 */

type View = 'pipeline' | 'mixture';

type FunnelStage = {
  label: string;
  note: string;
  widthPct: number;
  tag?: string;
  highlight?: boolean;
};

type DedupRow = {
  name: string;
  pct: number;
};

type MixtureRow = {
  name: string;
  pct: number;
  epochs: string;
};

const DEDUP_SCALE_MAX = 14; // Wiki-40B..RealNews all fit under 14%

const COPY = {
  en: {
    heading: 'From raw web crawl to a curated pretraining set',
    viewPipeline: 'Pipeline',
    viewMixture: 'Mixture',
    pipelineIntro:
      'A real, documented pipeline (FineWeb) that turns raw Common Crawl into a curated text dataset. Only the two endpoints — 96 snapshots in, the final token counts out — are measured numbers; the bar width at every stage in between is an illustrative taper, not a published per-stage yield.',
    stages: [
      {
        label: 'Raw Common Crawl',
        note: '96 monthly web-crawl snapshots — the base layer under almost every open pretraining corpus',
        widthPct: 96,
        tag: '96 snapshots',
      },
      {
        label: 'URL filtering',
        note: 'Drop domains on an adult-content blocklist, before any text is even pulled out',
        widthPct: 84,
      },
      {
        label: 'Extraction + language ID',
        note: 'trafilatura extracts text from raw HTML; a fastText classifier keeps only English pages scoring ≥ 0.65',
        widthPct: 70,
      },
      {
        label: 'Quality heuristics',
        note: 'Gopher/C4-style + custom filters — drop boilerplate, cookie-notice lines, pages that are mostly unpunctuated or duplicated lines',
        widthPct: 55,
      },
      {
        label: 'MinHash dedup',
        note: '5-grams → 112 hashes split into 14 buckets of 8 → drop documents that are ≥ 75% similar to one already kept',
        widthPct: 40,
      },
      {
        label: 'FineWeb (final)',
        note: 'The curated, deduplicated English web-text dataset',
        widthPct: 26,
        tag: '15T tokens',
        highlight: true,
      },
    ] as FunnelStage[],
    branchLabel: '+ classifier filter → FineWeb-Edu',
    branchNote: 'A Llama-3-70B-distilled educational-value scorer keeps only the highest-scoring subset',
    branchTag: '1.3T tokens',
    viewToggleAria: 'Pipeline or mixture view',
    dedupBarAria: (name: string, pct: number) => `${name}: ${pct}% of documents flagged as near-duplicate`,
    mixtureBarAria: (name: string, pct: number, epochs: string, epochsLabel: string) =>
      `${name}: ${pct}% of the mixture, ${epochs} (${epochsLabel})`,
    realTagLabel: 'measured / cited',
    illustrativeLabel: 'illustrative taper, not a measured yield',
    dedupHeading: 'Why dedup matters — measured, not illustrative',
    dedupBody:
      'Lee et al. 2021 measured how much of a raw web corpus is near-duplicate text before any dedup runs. It is a lot:',
    dedupRows: [
      { name: 'RealNews', pct: 13.63 },
      { name: 'LM1B', pct: 4.86 },
      { name: 'C4', pct: 3.04 },
      { name: 'Wiki-40B', pct: 0.39 },
    ] as DedupRow[],
    dedupFootnote:
      'Training on the undeduplicated version made models emit memorized training text roughly 10× more often — one 61-word C4 sentence was found duplicated over 60,000 times.',
    mixtureIntro:
      "One concrete, published example of a deliberate mixture — not raw crawl alone. LLaMA (Touvron et al. 2023) blended web text with books, code, and reference text in fixed, non-uniform ratios, up-sampling scarce high-quality sources across multiple epochs. This is LLaMA's recipe, not a universal law, and not Qwen's.",
    mixtureRows: [
      { name: 'Common Crawl', pct: 67.0, epochs: '1.10 epochs' },
      { name: 'C4', pct: 15.0, epochs: '1.06 epochs' },
      { name: 'GitHub', pct: 4.5, epochs: '0.64 epochs' },
      { name: 'Wikipedia', pct: 4.5, epochs: '2.45 epochs' },
      { name: 'Books', pct: 4.5, epochs: '2.23 epochs' },
      { name: 'ArXiv', pct: 2.5, epochs: '1.06 epochs' },
      { name: 'StackExchange', pct: 2.0, epochs: '1.03 epochs' },
    ] as MixtureRow[],
    epochsLabel: 'epochs over the run',
    qwenHonestyHeading: "What Qwen discloses about its own mixture",
    qwenHonesty: (
      <>
        The Qwen3 Technical Report discloses <strong>≈36T total tokens</strong> across three stages — Stage 1{' '}
        <span className="font-mono">&gt;30T</span>, Stage 2 <span className="font-mono">+≈5T</span>, Stage 3{' '}
        <span className="font-mono">hundreds of billions</span> — and broad category labels (STEM, code, books,
        multilingual, synthetic data from Qwen2.5-VL/Math/Coder). It names <strong>no</strong> deduplication method, no
        quality-filtering method, and <strong>no source-percentage breakdown</strong> — there is no CommonCrawl-%/Books-%
        table to show here the way there is for LLaMA above. Stanford CRFM&apos;s Foundation Model Transparency Index
        (Dec 2025) independently scores Alibaba/Qwen at <strong>0 — not disclosed</strong> — on data-acquisition
        methods, dataset sources, crawler/opt-out handling, and composition percentages.
      </>
    ),
  },
  zh: {
    heading: '从原始网页抓取到一份精心整理的预训练数据集',
    viewPipeline: 'Pipeline',
    viewMixture: 'Mixture',
    pipelineIntro:
      '一条真实、有文档记录的 pipeline（FineWeb），把原始 Common Crawl 变成一份精心整理的文本数据集。只有两端——进去的 96 份快照，出来的最终 token 数——是实测数字；中间每一级的条形宽度只是一种示意性的收窄，不是某个已公布的逐级产出比例。',
    stages: [
      {
        label: 'Raw Common Crawl',
        note: '96 份逐月网页抓取快照——几乎每一份开放预训练语料的底层来源',
        widthPct: 96,
        tag: '96 份快照',
      },
      {
        label: 'URL filtering',
        note: '在提取任何文本之前，先按成人内容黑名单剔除域名',
        widthPct: 84,
      },
      {
        label: 'Extraction + language ID',
        note: 'trafilatura 从原始 HTML 中抽取文本；fastText 分类器只保留英文得分 ≥ 0.65 的页面',
        widthPct: 70,
      },
      {
        label: 'Quality heuristics',
        note: 'Gopher/C4 风格 + 自定义过滤器——剔除样板文字、cookie 提示行，以及标点稀少或重复行占比过高的页面',
        widthPct: 55,
      },
      {
        label: 'MinHash dedup',
        note: '5-gram → 112 个哈希，分成 14 组每组 8 个 → 与已保留文档相似度 ≥ 75% 的文档被丢弃',
        widthPct: 40,
      },
      {
        label: 'FineWeb（最终）',
        note: '经过整理、去重的英文网页文本数据集',
        widthPct: 26,
        tag: '15T tokens',
        highlight: true,
      },
    ] as FunnelStage[],
    branchLabel: '+ classifier filter → FineWeb-Edu',
    branchNote: '一个由 Llama-3-70B 蒸馏出的教育价值评分器，只保留得分最高的子集',
    branchTag: '1.3T tokens',
    viewToggleAria: 'Pipeline 或 Mixture 视图切换',
    dedupBarAria: (name: string, pct: number) => `${name}：${pct}% 的文档被标记为近重复`,
    mixtureBarAria: (name: string, pct: number, epochs: string, epochsLabel: string) =>
      `${name}：占混合数据的 ${pct}%，${epochs}（${epochsLabel}）`,
    realTagLabel: '实测 / 有引用',
    illustrativeLabel: '示意性收窄，非实测产出',
    dedupHeading: '为什么去重重要——这是实测数字，不是示意',
    dedupBody: 'Lee 等人 2021 年实测了一份原始网页语料在做任何去重之前，有多少内容是近重复文本。答案是相当多：',
    dedupRows: [
      { name: 'RealNews', pct: 13.63 },
      { name: 'LM1B', pct: 4.86 },
      { name: 'C4', pct: 3.04 },
      { name: 'Wiki-40B', pct: 0.39 },
    ] as DedupRow[],
    dedupFootnote:
      '在未去重版本上训练，会让模型逐字吐出训练文本的频率高出约 10 倍——研究中发现 C4 里一句 61 个词的句子被逐字重复了超过 6 万次。',
    mixtureIntro:
      '一个具体的、已发表的「刻意配比」范例——而不是只用原始爬取数据。LLaMA（Touvron 等人，2023）把网页文本与书籍、代码、参考文献类文本按固定的非均匀比例混合，并对稀缺的高质量来源做多轮上采样。这是 LLaMA 自己的配方，不是普适规律，也不是 Qwen 的配方。',
    mixtureRows: [
      { name: 'Common Crawl', pct: 67.0, epochs: '1.10 epochs' },
      { name: 'C4', pct: 15.0, epochs: '1.06 epochs' },
      { name: 'GitHub', pct: 4.5, epochs: '0.64 epochs' },
      { name: 'Wikipedia', pct: 4.5, epochs: '2.45 epochs' },
      { name: 'Books', pct: 4.5, epochs: '2.23 epochs' },
      { name: 'ArXiv', pct: 2.5, epochs: '1.06 epochs' },
      { name: 'StackExchange', pct: 2.0, epochs: '1.03 epochs' },
    ] as MixtureRow[],
    epochsLabel: '本轮训练中的 epoch 数',
    qwenHonestyHeading: 'Qwen 对自己数据配比的披露',
    qwenHonesty: (
      <>
        Qwen3 Technical Report 披露了<strong>约 36T 总 token 数</strong>，分三个阶段——Stage 1{' '}
        <span className="font-mono">&gt;30T</span>、Stage 2 <span className="font-mono">+≈5T</span>、Stage 3{' '}
        <span className="font-mono">数千亿</span>——以及粗粒度的类别标签（STEM、代码、书籍、多语言、来自
        Qwen2.5-VL/Math/Coder 的合成数据）。但它<strong>没有</strong>点名任何去重方法、任何质量过滤方法，也
        <strong>没有来源百分比拆分</strong>——这里没法像上面 LLaMA 那样给出一张 CommonCrawl-%/Books-%
        的表。斯坦福 CRFM 的 Foundation Model Transparency Index（2025 年 12 月）独立评分中，Alibaba/Qwen 在数据获取方法、数据集来源、爬虫/opt-out
        处理、以及配比百分比这几项上都是<strong>0 分——未披露</strong>。
      </>
    ),
  },
} as const;

function Connector() {
  return (
    <div aria-hidden className="flex justify-center text-muted-foreground/60">
      <span className="text-[13px] leading-none">▾</span>
    </div>
  );
}

function FunnelStageBox({ stage, realTagLabel }: { stage: FunnelStage; realTagLabel: string }) {
  return (
    <div className="mx-auto" style={{ width: `${stage.widthPct}%` }}>
      <div
        className={
          'rounded-md border px-3 py-2 text-center ' +
          (stage.highlight ? 'border-primary bg-primary/10' : 'border-border bg-card')
        }
      >
        <div className={'text-[12.5px] font-semibold ' + (stage.highlight ? 'text-primary' : 'text-foreground')}>
          {stage.label}
        </div>
        <div className="mt-0.5 text-[11px] leading-snug text-muted-foreground">{stage.note}</div>
        {stage.tag ? (
          <div className="mt-1 inline-flex items-center gap-1 rounded-full border border-primary/40 bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
            {stage.tag} <span className="text-primary/70">· {realTagLabel}</span>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function DedupBar({ row, aria }: { row: DedupRow; aria: string }) {
  const widthPct = (row.pct / DEDUP_SCALE_MAX) * 100;
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between text-[11px]">
        <span className="text-foreground/85">{row.name}</span>
        <span className="font-mono text-muted-foreground">{row.pct}%</span>
      </div>
      <div
        className="h-3 w-full overflow-hidden rounded-sm border border-border bg-muted/40"
        role="img"
        aria-label={aria}
      >
        <div className="h-full rounded-[2px] bg-foreground/50" style={{ width: `${widthPct}%` }} />
      </div>
    </div>
  );
}

function MixtureBar({ row, aria }: { row: MixtureRow; aria: string }) {
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap items-baseline justify-between gap-x-2 text-[11px]">
        <span className="text-foreground/85">{row.name}</span>
        <span className="font-mono text-muted-foreground">
          {row.pct}% <span className="text-muted-foreground/70">· {row.epochs}</span>
        </span>
      </div>
      <div
        className="flex h-4 w-full overflow-hidden rounded-sm border border-border bg-muted/40"
        role="img"
        aria-label={aria}
      >
        <div className="h-full rounded-[2px] bg-primary/60" style={{ width: `${row.pct}%` }} />
      </div>
    </div>
  );
}

export function DataCurationFunnel() {
  const copy = COPY[useLocale()];
  const [view, setView] = React.useState<View>('pipeline');

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>
        <SegmentedToggle
          value={view}
          onChange={setView}
          options={[
            { value: 'pipeline', label: copy.viewPipeline },
            { value: 'mixture', label: copy.viewMixture },
          ]}
          ariaLabel={copy.viewToggleAria}
        />
      </div>

      {view === 'pipeline' ? (
        <>
          <p className="text-[12px] text-foreground/85">{copy.pipelineIntro}</p>

          <div className="space-y-1.5">
            {copy.stages.map((stage, i) => (
              <React.Fragment key={stage.label}>
                {i > 0 ? <Connector /> : null}
                <FunnelStageBox stage={stage} realTagLabel={copy.realTagLabel} />
              </React.Fragment>
            ))}
          </div>

          <div className="mx-auto flex items-start gap-2" style={{ width: '80%' }}>
            <span aria-hidden className="mt-2 text-muted-foreground/60">
              ↳
            </span>
            <div className="flex-1 rounded-md border border-dashed border-border bg-muted/20 px-3 py-2">
              <div className="text-[12px] font-semibold text-foreground">{copy.branchLabel}</div>
              <div className="mt-0.5 text-[11px] leading-snug text-muted-foreground">{copy.branchNote}</div>
              <div className="mt-1 inline-flex items-center gap-1 rounded-full border border-primary/40 bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
                {copy.branchTag} <span className="text-primary/70">· {copy.realTagLabel}</span>
              </div>
            </div>
          </div>

          <p className="text-[10px] text-muted-foreground">{copy.illustrativeLabel}</p>

          <div className="space-y-2 border-t border-border pt-3">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              {copy.dedupHeading}
            </div>
            <p className="text-[12px] text-foreground/85">{copy.dedupBody}</p>
            <div className="space-y-1.5">
              {copy.dedupRows.map((row) => (
                <DedupBar key={row.name} row={row} aria={copy.dedupBarAria(row.name, row.pct)} />
              ))}
            </div>
            <p className="text-[11px] text-muted-foreground">{copy.dedupFootnote}</p>
          </div>
        </>
      ) : (
        <>
          <p className="text-[12px] text-foreground/85">{copy.mixtureIntro}</p>

          <div className="space-y-2">
            {copy.mixtureRows.map((row) => (
              <MixtureBar
                key={row.name}
                row={row}
                aria={copy.mixtureBarAria(row.name, row.pct, row.epochs, copy.epochsLabel)}
              />
            ))}
          </div>

          <div className="rounded-md border border-border bg-muted/20 p-2 text-[12px] text-foreground/85">
            <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              {copy.qwenHonestyHeading}
            </div>
            {copy.qwenHonesty}
          </div>
        </>
      )}
    </div>
  );
}
