import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * SpeculativeDecodeDiagram — Multi-Token Prediction sub-chapter centerpiece: the
 * draft → verify → accept loop of self-speculative decoding with an MTP head.
 *
 * Self-contained: NO worker, NO model, NO WASM — pure SVG + React state, so it
 * server-renders for crawlers (createRoot replaces it on boot) and is robust
 * while the model downloads. The token strings are ILLUSTRATIVE — this widget
 * teaches the concept, it does not run MTP in the browser.
 *
 * ONE STEP of speculative decoding (verified against the mlx-node engine and
 * Leviathan–Chen-style speculative sampling):
 *   (a) DRAFT  — the cheap MTP head proposes D candidate tokens d₁…d_D, one
 *                after another (sequential, cheap).
 *   (b) VERIFY — the FULL model checks the window [last_committed, d₁, …, d_D]
 *                in ONE batched forward pass over D+1 positions (the expensive
 *                part, done exactly once).
 *   (c) ACCEPT — walk left→right; keep each draft dᵢ that matches the full
 *                model's own pick at position i; at the FIRST mismatch emit the
 *                full model's token (the "correction") and STOP. If ALL D drafts
 *                are accepted, also emit one BONUS token (the full model's pick
 *                at position D+1). Result: K accepted + 1 = K+1 real tokens from
 *                a single verify pass, K ∈ [0, D].
 *
 * The output is LOSSLESS — same distribution as plain one-token-at-a-time
 * decoding (identical text at temperature 0) — because a draft is kept only when
 * it matches the full model's own choice. Worst case (all rejected) = the one
 * full-model pass plain decode would run anyway, plus a small draft tax; best
 * case = D+1 tokens from that single verify pass.
 *
 * Phase state machine (within one step), 5 phases:
 *   0 commit  — show the committed prefix only.
 *   1 draft   — the MTP head's d₁…d_D appear, muted + dashed ("draft").
 *   2 verify  — the full model lights up all D+1 positions at once (one pass).
 *   3 accept  — left→right resolve: accepted drafts → green ✓; first mismatch →
 *               red ✗ replaced by the correction; or, on full accept, a bonus
 *               token appears.
 *   4 tally   — "tokens this pass: K+1" + running passes/tokens average.
 *
 * Two canned scenarios differ only in the per-position verify verdicts:
 *   A "partial accept": D=3, accept d₁,d₂, reject d₃ → correction → 3 tokens.
 *   B "full accept":    D=3, accept all → +1 bonus      → 4 tokens.
 *
 * Honors prefers-reduced-motion: no auto-play, renders the final (tally) frame
 * of Scenario A, and the Play button is hidden.
 */

type Verdict = 'accept' | 'reject';

type Scenario = {
  id: 'A' | 'B';
  /** committed prefix tokens (already emitted, locale-independent illustrative strings) */
  committed: string[];
  /** the MTP head's D draft tokens d₁…d_D */
  drafts: string[];
  /** verify verdict per draft position (length D) */
  verdicts: Verdict[];
  /** the full model's correction token at the first mismatch (only used if a reject exists) */
  correction: string;
  /** the bonus token emitted when ALL drafts are accepted (only used if no reject) */
  bonus: string;
};

// Two scenarios — illustrative tokens only. Strings stay English (they stand in
// for raw model tokens, not UI copy).
const SCENARIOS: Scenario[] = [
  {
    id: 'A',
    committed: ['The', 'capital', 'of', 'France', 'is'],
    drafts: ['Paris', ',', 'the'],
    verdicts: ['accept', 'accept', 'reject'],
    correction: 'a',
    bonus: '',
  },
  {
    id: 'B',
    committed: ['The', 'capital', 'of', 'France', 'is'],
    drafts: ['Paris', ',', 'a'],
    verdicts: ['accept', 'accept', 'accept'],
    correction: '',
    bonus: 'beautiful',
  },
];

const D = 3; // draft depth used by both scenarios
const PHASES = 5; // commit, draft, verify, accept, tally
const FRAME_MS = 2200;

// Tailwind palette hexes, used for SVG fills/strokes (SVG can't take TW classes).
const EMERALD = '#10b981'; // emerald-500 — accepted
const RED = '#ef4444'; // red-500 — rejected

/** Index of the first rejected draft, or -1 if all accepted (full accept). */
function firstReject(s: Scenario): number {
  return s.verdicts.findIndex((v) => v === 'reject');
}

/** Tokens this pass = (accepted count K) + 1 (correction or bonus). K∈[0,D]. */
function tokensThisPass(s: Scenario): number {
  const fr = firstReject(s);
  const k = fr === -1 ? D : fr; // accepted before the first mismatch
  return k + 1;
}

// ── Per-locale copy. COPY.en strings are the original English, verbatim. ─────
const COPY = {
  en: {
    title: 'Speculative decoding — draft, verify, accept',
    pause: '❚❚ Pause',
    play: '▶ Play',
    prevAria: 'Previous phase',
    nextAria: 'Next phase',
    scenarioLabel: 'scenario',
    scenarioA: 'partial accept',
    scenarioB: 'full accept',
    scenarioAAria: 'Scenario A — partial accept: two drafts kept, one corrected',
    scenarioBAria: 'Scenario B — full accept: all drafts kept plus a bonus token',
    committedLabel: 'committed so far',
    draftLabel: 'MTP draft',
    verifyLabel: 'full-model verify (one pass)',
    draftBadge: 'draft',
    correctionBadge: 'correction',
    bonusBadge: 'bonus',
    phases: ['commit', 'draft', 'verify', 'accept', 'tally'] as const,
    captions: {
      commit: (
        <>
          We have committed <strong>5 tokens</strong> so far. Plain decoding would now run the full model once to
          get exactly <strong>one</strong> more token. Speculative decoding tries to get several.
        </>
      ),
      draft: (
        <>
          The cheap <span className="font-mono">MTP</span> head proposes <strong>D = {D}</strong> draft tokens{' '}
          <span className="font-mono">d₁…d₃</span>, one after another — fast, but only a guess (shown dashed).
        </>
      ),
      verify: (
        <>
          The <strong>full model</strong> now checks the whole window{' '}
          <span className="font-mono">[is, d₁, d₂, d₃]</span> in <strong>one</strong> batched forward pass over{' '}
          <span className="font-mono">D+1 = {D + 1}</span> positions — this is the expensive part, done once.
        </>
      ),
      acceptPartial: (k: number) => (
        <>
          Walk left→right: the first <strong>{k}</strong> drafts match the full model&apos;s own pick, so they are{' '}
          <span className="text-emerald-600 dark:text-emerald-400">accepted ✓</span>. The next one mismatches —{' '}
          <span className="text-red-600 dark:text-red-400">rejected ✗</span> — so we emit the model&apos;s token
          instead (the correction) and stop.
        </>
      ),
      acceptFull: (
        <>
          Walk left→right: <strong>all {D}</strong> drafts match the full model&apos;s own pick —{' '}
          <span className="text-emerald-600 dark:text-emerald-400">all accepted ✓</span>. Since none was rejected,
          we also keep the model&apos;s pick at position <span className="font-mono">D+1</span> as a free{' '}
          <strong>bonus</strong> token.
        </>
      ),
      tally: (n: number) => (
        <>
          One verify pass emitted <strong>{n}</strong> real tokens. And the output is{' '}
          <strong>identical</strong> to plain one-at-a-time decoding — a draft is kept only when it matches the full
          model&apos;s own choice, so this is <em>lossless</em>, never a quality trade.
        </>
      ),
    },
    losslessChip: 'lossless · same distribution as plain decode',
    speedChip: (
      <>
        best case: <span className="font-mono">D+1 = {D + 1}</span> tokens / pass · worst case: 1 token + a small draft
        tax
      </>
    ),
    tokensThisPass: (n: number) => (
      <>
        tokens this pass: <span className="font-mono text-primary">{n}</span>
      </>
    ),
    runningTally: (passes: number, tokens: number, avg: string) => (
      <>
        passes: <span className="font-mono">{passes}</span> · tokens:{' '}
        <span className="font-mono">{tokens}</span> →{' '}
        <span className="font-mono text-primary">{avg}</span> per pass
      </>
    ),
    illustrativeCaption: "Illustrative tokens — the concept, not this model's real output.",
    ariaLabel:
      'Diagram of self-speculative decoding: a cheap MTP head drafts D candidate tokens, the full model verifies all D+1 positions in one batched pass, then accepted drafts turn green while the first mismatch is corrected in red — emitting K+1 lossless tokens from a single verify pass.',
    footnote:
      "Measured in this project's native engine: ~1.06–1.46× faster, ~1.3–1.46 tokens per verify pass at temperature 0 — the gain depends on how predictable the text is. Qwen3.5's default draft depth is 1.",
  },
  zh: {
    title: 'speculative decoding——draft、verify、accept',
    pause: '❚❚ 暂停',
    play: '▶ 播放',
    prevAria: '上一阶段',
    nextAria: '下一阶段',
    scenarioLabel: '场景',
    scenarioA: '部分接受',
    scenarioB: '全部接受',
    scenarioAAria: '场景 A——部分接受：保留两个 draft，纠正一个',
    scenarioBAria: '场景 B——全部接受：保留全部 draft 并额外赠送一个 token',
    committedLabel: '已提交',
    draftLabel: 'MTP draft',
    verifyLabel: '全模型 verify（一遍）',
    draftBadge: 'draft',
    correctionBadge: '纠正',
    bonusBadge: '赠送',
    phases: ['提交', 'draft', 'verify', 'accept', '统计'] as const,
    captions: {
      commit: (
        <>
          目前已经提交了 <strong>5 个 token</strong>。普通解码现在会把全模型跑一遍，恰好再得到{' '}
          <strong>一个</strong> token。speculative decoding 试图一次拿到好几个。
        </>
      ),
      draft: (
        <>
          廉价的 <span className="font-mono">MTP</span> head 一个接一个地提出 <strong>D = {D}</strong> 个 draft token{' '}
          <span className="font-mono">d₁…d₃</span>——很快，但只是猜测（用虚线表示）。
        </>
      ),
      verify: (
        <>
          <strong>全模型</strong> 现在在 <strong>一遍</strong> 批处理前向中检查整个窗口{' '}
          <span className="font-mono">[is, d₁, d₂, d₃]</span>，覆盖{' '}
          <span className="font-mono">D+1 = {D + 1}</span> 个位置——这是昂贵的部分，只做一次。
        </>
      ),
      acceptPartial: (k: number) => (
        <>
          从左到右扫描：前 <strong>{k}</strong> 个 draft 与全模型自己的选择一致，于是被{' '}
          <span className="text-emerald-600 dark:text-emerald-400">接受 ✓</span>。下一个不匹配——{' '}
          <span className="text-red-600 dark:text-red-400">拒绝 ✗</span>——于是改为输出模型的 token（纠正）并停止。
        </>
      ),
      acceptFull: (
        <>
          从左到右扫描：<strong>全部 {D} 个</strong> draft 都与全模型自己的选择一致——{' '}
          <span className="text-emerald-600 dark:text-emerald-400">全部接受 ✓</span>。由于没有任何拒绝，我们还把位置{' '}
          <span className="font-mono">D+1</span> 处模型的选择作为免费的 <strong>赠送</strong> token 一并保留。
        </>
      ),
      tally: (n: number) => (
        <>
          一次 verify 输出了 <strong>{n}</strong> 个真实 token。而且输出与逐个解码{' '}
          <strong>完全一致</strong>——只有当 draft 匹配全模型自己的选择时才会保留，所以这是 <em>无损</em>{' '}
          的，绝不牺牲质量。
        </>
      ),
    },
    losslessChip: '无损 · 与普通解码分布一致',
    speedChip: (
      <>
        最好情况：<span className="font-mono">D+1 = {D + 1}</span> token / 遍 · 最坏情况：1 个 token + 一点点草稿开销
      </>
    ),
    tokensThisPass: (n: number) => (
      <>
        本遍 token：<span className="font-mono text-primary">{n}</span>
      </>
    ),
    runningTally: (passes: number, tokens: number, avg: string) => (
      <>
        遍数：<span className="font-mono">{passes}</span> · token：{' '}
        <span className="font-mono">{tokens}</span> →{' '}
        <span className="font-mono text-primary">{avg}</span> / 遍
      </>
    ),
    illustrativeCaption: '示意 token——讲概念，并非本模型的真实输出。',
    ariaLabel:
      '自推测解码（self-speculative decoding）示意图：廉价的 MTP head 起草 D 个候选 token，全模型在一遍批处理中验证全部 D+1 个位置，随后被接受的 draft 变绿、第一个不匹配处用红色纠正——单次 verify 输出 K+1 个无损 token。',
    footnote:
      '在本项目原生引擎中实测：约 1.06–1.46× 加速，temperature 0 时每次 verify 约 1.3–1.46 个 token——增益取决于文本有多可预测。Qwen3.5 的默认 draft depth 为 1。',
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
const VB_H = 200;
const CELL_W = 92;
const CELL_H = 44;
const CELL_GAP = 8;
const ROW_Y = 70;
const ROW_X = 16;

// The committed prefix is shown compact (the last 2 cells), then a gap, then the
// verify window: the anchor (last committed) + the D draft slots.
const PREFIX_SHOWN = 2;
const WINDOW_GAP = 26;

function prefixCellX(i: number): number {
  return ROW_X + i * (CELL_W + CELL_GAP);
}
function windowStartX(): number {
  return ROW_X + PREFIX_SHOWN * (CELL_W + CELL_GAP) + WINDOW_GAP;
}
/** x of window slot j: j=0 is the anchor (last committed), j=1..D are draft slots. */
function windowCellX(j: number): number {
  return windowStartX() + j * (CELL_W + CELL_GAP);
}

export function SpeculativeDecodeDiagram() {
  const copy = COPY[useLocale()];
  const reducedMotion = usePrefersReducedMotion();
  const [scenarioIdx, setScenarioIdx] = React.useState(0);
  // Under reduced motion: jump to the final (tally) phase of Scenario A, never play.
  const [phase, setPhase] = React.useState(reducedMotion ? PHASES - 1 : 0);
  const [playing, setPlaying] = React.useState(!reducedMotion);
  // Running tally accumulated as the user/auto-play completes whole passes.
  const [passes, setPasses] = React.useState(0);
  const [tokens, setTokens] = React.useState(0);
  // Guard so one completed pass increments the running tally exactly once.
  const tallied = React.useRef(false);

  const scenario = SCENARIOS[scenarioIdx];
  const fr = firstReject(scenario);
  const isFullAccept = fr === -1;
  const nThisPass = tokensThisPass(scenario);

  React.useEffect(() => {
    if (!playing || reducedMotion) return;
    const t = window.setInterval(() => setPhase((p) => (p + 1) % PHASES), FRAME_MS);
    return () => window.clearInterval(t);
  }, [playing, reducedMotion]);

  // When we land on the tally phase, fold this pass into the running average once.
  React.useEffect(() => {
    if (phase === PHASES - 1 && !tallied.current) {
      tallied.current = true;
      setPasses((p) => p + 1);
      setTokens((t) => t + nThisPass);
    }
    if (phase !== PHASES - 1) {
      tallied.current = false;
    }
  }, [phase, nThisPass]);

  const step = (dir: number) => {
    setPlaying(false);
    setPhase((p) => Math.min(PHASES - 1, Math.max(0, p + dir)));
  };
  const pickScenario = (i: number) => {
    setPlaying(false);
    setScenarioIdx(i);
    setPhase(0);
    tallied.current = false;
  };

  // Phase booleans.
  const showDrafts = phase >= 1; // drafts visible from phase 1
  const verifying = phase === 2; // all D+1 positions highlighted at once
  const resolved = phase >= 3; // accept/reject coloring applied
  const isTally = phase === PHASES - 1;

  // The plain-language caption per phase.
  let caption: React.ReactNode;
  if (phase === 0) caption = copy.captions.commit;
  else if (phase === 1) caption = copy.captions.draft;
  else if (phase === 2) caption = copy.captions.verify;
  else if (phase === 3) caption = isFullAccept ? copy.captions.acceptFull : copy.captions.acceptPartial(fr);
  else caption = copy.captions.tally(nThisPass);

  const avg = passes > 0 ? (tokens / passes).toFixed(2) : '—';
  const anchor = scenario.committed[scenario.committed.length - 1];
  const shownPrefix = scenario.committed.slice(-PREFIX_SHOWN - 1, -1); // 2 cells before the anchor

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + controls */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <div className="flex items-center gap-1">
          {/* Scenario selector */}
          <div className="mr-1 flex items-center gap-1 text-[11px] text-muted-foreground">
            <span className="hidden sm:inline">{copy.scenarioLabel}:</span>
            <div className="flex overflow-hidden rounded border border-border">
              <button
                type="button"
                onClick={() => pickScenario(0)}
                aria-pressed={scenarioIdx === 0}
                aria-label={copy.scenarioAAria}
                className="px-2 py-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                style={{
                  background: scenarioIdx === 0 ? 'var(--primary)' : 'transparent',
                  color: scenarioIdx === 0 ? 'var(--background)' : 'var(--muted-foreground)',
                }}
              >
                A · {copy.scenarioA}
              </button>
              <button
                type="button"
                onClick={() => pickScenario(1)}
                aria-pressed={scenarioIdx === 1}
                aria-label={copy.scenarioBAria}
                className="border-l border-border px-2 py-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                style={{
                  background: scenarioIdx === 1 ? 'var(--primary)' : 'transparent',
                  color: scenarioIdx === 1 ? 'var(--background)' : 'var(--muted-foreground)',
                }}
              >
                B · {copy.scenarioB}
              </button>
            </div>
          </div>
          {!reducedMotion ? (
            <button
              type="button"
              onClick={() => setPlaying((p) => !p)}
              aria-pressed={playing}
              className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            >
              {playing ? copy.pause : copy.play}
            </button>
          ) : null}
          <button
            type="button"
            onClick={() => step(-1)}
            disabled={phase === 0}
            aria-label={copy.prevAria}
            className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary disabled:opacity-30"
          >
            ‹
          </button>
          <button
            type="button"
            onClick={() => step(1)}
            disabled={phase === PHASES - 1}
            aria-label={copy.nextAria}
            className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary disabled:opacity-30"
          >
            ›
          </button>
        </div>
      </div>

      {/* Phase rail — 5 dots */}
      <div className="flex items-center gap-1.5">
        {copy.phases.map((label, i) => {
          const active = i === phase;
          const passed = i < phase;
          return (
            <div key={label} className="flex items-center gap-1.5">
              <span
                className="h-2 w-2 rounded-full transition-all duration-300"
                style={{
                  background: active || passed ? 'var(--primary)' : 'var(--card)',
                  border: `1.5px solid ${active || passed ? 'var(--primary)' : 'var(--border)'}`,
                  transform: active ? 'scale(1.4)' : 'scale(1)',
                }}
              />
              <span
                className="font-mono text-[11px]"
                style={{ color: active ? 'var(--foreground)' : 'var(--muted-foreground)' }}
              >
                {label}
              </span>
            </div>
          );
        })}
      </div>

      {/* The diagram */}
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.ariaLabel}>
        {/* Section labels */}
        <text x={ROW_X} y={ROW_Y - 16} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
          {copy.committedLabel}
        </text>
        <text x={windowStartX()} y={ROW_Y - 16} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
          {showDrafts ? (verifying || resolved ? copy.verifyLabel : copy.draftLabel) : copy.draftLabel}
        </text>

        {/* Committed prefix cells (last 2 before the anchor) */}
        {shownPrefix.map((tok, i) => {
          const x = prefixCellX(i);
          return (
            <g key={`p${i}`}>
              <rect
                x={x}
                y={ROW_Y}
                width={CELL_W}
                height={CELL_H}
                rx={6}
                style={{ fill: 'var(--card)', stroke: 'var(--border)' }}
                strokeWidth={1.5}
              />
              <text
                x={x + CELL_W / 2}
                y={ROW_Y + CELL_H / 2 + 5}
                textAnchor="middle"
                style={{ fill: 'var(--muted-foreground)' }}
                className="font-mono text-[14px]"
              >
                {tok}
              </text>
            </g>
          );
        })}
        {/* ellipsis hinting there's more prefix to the left */}
        <text
          x={ROW_X - 4}
          y={ROW_Y + CELL_H / 2 + 5}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)' }}
          className="font-mono text-[14px]"
        >
          …
        </text>

        {/* Verify window: anchor (last committed) + D draft slots */}
        {/* verify-pass bracket over all D+1 positions */}
        {verifying ? (
          <rect
            x={windowCellX(0) - 5}
            y={ROW_Y - 6}
            width={(D + 1) * CELL_W + D * CELL_GAP + 10}
            height={CELL_H + 12}
            rx={9}
            fill="var(--primary)"
            fillOpacity={0.08}
            stroke="var(--primary)"
            strokeWidth={2}
          />
        ) : null}

        {/* anchor cell */}
        <g>
          <rect
            x={windowCellX(0)}
            y={ROW_Y}
            width={CELL_W}
            height={CELL_H}
            rx={6}
            style={{
              fill: verifying ? 'var(--primary)' : 'var(--card)',
              fillOpacity: verifying ? 0.16 : 1,
              stroke: verifying ? 'var(--primary)' : 'var(--border)',
            }}
            strokeWidth={verifying ? 2 : 1.5}
          />
          <text
            x={windowCellX(0) + CELL_W / 2}
            y={ROW_Y + CELL_H / 2 + 5}
            textAnchor="middle"
            style={{ fill: 'var(--foreground)' }}
            className="font-mono text-[14px] font-semibold"
          >
            {anchor}
          </text>
        </g>

        {/* D draft slots */}
        {scenario.drafts.map((draft, j) => {
          const slot = j + 1; // window slot index (anchor is 0)
          const x = windowCellX(slot);
          const accepted = resolved && scenario.verdicts[j] === 'accept';
          const rejected = resolved && scenario.verdicts[j] === 'reject';
          // After resolution, a rejected slot shows the CORRECTION instead of the draft.
          const display = rejected ? scenario.correction : draft;
          // Border/strokes: dashed while a draft; solid once resolved.
          let stroke = 'var(--border)';
          let fill = 'var(--card)';
          let fillOpacity = 1;
          let textColor = 'var(--muted-foreground)';
          let dash: string | undefined = showDrafts ? '5 4' : undefined;
          if (!showDrafts) {
            // phase 0: draft slots are empty placeholders
            fillOpacity = 0;
            stroke = 'transparent';
          } else if (verifying) {
            stroke = 'var(--primary)';
            fill = 'var(--primary)';
            fillOpacity = 0.16;
            textColor = 'var(--foreground)';
            dash = undefined;
          } else if (accepted) {
            stroke = EMERALD;
            fill = EMERALD;
            fillOpacity = 0.14;
            textColor = EMERALD;
            dash = undefined;
          } else if (rejected) {
            stroke = RED;
            fill = RED;
            fillOpacity = 0.14;
            textColor = RED;
            dash = undefined;
          } else {
            // phase 1 (draft proposed, not yet verified): muted + dashed
            textColor = 'var(--muted-foreground)';
          }
          return (
            <g key={`d${j}`} style={{ opacity: showDrafts ? 1 : 0, transition: 'opacity 300ms' }}>
              <rect
                x={x}
                y={ROW_Y}
                width={CELL_W}
                height={CELL_H}
                rx={6}
                fill={fill}
                fillOpacity={fillOpacity}
                stroke={stroke}
                strokeWidth={accepted || rejected || verifying ? 2 : 1.5}
                strokeDasharray={dash}
              />
              {showDrafts ? (
                <text
                  x={x + CELL_W / 2}
                  y={ROW_Y + CELL_H / 2 + 5}
                  textAnchor="middle"
                  fill={textColor}
                  className="font-mono text-[14px] font-semibold"
                >
                  {display}
                </text>
              ) : null}
              {/* draft index label dᵢ above each slot, phases 1-2 */}
              {showDrafts && !resolved ? (
                <text
                  x={x + CELL_W / 2}
                  y={ROW_Y - 16}
                  textAnchor="middle"
                  style={{ fill: 'var(--muted-foreground)' }}
                  className="font-mono text-[11px]"
                >
                  d{j + 1}
                </text>
              ) : null}
              {/* ✓ / ✗ verdict marks + badge, phase 3+ */}
              {resolved ? (
                <text
                  x={x + CELL_W - 8}
                  y={ROW_Y + 16}
                  textAnchor="end"
                  fill={accepted ? EMERALD : RED}
                  className="text-[13px] font-bold"
                >
                  {accepted ? '✓' : '✗'}
                </text>
              ) : null}
            </g>
          );
        })}

        {/* Bonus token cell (full-accept only), appears in phase 3+ to the right */}
        {resolved && isFullAccept ? (
          <g style={{ opacity: 1, transition: 'opacity 300ms' }}>
            <rect
              x={windowCellX(D + 1)}
              y={ROW_Y}
              width={CELL_W}
              height={CELL_H}
              rx={6}
              fill={EMERALD}
              fillOpacity={0.14}
              stroke={EMERALD}
              strokeWidth={2}
            />
            <text
              x={windowCellX(D + 1) + CELL_W / 2}
              y={ROW_Y + CELL_H / 2 + 5}
              textAnchor="middle"
              fill={EMERALD}
              className="font-mono text-[14px] font-semibold"
            >
              {scenario.bonus}
            </text>
            <text
              x={windowCellX(D + 1) + CELL_W / 2}
              y={ROW_Y - 16}
              textAnchor="middle"
              fill={EMERALD}
              className="text-[11px] font-semibold"
            >
              {copy.bonusBadge} +1
            </text>
          </g>
        ) : null}

        {/* draft / correction badge under the window (phase 1+ / phase 3+) */}
        {showDrafts && !resolved ? (
          <text
            x={windowCellX(1)}
            y={ROW_Y + CELL_H + 22}
            style={{ fill: 'var(--muted-foreground)' }}
            className="text-[11px]"
          >
            ← {copy.draftBadge} d1…d{D} (dashed = unverified)
          </text>
        ) : null}
        {resolved && !isFullAccept ? (
          <text
            x={windowCellX(fr + 1) + CELL_W / 2}
            y={ROW_Y + CELL_H + 22}
            textAnchor="middle"
            fill={RED}
            className="text-[11px] font-medium"
          >
            ↑ {copy.correctionBadge}
          </text>
        ) : null}
      </svg>

      {/* Illustrative-tokens caption */}
      <p className="text-[10px] italic text-muted-foreground/70">{copy.illustrativeCaption}</p>

      {/* Plain-language caption per phase */}
      <p className="min-h-[3.25rem] text-[13px] text-foreground/90">{caption}</p>

      {/* Status chips */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1.5 rounded border border-emerald-500/50 bg-emerald-500/10 px-2 py-0.5 text-[11px] text-emerald-700 dark:text-emerald-300">
          {copy.losslessChip}
        </span>
        <span className="inline-flex items-center gap-1.5 rounded border border-border bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">
          {copy.speedChip}
        </span>
      </div>

      {/* Tally — tokens this pass + running average */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border/60 pt-3">
        <div
          className="text-[12px] font-medium"
          style={{ color: isTally ? 'var(--foreground)' : 'var(--muted-foreground)' }}
        >
          {copy.tokensThisPass(nThisPass)}
        </div>
        <div className="text-[11px] text-muted-foreground">{copy.runningTally(passes, tokens, avg)}</div>
      </div>

      <p className="text-[10px] text-muted-foreground/70">{copy.footnote}</p>
    </div>
  );
}
