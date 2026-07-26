import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import {
  DASH,
  DIM,
  DiagramFrame,
  EMERALD,
  HatchDefs,
  hatchFill,
  PanBox,
  RED,
  RX,
  SW,
  useStepPlayer,
} from '../motion';

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
 *   1 draft   — the MTP head's d₁…d_D appear, dim + dashed ("draft").
 *   2 verify  — a RED bracket over all D+1 positions and a RED hatch inside each
 *               of them: one expensive pass, chewing on every position at once.
 *   3 accept  — left→right resolve: accepted drafts → EMERALD ✓; first mismatch →
 *               RED ✗ replaced by the correction; or, on full accept, a bonus
 *               token appears.
 *   4 tally   — "tokens this pass: K+1" + running passes/tokens average.
 *
 * Two canned scenarios differ only in the per-position verify verdicts:
 *   A "partial accept": D=3, accept d₁,d₂, reject d₃ → correction → 3 tokens.
 *   B "full accept":    D=3, accept all → +1 bonus      → 4 tokens.
 *
 * Animation model: `useStepPlayer` from `learn/motion`, replacing this file's
 * former hand-rolled `setInterval` + private `usePrefersReducedMotion` + play /
 * prev / next buttons. Reduced-motion readers still rest on the final (tally)
 * frame — that is `useStepPlayer`'s default `restFrame` — and still lose the
 * Play button but keep Step.
 *
 * ── THE PALETTE ────────────────────────────────────────────────────────────
 *
 * This diagram is `skin`'s two hues used for exactly what they mean, and the
 * accept/reject pair is a contract with two neighbours (PrefixCacheDiagram,
 * SpeculativeVerifyLive) that hardcode the same two hexes:
 *
 *   dashed, dim, neutral   a draft — proposed, not verified, NOT REAL YET. This
 *                          holds through phase 2 as well: the verify pass is
 *                          still running, so a guess is still a guess.
 *   RED + diagonal hatch   the one expensive full-model pass, happening now
 *   RED, solid             the correction the full model had to emit itself
 *   EMERALD, solid         an accepted draft, or the free bonus token — kept
 *
 * The one place this departs from `skin`'s "hue is permanent identity" rule is
 * the same departure BackpropFlowDiagram documents: every cell here has the
 * SAME job (it is a token slot), so there is no competing identity for hue to
 * lose, and the only thing the reader tracks is where the sweep is. "Not real
 * yet" is still the DASH and "busy this step" is still the HATCH — a draft is
 * never given a hue of its own.
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
//
// Play / Pause / Step are NOT here any more, and neither are the prev/next
// phase arrows: those verbs now belong to the shared `StepControls`, which
// carries its own en/zh pair so every animated widget says the same word.
const COPY = {
  en: {
    title: 'Speculative decoding — draft, verify, accept',
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
  const locale = useLocale();
  const copy = COPY[locale];
  // Reduced-motion readers rest on the LAST frame (the tally) — `useStepPlayer`'s
  // default `restFrame` — which is the same still this widget used to pin.
  const player = useStepPlayer(PHASES, { frameMs: FRAME_MS });
  const phase = player.frame;
  const [scenarioIdx, setScenarioIdx] = React.useState(0);
  // Running tally accumulated as the user/auto-play completes whole passes.
  const [passes, setPasses] = React.useState(0);
  const [tokens, setTokens] = React.useState(0);
  // Guard so one completed pass increments the running tally exactly once.
  const tallied = React.useRef(false);

  const scenario = SCENARIOS[scenarioIdx];
  const fr = firstReject(scenario);
  const isFullAccept = fr === -1;
  const nThisPass = tokensThisPass(scenario);

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

  const pickScenario = (i: number) => {
    setScenarioIdx(i);
    // `goTo` pauses first, so switching scenario parks the reader on the commit
    // phase of the new one — the same "stop and restart" the old handler did.
    player.goTo(0);
    tallied.current = false;
  };

  // Phase booleans.
  const showDrafts = phase >= 1; // drafts visible from phase 1
  const verifying = phase === 2; // the one expensive pass, chewing on D+1 positions
  const resolved = phase >= 3; // accept/reject coloring applied
  const isTally = phase === PHASES - 1;

  // The plain-language caption per phase.
  let caption: React.ReactNode;
  if (phase === 0) caption = copy.captions.commit;
  else if (phase === 1) caption = copy.captions.draft;
  else if (phase === 2) caption = copy.captions.verify;
  else if (phase === 3) caption = isFullAccept ? copy.captions.acceptFull : copy.captions.acceptPartial(fr);
  else caption = copy.captions.tally(nThisPass);

  // Every caption this sweep can reach, so DiagramFrame reserves the tallest and
  // the chapter body cannot hop when the beat changes. Phases 0-2 read the same
  // in both scenarios; phases 3 and 4 do NOT — the accept beat forks on
  // `isFullAccept` and the tally beat interpolates `tokensThisPass` — and the
  // scenario toggle can swap those at any frame, so BOTH scenarios' arms are
  // enumerated here, off the same two helpers the visible caption uses.
  const captions = [
    copy.captions.commit,
    copy.captions.draft,
    copy.captions.verify,
    ...SCENARIOS.flatMap((s) => {
      const r = firstReject(s);
      return [
        r === -1 ? copy.captions.acceptFull : copy.captions.acceptPartial(r),
        copy.captions.tally(tokensThisPass(s)),
      ];
    }),
  ];

  /** Skip every transition under reduced motion — Step is a state change, not motion. */
  const ease = player.reducedMotion ? undefined : 'stroke 300ms ease, stroke-opacity 300ms ease';
  const fade = player.reducedMotion ? undefined : 'opacity 300ms';

  const avg = passes > 0 ? (tokens / passes).toFixed(2) : '—';
  const anchor = scenario.committed[scenario.committed.length - 1];
  const shownPrefix = scenario.committed.slice(-PREFIX_SHOWN - 1, -1); // 2 cells before the anchor

  return (
    <DiagramFrame
      title={copy.title}
      player={player}
      locale={locale}
      caption={caption}
      captions={captions}
      note={copy.footnote}
    >
      {/* Scenario selector — a variant toggle, not a player control, so it stays
          with the widget and sits directly above the diagram body. */}
      <div className="flex flex-wrap items-center gap-1 text-[11px] text-muted-foreground">
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

      {/* The diagram.
          `min-w` is the reason this svg — and ONLY this svg — is wrapped in
          `PanBox`. A bare `w-full` svg does not reflow in a narrow column — it
          SCALES, text and all, so at a 375px viewport (a 320-unit column) every
          label here lands around 4.6 CSS px and stops being readable. The floor
          stops the shrink and `PanBox` pans to reach the rest.
          8px legibility floor / smallest size a reader must read: 8 * 760 / 11
          = 552.7 → 553. That smallest size is 11px, and all six of its uses are
          real prose — the two section labels, the d1…d3 draft indices, and the
          draft / correction / bonus badges — so none of them get to duck under
          the floor the way a true superscript would.
          The floor belongs to the svg ALONE, which is why `PanBox` wraps the
          svg and nothing else: the scenario toggle and the phase rail above,
          and the chips and tally below, stay DiagramFrame's own children — so
          they size to the visible width, keep the frame's `space-y-4` gaps, and
          stay reachable without panning. */}
      <PanBox locale={locale}>
        <svg
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          className="w-full min-w-[553px]"
          role="img"
          aria-label={copy.ariaLabel}
        >
          <HatchDefs />

          {/* Section labels. Bare `<text>`, NOT FrameLabel: both sit at ROW_Y - 16,
              a clear 10 units above the verify bracket's top edge (ROW_Y - 6), so
              there is no stroke under either one for a plate to erase. */}
          <text x={ROW_X} y={ROW_Y - 16} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
            {copy.committedLabel}
          </text>
          <text
            x={windowStartX()}
            y={ROW_Y - 16}
            style={{ fill: verifying ? RED : 'var(--muted-foreground)' }}
            className="text-[11px]"
          >
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
                  rx={RX}
                  style={{ fill: 'var(--card)', stroke: 'var(--border)' }}
                  strokeWidth={SW.OUTER}
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
          {/* The verify-pass bracket over all D+1 positions. RED because it IS the
              expensive full-model pass, which is exactly what `skin` reserves RED
              for — and a bare rect, NOT a PanelFrame: a 56-unit-tall PanelFrame
              insets its dashed inner border by 5, which lands one unit off the
              cell outlines it wraps and reads as a stray second stroke. This is a
              bracket AROUND boxes, not a stage box with contents drawn inside. */}
          {verifying ? (
            <rect
              x={windowCellX(0) - 5}
              y={ROW_Y - 6}
              width={(D + 1) * CELL_W + D * CELL_GAP + 10}
              height={CELL_H + 12}
              rx={RX}
              fill="none"
              stroke={RED}
              strokeWidth={SW.OUTER}
            />
          ) : null}

          {/* anchor cell — the last committed token. Permanently neutral: it is
              real in every frame. Verify does not change that, it only makes it
              BUSY, which is the hatch's job. */}
          <g>
            <rect
              x={windowCellX(0)}
              y={ROW_Y}
              width={CELL_W}
              height={CELL_H}
              rx={RX}
              style={{ fill: 'var(--card)', stroke: 'var(--border)' }}
              strokeWidth={SW.OUTER}
            />
            {verifying ? (
              <rect x={windowCellX(0)} y={ROW_Y} width={CELL_W} height={CELL_H} rx={RX} fill={hatchFill('red')} />
            ) : null}
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
            // "Becomes real" is `skin`'s three-part transform, applied all at once:
            // dash off, SW.INNER -> SW.OUTER, neutral -> hued. Note what is NOT in
            // that list: `verifying`. A draft under verification is still a guess
            // — the pass has not answered yet — so phases 1 AND 2 both stay dashed
            // and dim, and "the expensive thing is happening to this cell" is
            // carried by the hatch, which is orthogonal to both channels.
            const real = accepted || rejected;
            const hue = accepted ? EMERALD : rejected ? RED : undefined;
            const textColor = hue ?? (verifying ? 'var(--foreground)' : 'var(--muted-foreground)');
            return (
              <g key={`d${j}`} style={{ opacity: showDrafts ? 1 : 0, transition: fade }}>
                <rect
                  x={x}
                  y={ROW_Y}
                  width={CELL_W}
                  height={CELL_H}
                  rx={RX}
                  style={{
                    fill: 'var(--card)',
                    stroke: hue ?? 'var(--muted-foreground)',
                    transition: ease,
                  }}
                  strokeOpacity={real ? 1 : DIM.EMPTY_SLOT}
                  strokeWidth={real ? SW.OUTER : SW.INNER}
                  strokeDasharray={real ? undefined : DASH.BORDER}
                />
                {verifying ? (
                  <rect x={x} y={ROW_Y} width={CELL_W} height={CELL_H} rx={RX} fill={hatchFill('red')} />
                ) : null}
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
                {/* draft index label dᵢ above each slot — phase 1 ONLY, not 1-2.
                    On phase 2 the section label above swaps to the much longer
                    verify string, and in EN it collides: 'full-model verify (one
                    pass)' measures 143.9 units from x=242 (right edge 385.9) while
                    d₁'s box starts at 381.4 — a 4.5-unit overlap. (zh's shorter
                    string ends at 351.6 and clears by 29.8.) The drafts are still
                    labelled on phase 1 and named in the verify caption, so hiding
                    the dᵢ here costs nothing. */}
                {showDrafts && !verifying && !resolved ? (
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

          {/* Bonus token cell (full-accept only), appears in phase 3+ to the right.
              Real the moment it is drawn — the full model produced it — so it is
              solid + SW.OUTER + EMERALD, never dashed. */}
          {resolved && isFullAccept ? (
            <g style={{ opacity: 1, transition: fade }}>
              <rect
                x={windowCellX(D + 1)}
                y={ROW_Y}
                width={CELL_W}
                height={CELL_H}
                rx={RX}
                style={{ fill: 'var(--card)', stroke: EMERALD }}
                strokeWidth={SW.OUTER}
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
      </PanBox>

      {/* Illustrative-tokens caption */}
      <p className="text-[10px] italic text-muted-foreground/70">{copy.illustrativeCaption}</p>

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
    </DiagramFrame>
  );
}
