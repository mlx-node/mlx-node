import { cn } from '@/lib/utils';
import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Chapter 12 (KV cache & hybrid attention) supplement — the MECHANISM at the
 * token level: appending vs. updating-in-place.
 *
 * The chapter's hybrid-attention prose says GatedDeltaNet "compresses the
 * entire history into a fixed-size recurrent state … rolled forward at each
 * step the way an RNN's hidden state is." This widget makes that single
 * sentence physical by streaming the SAME tokens through two panels side by
 * side and letting the learner SEE the difference, token by token:
 *
 *   - FULL ATTENTION (the KV cache): every token APPENDS a new (K,V) entry.
 *     After k tokens there are k stored boxes. Memory ∝ N — it grows forever.
 *   - LINEAR ATTENTION (GatedDeltaNet-style running state): ONE fixed-size slot
 *     S that is UPDATED IN PLACE each step and never grows. Memory = constant.
 *
 * The running-state update is the leaky recurrence
 *
 *     S ← decay · S + u_t
 *
 * where u_t is the new token's scalar contribution. Unrolling it gives the
 * closed form the visualization draws as a stacked bar:
 *
 *     S_k = Σ_{i ≤ k} u_i · decay^(k − i)
 *
 * so every past token's *current* share of S is u_i · decay^age, age = k − i.
 * Older tokens fade as decay^age, which is exactly why a low decay forgets fast
 * and a high decay keeps a longer (but still lossy) memory horizon — the
 * memory-vs-recall trade-off the chapter's "linear attention is approximate"
 * section is about. The decay slider drives that fade directly.
 *
 * Pure presentational — no model, no worker, no WASM, no network. Only React
 * and `cn`. Fully keyboard-operable (native button + slider); auto-play is
 * disabled under prefers-reduced-motion (manual "feed next token" only).
 */

// A short, fixed token stream. Each token carries a small positive scalar
// "contribution" u_t — a stand-in for the value it writes into the running
// state. Hand-picked to read naturally and to keep S comfortably in range.
type Tok = { text: string; u: number };

const STREAM: ReadonlyArray<Tok> = [
  { text: 'The', u: 0.6 },
  { text: 'access', u: 0.9 },
  { text: 'code', u: 0.7 },
  { text: 'is', u: 0.4 },
  { text: 'X7', u: 1.0 },
  { text: 'then', u: 0.5 },
  { text: 'go', u: 0.8 },
];
const N = STREAM.length;

const DECAY_MIN = 0.5;
const DECAY_MAX = 0.95;
const DECAY_DEFAULT = 0.85;

// Auto-play cadence (motion allowed only).
const RUN_MS = 900;

// Per-locale copy. COPY.en is the original English, moved verbatim.
const COPY = {
  en: {
    title: 'Appending vs. updating in place',
    subtitle: (
      <>
        Same token stream, two memories: <span className="font-mono">cache grows</span> vs.{' '}
        <span className="font-mono">S ← decay·S + u</span>
      </>
    ),
    streamComplete: 'Stream complete',
    feedNext: 'Feed next token →',
    pause: 'Pause',
    replay: 'Replay',
    run: 'Run',
    reset: 'Reset',
    tokensCounter: (fed: number, n: number) => `${fed} / ${n} tokens`,
    fullPanelLabel: 'Full attention (KV cache)',
    appendsEvery: 'appends every token',
    emptyCache: 'empty cache',
    memoryEq: 'memory = ',
    unitWord: (n: number) => ` unit${n === 1 ? '' : 's'}`,
    growsNote: '(grows ∝ N)',
    linearPanelLabel: 'Linear state (GatedDeltaNet)',
    updatesOneSlot: 'updates one slot',
    oneFixedSlot: 'one fixed slot',
    shareTitle: (text: string, pct: string, age: number) => `${text}: ${pct}% of S (decay^${age})`,
    slotZero: 'one slot, currently zero',
    fadeNote: 'newest token brightest, older contributions fade as decayᵃᵍᵉ',
    unitOne: ' unit',
    constantNote: '(constant)',
    decayDesc: (horizon: number) =>
      `— controls how far back the state still remembers (≈ ${horizon} recent token${horizon === 1 ? '' : 's'} hold ≥10% weight)`,
    forgetsFastTitle: 'forgets fast — short memory horizon',
    forgetsFast: (v: string) => `${v} (forgets fast)`,
    longerMemoryTitle: 'remembers longer, but still lossy',
    longerMemory: (v: string) => `${v} (longer memory)`,
    tokensFedTile: 'tokens fed',
    fullAttentionTile: 'full attention',
    growsTile: '(grows)',
    linearStateTile: 'linear state',
    oneUnit: '1 unit',
    takeawayEnd: (n: number) =>
      `After ${n} tokens: full attention is storing ${n} entries (and would keep growing); the running state is still one slot.`,
    takeawayStart:
      'Feed a token to start. Watch the left side grow by one box each step while the right side updates one slot in place.',
    takeawayProgress: (fed: number) =>
      `${fed} token${fed === 1 ? '' : 's'} fed: full attention now holds ${fed} entries; the running state is still exactly one slot.`,
    liveSummary: (fed: number, n: number, s: string, lastText: string | undefined) =>
      `Tokens fed: ${fed} of ${n}. ` +
      `Full attention memory: ${fed} stored entr${fed === 1 ? 'y' : 'ies'} (grows one per token). ` +
      `Linear running state: 1 fixed slot (constant), current value S = ${s}` +
      (lastText ? `, after writing "${lastText}".` : '.'),
    rnnGloss: (
      <>
        That single running slot <strong>is a recurrent neural network&apos;s hidden state</strong>: a fixed-size
        memory rolled forward each step (<span className="font-mono">S ← decay·S + u</span>), like a running total or
        moving average. Full attention instead keeps <em>every</em> past token, so its memory grows without bound.
        That is the <strong>memory-vs-recall trade-off</strong> — a high <span className="font-mono">decay</span>{' '}
        remembers further back but still blurs the past together, so it can never pull out one exact old token the way
        the full cache can.
      </>
    ),
    footnote: (
      <>
        Illustrative — a scalar cartoon. GatedDeltaNet&apos;s real state is a matrix per head — shape{' '}
        <span className="font-mono">[16, 128, 128]</span> (value-heads × value-dim × key-dim) for the 0.8B model — and
        its real update is a gated <em>delta rule</em>,{' '}
        <span className="font-mono">S ← g·S + β·(v − (g·S)·k)·kᵀ</span>: it decays the old state by{' '}
        <span className="font-mono">g</span> and writes a β-weighted <em>correction</em> toward the new value, not a
        plain outer product. But the fixed-size-vs-growing contrast is exactly right.
      </>
    ),
  },
  zh: {
    title: '追加 vs 原地更新',
    subtitle: (
      <>
        同一条 token 流，两种记忆：<span className="font-mono">缓存增长</span> vs{' '}
        <span className="font-mono">S ← decay·S + u</span>
      </>
    ),
    streamComplete: '流已结束',
    feedNext: '喂入下一个 token →',
    pause: '暂停',
    replay: '重播',
    run: '运行',
    reset: '重置',
    tokensCounter: (fed: number, n: number) => `${fed} / ${n} token`,
    fullPanelLabel: '全量注意力（KV 缓存）',
    appendsEvery: '每个 token 都追加一项',
    emptyCache: '缓存为空',
    memoryEq: '内存 = ',
    unitWord: (_n: number) => ' 个单元',
    growsNote: '（随 N 增长）',
    linearPanelLabel: '线性状态（GatedDeltaNet）',
    updatesOneSlot: '原地更新一个槽位',
    oneFixedSlot: '一个固定槽位',
    shareTitle: (text: string, pct: string, age: number) => `${text}：占 S 的 ${pct}%（decay^${age}）`,
    slotZero: '只有一个槽位，当前为零',
    fadeNote: '最新的 token 最亮，更早的贡献按 decayᵃᵍᵉ 衰减变淡',
    unitOne: ' 个单元',
    constantNote: '（常数）',
    decayDesc: (horizon: number) =>
      `——控制状态还能记住多久以前（≈ 最近 ${horizon} 个 token 仍持有 ≥10% 的权重）`,
    forgetsFastTitle: '遗忘快——记忆视界短',
    forgetsFast: (v: string) => `${v}（遗忘快）`,
    longerMemoryTitle: '记得更久，但仍有损',
    longerMemory: (v: string) => `${v}（记得更久）`,
    tokensFedTile: '已喂入 token',
    fullAttentionTile: '全量注意力',
    growsTile: '（增长）',
    linearStateTile: '线性状态',
    oneUnit: '1 个单元',
    takeawayEnd: (n: number) =>
      `${n} 个 token 之后：全量注意力已存下 ${n} 条记录（而且还会继续增长）；运行状态仍然只有一个槽位。`,
    takeawayStart: '喂入一个 token 开始。注意左边每一步都多出一个格子，而右边只是原地更新同一个槽位。',
    takeawayProgress: (fed: number) =>
      `已喂入 ${fed} 个 token：全量注意力现在存着 ${fed} 条记录；运行状态仍然恰好是一个槽位。`,
    liveSummary: (fed: number, n: number, s: string, lastText: string | undefined) =>
      `已喂入 token：${fed} / ${n}。` +
      `全量注意力内存：已存 ${fed} 条记录（每个 token 增加一条）。` +
      `线性运行状态：1 个固定槽位（常数），当前值 S = ${s}` +
      (lastText ? `，刚写入“${lastText}”。` : '。'),
    rnnGloss: (
      <>
        那个单一的运行槽位<strong>就是循环神经网络的隐藏状态</strong>：一块固定大小的记忆，每一步向前滚动（
        <span className="font-mono">S ← decay·S + u</span>
        ），就像累计总和或滑动平均。全量注意力则保留<em>每一个</em>过去的
        token，所以它的内存无限增长。这就是<strong>记忆 vs 回忆的取舍</strong>——更高的{' '}
        <span className="font-mono">decay</span>{' '}
        记得更久，但仍会把过去糊在一起，永远无法像完整缓存那样精确取出某一个旧 token。
      </>
    ),
    footnote: (
      <>
        示意图——一个标量卡通。GatedDeltaNet 的真实状态是每个头一个矩阵——0.8B 模型的形状为{' '}
        <span className="font-mono">[16, 128, 128]</span>（value 头数 × value 维度 × key
        维度）——它的真实更新是带门控的<em>delta 规则</em>：
        <span className="font-mono">S ← g·S + β·(v − (g·S)·k)·kᵀ</span>：先用{' '}
        <span className="font-mono">g</span> 衰减旧状态，再向新值写入一个按 β 加权的<em>修正量</em>
        ，而不是简单的外积。但“固定大小 vs 持续增长”的对比是完全正确的。
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

// Decayed share of S contributed by the token fed at step `fedAt` (1-indexed),
// evaluated after `fed` tokens total: u · decay^(fed − fedAt).
function decayedShare(u: number, fedAt: number, fed: number, decay: number): number {
  const age = fed - fedAt;
  return age < 0 ? 0 : u * decay ** age;
}

export function RunningStateVsCache() {
  const copy = COPY[useLocale()];
  const reducedMotion = usePrefersReducedMotion();

  // How many tokens have been fed so far (0..N). Both panels read off this one
  // shared counter — the whole point is that the same stream drives both.
  const [fed, setFed] = React.useState(0);
  const [decay, setDecay] = React.useState(DECAY_DEFAULT);
  const [running, setRunning] = React.useState(false);

  const feedNext = React.useCallback(() => {
    // Functional update so rapid clicks / interval ticks never read a stale
    // count, and so we cap at N.
    setFed((f) => Math.min(N, f + 1));
  }, []);

  const reset = React.useCallback(() => {
    setRunning(false);
    setFed(0);
  }, []);

  // Auto-play: advance one token per tick, self-terminate at the stream end,
  // and tear down on unmount. Disabled entirely under reduced motion.
  React.useEffect(() => {
    if (reducedMotion || !running) return undefined;
    const id = window.setInterval(() => {
      setFed((f) => {
        if (f + 1 >= N) {
          setRunning(false);
          return N;
        }
        return f + 1;
      });
    }, RUN_MS);
    return () => window.clearInterval(id);
  }, [reducedMotion, running]);

  const atEnd = fed >= N;

  // Linear running state S = Σ u_i · decay^(fed − i) over the fed tokens.
  const shares = STREAM.map((t, i) => decayedShare(t.u, i + 1, fed, decay)).slice(0, fed);
  const S = shares.reduce((a, b) => a + b, 0);

  // The most recently fed token, for the "just wrote" highlight + readout.
  const lastTok = fed > 0 ? STREAM[fed - 1] : undefined;

  // Effective memory horizon: how many recent tokens still hold ≥10% of the
  // freshest token's weight. A purely illustrative read on "how far back the
  // state still remembers" at the current decay.
  const horizon = (() => {
    let h = 0;
    while (h < 64 && decay ** h >= 0.1) h += 1;
    return h;
  })();

  // Plain-language takeaway, recomputed from the live counts.
  const takeaway = atEnd ? copy.takeawayEnd(N) : fed === 0 ? copy.takeawayStart : copy.takeawayProgress(fed);

  // aria-live summary that conveys the growing-vs-updating structure for SR
  // users (the panels are DOM boxes, not a single SVG).
  const liveSummary = copy.liveSummary(fed, N, S.toFixed(2), lastTok ? lastTok.text.trim() : undefined);

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <div className="text-[11px] text-muted-foreground">{copy.subtitle}</div>
      </div>

      {/* ---- Token stream + transport controls ---- */}
      <div className="space-y-2 rounded-md border border-border/60 bg-muted/20 p-2">
        <div className="flex flex-wrap items-center gap-1.5" aria-hidden="true">
          {STREAM.map((t, i) => {
            const isFed = i < fed;
            const isNext = i === fed;
            return (
              <span
                key={`stream-${i}`}
                className={cn(
                  'rounded px-1.5 py-0.5 font-mono text-[11px] transition-colors',
                  isFed
                    ? 'bg-primary/15 text-foreground/90'
                    : isNext
                      ? 'border border-dashed border-primary/60 text-foreground/70'
                      : 'text-muted-foreground/50',
                )}
              >
                {t.text}
              </span>
            );
          })}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={feedNext}
            disabled={atEnd || running}
            className={cn(
              'rounded border border-border bg-muted/40 px-2.5 py-1 text-xs font-medium text-foreground/90',
              'transition-colors hover:bg-muted/70 disabled:opacity-40',
            )}
          >
            {atEnd ? copy.streamComplete : copy.feedNext}
          </button>
          {!reducedMotion ? (
            <button
              type="button"
              onClick={() => {
                if (atEnd) {
                  setFed(0);
                  setRunning(true);
                } else {
                  setRunning((p) => !p);
                }
              }}
              aria-pressed={running}
              className={cn(
                'rounded border border-border bg-muted/40 px-2.5 py-1 text-xs font-medium text-foreground/90',
                'transition-colors hover:bg-muted/70',
              )}
            >
              {running ? copy.pause : atEnd ? copy.replay : copy.run}
            </button>
          ) : null}
          <button
            type="button"
            onClick={reset}
            className={cn(
              'rounded border border-border px-2.5 py-1 text-xs font-medium text-muted-foreground',
              'transition-colors hover:text-foreground',
            )}
          >
            {copy.reset}
          </button>
          <span className="ml-auto font-mono text-[11px] text-muted-foreground">{copy.tokensCounter(fed, N)}</span>
        </div>
      </div>

      {/* ---- The two panels ---- */}
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {/* Full attention — appends, grows. */}
        <div className="space-y-2 rounded-md border border-border/60 bg-muted/10 p-2">
          <div className="flex items-baseline justify-between">
            <span className="rounded bg-amber-500/15 px-2 py-0.5 text-[11px] font-medium text-amber-700 dark:text-amber-300">
              {copy.fullPanelLabel}
            </span>
            <span className="text-[11px] text-muted-foreground">{copy.appendsEvery}</span>
          </div>

          <div className="flex min-h-[88px] flex-wrap content-start gap-1.5">
            {fed === 0 ? (
              <span className="text-[11px] italic text-muted-foreground/70">{copy.emptyCache}</span>
            ) : (
              STREAM.slice(0, fed).map((t, i) => {
                const isNewest = i === fed - 1;
                return (
                  <div
                    key={`kv-${i}`}
                    className={cn(
                      'flex w-[56px] flex-col items-center rounded border px-1 py-1 text-center',
                      isNewest ? 'border-amber-500/70 bg-amber-500/20' : 'border-border/60 bg-background',
                    )}
                  >
                    <span className="font-mono text-[10px] leading-tight text-foreground/90">{t.text.trim()}</span>
                    <span className="font-mono text-[9px] text-muted-foreground">
                      K,V<sub>{i + 1}</sub>
                    </span>
                  </div>
                );
              })
            )}
          </div>

          <div className="rounded border border-amber-500/30 bg-amber-500/5 px-2 py-1 text-[11px]">
            {copy.memoryEq}
            <span className="font-mono text-foreground/90">{fed}</span>
            {copy.unitWord(fed)} <span className="text-amber-700 dark:text-amber-300">{copy.growsNote}</span>
          </div>
        </div>

        {/* Linear running state — updates in place, constant size. */}
        <div className="space-y-2 rounded-md border border-border/60 bg-muted/10 p-2">
          <div className="flex items-baseline justify-between">
            <span className="rounded bg-violet-500/15 px-2 py-0.5 text-[11px] font-medium text-violet-700 dark:text-violet-300">
              {copy.linearPanelLabel}
            </span>
            <span className="text-[11px] text-muted-foreground">{copy.updatesOneSlot}</span>
          </div>

          <div className="flex min-h-[88px] flex-col justify-start gap-1.5">
            {/* The single fixed-size slot. Its contents change; the box never
                multiplies. The stacked fill shows each past token's decayed
                share u_i · decay^age, so older tokens visibly fade. */}
            <div className="flex items-baseline justify-between">
              <span className="font-mono text-[11px] text-foreground/85">S = {S.toFixed(2)}</span>
              <span className="text-[10px] text-muted-foreground">{copy.oneFixedSlot}</span>
            </div>
            <div
              className="flex h-12 w-full overflow-hidden rounded border border-violet-500/70 bg-muted/30"
              aria-hidden="true"
            >
              {fed === 0
                ? null
                : shares.map((share, i) => {
                    const frac = S > 0 ? share / S : 0;
                    const age = fed - (i + 1);
                    const isNewest = i === fed - 1;
                    // Older contributions fade toward the background; the freshest
                    // sits at full strength. decay^age also literally scales width
                    // (via share), so old tokens get both narrower and dimmer.
                    const opacity = 0.3 + 0.7 * decay ** age;
                    return (
                      <div
                        key={`share-${i}`}
                        className={cn('h-full', isNewest && 'ring-1 ring-inset ring-violet-200/80')}
                        style={{
                          width: `${(frac * 100).toFixed(2)}%`,
                          backgroundColor: 'var(--color-violet-500, #8b5cf6)',
                          opacity,
                        }}
                        title={copy.shareTitle(STREAM[i]?.text.trim() ?? '', (frac * 100).toFixed(0), age)}
                      />
                    );
                  })}
            </div>
            <div className="text-[10px] text-muted-foreground">{fed === 0 ? copy.slotZero : copy.fadeNote}</div>
          </div>

          <div className="rounded border border-violet-500/30 bg-violet-500/5 px-2 py-1 text-[11px]">
            {copy.memoryEq}
            <span className="font-mono text-foreground/90">1</span>
            {copy.unitOne} <span className="text-violet-700 dark:text-violet-300">{copy.constantNote}</span>
          </div>
        </div>
      </div>

      {/* ---- Decay slider ---- */}
      <div className="space-y-1">
        <label htmlFor="rsvc-decay" className="block text-xs text-muted-foreground">
          decay <span className="font-mono text-foreground/85">{decay.toFixed(2)}</span>{' '}
          <span className="text-muted-foreground/80">{copy.decayDesc(horizon)}</span>
        </label>
        <input
          id="rsvc-decay"
          type="range"
          min={DECAY_MIN}
          max={DECAY_MAX}
          step={0.01}
          value={decay}
          onChange={(e) => setDecay(Number(e.target.value))}
          className="w-full accent-primary"
          aria-valuetext={`decay ${decay.toFixed(2)}`}
        />
        <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
          <span title={copy.forgetsFastTitle}>{copy.forgetsFast(DECAY_MIN.toFixed(2))}</span>
          <span title={copy.longerMemoryTitle}>{copy.longerMemory(DECAY_MAX.toFixed(2))}</span>
        </div>
      </div>

      {/* ---- Readouts ---- */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3" aria-live="polite">
        <div className="rounded-md border border-border/60 bg-muted/20 p-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.tokensFedTile}</div>
          <div className="font-mono text-sm text-foreground/90">
            {fed} / {N}
          </div>
        </div>
        <div className="rounded-md border border-border/60 bg-muted/20 p-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.fullAttentionTile}</div>
          <div className="font-mono text-sm text-foreground/90">
            {fed}
            {copy.unitWord(fed)} <span className="text-[10px] text-muted-foreground">{copy.growsTile}</span>
          </div>
        </div>
        <div className="rounded-md border border-border/60 bg-muted/20 p-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.linearStateTile}</div>
          <div className="font-mono text-sm text-foreground/90">
            {copy.oneUnit} <span className="text-[10px] text-muted-foreground">{copy.constantNote}</span>
          </div>
        </div>
      </div>

      {/* Screen-reader summary of the live structure (panels are DOM boxes). */}
      <p className="sr-only" aria-live="polite">
        {liveSummary}
      </p>

      <div className="text-[12px] font-medium text-foreground/90" aria-live="polite">
        {takeaway}
      </div>

      {/* ---- The RNN gloss caption ---- */}
      <p className="text-[12px] leading-relaxed text-foreground/85">{copy.rnnGloss}</p>

      <p className="text-[10px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}
