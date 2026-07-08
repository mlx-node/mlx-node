import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Chapter 15 (post-training) supplement — the chapter's hook, made visible.
 *
 * Two lanes get the SAME prompt ("What is 2 + 2?"). The base-model lane
 * autocompletes it the way the open web would continue that string — with more
 * homework. The after-SFT lane answers it. Typewriter reveal, deterministic
 * and scripted: NO worker, NO model, NO WASM — the real base model isn't
 * loaded here (and the shipped checkpoint is already instruction-tuned).
 *
 * Animation model: a single tick counter advanced by setInterval inside a
 * guarded useEffect; each tick reveals one character in both lanes. Honors
 * prefers-reduced-motion (both lanes shown finished, no autoplay).
 */

const PROMPT = 'What is 2 + 2?';

// Scripted continuations. The base lane reads like a scraped worksheet — a
// plausible web continuation of the prompt string, not an answer.
const BASE_CONTINUATION = ' What is 3 + 5? What is 9 − 4? Worksheet 4, problem 2…';
const SFT_CONTINUATION = ' 2 + 2 = 4.';

const TICK_MS = 65;
const MAX_TICKS = Math.max(BASE_CONTINUATION.length, SFT_CONTINUATION.length);

// Per-locale copy. The prompt and the two scripted continuations are literal
// model-style text and stay English in both locales (the zh chapter quotes
// "What is 2 + 2?" in English too).
const COPY = {
  en: {
    header: 'Same prompt, before and after instruction tuning',
    replay: '↻ Replay',
    pause: '❚❚ Pause',
    play: '▶ Play',
    baseTitle: 'Base model (autocomplete)',
    baseSubtitle: 'continues the text — does not answer it',
    sftTitle: 'After SFT (assistant)',
    sftSubtitle: 'treats the question as something to answer',
    body: (
      <>
        Both lanes run the <em>same architecture</em> on the same prompt. The base model just predicts what plausibly
        follows that string on the web — and on the web, one homework question is usually followed by more homework.
        Instruction tuning shifts what counts as &ldquo;plausible&rdquo;: after seeing curated (question, answer) pairs,
        the likeliest continuation of a question <em>is</em> its answer.
      </>
    ),
    footnote: (
      <>
        Scripted illustration — the real base model isn&apos;t loaded here (the checkpoint this site ships is already
        instruction-tuned).
      </>
    ),
  },
  zh: {
    header: '同一个提示词，指令微调前后',
    replay: '↻ 重播',
    pause: '❚❚ 暂停',
    play: '▶ 播放',
    baseTitle: '基础模型（自动补全）',
    baseSubtitle: '接着写下去——并不回答',
    sftTitle: '经过 SFT（助手）',
    sftSubtitle: '把问题当成要回答的东西',
    body: (
      <>
        两条通道在同一个提示词上运行<em>同一套架构</em>。基础模型只是预测网络上最可能跟在这串文字后面的内容——
        而在网络上，一道家庭作业题后面通常跟着更多作业。指令微调改变了什么算“合理”：在见过精心整理的（问题，答案）
        对之后，一个问题最可能的续写<em>就是</em>它的答案。
      </>
    ),
    footnote: <>脚本化示意——这里并没有真的加载基础模型（本站附带的 checkpoint 已经过指令微调）。</>,
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

function Lane({
  title,
  subtitle,
  text,
  revealed,
  accent,
}: {
  title: string;
  subtitle: string;
  text: string;
  revealed: number;
  accent: boolean;
}) {
  const shown = text.slice(0, revealed);
  const laneDone = revealed >= text.length;
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[10px] uppercase tracking-wider text-muted-foreground">{title}</span>
        <span className="text-[10px] text-muted-foreground">{subtitle}</span>
      </div>
      <div
        className={[
          'min-h-12 rounded-md border p-2 text-[12px] leading-relaxed',
          accent ? 'border-primary/50 bg-primary/5' : 'border-border/60 bg-muted/30',
        ].join(' ')}
      >
        {/* the shared prompt chip */}
        <span className="mr-1 rounded border border-border bg-card px-1.5 py-0.5 font-mono text-[11px] text-foreground/90">
          {PROMPT}
        </span>
        <span className={['font-mono text-[11px]', accent ? 'text-foreground' : 'text-foreground/80'].join(' ')}>
          {shown}
        </span>
        {!laneDone ? (
          <span className="ml-0.5 font-mono text-[11px] text-muted-foreground" aria-hidden>
            ▮
          </span>
        ) : null}
      </div>
    </div>
  );
}

export function BaseVsInstructDiagram() {
  const copy = COPY[useLocale()];
  const reducedMotion = usePrefersReducedMotion();
  const [tick, setTick] = React.useState(reducedMotion ? MAX_TICKS : 0);
  const [playing, setPlaying] = React.useState(!reducedMotion);

  const done = tick >= MAX_TICKS;

  React.useEffect(() => {
    if (!playing || reducedMotion) return;
    const t = window.setInterval(() => {
      setTick((x) => {
        if (x + 1 >= MAX_TICKS) {
          setPlaying(false);
          return MAX_TICKS;
        }
        return x + 1;
      });
    }, TICK_MS);
    return () => window.clearInterval(t);
  }, [playing, reducedMotion]);

  const replay = () => {
    setTick(0);
    setPlaying(true);
  };

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        {!reducedMotion ? (
          <div className="flex items-center gap-1">
            {done ? (
              <button
                type="button"
                onClick={replay}
                className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              >
                {copy.replay}
              </button>
            ) : (
              <button
                type="button"
                onClick={() => setPlaying((p) => !p)}
                aria-pressed={playing}
                className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              >
                {playing ? copy.pause : copy.play}
              </button>
            )}
          </div>
        ) : null}
      </div>

      <Lane
        title={copy.baseTitle}
        subtitle={copy.baseSubtitle}
        text={BASE_CONTINUATION}
        revealed={Math.min(tick, BASE_CONTINUATION.length)}
        accent={false}
      />

      <Lane
        title={copy.sftTitle}
        subtitle={copy.sftSubtitle}
        text={SFT_CONTINUATION}
        revealed={Math.min(tick, SFT_CONTINUATION.length)}
        accent
      />

      <p className="text-[12px] text-foreground/85">{copy.body}</p>

      <p className="text-[10px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}
