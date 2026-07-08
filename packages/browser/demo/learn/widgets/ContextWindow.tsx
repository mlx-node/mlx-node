import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Chapter 16 (architecture) — the finite context window. A slider sets how long
 * the conversation has grown; the strip below shows the conversation as discrete
 * blocks of tokens. The window is a fixed-size frame holding only the most
 * recent WINDOW tokens — drag past it and the oldest blocks visibly DROP out of
 * the frame into the "forgotten" zone (dimmed, sunk, struck through).
 */

const WINDOW = 262_144; // Qwen3.5-0.8B max_position_embeddings
const SLIDER_MAX = WINDOW * 2;

// The conversation drawn as discrete blocks so "dropping the oldest" is
// visible: each chip stands for BLOCK_TOKENS tokens. 32 chips span SLIDER_MAX;
// the window holds exactly the newest 16 of them once the conversation is full.
const N_BLOCKS = 32;
const BLOCK_TOKENS = SLIDER_MAX / N_BLOCKS; // 16,384 tokens per chip

const fmt = (n: number) => Math.round(n).toLocaleString();

type ChipState = 'future' | 'in-window' | 'forgotten';

const COPY = {
  en: {
    header: 'The context window',
    windowReadout: (w: string) => `window = ${w} tokens`,
    soFar: 'Conversation so far: ',
    tokens: (n: string) => `${n} tokens`,
    sliderAria: 'Conversation length in tokens',
    stripAria: (nBlocks: number, blockTokens: string, forgotten: string, inWindow: string) =>
      `The conversation drawn as ${nBlocks} blocks of ${blockTokens} tokens. ${forgotten} tokens have dropped out of the fixed-size window into the forgotten zone; ${inWindow} tokens are still inside it.`,
    forgottenZone: '↓ forgotten zone — oldest tokens drop out here',
    windowZone: (w: string) => `window: newest ${w} tokens →`,
    chipForgotten: 'Dropped out of the window — gone',
    chipInWindow: 'In the window',
    chipFuture: 'Not spoken yet',
    inContext: 'In context: ',
    forgotten: 'Forgotten: ',
    body: (w: string) => (
      <>
        The model can only attend to tokens inside the window — it's a hard ceiling, not a sliding memory. Drag the
        slider past <span className="font-mono">{w}</span> and watch the oldest blocks drop out of the frame: to make
        room for each new token, one old token has to go, and anything dropped is gone unless it's re-supplied in a
        later prompt. There is no long-term store; the window is the model's <em>entire</em> working memory.
      </>
    ),
  },
  zh: {
    header: '上下文窗口',
    windowReadout: (w: string) => `窗口 = ${w} token`,
    soFar: '到目前为止的对话：',
    tokens: (n: string) => `${n} token`,
    sliderAria: '对话长度（token 数）',
    stripAria: (nBlocks: number, blockTokens: string, forgotten: string, inWindow: string) =>
      `把对话画成 ${nBlocks} 个区块，每块 ${blockTokens} 个 token。已有 ${forgotten} 个 token 掉出固定大小的窗口、落入遗忘区；窗口内还剩 ${inWindow} 个 token。`,
    forgottenZone: '↓ 遗忘区——最旧的 token 从这里掉出去',
    windowZone: (w: string) => `窗口：最新的 ${w} 个 token →`,
    chipForgotten: '已掉出窗口——再也回不来',
    chipInWindow: '在窗口内',
    chipFuture: '尚未说出',
    inContext: '在上下文中：',
    forgotten: '已遗忘：',
    body: (w: string) => (
      <>
        模型只能关注窗口之内的 token——这是一道硬上限，不是会滑动的记忆。把滑块拖过{' '}
        <span className="font-mono">{w}</span>，看看最旧的区块如何掉出画框：每进来一个新 token，就得有一个旧 token
        让位，而掉出去的内容除非在之后的提示词里重新提供，否则就永远消失了。这里没有长期存储；窗口就是模型
        <em>全部</em>的工作记忆。
      </>
    ),
  },
} as const;

function chipState(i: number, len: number): ChipState {
  const start = i * BLOCK_TOKENS;
  const end = start + BLOCK_TOKENS;
  if (len <= start) return 'future'; // not spoken yet
  // The window keeps the newest WINDOW tokens: indices >= len - WINDOW.
  if (end <= len - WINDOW) return 'forgotten';
  return 'in-window';
}

export function ContextWindow() {
  const copy = COPY[useLocale()];
  const [len, setLen] = React.useState(120_000);

  const inWindow = Math.min(len, WINDOW);
  const forgotten = Math.max(0, len - WINDOW);

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <div className="font-mono text-[11px] text-muted-foreground">{copy.windowReadout(fmt(WINDOW))}</div>
      </div>

      <label className="block text-[11px] text-muted-foreground">
        {copy.soFar}
        <span className="font-mono text-foreground/90">{copy.tokens(fmt(len))}</span>
        <input
          type="range"
          min={0}
          max={SLIDER_MAX}
          step={1000}
          value={len}
          onChange={(e) => setLen(Number(e.target.value))}
          className="mt-1 w-full accent-primary"
          aria-label={copy.sliderAria}
        />
      </label>

      <div
        className="space-y-1"
        role="img"
        aria-label={copy.stripAria(N_BLOCKS, fmt(BLOCK_TOKENS), fmt(forgotten), fmt(inWindow))}
      >
        {/* zone labels — the "forgotten" one lights up once anything drops */}
        <div className="flex justify-between text-[10px]">
          <span className={forgotten > 0 ? 'font-medium text-foreground/80' : 'text-muted-foreground/50'}>
            {copy.forgottenZone}
          </span>
          <span className="text-muted-foreground">{copy.windowZone(fmt(WINDOW))}</span>
        </div>
        <div className="flex h-9 w-full items-start gap-px">
          {Array.from({ length: N_BLOCKS }, (_, i) => {
            const state = chipState(i, len);
            return (
              <div
                key={i}
                className={[
                  'flex-1 rounded-sm border transition-all duration-200',
                  state === 'in-window' ? 'h-7 border-primary/50 bg-primary/30' : '',
                  state === 'forgotten' ? 'h-5 translate-y-3 border-border/40 bg-muted-foreground/15 opacity-50' : '',
                  state === 'future' ? 'h-7 border-dashed border-border/40 bg-transparent' : '',
                ].join(' ')}
                title={
                  state === 'forgotten'
                    ? copy.chipForgotten
                    : state === 'in-window'
                      ? copy.chipInWindow
                      : copy.chipFuture
                }
              />
            );
          })}
        </div>
      </div>

      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
        <span className="text-foreground/85">
          <span className="mr-1 inline-block h-2 w-2 rounded-sm bg-primary/40 align-middle" />
          {copy.inContext}
          <span className="font-mono">{fmt(inWindow)}</span>
        </span>
        <span className={forgotten > 0 ? 'text-foreground/85' : 'text-muted-foreground'}>
          <span className="mr-1 inline-block h-2 w-2 rounded-sm bg-muted-foreground/40 align-middle" />
          {copy.forgotten}
          <span className="font-mono">{fmt(forgotten)}</span>
        </span>
      </div>

      <p className="text-[12px] text-foreground/85">{copy.body(fmt(WINDOW))}</p>
    </div>
  );
}
