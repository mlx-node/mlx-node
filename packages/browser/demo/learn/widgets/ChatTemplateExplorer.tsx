import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import { SegmentedToggle } from '../scaffolding/SegmentedToggle';

/**
 * Chapter 15 (post-training) — the chat template, made literal.
 *
 * Static (no worker / no tokenizer call): a beginner's whole mental model of an
 * LLM is ChatGPT, so the key reveal is that the "assistant" is the same
 * next-token predictor fed a specially formatted string. Toggle between the raw
 * text the model actually sees (special-token markers and all) and the rendered
 * chat bubbles a user sees. The special tokens use the same dashed-border
 * styling as the Tokenization chapter.
 *
 * The format mirrors Qwen's real ChatML template (the app applies it via
 * `applyChatTemplate`): each turn is
 *   <|im_start|>{role}\n{content}<|im_end|>\n
 * and generation begins right after a trailing `<|im_start|>assistant\n`. The
 * real Qwen3.5 template ALWAYS injects a reasoning block right after that
 * assistant marker — `<think>\n` when thinking is on, or `<think>\n\n</think>\n\n`
 * when thinking is off — before the model's first generated token. A "show
 * reasoning markers" sub-toggle (tied to the live chat "think" pill) surfaces it.
 */

type Role = 'system' | 'user' | 'assistant';
type Turn = { role: Role; content: string };

const TURNS: Turn[] = [
  { role: 'system', content: 'You are a helpful assistant. Be concise.' },
  { role: 'user', content: 'What is 2 + 2?' },
  { role: 'assistant', content: '4' },
];

const ROLE_BG: Record<Role, string> = {
  system: 'oklch(0.72 0.04 280 / 0.18)',
  user: 'oklch(0.7 0.13 220 / 0.16)',
  assistant: 'oklch(0.72 0.15 150 / 0.16)',
};

type Mode = 'raw' | 'chat';

// Per-locale copy. The template text itself (turn contents, role names, and
// the special tokens <|im_start|> / <|im_end|> / <think>) is literal model I/O
// and stays byte-identical in both locales — only the labels around it change.
const COPY = {
  en: {
    header: 'The chat template',
    rawLabel: 'Raw text the model sees',
    chatLabel: 'Chat view',
    reasoningLabel: <>Reasoning (the &ldquo;think&rdquo; pill)</>,
    reasoningAria: 'Reasoning toggle',
    thinkingOn: 'thinking on',
    thinkingOff: 'thinking off',
    reasonsHere: '▮ the model reasons here, then closes ',
    andAnswers: ' and answers',
    generatesHere: '▮ the model generates the answer from here',
    body: (
      <>
        The "assistant" is the <em>same</em> next-token predictor from every other chapter — it's just fed a string
        wrapped in role markers. The special tokens <span className="font-mono">{'<|im_start|>'}</span> /{' '}
        <span className="font-mono">{'<|im_end|>'}</span> tell it whose turn it is and where a turn stops. Instruction
        tuning is what teaches it to continue a trailing <span className="font-mono">{'<|im_start|>assistant'}</span>{' '}
        with a helpful answer instead of, say, inventing a third user question. The real Qwen3.5 template always injects
        a <span className="font-mono">{'<think>'}</span> reasoning block right after that marker — open
        (<span className="font-mono">{'<think>\\n'}</span>) when the live chat&apos;s <em>think</em> pill is on, or
        pre-closed (<span className="font-mono">{'<think>\\n\\n</think>\\n\\n'}</span>) when it&apos;s off — which is the
        sub-toggle above. (A couple of rarer markers, like the tool-calling block below, are still elided for
        readability; the app fills those in for you.)
      </>
    ),
  },
  zh: {
    header: '对话模板',
    rawLabel: '模型实际看到的原始文本',
    chatLabel: '对话视图',
    reasoningLabel: <>推理（“think” 开关）</>,
    reasoningAria: '推理开关',
    thinkingOn: '思考开启',
    thinkingOff: '思考关闭',
    reasonsHere: '▮ 模型在这里推理，然后闭合 ',
    andAnswers: ' 并给出答案',
    generatesHere: '▮ 模型从这里开始生成答案',
    body: (
      <>
        这个“助手”和其他每一章里的下一个 token 预测器是<em>同一个</em>——它只是被喂进了一个用角色标记包裹起来的字符串。
        special token <span className="font-mono">{'<|im_start|>'}</span> /{' '}
        <span className="font-mono">{'<|im_end|>'}</span> 告诉它现在轮到谁说话、一轮在哪里结束。正是指令微调教会了它在结尾的{' '}
        <span className="font-mono">{'<|im_start|>assistant'}</span>{' '}
        之后接上一段有用的回答，而不是（比如）编出第三个用户问题。真实的 Qwen3.5 模板总会在那个标记之后立刻注入一个{' '}
        <span className="font-mono">{'<think>'}</span> 推理块——在线对话的 <em>think</em> 开关开启时是未闭合的（
        <span className="font-mono">{'<think>\\n'}</span>），关闭时是预先闭合的（
        <span className="font-mono">{'<think>\\n\\n</think>\\n\\n'}</span>）——也就是上面那个子开关。
        （少数更罕见的标记，比如下方的工具调用块，为了可读性仍被省略；应用会替你补全它们。）
      </>
    ),
  },
} as const;

function Special({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded border border-dashed border-border px-1 font-mono text-[10px] text-muted-foreground">
      {children}
    </span>
  );
}

export function ChatTemplateExplorer() {
  const copy = COPY[useLocale()];
  const [mode, setMode] = React.useState<Mode>('raw');
  // Mirrors the live chat "think" pill: when thinking is ON the template opens
  // an empty <think>\n for the model to reason into; when OFF it injects a
  // pre-closed <think>\n\n</think>\n\n so the model skips straight to the answer.
  const [thinking, setThinking] = React.useState(true);

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <SegmentedToggle
          value={mode}
          onChange={setMode}
          options={[
            { value: 'raw', label: copy.rawLabel },
            { value: 'chat', label: copy.chatLabel },
          ]}
        />
      </div>

      {mode === 'raw' ? (
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
          <span className="uppercase tracking-wider">{copy.reasoningLabel}</span>
          <SegmentedToggle
            value={thinking ? 'on' : 'off'}
            onChange={(v) => setThinking(v === 'on')}
            ariaLabel={copy.reasoningAria}
            options={[
              { value: 'on', label: copy.thinkingOn },
              { value: 'off', label: copy.thinkingOff },
            ]}
          />
        </div>
      ) : null}

      {mode === 'raw' ? (
        <pre className="overflow-x-auto rounded-md border border-border/60 bg-muted/30 p-3 font-mono text-[12px] leading-relaxed">
          {TURNS.map((t, i) => (
            <div key={i} className="whitespace-pre-wrap">
              <Special>{'<|im_start|>'}</Special>
              <span className="text-muted-foreground">{t.role}</span>
              {'\n'}
              <span className="text-foreground/90">{t.content}</span>
              <Special>{'<|im_end|>'}</Special>
              {'\n'}
            </div>
          ))}
          <div className="mt-1 whitespace-pre-wrap">
            <Special>{'<|im_start|>'}</Special>
            <span className="text-muted-foreground">assistant</span>
            {'\n'}
            {thinking ? (
              <>
                <Special>{'<think>'}</Special>
                {'\n'}
                <span className="text-primary">{copy.reasonsHere}</span>
                <Special>{'</think>'}</Special>
                <span className="text-primary">{copy.andAnswers}</span>
              </>
            ) : (
              <>
                <Special>{'<think>'}</Special>
                {'\n\n'}
                <Special>{'</think>'}</Special>
                {'\n\n'}
                <span className="text-primary">{copy.generatesHere}</span>
              </>
            )}
          </div>
        </pre>
      ) : (
        <div className="space-y-2">
          {TURNS.map((t, i) => (
            <div
              key={i}
              className="rounded-md border border-border/60 p-2"
              style={{ backgroundColor: ROLE_BG[t.role] }}
            >
              <div className="mb-0.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                {t.role}
              </div>
              <div className="text-[13px] text-foreground/90">{t.content}</div>
            </div>
          ))}
        </div>
      )}

      <p className="text-[12px] text-foreground/85">{copy.body}</p>
    </div>
  );
}
