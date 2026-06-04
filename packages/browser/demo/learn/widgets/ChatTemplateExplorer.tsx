import * as React from 'react';

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

function Special({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded border border-dashed border-border px-1 font-mono text-[10px] text-muted-foreground">
      {children}
    </span>
  );
}

export function ChatTemplateExplorer() {
  const [mode, setMode] = React.useState<Mode>('raw');
  // Mirrors the live chat "think" pill: when thinking is ON the template opens
  // an empty <think>\n for the model to reason into; when OFF it injects a
  // pre-closed <think>\n\n</think>\n\n so the model skips straight to the answer.
  const [thinking, setThinking] = React.useState(true);

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">The chat template</div>
        <SegmentedToggle
          value={mode}
          onChange={setMode}
          options={[
            { value: 'raw', label: 'Raw text the model sees' },
            { value: 'chat', label: 'Chat view' },
          ]}
        />
      </div>

      {mode === 'raw' ? (
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
          <span className="uppercase tracking-wider">Reasoning (the &ldquo;think&rdquo; pill)</span>
          <SegmentedToggle
            value={thinking ? 'on' : 'off'}
            onChange={(v) => setThinking(v === 'on')}
            ariaLabel="Reasoning toggle"
            options={[
              { value: 'on', label: 'thinking on' },
              { value: 'off', label: 'thinking off' },
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
                <span className="text-primary">▮ the model reasons here, then closes </span>
                <Special>{'</think>'}</Special>
                <span className="text-primary"> and answers</span>
              </>
            ) : (
              <>
                <Special>{'<think>'}</Special>
                {'\n\n'}
                <Special>{'</think>'}</Special>
                {'\n\n'}
                <span className="text-primary">▮ the model generates the answer from here</span>
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

      <p className="text-[12px] text-foreground/85">
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
      </p>
    </div>
  );
}
