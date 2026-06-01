import * as React from 'react';

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
 * and generation begins right after a trailing `<|im_start|>assistant\n`.
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

function ToggleButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={[
        'rounded px-2.5 py-1 text-xs font-medium transition-colors',
        active ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground',
      ].join(' ')}
    >
      {children}
    </button>
  );
}

export function ChatTemplateExplorer() {
  const [mode, setMode] = React.useState<Mode>('raw');

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">The chat template</div>
        <div className="inline-flex rounded-md border border-border p-0.5">
          <ToggleButton active={mode === 'raw'} onClick={() => setMode('raw')}>
            Raw text the model sees
          </ToggleButton>
          <ToggleButton active={mode === 'chat'} onClick={() => setMode('chat')}>
            Chat view
          </ToggleButton>
        </div>
      </div>

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
            <span className="text-primary">▮ the model generates from here</span>
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
        with a helpful answer instead of, say, inventing a third user question. This is the essential shape — to keep it
        readable we leave out a couple of markers the real Qwen3.5 template also threads in (including a reasoning
        block), which the app fills in for you.
      </p>
    </div>
  );
}
