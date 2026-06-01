import * as React from 'react';

import type { TokenInfo } from '../../../src/inspector-types';
import { Button } from '../../components/ui/button';
import { Textarea } from '../../components/ui/textarea';
import { tokenize } from '../../lib/tokenizer-client';
import { DemoCallout } from '../inspector/DemoCallout';
import { Prose } from '../Prose';
import { ChapterFrame } from '../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../scaffolding/learning-data';
import { BpeMergeWalkthrough } from '../widgets/BpeMergeWalkthrough';
import { MultilingualTokenMap } from '../widgets/MultilingualTokenMap';
import { TrickyCases } from '../widgets/TrickyCases';

/**
 * Chapter 1 — Tokenization.
 *
 * Plain TSX prose followed by `<TokenizerDemo />`, a live encode/decode
 * widget that calls the real Qwen3.5 tokenizer through the loaded worker.
 * When the model isn't loaded yet a yellow banner explains the fallback
 * and the chips render a heuristic whitespace split so the lesson is still
 * legible. The console.log convention mirrors chapter 4.
 */

const DEFAULT_PROMPT = 'The quick brown fox jumps over the lazy dog. Antidisestablishmentarianism. 你好';
const CODE_EXAMPLE = 'function fibonacci(n) { return n < 2 ? n : fibonacci(n-1) + fibonacci(n-2); }';
const MULTILINGUAL_EXAMPLE = 'Hello! 你好! ¡Hola! здравствуйте';

const TOKENIZE_DEBOUNCE_MS = 150;

// Soft pastel hues that cycle by token index. Tailwind classes are inlined
// here so the chip palette is a single source of truth.
const CHIP_PALETTE = [
  'bg-sky-100 dark:bg-sky-950/40 text-sky-900 dark:text-sky-100',
  'bg-amber-100 dark:bg-amber-950/40 text-amber-900 dark:text-amber-100',
  'bg-emerald-100 dark:bg-emerald-950/40 text-emerald-900 dark:text-emerald-100',
];

// Special-token text typically wraps `<|...|>` markers (e.g. `<|im_start|>`).
const SPECIAL_TOKEN_TEXT_RE = /<\|[^|]+\|>/;

/**
 * Scaffolding metadata for chapter 1 — drives the header, glossary,
 * takeaways, exercise, and quick-check rendered by `<ChapterFrame>`.
 * `chapterId` must match `CHAPTERS[0].id` in `learn/chapters.ts`.
 */
export const learning: ChapterLearningData = {
  chapterId: 'tokenization',
  objective: 'Explain why an LLM splits text into sub-word tokens instead of characters or words.',
  problem: 'Models need a finite vocabulary that balances sequence length against vocabulary size.',
  minutes: 7,
  glossary: [
    {
      term: 'token',
      definition: "The unit a transformer actually reads — an integer id pointing into the model's vocabulary.",
    },
    {
      term: 'BPE',
      definition:
        'Byte-Pair Encoding: start from bytes, repeatedly merge the most frequent adjacent pair until the vocabulary is the target size.',
    },
    {
      term: 'sub-word',
      definition: 'A token that is smaller than a word but larger than a character, e.g. "ization" or " the".',
    },
    {
      term: 'vocabulary',
      definition: 'The full set of tokens the model knows. Qwen3.5 ships about 248,320 entries.',
    },
    {
      term: 'special token',
      definition:
        'A reserved token like <|im_start|> that marks structure (chat turns, end-of-text) rather than literal text.',
    },
    {
      term: 'chars/token',
      definition:
        "Characters divided by tokens — a quick proxy for how efficiently a string fits the tokenizer's vocabulary.",
    },
  ],
  takeaways: [
    'Cost scales with tokens, not characters: billing, KV-cache, and latency are all per-token.',
    'A leading space is part of the token — "cat" and " cat" are usually different vocabulary entries.',
    'Different model families tokenize the same string differently, so token counts only compare within one tokenizer.',
  ],
  exercise: {
    prompt:
      'Using the tokenizer demo, find one sentence that gives you the highest chars/token ratio you can hit, and a different sentence that gives you the lowest.',
    answer:
      'Long English words (think "unbelievably", "internationalization") sit very high because BPE merges long common stems into one token. The low end is emoji, CJK punctuation, or unusual scripts that fall back to byte-level fragments — those routinely drop below 1 char/token.',
  },
  quiz: [
    {
      id: 'q1-why-subword',
      prompt: 'Why do modern LLMs use sub-word tokens instead of words?',
      options: [
        { id: 'a', label: 'Sub-words are always shorter than words.' },
        {
          id: 'b',
          label:
            'A word-level vocabulary would need to enumerate every name, typo, and rare word, which makes the embedding matrix impractically large.',
        },
        {
          id: 'c',
          label: 'Word-level tokenizers cannot represent punctuation, so they are unusable.',
        },
      ],
      correctId: 'b',
      explanation:
        'The vocabulary size has to be finite. Words mean millions of entries (long tail); characters mean very long sequences. Sub-words are the practical middle.',
    },
    {
      id: 'q2-chars-per-token',
      prompt: 'What does a higher chars/token ratio usually mean?',
      options: [
        {
          id: 'a',
          label: 'The string uses common patterns the tokenizer learned well, so each token covers more characters.',
        },
        {
          id: 'b',
          label: 'The model will hallucinate more on that input.',
        },
        {
          id: 'c',
          label: 'The text is in a language the model cannot understand.',
        },
      ],
      correctId: 'a',
      explanation:
        "Higher chars/token means the tokenizer's merges are an efficient fit for the text — typically prose in well-covered languages.",
    },
    {
      id: 'q3-leading-space',
      prompt: 'Are the tokens for "cat" at the start of a sentence and " cat" mid-sentence the same id?',
      options: [
        { id: 'a', label: 'Yes — BPE strips whitespace before encoding.' },
        {
          id: 'b',
          label:
            'No — BPE treats the leading space as part of the token, so they are usually two different vocabulary entries.',
        },
      ],
      correctId: 'b',
      explanation:
        'BPE operates on the raw byte stream including spaces, so a space-prefixed word is a distinct symbol from the bare word.',
    },
  ],
};

export function TokenizationChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>Tokenization: turning text into the model's vocabulary</h1>
        <p>
          Before a transformer can do anything with your sentence, the sentence has to be cut into pieces the model
          knows. Those pieces are called <strong>tokens</strong>, and the rule that decides where the cuts go is called
          the <em>tokenizer</em>. Tokens are not characters, and they are not words — they are something in between,
          chosen by an algorithm called <strong>Byte-Pair Encoding (BPE)</strong>. The widget on the right runs
          Qwen3.5's exact tokenizer in your browser. Type, watch the chips refresh.
        </p>

        <h2>Why sub-words?</h2>
        <p>
          Tokenizers sit in an awkward middle. If we made every <em>word</em> a token, the vocabulary would need
          millions of entries to cover the long tail of names, typos and made-up words — and the embedding matrix would
          balloon along with it. If we went the other way and made every <em>character</em> a token, the model would
          have to chew through sequences five-to-ten times longer than the same text in words, and every layer's cost is
          at least linear in sequence length.
        </p>
        <p>
          BPE finds a middle path: start from raw bytes (or characters), then repeatedly merge the most frequent
          adjacent pair into a new symbol until the vocabulary is the size you want. Common words like <code>the</code>{' '}
          end up as a single token; rarer words like <code>Antidisestablishmentarianism</code> decompose into shorter
          familiar pieces. The tokenizer doesn't know what a word means — it only knows what byte sequences are
          statistically frequent in the training corpus. Because that base alphabet is the 256 possible byte values, the
          tokenizer can represent literally <em>any</em> input — any Unicode character, emoji, or script, even ones it
          never saw in training — so there is no "unknown" or out-of-vocabulary token; unfamiliar text just breaks into
          more, smaller (byte) tokens.
        </p>

        <h2>What Qwen3.5 does specifically</h2>
        <p>
          Qwen3.5 ships with a BPE tokenizer of roughly <strong>248,320</strong> entries. A few quirks are worth
          knowing:
        </p>
        <ul>
          <li>
            <strong>Spaces live inside tokens.</strong> The BPE algorithm treats a space-prefixed word as a different
            symbol from the bare word, so <code>"cat"</code> at the start of a sentence and <code>" cat"</code>{' '}
            mid-sentence are usually two different tokens. The chips on the right mark a leading space with a small dot
            (<code>·</code>) so you can see this without it being invisible.
          </li>
          <li>
            <strong>Special tokens mark structure.</strong> Sequences like <code>&lt;|im_start|&gt;</code> and{' '}
            <code>&lt;|im_end|&gt;</code> aren't text the model generates — they're chat-turn boundaries used by
            Qwen3.5's chat template. The widget renders any token whose decoded text contains <code>&lt;|…|&gt;</code>{' '}
            with a dashed border so you can spot them.
          </li>
          <li>
            <strong>Multilingual coverage isn't uniform.</strong> Latin-script words are usually one or two tokens; CJK
            characters often land as one token each but unusual scripts can decompose into byte-level fragments. Try the
            multilingual example to see the difference.
          </li>
        </ul>

        <h2>Why this matters</h2>
        <p>
          Every downstream cost a language model has is paid <em>per token</em>: API billing, context-window limits,
          KV-cache memory, and the wall-clock latency of generation. A prompt that looks short to a human can be long to
          the model if it uses rare punctuation, code, or languages the tokenizer didn't see much of in training. The{' '}
          <strong>characters-per-token ratio</strong> in the stats footer is the simplest way to feel this: prose in
          well-supported languages usually lands near 4 chars/token, code closer to 2-3, and an emoji-heavy or
          rare-script string can drop below 1.
        </p>
        <p>
          Different model families use different tokenizers, and the same string can come out radically different
          lengths. That's why "tokens" is the lingua franca of LLM cost and capacity, not "characters" or "words."
        </p>

        <h2>Watching BPE build a token</h2>
        <p>
          The widget below walks through a stylised version of BPE on the input <code>"unbelievable"</code>. Each step
          merges one adjacent pair of fragments into a new symbol — the same operation BPE training runs many millions
          of times against a corpus to learn its merge order.
        </p>

        <BpeMergeWalkthrough />

        <h2>Tokenization quirks worth seeing</h2>
        <p>
          Most surprises beginners hit with LLM cost or behaviour trace back to one of these. Each row shows a small
          input alongside the tokens it usually decomposes into.
        </p>

        <TrickyCases />

        <MultilingualTokenMap />

        <p className="mt-6 text-muted-foreground">
          In the next chapter (<em>Embeddings</em>) we'll see how each of these token ids becomes a high-dimensional
          vector that the model can actually compute with — the bridge between integers and the continuous space
          attention operates in.
        </p>
      </Prose>
    </ChapterFrame>
  );
}

export type TokenizerDemoProps = {
  /** Ref to the shared MLX worker owned by the app shell. `null` while the
   *  model is not loaded — the demo shows a fallback banner and a heuristic
   *  whitespace split so the page is still legible. */
  workerRef: React.RefObject<Worker | null>;
  /** Ref to an AbortController bound to the current worker's lifetime. The
   *  app shell aborts this before calling `worker.terminate()` (model swap /
   *  reload kickoff) so an in-flight tokenize call rejects immediately
   *  instead of hanging for the full 60s timeout. */
  abortRef: React.RefObject<AbortController | null>;
};

type TokenizeStatus =
  | { kind: 'ok' }
  | { kind: 'fallback-no-worker' }
  | { kind: 'fallback-error'; error: string }
  | { kind: 'aborted' };

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

// When the worker isn't available we still want the chip strip to render
// something so the chapter doesn't look broken. This is a pure whitespace
// split — explicitly NOT BPE — and the banner above the chips tells the user
// what they're looking at.
function fallbackHeuristicTokens(prompt: string): TokenInfo[] {
  if (!prompt) return [];
  const pieces = prompt.match(/\s+|[^\s]+/g) ?? [];
  return pieces.map((text, i) => ({ id: -1 - i, text }));
}

// Map raw tokenizer text to display form: replace the BPE leading-space
// marker "Ġ" with a literal space (some tokenizer outputs use "Ġ", others
// already emit a literal space depending on decode flags), then render
// leading whitespace as "·" so the chip is visible.
function renderTokenText(raw: string): { display: string } {
  if (raw === '') return { display: '∅' };
  let text = raw;
  if (text.startsWith('Ġ')) {
    text = ' ' + text.slice(1);
  }
  text = text.replace(/Ġ/g, ' ');
  if (text.startsWith(' ')) {
    text = '·' + text.slice(1);
  }
  text = text.replace(/\n/g, '↵').replace(/\t/g, '→');
  if (text.length > 18) text = text.slice(0, 17) + '…';
  return { display: text };
}

function isSpecialToken(tok: TokenInfo): boolean {
  return SPECIAL_TOKEN_TEXT_RE.test(tok.text);
}

export function TokenizerDemo({ workerRef, abortRef }: TokenizerDemoProps) {
  const [prompt, setPrompt] = React.useState(DEFAULT_PROMPT);
  const [tokens, setTokens] = React.useState<TokenInfo[]>([]);
  const [status, setStatus] = React.useState<TokenizeStatus | null>(null);
  const [running, setRunning] = React.useState(false);

  // Each new request bumps a generation counter so a slow earlier reply
  // doesn't stomp the chips with stale data.
  const reqGenRef = React.useRef(0);
  // Per-call AbortController for the current in-flight tokenize. Aborting it
  // when a new keystroke arrives (or on unmount) cancels the worker call
  // itself, not just its state write — preventing forward-pass stacking on
  // fast typing.
  const tokenizeAbortRef = React.useRef<AbortController | null>(null);

  const runTokenize = React.useCallback(
    async (text: string) => {
      const myGen = ++reqGenRef.current;
      const worker = workerRef.current;
      if (!worker) {
        if (reqGenRef.current !== myGen) return;
        const fallback = fallbackHeuristicTokens(text);
        setTokens(fallback);
        setStatus({ kind: 'fallback-no-worker' });
        console.log('[tokenizer-demo] using heuristic fallback (model not loaded)', {
          promptLength: text.length,
          pieces: fallback.length,
        });
        return;
      }

      // Cancel any previous in-flight call before starting a new one. The
      // generation-counter guard below is kept as belt-and-suspenders for the
      // tiny window between abort dispatch and the listener firing.
      tokenizeAbortRef.current?.abort();
      const ctrl = new AbortController();
      tokenizeAbortRef.current = ctrl;

      // Chain the app-level abort (worker termination) into our local
      // controller so terminating the worker also tears down this call. We
      // either short-circuit synchronously or attach a one-shot listener that
      // we clean up in `finally`.
      const appSignal = abortRef.current?.signal;
      if (appSignal?.aborted) {
        ctrl.abort();
      }
      const onAppAbort = () => ctrl.abort();
      if (appSignal && !appSignal.aborted) {
        appSignal.addEventListener('abort', onAppAbort, { once: true });
      }

      setRunning(true);
      try {
        const result = await tokenize(worker, text, { signal: ctrl.signal });
        if (reqGenRef.current !== myGen) return;
        setTokens(result);
        setStatus({ kind: 'ok' });
      } catch (err) {
        if (reqGenRef.current !== myGen) return;
        if (isAbortError(err)) {
          // A newer call superseded us — let it set the next state. Only
          // surface "aborted" UI when the app-level signal (worker
          // termination) is the cause.
          if (appSignal?.aborted) {
            console.info('[tokenizer-demo] tokenize aborted (worker terminated)');
            setStatus({ kind: 'aborted' });
          }
        } else {
          const message = err instanceof Error ? err.message : String(err);
          console.error('[tokenizer-demo] tokenize failed; using heuristic fallback', err);
          const fallback = fallbackHeuristicTokens(text);
          setTokens(fallback);
          setStatus({ kind: 'fallback-error', error: message });
        }
      } finally {
        if (appSignal) {
          appSignal.removeEventListener('abort', onAppAbort);
        }
        if (tokenizeAbortRef.current === ctrl) {
          tokenizeAbortRef.current = null;
        }
        if (reqGenRef.current === myGen) setRunning(false);
      }
    },
    [workerRef, abortRef],
  );

  // Debounced re-tokenize on every prompt change. The worker call itself is
  // currently a short forward pass (we reach the tokenizer through
  // runForInspector), so the debounce keeps us from spamming the model on
  // every keystroke.
  React.useEffect(() => {
    const handle = setTimeout(() => {
      void runTokenize(prompt);
    }, TOKENIZE_DEBOUNCE_MS);
    return () => {
      clearTimeout(handle);
      // Cancel any pending in-flight tokenize on unmount / prompt change so
      // an outstanding worker call doesn't keep running after we're gone.
      tokenizeAbortRef.current?.abort();
    };
  }, [prompt, runTokenize]);

  const charCount = prompt.length;
  const tokenCount = tokens.length;
  const ratio = tokenCount > 0 ? (charCount / tokenCount).toFixed(2) : '—';

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <label htmlFor="tokenizer-demo-input" className="text-xs uppercase tracking-wider text-muted-foreground">
          Input
        </label>
        <Textarea
          id="tokenizer-demo-input"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={4}
          className="font-mono text-sm"
          placeholder="Type to tokenize..."
        />
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <Button variant="outline" size="sm" onClick={() => setPrompt(CODE_EXAMPLE)}>
            Try a code snippet
          </Button>
          <Button variant="outline" size="sm" onClick={() => setPrompt(MULTILINGUAL_EXAMPLE)}>
            Try multilingual
          </Button>
          <span className="text-muted-foreground">{running ? 'Tokenizing…' : 'Live encode on every keystroke.'}</span>
        </div>
      </div>

      {status?.kind === 'fallback-no-worker' ? (
        <div
          role="status"
          className="rounded-md border border-dashed border-amber-500/60 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300"
        >
          Showing a whitespace split — load the model first to see the real Qwen3.5 BPE tokens.
        </div>
      ) : null}
      {status?.kind === 'fallback-error' ? (
        <div
          role="alert"
          className="rounded-md border border-destructive/60 bg-destructive/10 px-3 py-2 text-xs text-destructive"
        >
          <strong>Tokenize failed.</strong> Showing a whitespace fallback instead. {status.error}
        </div>
      ) : null}
      {status?.kind === 'aborted' ? (
        <div
          role="status"
          className="rounded-md border border-dashed border-muted-foreground/40 bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
        >
          Tokenize cancelled — the model was reloaded. Type to retry.
        </div>
      ) : null}

      <div
        role="list"
        aria-label="Tokenized prompt"
        className="flex flex-wrap gap-1.5 rounded-md border border-border bg-background p-3 min-h-[3.5rem]"
      >
        {tokens.length === 0 ? (
          <span className="text-xs text-muted-foreground">(no tokens yet — type above)</span>
        ) : (
          tokens.map((tok, i) => <TokenChip key={i} token={tok} index={i} />)
        )}
      </div>

      <div aria-live="polite" className="rounded-md bg-muted/40 px-3 py-2 font-mono text-xs text-muted-foreground">
        {tokenCount} tokens · {charCount} characters · ratio: characters/token = {ratio}
      </div>

      <DemoCallout
        items={[
          'The dot prefix · marks a leading space — "cat" and " cat" are different tokens.',
          "Tokens with a dashed border are special tokens like <|im_start|> — Qwen's chat-template markers.",
          'Try the multilingual button: chars-per-token drops sharply for CJK and rare scripts.',
        ]}
      />
    </div>
  );
}

function TokenChip({ token, index }: { token: TokenInfo; index: number }) {
  const { display } = renderTokenText(token.text);
  const special = isSpecialToken(token);
  const palette = CHIP_PALETTE[index % CHIP_PALETTE.length]!;
  const idLabel = `id-tokenizer-demo-tok-${index}`;
  // Negative ids are synthesized by the heuristic fallback (no worker loaded).
  // Don't leak them into the tooltip or aria-described id label — they're
  // implementation detail, not real Qwen3.5 vocab indices.
  const isFallback = token.id < 0;
  const title = isFallback
    ? `fallback · ${JSON.stringify(token.text)}`
    : `id ${token.id} · ${JSON.stringify(token.text)}`;
  return (
    <span
      role="listitem"
      aria-describedby={idLabel}
      title={title}
      className={[
        'inline-flex items-center gap-1 rounded px-1.5 py-1 text-[13px] font-mono leading-none',
        special ? 'italic border border-dashed border-primary/60' : 'border border-transparent',
        palette,
      ].join(' ')}
    >
      <span>{display}</span>
      <span id={idLabel} className="text-[10px] opacity-60">
        {isFallback ? '·' : token.id}
      </span>
    </span>
  );
}
