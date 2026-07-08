import { Loader2Icon } from 'lucide-react';
import * as React from 'react';

import type { ScorePosition, ScoreTokensRun, TokenInfo } from '../../../src/inspector-types';
import type { Locale } from '../../lib/i18n';
import { useLocale } from '../../lib/i18n-react';
import { scoreTokens } from '../../lib/score-client';
import { tokenize } from '../../lib/tokenizer-client';
import { useFreeChatOptional } from '../../providers/free-chat';
import { useModelLoaderOptional } from '../../providers/model-loader';
import { renderTokenDisplay, TopKBars } from '../inspector/TopKBars';
import { RunButton } from '../scaffolding/RunButton';
import { SegmentedToggle } from '../scaffolding/SegmentedToggle';

/**
 * SpeculativeVerifyLive — the MTP sub-chapter's LIVE demo: the verify half of
 * speculative decoding running on the ACTUAL in-browser Qwen3.5-0.8B.
 *
 * A cheap NON-model drafter proposes D ≤ 4 tokens for a committed prefix:
 *   - prompt-lookup presets: a REAL n-gram lookup over the prefix's own token
 *     ids (the longest 2–3-token suffix that occurred earlier proposes the
 *     tokens that followed that earlier occurrence — exactly what prompt-lookup
 *     decoding does), or
 *   - "type your own": the reader's guessed continuation, tokenized, first ≤ 4
 *     tokens taken as the draft.
 * The real model then verifies ALL D draft tokens in ONE forward pass
 * (`scoreTokens`: one forward over [prefix + draft], per-position top-K
 * logits). Accept rule per position i: argmaxTokenId === draftTokenId (greedy,
 * temperature 0). At the first reject we show the model's REAL correction
 * token (its argmax) and roll back the rest — the lossless guarantee, live.
 *
 * The draft NEVER comes from Qwen's MTP head — this checkpoint ships zero
 * mtp.* tensors (the section's honesty close covers that); the point here is
 * the single-pass verify, which is real.
 *
 * PRERENDER SAFETY: this widget renders inside SSG-prerendered section HTML
 * where no app providers are mounted. It therefore reads the worker/model
 * state ONLY through the non-throwing `useFreeChatOptional()` /
 * `useModelLoaderOptional()` hooks and renders a static fallback frame when
 * they return null. No window/document access on the render path; all live
 * work happens in effects/handlers (which never run under renderToString).
 *
 * The prefix is tokenized RAW (the worker's `tokenize` message applies no chat
 * template) and `scoreTokens` feeds the ids verbatim — this is a raw
 * language-model continuation demo, and the copy says so.
 */

// ---------------------------------------------------------------------------
// Constants + presets
// ---------------------------------------------------------------------------

const TOP_K = 20; // requested per-position top-K (effective = min(20, 64, vocab))
const MAX_DRAFT = 4; // D ≤ 4 per the widget contract (worker allows 1..8)
const LOOKUP_MAX_NGRAM = 3; // longest prefix-suffix n-gram to look up (3 → 2)
const REVEAL_STEP_MS = 400;

// Accept/reject accents shared with SpeculativeDecodeDiagram's visual language.
const EMERALD = '#10b981'; // emerald-500 — accepted
const RED = '#ef4444'; // red-500 — rejected

type DraftSource = 'lookup' | 'typed';

type PresetDef = {
  id: 'capitals' | 'pangram' | 'typed';
  source: DraftSource;
  /** Committed prefix, tokenized raw (no chat template). Kept short (≤ ~30 tokens). */
  prefix: string;
};

const PRESETS: readonly PresetDef[] = [
  {
    // The last tokens "…of France is" also occur near the start; prompt-lookup
    // proposes the tokens that followed there: " Paris", ".", " The", " capital".
    id: 'capitals',
    source: 'lookup',
    prefix: 'The capital of France is Paris. The capital of Italy is Rome. The capital of France is',
  },
  {
    // "…quick brown fox jumps" repeats; the lookup proposes " over the lazy dog".
    id: 'pangram',
    source: 'lookup',
    prefix: 'The quick brown fox jumps over the lazy dog. Again: the quick brown fox jumps',
  },
  {
    id: 'typed',
    source: 'typed',
    prefix: 'The capital of France is',
  },
];

const DEFAULT_TYPED = ' Paris, the city of';

// Placeholder chips for the static (prerender / model-not-ready) frame: what
// preset 1's prompt-lookup is expected to propose. Display-only — the live
// path recomputes the draft from the real tokenizer.
const STATIC_CHIPS = ['Paris', '.', 'The', 'capital'];

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

/**
 * REAL prompt-lookup drafting over token ids: find the longest suffix of
 * `ids` (length LOOKUP_MAX_NGRAM down to 2) that also occurs earlier in
 * `ids`, scanning for the most recent earlier occurrence, and propose up to
 * `maxDraft` tokens that followed it. Returns null when no n-gram repeats.
 */
function promptLookupDraft(ids: number[], maxDraft: number): { start: number; ngram: number; count: number } | null {
  for (let n = Math.min(LOOKUP_MAX_NGRAM, ids.length - 1); n >= 2; n--) {
    const suffixStart = ids.length - n;
    for (let start = suffixStart - 1; start >= 0; start--) {
      let match = true;
      for (let j = 0; j < n; j++) {
        if (ids[start + j] !== ids[suffixStart + j]) {
          match = false;
          break;
        }
      }
      if (!match) continue;
      const followStart = start + n;
      const count = Math.min(maxDraft, ids.length - followStart);
      if (count > 0) return { start: followStart, ngram: n, count };
    }
  }
  return null;
}

function softmax(values: Float32Array): number[] {
  if (values.length === 0) return [];
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) if (values[i]! > max) max = values[i]!;
  const exps = new Array<number>(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    const e = Math.exp(values[i]! - max);
    exps[i] = e;
    sum += e;
  }
  if (sum === 0) return exps.map(() => 1 / exps.length);
  return exps.map((e) => e / sum);
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

// Same idiom as the sibling widgets (SpeculativeDecodeDiagram etc.): matchMedia
// read in the useState initializer is the deliberate course convention.
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

// ---------------------------------------------------------------------------
// Per-locale copy. Glossary terms (token, draft, verify, argmax, logits,
// prompt-lookup, top-K…) stay English per the course convention.
// ---------------------------------------------------------------------------

const COPY = {
  en: {
    title: 'Live verify — the real model, one forward pass',
    liveBadge: 'runs in your browser',
    staticCaption: 'Interactive demo — loads after the page boots.',
    loadCta: 'Load Qwen3.5-0.8B (~1.6 GB) to run this live',
    loadNote: 'Runs entirely on your device; nothing leaves your browser.',
    loadingLabel: 'Loading Qwen3.5-0.8B…',
    presetLabel: 'Draft source',
    presetNames: {
      capitals: 'Prompt-lookup · capitals',
      pangram: 'Prompt-lookup · pangram',
      typed: 'Type your own',
    },
    prefixLabel: 'Committed prefix (raw text — no chat template)',
    draftLabel: 'Proposed draft (not from the model)',
    typedInputLabel: 'Your guessed continuation',
    typedPlaceholder: 'e.g. " Paris, the city of"',
    typedHint: (d: number) => `Type a few words — the first ${d} tokens become the draft.`,
    lookupHint: (n: number) =>
      `Prompt-lookup: the last ${n} tokens of the prefix also occur earlier — the tokens that followed there become the draft.`,
    lookupMiss: 'Prompt-lookup found no repeated n-gram in this prefix, so there is nothing to draft.',
    typedEmpty: 'Type a continuation above to propose a draft.',
    preparing: 'tokenizing…',
    runCta: 'Run verify on the live model',
    runningLabel: 'Verifying…',
    errorPrefix: 'Verify failed.',
    correctionBadge: 'model’s pick',
    rolledBackBadge: 'rolled back',
    verifiedHeadline: (d: number) =>
      d === 1 ? '1 draft token verified in ONE forward pass' : `${d} draft tokens verified in ONE forward pass`,
    tallyAccepted: (k: number, d: number) => `${k} of ${d} accepted`,
    fullAcceptNote:
      'Every draft token matched the model’s own greedy pick — a real engine would commit all of them from this single pass.',
    correctionNote:
      'At the first mismatch the model’s own pick (beside the ✗) continues the text — that correction is why speculative decoding stays lossless.',
    barsSummary: (i: number, t: string, k: number) =>
      `position ${i + 1} · draft "${t}" · top-${k} logits, softmaxed over just these ${k} — renormalized share, not full-vocab probability`,
    barsRolledBack: 'computed in the same pass, discarded after the reject',
    barsDraftMissing: (t: string, k: number) => `The draft token "${t}" is not in the model’s top-${k} here.`,
    chipProposedAria: (t: string) => `draft token "${t}" — proposed, not yet verified`,
    chipAcceptedAria: (t: string) => `draft token "${t}" — accepted by the model`,
    chipRejectedAria: (t: string, c: string) => `draft token "${t}" — rejected; the model picked "${c}"`,
    chipRolledBackAria: (t: string) => `draft token "${t}" — rolled back after the first reject`,
    chipsAria: 'Draft tokens under verification',
    footnote:
      'The draft comes from prompt-lookup or your typing — a cheap, model-free drafter, not Qwen’s MTP head (this checkpoint ships no MTP weights; the section below has the full story). The verify is the real model.',
  },
  zh: {
    title: 'Verify 现场版——真实模型，一次前向',
    liveBadge: '在你的浏览器里运行',
    staticCaption: '互动演示——页面启动后加载。',
    loadCta: '加载 Qwen3.5-0.8B（约 1.6 GB）来实时运行',
    loadNote: '完全在你的设备上运行，数据不会离开浏览器。',
    loadingLabel: '正在加载 Qwen3.5-0.8B……',
    presetLabel: 'Draft 来源',
    presetNames: {
      capitals: 'Prompt-lookup · 首都',
      pangram: 'Prompt-lookup · 全字母句',
      typed: '自己输入',
    },
    prefixLabel: '已提交的前缀（原始文本——不套 chat template）',
    draftLabel: '提议的 draft（不来自模型）',
    typedInputLabel: '你猜的续写',
    typedPlaceholder: '例如 " Paris, the city of"',
    typedHint: (d: number) => `输入几个词——前 ${d} 个 token 会成为 draft。`,
    lookupHint: (n: number) =>
      `Prompt-lookup：前缀末尾的 ${n} 个 token 在前面也出现过——当时紧跟其后的 token 就成了 draft。`,
    lookupMiss: 'Prompt-lookup 在这个前缀里没找到重复的 n-gram，所以没有可提议的 draft。',
    typedEmpty: '在上面输入一段续写，才能提出 draft。',
    preparing: '正在 tokenize……',
    runCta: '在实时模型上运行 verify',
    runningLabel: '正在 verify……',
    errorPrefix: 'Verify 失败。',
    correctionBadge: '模型的选择',
    rolledBackBadge: '已回滚',
    verifiedHeadline: (d: number) => `一次前向就 verify 了 ${d} 个 draft token`,
    tallyAccepted: (k: number, d: number) => `${d} 个中接受了 ${k} 个`,
    fullAcceptNote: '每个 draft token 都与模型自己的 greedy 选择一致——真实引擎会把它们从这一次前向里全部提交。',
    correctionNote: '在第一个不匹配处，模型自己的选择（✗ 旁边）会接着往下写——正是这个纠正让推测式解码保持无损。',
    barsSummary: (i: number, t: string, k: number) =>
      `位置 ${i + 1} · draft "${t}" · top-${k} logits，只对这 ${k} 个做 softmax——是重新归一化的份额，不是全词表概率`,
    barsRolledBack: '同一次前向里算出，但在拒绝之后被丢弃',
    barsDraftMissing: (t: string, k: number) => `draft token "${t}" 不在模型这里的 top-${k} 里。`,
    chipProposedAria: (t: string) => `draft token "${t}"——已提议，尚未 verify`,
    chipAcceptedAria: (t: string) => `draft token "${t}"——被模型接受`,
    chipRejectedAria: (t: string, c: string) => `draft token "${t}"——被拒绝；模型选了 "${c}"`,
    chipRolledBackAria: (t: string) => `draft token "${t}"——在第一个拒绝之后被回滚`,
    chipsAria: '正在 verify 的 draft token',
    footnote:
      'Draft 来自 prompt-lookup 或你的输入——一个廉价的、不用模型的 drafter，而不是 Qwen 的 MTP head（这份 checkpoint 不带 MTP 权重；下面一节有完整说明）。Verify 用的是真实模型。',
  },
} as const;

// ---------------------------------------------------------------------------
// Internal state shapes
// ---------------------------------------------------------------------------

type PreparedDraft = {
  presetId: PresetDef['id'];
  prefixIds: number[];
  /** Draft token ids (length 1..MAX_DRAFT) — [] when nothing to propose. */
  draftIds: number[];
  /** Decoded text per draft id, index-aligned. */
  draftTexts: string[];
  /** Matched n-gram length for lookup drafts; null for typed drafts. */
  ngram: number | null;
};

type VerifyState =
  | { kind: 'idle' }
  | { kind: 'running' }
  | { kind: 'done'; result: ScoreTokensRun; verdicts: boolean[]; firstReject: number }
  | { kind: 'error'; message: string };

// ---------------------------------------------------------------------------
// Chips
// ---------------------------------------------------------------------------

type ChipStatus = 'proposed' | 'accepted' | 'rejected' | 'rolledback';

function chipStyle(status: ChipStatus): React.CSSProperties {
  switch (status) {
    case 'accepted':
      return { borderColor: EMERALD, background: `${EMERALD}24`, color: EMERALD, borderStyle: 'solid' };
    case 'rejected':
      return { borderColor: RED, background: `${RED}24`, color: RED, borderStyle: 'solid' };
    case 'rolledback':
      return {
        borderColor: 'var(--border)',
        color: 'var(--muted-foreground)',
        borderStyle: 'dashed',
        opacity: 0.4,
        textDecoration: 'line-through',
      };
    default:
      return { borderColor: 'var(--border)', color: 'var(--muted-foreground)', borderStyle: 'dashed' };
  }
}

function DraftChip({ text, status, ariaLabel }: { text: string; status: ChipStatus; ariaLabel: string }) {
  return (
    <span
      role="listitem"
      aria-label={ariaLabel}
      className="inline-flex items-center gap-1 rounded-md border px-2 py-1 font-mono text-[13px] font-semibold transition-all duration-300"
      style={chipStyle(status)}
    >
      {renderTokenDisplay(text)}
      {status === 'accepted' ? <span aria-hidden="true">✓</span> : null}
      {status === 'rejected' ? <span aria-hidden="true">✗</span> : null}
    </span>
  );
}

// ---------------------------------------------------------------------------
// The widget
// ---------------------------------------------------------------------------

export function SpeculativeVerifyLive() {
  const locale: Locale = useLocale();
  const copy = COPY[locale];
  const reducedMotion = usePrefersReducedMotion();

  // Optional (non-throwing) provider hooks — null under SSG prerender, where
  // this component must render its static frame without touching the model.
  const freeChat = useFreeChatOptional();
  const loader = useModelLoaderOptional();

  const [presetId, setPresetId] = React.useState<PresetDef['id']>(PRESETS[0]!.id);
  const preset = PRESETS.find((p) => p.id === presetId) ?? PRESETS[0]!;
  const [typedText, setTypedText] = React.useState(DEFAULT_TYPED);

  const [prepared, setPrepared] = React.useState<PreparedDraft | null>(null);
  const [prepError, setPrepError] = React.useState<string | null>(null);
  const [preparing, setPreparing] = React.useState(false);
  const [verify, setVerify] = React.useState<VerifyState>({ kind: 'idle' });
  // How many draft positions have been revealed by the post-run animation.
  const [revealed, setRevealed] = React.useState(0);
  // Bumped per successful run so TopKBars re-fires its grow-in animation.
  const [runKey, setRunKey] = React.useState(0);

  // Monotonic generation guards + per-run AbortControllers (TrainingPlayground
  // discipline): stale async results are dropped, unmount aborts everything.
  const prepGenRef = React.useRef(0);
  const runGenRef = React.useRef(0);
  const runAbortRef = React.useRef<AbortController | null>(null);

  React.useEffect(
    () => () => {
      prepGenRef.current++;
      runGenRef.current++;
      runAbortRef.current?.abort();
    },
    [],
  );

  const ready = loader?.status === 'ready';
  const workerRef = freeChat?.mlxWorkerRef ?? null;
  const appAbortRef = freeChat?.inspectorAbortRef ?? null;

  // -------------------------------------------------------------------------
  // Draft preparation: tokenize the prefix (raw — no chat template) and derive
  // the proposed draft chips. Lookup presets run the real prompt-lookup over
  // the prefix ids; typed mode tokenizes the reader's continuation and takes
  // the first ≤ MAX_DRAFT tokens. Debounced while typing.
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!ready || !workerRef) return;
    const worker = workerRef.current;
    if (!worker) return;
    const myGen = ++prepGenRef.current;
    const ctrl = new AbortController();
    const typedSnapshot = typedText;
    const currentPreset = PRESETS.find((p) => p.id === presetId) ?? PRESETS[0]!;

    const prepare = async () => {
      setPreparing(true);
      try {
        const prefixTokens: TokenInfo[] = await tokenize(worker, currentPreset.prefix, { signal: ctrl.signal });
        if (prepGenRef.current !== myGen) return;
        const prefixIds = prefixTokens.map((t) => t.id);
        let draftIds: number[] = [];
        let draftTexts: string[] = [];
        let ngram: number | null = null;
        if (currentPreset.source === 'lookup') {
          const hit = promptLookupDraft(prefixIds, MAX_DRAFT);
          if (hit) {
            const followed = prefixTokens.slice(hit.start, hit.start + hit.count);
            draftIds = followed.map((t) => t.id);
            draftTexts = followed.map((t) => t.text);
            ngram = hit.ngram;
          }
        } else if (typedSnapshot.trim().length > 0) {
          const typedTokens = await tokenize(worker, typedSnapshot, { signal: ctrl.signal });
          if (prepGenRef.current !== myGen) return;
          const taken = typedTokens.slice(0, MAX_DRAFT);
          draftIds = taken.map((t) => t.id);
          draftTexts = taken.map((t) => t.text);
        }
        if (prepGenRef.current !== myGen) return;
        setPrepared({ presetId: currentPreset.id, prefixIds, draftIds, draftTexts, ngram });
        setPrepError(null);
      } catch (err) {
        if (prepGenRef.current !== myGen || isAbortError(err)) return;
        setPrepared(null);
        setPrepError(err instanceof Error ? err.message : String(err));
      } finally {
        if (prepGenRef.current === myGen) setPreparing(false);
      }
    };

    // Debounce keystrokes in typed mode; preset switches tokenize immediately.
    const delay = currentPreset.source === 'typed' ? 250 : 0;
    const timer = window.setTimeout(() => void prepare(), delay);
    return () => {
      window.clearTimeout(timer);
      ctrl.abort();
    };
  }, [ready, workerRef, presetId, typedText]);

  // -------------------------------------------------------------------------
  // Reveal animation: after a run completes, uncover verdicts left → right in
  // REVEAL_STEP_MS steps, stopping at the first reject (later chips flip to
  // "rolled back"). Reduced motion reveals everything at once.
  // -------------------------------------------------------------------------
  const revealTarget =
    verify.kind === 'done' ? (verify.firstReject === -1 ? verify.result.draftLen : verify.firstReject + 1) : 0;
  React.useEffect(() => {
    if (verify.kind !== 'done') return;
    if (reducedMotion) {
      setRevealed(revealTarget);
      return;
    }
    setRevealed(0);
    let i = 0;
    const t = window.setInterval(() => {
      i++;
      setRevealed(i);
      if (i >= revealTarget) window.clearInterval(t);
    }, REVEAL_STEP_MS);
    return () => window.clearInterval(t);
  }, [verify, revealTarget, reducedMotion]);
  const revealDone = verify.kind === 'done' && revealed >= revealTarget;

  // -------------------------------------------------------------------------
  // The one live call: scoreTokens — ONE forward pass over [prefix + draft].
  // -------------------------------------------------------------------------
  async function handleRun() {
    if (!prepared || prepared.draftIds.length === 0) return;
    const worker = workerRef?.current ?? null;
    if (!worker) {
      setVerify({ kind: 'error', message: 'MLX worker is not available' });
      return;
    }
    const myGen = ++runGenRef.current;
    setVerify({ kind: 'running' });
    setRevealed(0);

    runAbortRef.current?.abort();
    const ctrl = new AbortController();
    runAbortRef.current = ctrl;
    // Bridge to the app-level inspector abort (worker teardown / model reload)
    // exactly once per run — the LmHeadDemo idiom.
    const appSignal = appAbortRef?.current?.signal;
    const onAppAbort = () => ctrl.abort();
    if (appSignal?.aborted) ctrl.abort();
    else appSignal?.addEventListener('abort', onAppAbort, { once: true });

    try {
      const result = await scoreTokens(
        worker,
        { prefixIds: prepared.prefixIds, draftIds: prepared.draftIds, topK: TOP_K },
        { signal: ctrl.signal },
      );
      if (runGenRef.current !== myGen) return;
      const verdicts = result.positions.map((p) => p.argmaxTokenId === p.draftTokenId);
      const firstReject = verdicts.findIndex((v) => !v);
      setRunKey((k) => k + 1);
      setVerify({ kind: 'done', result, verdicts, firstReject });
    } catch (err) {
      if (runGenRef.current !== myGen) return;
      if (isAbortError(err)) {
        setVerify({ kind: 'idle' });
      } else {
        setVerify({ kind: 'error', message: err instanceof Error ? err.message : String(err) });
      }
    } finally {
      appSignal?.removeEventListener('abort', onAppAbort);
      if (runAbortRef.current === ctrl) runAbortRef.current = null;
    }
  }

  function pickPreset(id: PresetDef['id']) {
    setPresetId(id);
    setPrepared(null);
    setPrepError(null);
    setVerify({ kind: 'idle' });
    setRevealed(0);
  }

  function handleTypedChange(v: string) {
    setTypedText(v);
    setPrepError(null);
    setVerify({ kind: 'idle' });
    setRevealed(0);
  }

  // -------------------------------------------------------------------------
  // Render helpers
  // -------------------------------------------------------------------------

  const frame = (children: React.ReactNode) => (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <span className="inline-flex items-center gap-1.5 rounded border border-primary/40 bg-primary/10 px-2 py-0.5 text-[11px] text-primary">
          {copy.liveBadge}
        </span>
      </div>
      {children}
    </div>
  );

  const staticPreview = (
    <>
      <div className="space-y-1">
        <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.prefixLabel}</div>
        <div className="overflow-x-auto rounded-md border border-border bg-muted/40 p-2 font-mono text-[12px] text-foreground/80">
          {PRESETS[0]!.prefix}
        </div>
      </div>
      <div className="space-y-1">
        <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.draftLabel}</div>
        <div role="list" aria-label={copy.chipsAria} className="flex flex-wrap items-center gap-1.5">
          {STATIC_CHIPS.map((t, i) => (
            <DraftChip key={i} text={t} status="proposed" ariaLabel={copy.chipProposedAria(t)} />
          ))}
        </div>
      </div>
    </>
  );

  // ---- State 1: prerender / no providers → static frame, nothing live. ----
  if (!freeChat || !loader) {
    return frame(
      <>
        {staticPreview}
        <p className="text-[11px] italic text-muted-foreground/70">{copy.staticCaption}</p>
      </>,
    );
  }

  // ---- State 2: providers present, model not ready → load CTA / spinner. ----
  if (!ready) {
    const loading = loader.status === 'loading';
    return frame(
      <>
        {staticPreview}
        {loading ? (
          <div className="flex items-center gap-2 text-xs text-muted-foreground" role="status" aria-live="polite">
            <Loader2Icon className="h-3.5 w-3.5 animate-spin text-primary" aria-hidden="true" />
            <span>{loader.loadingText || copy.loadingLabel}</span>
          </div>
        ) : (
          <div className="flex flex-col items-start gap-1.5">
            <button
              type="button"
              onClick={() => loader.kickoffLoad()}
              className="rounded-lg border border-primary/50 bg-primary/10 px-4 py-2 text-sm font-medium text-primary transition-colors hover:bg-primary/20 focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
            >
              {copy.loadCta}
            </button>
            <span className="text-[11px] text-muted-foreground">{copy.loadNote}</span>
            {loader.status === 'error' && loader.errorBanner ? (
              <span className="text-[11px] text-muted-foreground" role="alert">
                {copy.errorPrefix} {loader.errorBanner}
              </span>
            ) : null}
          </div>
        )}
      </>,
    );
  }

  // ---- State 3: live. ----
  const draftReady = prepared !== null && prepared.presetId === preset.id && prepared.draftIds.length > 0;
  const chips = draftReady ? prepared.draftTexts : [];
  const running = verify.kind === 'running';
  const done = verify.kind === 'done' ? verify : null;
  const accepted = done ? (done.firstReject === -1 ? done.result.draftLen : done.firstReject) : 0;

  const chipStatusAt = (i: number): ChipStatus => {
    if (!done || i >= revealed) {
      if (done && revealDone && done.firstReject !== -1 && i > done.firstReject) return 'rolledback';
      return 'proposed';
    }
    return done.verdicts[i] ? 'accepted' : 'rejected';
  };

  const chipAriaAt = (i: number, text: string): string => {
    const status = chipStatusAt(i);
    const display = renderTokenDisplay(text);
    if (status === 'accepted') return copy.chipAcceptedAria(display);
    if (status === 'rejected') {
      const correction = done ? renderTokenDisplay(done.result.positions[i]?.topKTexts[0] ?? '') : '';
      return copy.chipRejectedAria(display, correction);
    }
    if (status === 'rolledback') return copy.chipRolledBackAria(display);
    return copy.chipProposedAria(display);
  };

  const sourceHint =
    preset.source === 'typed'
      ? copy.typedHint(MAX_DRAFT)
      : draftReady && prepared.ngram !== null
        ? copy.lookupHint(prepared.ngram)
        : prepared !== null && prepared.presetId === preset.id
          ? copy.lookupMiss
          : null;

  return frame(
    <>
      {/* Preset picker */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.presetLabel}</span>
        <SegmentedToggle
          value={presetId}
          onChange={(v) => pickPreset(v as PresetDef['id'])}
          ariaLabel={copy.presetLabel}
          disabled={running}
          wrap
          options={PRESETS.map((p) => ({ value: p.id, label: copy.presetNames[p.id] }))}
        />
      </div>

      {/* Committed prefix, raw */}
      <div className="space-y-1">
        <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.prefixLabel}</div>
        <div className="overflow-x-auto rounded-md border border-border bg-muted/40 p-2 font-mono text-[12px] text-foreground/80">
          {preset.prefix}
        </div>
      </div>

      {/* Typed continuation input */}
      {preset.source === 'typed' ? (
        <div className="space-y-1">
          <label htmlFor="spec-verify-typed" className="text-[11px] uppercase tracking-wider text-muted-foreground">
            {copy.typedInputLabel}
          </label>
          <input
            id="spec-verify-typed"
            type="text"
            value={typedText}
            onChange={(e) => handleTypedChange(e.target.value)}
            placeholder={copy.typedPlaceholder}
            maxLength={80}
            disabled={running}
            className="w-full rounded-md border border-input bg-transparent px-3 py-2 font-mono text-sm outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50"
          />
        </div>
      ) : null}

      {/* Draft chips + correction */}
      <div className="space-y-1">
        <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.draftLabel}</div>
        <div role="list" aria-label={copy.chipsAria} className="flex flex-wrap items-center gap-1.5">
          {chips.map((text, i) => {
            const status = chipStatusAt(i);
            return (
              <React.Fragment key={`${runKey}-${i}`}>
                <DraftChip text={text} status={status} ariaLabel={chipAriaAt(i, text)} />
                {status === 'rejected' && done ? (
                  <span
                    className="inline-flex items-center gap-1 rounded-md border px-2 py-1 font-mono text-[13px] font-semibold"
                    style={{ borderColor: EMERALD, background: `${EMERALD}24`, color: EMERALD }}
                    aria-label={`${copy.correctionBadge}: ${renderTokenDisplay(done.result.positions[i]?.topKTexts[0] ?? '')}`}
                  >
                    → {renderTokenDisplay(done.result.positions[i]?.topKTexts[0] ?? '')}
                    <span className="text-[10px] font-normal opacity-80">{copy.correctionBadge}</span>
                  </span>
                ) : null}
              </React.Fragment>
            );
          })}
          {chips.length === 0 ? (
            <span className="text-[12px] text-muted-foreground">
              {preparing
                ? copy.preparing
                : prepError !== null
                  ? `${copy.errorPrefix} ${prepError}`
                  : preset.source === 'typed'
                    ? copy.typedEmpty
                    : (sourceHint ?? copy.preparing)}
            </span>
          ) : null}
        </div>
        {chips.length > 0 && sourceHint ? <p className="text-[11px] text-muted-foreground">{sourceHint}</p> : null}
      </div>

      {/* Run */}
      <div className="flex flex-wrap items-center gap-2">
        <RunButton
          onClick={() => void handleRun()}
          running={running}
          label={copy.runCta}
          runningLabel={copy.runningLabel}
          disabled={!draftReady || preparing}
        />
      </div>

      {/* Error — inline, muted, non-blocking; controls stay usable (idle-equivalent). */}
      {verify.kind === 'error' ? (
        <p className="text-[12px] text-muted-foreground" role="alert">
          <strong>{copy.errorPrefix}</strong> {verify.message}
        </p>
      ) : null}

      {/* Tally + per-position real top-K bars */}
      {done && revealDone ? (
        <div className="space-y-2 border-t border-border/60 pt-3">
          <div className="flex flex-wrap items-baseline justify-between gap-2" role="status" aria-live="polite">
            <span className="text-[13px] font-medium text-foreground">
              {copy.verifiedHeadline(done.result.draftLen)}
            </span>
            <span className="font-mono text-[12px] text-muted-foreground">
              {copy.tallyAccepted(accepted, done.result.draftLen)}
            </span>
          </div>
          <p className="text-[12px] text-muted-foreground">
            {done.firstReject === -1 ? copy.fullAcceptNote : copy.correctionNote}
          </p>
          <div className="space-y-1.5">
            {done.result.positions.map((pos: ScorePosition, i: number) => {
              const probs = softmax(pos.topKLogits);
              const rolledBack = done.firstReject !== -1 && i > done.firstReject;
              const draftDisplay = renderTokenDisplay(
                pos.topKTexts[pos.topKIds.indexOf(pos.draftTokenId)] ?? chips[i] ?? '',
              );
              const draftInTopK = pos.topKIds.includes(pos.draftTokenId);
              return (
                <details key={`${runKey}-${i}`} className="rounded-md border border-border bg-muted/20 px-2 py-1.5">
                  <summary
                    className="cursor-pointer text-[12px] text-foreground/85"
                    style={rolledBack ? { opacity: 0.5 } : undefined}
                  >
                    {copy.barsSummary(i, draftDisplay, done.result.topK)}
                    {rolledBack ? (
                      <span className="ml-1 text-[11px] italic text-muted-foreground">— {copy.barsRolledBack}</span>
                    ) : null}
                  </summary>
                  <div className="mt-2 space-y-1.5">
                    {!draftInTopK ? (
                      <p className="text-[11px] text-muted-foreground">
                        {copy.barsDraftMissing(draftDisplay, done.result.topK)}
                      </p>
                    ) : null}
                    <TopKBars
                      ids={pos.topKIds}
                      probs={probs}
                      texts={pos.topKTexts}
                      sampledTokenId={pos.draftTokenId}
                      runKey={runKey}
                    />
                  </div>
                </details>
              );
            })}
          </div>
        </div>
      ) : null}

      <p className="text-[10px] text-muted-foreground/70">{copy.footnote}</p>
    </>,
  );
}
