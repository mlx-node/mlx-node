import { ArrowLeftIcon, MessageSquareIcon } from 'lucide-react';
import * as React from 'react';

import type { AttentionRun } from '../../src/inspector-types';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Textarea } from '../components/ui/textarea';
import { runForInspector } from '../lib/inspector-client';
import { CHAPTERS, findChapter, type ChapterMeta } from './chapters';
import { cleanupTokenText, renderTokenDisplay } from './inspector/TopKBars';
import { RunButton } from './scaffolding/RunButton';
import { useRunFlash } from './scaffolding/useRunFlash';

/** Chapter number for a stable chapter id, looked up from the registry. The
 *  forward-pass diagram labels its stages by id; deriving the number here (vs
 *  hardcoding it) keeps the chips from desyncing when chapters are renumbered. */
const chNum = (id: string): number => findChapter(id)?.number ?? 0;

export type ChapterIndexProps = {
  onOpenChapter: (chapterId: string) => void;
  onBackToLanding: () => void;
  onOpenFreeChat: () => void;
  workerRef: React.RefObject<Worker | null>;
  abortRef: React.RefObject<AbortController | null>;
};

// (Previous "curriculum phase" colour bands lived here. They drove the
// horizontal pill rail that has been replaced by the real-forward-pass
// flow below — see {@link ForwardPassFlow}.)

export function ChapterIndex({
  onOpenChapter,
  onBackToLanding,
  onOpenFreeChat,
  workerRef,
  abortRef,
}: ChapterIndexProps) {
  return (
    <div className="absolute inset-0 z-10 overflow-y-auto bg-background">
      <div className="mx-auto flex w-full max-w-5xl flex-col px-6 py-10">
        {/* Top bar */}
        <div className="mb-8 flex items-center justify-between">
          <button
            type="button"
            onClick={onBackToLanding}
            className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
          >
            <ArrowLeftIcon className="size-4" />
            Back
          </button>
          <Button variant="outline" size="sm" onClick={onOpenFreeChat} className="gap-2">
            <MessageSquareIcon className="size-4" />
            Open free chat
          </Button>
        </div>

        {/* Title */}
        <div className="mb-8">
          <div className="mb-2 font-mono text-xs uppercase tracking-[0.2em] text-primary">
            Learn LLMs · powered by Qwen3.5 in your browser
          </div>
          <h1 className="text-4xl font-semibold tracking-tight text-foreground">Chapters</h1>
          <p className="mt-3 max-w-2xl text-muted-foreground">
            {CHAPTERS.length} guided lessons that explain how a modern transformer LLM works, using the real model
            running in your browser via WebGPU. Read the prose on the left, then poke at the live model on the right.
          </p>
        </div>

        {/* Real forward-pass flow — runs the live model, captures per-layer
            data, and replays it through an Apple-Wallet-style card stack
            that shuffles between layers, with full vs linear-attention
            cards getting distinct treatments. */}
        <ForwardPassFlow onOpenChapter={onOpenChapter} workerRef={workerRef} abortRef={abortRef} />

        {/* Grid of chapter cards */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {CHAPTERS.map((chapter) => (
            <ChapterCard
              key={chapter.id}
              chapter={chapter}
              onOpen={() => (chapter.available ? onOpenChapter(chapter.id) : undefined)}
            />
          ))}
        </div>

        <p className="mt-10 text-xs text-muted-foreground">
          Each chapter is self-contained, but the suggested order builds the model from the bottom up — text in,
          attention through the middle, sampling out.
        </p>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------------------
// ForwardPassFlow — real model run + Apple-Wallet card-stack replay.
// ----------------------------------------------------------------------------
//
// The diagram is hybrid: a left rail with the traveling "hidden state" dot
// + a card-stack viewport where 24 layer cards are absolutely positioned
// and transforms shift in lock-step as the active layer changes. Each card
// shows its sub-stage list; the active sub-stage gets a highlight bar that
// slides between rows.
//
// Layer types: full-attention layers (Qwen3_5Attention) get a cyan ring
// and a 12-step internal sequence ending in MLP + residuals. Linear layers
// (GatedDeltaNet — most of Qwen3.5) get an amber ring and a GDN-specific
// 10-step sequence. The card visual + internal animation match the layer
// type so you can see when the model switches mode.

const NUM_LAYERS = 24;
const DEFAULT_PROMPT = 'The cat sat on the';

// How many predicted tokens the reveal shows. This diagram runs the model
// in *completion* mode (no chat template — see handleRun), and we measured
// that greedy raw completion stays clean for the first few tokens, then
// degenerates into verbatim repetition loops past ~5–6. So we cap the
// revealed window: rainbow-highlight the first predicted token, show the
// next few in muted gray, and stop before the loop sets in.
const REVEAL_TOKEN_CAP = 5;

// Note: dedup boundary (echoTokenCount) and special-token stopping come
// straight from the Rust inspector (see
// inspector.rs::AttentionRunNapi.echo_token_count and the stop_token_ids
// set in qwen3_5/model.rs::run_for_inspector_sync). The frontend just
// reads the structured count instead of regex-matching token text — the
// tokenizer is the source of truth for what counts as a stop marker. In
// completion mode the model continues the prompt rather than restating
// it, so echoTokenCount is ~always 0 and the dedup is a harmless no-op.

/** A sub-stage inside one of the 24 layer cards. */
type SubStage = { id: string; label: string; durationMs: number };

/** Full-attention layer (e.g. Qwen3.5 layers 3, 7, 11, 15, 19, 23).
 *  Each step has a >= 65 ms dwell so the human eye can still follow at 1×
 *  but the whole replay feels snappy. Earlier values were 2× this; ½×
 *  restores the slower-paced reading speed for first-time learners. */
const FULL_ATTN_SUBS: readonly SubStage[] = [
  { id: 'pre_norm', label: 'RMSNorm (pre)', durationMs: 75 },
  { id: 'qkv_proj', label: 'Q · K · V projection', durationMs: 75 },
  { id: 'rope', label: 'RoPE on Q · K', durationMs: 90 },
  { id: 'gqa_split', label: '8 Q heads · 2 KV heads (GQA)', durationMs: 90 },
  { id: 'attn_score', label: 'Q @ Kᵀ · scale', durationMs: 90 },
  { id: 'causal_mask', label: 'Causal mask', durationMs: 75 },
  { id: 'softmax', label: 'Softmax', durationMs: 100 },
  { id: 'attn_v', label: 'Attn @ V · out-proj', durationMs: 90 },
  { id: 'residual_attn', label: '+ residual', durationMs: 65 },
  { id: 'post_norm', label: 'RMSNorm (post)', durationMs: 75 },
  { id: 'mlp', label: 'MLP block (SwiGLU)', durationMs: 125 },
  { id: 'residual_mlp', label: '+ residual', durationMs: 65 },
];
// Sum ≈ 1.015s per full-attention layer at 1× (was 2.03s before halving).

/** Linear-attention layer (GatedDeltaNet — most of Qwen3.5's layers). */
const LINEAR_ATTN_SUBS: readonly SubStage[] = [
  { id: 'pre_norm', label: 'RMSNorm (pre)', durationMs: 75 },
  { id: 'gdn_gates', label: 'b · a · c · g gates', durationMs: 100 },
  { id: 'gdn_qkv', label: 'q · k · v projections', durationMs: 90 },
  { id: 'gdn_recurrent', label: 'S ← α·S + β·k⊗v (recurrent state)', durationMs: 140 },
  { id: 'gdn_output_gate', label: 'Gated output', durationMs: 90 },
  { id: 'gdn_out_proj', label: 'Out projection', durationMs: 75 },
  { id: 'residual_attn', label: '+ residual', durationMs: 65 },
  { id: 'post_norm', label: 'RMSNorm (post)', durationMs: 75 },
  { id: 'mlp', label: 'MLP block (SwiGLU)', durationMs: 125 },
  { id: 'residual_mlp', label: '+ residual', durationMs: 65 },
];
// Sum ≈ 0.90s per linear-attention layer at 1× (was 1.80s before halving).

/** Chapter links for sub-stage rows. Both layer types share most rows. */
const SUB_CHAPTER: Record<string, string | undefined> = {
  pre_norm: 'rmsnorm',
  qkv_proj: 'attention',
  rope: 'rope',
  gqa_split: 'multi-head-gqa',
  attn_score: 'attention',
  causal_mask: 'attention',
  softmax: 'attention',
  attn_v: 'attention',
  residual_attn: 'full-block',
  post_norm: 'rmsnorm',
  mlp: 'mlp',
  residual_mlp: 'full-block',
  gdn_gates: 'kv-cache',
  gdn_qkv: 'kv-cache',
  gdn_recurrent: 'kv-cache',
  gdn_output_gate: 'kv-cache',
  gdn_out_proj: 'kv-cache',
};

/** Stage ids. The layer scenes carry both the layer index and the sub-stage
 *  id (a string from one of the layer-type sequences). */
type StageId =
  | 'tokenize'
  | 'embedding'
  | { kind: 'layer'; index: number; subId: string }
  | 'final_norm'
  | 'lm_head'
  | 'sampling';

function isLayerStage(s: StageId): s is { kind: 'layer'; index: number; subId: string } {
  return typeof s === 'object' && s !== null && (s as { kind?: string }).kind === 'layer';
}

/** Dwell for the fixed top/bottom stages. Halved from the prior values so
 *  the whole replay is 2× faster by default; the ½× speed button restores
 *  the slower pace for first-time readers, and 4× is for impatient
 *  re-viewers. */
const FIXED_DWELL_MS: Record<string, number> = {
  tokenize: 300,
  embedding: 300,
  final_norm: 300,
  lm_head: 400,
  sampling: 750,
};

type Scene = { stage: StageId; durationMs: number };

const DEFAULT_FULL_ATTN_SET = new Set<number>([3, 7, 11, 15, 19, 23]);

function buildScenes(fullAttn: Set<number>): Scene[] {
  const scenes: Scene[] = [];
  scenes.push({ stage: 'tokenize', durationMs: FIXED_DWELL_MS.tokenize! });
  scenes.push({ stage: 'embedding', durationMs: FIXED_DWELL_MS.embedding! });
  for (let i = 0; i < NUM_LAYERS; i++) {
    const subs = fullAttn.has(i) ? FULL_ATTN_SUBS : LINEAR_ATTN_SUBS;
    for (const sub of subs) {
      scenes.push({ stage: { kind: 'layer', index: i, subId: sub.id }, durationMs: sub.durationMs });
    }
  }
  scenes.push({ stage: 'final_norm', durationMs: FIXED_DWELL_MS.final_norm! });
  scenes.push({ stage: 'lm_head', durationMs: FIXED_DWELL_MS.lm_head! });
  scenes.push({ stage: 'sampling', durationMs: FIXED_DWELL_MS.sampling! });
  return scenes;
}

type RunStatus =
  | { kind: 'idle' }
  | { kind: 'running' } // worker call in flight, before replay starts
  | { kind: 'replaying' }
  | { kind: 'paused' }
  | { kind: 'done' }
  | { kind: 'error'; error: string }
  | { kind: 'aborted' }
  | { kind: 'empty-prompt' };

type Speed = 0.5 | 1 | 2 | 4;

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

type ForwardPassFlowProps = {
  onOpenChapter: (chapterId: string) => void;
  workerRef: React.RefObject<Worker | null>;
  abortRef: React.RefObject<AbortController | null>;
};

function ForwardPassFlow({ onOpenChapter, workerRef, abortRef }: ForwardPassFlowProps) {
  const [prompt, setPrompt] = React.useState<string>(DEFAULT_PROMPT);
  const [status, setStatus] = React.useState<RunStatus>({ kind: 'idle' });
  const [run, setRun] = React.useState<AttentionRun | null>(null);
  const [activeIdx, setActiveIdx] = React.useState<number>(0);
  const [speed, setSpeed] = React.useState<Speed>(1);
  const reducedMotion = usePrefersReducedMotion();

  const fullAttentionLayerIndices = React.useMemo(() => {
    if (!run) return DEFAULT_FULL_ATTN_SET;
    const set = new Set<number>();
    // `run.attention` carries an entry for EVERY layer (the Rust side emits
    // a stub `kind: 'linear'` entry with empty scores for linear layers so
    // the frontend can reason about layer order). We only want the layers
    // whose `kind === 'full'` to render as full-attention cards.
    for (const a of run.attention ?? []) {
      if (a.kind === 'full') set.add(a.layerIndex);
    }
    // modelMeta carries only the full-attention indices by contract, so
    // unioning it is safe (and serves as a fallback when `attention` is
    // empty — e.g. an inspector request that disabled attention capture).
    for (const i of run.modelMeta?.fullAttentionLayerIndices ?? []) set.add(i);
    // Fall back to the default if the run didn't carry any (e.g. pre-run).
    return set.size === 0 ? DEFAULT_FULL_ATTN_SET : set;
  }, [run]);

  const scenes = React.useMemo(() => buildScenes(fullAttentionLayerIndices), [fullAttentionLayerIndices]);

  // Track which layer card is on the front of the stack. The dot uses this
  // too to know when it's "inside the layer block" vs at head/tail.
  const activeStage: StageId = scenes[activeIdx]?.stage ?? 'tokenize';
  const activeLayer = isLayerStage(activeStage) ? activeStage : null;

  // Total replay duration in seconds at 1× speed (for the helper copy).
  const totalReplayMs = React.useMemo(() => scenes.reduce((sum, s) => sum + s.durationMs, 0), [scenes]);

  // -- worker call lifecycle ----------------------------------------------
  const runGenRef = React.useRef(0);
  const runAbortRef = React.useRef<AbortController | null>(null);
  const didAutoRunRef = React.useRef(false);
  const resultFlashRef = useRunFlash<HTMLDivElement>(status.kind === 'running' || status.kind === 'replaying');

  const handleRun = React.useCallback(async () => {
    if (prompt.trim().length === 0) {
      setStatus({ kind: 'empty-prompt' });
      return;
    }
    const myGen = ++runGenRef.current;
    setStatus({ kind: 'running' });
    // Reset the cursor to the top while the worker call is in flight. This
    // makes the "Re-run" gesture visually concrete — the cards pop back to
    // layer 0 before the new replay starts.
    setActiveIdx(0);

    const worker = workerRef.current;
    if (!worker) {
      setStatus({ kind: 'error', error: 'Worker is unavailable. Reload the page.' });
      return;
    }

    runAbortRef.current?.abort();
    const ctrl = new AbortController();
    runAbortRef.current = ctrl;

    const appSignal = abortRef.current?.signal;
    if (appSignal?.aborted) ctrl.abort();
    const onAppAbort = () => ctrl.abort();
    if (appSignal && !appSignal.aborted) appSignal.addEventListener('abort', onAppAbort, { once: true });

    try {
      const result = await runForInspector(
        worker,
        {
          prompt,
          attention: true,
          // NOTE: we deliberately do NOT request `hiddenStates` here.
          // Requesting it routes every decode step through the
          // inspector-aware forward path (see qwen3_5/model.rs ~L5474),
          // which `.eval()`s per-layer stats mid-pass and serializes the
          // GPU pipeline. With `maxNewTokens > 1` that turns one fast
          // multi-token decode into N slow ones. We don't render
          // hidden-state stats in the stacked-layer diagram anyway, so
          // this is free perf. If a future tooltip needs L2 norms,
          // re-enable it AND drop maxNewTokens back to 1 for that run.
          logits: { topK: 12 },
          // Short decode budget. This diagram teaches next-token
          // prediction, so we reveal only the first few predicted tokens
          // (rainbow first + gray continuation, capped at
          // REVEAL_TOKEN_CAP). We measured greedy decoding on this model:
          // the first ~5 tokens are clean and on-topic ("The cat sat on
          // the" → " floor, and the cat"), but past ~5–6 tokens raw
          // greedy generation degenerates into verbatim repetition loops.
          // Generate a hair more than the reveal cap (8 vs 5) for echo
          // headroom; the UI never shows past REVEAL_TOKEN_CAP. The
          // forward-pass animation is keyed to the *prefill* trace
          // (captured once), so the decode budget only affects the reveal
          // text, not the per-layer visualization.
          maxNewTokens: 8,
          // Completion mode — do NOT wrap the prompt in the chat template.
          // The diagram visualizes "one forward pass → the next token",
          // i.e. text completion ("The cat sat on the" → " floor"), not a
          // chat assistant's reply. We A/B-tested both: the chat template
          // makes the model *reply* to the prompt instead of *continuing*
          // it, which muddies the next-token lesson. (Contrary to an
          // earlier note here, raw greedy does NOT emit junk tokens on
          // natural prompts — its only failure mode is the long-window
          // repetition handled by the short reveal cap above.)
          applyChatTemplate: false,
        },
        {
          signal: ctrl.signal,
          // Even with the hiddenStates capture dropped, generating 32
          // tokens with a chat-template-wrapped prompt + attention
          // capture can take 30–60s on the slower WASM/WebGPU paths.
          // 60s default is too tight; give ourselves real headroom.
          timeoutMs: 180_000,
        },
      );
      if (runGenRef.current !== myGen) return;
      setRun(result);
      // Reset to scene 0 and start replay. The effect below schedules the
      // per-scene transitions; here we just flip status into 'replaying'.
      setActiveIdx(0);
      if (reducedMotion) {
        // Skip the long replay — snap to done. Read the *future* scene
        // count from the captured run via a fresh buildScenes call so we
        // don't read stale `scenes` here.
        const set = new Set<number>();
        // Same filter as the memo above: only `kind === 'full'` counts.
        for (const a of result.attention ?? []) {
          if (a.kind === 'full') set.add(a.layerIndex);
        }
        for (const i of result.modelMeta?.fullAttentionLayerIndices ?? []) set.add(i);
        const finalScenes = buildScenes(set.size === 0 ? DEFAULT_FULL_ATTN_SET : set);
        setActiveIdx(finalScenes.length - 1);
        setStatus({ kind: 'done' });
      } else {
        setStatus({ kind: 'replaying' });
      }
    } catch (err) {
      if (runGenRef.current !== myGen) return;
      if (isAbortError(err)) {
        if (appSignal?.aborted) setStatus({ kind: 'aborted' });
        else setStatus({ kind: 'idle' });
      } else {
        const message = err instanceof Error ? err.message : String(err);
        setStatus({ kind: 'error', error: message });
      }
    } finally {
      if (appSignal) appSignal.removeEventListener('abort', onAppAbort);
      if (runAbortRef.current === ctrl) runAbortRef.current = null;
    }
  }, [prompt, workerRef, abortRef, reducedMotion]);

  // Auto-fire one run on mount — the route gates rendering on
  // status === 'ready' so the worker is already alive here.
  React.useEffect(() => {
    if (didAutoRunRef.current) return;
    didAutoRunRef.current = true;
    void handleRun();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Cleanup on unmount.
  React.useEffect(
    () => () => {
      runAbortRef.current?.abort();
    },
    [],
  );

  // -- replay timer -------------------------------------------------------
  //
  // We drive the scene cursor with a single scheduled timeout, restarted on
  // each scene transition. Pause stores the remaining time in the current
  // scene (so resume can schedule that remainder rather than a full dwell).
  // Re-running clears everything via the runGenRef/status flip.
  const sceneStartRef = React.useRef<number>(0); // performance.now() when the current scene began
  const sceneRemainingRef = React.useRef<number | null>(null); // ms remaining (used on resume)

  React.useEffect(() => {
    if (status.kind !== 'replaying') return undefined;
    const scene = scenes[activeIdx];
    if (!scene) return undefined;
    const fullDwell = scene.durationMs / speed;
    // Honour leftover time from a pause within the same scene. The pause
    // handler stores `sceneRemainingRef`; the resume handler leaves it set
    // so this effect picks it up. Any other transition (scene change,
    // re-run) clears it.
    const dwell = sceneRemainingRef.current != null ? sceneRemainingRef.current : fullDwell;
    sceneStartRef.current = performance.now();

    const isLast = activeIdx >= scenes.length - 1;
    const t = setTimeout(() => {
      sceneRemainingRef.current = null;
      if (isLast) {
        setStatus({ kind: 'done' });
      } else {
        setActiveIdx((i) => Math.min(scenes.length - 1, i + 1));
      }
    }, dwell);

    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeIdx, status.kind, speed, scenes]);

  // Clear any stashed remaining time when the scene changes (so the next
  // scene starts at its full dwell, not the previous scene's leftover).
  React.useEffect(() => {
    sceneRemainingRef.current = null;
  }, [activeIdx]);

  // Pause / resume.
  const pause = React.useCallback(() => {
    if (status.kind !== 'replaying') return;
    const elapsed = performance.now() - sceneStartRef.current;
    const scene = scenes[activeIdx];
    const baseDwell = scene ? scene.durationMs / speed : 0;
    sceneRemainingRef.current = Math.max(0, baseDwell - elapsed);
    setStatus({ kind: 'paused' });
  }, [status.kind, activeIdx, speed, scenes]);

  const resume = React.useCallback(() => {
    if (status.kind !== 'paused') return;
    setStatus({ kind: 'replaying' });
    // When status flips to 'replaying' the scheduling effect above sees the
    // current activeIdx and re-schedules. The stashed remaining time from
    // pause() is picked up automatically.
  }, [status.kind]);

  // Speed change while replaying: leave activeIdx where it is, the
  // scheduling effect re-fires because `speed` is a dep.
  const onSpeedClick = (s: Speed) => () => setSpeed(s);

  // Status pill content.
  const layerProgress = activeLayer ? activeLayer.index + 1 : 0;
  const statusText = (() => {
    switch (status.kind) {
      case 'idle':
        return 'Ready to run.';
      case 'running':
        return 'Running real inference…';
      case 'replaying':
        if (activeLayer) return `Replaying · layer ${layerProgress} / ${NUM_LAYERS}`;
        if (activeStage === 'tokenize') return 'Replaying · tokenize';
        if (activeStage === 'embedding') return 'Replaying · embedding';
        if (activeStage === 'final_norm') return 'Replaying · final RMSNorm';
        if (activeStage === 'lm_head') return 'Replaying · LM head';
        if (activeStage === 'sampling') return 'Replaying · sampling';
        return 'Replaying…';
      case 'paused':
        return activeLayer ? `Paused · layer ${layerProgress} / ${NUM_LAYERS}` : 'Paused';
      case 'done':
        return 'Done — token revealed below.';
      case 'aborted':
        return 'Run cancelled.';
      case 'empty-prompt':
        return 'Enter a prompt to run.';
      case 'error':
        return `Error: ${status.error}`;
    }
  })();

  const isBusy = status.kind === 'running' || status.kind === 'replaying' || status.kind === 'paused';
  // Consume the structured Rust response directly: no string-match dedup,
  // no regex stripping. The Rust inspector already:
  //   - breaks the decode at `<|im_end|>` / `<|endoftext|>`
  //     (qwen3_5/model.rs::run_for_inspector_sync builds stop_token_ids
  //      from the tokenizer and pops the stop token off the array)
  //   - reports `echoTokenCount` — the count of leading tokens whose
  //     cumulative text matches the prompt as a prefix. In completion
  //     mode (no chat template) the model continues the prompt rather
  //     than restating it, so this is ~always 0; we keep the slice so a
  //     rare restatement still renders muted.
  // The frontend slices the array, then caps the revealed window to
  // REVEAL_TOKEN_CAP tokens (raw greedy loops past ~5 — see handleRun).
  const allTokens = run?.generatedTokens ?? (run ? [run.generatedToken] : []);
  const echoCountFromRust = run?.echoTokenCount ?? 0;
  const echoTokenCount = Math.max(0, Math.min(echoCountFromRust, allTokens.length));
  const echoTokens = allTokens.slice(0, echoTokenCount);
  const newTokens = allTokens.slice(echoTokenCount, echoTokenCount + REVEAL_TOKEN_CAP);
  const newTokenCount = newTokens.length;
  // The rainbow "next token" highlight must come from the SAME capped,
  // post-echo window as the count (`newTokenCount`) and the rest-text
  // (`dedupedReplyDisplay`) — never from `run.generatedToken` directly. If the
  // continuation is empty (echo consumed the whole window, or a cached/compat
  // run reported `generatedTokens: []`), this is `undefined`, so the highlight,
  // the reveal text, and the count all agree on "nothing to show" rather than
  // the ghost rendering an echoed token while the count reads 0.
  const firstNewTokenInfo = newTokens[0];
  // The predicted continuation — the capped window of completion tokens.
  // This is what we show in the bottom reveal as the model's prediction.
  // We use `cleanupTokenText` (not `renderTokenDisplay`) to apply BPE cleanup
  // (Ġ → space, newline glyphs) WITHOUT the 17-char truncation that
  // `renderTokenDisplay` enforces for single-token chips.
  const dedupedReplyDisplay = run ? cleanupTokenText(newTokens.map((t) => t.text).join('')) : '';
  // The echoed prefix (for muted-tone rendering in the reveal). Empty
  // when there's no echo.
  const echoedPrefixDisplay =
    run && echoTokens.length > 0 ? cleanupTokenText(echoTokens.map((t) => t.text).join('')) : '';
  // For the FIRST-NEW-token highlight (rainbow chip in the ghost overlay
  // and the reveal). Truncated to 17 chars for single-token chip widths.
  const firstNewTokenText = firstNewTokenInfo ? renderTokenDisplay(firstNewTokenInfo.text) : '';
  // Full-length (untruncated) form, used as the anchor for slicing the
  // "rest" of the deduped reply in the bottom reveal.
  const firstNewTokenTextFull = firstNewTokenInfo ? cleanupTokenText(firstNewTokenInfo.text) : '';
  const showTokenReveal = (status.kind === 'replaying' && activeStage === 'sampling') || status.kind === 'done';

  return (
    <div className="mb-8 space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="font-mono text-xs uppercase tracking-[0.15em] text-muted-foreground">
          One forward pass, end to end — real model run
        </div>
        <div className="text-[11px] text-muted-foreground">
          The card on top is the current layer. Click any sub-stage to open its chapter.
        </div>
      </div>

      {/* Prompt + controls row */}
      <div className="space-y-2 rounded-md border border-dashed border-border bg-muted/20 px-3 py-2">
        <div className="flex flex-wrap items-end gap-2">
          <div className="min-w-[16rem] flex-1">
            <label className="block font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Prompt</label>
            {/*
              Ghost-prediction overlay — same pattern as chapter 4 (Attention)
              and chapter 5 (GQA). A sibling div sits behind the textarea,
              mirrors the prompt invisibly (text-transparent), then renders the
              actually-sampled next token at the end with the rainbow shimmer.
              The textarea has `bg-transparent` so the ghost shows through.

              Visibility gate:
              - `run` exists (we've completed at least one forward pass)
              - `prompt === run.prompt` (the prompt hasn't been edited since)
              - `status.kind === 'done'` (the *visible* replay has finished —
                even though the real inference completes in <1 s and we know
                the token long before, holding the reveal until the dot has
                walked all 24 layers and the sampling stage finishes prevents
                the "answer-before-the-work" optics where the predicted token
                appears at layer 19/24 and makes the diagram look fake).
            */}
            <div className="relative mt-1">
              <Textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={2}
                className="relative z-[1] bg-transparent font-mono text-sm"
                placeholder="The cat sat on the"
              />
              {run && prompt === run.prompt && status.kind === 'done' ? (
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-0 z-0 whitespace-pre-wrap break-words rounded-md border border-transparent px-3 py-2 font-mono text-sm text-transparent"
                >
                  {prompt}
                  {/* Multi-token ghost: the FIRST new token gets the rainbow
                      shimmer (drawing the learner's eye to "this is the model's
                      next prediction"), and the rest of the deduped reply is
                      rendered in muted gray so the learner can see the full
                      completion as a faint continuation — like a Copilot
                      ghost-text completion. The prompt itself stays invisible
                      (text-transparent on the wrapper) so it lines up under
                      the textarea text. */}
                  <span className="rainbow-text-animated font-mono text-sm">{firstNewTokenTextFull}</span>
                  {dedupedReplyDisplay.length > firstNewTokenTextFull.length ? (
                    <span className="font-mono text-sm text-muted-foreground/60">
                      {dedupedReplyDisplay.slice(firstNewTokenTextFull.length)}
                    </span>
                  ) : null}
                </div>
              ) : null}
            </div>
          </div>
          <div className="flex flex-col items-end gap-2">
            <RunButton
              onClick={handleRun}
              running={isBusy}
              label={status.kind === 'done' ? 'Re-run' : 'Run'}
              runningLabel={status.kind === 'running' ? 'Running…' : 'Replaying…'}
            />
            <div className="flex items-center gap-1">
              {status.kind === 'paused' ? (
                <Button size="sm" variant="outline" onClick={resume} className="h-7 px-2 text-xs">
                  Play
                </Button>
              ) : (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={pause}
                  disabled={status.kind !== 'replaying'}
                  className="h-7 px-2 text-xs"
                >
                  Pause
                </Button>
              )}
            </div>
            <div className="flex items-center gap-1" role="group" aria-label="Replay speed">
              <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Speed</span>
              {([0.5, 1, 2, 4] as const).map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={onSpeedClick(s)}
                  aria-pressed={speed === s}
                  className={[
                    'rounded border px-1.5 py-0.5 font-mono text-[11px]',
                    speed === s
                      ? 'border-primary/60 bg-primary/15 text-foreground'
                      : 'border-border bg-background text-muted-foreground hover:bg-muted/40',
                  ].join(' ')}
                >
                  {s === 0.5 ? '½×' : `${s}×`}
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-[11px]">
          <span
            role="status"
            aria-live="polite"
            className={[
              'inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 font-mono',
              status.kind === 'error'
                ? 'border-destructive/60 bg-destructive/10 text-destructive'
                : status.kind === 'empty-prompt'
                  ? 'border-amber-500/40 bg-amber-500/10 text-foreground'
                  : 'border-border bg-muted/40 text-foreground',
            ].join(' ')}
          >
            {(status.kind === 'running' || status.kind === 'replaying') && (
              <span aria-hidden="true" className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-primary" />
            )}
            {statusText}
          </span>
          <span className="text-muted-foreground">
            Real inference, real per-layer trace — replayed through a card stack at ~0.9–1.0 s/layer so you can watch
            each sub-stage fire. Total replay ~{Math.round(totalReplayMs / 1000)}s at 1× speed (½× to slow down for
            reading, 4× to skim).
          </span>
        </div>
      </div>

      <div ref={resultFlashRef} className="rounded-md">
        <StageStack
          activeStage={activeStage}
          activeLayer={activeLayer}
          fullAttentionLayerIndices={fullAttentionLayerIndices}
          onOpenChapter={onOpenChapter}
          firstNewTokenText={firstNewTokenText}
          firstNewTokenTextFull={firstNewTokenTextFull}
          dedupedReplyDisplay={dedupedReplyDisplay}
          echoedPrefixDisplay={echoedPrefixDisplay}
          newTokenCount={newTokenCount}
          showTokenReveal={showTokenReveal}
          reducedMotion={reducedMotion}
        />
      </div>

      <p className="text-xs text-muted-foreground">
        Watch the stack: full-attention layers (cyan) compute Q · Kᵀ · softmax over all past tokens; linear layers
        (amber, GatedDeltaNet) maintain a recurrent state — Qwen3.5 interleaves them 1:3.
      </p>

      <p className="text-xs text-muted-foreground">
        This is one turn of the loop. The model appends that next token and runs the whole stack again — but on the next
        pass it doesn&apos;t recompute the prefix: the{' '}
        <button
          type="button"
          onClick={() => onOpenChapter('kv-cache')}
          className="underline underline-offset-2 hover:text-foreground"
        >
          KV cache (chapter {chNum('kv-cache')})
        </button>{' '}
        is reused — the prefix isn&apos;t recomputed, so each new token is far cheaper than reprocessing the whole
        context. Generation is this forward pass, looped.
      </p>

      <p className="text-xs text-muted-foreground">
        Honest framing: this diagram shows <strong>greedy</strong> completion — it always takes the single highest-logit
        token, so the same prompt gives the same answer every time here. Real generation usually{' '}
        <button
          type="button"
          onClick={() => onOpenChapter('sampling')}
          className="underline underline-offset-2 hover:text-foreground"
        >
          samples
        </button>{' '}
        from the distribution, so an identical prompt can come out differently from run to run. The revealed
        continuation is also <strong>capped</strong> at {REVEAL_TOKEN_CAP} tokens — raw greedy decoding loops past that
        — so this is a window onto the first few predictions, not a full reply. See the{' '}
        <button
          type="button"
          onClick={() => onOpenChapter('architecture')}
          className="underline underline-offset-2 hover:text-foreground"
        >
          whole-model chapter
        </button>{' '}
        for the finished picture and its limits.
      </p>
    </div>
  );
}

// ----------------------------------------------------------------------------
// StageStack — left rail dot + card-stack viewport.
// ----------------------------------------------------------------------------

type StageStackProps = {
  activeStage: StageId;
  activeLayer: { kind: 'layer'; index: number; subId: string } | null;
  fullAttentionLayerIndices: Set<number>;
  onOpenChapter: (chapterId: string) => void;
  /** The first predicted token, used for the highlighted lead-in of the
   *  reveal. Truncated to 17 chars + ellipsis for chip width. */
  firstNewTokenText: string;
  /** Same first predicted token with BPE cleanup but NO length truncation;
   *  used as the slice anchor against `dedupedReplyDisplay` so the
   *  "rest" of the continuation is computed against matching transform
   *  rules. */
  firstNewTokenTextFull: string;
  /** The predicted continuation (capped reveal window, BPE-cleaned). In
   *  the rare case the completion restated the prompt, that echo prefix
   *  is stripped from this. */
  dedupedReplyDisplay: string;
  /** The portion of the completion that restated the prompt (BPE-cleaned).
   *  Rendered muted before the rainbow continuation. Empty in the normal
   *  completion case — the model continues rather than restates. */
  echoedPrefixDisplay: string;
  /** Number of predicted tokens shown (the reveal window, ≤ REVEAL_TOKEN_CAP). */
  newTokenCount: number;
  showTokenReveal: boolean;
  reducedMotion: boolean;
};

const FIXED_HEAD: { id: 'tokenize' | 'embedding'; label: string; chapterId: string; chapterNum: number }[] = [
  { id: 'tokenize', label: 'Tokenize', chapterId: 'tokenization', chapterNum: chNum('tokenization') },
  { id: 'embedding', label: 'Embedding lookup', chapterId: 'embeddings', chapterNum: chNum('embeddings') },
];

const FIXED_TAIL: {
  id: 'final_norm' | 'lm_head' | 'sampling';
  label: string;
  chapterId: string;
  chapterNum: number;
}[] = [
  { id: 'final_norm', label: 'Final RMSNorm', chapterId: 'rmsnorm', chapterNum: chNum('rmsnorm') },
  { id: 'lm_head', label: 'LM head → 248K logits', chapterId: 'lm-head', chapterNum: chNum('lm-head') },
  { id: 'sampling', label: 'Sampling → next token', chapterId: 'sampling', chapterNum: chNum('sampling') },
];

/** Card-stack viewport height in px. Tall enough for the front card body. */
const STACK_VIEWPORT_HEIGHT = 360;

function StageStack({
  activeStage,
  activeLayer,
  fullAttentionLayerIndices,
  onOpenChapter,
  firstNewTokenText,
  firstNewTokenTextFull,
  dedupedReplyDisplay,
  echoedPrefixDisplay,
  newTokenCount,
  showTokenReveal,
  reducedMotion,
}: StageStackProps) {
  const inLayerBlock = activeLayer !== null;
  const activeLayerIndex = activeLayer?.index ?? null;
  const activeSubId = activeLayer?.subId ?? null;

  // Dot y-position — five logical slots: tokenize, embedding, layer-block,
  // final_norm, lm_head, sampling. Each fixed chip has its own slot; the
  // layer block has ONE shared slot anchored to the front-card centre.
  const dotSlot: 'tokenize' | 'embedding' | 'layer' | 'final_norm' | 'lm_head' | 'sampling' = isLayerStage(activeStage)
    ? 'layer'
    : activeStage;

  return (
    <div className="rounded-md border border-border bg-background p-3">
      <div className="relative">
        {/* Left rail with the traveling dot. The rail spans head + stack +
            tail; the dot sits in a single y-slot per logical phase. */}
        <DotRail slot={dotSlot} activeSubId={activeSubId} reducedMotion={reducedMotion} />

        {/* Right column: head chips, card stack, tail chips. */}
        <div className="flex flex-col gap-1.5 pl-10">
          {FIXED_HEAD.map((s) => (
            <FixedChip
              key={s.id}
              id={s.id}
              label={s.label}
              chapterId={s.chapterId}
              chapterNum={s.chapterNum}
              active={s.id === activeStage}
              onOpenChapter={onOpenChapter}
            />
          ))}

          {/* Card-stack viewport — Apple-Wallet-style coordinated shuffle. */}
          <div className="mt-2 rounded-md border border-dashed border-emerald-500/40 bg-emerald-500/[0.04] p-2">
            <div className="mb-2 flex items-center justify-between px-1 text-[10px] font-mono tracking-wider text-emerald-700/80 dark:text-emerald-300/80">
              <span>
                × {NUM_LAYERS} LAYERS · chapter {chNum('full-block')} (full block)
              </span>
              <span
                className={[
                  'rounded-md border px-2 py-0.5 font-mono text-[11px] tracking-wider',
                  inLayerBlock
                    ? 'border-primary/60 bg-primary/10 text-foreground'
                    : 'border-border bg-muted/40 text-muted-foreground',
                ].join(' ')}
              >
                Layer {(activeLayerIndex ?? -1) + 1} / {NUM_LAYERS}
              </span>
            </div>

            <div
              className="relative w-full"
              style={{ height: STACK_VIEWPORT_HEIGHT, perspective: '1200px' }}
              data-stage-id="card-stack"
            >
              {Array.from({ length: NUM_LAYERS }, (_, i) => {
                const isFull = fullAttentionLayerIndices.has(i);
                // When the diagram is outside the layer block, anchor the
                // stack to layer 0 (head) or layer 23 (tail) so the cards
                // settle in a stable resting state.
                const stackAnchor =
                  activeLayerIndex ?? (activeStage === 'tokenize' || activeStage === 'embedding' ? 0 : NUM_LAYERS - 1);
                const rel = i - stackAnchor;
                return (
                  <LayerCard
                    key={i}
                    index={i}
                    isFull={isFull}
                    rel={rel}
                    activeSubId={inLayerBlock && rel === 0 ? activeSubId : null}
                    onOpenChapter={onOpenChapter}
                    reducedMotion={reducedMotion}
                  />
                );
              })}
            </div>
          </div>

          {FIXED_TAIL.map((s) => (
            <FixedChip
              key={s.id}
              id={s.id}
              label={s.label}
              chapterId={s.chapterId}
              chapterNum={s.chapterNum}
              active={s.id === activeStage}
              onOpenChapter={onOpenChapter}
            />
          ))}

          {/* Predicted-continuation reveal. Three layered pieces:
                1. Echoed prefix (muted) — only present in the rare case
                   the completion restates the prompt; greyed-out so it
                   stays legible without stealing focus. Empty in the
                   normal completion case.
                2. First predicted token highlighted in rainbow — the
                   "next token" this whole diagram is about.
                3. The next few predicted tokens in normal foreground
                   tone (the window is capped at REVEAL_TOKEN_CAP). */}
          <div
            className="mt-2 flex flex-col items-center gap-1 transition-opacity duration-300"
            style={{ opacity: showTokenReveal ? 1 : 0 }}
            aria-hidden={!showTokenReveal}
          >
            <span className="font-mono text-[11px] text-muted-foreground">next tokens →</span>
            <span
              className="max-w-[42rem] rounded-md border border-primary/60 bg-primary/10 px-3 py-1.5 text-center font-mono text-sm font-semibold text-foreground"
              data-stage-id="generated-reply"
            >
              {echoedPrefixDisplay ? <span className="text-muted-foreground/60">{echoedPrefixDisplay}</span> : null}
              <span className="rainbow-text-animated">{firstNewTokenText || '…'}</span>
              {newTokenCount > 1 && dedupedReplyDisplay !== firstNewTokenTextFull ? (
                <span className="text-foreground/80">{dedupedReplyDisplay.slice(firstNewTokenTextFull.length)}</span>
              ) : null}
            </span>
            <span className="font-mono text-[10px] text-muted-foreground/70">
              {echoedPrefixDisplay ? 'first new token highlighted · ' : 'first token highlighted · '}
              {newTokenCount} predicted token{newTokenCount === 1 ? '' : 's'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------------------
// DotRail — left-gutter rail with five logical y-slots.
// ----------------------------------------------------------------------------
//
// The dot's vertical position is driven by which logical phase we're in:
// the two head chips, the entire card-stack region (single anchored y),
// and the three tail chips. During the layer block the dot stays put and
// pulses with each sub-stage swap.

type DotRailProps = {
  slot: 'tokenize' | 'embedding' | 'layer' | 'final_norm' | 'lm_head' | 'sampling';
  activeSubId: string | null;
  reducedMotion: boolean;
};

// y offsets in px, measured from the rail's top. The layer slot anchors to
// the vertical centre of the card-stack viewport; the rest line up with the
// fixed chips' centres. These numbers are coupled to the chip heights +
// gap-1.5 (6 px) used in the column.
const SLOT_Y: Record<DotRailProps['slot'], number> = {
  tokenize: 16, // first chip centre (py-1.5 + line height ≈ 32 px tall)
  embedding: 54, // second chip centre
  layer: 92 + STACK_VIEWPORT_HEIGHT / 2 + 24, // mid of the viewport (after the in-viewport header bar)
  final_norm: 92 + STACK_VIEWPORT_HEIGHT + 50 + 16, // first tail chip centre
  lm_head: 92 + STACK_VIEWPORT_HEIGHT + 50 + 54,
  sampling: 92 + STACK_VIEWPORT_HEIGHT + 50 + 92,
};

function DotRail({ slot, activeSubId, reducedMotion }: DotRailProps) {
  const y = SLOT_Y[slot];
  // Pulse the dot on each sub-stage swap during the layer block. Using
  // `activeSubId` as a `key` triggers React to remount the inner pulse
  // element, restarting its CSS keyframe.
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none absolute left-3 top-0 z-10"
      style={{
        width: 0,
        height: 0,
        transform: `translateY(${y}px)`,
        transition: reducedMotion ? 'none' : 'transform 380ms cubic-bezier(0.4, 0, 0.2, 1)',
      }}
    >
      <span
        className="block rounded-full bg-primary/25 blur-[6px]"
        style={{ position: 'absolute', left: -12, top: -12, width: 24, height: 24 }}
      />
      <span
        key={slot === 'layer' ? (activeSubId ?? 'idle') : slot}
        className="block rounded-full bg-primary shadow-[0_0_6px_var(--primary)]"
        style={{
          position: 'absolute',
          left: -5,
          top: -5,
          width: 10,
          height: 10,
          animation: reducedMotion ? undefined : 'dotPulse 240ms ease-out',
        }}
      />
      <style>{`@keyframes dotPulse { 0% { transform: scale(1); } 50% { transform: scale(1.15); } 100% { transform: scale(1); } }`}</style>
    </div>
  );
}

// ----------------------------------------------------------------------------
// FixedChip — the small chips used for tokenize / embedding / norm / head /
// sampling above and below the card stack.
// ----------------------------------------------------------------------------

function FixedChip({
  id,
  label,
  chapterId,
  chapterNum,
  active,
  onOpenChapter,
}: {
  id: StageId & string;
  label: string;
  chapterId: string;
  chapterNum: number;
  active: boolean;
  onOpenChapter: (chapterId: string) => void;
}) {
  return (
    <div
      data-stage-id={id}
      role="button"
      tabIndex={0}
      aria-label={`Open chapter ${chapterNum}: ${label}`}
      onClick={() => onOpenChapter(chapterId)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onOpenChapter(chapterId);
        }
      }}
      className={[
        'flex cursor-pointer items-center justify-between rounded-md border px-3 py-1.5 text-sm transition-colors hover:bg-muted/30',
        active ? 'border-primary bg-primary/10 text-foreground' : 'border-border bg-background text-foreground',
      ].join(' ')}
      style={{ outline: 'none' }}
    >
      <span className={active ? 'font-medium' : ''}>{label}</span>
      <span
        className={[
          'rounded-md border px-1.5 py-0.5 font-mono text-[10px]',
          active
            ? 'border-primary/60 bg-primary/10 text-foreground'
            : 'border-border bg-muted/40 text-muted-foreground',
        ].join(' ')}
      >
        ch {chapterNum}
      </span>
    </div>
  );
}

// ----------------------------------------------------------------------------
// LayerCard — one of the 24 absolutely-positioned cards in the stack.
// ----------------------------------------------------------------------------
//
// `rel` is `index - activeLayer` (or the anchor for non-layer scenes). The
// card translates / scales / fades in lock-step with every other card on
// each `rel` change → Apple-Wallet shuffle effect.

type LayerCardProps = {
  index: number;
  isFull: boolean;
  rel: number;
  /** Active sub-stage id when this card is the front (`rel === 0`); null
   *  otherwise so peeking cards don't render a highlight bar. */
  activeSubId: string | null;
  onOpenChapter: (chapterId: string) => void;
  reducedMotion: boolean;
};

function LayerCard({ index, isFull, rel, activeSubId, onOpenChapter, reducedMotion }: LayerCardProps) {
  // Map `rel` to a transform / opacity slot. Cards beyond ±3 fade away.
  const { translateY, scale, opacity, zIndex } = (() => {
    if (rel === 0) return { translateY: 0, scale: 1, opacity: 1, zIndex: 30 };
    if (rel === 1) return { translateY: 50, scale: 0.94, opacity: 0.55, zIndex: 25 };
    if (rel === 2) return { translateY: 85, scale: 0.9, opacity: 0.32, zIndex: 20 };
    if (rel === 3) return { translateY: 110, scale: 0.87, opacity: 0.18, zIndex: 15 };
    if (rel >= 4) return { translateY: 160, scale: 0.85, opacity: 0, zIndex: 10 };
    if (rel === -1) return { translateY: -220, scale: 0.96, opacity: 0, zIndex: 5 };
    // rel <= -2 — history slides further off the top.
    return { translateY: -260, scale: 0.95, opacity: 0, zIndex: 1 };
  })();

  const cardState = rel === 0 ? 'front' : rel > 0 ? 'peek' : 'history';
  const chapterId = isFull ? 'attention' : 'kv-cache';
  const chapterNum = chNum(chapterId);
  const subs = isFull ? FULL_ATTN_SUBS : LINEAR_ATTN_SUBS;

  return (
    <div
      data-layer-index={index}
      data-card-state={cardState}
      data-stage-id={`layer:${index}`}
      className={[
        'absolute left-2 right-2 rounded-lg border bg-background/95 backdrop-blur-sm',
        isFull
          ? 'border-cyan-500/40 ring-1 ring-cyan-500/30 dark:border-cyan-400/40 dark:ring-cyan-400/30'
          : 'border-amber-500/40 ring-1 ring-amber-500/30 dark:border-amber-400/40 dark:ring-amber-400/30',
        rel >= 4 || rel <= -1 ? 'pointer-events-none' : '',
      ].join(' ')}
      style={{
        top: 0,
        transform: `translate3d(0, ${translateY}px, 0) scale(${scale})`,
        transformOrigin: 'top center',
        opacity,
        zIndex,
        transition: reducedMotion ? 'none' : 'transform 380ms cubic-bezier(0.34, 1.4, 0.64, 1), opacity 320ms ease-out',
        boxShadow: rel === 0 ? '0 12px 32px -8px rgba(0,0,0,0.25)' : '0 4px 12px -4px rgba(0,0,0,0.15)',
      }}
    >
      {/* Card header — always visible even when peeking. */}
      <div className="flex items-center justify-between gap-2 border-b border-border/50 px-3 py-2">
        <span className="flex items-center gap-2">
          <span className="font-mono tabular-nums text-sm font-semibold text-foreground">
            #{index.toString().padStart(2, '0')}
          </span>
          <span
            className={[
              'rounded border px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider',
              isFull
                ? 'border-cyan-500/50 bg-cyan-500/10 text-cyan-700 dark:text-cyan-300'
                : 'border-amber-500/50 bg-amber-500/10 text-amber-700 dark:text-amber-300',
            ].join(' ')}
            title={isFull ? 'Full softmax attention layer' : 'Linear-attention layer (GatedDeltaNet)'}
          >
            {isFull ? 'FULL ATTN' : 'LINEAR · GDN'}
          </span>
          <span className="text-xs text-muted-foreground">decoder layer</span>
        </span>
        <button
          type="button"
          onClick={() => onOpenChapter(chapterId)}
          className="rounded-md border border-border bg-muted/40 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground hover:bg-muted/60"
        >
          ch {chapterNum}
        </button>
      </div>

      {/* Card body — sub-stage list. Visually meaningful on the front card;
          gets clipped under cards above for peeks. */}
      <div className="space-y-0.5 px-2 py-2">
        {subs.map((sub) => {
          const isActive = activeSubId === sub.id;
          const subChapter = SUB_CHAPTER[sub.id];
          return (
            <div
              key={sub.id}
              data-stage-id={`layer:${index}:${sub.id}`}
              role={subChapter ? 'button' : undefined}
              tabIndex={subChapter && rel === 0 ? 0 : -1}
              onClick={subChapter && rel === 0 ? () => onOpenChapter(subChapter) : undefined}
              onKeyDown={(e) => {
                if (!subChapter || rel !== 0) return;
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onOpenChapter(subChapter);
                }
              }}
              className={[
                'flex items-center gap-2 rounded-sm border-l-2 px-2 py-1 font-mono text-[11px]',
                reducedMotion ? '' : 'transition-colors duration-200',
                isActive ? 'border-primary bg-primary/15 text-foreground' : 'border-transparent text-muted-foreground',
                subChapter && rel === 0 ? 'cursor-pointer hover:bg-muted/30' : '',
              ].join(' ')}
            >
              <span aria-hidden="true" className="text-muted-foreground/40">
                └─
              </span>
              <span>{sub.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

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

function ChapterCard({ chapter, onOpen }: { chapter: ChapterMeta; onOpen: () => void }) {
  const interactive = chapter.available;
  return (
    <Card
      onClick={interactive ? onOpen : undefined}
      role={interactive ? 'button' : undefined}
      tabIndex={interactive ? 0 : -1}
      onKeyDown={(e) => {
        if (!interactive) return;
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onOpen();
        }
      }}
      className={
        interactive ? 'cursor-pointer transition-colors hover:bg-card/80 hover:border-primary/40' : 'opacity-60'
      }
      aria-disabled={!interactive}
    >
      <CardHeader>
        <div className="mb-1 flex items-center justify-between">
          <span className="font-mono text-xs text-muted-foreground">
            Ch. {chapter.number.toString().padStart(2, '0')}
          </span>
          {chapter.available ? <Badge variant="default">Ready</Badge> : <Badge variant="secondary">Coming soon</Badge>}
        </div>
        <CardTitle className="text-lg">{chapter.title}</CardTitle>
        <CardDescription>{chapter.blurb}</CardDescription>
      </CardHeader>
      <CardContent>
        <span className={interactive ? 'text-sm text-primary' : 'text-sm text-muted-foreground'}>
          {interactive ? 'Open chapter →' : 'Not yet authored'}
        </span>
      </CardContent>
    </Card>
  );
}
