import { Loader2Icon } from 'lucide-react';
import * as React from 'react';

import type { TokenInfo } from '../../../../src/inspector-types';
import type { Locale } from '../../../lib/i18n';
import { useLocale } from '../../../lib/i18n-react';
import { lensReadout } from '../../../lib/lens-client';
import { tokenize } from '../../../lib/tokenizer-client';
import { useFreeChatOptional } from '../../../providers/free-chat';
import { useModelLoaderOptional } from '../../../providers/model-loader';
import { renderTokenDisplay } from '../../inspector/TopKBars';
import { RunButton } from '../../scaffolding/RunButton';
import { SegmentedToggle } from '../../scaffolding/SegmentedToggle';
import { ArgmaxGrid, type CellRef } from './ArgmaxGrid';
import { pinColor } from './colors';
import { LensTooltip } from './LensTooltip';
import { PinChips } from './PinChips';
import { RankHeatmap } from './RankHeatmap';
import { buildLensSlice, type LensSliceData } from './types';

/**
 * LogitLensLive — the interpretability chapter's LIVE demo: the LOGIT LENS on
 * the actual in-browser Qwen3.5-0.8B. One forward pass reads the model's
 * vocabulary out of every intermediate residual boundary `h_ℓ` through the tied
 * unembedding `W_U` (`lensReadout`, useJacobian:false). The reader picks a short
 * prompt, runs it, and watches the per-layer argmax token go from gibberish in
 * the middle layers to the real answer in the late layers, while a pinned
 * answer token climbs the full-vocab ranks only near the top.
 *
 * This is the plain logit lens — no fitted Jacobian (the browser ships no lens
 * pack), which the UI states honestly. A later "Jacobian lens" chapter is what
 * sharpens those blurry middle layers; this widget shows the blind spot the
 * Jacobian lens fixes, it does not run it.
 *
 * PRERENDER SAFETY: rendered inside SSG-prerendered section HTML with no app
 * providers mounted. It reads worker/model state ONLY through the non-throwing
 * `useFreeChatOptional()` / `useModelLoaderOptional()` hooks and renders a
 * static frame when they return null. No window/document on the render path;
 * all live work happens in handlers/effects (never run under renderToString).
 */

// ---------------------------------------------------------------------------
// Constants + presets
// ---------------------------------------------------------------------------

// Curated boundary subset: span the 25-boundary stack without reading all of
// it — keeps lensReadout fast and the grid legible.
const LAYERS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24];
const TOP_K = 5;

type PresetDef = {
  id: 'france' | 'opposite' | 'sky';
  /** Raw prompt text — tokenized with NO chat template. Kept short (≤ ~12 tokens). */
  prompt: string;
  /** Answer/concept tokens to rank-track. Each is tokenized as " <concept>"
   *  (leading space) and its FIRST token id is pinned. */
  concepts: string[];
};

const PRESETS: readonly PresetDef[] = [
  { id: 'france', prompt: 'The capital of France is', concepts: ['Paris', 'London'] },
  { id: 'opposite', prompt: 'The opposite of hot is', concepts: ['cold', 'warm'] },
  { id: 'sky', prompt: 'The sky is', concepts: ['blue', 'grey'] },
];

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

// Same idiom as SpeculativeVerifyLive and the sibling widgets: matchMedia read
// in the useState initializer is the deliberate course convention — not a bug.
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
// Per-locale copy. Glossary terms (logit lens, token, layer, rank, argmax,
// top-K, Jacobian, Qwen3.5, unembedding) stay English per the course convention.
// ---------------------------------------------------------------------------

const COPY = {
  en: {
    title: 'Logit lens — reading the vocabulary out of every layer',
    liveBadge: 'runs in your browser',
    lensBadge: 'logit lens · Jacobian off',
    staticCaption: 'Interactive demo — loads after the page boots.',
    staticIntro:
      'Read the model’s vocabulary straight out of a middle residual layer and you mostly get gibberish; only the late layers surface the real answer. Run it live on Qwen3.5-0.8B and watch the per-layer top token resolve with depth.',
    loadCta: 'Load Qwen3.5-0.8B (~1.6 GB) to run this live',
    loadNote: 'Runs entirely on your device; nothing leaves your browser.',
    loadingLabel: 'Loading Qwen3.5-0.8B…',
    errorPrefix: 'Lens read failed.',
    presetLabel: 'Prompt',
    presetNames: {
      france: 'Capital of France',
      opposite: 'Opposite of hot',
      sky: 'The sky is',
    } as Record<PresetDef['id'], string>,
    promptLabel: 'Prompt (raw text — no chat template)',
    pinnedLabel: 'Answer tokens to rank-track',
    runCta: 'Run the logit lens',
    runningLabel: 'Reading layers…',
    emptyHint: 'Run the lens to read every layer’s top token for this prompt.',
    gridTitle: 'Per-layer top token (deepest at the top)',
    gridAria: (nLayers: number, promptLen: number) =>
      `Logit-lens argmax grid: ${nLayers} residual boundaries as rows, deepest at the top, by ${promptLen} prompt positions as columns. Each cell is the model’s top token read at that depth through the final unembedding; the highlighted final column is the next-token prediction, which sharpens into the real answer only in the late layers.`,
    finalColNote: 'The highlighted final column is the next-token prediction — read it top-to-bottom to see depth resolve the answer.',
    heatmapTitle: 'Answer-token rank by depth (final position)',
    heatmapAria: (nPins: number) =>
      `Rank heatmap: ${nPins} pinned answer tokens as columns, residual boundaries as rows with the deepest at the top. Brighter cells mean a better (lower) full-vocab rank at the final position; the correct answer brightens only near the top of the stack.`,
    legendLabel: 'rank →',
    legendBright: 'rank 1',
    legendDark: '999+',
    tooltipHeader: (layer: number, pos: number, tokenText: string) =>
      `ℓ${layer} · position ${pos + 1} · after "${tokenText}"`,
    tooltipProbLabel: 'full-vocab probability',
    tooltipHint: 'Hover or tap any cell to see that layer’s top-K read.',
    pinsAria: 'Rank-tracked answer tokens',
    pinBestRank: (n: number) => `best rank ${n}`,
    pinSurfaces: (layer: number) => `surfaces at ℓ${layer}`,
    pinNeverSurfaces: (k: number) => `never enters top-${k}`,
    pinPartial: 'concept split into multiple tokens — only its first token is tracked',
    jacobianTease:
      'Those blurry middle layers are the logit lens’s blind spot — a later Jacobian lens is what sharpens them. This widget shows the blind spot; it does not run the Jacobian lens.',
    footnote:
      'The logit lens reads each intermediate residual `h_ℓ` through the model’s final unembedding W_U — an off-distribution shortcut, so middle-layer reads are approximate. Probabilities are honest full-vocab softmax; ranks are 1-based over the whole vocab, capped at 999.',
  },
  zh: {
    title: 'Logit lens——从每一层里读出词表',
    liveBadge: '在你的浏览器里运行',
    lensBadge: 'logit lens · 不用 Jacobian',
    staticCaption: '互动演示——页面启动后加载。',
    staticIntro:
      '直接从中间的 residual layer 把模型的词表读出来，多半是乱码；只有靠后的 layer 才会浮现真正的答案。在 Qwen3.5-0.8B 上实时运行，看每一层的 top token 如何随深度逐渐收敛。',
    loadCta: '加载 Qwen3.5-0.8B（约 1.6 GB）来实时运行',
    loadNote: '完全在你的设备上运行，数据不会离开浏览器。',
    loadingLabel: '正在加载 Qwen3.5-0.8B……',
    errorPrefix: 'Lens 读取失败。',
    presetLabel: '提示',
    presetNames: {
      france: '法国的首都',
      opposite: 'hot 的反义词',
      sky: '天空是',
    } as Record<PresetDef['id'], string>,
    promptLabel: '提示（原始文本——不套 chat template）',
    pinnedLabel: '要追踪 rank 的答案 token',
    runCta: '运行 logit lens',
    runningLabel: '正在逐层读取……',
    emptyHint: '运行 lens，读出这个提示在每一层的 top token。',
    gridTitle: '每一层的 top token（最深的在最上面）',
    gridAria: (nLayers: number, promptLen: number) =>
      `Logit-lens argmax 网格：${nLayers} 个 residual 边界作为行（最深的在最上面），${promptLen} 个提示位置作为列。每个格子是在该深度经过最终 unembedding 读出的 top token；高亮的最后一列是下一个 token 的预测，只有在靠后的 layer 才会收敛到真正的答案。`,
    finalColNote: '高亮的最后一列是下一个 token 的预测——从上往下读，就能看到深度如何把答案解出来。',
    heatmapTitle: '答案 token 的 rank 随深度变化（最后一个位置）',
    heatmapAria: (nPins: number) =>
      `Rank 热力图：${nPins} 个被 pin 的答案 token 作为列，residual 边界作为行（最深的在最上面）。格子越亮表示在最后位置的全词表 rank 越好（越低）；正确答案只有在最上面几层才会变亮。`,
    legendLabel: 'rank →',
    legendBright: 'rank 1',
    legendDark: '999+',
    tooltipHeader: (layer: number, pos: number, tokenText: string) =>
      `ℓ${layer} · 位置 ${pos + 1} · 在 "${tokenText}" 之后`,
    tooltipProbLabel: '全词表概率',
    tooltipHint: '悬停或点按任意格子，查看该层的 top-K 读数。',
    pinsAria: '被追踪 rank 的答案 token',
    pinBestRank: (n: number) => `最好 rank ${n}`,
    pinSurfaces: (layer: number) => `在 ℓ${layer} 浮现`,
    pinNeverSurfaces: (k: number) => `始终进不了 top-${k}`,
    pinPartial: '这个概念被切成了多个 token——只追踪它的第一个 token',
    jacobianTease:
      '中间那些模糊的 layer 正是 logit lens 的盲区——后面的 Jacobian lens 才能把它们锐化。这个 widget 展示的是盲区，并不运行 Jacobian lens。',
    footnote:
      'Logit lens 把每个中间 residual `h_ℓ` 直接经过模型最终的 unembedding W_U 读出——这是一条离分布的捷径，所以中间层的读数只是近似。概率是诚实的全词表 softmax；rank 是全词表 1-based，上限 999。',
  },
} as const;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

type RunState =
  | { kind: 'idle' }
  | { kind: 'running' }
  | { kind: 'done'; slice: LensSliceData; partialFlags: boolean[] }
  | { kind: 'error'; message: string };

// ---------------------------------------------------------------------------
// The widget
// ---------------------------------------------------------------------------

export function LogitLensLive() {
  const locale: Locale = useLocale();
  const copy = COPY[locale];
  const reducedMotion = usePrefersReducedMotion();

  // Optional (non-throwing) provider hooks — null under SSG prerender.
  const freeChat = useFreeChatOptional();
  const loader = useModelLoaderOptional();

  const [presetId, setPresetId] = React.useState<PresetDef['id']>(PRESETS[0]!.id);
  const preset = PRESETS.find((p) => p.id === presetId) ?? PRESETS[0]!;

  const [run, setRun] = React.useState<RunState>({ kind: 'idle' });
  const [selected, setSelected] = React.useState<CellRef | null>(null);
  const [hovered, setHovered] = React.useState<CellRef | null>(null);
  const [runKey, setRunKey] = React.useState(0);

  // Monotonic generation guard + per-run AbortController + unmount cleanup.
  const runGenRef = React.useRef(0);
  const runAbortRef = React.useRef<AbortController | null>(null);
  React.useEffect(
    () => () => {
      runGenRef.current++;
      runAbortRef.current?.abort();
    },
    [],
  );

  const ready = loader?.status === 'ready';
  const workerRef = freeChat?.mlxWorkerRef ?? null;
  const appAbortRef = freeChat?.inspectorAbortRef ?? null;

  // -------------------------------------------------------------------------
  // The one live path: tokenize the prompt + concept pins (raw, no chat
  // template), then a single lensReadout forward pass over the prompt.
  // -------------------------------------------------------------------------
  async function handleRun() {
    const worker = workerRef?.current ?? null;
    if (!worker) {
      setRun({ kind: 'error', message: 'MLX worker is not available' });
      return;
    }
    const myGen = ++runGenRef.current;
    setRun({ kind: 'running' });
    setSelected(null);
    setHovered(null);

    runAbortRef.current?.abort();
    const ctrl = new AbortController();
    runAbortRef.current = ctrl;
    const appSignal = appAbortRef?.current?.signal;
    const onAppAbort = () => ctrl.abort();
    if (appSignal?.aborted) ctrl.abort();
    else appSignal?.addEventListener('abort', onAppAbort, { once: true });

    try {
      const promptTokens: TokenInfo[] = await tokenize(worker, preset.prompt, { signal: ctrl.signal });
      if (runGenRef.current !== myGen) return;
      const promptIds = promptTokens.map((t) => t.id);
      if (promptIds.length === 0) throw new Error('prompt tokenized to zero tokens');

      // Pin each concept by its first token (leading space → mid-sentence form).
      const pinnedIds: number[] = [];
      const partialFlags: boolean[] = [];
      for (const concept of preset.concepts) {
        const conceptTokens = await tokenize(worker, ` ${concept}`, { signal: ctrl.signal });
        if (runGenRef.current !== myGen) return;
        if (conceptTokens.length === 0) continue;
        pinnedIds.push(conceptTokens[0]!.id);
        partialFlags.push(conceptTokens.length > 1);
      }

      const result = await lensReadout(
        worker,
        { promptIds, layers: LAYERS, topK: TOP_K, pinnedIds, useJacobian: false },
        { signal: ctrl.signal },
      );
      if (runGenRef.current !== myGen) return;

      const slice = buildLensSlice(result);
      setRunKey((k) => k + 1);
      setRun({ kind: 'done', slice, partialFlags });
      // Default the tooltip to the output boundary at the final position.
      setSelected({ layerIdx: slice.layers.length - 1, pos: slice.promptLen - 1 });
    } catch (err) {
      if (runGenRef.current !== myGen) return;
      if (isAbortError(err)) setRun({ kind: 'idle' });
      else setRun({ kind: 'error', message: err instanceof Error ? err.message : String(err) });
    } finally {
      appSignal?.removeEventListener('abort', onAppAbort);
      if (runAbortRef.current === ctrl) runAbortRef.current = null;
    }
  }

  function pickPreset(id: PresetDef['id']) {
    setPresetId(id);
    setRun({ kind: 'idle' });
    setSelected(null);
    setHovered(null);
  }

  // -------------------------------------------------------------------------
  // Render helpers
  // -------------------------------------------------------------------------

  const frame = (children: React.ReactNode) => (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="inline-flex items-center gap-1.5 rounded border border-border bg-muted/40 px-2 py-0.5 text-[11px] text-muted-foreground">
            {copy.lensBadge}
          </span>
          <span className="inline-flex items-center gap-1.5 rounded border border-primary/40 bg-primary/10 px-2 py-0.5 text-[11px] text-primary">
            {copy.liveBadge}
          </span>
        </div>
      </div>
      {children}
    </div>
  );

  const promptBlock = (promptText: string) => (
    <div className="space-y-1">
      <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.promptLabel}</div>
      <div className="overflow-x-auto rounded-md border border-border bg-muted/40 p-2 font-mono text-[12px] text-foreground/80">
        {promptText}
      </div>
    </div>
  );

  const conceptPreview = (concepts: string[]) => (
    <div className="space-y-1">
      <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.pinnedLabel}</div>
      <div className="flex flex-wrap items-center gap-2">
        {concepts.map((c, i) => {
          const color = pinColor(i);
          return (
            <span
              key={c}
              className="inline-flex items-center gap-1 rounded-md border px-2 py-1 font-mono text-[12px] font-semibold"
              style={{ borderColor: color, background: `${color}1a`, color }}
            >
              {renderTokenDisplay(` ${c}`)}
            </span>
          );
        })}
      </div>
    </div>
  );

  const staticPreview = (
    <>
      {promptBlock(PRESETS[0]!.prompt)}
      {conceptPreview(PRESETS[0]!.concepts)}
      <p className="text-[12px] text-muted-foreground">{copy.staticIntro}</p>
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
  const done = run.kind === 'done' ? run : null;
  const running = run.kind === 'running';
  const active = hovered ?? selected;

  // Per-pin accents + argmax-match colour map, from the completed run.
  const pinTexts = done ? done.slice.pinned.map((p) => p.tokenText) : [];
  const pinColors = done ? done.slice.pinned.map((_p, i) => pinColor(i)) : [];
  const colorByPinnedId = new Map<number, string>();
  if (done) done.slice.pinned.forEach((p, i) => colorByPinnedId.set(p.tokenId, pinColor(i)));
  const finalPos = done ? done.slice.promptLen - 1 : 0;

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

      {promptBlock(preset.prompt)}
      {conceptPreview(preset.concepts)}

      {/* Run */}
      <div className="flex flex-wrap items-center gap-2">
        <RunButton
          onClick={() => void handleRun()}
          running={running}
          label={copy.runCta}
          runningLabel={copy.runningLabel}
        />
      </div>

      {/* Error — inline, muted, non-blocking. */}
      {run.kind === 'error' ? (
        <p className="text-[12px] text-muted-foreground" role="alert">
          <strong>{copy.errorPrefix}</strong> {run.message}
        </p>
      ) : null}

      {done ? (
        <div className={['space-y-4 border-t border-border/60 pt-3', reducedMotion ? '' : 'transition-all'].join(' ')}>
          {/* Argmax grid */}
          <div className="space-y-1.5">
            <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.gridTitle}</div>
            <ArgmaxGrid
              slice={done.slice}
              colorByPinnedId={colorByPinnedId}
              selected={selected}
              onHover={setHovered}
              onSelect={setSelected}
              ariaLabel={copy.gridAria(done.slice.layers.length, done.slice.promptLen)}
            />
            <p className="text-[11px] text-muted-foreground">{copy.finalColNote}</p>
          </div>

          {/* Per-cell tooltip */}
          {active ? (
            <LensTooltip
              slice={done.slice}
              layerIdx={active.layerIdx}
              pos={active.pos}
              runKey={runKey}
              header={copy.tooltipHeader(
                done.slice.layers[active.layerIdx]!,
                active.pos,
                renderTokenDisplay(done.slice.tokens[active.pos]?.text ?? ''),
              )}
              probLabel={copy.tooltipProbLabel}
            />
          ) : (
            <p className="text-[11px] text-muted-foreground">{copy.tooltipHint}</p>
          )}

          {/* Pinned answer-token rank tracks */}
          {done.slice.pinned.length > 0 ? (
            <div className="space-y-2">
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.heatmapTitle}</div>
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:gap-6">
                <RankHeatmap
                  slice={done.slice}
                  finalPos={finalPos}
                  pinTexts={pinTexts}
                  pinColors={pinColors}
                  ariaLabel={copy.heatmapAria(done.slice.pinned.length)}
                  legendLabel={copy.legendLabel}
                  brightLabel={copy.legendBright}
                  darkLabel={copy.legendDark}
                />
                <PinChips
                  slice={done.slice}
                  finalPos={finalPos}
                  pinTexts={pinTexts}
                  pinColors={pinColors}
                  partialFlags={done.partialFlags}
                  copy={{
                    chipsAria: copy.pinsAria,
                    bestRank: copy.pinBestRank,
                    surfaces: copy.pinSurfaces,
                    neverSurfaces: copy.pinNeverSurfaces,
                    partial: copy.pinPartial,
                  }}
                />
              </div>
            </div>
          ) : null}
        </div>
      ) : running ? null : (
        <p className="text-[12px] text-muted-foreground">{copy.emptyHint}</p>
      )}

      <p className="text-[11px] text-muted-foreground">{copy.jacobianTease}</p>
      <p className="text-[10px] text-muted-foreground/70">{copy.footnote}</p>
    </>,
  );
}

export default LogitLensLive;
