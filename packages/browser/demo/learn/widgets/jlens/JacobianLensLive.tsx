import { Loader2Icon } from 'lucide-react';
import * as React from 'react';

import type { TokenInfo } from '../../../../src/inspector-types';
import type { Locale } from '../../../lib/i18n';
import { useLocale } from '../../../lib/i18n-react';
import { lensReadout, loadLensPack } from '../../../lib/lens-client';
import { tokenize } from '../../../lib/tokenizer-client';
import { useFreeChatOptional } from '../../../providers/free-chat';
import { useModelLoaderOptional } from '../../../providers/model-loader';
import { renderTokenDisplay } from '../../inspector/TopKBars';
import { RunButton } from '../../scaffolding/RunButton';
import { SegmentedToggle } from '../../scaffolding/SegmentedToggle';
import { ArgmaxGrid, type CellRef } from './ArgmaxGrid';
import { BAKED } from './baked';
import { pinColor } from './colors';
import { BAND, JACOBIAN_LAYERS, JACOBIAN_PRESETS, type JacobianPreset } from './jacobian-presets';
import { LensTooltip } from './LensTooltip';
import { PinChips } from './PinChips';
import { RankHeatmap } from './RankHeatmap';
import { reviveRun } from './revive';
import { buildLensSlice, type LensSliceData } from './types';

/**
 * JacobianLensLive — the fitted J-lens demo for the interpretability chapter.
 *
 * The 3 committed `baked/<slug>.json` frames carry BOTH a logit and a jacobian
 * `LensReadoutRun` per curated prompt, so the DEFAULT experience needs NO model
 * and NO lens pack: on load (including SSG prerender) the widget revives the
 * baked jacobian frame for the headline preset and renders it through the SAME
 * jlens children as the live logit-lens widget. The LOGIT | JACOBIAN toggle
 * flips between that preset's baked logit vs jacobian run instantly — the whole
 * lesson (what the fitted Jacobian buys mid-band) is taught with zero downloads.
 *
 * "Compute live on your device" is the OPTIONAL prove-it path: it recomputes the
 * current preset live on the in-browser Qwen3.5-0.8B (existing ~1.6 GB model
 * gate) and, for JACOBIAN mode only, one-time loads the fitted pack
 * (`loadLensPack`, +~46 MB). Most readers never trigger either download.
 *
 * PRERENDER SAFETY: the baked frame renders with `freeChat === null &&
 * loader === null`. Provider state is read ONLY through the non-throwing
 * `useFreeChatOptional()` / `useModelLoaderOptional()` hooks; no window/document
 * on the render path; all live work is in handlers/effects.
 *
 * This widget does NOT modify LogitLensLive or the logit-lens page — it reuses
 * the jlens children (ArgmaxGrid / RankHeatmap / PinChips / LensTooltip) and the
 * single `LensSliceData` view-model unchanged.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Live recompute matches the baked frames: same top-K as scripts/jlens/bake.mts.
const TOP_K = 10;
// LIVE-only "show all layers" axis: the full residual boundary axis 1..24 (the
// baked frame stays on the curated JACOBIAN_LAYERS subset). Reveals the noisy
// early band (1..5) the default view intentionally hides.
const ALL_LAYERS: number[] = [...BAND.boundaries];

type LensMode = 'logit' | 'jacobian';

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

// matchMedia read in the useState initializer is the deliberate course
// convention (mirrors LogitLensLive.tsx:76-88) — not a bug.
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
// Per-locale copy. Glossary terms (logit lens, Jacobian, token, layer, rank,
// argmax, top-K, W_U, Qwen3.5, unembedding) stay English per the course
// convention. Per-preset `blurb` copy lives in jacobian-presets.ts.
// ---------------------------------------------------------------------------

const COPY = {
  en: {
    title: 'Jacobian lens — the same layers, sharpened',
    sourceBaked: 'precomputed offline · fitted on this exact Qwen3.5-0.8B',
    sourceLive: 'computed live on your device',
    lensLogit: 'logit lens · Jacobian off',
    lensJacobianApplied: 'Jacobian lens · fitted J applied',
    lensJacobianIdentity: 'Jacobian requested · identity only',
    modeLabel: 'Lens',
    modeAria: 'Lens: logit lens or Jacobian lens',
    modeLogit: 'LOGIT',
    modeJacobian: 'JACOBIAN',
    presetLabel: 'Prompt',
    presetNames: {
      'french-season': 'French season',
      'spanish-opposite': 'Spanish opposite',
      'arithmetic-parens': 'Arithmetic',
    } as Record<string, string>,
    promptLabel: 'Prompt (raw text — no chat template)',
    pinnedLabel: 'Concept tokens to rank-track',
    gridTitle: 'Per-layer top token (deepest at the top)',
    gridAria: (nLayers: number, promptLen: number, mode: LensMode) =>
      `${mode === 'jacobian' ? 'Jacobian-lens' : 'Logit-lens'} argmax grid: ${nLayers} residual boundaries as rows, deepest at the top, by ${promptLen} prompt positions as columns. Each cell is the top token read at that depth${mode === 'jacobian' ? ' through the fitted per-layer Jacobian and the unembedding' : ' straight through the unembedding'}; the highlighted final column is the next-token prediction.`,
    finalColNote: 'The highlighted final column is the next-token prediction — read it bottom-to-top (shallow → deep) to watch depth resolve the answer.',
    identityNote:
      'The output boundary ℓ24 is J = I by construction, so its Jacobian and logit reads are identical — the fitted Jacobian acts mid-stack, not at the output.',
    tooltipHeader: (layer: number, pos: number, tokenText: string) =>
      `ℓ${layer} · position ${pos + 1} · after "${tokenText}"`,
    tooltipProbLabel: 'full-vocab probability',
    tooltipHint: 'Hover or tap any cell to see that layer’s top-K read.',
    heatmapTitle: 'Concept-token rank by depth (final position)',
    heatmapAria: (nPins: number) =>
      `Rank heatmap: ${nPins} pinned concept tokens as columns, residual boundaries as rows with the deepest at the top. Brighter cells mean a better (lower) full-vocab rank at the final position.`,
    legendLabel: 'rank →',
    legendBright: 'rank 1',
    legendDark: '999+',
    surfaceFormNote:
      'The heatmap tracks the pinned surface-form’s rank (the leading-space token), so a concept can sit around rank 4 mid-stack even where the grid’s top token already reads that word — the grid shows the argmax, the heatmap the pinned form’s full-vocab rank.',
    pinsAria: 'Rank-tracked concept tokens',
    pinBestRank: (n: number) => `best rank ${n}`,
    pinSurfaces: (layer: number) => `surfaces at ℓ${layer}`,
    pinNeverSurfaces: (k: number) => `never enters top-${k}`,
    pinPartial: 'concept split into multiple tokens — only its first token is tracked',
    caveat:
      'Read honestly: the first ~third of the stack is noisy, each pin tracks a single surface-form token, and some readouts resist a clean interpretation.',
    // Live affordance
    loadCta: 'Load Qwen3.5-0.8B (~1.6 GB) to recompute this live',
    loadNote: 'Runs entirely on your device; nothing leaves your browser.',
    loadingLabel: 'Loading Qwen3.5-0.8B…',
    errorPrefix: 'Model load failed.',
    axisLabel: 'Live layers',
    axisAria: 'Live recompute layer axis',
    axisCurated: 'Curated',
    axisAll: 'All (1–24)',
    axisNote: 'Live-only — the baked frame stays on the curated boundary subset.',
    computeCta: 'Compute live on your device',
    computeRunningLabel: 'Recomputing…',
    computeNoteLogit: 'Recompute this logit-lens frame live on your in-browser Qwen3.5-0.8B.',
    computeNoteJacobian: 'Recompute this Jacobian-lens frame live — needs the fitted Jacobian pack (+~46 MB) once.',
    packConsentTitle: 'The Jacobian lens needs its fitted pack',
    packConsentCta: 'Load the fitted Jacobian pack (+~46 MB, 23 Jacobians)',
    packConsentNote: 'One-time download, cached for the session; a second Jacobian run reuses it.',
    packConsentConfirm: 'Load pack & compute',
    packConsentCancel: 'Cancel',
    liveErrorPrefix: 'Live recompute failed.',
  },
  zh: {
    title: 'Jacobian lens——同样的层，但更锐利',
    sourceBaked: '离线预计算 · 在这颗 Qwen3.5-0.8B 上拟合',
    sourceLive: '在你的设备上实时计算',
    lensLogit: 'logit lens · 不用 Jacobian',
    lensJacobianApplied: 'Jacobian lens · 已应用拟合的 J',
    lensJacobianIdentity: '请求了 Jacobian · 仅 identity',
    modeLabel: 'Lens',
    modeAria: 'Lens：logit lens 或 Jacobian lens',
    modeLogit: 'LOGIT',
    modeJacobian: 'JACOBIAN',
    presetLabel: '提示',
    presetNames: {
      'french-season': '法语·季节',
      'spanish-opposite': '西语·反义',
      'arithmetic-parens': '算术',
    } as Record<string, string>,
    promptLabel: '提示（原始文本——不套 chat template）',
    pinnedLabel: '要追踪 rank 的概念 token',
    gridTitle: '每一层的 top token（最深的在最上面）',
    gridAria: (nLayers: number, promptLen: number, mode: LensMode) =>
      `${mode === 'jacobian' ? 'Jacobian-lens' : 'Logit-lens'} argmax 网格：${nLayers} 个 residual 边界作为行（最深的在最上面），${promptLen} 个提示位置作为列。每个格子是在该深度${mode === 'jacobian' ? '经过拟合的逐层 Jacobian 和 unembedding' : '直接经过 unembedding'}读出的 top token；高亮的最后一列是下一个 token 的预测。`,
    finalColNote: '高亮的最后一列是下一个 token 的预测——从下往上读（从浅层到深层），看深度如何把答案解出来。',
    identityNote:
      '输出边界 ℓ24 按构造就是 J = I，所以它的 Jacobian 读数和 logit 读数完全相同——拟合的 Jacobian 作用在中间层，而不是在输出处。',
    tooltipHeader: (layer: number, pos: number, tokenText: string) =>
      `ℓ${layer} · 位置 ${pos + 1} · 在 "${tokenText}" 之后`,
    tooltipProbLabel: '全词表概率',
    tooltipHint: '悬停或点按任意格子，查看该层的 top-K 读数。',
    heatmapTitle: '概念 token 的 rank 随深度变化（最后一个位置）',
    heatmapAria: (nPins: number) =>
      `Rank 热力图：${nPins} 个被 pin 的概念 token 作为列，residual 边界作为行（最深的在最上面）。格子越亮表示在最后位置的全词表 rank 越好（越低）。`,
    legendLabel: 'rank →',
    legendBright: 'rank 1',
    legendDark: '999+',
    surfaceFormNote:
      '热力图追踪的是被 pin 的表层形式（带前导空格的那个 token）的 rank，所以一个概念在中间层可能停在 rank 4 附近，即使网格里的 top token 已经读出这个词——网格看的是 argmax，热力图看的是被 pin 形式的全词表 rank。',
    pinsAria: '被追踪 rank 的概念 token',
    pinBestRank: (n: number) => `最好 rank ${n}`,
    pinSurfaces: (layer: number) => `在 ℓ${layer} 浮现`,
    pinNeverSurfaces: (k: number) => `始终进不了 top-${k}`,
    pinPartial: '这个概念被切成了多个 token——只追踪它的第一个 token',
    caveat:
      '诚实地读：栈的前三分之一是有噪声的，每个 pin 只追踪一个表层形式的 token，有些读数本身就难以干净地解释。',
    // Live affordance
    loadCta: '加载 Qwen3.5-0.8B（约 1.6 GB）来实时重算',
    loadNote: '完全在你的设备上运行，数据不会离开浏览器。',
    loadingLabel: '正在加载 Qwen3.5-0.8B……',
    errorPrefix: '模型加载失败。',
    axisLabel: '实时层',
    axisAria: '实时重算的层范围',
    axisCurated: '精选',
    axisAll: '全部（1–24）',
    axisNote: '仅实时——baked 帧固定在精选的边界子集上。',
    computeCta: '在你的设备上实时计算',
    computeRunningLabel: '正在重算……',
    computeNoteLogit: '在你浏览器里的 Qwen3.5-0.8B 上实时重算这个 logit-lens 帧。',
    computeNoteJacobian: '实时重算这个 Jacobian-lens 帧——需要一次性加载拟合的 Jacobian pack（约 +46 MB）。',
    packConsentTitle: 'Jacobian lens 需要它拟合的 pack',
    packConsentCta: '加载拟合的 Jacobian pack（约 +46 MB，23 个 Jacobian）',
    packConsentNote: '一次性下载，本会话内缓存；第二次 Jacobian 运行会复用它。',
    packConsentConfirm: '加载 pack 并计算',
    packConsentCancel: '取消',
    liveErrorPrefix: '实时重算失败。',
  },
} as const;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

type LiveState =
  | { kind: 'idle' }
  | { kind: 'running' }
  | {
      kind: 'done';
      slice: LensSliceData;
      partialFlags: boolean[];
      forSlug: string;
      forMode: LensMode;
    }
  | { kind: 'error'; message: string };

// ---------------------------------------------------------------------------
// The widget
// ---------------------------------------------------------------------------

export function JacobianLensLive() {
  const locale: Locale = useLocale();
  const copy = COPY[locale];
  const reducedMotion = usePrefersReducedMotion();

  // Optional (non-throwing) provider hooks — null under SSG prerender.
  const freeChat = useFreeChatOptional();
  const loader = useModelLoaderOptional();

  const [presetSlug, setPresetSlug] = React.useState<string>(JACOBIAN_PRESETS[0]!.slug);
  const preset: JacobianPreset = JACOBIAN_PRESETS.find((p) => p.slug === presetSlug) ?? JACOBIAN_PRESETS[0]!;
  const bakedFile = BAKED[preset.slug] ?? BAKED[JACOBIAN_PRESETS[0]!.slug]!;

  const [lensMode, setLensMode] = React.useState<LensMode>('jacobian');

  const [live, setLive] = React.useState<LiveState>({ kind: 'idle' });
  const [packConsent, setPackConsent] = React.useState(false);
  const [showAllLayers, setShowAllLayers] = React.useState(false);
  const [packLoaded, setPackLoaded] = React.useState(false);
  const packLoadedRef = React.useRef(false);

  const [selected, setSelected] = React.useState<CellRef | null>(() => ({
    layerIdx: BAKED[JACOBIAN_PRESETS[0]!.slug]!.layers.length - 1,
    pos: BAKED[JACOBIAN_PRESETS[0]!.slug]!.logit.promptLen - 1,
  }));
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
  const providers = freeChat != null && loader != null;

  // -------------------------------------------------------------------------
  // Baked frame for (preset, mode) — the default, model-free, SSG-safe view.
  // -------------------------------------------------------------------------
  const bakedRun = lensMode === 'jacobian' ? bakedFile.jacobian : bakedFile.logit;
  const bakedSlice = React.useMemo(() => buildLensSlice(reviveRun(bakedRun)), [bakedRun]);

  // A completed live recompute overrides the baked frame ONLY for the exact
  // (preset, mode) it was computed for (we reset live → idle on any switch, so
  // this is belt-and-suspenders).
  const liveDone = live.kind === 'done' && live.forSlug === preset.slug && live.forMode === lensMode ? live : null;
  const displaySlice = liveDone ? liveDone.slice : bakedSlice;
  const displayPartialFlags = liveDone ? liveDone.partialFlags : bakedFile.partialFlags;
  const isLive = liveDone != null;
  const running = live.kind === 'running';

  // -------------------------------------------------------------------------
  // Switching preset / mode resets to the baked frame and its default cell.
  // -------------------------------------------------------------------------
  function resetToBaked(slug: string, mode: LensMode) {
    setLive({ kind: 'idle' });
    setPackConsent(false);
    setHovered(null);
    const file = BAKED[slug];
    setSelected(file ? { layerIdx: file.layers.length - 1, pos: file.logit.promptLen - 1 } : null);
    setRunKey((k) => k + 1);
    void mode;
  }

  function pickPreset(slug: string) {
    setPresetSlug(slug);
    resetToBaked(slug, lensMode);
  }

  function pickMode(mode: LensMode) {
    setLensMode(mode);
    resetToBaked(preset.slug, mode);
  }

  // -------------------------------------------------------------------------
  // Live recompute: tokenize the prompt + concept pins (raw, no chat template)
  // exactly as LogitLensLive/bake do, one lensReadout forward pass. For JACOBIAN
  // mode, load the fitted pack first (once). Mirrors LogitLensLive's
  // monotonic-generation + AbortController + app-abort cancellation discipline.
  // -------------------------------------------------------------------------
  async function runLive() {
    const worker = workerRef?.current ?? null;
    if (!worker) {
      setLive({ kind: 'error', message: 'MLX worker is not available' });
      return;
    }
    const myGen = ++runGenRef.current;
    setLive({ kind: 'running' });
    setSelected(null);
    setHovered(null);

    runAbortRef.current?.abort();
    const ctrl = new AbortController();
    runAbortRef.current = ctrl;
    const appSignal = appAbortRef?.current?.signal;
    const onAppAbort = () => ctrl.abort();
    if (appSignal?.aborted) ctrl.abort();
    else appSignal?.addEventListener('abort', onAppAbort, { once: true });

    const wantJacobian = lensMode === 'jacobian';
    const layers = showAllLayers ? ALL_LAYERS : JACOBIAN_LAYERS;
    const forSlug = preset.slug;
    const forMode: LensMode = lensMode;

    try {
      // JACOBIAN needs the fitted pack resident before the first readout. The
      // backend HARD-ERRORS if a non-final layer lacks a J, so never silently
      // fall back to logit — surface loadLensPack's error honestly.
      if (wantJacobian && !packLoadedRef.current) {
        await loadLensPack(worker, { signal: ctrl.signal });
        if (runGenRef.current !== myGen) return;
        packLoadedRef.current = true;
        setPackLoaded(true);
      }

      const promptTokens: TokenInfo[] = await tokenize(worker, preset.prompt, { signal: ctrl.signal });
      if (runGenRef.current !== myGen) return;
      const promptIds = promptTokens.map((t) => t.id);
      if (promptIds.length === 0) throw new Error('prompt tokenized to zero tokens');

      // Pin each concept by its first token (leading space → mid-sentence form).
      // Must match scripts/jlens/bake.mts, or a live run would disagree with the
      // baked frame: when the leading space becomes its own token (` 7` → [' ', '7'])
      // the pin would track bare whitespace, so fall back to the space-less form.
      const pinnedIds: number[] = [];
      const partialFlags: boolean[] = [];
      for (const concept of preset.concepts) {
        let conceptTokens = await tokenize(worker, ` ${concept}`, { signal: ctrl.signal });
        if (runGenRef.current !== myGen) return;
        if (conceptTokens.length > 0 && conceptTokens[0]!.text.trim() === '') {
          conceptTokens = await tokenize(worker, concept, { signal: ctrl.signal });
          if (runGenRef.current !== myGen) return;
        }
        if (conceptTokens.length === 0) continue;
        pinnedIds.push(conceptTokens[0]!.id);
        partialFlags.push(conceptTokens.length > 1);
      }

      const result = await lensReadout(
        worker,
        { promptIds, layers, topK: TOP_K, pinnedIds, useJacobian: wantJacobian },
        { signal: ctrl.signal },
      );
      if (runGenRef.current !== myGen) return;

      // Honesty guard: a jacobian request that did not actually apply a J is a
      // bug, not a logit frame relabeled — hard-fail loudly.
      if (wantJacobian && result.jacobianApplied !== true) {
        throw new Error(
          'the fitted Jacobian did not apply on the non-final boundaries — the pack may be missing or stale',
        );
      }

      const slice = buildLensSlice(result);
      setRunKey((k) => k + 1);
      setLive({ kind: 'done', slice, partialFlags, forSlug, forMode });
      setSelected({ layerIdx: slice.layers.length - 1, pos: slice.promptLen - 1 });
    } catch (err) {
      if (runGenRef.current !== myGen) return;
      if (isAbortError(err)) setLive({ kind: 'idle' });
      else setLive({ kind: 'error', message: err instanceof Error ? err.message : String(err) });
    } finally {
      appSignal?.removeEventListener('abort', onAppAbort);
      if (runAbortRef.current === ctrl) runAbortRef.current = null;
    }
  }

  function handleComputeClick() {
    // JACOBIAN mode requires the +~46 MB pack before its first live readout:
    // gate the download behind explicit consent (a no-op once loaded).
    if (lensMode === 'jacobian' && !packLoaded) {
      setPackConsent(true);
      return;
    }
    void runLive();
  }

  function confirmPackAndCompute() {
    setPackConsent(false);
    void runLive();
  }

  // -------------------------------------------------------------------------
  // Derived view state
  // -------------------------------------------------------------------------
  const active = hovered ?? selected;
  const pinTexts = displaySlice.pinned.map((p) => p.tokenText);
  const pinColors = displaySlice.pinned.map((_p, i) => pinColor(i));
  const colorByPinnedId = new Map<number, string>();
  displaySlice.pinned.forEach((p, i) => colorByPinnedId.set(p.tokenId, pinColor(i)));
  const finalPos = displaySlice.promptLen - 1;

  const sourceBadge = isLive ? copy.sourceLive : copy.sourceBaked;
  const lensBadge =
    lensMode === 'jacobian'
      ? displaySlice.jacobianApplied
        ? copy.lensJacobianApplied
        : copy.lensJacobianIdentity
      : copy.lensLogit;

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="inline-flex items-center gap-1.5 rounded border border-border bg-muted/40 px-2 py-0.5 text-[11px] text-muted-foreground">
            {lensBadge}
          </span>
          <span
            className={[
              'inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-[11px]',
              isLive ? 'border-primary/40 bg-primary/10 text-primary' : 'border-border bg-muted/40 text-muted-foreground',
            ].join(' ')}
          >
            {sourceBadge}
          </span>
        </div>
      </div>

      {/* Preset + lens-mode pickers */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.presetLabel}</span>
          <SegmentedToggle
            value={preset.slug}
            onChange={(v) => pickPreset(v)}
            ariaLabel={copy.presetLabel}
            disabled={running}
            wrap
            options={JACOBIAN_PRESETS.map((p) => ({ value: p.slug, label: copy.presetNames[p.slug] ?? p.slug }))}
          />
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.modeLabel}</span>
          <SegmentedToggle
            value={lensMode}
            onChange={(v) => pickMode(v as LensMode)}
            ariaLabel={copy.modeAria}
            disabled={running}
            options={[
              { value: 'logit', label: copy.modeLogit },
              { value: 'jacobian', label: copy.modeJacobian },
            ]}
          />
        </div>
      </div>

      {/* Prompt + concepts */}
      <div className="space-y-1">
        <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.promptLabel}</div>
        <div className="overflow-x-auto rounded-md border border-border bg-muted/40 p-2 font-mono text-[12px] text-foreground/80">
          {preset.prompt}
        </div>
      </div>
      <div className="space-y-1">
        <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.pinnedLabel}</div>
        <div className="flex flex-wrap items-center gap-2">
          {preset.concepts.map((c, i) => {
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

      {/* Per-preset honest framing (authored in jacobian-presets.ts). */}
      <p className="text-[12px] text-muted-foreground">{preset.blurb}</p>

      {/* THE FRAME (baked by default; a live recompute overrides it). */}
      <div className={['space-y-4 border-t border-border/60 pt-3', reducedMotion ? '' : 'transition-all'].join(' ')}>
        {/* Argmax grid */}
        <div className="space-y-1.5">
          <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.gridTitle}</div>
          <ArgmaxGrid
            slice={displaySlice}
            colorByPinnedId={colorByPinnedId}
            selected={selected}
            onHover={setHovered}
            onSelect={setSelected}
            ariaLabel={copy.gridAria(displaySlice.layers.length, displaySlice.promptLen, lensMode)}
          />
          <p className="text-[11px] text-muted-foreground">{copy.finalColNote}</p>
          {lensMode === 'jacobian' ? (
            <p className="text-[11px] text-muted-foreground/80">{copy.identityNote}</p>
          ) : null}
        </div>

        {/* Per-cell tooltip */}
        {active ? (
          <LensTooltip
            slice={displaySlice}
            layerIdx={active.layerIdx}
            pos={active.pos}
            runKey={runKey}
            header={copy.tooltipHeader(
              displaySlice.layers[active.layerIdx] ?? 0,
              active.pos,
              renderTokenDisplay(displaySlice.tokens[active.pos]?.text ?? ''),
            )}
            probLabel={copy.tooltipProbLabel}
          />
        ) : (
          <p className="text-[11px] text-muted-foreground">{copy.tooltipHint}</p>
        )}

        {/* Pinned concept-token rank tracks */}
        {displaySlice.pinned.length > 0 ? (
          <div className="space-y-2">
            <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.heatmapTitle}</div>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:gap-6">
              <RankHeatmap
                slice={displaySlice}
                finalPos={finalPos}
                pinTexts={pinTexts}
                pinColors={pinColors}
                ariaLabel={copy.heatmapAria(displaySlice.pinned.length)}
                legendLabel={copy.legendLabel}
                brightLabel={copy.legendBright}
                darkLabel={copy.legendDark}
              />
              <PinChips
                slice={displaySlice}
                finalPos={finalPos}
                pinTexts={pinTexts}
                pinColors={pinColors}
                partialFlags={displayPartialFlags}
                copy={{
                  chipsAria: copy.pinsAria,
                  bestRank: copy.pinBestRank,
                  surfaces: copy.pinSurfaces,
                  neverSurfaces: copy.pinNeverSurfaces,
                  partial: copy.pinPartial,
                }}
              />
            </div>
            <p className="text-[11px] text-muted-foreground/80">{copy.surfaceFormNote}</p>
          </div>
        ) : null}
      </div>

      {/* Live recompute affordance (only when the app providers are mounted). */}
      {providers ? (
        <div className="space-y-2 border-t border-border/60 pt-3">
          {!ready ? (
            loader?.status === 'loading' ? (
              <div className="flex items-center gap-2 text-xs text-muted-foreground" role="status" aria-live="polite">
                <Loader2Icon className="h-3.5 w-3.5 animate-spin text-primary" aria-hidden="true" />
                <span>{loader.loadingText || copy.loadingLabel}</span>
              </div>
            ) : (
              <div className="flex flex-col items-start gap-1.5">
                <button
                  type="button"
                  onClick={() => loader?.kickoffLoad()}
                  className="rounded-lg border border-primary/50 bg-primary/10 px-4 py-2 text-sm font-medium text-primary transition-colors hover:bg-primary/20 focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
                >
                  {copy.loadCta}
                </button>
                <span className="text-[11px] text-muted-foreground">{copy.loadNote}</span>
                {loader?.status === 'error' && loader.errorBanner ? (
                  <span className="text-[11px] text-muted-foreground" role="alert">
                    {copy.errorPrefix} {loader.errorBanner}
                  </span>
                ) : null}
              </div>
            )
          ) : (
            <>
              {/* Live-only layer axis */}
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.axisLabel}</span>
                <SegmentedToggle
                  value={showAllLayers}
                  onChange={(v) => setShowAllLayers(v)}
                  ariaLabel={copy.axisAria}
                  disabled={running}
                  options={[
                    { value: false, label: copy.axisCurated },
                    { value: true, label: copy.axisAll },
                  ]}
                />
                <span className="text-[11px] text-muted-foreground">{copy.axisNote}</span>
              </div>

              {/* Consent gate for the fitted pack (JACOBIAN, first live run). */}
              {packConsent ? (
                <div className="space-y-2 rounded-md border border-primary/40 bg-primary/5 p-2.5">
                  <div className="text-[12px] font-medium text-foreground">{copy.packConsentTitle}</div>
                  <div className="text-[11px] text-muted-foreground">{copy.packConsentCta}</div>
                  <div className="text-[11px] text-muted-foreground">{copy.packConsentNote}</div>
                  <div className="flex flex-wrap items-center gap-2 pt-0.5">
                    <button
                      type="button"
                      onClick={confirmPackAndCompute}
                      className="rounded-lg border border-primary/50 bg-primary/10 px-3 py-1.5 text-xs font-medium text-primary transition-colors hover:bg-primary/20 focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
                    >
                      {copy.packConsentConfirm}
                    </button>
                    <button
                      type="button"
                      onClick={() => setPackConsent(false)}
                      className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
                    >
                      {copy.packConsentCancel}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-start gap-1.5">
                  <RunButton
                    onClick={handleComputeClick}
                    running={running}
                    label={copy.computeCta}
                    runningLabel={copy.computeRunningLabel}
                  />
                  <span className="text-[11px] text-muted-foreground">
                    {lensMode === 'jacobian' ? copy.computeNoteJacobian : copy.computeNoteLogit}
                  </span>
                </div>
              )}

              {live.kind === 'error' ? (
                <p className="text-[12px] text-muted-foreground" role="alert">
                  <strong>{copy.liveErrorPrefix}</strong> {live.message}
                </p>
              ) : null}
            </>
          )}
        </div>
      ) : null}

      {/* Compact caveat (the full D12 caveat box is the T4.4b prose task). */}
      <p className="text-[10px] text-muted-foreground/70">{copy.caveat}</p>
    </div>
  );
}

export default JacobianLensLive;
