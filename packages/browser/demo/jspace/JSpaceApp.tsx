import * as React from 'react';

import type { LensPinned, LensReadoutRun, TokenInfo } from '../../src/inspector-types';
import { LENS_MAX_PINNED, LENS_MAX_POSITIONS } from '../../src/inspector-types';
import { lensReadout, loadLensPack } from '../lib/lens-client';
import { tokenize } from '../lib/tokenizer-client';
import { useFreeChat } from '../providers/free-chat';
import { useModelLoader } from '../providers/model-loader';
import { pinColor } from '../jlens-core/colors';
import { derivePins } from '../jlens-core/derive-pins';
import { JACOBIAN_LAYERS, JACOBIAN_PRESETS } from '../jlens-core/jacobian-presets';
import {
  applyPermalink,
  encodePermalink,
  type JSpaceDefaults,
  type LensMode,
} from '../jlens-core/permalink';
import { reviveRun, type BakedFile } from '../jlens-core/revive';
import { compareToBakedFrame, type SelfTestVerdict } from '../jlens-core/self-test';
import { buildLensSlice, type LensSliceData } from '../jlens-core/types';
import { LensTooltip } from '../learn/widgets/jlens/LensTooltip';
import { cleanupTokenText, renderTokenDisplay } from '../learn/inspector/TopKBars';
import { SegmentedToggle } from '../learn/scaffolding/SegmentedToggle';
import { ArgmaxGridCanvas, normalizeSelected, type CellRef } from './ArgmaxGridCanvas';
import { ByLayerStrip } from './ByLayerStrip';
import { ByPosStrip } from './ByPosStrip';
import { PinManager } from './PinManager';
import { PromptTokens } from './PromptTokens';
import { RankChart } from './RankChart';
import { RankHeatmapCanvas } from './RankHeatmapCanvas';
import { STARTERS, STARTER_SLUGS } from './starters';
import { useLensRun } from './useLensRun';

// ---------------------------------------------------------------------------
// Read settings. The Jacobian pack only carries fitted Jacobians for
// JACOBIAN_LAYERS, so jacobian mode is limited to those 11 boundaries; the plain
// logit lens can read every boundary, so it shows the full 24-row stack. topK
// matches the offline bake so the self-test compares like against like.
// ---------------------------------------------------------------------------
const LOGIT_LAYERS: number[] = Array.from({ length: 24 }, (_, i) => i + 1); // 1..24
const TOP_K = 10;

const DEFAULTS: JSpaceDefaults = { mode: 'logit', pins: [], sel: null };

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

function layersFor(mode: LensMode): number[] {
  return mode === 'jacobian' ? JACOBIAN_LAYERS : LOGIT_LAYERS;
}

/** Jacobian activation lifecycle: one-time pack load + silent self-test. */
type JacState =
  | { status: 'idle' }
  | { status: 'activating' }
  | { status: 'ok'; verdict: SelfTestVerdict }
  | { status: 'failed'; verdict: SelfTestVerdict }
  | { status: 'error'; message: string };

export default function JSpaceApp() {
  const { status: modelStatus, kickoffLoad, loadingText, loadKickoff } = useModelLoader();
  const { mlxWorkerRef, inspectorAbortRef } = useFreeChat();
  const modelReady = modelStatus === 'ready';

  // ---- state --------------------------------------------------------------
  const [prompt, setPrompt] = React.useState('');
  const [mode, setMode] = React.useState<LensMode>(DEFAULTS.mode);
  const [pins, setPins] = React.useState<number[]>([]);
  const [activePinIdx, setActivePinIdx] = React.useState<number | null>(null);
  const [selected, setSelected] = React.useState<CellRef | null>(null);
  const [hovered, setHovered] = React.useState<CellRef | null>(null);
  const [showWhitespace, setShowWhitespace] = React.useState(false);
  const [starterSlug, setStarterSlug] = React.useState<string>(STARTER_SLUGS[0]!);
  const [tokenCount, setTokenCount] = React.useState<number | null>(null);
  const [runError, setRunError] = React.useState<string | null>(null);
  const [jac, setJac] = React.useState<JacState>({ status: 'idle' });

  // ---- refs (async-closure-safe mirrors + one-shot guards) ----------------
  const pinsRef = React.useRef(pins);
  pinsRef.current = pins;
  const committedPromptIdsRef = React.useRef<number[] | null>(null);
  const packLoadedRef = React.useRef(false);
  const jacActivatedRef = React.useRef(false);
  const hashAppliedRef = React.useRef(false);
  const lastWrittenHashRef = React.useRef<string | null>(null);

  // A fresh model load resets the worker-global `lensPackLoaded` (mlx-worker.ts),
  // so our belief that the pack is resident + the self-test verdict must reset
  // too — we then re-consult the worker (loadLensPack returns `alreadyLoaded`)
  // rather than trusting a stale ref.
  React.useEffect(() => {
    packLoadedRef.current = false;
    jacActivatedRef.current = false;
    setJac({ status: 'idle' });
  }, [loadKickoff]);

  // -------------------------------------------------------------------------
  // Execution: single-flight lensReadout with the hard jacobianApplied guard.
  // -------------------------------------------------------------------------
  const lensRun = useLensRun(async (args, signal) => {
    const worker = mlxWorkerRef.current;
    if (!worker) throw new Error('MLX worker is not available');
    const result = await lensReadout(
      worker,
      {
        promptIds: args.promptIds,
        layers: args.layers,
        topK: args.topK,
        pinnedIds: args.pinnedIds,
        useJacobian: args.useJacobian,
      },
      { signal },
    );
    // HARD invariant (kept from the bake + lesson): a useJacobian:true request
    // that silently downgraded is a BUG — never relabel a logit frame Jacobian.
    if (args.useJacobian && result.jacobianApplied !== true) {
      throw new Error('Jacobian readout downgraded to logit (jacobianApplied=false) — refusing to mislabel.');
    }
    return result;
  });

  // -------------------------------------------------------------------------
  // Jacobian activation: 46 MB pack + one-time silent self-test.
  // -------------------------------------------------------------------------
  async function runSelfTest(worker: Worker): Promise<SelfTestVerdict> {
    const signal = inspectorAbortRef.current?.signal ?? undefined;
    const preset = JACOBIAN_PRESETS.find((p) => p.slug === 'french-season');
    if (!preset) throw new Error('french-season preset is missing');
    const toks = await tokenize(worker, preset.prompt, { signal });
    const promptIds = toks.map((t) => t.id);
    if (promptIds.length === 0) throw new Error('self-test prompt tokenized to zero tokens');
    // derivePins is used ONLY here (Constraint 8: never in PinManager) — it
    // reproduces the four french-season concept pins [3098, 7094, 40297, 4845].
    const { pinnedIds } = await derivePins(preset.concepts, (text) => tokenize(worker, text, { signal }));
    const live = await lensReadout(
      worker,
      { promptIds, layers: JACOBIAN_LAYERS, topK: TOP_K, pinnedIds, useJacobian: true },
      { signal },
    );
    if (live.useJacobian && live.jacobianApplied !== true) {
      throw new Error('self-test: Jacobian readout downgraded (jacobianApplied=false)');
    }
    const baked = reviveRun((STARTERS['french-season'] as unknown as BakedFile).jacobian);
    return compareToBakedFrame(live, baked);
  }

  async function ensureJacobianReady(): Promise<'ok' | 'failed' | 'unavailable'> {
    if (modelStatus !== 'ready') return 'unavailable';
    const worker = mlxWorkerRef.current;
    if (!worker) return 'unavailable';
    if (jacActivatedRef.current) return jac.status === 'failed' ? 'failed' : 'ok';
    setJac({ status: 'activating' });
    try {
      // Idempotent 46 MB fetch + GPU upload; trust the worker's returned
      // already-loaded flag over packLoadedRef (which a model reload invalidates).
      await loadLensPack(worker, { signal: inspectorAbortRef.current?.signal ?? undefined });
      packLoadedRef.current = true;
      const verdict = await runSelfTest(worker);
      jacActivatedRef.current = true;
      if (verdict.ok) {
        setJac({ status: 'ok', verdict });
        return 'ok';
      }
      // The self-test caught garbage — refuse the "verified" badge, warn loudly.
      setJac({ status: 'failed', verdict });
      return 'failed';
    } catch (err) {
      if (isAbortError(err)) {
        setJac({ status: 'idle' });
        return 'unavailable';
      }
      setJac({ status: 'error', message: err instanceof Error ? err.message : String(err) });
      return 'unavailable';
    }
  }

  // -------------------------------------------------------------------------
  // Dispatch a readout (single-flight). Jacobian runs activate first.
  // -------------------------------------------------------------------------
  async function runReadout(promptIds: number[], pinsArr: number[], m: LensMode): Promise<void> {
    if (m === 'jacobian') {
      const trust = await ensureJacobianReady();
      if (trust === 'unavailable') return; // model/pack not ready — error already surfaced
    }
    committedPromptIdsRef.current = promptIds;
    await lensRun.run({
      promptIds,
      layers: layersFor(m),
      topK: TOP_K,
      pinnedIds: pinsArr,
      useJacobian: m === 'jacobian',
    });
  }

  async function handleRun(): Promise<void> {
    setRunError(null);
    if (modelStatus !== 'ready') {
      kickoffLoad();
      return;
    }
    const worker = mlxWorkerRef.current;
    if (!worker) {
      setRunError('MLX worker is not available.');
      return;
    }
    let toks: TokenInfo[];
    try {
      toks = await tokenize(worker, prompt);
    } catch (err) {
      if (isAbortError(err)) return;
      setRunError(err instanceof Error ? err.message : String(err));
      return;
    }
    const promptIds = toks.map((t) => t.id);
    setTokenCount(promptIds.length); // live token counter reflects this submit
    if (promptIds.length === 0) {
      setRunError('Prompt tokenized to zero tokens.');
      return;
    }
    // CLIENT CAP: refuse to dispatch beyond LENS_MAX_POSITIONS (never a bare 128).
    if (promptIds.length > LENS_MAX_POSITIONS) {
      setRunError(`Prompt is ${promptIds.length} tokens; the maximum is ${LENS_MAX_POSITIONS}. Trim it to run.`);
      return;
    }
    setSelected(null);
    setHovered(null);
    await runReadout(promptIds, pinsRef.current, mode);
  }

  function handleModeChange(next: LensMode): void {
    if (next === mode) return;
    setMode(next);
    setSelected(null);
    setHovered(null);
    const hasLiveRun = committedPromptIdsRef.current !== null;
    if (next === 'jacobian' && modelStatus === 'ready') {
      void (async () => {
        const trust = await ensureJacobianReady();
        if (hasLiveRun && trust !== 'unavailable') {
          await runReadout(committedPromptIdsRef.current!, pinsRef.current, next);
        }
      })();
    } else if (hasLiveRun) {
      // → logit, or → jacobian while the model isn't loaded (shows baked starter).
      void runReadout(committedPromptIdsRef.current!, pinsRef.current, next);
    }
  }

  // ---- pin editing (live runs only) ---------------------------------------
  function addPin(id: number): void {
    if (pins.length >= LENS_MAX_PINNED) return; // client-side cap, refuse the 9th
    if (pins.includes(id)) return;
    const next = [...pins, id];
    setPins(next);
    setActivePinIdx(next.length - 1);
    const ids = committedPromptIdsRef.current;
    if (ids) void runReadout(ids, next, mode);
  }

  function removePin(id: number): void {
    const idx = pins.indexOf(id);
    if (idx < 0) return;
    const next = pins.filter((x) => x !== id);
    setPins(next);
    setActivePinIdx(next.length === 0 ? null : Math.min(activePinIdx ?? 0, next.length - 1));
    const ids = committedPromptIdsRef.current;
    if (ids) void runReadout(ids, next, mode);
  }

  // -------------------------------------------------------------------------
  // Permalink — read the HASH once on mount (never runs, never downloads).
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (hashAppliedRef.current) return;
    hashAppliedRef.current = true;
    const hash = typeof window !== 'undefined' ? window.location.hash : '';
    const restored = applyPermalink(DEFAULTS, hash);
    setPrompt(restored.prompt);
    setMode(restored.mode);
    setPins(restored.pins);
    setSelected(restored.sel);
    setActivePinIdx(restored.pins.length > 0 ? 0 : null);
    lastWrittenHashRef.current = hash.startsWith('#') ? hash.slice(1) : hash;
  }, []);

  // Write the permalink to the HASH with replaceState — NEVER navigate (the root
  // route's searchSchema would strip it). Skips redundant writes + the cold
  // default so a fresh visit keeps a clean URL.
  React.useEffect(() => {
    if (!hashAppliedRef.current) return;
    if (typeof window === 'undefined') return;
    const encoded = encodePermalink({ prompt, mode, pins, sel: selected });
    if (encoded === lastWrittenHashRef.current) return;
    const isColdDefault = prompt === '' && pins.length === 0 && selected === null && mode === DEFAULTS.mode;
    if (isColdDefault && (lastWrittenHashRef.current === null || lastWrittenHashRef.current === '')) return;
    lastWrittenHashRef.current = encoded;
    const url = `${window.location.pathname}${window.location.search}#${encoded}`;
    window.history.replaceState(window.history.state, '', url);
  }, [prompt, mode, pins, selected]);

  // -------------------------------------------------------------------------
  // What to render: LIVE run › model-free STARTER grid (cold) › skeleton
  // (a custom prompt not yet run — NEVER a starter grid under someone's prompt).
  // -------------------------------------------------------------------------
  const liveResult = lensRun.state.status === 'done' ? lensRun.state.result : null;
  const view = React.useMemo((): {
    kind: 'live' | 'starter' | 'skeleton';
    slice: LensSliceData | null;
    pinned: LensPinned[];
  } => {
    if (liveResult) {
      const slice = buildLensSlice(liveResult);
      return { kind: 'live', slice, pinned: liveResult.pinned };
    }
    if (prompt.trim() === '') {
      const frame = (STARTERS[starterSlug] ?? STARTERS['french-season']) as BakedFile;
      const run = mode === 'jacobian' ? frame.jacobian : frame.logit;
      const slice = buildLensSlice(reviveRun(run));
      return { kind: 'starter', slice, pinned: slice.pinned };
    }
    return { kind: 'skeleton', slice: null, pinned: [] };
  }, [liveResult, prompt, mode, starterSlug]);

  const editable = view.kind === 'live';
  const slice = view.slice;
  const pinnedForView = view.pinned;
  const effectiveActiveIdx =
    pinnedForView.length === 0 ? null : Math.min(Math.max(activePinIdx ?? 0, 0), pinnedForView.length - 1);

  const activeCellRef = slice ? normalizeSelected(hovered ?? selected, slice) : null;
  const colorByPinnedId = React.useMemo(() => {
    const m = new Map<number, string>();
    pinnedForView.forEach((p, i) => m.set(p.tokenId, pinColor(i)));
    return m;
  }, [pinnedForView]);

  // labelOf for pins: read the display text from the current view's pinned track.
  const pinLabel = React.useCallback(
    (id: number) => pinnedForView.find((p) => p.tokenId === id)?.tokenText ?? String(id),
    [pinnedForView],
  );

  // The token the Add control would pin: the selected cell's top token (live only).
  const addCandidate =
    editable && slice && activeCellRef
      ? {
          id: slice.cellAt(activeCellRef.layerIdx, activeCellRef.pos).argmaxId,
          text: slice.cellAt(activeCellRef.layerIdx, activeCellRef.pos).topKTexts[0] ?? '',
        }
      : null;

  // RankChart: each pin's rank vs depth at the selected (or final) position.
  const chartPos = slice ? Math.min(activeCellRef?.pos ?? slice.promptLen - 1, slice.promptLen - 1) : 0;
  const chartPoints = slice
    ? pinnedForView.map((_p, pi) =>
        slice.layers.map((layerNum, layerIdx) => ({ x: layerNum, rank: slice.rankAt(pi, layerIdx, chartPos) })),
      )
    : [];
  const chartColors = pinnedForView.map((_p, i) => pinColor(i));
  const selectedLayerNum = slice && activeCellRef ? slice.layers[activeCellRef.layerIdx] ?? null : null;

  const running = lensRun.state.status === 'running';
  const activating = jac.status === 'activating';

  // ---- render -------------------------------------------------------------
  return (
    <main className="mx-auto max-w-[110rem] space-y-5 px-4 py-6">
      <header className="space-y-1">
        <h1 className="text-xl font-semibold">J-Space</h1>
        <p className="text-sm text-muted-foreground">
          Every layer’s guess, at every position — computed on your device.
        </p>
      </header>

      {!modelReady ? (
        <section
          aria-labelledby="jspace-consent"
          className="space-y-2 rounded-md border border-border bg-muted/20 p-3"
        >
          <h2 id="jspace-consent" className="text-sm font-semibold">
            Run the model on your device
          </h2>
          <p className="text-[13px] text-muted-foreground">
            Reading your own prompt downloads about <strong>1.6 GB</strong> of model weights, and a further{' '}
            <strong>46 MB</strong> the first time you switch to the Jacobian lens. Nothing is downloaded
            until you press Run. The starter grid below needs no model.
          </p>
          {modelStatus === 'loading' ? (
            <p className="text-xs text-muted-foreground" role="status" aria-live="polite">
              {loadingText || 'Loading Qwen3.5-0.8B…'}
            </p>
          ) : (
            <button
              type="button"
              onClick={() => kickoffLoad()}
              className="rounded-lg border border-primary/50 bg-primary/10 px-4 py-2 text-sm font-medium text-primary transition-colors hover:bg-primary/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            >
              Download and run (~1.6 GB)
            </button>
          )}
        </section>
      ) : null}

      {/* Prompt editor + controls */}
      <section className="space-y-2">
        <label htmlFor="jspace-prompt" className="text-[11px] uppercase tracking-wider text-muted-foreground">
          Prompt (raw text — no chat template)
        </label>
        <textarea
          id="jspace-prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            // Enter runs; Shift+Enter inserts a newline. Runs fire on Enter,
            // never on every keystroke.
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              void handleRun();
            }
          }}
          rows={2}
          placeholder="Type a prompt, then press Enter to read every layer…"
          className="w-full resize-y rounded-md border border-border bg-background p-2 font-mono text-[13px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
        />
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => void handleRun()}
            disabled={running || activating}
            className="rounded-md border border-primary/50 bg-primary/10 px-3 py-1.5 text-sm font-medium text-primary transition-colors hover:bg-primary/20 disabled:pointer-events-none disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          >
            {!modelReady ? 'Download & run' : running ? 'Reading layers…' : 'Run'}
          </button>

          <SegmentedToggle
            value={mode}
            onChange={handleModeChange}
            ariaLabel="Lens mode"
            disabled={running || activating}
            options={[
              { value: 'logit' as LensMode, label: 'Logit' },
              { value: 'jacobian' as LensMode, label: 'Jacobian' },
            ]}
          />

          <SegmentedToggle
            value={showWhitespace}
            onChange={setShowWhitespace}
            ariaLabel="Whitespace visibility"
            options={[
              { value: false, label: 'Hide ws' },
              { value: true, label: 'Show ws' },
            ]}
          />

          {/* Live token counter — char count is live, token count reflects the
              last submit. Turns red past the cap. Never a bare 128. */}
          <span className="text-[11px] text-muted-foreground">
            {prompt.length} chars ·{' '}
            <span className={tokenCount !== null && tokenCount > LENS_MAX_POSITIONS ? 'font-semibold text-destructive' : ''}>
              {tokenCount ?? '—'} / {LENS_MAX_POSITIONS} tokens
            </span>
          </span>
        </div>

        {/* Jacobian activation status / self-test verdict */}
        {activating ? (
          <p className="text-[12px] text-muted-foreground" role="status" aria-live="polite">
            Loading the fitted-Jacobian lens pack (~46 MB) and self-testing against the baked reference…
          </p>
        ) : jac.status === 'ok' ? (
          <span className="inline-flex items-center gap-1.5 rounded border border-primary/40 bg-primary/10 px-2 py-0.5 text-[11px] text-primary">
            fitted Jacobian · self-test passed (agreement {jac.verdict.topOneAgreement.toFixed(2)}, best-rank Δ{' '}
            {jac.verdict.worstPinDelta})
          </span>
        ) : jac.status === 'failed' ? (
          <p className="text-[12px] text-destructive" role="alert">
            <strong>Jacobian self-test failed.</strong> The browser could not reproduce the baked reference
            (top-1 agreement {jac.verdict.topOneAgreement.toFixed(4)}, worst pinned best-rank delta{' '}
            {jac.verdict.worstPinDelta}). The fitted-Jacobian badge is withheld — treat any Jacobian readout
            with suspicion.
          </p>
        ) : jac.status === 'error' ? (
          <p className="text-[12px] text-destructive" role="alert">
            <strong>Jacobian lens unavailable.</strong> {jac.message}
          </p>
        ) : null}

        {runError ? (
          <p className="text-[12px] text-destructive" role="alert">
            <strong>Lens read failed.</strong> {runError}
          </p>
        ) : lensRun.state.status === 'error' ? (
          <p className="text-[12px] text-destructive" role="alert">
            <strong>Lens read failed.</strong> {lensRun.state.message}
          </p>
        ) : null}
      </section>

      {/* Starter chips — only under the model-free starter grid. */}
      {view.kind === 'starter' ? (
        <section className="space-y-1">
          <span className="text-[11px] uppercase tracking-wider text-muted-foreground">
            Starter (no model needed)
          </span>
          <div className="flex flex-wrap items-center gap-2">
            <SegmentedToggle
              value={starterSlug}
              onChange={setStarterSlug}
              ariaLabel="Starter preset"
              wrap
              options={STARTER_SLUGS.map((slug) => ({
                value: slug,
                label: (STARTERS[slug] as BakedFile).prompt,
              }))}
            />
          </div>
        </section>
      ) : null}

      {/* The main grid + panels */}
      {view.kind === 'skeleton' || !slice ? (
        <section className="space-y-2">
          <div className="rounded-md border border-border bg-background p-3 font-mono text-[13px]" style={{ whiteSpace: 'pre-wrap' }}>
            {prompt}
          </div>
          {pins.length > 0 ? (
            <p className="text-[12px] text-muted-foreground">
              Restored pins: {pins.map((id) => `#${id}`).join(', ')} — run to see their rank tracks.
            </p>
          ) : null}
          <div className="flex min-h-[8rem] items-center justify-center rounded-md border border-dashed border-border bg-muted/10 text-sm text-muted-foreground">
            Run to compute — press Run to read every layer for this prompt.
          </div>
        </section>
      ) : (
        <section className="space-y-4">
          {/* Prompt tokens = position axis */}
          <PromptTokens
            tokens={slice.tokens}
            selectedPos={activeCellRef?.pos ?? null}
            showWhitespace={showWhitespace}
            onSelectPos={(pos) =>
              setSelected((prev) => ({ layerIdx: prev?.layerIdx ?? slice.layers.length - 1, pos }))
            }
          />

          {/* Argmax grid */}
          <div className="space-y-1.5">
            <div className="text-[11px] uppercase tracking-wider text-muted-foreground">
              Per-layer top token (deepest at the top)
              {view.kind === 'starter' ? ' · baked starter (no model)' : slice.jacobianApplied ? ' · fitted Jacobian' : ' · logit lens'}
            </div>
            <ArgmaxGridCanvas
              slice={slice}
              colorByPinnedId={colorByPinnedId}
              selected={selected}
              onHover={setHovered}
              onSelect={setSelected}
              showWhitespace={showWhitespace}
              ariaLabel={`Argmax grid: ${slice.layers.length} residual boundaries as rows (deepest on top) by ${slice.promptLen} prompt positions as columns.`}
            />
          </div>

          {/* Per-cell tooltip */}
          {activeCellRef ? (
            <LensTooltip
              slice={slice}
              layerIdx={activeCellRef.layerIdx}
              pos={activeCellRef.pos}
              runKey={lensRun.runKey}
              header={`ℓ${slice.layers[activeCellRef.layerIdx]} · position ${activeCellRef.pos + 1} · after "${renderTokenDisplay(slice.tokens[activeCellRef.pos]?.text ?? '')}"`}
              probLabel="full-vocab probability"
            />
          ) : (
            <p className="text-[11px] text-muted-foreground">Hover or tap any cell to see that layer’s top-K read.</p>
          )}

          {/* Pins + rank field + rank-by-depth chart */}
          <div className="space-y-3">
            <PinManager
              pins={pinnedForView.map((p) => p.tokenId)}
              labelOf={pinLabel}
              activePinIdx={effectiveActiveIdx}
              onSelectActive={setActivePinIdx}
              maxPins={LENS_MAX_PINNED}
              addCandidate={addCandidate}
              onAddPin={editable ? addPin : undefined}
              onRemovePin={editable ? removePin : undefined}
            />

            {pinnedForView.length > 0 && effectiveActiveIdx !== null ? (
              <div className="flex flex-col gap-4 lg:flex-row lg:items-start">
                <div className="space-y-1">
                  <div className="text-[11px] uppercase tracking-wider text-muted-foreground">
                    Rank of “{cleanupTokenText(pinnedForView[effectiveActiveIdx]?.tokenText ?? '')}” by position × layer
                  </div>
                  <RankHeatmapCanvas
                    slice={slice}
                    pinnedIdx={effectiveActiveIdx}
                    selected={selected}
                    onSelect={setSelected}
                  />
                </div>
                <div className="min-w-0 flex-1 space-y-1">
                  <div className="text-[11px] uppercase tracking-wider text-muted-foreground">
                    Pinned-token rank by depth (position {chartPos + 1})
                  </div>
                  <RankChart
                    points={chartPoints}
                    colors={chartColors}
                    xLabel="residual boundary ℓ"
                    selectedX={selectedLayerNum}
                  />
                </div>
              </div>
            ) : null}
          </div>

          {/* Cross-sections at the selected cell */}
          {activeCellRef ? (
            <div className="grid gap-4 lg:grid-cols-2">
              <ByLayerStrip slice={slice} pos={activeCellRef.pos} selected={selected} onSelect={setSelected} />
              <ByPosStrip slice={slice} layerIdx={activeCellRef.layerIdx} selected={selected} onSelect={setSelected} />
            </div>
          ) : null}
        </section>
      )}
    </main>
  );
}
