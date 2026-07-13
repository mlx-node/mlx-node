import * as React from 'react';
import { Link } from '@tanstack/react-router';

import type { LensPinned, TokenInfo } from '../../src/inspector-types';
import { LENS_MAX_PINNED, LENS_MAX_POSITIONS } from '../../src/inspector-types';
import { lensReadout, loadLensPack } from '../lib/lens-client';
import { readStoredLocale, type Locale } from '../lib/i18n';
import { tokenize } from '../lib/tokenizer-client';
import { useFreeChat } from '../providers/free-chat';
import { useModelLoader } from '../providers/model-loader';
import { pinColor } from '../jlens-core/colors';
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
import { composeAbort } from './compose-abort';
import { isColdPrompt } from './cold-prompt';
import { ByPosStrip } from './ByPosStrip';
import { PinManager } from './PinManager';
import { PromptTokens } from './PromptTokens';
import { RankChart } from './RankChart';
import { RankHeatmapCanvas } from './RankHeatmapCanvas';
import { STARTERS, STARTER_SLUGS } from './starters';
import { GALLERY } from './starters/gallery';
import { useLensRun } from './useLensRun';

// ---------------------------------------------------------------------------
// Read settings. BOTH lenses read every residual boundary 1..24 so the LOGIT ↔
// JACOBIAN toggle is directly comparable and never shrinks the grid: the shipped
// fitted-Jacobian pack carries a fitted J for every boundary 1..23 (24 is J=I,
// the plain unembedding), so Jacobian mode is NOT limited to a subset. The
// 11-boundary JACOBIAN_LAYERS sample is used ONLY by the self-test oracle below
// to replay the committed 11-layer baked reference frame. topK matches the
// offline bake so the self-test compares like against like.
// ---------------------------------------------------------------------------
const ALL_LAYERS: number[] = Array.from({ length: 24 }, (_, i) => i + 1); // 1..24
const TOP_K = 10;

const DEFAULTS: JSpaceDefaults = { mode: 'logit', pins: [], sel: null };

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

/** Both lenses read the full boundary stack 1..24 (see {@link ALL_LAYERS}), so a
 *  mode toggle never shrinks the grid. `mode` is accepted to keep the call sites
 *  uniform and to document that the two modes stay aligned. Exported for the
 *  layer-parity unit test. */
export function layersFor(mode: LensMode): number[] {
  return mode === 'jacobian' ? ALL_LAYERS : ALL_LAYERS;
}

// Pin ids for a baked gallery frame — model-free (the frame already carries them,
// as SerializedPinned[] with a plain `tokenId`). At MODULE scope so it can seed the
// `pins` useState. Prefer the jacobian run's pins (all vetted tiles open jacobian),
// fall back to the logit run, then to no pins for an unknown slug.
function pinsForSlug(slug: string): number[] {
  const frame = STARTERS[slug];
  const pinned = frame?.jacobian?.pinned ?? frame?.logit?.pinned ?? [];
  return pinned.map((p) => p.tokenId);
}

/** Jacobian activation lifecycle: one-time pack load + silent self-test. */
type JacState =
  | { status: 'idle' }
  | { status: 'activating' }
  | { status: 'ok'; verdict: SelfTestVerdict }
  | { status: 'failed'; verdict: SelfTestVerdict }
  | { status: 'error'; message: string };

// ---------------------------------------------------------------------------
// Bilingual copy for the model-free gallery surfaces (names · hooks · blurbs ·
// bandNote). /jspace has no LocaleProvider, so `locale` is resolved from the
// stored preference in an effect (see the component). Glossary terms — logit
// lens, Jacobian, J-lens, token, rank, ℓ — stay English per the zh convention.
// `galleryLabel`, `presetNames`, `galleryHooks`, `bandNote` are declared now but
// consumed by the Task-6 gallery launcher; `blurbs` powers the D1 caption today.
// ---------------------------------------------------------------------------
const JSPACE_COPY = {
  en: {
    galleryLabel: 'Examples · no model needed',
    presetNames: {
      'french-season': 'French season', 'arith-inner-sum': 'Inner sum',
      'arith-precedence': 'Operator precedence', 'arith-fewshot': 'Few-shot sum',
      'grammar-error': 'Grammar error', 'giza-continent': 'Giza continent',
      'int-cast-error': 'Bad int() cast', 'eiffel-capital': 'Eiffel capital',
    } as Record<string, string>,
    galleryHooks: {
      'french-season': "Mid-stack the lens surfaces 'season' then 'summer' before the answer 'automne'.",
      'arith-inner-sum': "The unspoken inner sum '3' (1+2) surfaces at ℓ18; the model then answers 6.",
      'arith-precedence': "Precedence: the product '8' (4×2) is held mid-stack before adding 3.",
      'arith-fewshot': "The inner sum '5' (2+3) shows up across ℓ20–23 in the few-shot analogy.",
      'grammar-error': "After 'childs', the judgment 'incorrect' surfaces — the correction never does.",
      'giza-continent': "Faint: the answer 'Africa' surfaces mid-stack while the real output is a degenerate '1'; the Egypt bridge hop never appears.",
      'int-cast-error': "Faint: a generic 'error' concept for int('hello') — never 'ValueError'.",
      'eiffel-capital': "Faint: the answer 'Paris' surfaces; the France bridge hop never does.",
    } as Record<string, string>,
    bandNote: (onset: number, peak: number) =>
      `Legibility band: intermediate concepts start surfacing around ℓ${onset}, peaking near ℓ${peak}.`,
    blurbs: {
      'french-season': "Headline: mid-stack the J-lens surfaces the abstract concept ('season' near rank 1, 'summer' near rank 2, around boundaries 16-17) where the plain logit lens is still ranked 999+. A clean mid-band concept cluster. The target token is 'automne' (autumn).",
      'arith-inner-sum': "Jacobian lens surfaces the unspoken inner sum '3' (1+2) at rank 2 by ℓ18, then the model correctly outputs 6. The logit lens shows '3' only at the final layer (rank 6). '3' is in neither the prompt nor the output.",
      'arith-precedence': "Precedence in action: the Jacobian lens holds the unspoken product '8' (4×2) at rank 0 around ℓ20, then the model answers 11. The logit lens buries '8' at rank 7 in the last layer only.",
      'arith-fewshot': "The unspoken inner sum '5' (2+3) sits at rank 2 across ℓ20–23 in the Jacobian lens; the logit lens shows it only at the final layer. '5' appears in neither the prompt nor the output.",
      'grammar-error': "After the error 'childs', both lenses raise the unspoken judgment 'incorrect' to rank 1–2 from ℓ17 — genuine error detection. Honest caveat: the correction 'children' never surfaces, only the flag.",
      'giza-continent': "Faint but real: the Jacobian lens surfaces the unspoken answer 'Africa' (around rank 2, ℓ17–18) while the logit lens shows nothing and the model's actual greedy output is a degenerate '1'. Honest caveat: the Egypt bridge hop never appears on this 0.8B model.",
      'int-cast-error': "Faint but real: the Jacobian lens raises a generic 'error' concept to rank 1 at ℓ18 for the invalid int('hello') cast (the logit lens shows nothing). Never 'ValueError' or 'invalid' as the paper's larger models show.",
      'eiffel-capital': "Faint but real: the Jacobian lens surfaces the unspoken answer 'Paris' at rank 5–6 around ℓ18 while the model's real output is a degenerate '1'; the France bridge hop never appears. The logit lens shows nothing.",
    } as Record<string, string>,
  },
  zh: {
    galleryLabel: '示例 · 无需模型',
    presetNames: {
      'french-season': '法语·季节', 'arith-inner-sum': '内层求和',
      'arith-precedence': '运算优先级', 'arith-fewshot': '少样本求和',
      'grammar-error': '语法错误', 'giza-continent': '吉萨·大洲',
      'int-cast-error': 'int() 转换错误', 'eiffel-capital': '埃菲尔·首都',
    } as Record<string, string>,
    galleryHooks: {
      'french-season': "中间层 lens 先浮现 'season'，再是 'summer'，然后才是答案 'automne'。",
      'arith-inner-sum': "未说出口的内层和 '3'（1+2）在 ℓ18 浮现，随后模型答出 6。",
      'arith-precedence': "优先级：乘积 '8'（4×2）在中间层被暂存，之后才加上 3。",
      'arith-fewshot': "内层和 '5'（2+3）在 ℓ20–23 的少样本类比中出现。",
      'grammar-error': "在 'childs' 之后，判断词 'incorrect' 浮现——但正确写法从未出现。",
      'giza-continent': "微弱：答案 'Africa' 在中间层浮现，而真实输出是退化的 '1'；埃及这一跳桥概念从未出现。",
      'int-cast-error': "微弱：int('hello') 只浮现出泛化的 'error'——从不是 'ValueError'。",
      'eiffel-capital': "微弱：答案 'Paris' 浮现，但法国这一跳桥概念从未出现。",
    } as Record<string, string>,
    bandNote: (onset: number, peak: number) =>
      `可读性区间：中间概念大约从 ℓ${onset} 开始浮现，在 ℓ${peak} 附近达到峰值。`,
    blurbs: {
      'french-season': "标题示例：J-lens 在中间层浮现出抽象概念（'season' 接近 rank 1，'summer' 接近 rank 2，大约在边界 16-17），而普通的 logit lens 仍停留在 rank 999+。一个干净的中间层概念簇。目标 token 是 'automne'（autumn，秋天）。",
      'arith-inner-sum': "J-lens 在 ℓ18 以 rank 2 浮现出未说出口的内层和 '3'（1+2），随后模型正确输出 6。logit lens 仅在最后一层（rank 6）显示 '3'。'3' 既不在 prompt 也不在输出中。",
      'arith-precedence': "优先级实况：J-lens 在 ℓ20 附近以 rank 0 暂存乘积 '8'（4×2），随后模型答出 11。logit lens 仅在最后一层把 '8' 埋在 rank 7。",
      'arith-fewshot': "未说出口的内层和 '5'（2+3）在 J-lens 中横跨 ℓ20–23 稳定处于 rank 2；logit lens 仅在最后一层显示。'5' 既不在 prompt 也不在输出中。",
      'grammar-error': "在错误的 'childs' 之后，两种 lens 都从 ℓ17 起把判断词 'incorrect' 抬到 rank 1–2——真实的错误检测。诚实提醒：正确写法 'children' 从未浮现，只有错误标记。",
      'giza-continent': "微弱但真实：J-lens 在 ℓ17–18 附近以 rank 2 浮现出未说出口的答案 'Africa'，而 logit lens 毫无显示，模型真实的贪心输出是退化的 '1'。诚实提醒：埃及这一跳桥概念在 0.8B 模型上从未出现。",
      'int-cast-error': "微弱但真实：J-lens 在 ℓ18 为非法的 int('hello') 转换把泛化的 'error' 概念抬到 rank 1（logit lens 毫无显示）。从不是论文中大模型显示的 'ValueError' 或 'invalid'。",
      'eiffel-capital': "微弱但真实：J-lens 在 ℓ18 附近以 rank 5–6 浮现出未说出口的答案 'Paris'，而模型真实输出是退化的 '1'；法国这一跳桥概念从未出现。logit lens 毫无显示。",
    } as Record<string, string>,
  },
} as const;

export default function JSpaceApp() {
  const { status: modelStatus, kickoffLoad, loadingText, loadKickoff } = useModelLoader();
  const { mlxWorkerRef, inspectorAbortRef } = useFreeChat();
  const modelReady = modelStatus === 'ready';

  // ---- state --------------------------------------------------------------
  const [prompt, setPrompt] = React.useState('');
  const [mode, setMode] = React.useState<LensMode>(DEFAULTS.mode);
  // Seed the headline example's concept pins so first paint threads (and a later
  // live run inherits them). The mount permalink-restore effect below overrides
  // this: a real permalink wins with its own pins; a cold visit resets to [].
  const [pins, setPins] = React.useState<number[]>(pinsForSlug(STARTER_SLUGS[0]!));
  const [activePinIdx, setActivePinIdx] = React.useState<number | null>(null);
  const [selected, setSelected] = React.useState<CellRef | null>(null);
  const [hovered, setHovered] = React.useState<CellRef | null>(null);
  // Sticky "last engaged cell". The per-cell readout and the cross-section strips
  // are MOUNT-gated on `activeCellRef` (= hovered ?? selected). `hovered` flips to
  // null the instant the cursor leaves the grid, so without this a hover-only
  // session (no click) would UNMOUNT those large sections on every mouse-out —
  // which reflows the page (worst when scrolled to the bottom, where the shrink
  // clamps the scroll and re-slides a cell under the cursor, re-arming hover) in an
  // unmount→reflow→re-hover blink loop. Persisting the last engaged cell keeps them
  // mounted; hover still drives their CONTENT. Reset on every new run / mode change.
  const [focusCell, setFocusCell] = React.useState<CellRef | null>(null);
  const [showWhitespace, setShowWhitespace] = React.useState(false);
  const [starterSlug, setStarterSlug] = React.useState<string>(STARTER_SLUGS[0]!);
  const [tokenCount, setTokenCount] = React.useState<number | null>(null);
  const [runError, setRunError] = React.useState<string | null>(null);
  const [jac, setJac] = React.useState<JacState>({ status: 'idle' });
  // True from `handleRun` entry until the dispatched run fully settles — covers the
  // pre-dispatch `await tokenize(...)` window where neither `running` nor `activating`
  // is set yet, so the mode toggle can't change out from under an in-flight run and a
  // second Enter/Run can't start a duplicate (F1).
  const [preparing, setPreparing] = React.useState(false);

  // ---- locale (no LocaleProvider on /jspace) ------------------------------
  // SSG first paint = EN (prerender stays deterministic); a zh visitor arriving
  // from the /zh landing already has `mlx:preferences:locale = 'zh'` stored, so
  // the gallery/captions flip to zh after mount. `useLocale()` would always be EN
  // here because this standalone route has no provider — read storage in an effect.
  const [locale, setLocale] = React.useState<Locale>('en');
  React.useEffect(() => {
    const s = readStoredLocale();
    if (s) setLocale(s);
  }, []);
  const copy = JSPACE_COPY[locale];
  const galleryEntry = GALLERY.find((g) => g.slug === starterSlug) ?? null;

  // ---- refs (async-closure-safe mirrors + one-shot guards) ----------------
  const pinsRef = React.useRef(pins);
  pinsRef.current = pins;
  const committedPromptIdsRef = React.useRef<number[] | null>(null);
  const packLoadedRef = React.useRef(false);
  const jacActivatedRef = React.useRef(false);
  // Single-flight promise for `ensureJacobianReady`: all entry points (Enter, mode
  // toggle, addPin/removePin → runReadout) coalesce onto ONE 46 MB download + ONE
  // self-test instead of racing concurrent activations (F2).
  const jacActivationRef = React.useRef<Promise<'ok' | 'failed' | 'unavailable'> | null>(null);
  const hashAppliedRef = React.useRef(false);
  const lastWrittenHashRef = React.useRef<string | null>(null);
  // The permalink WRITE effect's first invocation is always the mount pass, where
  // the restore effect has already established the authoritative initial state and
  // set `lastWrittenHashRef` to the incoming hash — so the write effect must not
  // persist anything then. Skip that one pass: without it, `pins` is still the
  // headline SEED (the restore effect's setPins hasn't reconciled within the same
  // effect flush), so the cold-default skip below misfires and a fresh visit
  // writes a spurious `#p=&mode=l` instead of keeping a clean URL.
  const firstHashWriteRef = React.useRef(true);
  // A selection restored from a permalink onto a CUSTOM prompt can't render until
  // the reader runs (skeleton until then) and `handleRun` clears `selected` before
  // dispatch — so hold it here and re-apply it ONCE after the first slice arrives,
  // clamped to that slice. Consumed once; later runs never re-apply it.
  const pendingSelRef = React.useRef<CellRef | null>(null);

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
      // Cancel on EITHER the component lifetime (useLensRun's own signal) OR the
      // worker generation serving this readout: a post-ready worker/GPU error →
      // retry terminates + replaces the worker (app.tsx aborts inspectorAbortRef
      // on teardown), which rejects this readout promptly so the queue drains
      // instead of stranding the next run behind a hung dispatch for 60 s.
      { signal: composeAbort(signal, inspectorAbortRef.current?.signal) },
    );
    // HARD invariant (kept from the bake + lesson): a useJacobian:true request
    // that silently downgraded is a BUG — never relabel a logit frame Jacobian.
    if (args.useJacobian && result.jacobianApplied !== true) {
      throw new Error('Jacobian readout downgraded to logit (jacobianApplied=false) — refusing to mislabel.');
    }
    return result;
  });
  const { reset: resetRun } = lensRun;

  // A fresh model load (loadKickoff bump — cold load OR a retry that terminates +
  // replaces the worker) resets the worker-global `lensPackLoaded` (mlx-worker.ts),
  // so our belief that the pack is resident + the self-test verdict must reset too
  // (we then re-consult the worker; loadLensPack returns `alreadyLoaded`). It also
  // starts a NEW worker generation, so the committed prompt + any displayed frame
  // computed on the previous worker are stale — drop them so recovery shows the
  // model-free view until the reader re-runs, never a dead-worker frame.
  React.useEffect(() => {
    packLoadedRef.current = false;
    jacActivatedRef.current = false;
    jacActivationRef.current = null;
    committedPromptIdsRef.current = null;
    resetRun();
    setJac({ status: 'idle' });
  }, [loadKickoff, resetRun]);

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
    // Replay the EXACT baked french-season frame: source the pins from the frame we
    // validate against, NOT derivePins. The gallery bake now re-pins each concept to
    // the token that actually SURFACES it (robust pin resolution, Task-2 fix #1), so
    // a baked pin id can differ from derivePins' first-token guess (e.g. 'season' →
    // the standalone token 16306, not the ' season' fragment 3098). Re-deriving here
    // would structurally mismatch the baked frame and spuriously fail the self-test.
    // The garbage detector is the per-cell top-1 agreement, which is pin-independent.
    // (This is the /jspace app's only former derivePins use; PinManager never used it
    // either — Constraint 8 — so the app no longer imports derivePins at all.)
    const baked = reviveRun((STARTERS['french-season'] as unknown as BakedFile).jacobian);
    const pinnedIds = baked.pinned.map((p) => p.tokenId);
    const live = await lensReadout(
      worker,
      { promptIds, layers: JACOBIAN_LAYERS, topK: TOP_K, pinnedIds, useJacobian: true },
      { signal },
    );
    // We REQUESTED useJacobian:true, so the response MUST report jacobianApplied:true —
    // check unconditionally (a downgraded useJacobian:false response must not slip past).
    if (live.jacobianApplied !== true) {
      throw new Error('self-test: Jacobian readout downgraded (jacobianApplied=false)');
    }
    return compareToBakedFrame(live, baked);
  }

  async function ensureJacobianReady(): Promise<'ok' | 'failed' | 'unavailable'> {
    if (modelStatus !== 'ready') return 'unavailable';
    const worker = mlxWorkerRef.current;
    if (!worker) return 'unavailable';
    if (jacActivatedRef.current) return jac.status === 'failed' ? 'failed' : 'ok';
    if (jacActivationRef.current) return jacActivationRef.current; // coalesce concurrent callers
    const p = (async (): Promise<'ok' | 'failed' | 'unavailable'> => {
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
    })();
    jacActivationRef.current = p;
    try {
      return await p;
    } finally {
      jacActivationRef.current = null;
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
    // `preparing` guards the whole run — including the pre-dispatch `await tokenize`
    // window — so the mode toggle and a second run stay disabled until this settles.
    setPreparing(true);
    try {
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
        // Bind tokenization to the worker generation too, so a worker teardown
        // (retry/replacement) rejects it promptly instead of hanging to timeout.
        toks = await tokenize(worker, prompt, { signal: inspectorAbortRef.current?.signal ?? undefined });
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
      setFocusCell(null); // new run → back to the "hover a cell" placeholder
      await runReadout(promptIds, pinsRef.current, mode);
    } finally {
      setPreparing(false);
    }
  }

  function handleModeChange(next: LensMode): void {
    if (next === mode) return;
    setMode(next);
    setSelected(null);
    setHovered(null);
    setFocusCell(null); // mode swap → back to the "hover a cell" placeholder
    // User diverged from any permalink-restored state — cancel the one-shot
    // restore so a later run does not resurrect a stale selection (F3).
    pendingSelRef.current = null;
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

  // Open a baked gallery example: model-free, always available (even after the user
  // typed a custom prompt). Blanks the prompt back to cold so the baked starter grid
  // shows, drops any live frame (the view memo prefers it otherwise), opens in the
  // tile's vetted lens, and seeds the concept pins. Clears the SAME selection/status
  // state that handleRun + handleModeChange clear so no stale cell/verdict carries in.
  const openStarter = React.useCallback(
    (slug: string): void => {
      const entry = GALLERY.find((g) => g.slug === slug);
      if (!entry) return;
      setStarterSlug(slug);
      setPrompt(''); // cold → view.kind === 'starter'
      resetRun(); // drop any live frame (the memo prefers it otherwise)
      committedPromptIdsRef.current = null;
      setMode(entry.defaultMode); // all vetted tiles open in jacobian
      setPins(pinsForSlug(slug)); // pins are number[] (token ids) — no conversion
      setActivePinIdx(null);
      setSelected(null);
      setHovered(null);
      setFocusCell(null);
      setTokenCount(null);
      setRunError(null);
      pendingSelRef.current = null; // cancel any pending permalink restore
    },
    [resetRun],
  );

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
    setSelected(restored.sel); // renders immediately for a STARTER permalink (grid is present)
    pendingSelRef.current = restored.sel; // …and survives the first Run for a CUSTOM permalink
    setActivePinIdx(restored.pins.length > 0 ? 0 : null);
    lastWrittenHashRef.current = hash.startsWith('#') ? hash.slice(1) : hash;
  }, []);

  // Re-apply a permalink-restored selection ONCE after the first live slice: the
  // custom-prompt path starts as a skeleton and `handleRun` clears `selected`, so
  // without this the shared cell would be lost. Clamp to the real slice and consume.
  React.useEffect(() => {
    if (lensRun.state.status !== 'done' || pendingSelRef.current === null) return;
    const slice = buildLensSlice(lensRun.state.result);
    setSelected(normalizeSelected(pendingSelRef.current, slice));
    pendingSelRef.current = null;
  }, [lensRun.state]);

  // Write the permalink to the HASH with replaceState — NEVER navigate (the root
  // route's searchSchema would strip it). Skips redundant writes + the cold
  // default so a fresh visit keeps a clean URL.
  React.useEffect(() => {
    if (!hashAppliedRef.current) return;
    if (typeof window === 'undefined') return;
    // Never persist the pre-restore mount transient (see firstHashWriteRef).
    if (firstHashWriteRef.current) {
      firstHashWriteRef.current = false;
      return;
    }
    const encoded = encodePermalink({ prompt, mode, pins, sel: selected });
    if (encoded === lastWrittenHashRef.current) return;
    const isColdDefault = isColdPrompt(prompt) && pins.length === 0 && selected === null && mode === DEFAULTS.mode;
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
    if (isColdPrompt(prompt)) {
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

  // A deliberate selection (keyboard nav or click) wins over a stale hover: clear
  // `hovered` so `activeCellRef` (= hovered ?? selected) follows the selection.
  // Otherwise the canvas ring + aria-activedescendant (which track `selected`)
  // diverge from the tooltip / charts / cross-sections / Pin target (which track
  // activeCellRef), and Pin could pin the hovered cell while another is selected.
  const selectCell = React.useCallback((ref: CellRef | null) => {
    setHovered(null);
    setSelected(ref);
    // A deliberate selection is a user divergence — drop any pending permalink
    // restore so it can't overwrite this selection after the next run (F3).
    pendingSelRef.current = null;
  }, []);

  // The user's live pointer/keyboard target; `focusCell` persists the last one so
  // `activeCellRef` never collapses to null mid-session (see focusCell above).
  const engagedCell = hovered ?? selected;
  React.useEffect(() => {
    if (engagedCell) setFocusCell(engagedCell);
  }, [engagedCell?.layerIdx, engagedCell?.pos]);
  const activeCellRef = slice ? normalizeSelected(engagedCell ?? focusCell, slice) : null;
  const colorByPinnedId = React.useMemo(() => {
    const m = new Map<number, string>();
    pinnedForView.forEach((p, i) => m.set(p.tokenId, pinColor(i)));
    return m;
  }, [pinnedForView]);

  // Legibility band for the argmax grid gutter: only on the model-free baked
  // starter in jacobian mode (the band is a per-example vetted claim about the
  // BAKED frame; a live logit read or a custom prompt has no such band).
  const band = React.useMemo(
    () => (view.kind === 'starter' && mode === 'jacobian' && galleryEntry ? galleryEntry.band : null),
    [view.kind, mode, galleryEntry],
  );

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

  // RankChart 1: each pin's rank vs DEPTH at the selected (or final) position.
  const chartPos = slice ? Math.min(activeCellRef?.pos ?? slice.promptLen - 1, slice.promptLen - 1) : 0;
  const chartPoints = slice
    ? pinnedForView.map((_p, pi) =>
        slice.layers.map((layerNum, layerIdx) => ({ x: layerNum, rank: slice.rankAt(pi, layerIdx, chartPos) })),
      )
    : [];
  const chartColors = pinnedForView.map((_p, i) => pinColor(i));
  const selectedLayerNum = slice && activeCellRef ? slice.layers[activeCellRef.layerIdx] ?? null : null;

  // RankChart 2: each pin's rank vs POSITION at the selected (or deepest) layer.
  const chartLayerIdx = slice ? (activeCellRef?.layerIdx ?? slice.layers.length - 1) : 0;
  const posChartPoints = slice
    ? pinnedForView.map((_p, pi) =>
        Array.from({ length: slice.promptLen }, (_, pos) => ({ x: pos + 1, rank: slice.rankAt(pi, chartLayerIdx, pos) })),
      )
    : [];
  // The layer actually charted, so the header is honest even with no selection.
  const posChartLayerNum = slice ? slice.layers[chartLayerIdx] ?? null : null;

  const running = lensRun.state.status === 'running';
  const activating = jac.status === 'activating';

  // ---- render -------------------------------------------------------------
  return (
    // /jspace is a standalone route rendered directly inside `.app-root`
    // (position: relative; h-dvh; overflow-hidden — demo/styles.css), whose body is
    // `overflow: hidden`, so the document never scrolls. Like LessonLayout and the
    // chat layer, /jspace must OWN its scroll surface. `absolute inset-0` pins this
    // <main> to all four edges of the shell — fixing its height to the viewport so
    // `overflow-y-auto` scrolls internally instead of clipping; `max-w-[110rem]
    // mx-auto` still centers the body (auto margins on an inset-0 box). Without this
    // the live view (grid + heatmap + charts + cross-sections) is unreachable below
    // the fold.
    <main className="jspace-scroll absolute inset-0 mx-auto max-w-[82rem] space-y-8 overflow-y-auto px-6 py-10 md:px-10 md:py-14">
      <header className="space-y-3 border-b border-border/60 pb-8">
        {/* Back to the course landing — /jspace is a standalone route with no
            shared chrome, so without this it is a dead end. Not locale-prefixed
            (there is no /zh/jspace), matching how the landing links INTO /jspace. */}
        <Link
          to="/"
          search={(prev) => prev}
          className="inline-flex items-center gap-1.5 rounded-sm font-mono text-[10px] uppercase tracking-[0.18em] text-[color:var(--text-dim)] transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
        >
          <span aria-hidden>←</span> How LLMs Work
        </Link>
        <span className="inline-flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-[color:var(--text-dim)]">
          <span
            aria-hidden
            className="h-1.5 w-1.5 rounded-full bg-primary shadow-[0_0_8px_var(--book-pink)]"
          />
          Jacobian lens · on-device
        </span>
        <h1 className="font-display text-4xl font-normal leading-[1.05] tracking-tight text-foreground md:text-5xl">
          J-<span className="italic text-primary">Space</span>
        </h1>
        <p className="max-w-[48ch] text-sm leading-relaxed text-[color:var(--text-dim)]">
          Every layer’s guess, at every position — computed on your device.
        </p>
      </header>

      {!modelReady ? (
        <section aria-labelledby="jspace-consent" className="jspace-panel space-y-3 p-5">
          <h2 id="jspace-consent" className="font-display text-lg font-normal tracking-tight text-foreground">
            Run the model on your device
          </h2>
          <p className="text-[13px] leading-relaxed text-[color:var(--text-dim)]">
            Reading your own prompt downloads about <strong className="font-medium text-foreground">1.6 GB</strong>{' '}
            of model weights, and a further{' '}
            <strong className="font-medium text-foreground">46 MB</strong> the first time you switch to the
            Jacobian lens. Nothing is downloaded until you press Run. The starter grid below needs no model.
          </p>
          {modelStatus === 'loading' ? (
            <p
              className="font-mono text-[11px] uppercase tracking-[0.16em] text-[color:var(--text-dim)]"
              role="status"
              aria-live="polite"
            >
              {loadingText || 'Loading Qwen3.5-0.8B…'}
            </p>
          ) : (
            <button
              type="button"
              onClick={() => kickoffLoad()}
              className="inline-flex items-center gap-2 rounded-lg border border-primary/40 bg-primary/10 px-4 py-2 text-sm font-medium text-primary shadow-[inset_0_1px_0_rgba(255,255,255,0.06)] transition-colors hover:border-primary/60 hover:bg-primary/15 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50"
            >
              Download and run (~1.6 GB)
            </button>
          )}
        </section>
      ) : null}

      {/* Prompt editor + controls */}
      <section className="jspace-panel space-y-4 p-5">
        <label
          htmlFor="jspace-prompt"
          className="block font-mono text-[10px] uppercase tracking-[0.18em] text-[color:var(--text-dim)]"
        >
          Prompt (raw text — no chat template)
        </label>
        <textarea
          id="jspace-prompt"
          value={prompt}
          onChange={(e) => {
            setPrompt(e.target.value);
            // Editing the prompt is a user divergence — cancel any pending
            // permalink-restored selection so a later run can't resurrect it (F3).
            pendingSelRef.current = null;
          }}
          onKeyDown={(e) => {
            // Enter runs; Shift+Enter inserts a newline. Runs fire on Enter,
            // never on every keystroke.
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              if (running || activating || preparing) return;
              void handleRun();
            }
          }}
          rows={2}
          placeholder="Type a prompt, then press Enter to read every layer…"
          className="w-full resize-y rounded-lg border border-border bg-background/50 p-3.5 font-mono text-[13px] leading-relaxed text-foreground placeholder:text-[color:var(--text-muted)] shadow-inner transition-colors focus-visible:border-primary/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
        />
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <button
            type="button"
            onClick={() => void handleRun()}
            disabled={running || activating || preparing}
            className="inline-flex items-center gap-2 rounded-lg border border-primary/40 bg-primary/10 px-4 py-2 text-sm font-medium text-primary shadow-[inset_0_1px_0_rgba(255,255,255,0.06)] transition-colors hover:border-primary/60 hover:bg-primary/15 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50"
          >
            {!modelReady ? 'Download & run' : running ? 'Reading layers…' : 'Run'}
          </button>

          <SegmentedToggle
            value={mode}
            onChange={handleModeChange}
            ariaLabel="Lens mode"
            disabled={running || activating || preparing}
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
          <span className="ml-auto font-mono text-[11px] tabular-nums text-[color:var(--text-dim)]">
            {prompt.length} chars ·{' '}
            <span className={tokenCount !== null && tokenCount > LENS_MAX_POSITIONS ? 'font-semibold text-destructive' : ''}>
              {tokenCount ?? '—'} / {LENS_MAX_POSITIONS} tokens
            </span>
          </span>
        </div>

        {/* Jacobian activation status / self-test verdict */}
        {activating ? (
          <p
            className="font-mono text-[11px] uppercase tracking-[0.16em] text-[color:var(--text-dim)]"
            role="status"
            aria-live="polite"
          >
            Loading the fitted-Jacobian lens pack (~46 MB) and self-testing against the baked reference…
          </p>
        ) : jac.status === 'ok' ? (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-[color:var(--qwen)]/45 bg-[color:var(--qwen)]/12 px-2.5 py-1 font-mono text-[11px] text-[color:var(--qwen-light)]">
            fitted Jacobian · self-test passed (agreement {jac.verdict.topOneAgreement.toFixed(2)}, best-rank Δ{' '}
            {jac.verdict.worstPinDelta})
          </span>
        ) : jac.status === 'failed' ? (
          <p className="text-[12px] leading-relaxed text-destructive" role="alert">
            <strong>Jacobian self-test failed.</strong> The browser could not reproduce the baked reference
            (top-1 agreement {jac.verdict.topOneAgreement.toFixed(4)}, worst pinned best-rank delta{' '}
            {jac.verdict.worstPinDelta}). The fitted-Jacobian badge is withheld — treat any Jacobian readout
            with suspicion.
          </p>
        ) : jac.status === 'error' ? (
          <p className="text-[12px] leading-relaxed text-destructive" role="alert">
            <strong>Jacobian lens unavailable.</strong> {jac.message}
          </p>
        ) : jac.status === 'idle' && modelReady ? (
          <p className="text-[12px] leading-relaxed text-[color:var(--text-dim)]">
            Switching to the Jacobian lens downloads a ~46 MB fitted-lens pack (once).
          </p>
        ) : null}

        {runError ? (
          <p className="text-[12px] leading-relaxed text-destructive" role="alert">
            <strong>Lens read failed.</strong> {runError}
          </p>
        ) : lensRun.state.status === 'error' ? (
          <p className="text-[12px] leading-relaxed text-destructive" role="alert">
            <strong>Lens read failed.</strong> {lensRun.state.message}
          </p>
        ) : null}
      </section>

      {/* Gallery launcher — ALWAYS available (model-free baked frames). Rendered
          unconditionally so it persists after the user types a custom prompt; the
          cards highlight only WHILE a starter is displayed (launcher, not mirror). */}
      <section aria-labelledby="jspace-gallery" className="space-y-2">
        <span
          id="jspace-gallery"
          className="font-mono text-[10px] uppercase tracking-[0.18em] text-[color:var(--text-dim)]"
        >
          {copy.galleryLabel}
        </span>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {STARTER_SLUGS.map((slug) => {
            const active = view.kind === 'starter' && slug === starterSlug;
            return (
              <button
                key={slug}
                type="button"
                aria-pressed={active}
                onClick={() => openStarter(slug)}
                className={[
                  'rounded-lg border p-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                  active
                    ? 'border-primary/50 bg-primary/10'
                    : 'border-border/60 bg-card/30 hover:border-primary/40 hover:bg-primary/5',
                ].join(' ')}
              >
                <span className="block font-display text-sm text-foreground">{copy.presetNames[slug] ?? slug}</span>
                <span className="mt-0.5 block text-[11px] leading-snug text-[color:var(--text-dim)]">
                  {copy.galleryHooks[slug]}
                </span>
              </button>
            );
          })}
        </div>
        {/* D1 — per-example blurb caption, bilingual (from JSPACE_COPY). Only WHILE
            the model-free starter grid is displayed; the empirical claim for the
            selected example, kept honest (WEAK tiles say "Faint…"/"微弱…"). */}
        {view.kind === 'starter' && galleryEntry ? (
          <p className="max-w-[62ch] text-[12px] leading-relaxed text-[color:var(--text-dim)]">
            {copy.blurbs[galleryEntry.slug]}
          </p>
        ) : null}
      </section>

      {/* The main grid + panels */}
      {view.kind === 'skeleton' || !slice ? (
        <section className="space-y-4">
          <div
            className="jspace-panel p-4 font-mono text-[13px] leading-relaxed text-foreground"
            style={{ whiteSpace: 'pre-wrap' }}
          >
            {prompt}
          </div>
          {pins.length > 0 ? (
            <p className="text-[12px] leading-relaxed text-[color:var(--text-dim)]">
              Restored pins: {pins.map((id) => `#${id}`).join(', ')} — run to see their rank tracks.
            </p>
          ) : null}
          <div className="flex min-h-[10rem] flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border/70 bg-card/30 p-10 text-center">
            <p className="font-display text-lg font-normal tracking-tight text-foreground">Run to compute</p>
            <p className="text-[13px] text-[color:var(--text-dim)]">
              Press Run to read every layer for this prompt.
            </p>
          </div>
        </section>
      ) : (
        <section className="space-y-6">
          {/* Prompt tokens = position axis */}
          <div className="space-y-2">
            <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-[color:var(--text-dim)]">
              Prompt · position axis
            </span>
            <PromptTokens
              tokens={slice.tokens}
              selectedPos={activeCellRef?.pos ?? null}
              showWhitespace={showWhitespace}
              onSelectPos={(pos) => {
                setHovered(null); // a deliberate position pick wins over a stale hover
                setSelected((prev) => ({ layerIdx: prev?.layerIdx ?? slice.layers.length - 1, pos }));
                // A deliberate position pick is a user divergence — drop any
                // pending permalink-restored selection (F3).
                pendingSelRef.current = null;
              }}
            />
          </div>

          {/* Argmax grid (instrument) docked beside the per-cell readout on lg.
              SOURCE ORDER stays grid-then-detail so selectCell / aria-activedescendant
              / the permalink one-shot restore are unaffected; mobile stacks. */}
          <div className="lg:grid lg:grid-cols-[minmax(0,1fr)_19rem] lg:gap-6 lg:items-start">
            {/* LEFT — the argmax grid panel. min-w-0 keeps the inner
                .jspace-grid-scroll horizontally scrollable instead of blowing out. */}
            <div className="jspace-panel min-w-0 space-y-3 p-4 lg:p-5">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-[color:var(--text-dim)]">
                  Per-layer top token · deepest at top
                </span>
                {view.kind === 'starter' ? (
                  <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.16em] text-[color:var(--text-dim)]">
                    baked starter
                  </span>
                ) : slice.jacobianApplied ? (
                  <span className="rounded-full border border-[color:var(--qwen)]/45 bg-[color:var(--qwen)]/12 px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.16em] text-[color:var(--qwen-light)]">
                    fitted Jacobian
                  </span>
                ) : (
                  <span className="rounded-full border border-primary/40 bg-primary/10 px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.16em] text-primary">
                    logit lens
                  </span>
                )}
              </div>
              <div className="overflow-hidden rounded-lg border border-border/60">
                <ArgmaxGridCanvas
                  slice={slice}
                  colorByPinnedId={colorByPinnedId}
                  selected={selected}
                  onHover={setHovered}
                  onSelect={selectCell}
                  showWhitespace={showWhitespace}
                  band={band}
                  ariaLabel={`Argmax grid: ${slice.layers.length} residual boundaries as rows (deepest on top) by ${slice.promptLen} prompt positions as columns.`}
                />
              </div>
            </div>

            {/* RIGHT — the per-cell readout, sticky within the page scroller on lg. */}
            <div className="mt-4 lg:mt-0 lg:sticky lg:top-6">
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
                <p className="flex min-h-[6rem] items-center justify-center rounded-lg border border-dashed border-border/60 bg-card/30 p-4 text-center font-mono text-[11px] leading-relaxed text-[color:var(--text-dim)]">
                  Hover or tap any cell to see that layer’s top-K read.
                </p>
              )}
            </div>
          </div>

          {/* Pins + rank field + rank charts */}
          <div className="space-y-4">
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
              <div className="space-y-5">
                {/* Rank heatmap — full-width panel; the flex item carries min-w-0
                    flex-1 so the intrinsic 8750px canvas stays inside its own
                    .jspace-grid-scroll scroller instead of overflowing the page (F6). */}
                <div className="flex flex-col gap-5 lg:flex-row lg:items-start">
                  <div className="jspace-panel min-w-0 flex-1 space-y-2 p-4">
                    <div className="jspace-eyebrow">
                      Rank of “{cleanupTokenText(pinnedForView[effectiveActiveIdx]?.tokenText ?? '')}” by position × layer
                    </div>
                    <RankHeatmapCanvas
                      slice={slice}
                      pinnedIdx={effectiveActiveIdx}
                      selected={selected}
                      onSelect={selectCell}
                    />
                  </div>
                </div>

                {/* Rank-vs-depth and rank-vs-position, a matched pair below (F5). */}
                <div className="flex flex-col gap-5 lg:flex-row lg:items-start">
                  <div className="jspace-panel min-w-0 flex-1 space-y-2 p-4">
                    <div className="jspace-eyebrow">Pinned-token rank by depth (position {chartPos + 1})</div>
                    <RankChart
                      points={chartPoints}
                      colors={chartColors}
                      xLabel="residual boundary ℓ"
                      selectedX={selectedLayerNum}
                    />
                  </div>
                  <div className="jspace-panel min-w-0 flex-1 space-y-2 p-4">
                    <div className="jspace-eyebrow">
                      Pinned-token rank across positions (ℓ{selectedLayerNum ?? posChartLayerNum})
                    </div>
                    <RankChart
                      points={posChartPoints}
                      colors={chartColors}
                      xLabel="prompt position"
                      selectedX={(activeCellRef?.pos ?? 0) + 1}
                    />
                  </div>
                </div>
              </div>
            ) : null}
          </div>

          {/* Cross-sections at the selected cell */}
          {activeCellRef ? (
            <div className="grid gap-5 lg:grid-cols-2">
              <ByLayerStrip slice={slice} pos={activeCellRef.pos} selected={selected} onSelect={selectCell} />
              <ByPosStrip slice={slice} layerIdx={activeCellRef.layerIdx} selected={selected} onSelect={selectCell} />
            </div>
          ) : null}
        </section>
      )}
    </main>
  );
}
