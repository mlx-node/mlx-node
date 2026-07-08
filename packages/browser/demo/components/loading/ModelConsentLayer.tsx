// ModelConsentLayer — the consent gate for a chapter's live "Try it now" demo.
//
// The chapter body/prose always renders without the model. This layer wraps the
// model-dependent demo panel and keeps the ~1.6 GB download (or the WebGPU
// device init) behind an EXPLICIT user click. Nothing auto-downloads.
//
// States (see selectConsentLayerState):
//   ready   — the demo's resource is up → render {children} (the live demo).
//             We mount the live demo ONLY when ready so it never sees a null
//             worker at mount (the existing demos assume a live worker).
//   prompt  — frozen panel: a one-line description + a primary CTA button that
//             kicks off the load (or opens the local-model picker when no
//             hosted model is available).
//   loading — spinner + loadingText + progress (reuses <PanelLoading>); CTA
//             disabled.
//   error   — the error message + a Retry button that re-triggers the kickoff.

import type { ReactNode } from 'react';

import type { Locale } from '../../lib/i18n';
import { useLocale } from '../../lib/i18n-react';
import { triggerLocalPicker } from '../../lib/local-model-picker';
import { type ConsentMode, selectConsentLayerState } from '../../lib/model-loader-state';
import { useModelLoader } from '../../providers/model-loader';
import { PanelLoading } from './PanelLoading';

export type ModelConsentLayerProps = {
  /**
   * Which resource the wrapped demo needs:
   *   "model"  — the full ~1.6 GB model (gates on status === 'ready').
   *   "device" — only the WebGPU device + WASM runtime, no download (gates on
   *              deviceReady; a full 'ready' implies the device is up too).
   */
  mode: ConsentMode;
  /** The live demo. Mounted ONLY when the resource is ready. */
  children: ReactNode;
};

type ConsentCopy = {
  title: string;
  subtext: string;
  cta: string;
};

const COPY: Record<Locale, Record<ConsentMode, ConsentCopy>> = {
  en: {
    model: {
      title: 'Run this live in your browser',
      subtext: 'Loads the real Qwen3.5-0.8B (~1.6 GB) — runs entirely on your device; nothing leaves your browser.',
      cta: 'Load model',
    },
    device: {
      title: 'Start the in-browser trainer',
      subtext: 'Initializes the WebGPU device in your browser — no model download.',
      cta: 'Initialize',
    },
  },
  zh: {
    model: {
      title: '在你的浏览器里实时运行',
      subtext: '加载真实的 Qwen3.5-0.8B（约 1.6 GB）——完全在你的设备上运行，数据不会离开浏览器。',
      cta: '加载模型',
    },
    device: {
      title: '启动浏览器内训练器',
      subtext: '在浏览器中初始化 WebGPU 设备——无需下载模型。',
      cta: '初始化',
    },
  },
};

// Status/error chrome around the consent panel, per locale.
const CHROME = {
  en: {
    hostedUnavailable:
      'This live demo needs the model. No hosted model is available here — choose a local model directory to run it. Everything runs on your device.',
    chooseLocal: 'Choose local model',
    preparing: 'Preparing the model…',
    errorTitle: 'Couldn’t load the model',
    errorFallback: 'Something went wrong while loading. Please try again.',
    retry: 'Retry',
  },
  zh: {
    hostedUnavailable: '这个在线演示需要模型。当前没有可用的托管模型——请选择本地模型目录来运行。一切都在你的设备上进行。',
    chooseLocal: '选择本地模型',
    preparing: '正在准备模型……',
    errorTitle: '模型加载失败',
    errorFallback: '加载过程中出了点问题，请重试。',
    retry: '重试',
  },
} satisfies Record<Locale, Record<string, string>>;

export function ModelConsentLayer({ mode, children }: ModelConsentLayerProps) {
  const locale = useLocale();
  const {
    status,
    deviceReady,
    loadKickoff,
    loadingText,
    loadingProgress,
    errorBanner,
    hostedModelAvailable,
    kickoffLoad,
    kickoffDeviceOnly,
  } = useModelLoader();

  const state = selectConsentLayerState({ mode, status, deviceReady, loadKickoff });

  // Ready — mount the real demo. This is the only branch that renders children,
  // preserving the invariant that demos see a live worker at mount.
  if (state === 'ready') {
    return <>{children}</>;
  }

  const copy = COPY[locale][mode];
  const chrome = CHROME[locale];

  // A model-mode demo with no hosted model available can't auto-download — the
  // user must point at a local model directory. The device mode never fetches
  // the model, so it is unaffected.
  const hostedUnavailable = mode === 'model' && hostedModelAvailable === false;

  // This layer only ever wraps the chapter route's demo, which AUTO-STARTS the
  // load on entry (see routes/chapters.$chapterId.tsx). A 'prompt' state here
  // therefore does NOT mean "click to begin" — the load is already coming, or
  // the hosted-model probe (hostedModelAvailable === null) is still resolving.
  // Showing a "Load model" CTA in that window is misleading (the user thinks a
  // click is required) and is the source of the "I still had to click" report.
  // Render the loading spinner instead — EXCEPT when there is genuinely no
  // hosted model to auto-fetch (hostedUnavailable): that path needs a manual
  // local-model-directory pick (a real user gesture), so it keeps the CTA.
  const effectiveState = state === 'prompt' && !hostedUnavailable ? 'loading' : state;

  // The kickoff for this mode's CTA. For model mode without a hosted model, we
  // open the local-model directory picker instead.
  const kickoff = () => {
    if (mode === 'device') {
      kickoffDeviceOnly();
      return;
    }
    if (hostedUnavailable) {
      triggerLocalPicker();
      return;
    }
    kickoffLoad();
  };

  // Loading — reuse the existing PanelLoading visuals (spinner + progress +
  // aria-live announcement). The CTA is implicitly "disabled" because we render
  // the spinner instead of the button. `loadingText` is empty in the brief
  // pre-kickoff/probe window, so fall back to a neutral "Preparing…" label.
  if (effectiveState === 'loading') {
    return <PanelLoading status={loadingText || chrome.preparing} progress={loadingProgress} />;
  }

  // Error — surface the message + a Retry that re-triggers the kickoff.
  if (effectiveState === 'error') {
    return (
      <div className="flex flex-col items-start gap-3 rounded-lg border border-destructive/50 bg-destructive/5 p-4">
        <p className="text-sm font-medium text-foreground">{chrome.errorTitle}</p>
        <p className="text-xs leading-relaxed text-muted-foreground" role="alert">
          {errorBanner ?? chrome.errorFallback}
        </p>
        <button
          type="button"
          onClick={kickoff}
          className="rounded-lg border border-border bg-muted px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-muted/70"
        >
          {chrome.retry}
        </button>
      </div>
    );
  }

  // Prompt — the frozen consent panel. Real <button>, focusable, with an
  // aria-live region kept present (empty here) to mirror PanelLoading's
  // announcement contract once loading starts.
  return (
    <div className="flex flex-col items-start gap-3 rounded-lg border border-border bg-muted/20 p-5">
      <p className="text-sm font-medium text-foreground">{copy.title}</p>
      <p className="max-w-prose text-xs leading-relaxed text-muted-foreground">
        {hostedUnavailable ? chrome.hostedUnavailable : copy.subtext}
      </p>
      <button
        type="button"
        onClick={kickoff}
        className="rounded-lg border border-primary/50 bg-primary/10 px-4 py-2 text-sm font-medium text-primary transition-colors hover:bg-primary/20 focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
      >
        {hostedUnavailable ? chrome.chooseLocal : copy.cta}
      </button>
      {/* Polite aria-live region: empty in the prompt state, but present so the
          transition into loading (announced by <PanelLoading>) has a stable
          landmark. */}
      <span className="sr-only" role="status" aria-live="polite" />
    </div>
  );
}
