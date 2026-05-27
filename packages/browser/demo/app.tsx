import type { ChatStreamChunk, ToolCallResult, ToolDefinition } from '@mlx-node/core';
import { ToolCallTagBuffer } from '@mlx-node/lm/tools';
import { createCodePlugin } from '@streamdown/code';
import { type ChangeEvent, useEffect, useReducer, useRef, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Streamdown } from 'streamdown';

import { createSabRingOverHeap } from '../src/chat-stream-sab.js';
import mlxWorkerUrl from '../src/mlx-worker.ts?worker&url';
import {
  couldStillBeReasoningTextPrefix,
  looksLikeReasoningText,
  sanitizeAssistantText,
  sanitizeThinkingText as sanitizeThinkingMarkup,
  splitAssistantThinking,
} from '../src/generated-text.js';
import { workerAssetUrl } from '../src/worker-asset-url.js';
import { ChatHeader } from './components/chat/ChatHeader';
import { InlinePreviewCard } from './components/chat/InlinePreviewCard';
import { InspectorDrawer } from './components/chat/InspectorDrawer';
import { PowerComposer } from './components/chat/PowerComposer';
import { TelemetryStrip } from './components/chat/TelemetryStrip';
import { Landing } from './components/landing/Landing';
import { Loading, type LoadingProgress } from './components/loading/Loading';
import { BrowserChatSessionAdapter, type BrowserChatMessage } from './lib/browser-chat-session';
import {
  type ScreenState,
  type ProfileLikeStats,
  cycleReasoningEffort,
  parseScreenFromUrl,
  reduceScreen,
  screenToUrlQuery,
} from './lib/screen-state';
import { ChapterIndex } from './learn/ChapterIndex';
import { LessonLayout } from './learn/LessonLayout';
import { findChapter } from './learn/chapters';
import { AttentionChapterBody, AttentionDemo } from './learn/chapters/03-attention';
import { MultiheadGqaChapterBody, MultiheadGqaDemo } from './learn/chapters/04-multihead-gqa';
import { TokenizationChapterBody, TokenizerDemo } from './learn/chapters/01-tokenization';
import { EmbeddingsChapterBody, EmbeddingsDemo } from './learn/chapters/02-embeddings';
import { RopeChapterBody, RopeDemo } from './learn/chapters/05-rope';
import { RmsNormChapterBody, RmsNormDemo } from './learn/chapters/06-rms-norm';
import { MlpChapterBody, MlpDemo } from './learn/chapters/07-mlp';
import { FullBlockChapterBody, FullBlockDemo } from './learn/chapters/08-full-block';
import { SamplingChapterBody, SamplingDemo } from './learn/chapters/09-sampling';
import { KvCacheChapterBody, KvCacheDemo } from './learn/chapters/10-kv-cache';

const DEFAULT_INITIAL_SCREEN: ScreenState = { kind: 'landing' };

// useReducer's lazy-init: read the URL once at boot so deep links like
// /?screen=chapter&chapterId=tokenization land on the right screen directly
// instead of bouncing through landing. Invalid / transient URLs fall back to
// landing — see parseScreenFromUrl for the validation rules.
function getInitialScreen(): ScreenState {
  return parseScreenFromUrl() ?? DEFAULT_INITIAL_SCREEN;
}
import 'streamdown/styles.css';
import './styles.css';

type StatusState = 'info' | 'ready' | 'error';
type ReasoningEffort = 'off' | 'low' | 'medium' | 'high';
const DEFAULT_MODEL_LABEL = 'qwen3.5-0.8b-mlx-bf16';
const DEFAULT_MODEL_URL = '/model';
const MAX_BROWSER_OUTPUT_TOKENS = 36864;
const DEFAULT_BROWSER_OUTPUT_TOKENS = 16834;
const DEFAULT_BROWSER_TEMPERATURE = 0.6;
const APP_PREVIEW_TOOL_NAME = 'create_app_preview';
const MAX_TOOL_CONTINUATIONS = 2;
const AUTO_CONTINUE_AFTER_APP_PREVIEW = false;
const BASE_SYSTEM_PROMPT = 'You are a helpful assistant. Be concise.';
const APP_PREVIEW_SYSTEM_PROMPT = [
  BASE_SYSTEM_PROMPT,
  `When tools are enabled and the user asks you to build, write, prototype, preview, or render an app, UI, animation, game, or HTML/CSS/JS demo, call ${APP_PREVIEW_TOOL_NAME} exactly once with a complete self-contained runnable HTML document in the html argument.`,
  'Do not print raw HTML/CSS/JS outside the tool call.',
].join(' ');
const streamdownPlugins = {
  code: createCodePlugin({
    themes: ['github-dark-high-contrast', 'github-dark-high-contrast'],
  }),
};

function systemPromptForTools(toolsEnabled: boolean) {
  return toolsEnabled ? APP_PREVIEW_SYSTEM_PROMPT : BASE_SYSTEM_PROMPT;
}

type ModelSource = {
  modelFiles?: File[];
  modelUrl?: string;
  label?: string;
};

function normalizeModelBaseUrl(value: string | undefined | null) {
  const trimmed = value?.trim() || DEFAULT_MODEL_URL;
  return trimmed.replace(/\/+$/g, '') || DEFAULT_MODEL_URL;
}

function resolveConfiguredModelUrl(params: URLSearchParams) {
  return normalizeModelBaseUrl(
    params.get('model_url') ??
      params.get('modelUrl') ??
      params.get('model') ??
      (import.meta.env.VITE_MLX_MODEL_URL as string | undefined),
  );
}

function resolveConfiguredModelLabel(params: URLSearchParams) {
  return (
    params.get('model_label') ??
    params.get('modelLabel') ??
    (import.meta.env.VITE_MLX_MODEL_LABEL as string | undefined) ??
    DEFAULT_MODEL_LABEL
  );
}

function modelAssetUrl(modelUrl: string, assetPath: string) {
  const base = `${normalizeModelBaseUrl(modelUrl)}/`;
  return new URL(
    assetPath
      .replace(/^\/+/g, '')
      .split('/')
      .map((part) => encodeURIComponent(part))
      .join('/'),
    new URL(base, location.href),
  ).href;
}

async function hasHostedModel(modelUrl: string): Promise<boolean> {
  try {
    const resp = await fetch(`${modelAssetUrl(modelUrl, 'config.json')}?probe=${Date.now()}`, {
      cache: 'no-store',
      headers: { Accept: 'application/json' },
    });
    if (!resp.ok) return false;
    const contentType = resp.headers.get('content-type') ?? '';
    const text = await resp.text();
    if (contentType.includes('text/html')) return false;
    return !text.trimStart().startsWith('<!doctype');
  } catch {
    return false;
  }
}

function isLocalDevHost() {
  return location.hostname === 'localhost' || location.hostname === '127.0.0.1' || location.hostname === '::1';
}

function modelSourceFromLocalFiles(files: File[]): ModelSource | null {
  if (files.length === 0) return null;
  const firstPath = (files[0] as File & { webkitRelativePath?: string }).webkitRelativePath || files[0]!.name;
  const label = firstPath.split('/')[0] || 'local model';
  return { modelFiles: files, label };
}

function positiveFiniteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null;
}

function nonNegativeFiniteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
}

function clampPct(value: number) {
  return Math.max(0, Math.min(100, value));
}

function loadingProgressFromWorker(data: Record<string, unknown>): LoadingProgress | null {
  const loadedBytes = nonNegativeFiniteNumber(data.loadedBytes);
  const totalBytes = nonNegativeFiniteNumber(data.totalBytes);
  const explicitPct = nonNegativeFiniteNumber(data.pct);
  const pct =
    explicitPct != null
      ? clampPct(explicitPct)
      : loadedBytes != null && totalBytes != null && totalBytes > 0
        ? clampPct((loadedBytes / totalBytes) * 100)
        : null;

  if (pct == null) {
    return null;
  }

  const file = typeof data.file === 'string' && data.file.trim() ? data.file : undefined;
  const cacheSource = typeof data.cacheSource === 'string' && data.cacheSource.trim() ? data.cacheSource : undefined;

  return {
    pct,
    ...(loadedBytes != null ? { loadedBytes } : {}),
    ...(totalBytes != null ? { totalBytes } : {}),
    ...(file ? { file } : {}),
    ...(cacheSource ? { cacheSource } : {}),
  };
}

function pushVisibleToolText(buffer: ToolCallTagBuffer, delta: string) {
  const { safeText, tagFound, cleanPrefix } = buffer.push(delta);
  return tagFound ? cleanPrefix : safeText;
}

const APP_PREVIEW_TOOL: ToolDefinition = {
  type: 'function',
  function: {
    name: APP_PREVIEW_TOOL_NAME,
    description:
      'Create or update the live browser preview when the user asks to build, write, prototype, preview, or render an app, UI, animation, game, or HTML/CSS/JS demo. For those requests, call this function instead of returning a markdown code block.',
    parameters: {
      type: 'object',
      properties: JSON.stringify({
        title: {
          type: 'string',
          description: 'Short display title for the preview.',
        },
        html: {
          type: 'string',
          description:
            'Complete self-contained runnable HTML document for the preview iframe. Include all required markup and inline <style> and <script> unless separate css/js fields make the call shorter.',
        },
        css: {
          type: 'string',
          description: 'Optional CSS for the app. Keep it self-contained and responsive.',
        },
        js: {
          type: 'string',
          description: 'Optional client-side JavaScript. Do not use external dependencies.',
        },
      }),
      required: ['title', 'html'],
    },
  },
};

const DIRECTORY_PICKER_INPUT_PROPS = {
  webkitdirectory: '',
  directory: '',
} as const;

type ChatResult = {
  text?: string;
  rawText?: string;
  thinking?: string | null;
  finishReason?: string | null;
  toolCalls?: ToolCallResult[];
  numTokens?: number;
  performance?: {
    ttftMs: number;
    prefillTokensPerSecond: number;
    decodeTokensPerSecond: number;
  } | null;
};

type ProfileStats = {
  numTokens: number;
  totalDispatches: number;
  totalPassEnds: number;
  bridgeRpcCount: number;
  bridgeByFn: Record<number, number>;
  gpuRpcCount: number;
  gpuByFn: Record<number, number>;
  poolHits?: number;
  poolMisses?: number;
  gpuPoolHits?: number;
  gpuPoolMisses?: number;
  bindGroupCacheHits?: number;
  bindGroupCacheMisses?: number;
  uniformHotHits?: number;
  uniformHotMisses?: number;
  spinHits?: number;
  spinMisses?: number;
  spinBudget?: number;
  diagCreateAll?: number;
  diagCreateMappedCopyDst?: number;
  diagCreateMappedNoCopyDst?: number;
  diagReleaseAll?: number;
  diagReleaseUnknownHandle?: number;
  diagReleaseUnpoolable?: number;
  diagPoolEvictions?: number;
  diagBatchAttempt?: number;
  diagBatchStaged?: number;
  diagBatchDeferredBlock?: number;
  diagBatchStageRefused?: number;
};

function App() {
  const statusRef = useRef<HTMLSpanElement>(null);
  const chatRef = useRef<HTMLDivElement>(null);
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const sendRef = useRef<HTMLButtonElement>(null);
  const imageButtonRef = useRef<HTMLButtonElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const modelDirInputRef = useRef<HTMLInputElement>(null);
  const temperatureInputRef = useRef<HTMLInputElement>(null);
  const maxOutputTokensInputRef = useRef<HTMLInputElement>(null);
  const reasoningEffortRef = useRef<ReasoningEffort>('off');
  const initialUrlParams = new URLSearchParams(location.search);
  const configuredModelUrl = resolveConfiguredModelUrl(initialUrlParams);
  const configuredModelLabel = resolveConfiguredModelLabel(initialUrlParams);
  const initialAppToolsEnabled = initialUrlParams.get('tools') === '1' || initialUrlParams.get('app_preview') === '1';
  const [appToolsEnabled, setAppToolsEnabledState] = useState(initialAppToolsEnabled);
  const [reasoningEffort, setReasoningEffortState] = useState<ReasoningEffort>('off');
  const [screen, dispatchScreen] = useReducer(reduceScreen, undefined, getInitialScreen);
  const [loadKickoff, setLoadKickoff] = useState(0);
  const [loadingText, setLoadingText] = useState<string | null>(null);
  const [loadingProgress, setLoadingProgress] = useState<LoadingProgress | null>(null);
  const [errorBanner, setErrorBannerState] = useState<string | null>(null);
  const [telemetryStats, setTelemetryStats] = useState<ProfileLikeStats | null>(null);
  const [prefillTokensPerSec, setPrefillTokensPerSec] = useState<number | null>(null);
  const [decodeTokensPerSec, setDecodeTokensPerSec] = useState<number | null>(null);
  const [pendingModelSource, setPendingModelSource] = useState<ModelSource | null>(null);
  const [modelLine, setModelLine] = useState<string>(configuredModelLabel);
  const [hostedModelAvailable, setHostedModelAvailable] = useState<boolean | null>(null);
  const appToolsEnabledRef = useRef(initialAppToolsEnabled);
  const initialMaxOutputTokens = Math.min(
    MAX_BROWSER_OUTPUT_TOKENS,
    Math.max(
      1,
      Number.parseInt(
        initialUrlParams.get('max_new_tokens') ??
          initialUrlParams.get('maxOutputTokens') ??
          `${DEFAULT_BROWSER_OUTPUT_TOKENS}`,
        10,
      ) || DEFAULT_BROWSER_OUTPUT_TOKENS,
    ),
  );
  const parsedInitialTemperature = Number.parseFloat(
    initialUrlParams.get('temperature') ?? initialUrlParams.get('temp') ?? `${DEFAULT_BROWSER_TEMPERATURE}`,
  );
  const initialTemperature = Number.isFinite(parsedInitialTemperature)
    ? Math.min(2, Math.max(0, parsedInitialTemperature))
    : DEFAULT_BROWSER_TEMPERATURE;
  const [temperatureValue, setTemperatureValue] = useState(initialTemperature);
  const [maxTokensValue, setMaxTokensValue] = useState(initialMaxOutputTokens);
  const temperatureValueRef = useRef(initialTemperature);
  const maxTokensValueRef = useRef(initialMaxOutputTokens);
  const [generating, setGeneratingState] = useState(false);
  const [sendDisabled, setSendDisabledState] = useState(true);
  // Shared handle to the MLX inference worker so chapter components can post
  // their own `runForInspector` requests without rerouting through the main
  // chat path. The worker itself is owned by the load-model effect below;
  // this ref is populated when the worker comes up and nulled on teardown.
  const mlxWorkerRef = useRef<Worker | null>(null);
  // Abort signal for in-flight inspector calls. The load-model effect's
  // cleanup function aborts this BEFORE calling `worker.terminate()`, so any
  // chapter-side `runForInspector` Promise rejects within a tick instead of
  // hanging the full 60s timeout waiting for a reply from a dead worker.
  // A fresh controller is installed for each kickoff cycle.
  const inspectorAbortRef = useRef<AbortController | null>(null);
  // Prompt currently being inspected in the chat drawer. `null` = drawer
  // closed. Set when the user clicks the "Inspect" button on a finished
  // assistant message; the InspectorDrawer then re-runs the model on this
  // exact prompt with attention + top-K logits captured.
  const [inspectedPrompt, setInspectedPrompt] = useState<string | null>(null);

  useEffect(() => {
    const modelDirInput = modelDirInputRef.current;
    if (!modelDirInput) return;
    modelDirInput.setAttribute('webkitdirectory', '');
    modelDirInput.setAttribute('directory', '');
  }, []);

  // URL <-> screen state bridge. Single source of truth for writes lives here:
  // every screen change calls history.replaceState so a reload restores the
  // current screen. We use replaceState (not pushState) because intra-app
  // navigation should not bloat the back stack — the user's back button
  // takes them OUT of the app, not through every chapter they viewed.
  // The `loading` screen is skipped (screenToUrlQuery returns null) so
  // reloading mid-load doesn't restore an unrecoverable transient state.
  useEffect(() => {
    const query = screenToUrlQuery(screen);
    if (query === null) return;
    try {
      const existing = window.location.search.startsWith('?')
        ? window.location.search.slice(1)
        : window.location.search;
      if (existing === query) return;
      const newUrl = `${window.location.pathname}?${query}`;
      window.history.replaceState(null, '', newUrl);
    } catch {
      // history.replaceState can throw in sandboxed iframes / very restricted
      // contexts. The app must keep working even if URL sync is impossible.
    }
  }, [screen]);

  // Browser back/forward → re-parse the URL and restore via the `restore`
  // reducer event. Bound once at mount with an empty dep array; the handler
  // reads from window.location at call time so the closure stays trivial.
  useEffect(() => {
    const handler = () => {
      const restored = parseScreenFromUrl();
      dispatchScreen({ type: 'restore', screen: restored ?? DEFAULT_INITIAL_SCREEN });
    };
    window.addEventListener('popstate', handler);
    return () => window.removeEventListener('popstate', handler);
  }, []);

  useEffect(() => {
    let cancelled = false;
    hasHostedModel(configuredModelUrl).then((available) => {
      if (!cancelled) setHostedModelAvailable(available);
    });
    return () => {
      cancelled = true;
    };
  }, [configuredModelUrl]);

  // Auto-kick-off the model load when the user enters the learning flow —
  // either by clicking "Start learning" on the landing, or by opening a
  // chapter URL directly (bookmark, popstate). Chapters fall back to mock
  // data while the worker isn't up, so without this the user sees "example
  // data" banners until they navigate to chat and click Load Model. We only
  // kickoff once (guarded on loadKickoff === 0); subsequent screen changes
  // reuse the existing worker. We skip kickoff if no hosted model is
  // available — that path requires the user to pick a local model directory
  // and we don't want to auto-open the directory picker.
  useEffect(() => {
    if (loadKickoff !== 0) return;
    if (screen.kind !== 'chapter' && screen.kind !== 'chapter_index') return;
    if (hostedModelAvailable === false) return;
    if (hostedModelAvailable === null) return; // still probing
    setPendingModelSource(null);
    setErrorBannerState(null);
    setLoadingProgress(null);
    setLoadKickoff(1);
  }, [screen.kind, loadKickoff, hostedModelAvailable]);

  function handleLocalModelInputChange(event: ChangeEvent<HTMLInputElement>) {
    const source = modelSourceFromLocalFiles(Array.from(event.currentTarget.files ?? []));
    event.currentTarget.value = '';
    if (!source) return;
    setPendingModelSource(source);
    setErrorBannerState(null);
    setLoadingProgress(null);
    setLoadKickoff((k) => k + 1);
    dispatchScreen({ type: 'load_kickoff' });
  }

  useEffect(() => {
    if (loadKickoff === 0) {
      return;
    }
    const statusEl = statusRef.current!;
    const chatEl = chatRef.current!;
    const promptEl = promptRef.current!;
    const sendBtn = sendRef.current!;
    const imageBtn = imageButtonRef.current!;
    const imageInput = imageInputRef.current!;
    const temperatureInput = temperatureInputRef.current!;
    const maxOutputTokensInput = maxOutputTokensInputRef.current!;

    if (!statusEl || !chatEl || !promptEl || !sendBtn || !imageBtn || !imageInput || !maxOutputTokensInput) {
      return;
    }
    let activeModelLabel = configuredModelLabel;
    let activeChatMaxNewTokens = DEFAULT_BROWSER_OUTPUT_TOKENS;

    if (navigator.storage?.persist) {
      void navigator.storage
        .persist()
        .then((persisted) => {
          log(
            persisted
              ? 'Browser storage persistence granted.'
              : 'Browser storage persistence unavailable; cached models may be evicted under storage pressure.',
          );
        })
        .catch((error) => {
          log(`Browser storage persistence request failed: ${String(error)}`);
        });
    }

    function setStatus(text: string, state: StatusState = 'info', progress: LoadingProgress | null = null) {
      statusEl.textContent = text;
      statusEl.className = `status-pill ${state}`;
      setLoadingText(text);
      setLoadingProgress(progress);
      if (state === 'ready') {
        dispatchScreen({ type: 'model_ready' });
      } else if (state === 'error') {
        setErrorBannerState(text);
        dispatchScreen({ type: 'model_error' });
      }
    }

    function setAppToolsEnabled(enabled: boolean) {
      appToolsEnabledRef.current = enabled;
      setAppToolsEnabledState(enabled);
    }

    function log(msg: string) {
      // eslint-disable-next-line no-console
      console.log(`[mlx] ${msg}`);
    }

    function compactModelLabel(label: string) {
      return label;
    }

    function readMaxOutputTokens() {
      const parsed = Number.parseInt(`${maxTokensValueRef.current}`, 10);
      const clamped = Math.min(
        MAX_BROWSER_OUTPUT_TOKENS,
        Math.max(1, Number.isFinite(parsed) ? parsed : DEFAULT_BROWSER_OUTPUT_TOKENS),
      );
      maxTokensValueRef.current = clamped;
      if (`${clamped}` !== maxOutputTokensInput.value) {
        maxOutputTokensInput.value = `${clamped}`;
      }
      setMaxTokensValue((current) => (current === clamped ? current : clamped));
      return clamped;
    }

    function readTemperature() {
      const parsed = Number.parseFloat(`${temperatureValueRef.current}`);
      const clamped = Math.min(2, Math.max(0, Number.isFinite(parsed) ? parsed : DEFAULT_BROWSER_TEMPERATURE));
      const formatted = Number.isInteger(clamped) ? `${clamped}` : `${Math.round(clamped * 100) / 100}`;
      temperatureValueRef.current = clamped;
      if (formatted !== temperatureInput.value) {
        temperatureInput.value = formatted;
      }
      setTemperatureValue((current) => (current === clamped ? current : clamped));
      return clamped;
    }

    let imageCapabilityKnown = false;
    let supportsImages = false;
    let pendingImage: Uint8Array | null = null;

    function setImageAttached(attached: boolean) {
      imageBtn.dataset.attached = attached ? 'true' : 'false';
      imageBtn.dataset.unsupported = supportsImages ? 'false' : 'true';
      imageBtn.title = supportsImages
        ? attached
          ? 'Image attached. Click to replace'
          : 'Attach image'
        : imageCapabilityKnown
          ? 'Image input unavailable for this model'
          : 'Image input available after model loads';
      imageBtn.setAttribute('aria-label', imageBtn.title);
    }

    function setImageCapability(enabled: boolean) {
      imageCapabilityKnown = true;
      supportsImages = enabled;
      imageBtn.disabled = !enabled;
      if (!enabled) {
        pendingImage = null;
        imageInput.value = '';
      }
      setImageAttached(pendingImage !== null);
    }

    function autosizePrompt() {
      promptEl.style.height = 'auto';
      promptEl.style.height = `${Math.min(promptEl.scrollHeight, 132)}px`;
    }

    function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
      const copy = new Uint8Array(bytes.byteLength);
      copy.set(bytes);
      return copy.buffer;
    }

    function asRecord(value: unknown): Record<string, unknown> {
      if (typeof value === 'string') {
        try {
          const parsed = JSON.parse(value);
          return asRecord(parsed);
        } catch {
          return {};
        }
      }
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        return value as Record<string, unknown>;
      }
      return {};
    }

    function stringArg(args: Record<string, unknown>, key: string) {
      const value = args[key];
      return typeof value === 'string' ? value : '';
    }

    function stripRawToolMarkupForDisplay(value: string | null | undefined) {
      let input = value ?? '';
      let output = '';

      while (input.length > 0) {
        const openIndex = input.indexOf('<tool_call>');
        if (openIndex < 0) {
          output += input;
          break;
        }

        output += input.slice(0, openIndex);
        const bodyStart = openIndex + '<tool_call>'.length;
        const closeIndex = input.indexOf('</tool_call>', bodyStart);
        if (closeIndex < 0) {
          break;
        }
        input = input.slice(closeIndex + '</tool_call>'.length);
      }

      const orphanMarkers = ['<function=', '<parameter='];
      let orphanIndex = -1;
      for (const marker of orphanMarkers) {
        const index = output.indexOf(marker);
        if (index >= 0 && (orphanIndex < 0 || index < orphanIndex)) {
          orphanIndex = index;
        }
      }
      if (orphanIndex >= 0) {
        output = output.slice(0, orphanIndex);
      }

      return output.trim();
    }

    function hasRawToolMarkup(value: string | null | undefined) {
      const text = value ?? '';
      return text.includes('<tool_call>') || text.includes('<function=') || text.includes('<parameter=');
    }

    function makeUnparsedToolCallCard(finishReason: string, maxNewTokens: number): ToolCallResult {
      return {
        id: 'call_unparsed_app_preview',
        name: APP_PREVIEW_TOOL_NAME,
        arguments: {},
        status: 'parse_error',
        rawContent: '',
        error:
          `Incomplete preview tool call: model stopped before emitting a complete ` +
          `<tool_call>...</tool_call> block (finish_reason=${finishReason}, max_new_tokens=${maxNewTokens}).`,
      };
    }

    function escapeHtmlText(text: string) {
      return text.replace(/[<>&"]/g, (char) => {
        switch (char) {
          case '<':
            return '&lt;';
          case '>':
            return '&gt;';
          case '&':
            return '&amp;';
          case '"':
            return '&quot;';
          default:
            return char;
        }
      });
    }

    function escapeStyleBlock(text: string) {
      return text.replace(/<\/style/gi, '<\\/style');
    }

    function escapeScriptBlock(text: string) {
      return text.replace(/<\/script/gi, '<\\/script');
    }

    function wrapPreviewScript(text: string) {
      return `(() => {\n${text}\n})();`;
    }

    function isClassicScript(script: HTMLScriptElement) {
      const type = script.getAttribute('type')?.trim().toLowerCase();
      return !type || type === 'text/javascript' || type === 'application/javascript';
    }

    function wrapInlinePreviewScripts(root: ParentNode) {
      root.querySelectorAll('script').forEach((script) => {
        if (!(script instanceof HTMLScriptElement) || !isClassicScript(script)) {
          return;
        }
        const source = script.textContent ?? '';
        if (!source.trim()) return;
        script.textContent = wrapPreviewScript(source);
      });
    }

    function sanitizePreviewHtmlFragment(html: string) {
      if (!/<script[\s>]/i.test(html)) return html;
      const template = document.createElement('template');
      template.innerHTML = html;
      wrapInlinePreviewScripts(template.content);
      return template.innerHTML;
    }

    function sanitizePreviewDocument(html: string) {
      if (!/<script[\s>]/i.test(html)) return html;
      const doc = new DOMParser().parseFromString(html, 'text/html');
      wrapInlinePreviewScripts(doc);
      const doctype = doc.doctype ? `<!doctype ${doc.doctype.name}>` : '<!doctype html>';
      return `${doctype}\n${doc.documentElement.outerHTML}`;
    }

    function buildPreviewDocument(args: Record<string, unknown>) {
      const title = stringArg(args, 'title') || 'App preview';
      const rawHtml = stringArg(args, 'html');
      const css = stringArg(args, 'css');
      const js = stringArg(args, 'js') || stringArg(args, 'javascript');
      const isDocument = /<!doctype|<html[\s>]/i.test(rawHtml);
      const html = isDocument ? sanitizePreviewDocument(rawHtml) : sanitizePreviewHtmlFragment(rawHtml);
      const cssTag = css ? `<style>\n${escapeStyleBlock(css)}\n</style>` : '';
      const jsTag = js ? `<script>\n${escapeScriptBlock(wrapPreviewScript(js))}\n</script>` : '';
      if (isDocument) {
        let doc = html;
        if (cssTag) {
          doc = /<\/head>/i.test(doc) ? doc.replace(/<\/head>/i, `${cssTag}\n</head>`) : `${cssTag}\n${doc}`;
        }
        if (jsTag) {
          doc = /<\/body>/i.test(doc) ? doc.replace(/<\/body>/i, `${jsTag}\n</body>`) : `${doc}\n${jsTag}`;
        }
        return doc;
      }
      return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <base target="_blank" />
    <title>${escapeHtmlText(title)}</title>
    ${cssTag}
  </head>
  <body>
    ${html}
    ${jsTag}
  </body>
</html>`;
    }

    function renderInlinePreview(bubbleEl: HTMLElement, title: string, srcdoc: string) {
      // Create a child div in the bubble, mount React InlinePreviewCard into it
      let host = bubbleEl.querySelector<HTMLDivElement>(':scope > .inline-preview-host');
      if (!host) {
        host = document.createElement('div');
        host.className = 'inline-preview-host';
        bubbleEl.appendChild(host);
      }
      // Reuse a previously-mounted root if any (avoid re-creating)
      const existing = (host as HTMLDivElement & { __reactRoot?: Root }).__reactRoot;
      const root = existing ?? createRoot(host);
      (host as HTMLDivElement & { __reactRoot?: Root }).__reactRoot = root;
      root.render(<InlinePreviewCard title={title} srcdoc={srcdoc} />);
    }

    function compactToolArguments(call: ToolCallResult) {
      const args = asRecord(call.arguments);
      const title = stringArg(args, 'title');
      const summary = title
        ? { title }
        : Object.fromEntries(
            Object.keys(args)
              .slice(0, 3)
              .map((key) => [key, args[key]]),
          );
      return JSON.stringify(summary);
    }

    function appendToolCell(call: ToolCallResult, state: 'calling' | 'result' | 'error') {
      const toolDiv = document.createElement('div');
      toolDiv.className = `message tool-card ${state}`;

      const header = document.createElement('div');
      header.className = 'tool-card-header';

      const bullet = document.createElement('span');
      bullet.className = 'tool-card-bullet';
      bullet.setAttribute('aria-hidden', 'true');
      header.appendChild(bullet);

      const title = document.createElement('div');
      title.className = 'tool-card-title';
      title.textContent = `${state === 'calling' ? 'Calling' : state === 'result' ? 'Called' : 'Tool error'} ${call.name}`;
      header.appendChild(title);

      const meta = document.createElement('code');
      meta.className = 'tool-card-meta';
      meta.textContent = compactToolArguments(call);
      header.appendChild(meta);

      const body = document.createElement('div');
      body.className = 'tool-card-body';

      toolDiv.appendChild(header);
      toolDiv.appendChild(body);
      chatEl.appendChild(toolDiv);
      chatEl.scrollTop = chatEl.scrollHeight;
      return toolDiv;
    }

    function updateToolCell(toolDiv: HTMLElement, call: ToolCallResult, state: 'result' | 'error', text: string) {
      toolDiv.className = `message tool-card ${state}`;
      const title = toolDiv.querySelector('.tool-card-title');
      if (title) {
        title.textContent = `${state === 'result' ? 'Called' : 'Tool error'} ${call.name}`;
      }
      const body = toolDiv.querySelector('.tool-card-body');
      if (body) {
        body.textContent = text;
      }
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    function executeToolCall(call: ToolCallResult): BrowserChatMessage {
      if (call.status !== 'ok') {
        return {
          role: 'tool',
          toolCallId: call.id,
          content: JSON.stringify({
            ok: false,
            error: call.error || `Tool call parse failed: ${call.status}`,
          }),
        };
      }

      if (call.name !== APP_PREVIEW_TOOL_NAME) {
        return {
          role: 'tool',
          toolCallId: call.id,
          content: JSON.stringify({
            ok: false,
            error: `Unknown browser tool: ${call.name}`,
          }),
        };
      }

      const args = asRecord(call.arguments);
      const title = stringArg(args, 'title') || 'App preview';
      const html = stringArg(args, 'html');
      const css = stringArg(args, 'css');
      const js = stringArg(args, 'js') || stringArg(args, 'javascript');
      if (!html && !css && !js) {
        const keys = Object.keys(args);
        const rawPreview = (call.rawContent ?? '').slice(0, 500);
        log(
          `[TOOL] ${APP_PREVIEW_TOOL_NAME} missing content keys=${keys.join(',') || '(none)'} raw=${
            rawPreview || '(empty)'
          }`,
        );
        return {
          role: 'tool',
          toolCallId: call.id,
          content: JSON.stringify({
            ok: false,
            error: `create_app_preview requires html, css, or js content. Parsed keys: ${
              keys.join(', ') || '(none)'
            }.`,
          }),
        };
      }

      previewSequence++;
      const srcdoc = buildPreviewDocument(args);
      // Mount the inline preview card under the assistant bubble that owns
      // this tool call. Falls back to appending under the chat scroll
      // container if there is no current assistant bubble (e.g. the assistant
      // turn has already finalized before the tool fires).
      const previewHost: HTMLElement = currentAssistantDiv ?? chatEl;
      renderInlinePreview(previewHost, title, srcdoc);
      log(`[TOOL] ${APP_PREVIEW_TOOL_NAME} rendered "${title}"`);

      return {
        role: 'tool',
        toolCallId: call.id,
        content: JSON.stringify({
          ok: true,
          previewId: `preview-${previewSequence}`,
          title,
          message: 'The app is rendered in the browser preview iframe.',
        }),
      };
    }

    setAppToolsEnabled(appToolsEnabledRef.current);

    const chatSession = new BrowserChatSessionAdapter(BASE_SYSTEM_PROMPT, systemPromptForTools);

    let currentAssistantDiv: HTMLDivElement | null = null;
    let currentThinkingDiv: HTMLDetailsElement | null = null;
    let currentResponseDiv: HTMLDivElement | null = null;
    let currentToolCallIndicatorDiv: HTMLDivElement | null = null;
    let currentReasoningVisible = false;
    let isInThinking = false;
    let streamTokenCount = 0;
    let toolContinuationCount = 0;
    let previewSequence = 0;

    let reasoningBuffer = '';
    let contentQueue = '';
    let responseRenderText = '';
    let contentPrefixBuffer = '';
    let contentPrefixResolved = false;
    let contentReasoningPrefixActive = false;
    let currentUserPrompt = '';
    let rafHandle: number | null = null;
    let scrollDirty = false;
    let reasoningHasContent = false;
    let contentHasContent = false;
    let sharedWasmMemory: WebAssembly.Memory | null = null;
    let activeReaderAbort: AbortController | null = null;
    let streamT0 = 0;
    const markdownRoots = new Map<HTMLElement, ReturnType<typeof createRoot>>();
    const contentToolCallDisplayBuffer = new ToolCallTagBuffer();
    const reasoningToolCallDisplayBuffer = new ToolCallTagBuffer();

    function renderStreamdown(container: HTMLElement, text: string, isStreaming: boolean, autoScroll = false) {
      let root = markdownRoots.get(container);
      if (!root) {
        root = createRoot(container);
        markdownRoots.set(container, root);
      }
      root.render(
        <Streamdown
          mode={isStreaming ? 'streaming' : 'static'}
          className="streamdown-render"
          animated={false}
          isAnimating={isStreaming}
          parseIncompleteMarkdown={isStreaming}
          plugins={streamdownPlugins}
        >
          {text}
        </Streamdown>,
      );
      if (autoScroll) {
        requestAnimationFrame(() => {
          if (container.isConnected) {
            container.scrollTop = container.scrollHeight;
          }
        });
      }
    }

    function unmountAllStreamdown() {
      for (const root of markdownRoots.values()) {
        root.unmount();
      }
      markdownRoots.clear();
    }

    function createAssistantMessage() {
      const assistantDiv = document.createElement('div');
      assistantDiv.className = 'message assistant';
      // Stash the prompt that produced this assistant message so the per-
      // message Inspect button (revealed in finalizeAssistantMessage) can
      // hand it to the InspectorDrawer for a fresh runForInspector pass.
      if (currentUserPrompt) {
        assistantDiv.dataset.userPrompt = currentUserPrompt;
      }

      const thinkingDiv = document.createElement('details');
      thinkingDiv.className = 'thinking';
      const thinkingSummary = document.createElement('summary');
      thinkingSummary.textContent = 'Thinking...';
      thinkingDiv.appendChild(thinkingSummary);
      const thinkingContent = document.createElement('div');
      thinkingContent.className = 'thinking-content';
      thinkingDiv.appendChild(thinkingContent);
      assistantDiv.appendChild(thinkingDiv);

      const responseDiv = document.createElement('div');
      responseDiv.className = 'response-content';
      assistantDiv.appendChild(responseDiv);

      const toolCallIndicator = document.createElement('div');
      toolCallIndicator.className = 'streaming-tool-call';
      toolCallIndicator.hidden = true;
      const indicatorPulse = document.createElement('span');
      indicatorPulse.className = 'streaming-tool-call-pulse';
      indicatorPulse.setAttribute('aria-hidden', 'true');
      toolCallIndicator.appendChild(indicatorPulse);
      const indicatorLabel = document.createElement('span');
      indicatorLabel.className = 'streaming-tool-call-label';
      indicatorLabel.textContent = 'Creating app preview';
      toolCallIndicator.appendChild(indicatorLabel);
      const indicatorDots = document.createElement('span');
      indicatorDots.className = 'streaming-tool-call-dots';
      indicatorDots.setAttribute('aria-hidden', 'true');
      for (let i = 0; i < 3; i++) {
        indicatorDots.appendChild(document.createElement('i'));
      }
      toolCallIndicator.appendChild(indicatorDots);
      assistantDiv.appendChild(toolCallIndicator);

      chatEl.appendChild(assistantDiv);
      chatEl.scrollTop = chatEl.scrollHeight;

      currentAssistantDiv = assistantDiv;
      currentThinkingDiv = thinkingDiv;
      currentResponseDiv = responseDiv;
      currentToolCallIndicatorDiv = toolCallIndicator;
      isInThinking = false;

      thinkingDiv.style.display = 'none';
      reasoningBuffer = '';
      contentQueue = '';
      responseRenderText = '';
      contentPrefixBuffer = '';
      contentPrefixResolved = false;
      contentReasoningPrefixActive = false;
      contentToolCallDisplayBuffer.reset();
      reasoningToolCallDisplayBuffer.reset();
      reasoningHasContent = false;
      contentHasContent = false;
      if (rafHandle != null) {
        cancelAnimationFrame(rafHandle);
        rafHandle = null;
      }
      scrollDirty = false;
    }

    function scheduleFlush() {
      if (rafHandle != null) return;
      rafHandle = requestAnimationFrame(flushTick);
    }

    function couldStillBePromptEchoPrefix(buffer: string, prompt: string) {
      const normalizedPrompt = prompt.trim().toLowerCase();
      if (!normalizedPrompt) return false;

      const normalizedBuffer = buffer
        .trimStart()
        .replace(/^(?:\((?:user)\)|user)\s*:\s*/i, '')
        .toLowerCase();
      if (!normalizedBuffer) return true;
      if (normalizedPrompt.startsWith(normalizedBuffer)) return true;

      if (normalizedBuffer.startsWith(normalizedPrompt)) {
        const rest = normalizedBuffer.slice(normalizedPrompt.length);
        return !/[\r\n]/.test(rest);
      }

      return false;
    }

    function sanitizeThinkingText(thinking: string | null | undefined, latestUserText?: string) {
      const cleaned = sanitizeThinkingMarkup(splitAssistantThinking(thinking, latestUserText).thinking || thinking);
      if (/^[A-Z]$/u.test(cleaned)) return '';
      return cleaned;
    }

    function mergeThinkingText(current: string, next: string) {
      const left = sanitizeThinkingText(current);
      const right = sanitizeThinkingText(next);
      if (!right) return left;
      if (!left) return right;
      if (left.includes(right)) return left;
      if (right.includes(left)) return right;
      return `${left}\n\n${right}`;
    }

    function setStreamingThinkingText(text: string, open = true) {
      if (!currentThinkingDiv || !currentReasoningVisible) return;
      const cleaned = sanitizeThinkingText(text);
      if (!cleaned) return;
      const thinkingContentEl = currentThinkingDiv.querySelector('.thinking-content') as HTMLElement | null;
      const summary = currentThinkingDiv.querySelector('summary') as HTMLElement | null;
      if (summary) summary.textContent = 'Thought process';
      if (thinkingContentEl) renderStreamdown(thinkingContentEl, cleaned, true, true);
      currentThinkingDiv.style.display = '';
      currentThinkingDiv.open = open;
      reasoningHasContent = true;
      scrollDirty = true;
      scheduleFlush();
    }

    function showStreamingToolCallIndicator() {
      if (!currentAssistantDiv || !currentToolCallIndicatorDiv) return;
      currentAssistantDiv.classList.add('tool-streaming');
      currentToolCallIndicatorDiv.hidden = false;
      scrollDirty = true;
      scheduleFlush();
    }

    function hideStreamingToolCallIndicator() {
      currentAssistantDiv?.classList.remove('tool-streaming');
      if (currentToolCallIndicatorDiv) {
        currentToolCallIndicatorDiv.hidden = true;
      }
    }

    function looksLikeReasoningLeak(text: string) {
      return looksLikeReasoningText(text);
    }

    function hasExplicitThinkingBoundary(text: string) {
      return /(?:<\/\s*(?:think|longcat_think)\s*>|^\s*<details\b[\s\S]*<summary\b[\s\S]*(?:thought\s+process|thinking\s+process))/iu.test(
        text,
      );
    }

    function stripAssistantAnswerBoundary(text: string) {
      return text
        .replace(/^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:final\s+answer|answer|response)\s*:?\s*(?:\*\*)?\s*/iu, '')
        .trimStart();
    }

    function hasAssistantAnswerBoundary(text: string) {
      return stripAssistantAnswerBoundary(text) !== text.trimStart() && stripAssistantAnswerBoundary(text).length > 0;
    }

    function couldStillBeAssistantAnswerBoundaryPrefix(text: string) {
      const probe = text
        .trimStart()
        .toLowerCase()
        .replace(/^(?:#{1,6}\s*)/u, '')
        .replace(/^\*{1,2}/u, '')
        .trimStart();
      if (!probe) return true;
      return ['final answer', 'answer', 'response'].some(
        (label) => label.startsWith(probe) && probe.length < label.length,
      );
    }

    function appendStreamedToken(deltaText: string, isReasoning: boolean) {
      if (!currentAssistantDiv || !currentThinkingDiv || !currentResponseDiv) return;
      if (!deltaText) return;

      streamTokenCount++;
      setStatus(`Generating... ${streamTokenCount} tokens`, 'info');

      if (isReasoning) {
        const visibleReasoning = appToolsEnabledRef.current
          ? pushVisibleToolText(reasoningToolCallDisplayBuffer, deltaText)
          : deltaText;
        if (!currentReasoningVisible) {
          if (appToolsEnabledRef.current && reasoningToolCallDisplayBuffer.suppressed) {
            showStreamingToolCallIndicator();
          }
          return;
        }
        if (visibleReasoning.length === 0) {
          if (appToolsEnabledRef.current && reasoningToolCallDisplayBuffer.suppressed) {
            showStreamingToolCallIndicator();
          }
          return;
        }
        if (appToolsEnabledRef.current && reasoningToolCallDisplayBuffer.suppressed) {
          showStreamingToolCallIndicator();
        }
        reasoningBuffer += visibleReasoning;
        setStreamingThinkingText(reasoningBuffer, true);
        return;
      } else {
        let text = contentHasContent ? deltaText : deltaText.replace(/^\s+/, '');
        if (text.length === 0) return;
        if (!contentPrefixResolved && (currentUserPrompt || (currentReasoningVisible && !contentHasContent))) {
          contentPrefixBuffer += text;
          if (currentReasoningVisible && !contentHasContent) {
            const split = splitAssistantThinking(contentPrefixBuffer, currentUserPrompt);
            if (contentReasoningPrefixActive && hasAssistantAnswerBoundary(contentPrefixBuffer)) {
              text = stripAssistantAnswerBoundary(contentPrefixBuffer);
              contentPrefixBuffer = '';
              contentReasoningPrefixActive = false;
              contentPrefixResolved = true;
              if (text.length === 0) return;
            } else if (contentReasoningPrefixActive && couldStillBeAssistantAnswerBoundaryPrefix(contentPrefixBuffer)) {
              return;
            } else if (
              contentReasoningPrefixActive ||
              split.thinking ||
              hasExplicitThinkingBoundary(contentPrefixBuffer) ||
              looksLikeReasoningLeak(contentPrefixBuffer)
            ) {
              reasoningBuffer = mergeThinkingText(reasoningBuffer, split.thinking || contentPrefixBuffer);
              setStreamingThinkingText(reasoningBuffer, true);
              text = split.text;
              contentPrefixBuffer = '';
              contentReasoningPrefixActive = text.length === 0;
              contentPrefixResolved = text.length > 0;
              if (text.length === 0) return;
            } else if (couldStillBeReasoningTextPrefix(contentPrefixBuffer) && contentPrefixBuffer.length < 512) {
              return;
            }
          }
          if (!contentPrefixResolved) {
            const cleaned = sanitizeAssistantText(contentPrefixBuffer, currentUserPrompt);
            const shouldWait =
              cleaned === contentPrefixBuffer.trimStart() &&
              couldStillBePromptEchoPrefix(contentPrefixBuffer, currentUserPrompt) &&
              contentPrefixBuffer.length < currentUserPrompt.length + 24;
            if (shouldWait) return;
            text = cleaned;
            contentPrefixBuffer = '';
            contentPrefixResolved = true;
            if (text.length === 0) return;
          }
        }
        const visibleText = appToolsEnabledRef.current ? pushVisibleToolText(contentToolCallDisplayBuffer, text) : text;
        if (visibleText.length === 0) {
          if (appToolsEnabledRef.current && contentToolCallDisplayBuffer.suppressed) {
            showStreamingToolCallIndicator();
          }
          return;
        }
        if (appToolsEnabledRef.current && contentToolCallDisplayBuffer.suppressed) {
          showStreamingToolCallIndicator();
        } else {
          hideStreamingToolCallIndicator();
        }
        contentHasContent = true;
        contentQueue += visibleText;
      }
      scheduleFlush();
    }

    function flushTick() {
      rafHandle = null;

      if (!currentAssistantDiv || !currentThinkingDiv || !currentResponseDiv) {
        contentQueue = '';
        return;
      }

      if (contentQueue.length > 0) {
        const reveal = Math.max(1, Math.ceil(contentQueue.length / 20));
        const slice = contentQueue.slice(0, reveal);
        contentQueue = contentQueue.slice(reveal);

        if (isInThinking) {
          isInThinking = false;
          const summary = currentThinkingDiv.querySelector('summary') as HTMLElement | null;
          if (summary) summary.textContent = 'Thought process';
          currentThinkingDiv.open = false;
        }
        responseRenderText += slice;
        renderStreamdown(currentResponseDiv, responseRenderText, true);
        scrollDirty = true;
      }

      if (scrollDirty) {
        chatEl.scrollTop = chatEl.scrollHeight;
        scrollDirty = false;
      }

      if (contentQueue.length > 0) {
        scheduleFlush();
      }
    }

    function drainQueuesSync() {
      if (currentAssistantDiv && currentThinkingDiv && currentResponseDiv) {
        if (contentQueue.length > 0) {
          hideStreamingToolCallIndicator();
          if (isInThinking) {
            isInThinking = false;
            const summary = currentThinkingDiv.querySelector('summary') as HTMLElement | null;
            if (summary) summary.textContent = 'Thought process';
            currentThinkingDiv.open = false;
          }
          responseRenderText += contentQueue;
          renderStreamdown(currentResponseDiv, responseRenderText, true);
        }
      }
      reasoningBuffer = '';
      contentQueue = '';
      contentPrefixBuffer = '';
      contentPrefixResolved = false;
      contentToolCallDisplayBuffer.reset();
      reasoningToolCallDisplayBuffer.reset();
      hideStreamingToolCallIndicator();
      if (rafHandle != null) {
        cancelAnimationFrame(rafHandle);
        rafHandle = null;
      }
      scrollDirty = false;
    }

    function finalizeAssistantMessage(text: string, thinking: string | null) {
      if (!currentAssistantDiv || !currentThinkingDiv || !currentResponseDiv) return;

      drainQueuesSync();

      const trimmedThinking = thinking?.trim() || '';
      const shouldShowThinking = currentReasoningVisible && trimmedThinking.length > 3;
      if (shouldShowThinking) {
        const thinkingContentEl = currentThinkingDiv.querySelector('.thinking-content') as HTMLElement | null;
        if (thinkingContentEl) renderStreamdown(thinkingContentEl, trimmedThinking, false);
        const summary = currentThinkingDiv.querySelector('summary') as HTMLElement | null;
        if (summary) summary.textContent = 'Thought process';
        currentThinkingDiv.open = false;
        currentThinkingDiv.style.display = '';
      } else {
        currentThinkingDiv.style.display = 'none';
      }

      if (text && text.length > 0) {
        responseRenderText = text;
        renderStreamdown(currentResponseDiv, responseRenderText, false);
      } else {
        if (!shouldShowThinking) {
          currentAssistantDiv.remove();
          currentAssistantDiv = null;
          currentThinkingDiv = null;
          currentResponseDiv = null;
          currentToolCallIndicatorDiv = null;
          chatEl.scrollTop = chatEl.scrollHeight;
          return;
        }
        responseRenderText = '';
        renderStreamdown(currentResponseDiv, '', false);
      }
      currentAssistantDiv.classList.add('done');

      // Append the per-message Inspect button now that streaming completed.
      // Hidden during streaming (per Task 18 spec) — only revealed after the
      // message is final. Click is handled by event delegation on chatEl
      // below; the button just carries the prompt via data-user-prompt.
      const promptForInspect = currentAssistantDiv.dataset.userPrompt;
      if (promptForInspect && !currentAssistantDiv.querySelector('.assistant-inspect-row')) {
        const inspectRow = document.createElement('div');
        inspectRow.className = 'assistant-inspect-row';
        const inspectBtn = document.createElement('button');
        inspectBtn.type = 'button';
        inspectBtn.className = 'assistant-inspect-btn';
        inspectBtn.dataset.action = 'inspect';
        inspectBtn.textContent = 'Inspect';
        inspectBtn.title = 'Re-run the model on this prompt and show attention + top-K logits.';
        inspectRow.appendChild(inspectBtn);
        currentAssistantDiv.appendChild(inspectRow);
      }

      chatEl.scrollTop = chatEl.scrollHeight;

      currentAssistantDiv = null;
      currentThinkingDiv = null;
      currentResponseDiv = null;
      currentToolCallIndicatorDiv = null;
    }

    let worker = new Worker(workerAssetUrl(mlxWorkerUrl), {
      type: 'module',
    });
    mlxWorkerRef.current = worker;
    // Install a fresh AbortController for inspector calls bound to this
    // worker. The cleanup function below aborts it before terminating the
    // worker so in-flight `runForInspector` Promises in chapter components
    // reject immediately instead of waiting on the dead worker's reply.
    inspectorAbortRef.current = new AbortController();

    function postChatRequest() {
      const toolsEnabled = appToolsEnabledRef.current;
      const requestReasoningEffort: ReasoningEffort = reasoningEffortRef.current;
      const maxNewTokens = readMaxOutputTokens();
      activeChatMaxNewTokens = maxNewTokens;
      const config: Record<string, unknown> = {
        maxNewTokens,
        temperature: readTemperature(),
        reportPerformance: true,
      };
      if (toolsEnabled) {
        config.tools = [APP_PREVIEW_TOOL];
        config.allowToolCallsInReasoning = true;
        config.maxConsecutiveTokens = 256;
        config.maxNgramRepeats = 8;
      }
      worker.postMessage({
        type: 'chat',
        messages: chatSession.requestMessages(toolsEnabled),
        config,
        useSab,
        mode: modeParam ?? undefined,
        reasoningEffort: requestReasoningEffort,
      });
    }

    function finishChatTurn() {
      toolContinuationCount = 0;
      setStatus(`${compactModelLabel(activeModelLabel)} - Ready`, 'ready');
      sendBtn.disabled = false;
      setSendDisabledState(false);
      setGeneratingState(false);
      promptEl.disabled = false;
      promptEl.focus();
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    function finalizeFromResult(result: ChatResult) {
      const latestUserText = chatSession.latestUserContent() ?? currentUserPrompt;
      const textParts = splitAssistantThinking(result.text, latestUserText);
      const rawParts = splitAssistantThinking(result.rawText, latestUserText);
      const toolsEnabled = appToolsEnabledRef.current;
      const finishReason = result.finishReason || 'unknown';
      const rawToolMarkup =
        toolsEnabled &&
        (hasRawToolMarkup(result.text) ||
          hasRawToolMarkup(result.rawText) ||
          hasRawToolMarkup(result.thinking) ||
          hasRawToolMarkup(reasoningBuffer));
      let text =
        textParts.text || sanitizeAssistantText(result.text, latestUserText) || (toolsEnabled ? '' : rawParts.text);
      let thinking = sanitizeThinkingText(result.thinking ?? reasoningBuffer, latestUserText);
      thinking = mergeThinkingText(thinking, textParts.thinking);
      if (!toolsEnabled) {
        thinking = mergeThinkingText(thinking, rawParts.thinking);
      } else {
        text = stripRawToolMarkupForDisplay(text);
        thinking = stripRawToolMarkupForDisplay(thinking);
      }
      const toolCalls = (result.toolCalls ?? []).filter(Boolean);
      const okToolCalls = toolsEnabled ? toolCalls.filter((call) => call.status === 'ok') : [];
      const suppressedToolCalls = toolsEnabled ? [] : toolCalls.filter((call) => call.status === 'ok');
      const failedToolCalls = toolCalls.filter((call) => call.status !== 'ok');
      const displayText =
        text ||
        (okToolCalls.length > 0
          ? ''
          : suppressedToolCalls.length > 0
            ? 'Tool call skipped: app preview is disabled.'
            : '');
      finalizeAssistantMessage(displayText, thinking || null);
      if (finishReason === 'error') {
        chatSession.rollbackPendingTurn();
      } else {
        chatSession.commitAssistantResult(text, okToolCalls);
      }

      if (result.performance) {
        const prefillTps = positiveFiniteNumber(result.performance.prefillTokensPerSecond);
        const decodeTps = positiveFiniteNumber(result.performance.decodeTokensPerSecond);
        const ttftMs =
          Number.isFinite(result.performance.ttftMs) && result.performance.ttftMs >= 0
            ? result.performance.ttftMs.toFixed(0)
            : 'n/a';
        const prefill = prefillTps ? ` | Prefill ${prefillTps.toFixed(1)} tok/s` : '';
        const decode = decodeTps ? decodeTps.toFixed(1) : 'n/a';
        log(
          `${result.numTokens} tokens | finish ${finishReason} | TTFT ${ttftMs}ms${prefill} | Decode ${decode} tok/s`,
        );
        setPrefillTokensPerSec(prefillTps);
        setDecodeTokensPerSec(decodeTps);
      } else {
        log(`${result.numTokens ?? 0} tokens | finish ${finishReason}`);
        setPrefillTokensPerSec(null);
        setDecodeTokensPerSec(null);
      }

      for (const call of failedToolCalls) {
        const toolCell = appendToolCell(call, 'error');
        updateToolCell(toolCell, call, 'error', call.error || `Tool call parse failed: ${call.status}`);
      }
      if (suppressedToolCalls.length > 0) {
        for (const call of suppressedToolCalls) {
          const toolCell = appendToolCell(call, 'error');
          updateToolCell(toolCell, call, 'error', 'Tool call ignored: app preview is disabled.');
        }
      }
      if (rawToolMarkup && okToolCalls.length === 0) {
        const call = makeUnparsedToolCallCard(finishReason, activeChatMaxNewTokens);
        const toolCell = appendToolCell(call, 'error');
        updateToolCell(toolCell, call, 'error', call.error || '');
      }

      if (okToolCalls.length > 0) {
        for (const call of okToolCalls) {
          const toolCell = appendToolCell(call, 'calling');
          const toolMessage = executeToolCall(call);
          const toolResult = asRecord(toolMessage.content);
          const ok = toolResult.ok === true;
          updateToolCell(
            toolCell,
            call,
            ok ? 'result' : 'error',
            ok
              ? `Preview updated: ${stringArg(toolResult, 'title') || call.name}`
              : stringArg(toolResult, 'error') || `Tool failed: ${call.name}`,
          );
          try {
            chatSession.commitToolResult(toolMessage);
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            updateToolCell(toolCell, call, 'error', message);
            log(`[TOOL] ${message}`);
          }
        }
        if (!AUTO_CONTINUE_AFTER_APP_PREVIEW) {
          finishChatTurn();
          return;
        }
        if (toolContinuationCount < MAX_TOOL_CONTINUATIONS) {
          toolContinuationCount++;
          setStatus('Tool complete - generating follow-up...', 'info');
          currentReasoningVisible = reasoningEffortRef.current !== 'off';
          createAssistantMessage();
          postChatRequest();
          return;
        }
        const call = okToolCalls[okToolCalls.length - 1]!;
        const toolCell = appendToolCell(call, 'error');
        updateToolCell(toolCell, call, 'error', 'Tool loop stopped: max continuations reached.');
      }

      finishChatTurn();
    }

    function resetStreamingUi() {
      reasoningBuffer = '';
      contentQueue = '';
      responseRenderText = '';
      contentToolCallDisplayBuffer.reset();
      reasoningToolCallDisplayBuffer.reset();
      hideStreamingToolCallIndicator();
      if (rafHandle != null) {
        cancelAnimationFrame(rafHandle);
        rafHandle = null;
      }
      scrollDirty = false;
    }

    const handleWorkerMessage = (e: MessageEvent) => {
      const { type, ...data } = e.data;

      switch (type) {
        case 'log':
          console.log(data.message);
          break;

        case 'progress':
          log(data.message);
          if (data.step === 'chat') {
            break;
          } else {
            const progress = loadingProgressFromWorker(data);
            if (data.step === 'download' && progress) {
              const cacheSource = progress.cacheSource?.toLowerCase();
              const prefix = cacheSource === 'browser' ? 'Loading cached weights' : 'Downloading weights';
              setStatus(`${prefix}... ${Math.round(progress.pct)}%`, 'info', progress);
            } else {
              setStatus(data.message, 'info', progress);
            }
          }
          break;

        case 'ready':
          if (typeof data.modelLabel === 'string' && data.modelLabel) {
            activeModelLabel = data.modelLabel;
          }
          setModelLine(activeModelLabel);
          log('Model ready!');
          setStatus(`${compactModelLabel(activeModelLabel)} - Ready`, 'ready');
          sharedWasmMemory = (data as { sharedMemory?: WebAssembly.Memory }).sharedMemory ?? null;
          promptEl.disabled = false;
          sendBtn.disabled = false;
          setSendDisabledState(false);
          setGeneratingState(false);
          setImageCapability((data as { supportsImages?: boolean }).supportsImages === true);
          break;

        case 'stream-sab-open': {
          if (!sharedWasmMemory) {
            log('Error: received stream-sab-open before shared WASM memory');
            break;
          }
          streamT0 = performance.now();
          const ringOffset = (data as { ringOffset: number }).ringOffset;
          const size = (data as { size: number }).size;
          const ring = createSabRingOverHeap(sharedWasmMemory, ringOffset, size);
          activeReaderAbort = ring.reader(
            (chunk: ChatStreamChunk) => {
              if (chunk.done) {
                const elapsed = ((performance.now() - streamT0) / 1000).toFixed(1);
                log(`chatStream completed in ${elapsed}s`);
                finalizeFromResult({
                  text: chunk.text,
                  rawText: chunk.rawText,
                  thinking: chunk.thinking ?? null,
                  finishReason: chunk.finishReason,
                  toolCalls: chunk.toolCalls,
                  numTokens: chunk.numTokens,
                  performance: chunk.performance ?? null,
                });
                activeReaderAbort?.abort();
                activeReaderAbort = null;
                worker.postMessage({
                  type: 'stream-finalize',
                  numTokens: chunk.numTokens ?? 0,
                });
              } else {
                appendStreamedToken(chunk.text, chunk.isReasoning ?? false);
              }
            },
            (err: Error) => {
              log(`Stream error: ${err.message}`);
              chatSession.rollbackPendingTurn();
              resetStreamingUi();
              if (currentResponseDiv) {
                renderStreamdown(currentResponseDiv, `Error: ${err.message}`, false);
              }
              setStatus('Error', 'error');
              sendBtn.disabled = false;
              setSendDisabledState(false);
              setGeneratingState(false);
              promptEl.disabled = false;
              currentAssistantDiv = null;
              currentThinkingDiv = null;
              currentResponseDiv = null;
              currentToolCallIndicatorDiv = null;
              activeReaderAbort?.abort();
              activeReaderAbort = null;
              worker.postMessage({ type: 'stream-finalize', numTokens: 0 });
            },
          );
          break;
        }

        case 'result':
          finalizeFromResult(data as ChatResult);
          break;

        case 'profile':
          logProfile((data as { stats: ProfileStats }).stats);
          break;

        case 'error':
          log(`Error: ${data.message}`);
          logStack(data.stack);
          chatSession.rollbackPendingTurn();
          resetStreamingUi();
          if (currentResponseDiv) {
            renderStreamdown(currentResponseDiv, `Error: ${data.message}`, false);
          }
          setStatus('Error', 'error');
          sendBtn.disabled = false;
          setSendDisabledState(false);
          setGeneratingState(false);
          promptEl.disabled = false;
          currentAssistantDiv = null;
          currentThinkingDiv = null;
          currentResponseDiv = null;
          currentToolCallIndicatorDiv = null;
          if (activeReaderAbort) {
            activeReaderAbort.abort();
            activeReaderAbort = null;
          }
          break;

        case 'bridge-poisoned': {
          const reason = (data as { reason?: string }).reason ?? 'unknown';
          log(`Bridge poisoned (${reason}). The GPU worker is unresponsive - reload the page to recover.`);
          chatSession.rollbackPendingTurn();
          setStatus('Bridge poisoned - reload required', 'error');
          sendBtn.disabled = true;
          setSendDisabledState(true);
          setGeneratingState(false);
          promptEl.disabled = true;
          imageBtn.disabled = true;
          break;
        }
      }
    };
    worker.onmessage = handleWorkerMessage;

    function logProfile(s: ProfileStats) {
      setTelemetryStats({
        numTokens: s.numTokens,
        gpuRpcCount: s.gpuRpcCount,
        poolHits: s.poolHits,
        poolMisses: s.poolMisses,
      });
      const n = Math.max(1, s.numTokens);
      const dpt = s.totalDispatches / n;
      const rpt = s.bridgeRpcCount / n;
      const ept = s.totalPassEnds / n;
      const gptRpc = s.gpuRpcCount / n;
      const allOps = Object.entries(s.gpuByFn)
        .map(([k, v]) => ({ op: Number(k), count: v }))
        .filter(({ count }) => count > 0)
        .sort((a, b) => b.count - a.count)
        .map(({ op, count }) => `${rpcFnName(op)}(${count}, ${(count / n).toFixed(1)}/t)`)
        .join(', ');

      log(
        `[profile] Decode ${s.numTokens} tok | dispatches=${s.totalDispatches} (${dpt.toFixed(1)}/tok) | ` +
          `bridgeRPCs=${s.bridgeRpcCount} (${rpt.toFixed(1)}/tok) | ` +
          `gpuRPCs=${s.gpuRpcCount} (${gptRpc.toFixed(1)}/tok) | ` +
          `pass_ends=${s.totalPassEnds} (${ept.toFixed(1)}/tok)`,
      );
      log(`[profile] opcodes: ${allOps}`);

      const ph = s.poolHits ?? 0;
      const pm = s.poolMisses ?? 0;
      const pt = ph + pm;
      const phRate = pt > 0 ? ((ph / pt) * 100).toFixed(1) : '0.0';
      log(
        `[profile] pool: hits=${ph} (${(ph / n).toFixed(1)}/tok) misses=${pm} (${(pm / n).toFixed(1)}/tok) hitRate=${phRate}%`,
      );

      const gph = s.gpuPoolHits ?? 0;
      const gpm = s.gpuPoolMisses ?? 0;
      const gpt = gph + gpm;
      const gphRate = gpt > 0 ? ((gph / gpt) * 100).toFixed(1) : '0.0';
      log(
        `[profile] gpu-pool: hits=${gph} (${(gph / n).toFixed(1)}/tok) misses=${gpm} (${(gpm / n).toFixed(1)}/tok) hitRate=${gphRate}%`,
      );

      if (s.bindGroupCacheHits != null && s.bindGroupCacheMisses != null) {
        const bgh = s.bindGroupCacheHits;
        const bgm = s.bindGroupCacheMisses;
        const bgt = bgh + bgm;
        const bghRate = bgt > 0 ? ((bgh / bgt) * 100).toFixed(1) : '0.0';
        log(
          `[profile] bg-cache: hits=${bgh} (${(bgh / n).toFixed(1)}/tok) misses=${bgm} (${(bgm / n).toFixed(1)}/tok) hitRate=${bghRate}%`,
        );
      }

      if (s.uniformHotHits != null && s.uniformHotMisses != null) {
        const uhh = s.uniformHotHits;
        const uhm = s.uniformHotMisses;
        const uht = uhh + uhm;
        const uhhRate = uht > 0 ? ((uhh / uht) * 100).toFixed(1) : '0.0';
        log(
          `[profile] uniform-hot: hits=${uhh} (${(uhh / n).toFixed(1)}/tok) misses=${uhm} (${(uhm / n).toFixed(1)}/tok) hitRate=${uhhRate}%`,
        );
      }

      if (s.spinHits != null && s.spinMisses != null) {
        const sh = s.spinHits;
        const sm = s.spinMisses;
        const st = sh + sm;
        const shRate = st > 0 ? ((sh / st) * 100).toFixed(1) : '0.0';
        const sb = s.spinBudget ?? 0;
        log(`[profile] spin: hits=${sh} misses=${sm} hitRate=${shRate}% budget=${sb}`);
      }

      log(
        `[profile] diag: createAll=${s.diagCreateAll ?? 0} (mappedCopyDst=${s.diagCreateMappedCopyDst ?? 0}, mappedNoCopyDst=${s.diagCreateMappedNoCopyDst ?? 0}) | releaseAll=${s.diagReleaseAll ?? 0} (unknownHandle=${s.diagReleaseUnknownHandle ?? 0}, unpoolable=${s.diagReleaseUnpoolable ?? 0}, evictions=${s.diagPoolEvictions ?? 0})`,
      );
      log(
        `[profile] batch: attempt=${s.diagBatchAttempt ?? 0} staged=${s.diagBatchStaged ?? 0} deferredBlock=${s.diagBatchDeferredBlock ?? 0} stageRefused=${s.diagBatchStageRefused ?? 0}`,
      );
    }

    function logStack(stack: string | undefined) {
      if (!stack) return;
      for (const line of stack.split('\n').slice(0, 12)) {
        if (line.trim()) log(`  ${line}`);
      }
    }

    const handleWorkerError = (e: ErrorEvent) => {
      e.preventDefault();
      const inner = e.error instanceof Error ? e.error.message : '';
      log(`Worker error: ${e.message || inner || String(e)}`);
      log(`  file: ${e.filename}`);
      log(`  line: ${e.lineno}, col: ${e.colno}`);
      if (e.error instanceof Error) logStack(e.error.stack);
      chatSession.rollbackPendingTurn();
      setStatus('Worker error', 'error');
    };
    worker.onerror = handleWorkerError;

    const onMessageError = (e: MessageEvent) => {
      log(`Worker messageerror: ${e}`);
      chatSession.rollbackPendingTurn();
      setStatus('Worker error', 'error');
      sendBtn.disabled = false;
      setSendDisabledState(false);
      setGeneratingState(false);
      promptEl.disabled = false;
    };

    worker.addEventListener('messageerror', onMessageError);

    const urlParams = new URLSearchParams(location.search);
    const packBf16 = urlParams.get('pack_bf16') === '1';
    const sdpaFallback = urlParams.get('sdpa_fallback') === '1';
    const profile = urlParams.get('profile') === '1';
    const dispatchBatch = urlParams.get('dispatch_batch') !== '0';
    const compileMlp = urlParams.get('compile_mlp') !== '0';
    const compileGdnPre = urlParams.get('compile_gdn_pre') !== '0';
    const compileGdnPost = urlParams.get('compile_gdn_post') !== '0';
    const compileGdnG = urlParams.get('compile_gdn_g') !== '0';
    const enableVlm = urlParams.get('disable_vlm') !== '1';
    const fuseDispatch = urlParams.get('fuse_dispatch') !== '0';
    const useSab = urlParams.get('stream_sab') !== '0';
    const weightUploadBatchMb = Number.parseFloat(urlParams.get('weight_upload_batch_mb') ?? '');
    const weightUploadBatchBytes =
      Number.isFinite(weightUploadBatchMb) && weightUploadBatchMb > 0
        ? Math.round(weightUploadBatchMb * 1024 * 1024)
        : undefined;
    const modeParam = urlParams.get('mode') as 'sab' | 'tsfn' | 'baseline' | null;
    const flagBadges =
      (packBf16 ? ' (pack_bf16=1)' : '') +
      (sdpaFallback ? ' (sdpa_fallback=1)' : '') +
      (profile ? ' (profile=1)' : '') +
      (dispatchBatch ? ' (dispatch_batch=1)' : '') +
      (compileMlp ? ' (compile_mlp=1)' : '') +
      (compileGdnPre ? ' (compile_gdn_pre=1)' : '') +
      (compileGdnPost ? ' (compile_gdn_post=1)' : '') +
      (compileGdnG ? ' (compile_gdn_g=1)' : '') +
      (fuseDispatch ? '' : ' (fuse_dispatch=0)') +
      (enableVlm ? '' : ' (disable_vlm=1)') +
      (useSab ? '' : ' (stream_sab=0)') +
      (weightUploadBatchBytes ? ` (weight_upload_batch_mb=${Math.round(weightUploadBatchBytes / 1024 / 1024)})` : '') +
      (modeParam ? ` (mode=${modeParam})` : '');

    function resetForModelLoad(label?: string) {
      activeModelLabel = label ?? configuredModelLabel;
      setModelLine(activeModelLabel);
      setTelemetryStats(null);
      setPrefillTokensPerSec(null);
      setDecodeTokensPerSec(null);
      activeReaderAbort?.abort();
      activeReaderAbort = null;
      sharedWasmMemory = null;
      imageCapabilityKnown = false;
      supportsImages = false;
      pendingImage = null;
      imageInput.value = '';
      setImageAttached(false);
      resetStreamingUi();
      unmountAllStreamdown();
      currentAssistantDiv = null;
      currentThinkingDiv = null;
      currentResponseDiv = null;
      currentToolCallIndicatorDiv = null;
      chatSession.reset();
      chatEl.replaceChildren();
      promptEl.disabled = true;
      sendBtn.disabled = true;
      setSendDisabledState(true);
      setGeneratingState(false);
      imageBtn.disabled = true;
      setStatus('Initializing...', 'info');
      log(`Starting MLX Worker${flagBadges}${label ? ` (${label})` : ''}...`);
    }

    function startWorker(
      source: {
        modelFiles?: File[];
        modelUrl?: string;
        label?: string;
      } = {},
    ) {
      resetForModelLoad(source.label);
      worker.postMessage({
        type: 'init',
        wasmUrl: new URL(`/mlx-core.opt.wasm?v=${Date.now()}`, location.href).href,
        modelUrl: source.modelUrl ?? configuredModelUrl,
        modelLabel: source.label ?? configuredModelLabel,
        modelFiles: source.modelFiles,
        packBf16,
        sdpaFallback,
        profile,
        dispatchBatch,
        compileMlp,
        compileGdnPre,
        compileGdnPost,
        compileGdnG,
        enableVlm,
        fuseDispatch,
        weightUploadBatchBytes,
      });
    }

    startWorker(pendingModelSource ?? undefined);

    function handleSend() {
      const text = promptEl.value.trim();
      if (!text) return;

      promptEl.value = '';
      autosizePrompt();
      setTelemetryStats(null);
      setPrefillTokensPerSec(null);
      setDecodeTokensPerSec(null);
      sendBtn.disabled = true;
      setSendDisabledState(true);
      setGeneratingState(true);
      promptEl.disabled = true;

      if (pendingImage && !supportsImages) {
        log('Image removed: the loaded model does not support vision input.');
        pendingImage = null;
        imageInput.value = '';
        setImageAttached(false);
      }

      const userDiv = document.createElement('div');
      userDiv.className = 'message user';
      userDiv.textContent = text;
      if (pendingImage) {
        const img = document.createElement('img');
        img.src = URL.createObjectURL(new Blob([toArrayBuffer(pendingImage)]));
        img.className = 'attached-image';
        userDiv.appendChild(img);
      }
      chatEl.appendChild(userDiv);

      const msg: BrowserChatMessage = {
        role: 'user',
        content: text,
      };
      if (pendingImage) {
        msg.images = [pendingImage];
        pendingImage = null;
        imageInput.value = '';
        setImageAttached(false);
      }
      try {
        chatSession.beginUserTurn(msg);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        log(`Error: ${message}`);
        setStatus('Session error', 'error');
        sendBtn.disabled = false;
        setSendDisabledState(false);
        setGeneratingState(false);
        promptEl.disabled = false;
        return;
      }

      setStatus('Generating...', 'info');
      streamTokenCount = 0;
      toolContinuationCount = 0;
      isInThinking = false;
      currentUserPrompt = text;
      currentReasoningVisible = reasoningEffortRef.current !== 'off';
      createAssistantMessage();

      postChatRequest();
    }

    const onPromptKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    };

    const onImageClick = () => {
      if (!supportsImages) {
        setStatus('Image input unavailable', 'info');
        log('Image input unavailable for this model.');
        return;
      }
      imageInput.click();
    };
    const onImageChange = async () => {
      const file = imageInput.files?.[0];
      if (!file) return;
      if (!supportsImages) {
        imageInput.value = '';
        setStatus('Image input unavailable', 'info');
        log('Image ignored: the loaded model does not support vision input.');
        return;
      }
      pendingImage = new Uint8Array(await file.arrayBuffer());
      imageInput.value = '';
      log(`Image attached: ${file.name} (${(pendingImage.length / 1024).toFixed(0)} KB)`);
      setImageAttached(true);
    };

    const onPaste = async (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          if (!supportsImages) {
            setStatus('Image input unavailable', 'info');
            log('Image paste ignored: the loaded model does not support vision input.');
            break;
          }
          const file = item.getAsFile();
          if (!file) continue;
          pendingImage = new Uint8Array(await file.arrayBuffer());
          log(`Image pasted (${(pendingImage.length / 1024).toFixed(0)} KB)`);
          setImageAttached(true);
          break;
        }
      }
    };

    // Event delegation for the per-assistant-message "Inspect" button. The
    // buttons are appended imperatively in finalizeAssistantMessage; this
    // listener fishes the prompt off the closest .message.assistant ancestor
    // and lifts it into React state so the InspectorDrawer mounts.
    const onChatClick = (e: MouseEvent) => {
      const target = e.target;
      if (!(target instanceof HTMLElement)) return;
      const btn = target.closest('[data-action="inspect"]');
      if (!btn) return;
      const messageEl = btn.closest('.message.assistant') as HTMLElement | null;
      const promptForInspect = messageEl?.dataset.userPrompt;
      if (promptForInspect) {
        setInspectedPrompt(promptForInspect);
      }
    };

    sendBtn.addEventListener('click', handleSend);
    promptEl.addEventListener('keydown', onPromptKeyDown);
    promptEl.addEventListener('input', autosizePrompt);
    imageBtn.addEventListener('click', onImageClick);
    imageInput.addEventListener('change', onImageChange);
    document.addEventListener('paste', onPaste);
    chatEl.addEventListener('click', onChatClick);

    // Expose a global reset hook for the new ChatHeader's "Reset Chat" button.
    // Clears the chat DOM, resets the local session history (keeping the
    // system prompt), and tears down any in-flight streaming UI state.
    (window as unknown as { __mlxResetChat?: () => void }).__mlxResetChat = () => {
      activeReaderAbort?.abort();
      activeReaderAbort = null;
      resetStreamingUi();
      unmountAllStreamdown();
      currentAssistantDiv = null;
      currentThinkingDiv = null;
      currentResponseDiv = null;
      currentToolCallIndicatorDiv = null;
      chatSession.reset();
      chatEl.replaceChildren();
      // Close the inspector drawer if it's open — its prompt no longer
      // corresponds to any visible message after the reset.
      setInspectedPrompt(null);
    };

    return () => {
      sendBtn.removeEventListener('click', handleSend);
      promptEl.removeEventListener('keydown', onPromptKeyDown);
      promptEl.removeEventListener('input', autosizePrompt);
      imageBtn.removeEventListener('click', onImageClick);
      imageInput.removeEventListener('change', onImageChange);
      document.removeEventListener('paste', onPaste);
      chatEl.removeEventListener('click', onChatClick);
      worker.removeEventListener('messageerror', onMessageError);
      activeReaderAbort?.abort();
      if (rafHandle != null) cancelAnimationFrame(rafHandle);
      unmountAllStreamdown();
      // Abort any in-flight inspector calls BEFORE terminating the worker.
      // After `terminate()` the worker can never reply, so an outstanding
      // `runForInspector` Promise would otherwise hang for the full 60s
      // timeout. Signalling abort lets chapter components show the soft
      // "cancelled" banner within a tick.
      inspectorAbortRef.current?.abort();
      inspectorAbortRef.current = null;
      worker.terminate();
      if (mlxWorkerRef.current === worker) {
        mlxWorkerRef.current = null;
      }
      delete (window as unknown as { __mlxResetChat?: () => void }).__mlxResetChat;
    };
  }, [configuredModelLabel, configuredModelUrl, loadKickoff, pendingModelSource]);

  return (
    <div className="app-root">
      <div className={`chat-layer ${screen.kind === 'chat' ? 'visible' : ''}`}>
        <div className="app-shell">
          <ChatHeader
            onReset={() => {
              // The existing useEffect installs a global reset hook on
              // window.__mlxResetChat when the chat session is established. If
              // it's set, call it; the chat DOM will clear and the underlying
              // chat session will be reset.
              if ((window as unknown as { __mlxResetChat?: () => void }).__mlxResetChat) {
                (window as unknown as { __mlxResetChat: () => void }).__mlxResetChat();
              }
              dispatchScreen({ type: 'reset_chat' });
            }}
          />
          {/*
        Hidden ref-only status element. The legacy app-header rendered the
        Initializing status pill; we kept the ref so the existing useEffect's
        statusEl.textContent / statusEl.className mutations remain safe. The
        new ChatHeader shows its own static "Ready on WebGPU" status.
      */}
          <span ref={statusRef} id="status" style={{ display: 'none' }} aria-hidden="true" />

          <div className="chat-messages" id="chat" ref={chatRef} />

          <TelemetryStrip
            stats={telemetryStats}
            prefillTokensPerSecond={prefillTokensPerSec}
            decodeTokensPerSecond={decodeTokensPerSec}
            modelLine={modelLine}
          />

          <PowerComposer
            textareaRef={promptRef}
            sendRef={sendRef}
            imageButtonRef={imageButtonRef}
            imageInputRef={imageInputRef}
            reasoningEffort={reasoningEffort}
            onCycleReasoning={() => {
              const next = cycleReasoningEffort(reasoningEffort);
              reasoningEffortRef.current = next;
              setReasoningEffortState(next);
            }}
            temperature={temperatureValue}
            onTemperatureChange={(v) => {
              temperatureValueRef.current = v;
              setTemperatureValue(v);
              if (temperatureInputRef.current) temperatureInputRef.current.value = `${v}`;
            }}
            maxTokens={maxTokensValue}
            onMaxTokensChange={(v) => {
              maxTokensValueRef.current = v;
              setMaxTokensValue(v);
              if (maxOutputTokensInputRef.current) maxOutputTokensInputRef.current.value = `${v}`;
            }}
            toolsEnabled={appToolsEnabled}
            onToggleTools={() => {
              const next = !appToolsEnabled;
              setAppToolsEnabledState(next);
              appToolsEnabledRef.current = next;
            }}
            generating={generating}
            sendDisabled={sendDisabled}
          />
        </div>
        {inspectedPrompt !== null && screen.kind === 'chat' ? (
          <InspectorDrawer
            prompt={inspectedPrompt}
            workerRef={mlxWorkerRef}
            abortRef={inspectorAbortRef}
            onClose={() => setInspectedPrompt(null)}
          />
        ) : null}
      </div>

      {/*
        Always-mounted hidden file input for picking a local model directory.
        Lives at the App root so the landing screen can open it before a model
        worker exists; React captures the selected files and starts the worker
        directly from those local File objects.
      */}
      <input
        id="model-dir-input"
        ref={modelDirInputRef}
        type="file"
        multiple
        {...DIRECTORY_PICKER_INPUT_PROPS}
        className="hidden"
        onChange={handleLocalModelInputChange}
      />
      {/*
        Hidden inputs preserved so the existing useEffect's readTemperature() and
        readMaxOutputTokens() helpers (which read from .value) keep working
        unchanged. The PowerComposer mirrors React state into these via its
        onTemperatureChange / onMaxTokensChange callbacks.
      */}
      <input ref={temperatureInputRef} type="hidden" defaultValue={initialTemperature} />
      <input ref={maxOutputTokensInputRef} type="hidden" defaultValue={initialMaxOutputTokens} />

      {screen.kind === 'landing' && (
        <Landing
          onLoad={() => {
            if (hostedModelAvailable === false) {
              modelDirInputRef.current?.click();
              return;
            }
            setPendingModelSource(null);
            setErrorBannerState(null);
            setLoadingProgress(null);
            setLoadKickoff((k) => k + 1);
            dispatchScreen({ type: 'load_kickoff' });
          }}
          onLocalModel={() => modelDirInputRef.current?.click()}
          onStartLearning={() => dispatchScreen({ type: 'start_learning' })}
          errorBanner={errorBanner}
          hostedModelAvailable={hostedModelAvailable}
        />
      )}
      {screen.kind === 'chapter_index' && (
        <ChapterIndex
          onOpenChapter={(chapterId) => dispatchScreen({ type: 'open_chapter', chapterId })}
          onBackToLanding={() => dispatchScreen({ type: 'back_to_landing' })}
          onOpenFreeChat={() => {
            if (hostedModelAvailable === false) {
              modelDirInputRef.current?.click();
              return;
            }
            setPendingModelSource(null);
            setErrorBannerState(null);
            setLoadingProgress(null);
            setLoadKickoff((k) => k + 1);
            dispatchScreen({ type: 'open_free_chat' });
          }}
        />
      )}
      {screen.kind === 'chapter' && (() => {
        // chapter.id is guaranteed present by the discriminated-union typing —
        // `open_chapter` carries it on the event and the reducer writes it into
        // the state. `findChapter` may still return undefined if the id is a
        // stale value (e.g. a removed chapter), so we narrow defensively.
        const chapter = findChapter(screen.chapterId);
        if (!chapter) {
          return (
            <div className="text-muted-foreground p-6">
              Unknown chapter: <span className="font-mono">{screen.chapterId}</span>
              <div className="mt-4">
                <button
                  type="button"
                  className="underline"
                  onClick={() => dispatchScreen({ type: 'back_to_index' })}
                >
                  Back to chapter index
                </button>
              </div>
            </div>
          );
        }
        return (
          <LessonLayout
            current={chapter}
            onOpenChapter={(chapterId) => dispatchScreen({ type: 'open_chapter', chapterId })}
            onBackToIndex={() => dispatchScreen({ type: 'back_to_index' })}
            onOpenFreeChat={() => {
              if (hostedModelAvailable === false) {
                modelDirInputRef.current?.click();
                return;
              }
              setPendingModelSource(null);
              setErrorBannerState(null);
              setLoadingProgress(null);
              setLoadKickoff((k) => k + 1);
              dispatchScreen({ type: 'open_free_chat' });
            }}
            tryItPanel={
              chapter.id === 'attention' ? (
                <AttentionDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
              ) : chapter.id === 'multi-head-gqa' ? (
                <MultiheadGqaDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
              ) : chapter.id === 'tokenization' ? (
                <TokenizerDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
              ) : chapter.id === 'embeddings' ? (
                <EmbeddingsDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
              ) : chapter.id === 'rope' ? (
                <RopeDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
              ) : chapter.id === 'rmsnorm' ? (
                <RmsNormDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
              ) : chapter.id === 'mlp' ? (
                <MlpDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
              ) : chapter.id === 'full-block' ? (
                <FullBlockDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
              ) : chapter.id === 'sampling' ? (
                <SamplingDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
              ) : chapter.id === 'kv-cache' ? (
                <KvCacheDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
              ) : null
            }
          >
            {chapter.id === 'attention' ? (
              <AttentionChapterBody />
            ) : chapter.id === 'multi-head-gqa' ? (
              <MultiheadGqaChapterBody />
            ) : chapter.id === 'tokenization' ? (
              <TokenizationChapterBody />
            ) : chapter.id === 'embeddings' ? (
              <EmbeddingsChapterBody />
            ) : chapter.id === 'rope' ? (
              <RopeChapterBody />
            ) : chapter.id === 'rmsnorm' ? (
              <RmsNormChapterBody />
            ) : chapter.id === 'mlp' ? (
              <MlpChapterBody />
            ) : chapter.id === 'full-block' ? (
              <FullBlockChapterBody />
            ) : chapter.id === 'sampling' ? (
              <SamplingChapterBody />
            ) : chapter.id === 'kv-cache' ? (
              <KvCacheChapterBody />
            ) : (
              <div className="text-muted-foreground">
                This chapter is not yet authored.
              </div>
            )}
          </LessonLayout>
        );
      })()}
      {screen.kind === 'loading' && <Loading status={loadingText} progress={loadingProgress} />}
    </div>
  );
}

function rpcFnName(fn: number): string {
  switch (fn) {
    case 1:
      return 'CREATE_INSTANCE';
    case 2:
      return 'INSTANCE_REQUEST_ADAPTER';
    case 3:
      return 'INSTANCE_RELEASE';
    case 4:
      return 'ADAPTER_REQUEST_DEVICE';
    case 5:
      return 'ADAPTER_RELEASE';
    case 6:
      return 'ADAPTER_GET_PROPERTIES';
    case 10:
      return 'DEVICE_CREATE_BUFFER';
    case 11:
      return 'DEVICE_CREATE_SHADER_MODULE';
    case 12:
      return 'DEVICE_CREATE_COMPUTE_PIPELINE';
    case 13:
      return 'DEVICE_CREATE_BIND_GROUP';
    case 14:
      return 'DEVICE_CREATE_COMMAND_ENCODER';
    case 15:
      return 'DEVICE_GET_QUEUE';
    case 16:
      return 'DEVICE_GET_LIMITS';
    case 17:
      return 'DEVICE_SET_ERROR_CALLBACK';
    case 18:
      return 'DEVICE_SET_LOST_CALLBACK';
    case 19:
      return 'DEVICE_RELEASE';
    case 20:
      return 'QUEUE_SUBMIT';
    case 21:
      return 'QUEUE_WRITE_BUFFER';
    case 22:
      return 'QUEUE_ON_SUBMITTED_WORK_DONE';
    case 23:
      return 'QUEUE_RELEASE';
    case 30:
      return 'CMD_ENCODER_BEGIN_COMPUTE_PASS';
    case 31:
      return 'CMD_ENCODER_COPY_BUFFER';
    case 32:
      return 'CMD_ENCODER_FINISH';
    case 33:
      return 'CMD_ENCODER_RELEASE';
    case 34:
      return 'CMD_BUFFER_RELEASE';
    case 40:
      return 'COMPUTE_PASS_SET_PIPELINE';
    case 41:
      return 'COMPUTE_PASS_SET_BIND_GROUP';
    case 42:
      return 'COMPUTE_PASS_DISPATCH';
    case 43:
      return 'COMPUTE_PASS_END';
    case 44:
      return 'COMPUTE_PASS_RELEASE';
    case 50:
      return 'BUFFER_GET_SIZE';
    case 51:
      return 'BUFFER_GET_MAPPED_RANGE';
    case 52:
      return 'BUFFER_GET_CONST_MAPPED_RANGE';
    case 53:
      return 'BUFFER_UNMAP';
    case 54:
      return 'BUFFER_MAP_ASYNC';
    case 55:
      return 'BUFFER_DESTROY';
    case 56:
      return 'BUFFER_RELEASE';
    case 60:
      return 'PIPELINE_GET_BIND_GROUP_LAYOUT';
    case 61:
      return 'PIPELINE_RELEASE';
    case 70:
      return 'BIND_GROUP_RELEASE';
    case 71:
      return 'BIND_GROUP_LAYOUT_RELEASE';
    case 72:
      return 'SHADER_MODULE_RELEASE';
    case 80:
      return 'POLL';
    case 90:
      return 'ADD_GPU_BUFFER';
    case 91:
      return 'FUSED_DISPATCH';
    case 92:
      return 'FUSED_DISPATCH_2BG';
    case 93:
      return 'FUSED_SUBMIT';
    case 94:
      return 'FUSED_BG_DISPATCH';
    case 95:
      return 'CREATE_BUFFER_FROM_DATA';
    case 96:
      return 'FUSED_FULL_DISPATCH';
    case 97:
      return 'FUSED_DISPATCH_WITH_UNIFORM';
    case 98:
      return 'FUSED_COPY_BUFFER';
    case 99:
      return 'GET_STATS';
    case 102:
      return 'BUFFER_RELEASE_BATCH';
    case 103:
      return 'DISPATCH_BATCH';
    default:
      return `fn${fn}`;
  }
}

const rootHost = document.getElementById('root')!;
const rootGlobal = globalThis as typeof globalThis & {
  __mlxBrowserDemoRoot?: Root;
};
const appRoot = rootGlobal.__mlxBrowserDemoRoot ?? createRoot(rootHost);
rootGlobal.__mlxBrowserDemoRoot = appRoot;
appRoot.render(<App />);
