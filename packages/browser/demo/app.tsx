import type {
  ChatStreamChunk,
  ToolCallResult,
  ToolDefinition,
} from "@mlx-node/core";

import { ArrowUp, Cpu, ImagePlus, Mic } from "lucide-react";
import { useEffect, useReducer, useRef, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Streamdown } from "streamdown";

import { createSabRingOverHeap } from "../src/chat-stream-sab.js";
import {
  type ScreenState,
  type ProfileLikeStats,
  reduceScreen,
} from "./lib/screen-state";
import { ChatHeader } from "./components/chat/ChatHeader";
import { InlinePreviewCard } from "./components/chat/InlinePreviewCard";
import { TelemetryStrip } from "./components/chat/TelemetryStrip";
import { Landing } from "./components/landing/Landing";
import { Loading } from "./components/loading/Loading";
import { Button } from "./components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./components/ui/card";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./components/ui/select";
import { Switch } from "./components/ui/switch";
import { Textarea } from "./components/ui/textarea";
import {
  sanitizeAssistantText,
  sanitizeThinkingText as sanitizeThinkingMarkup,
  splitAssistantThinking,
} from "../src/generated-text.js";
import "streamdown/styles.css";
import "./styles.css";

type StatusState = "info" | "ready" | "error";
type ReasoningEffort = "off" | "low" | "medium" | "high";
type PreviewState = "idle" | "calling" | "rendered" | "error";
const DEFAULT_MODEL_LABEL = "qwen3.5-0.8b-mlx-bf16";
const MAX_BROWSER_OUTPUT_TOKENS = 36864;
const DEFAULT_BROWSER_OUTPUT_TOKENS = 1024;
const DEFAULT_BROWSER_TEMPERATURE = 0.6;
const APP_PREVIEW_TOOL_NAME = "create_app_preview";
const MAX_TOOL_CONTINUATIONS = 2;
const AUTO_CONTINUE_AFTER_APP_PREVIEW = false;

type BrowserToolCall = {
  id?: string;
  name: string;
  arguments: string;
};

type BrowserChatMessage = {
  role: string;
  content: string;
  images?: Uint8Array[];
  toolCalls?: BrowserToolCall[];
  toolCallId?: string;
};

class ToolCallDisplayBuffer {
  private pending = "";
  private suppressing = false;
  private seenToolCall = false;
  private readonly openTag = "<tool_call>";
  private readonly closeTag = "</tool_call>";

  reset() {
    this.pending = "";
    this.suppressing = false;
    this.seenToolCall = false;
  }

  isToolCallActive() {
    return this.suppressing || this.seenToolCall;
  }

  push(delta: string) {
    this.pending += delta;
    let visible = "";

    while (this.pending.length > 0) {
      if (this.suppressing) {
        const closeIndex = this.pending.indexOf(this.closeTag);
        if (closeIndex < 0) {
          this.pending = this.keepPossiblePrefix(this.pending, this.closeTag);
          return visible;
        }
        this.pending = this.pending.slice(closeIndex + this.closeTag.length);
        this.suppressing = false;
        continue;
      }

      const openIndex = this.pending.indexOf(this.openTag);
      if (openIndex >= 0) {
        visible += this.pending.slice(0, openIndex);
        this.pending = this.pending.slice(openIndex + this.openTag.length);
        this.suppressing = true;
        this.seenToolCall = true;
        continue;
      }

      const keepLength = this.possiblePrefixLength(this.pending, this.openTag);
      if (keepLength > 0) {
        visible += this.pending.slice(0, this.pending.length - keepLength);
        this.pending = this.pending.slice(-keepLength);
        return visible;
      }

      visible += this.pending;
      this.pending = "";
    }

    return visible;
  }

  private keepPossiblePrefix(text: string, marker: string) {
    const keepLength = this.possiblePrefixLength(text, marker);
    return keepLength > 0 ? text.slice(-keepLength) : "";
  }

  private possiblePrefixLength(text: string, marker: string) {
    const maxLength = Math.min(text.length, marker.length - 1);
    for (let length = maxLength; length > 0; length--) {
      if (marker.startsWith(text.slice(-length))) return length;
    }
    return 0;
  }
}

const APP_PREVIEW_TOOL: ToolDefinition = {
  type: "function",
  function: {
    name: APP_PREVIEW_TOOL_NAME,
    description:
      "Render a complete, self-contained HTML/CSS/JavaScript app in the browser preview iframe.",
    parameters: {
      type: "object",
      properties: JSON.stringify({
        title: {
          type: "string",
          description: "Short title for the app preview.",
        },
        html: {
          type: "string",
          description:
            "Body HTML for the app. Include meaningful semantic structure.",
        },
        css: {
          type: "string",
          description:
            "CSS for the app. Keep it self-contained and responsive.",
        },
        js: {
          type: "string",
          description:
            "Client-side JavaScript for app interactions. Do not use external dependencies.",
        },
      }),
      required: ["html", "css", "js"],
    },
  },
};

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
  const workspaceGridRef = useRef<HTMLElement>(null);
  const chatRef = useRef<HTMLDivElement>(null);
  const previewSurfaceRef = useRef<HTMLDivElement>(null);
  const previewFrameRef = useRef<HTMLIFrameElement>(null);
  const previewTitleRef = useRef<HTMLDivElement>(null);
  const previewMetaRef = useRef<HTMLDivElement>(null);
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const sendRef = useRef<HTMLButtonElement>(null);
  const imageButtonRef = useRef<HTMLButtonElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const modelDirInputRef = useRef<HTMLInputElement>(null);
  const composerModelLabelRef = useRef<HTMLSpanElement>(null);
  const temperatureInputRef = useRef<HTMLInputElement>(null);
  const maxOutputTokensInputRef = useRef<HTMLInputElement>(null);
  const reasoningEffortRef = useRef<ReasoningEffort>("off");
  const initialUrlParams = new URLSearchParams(location.search);
  const initialAppToolsEnabled =
    initialUrlParams.get("tools") === "1" ||
    initialUrlParams.get("app_preview") === "1";
  const [appToolsEnabled, setAppToolsEnabledState] = useState(
    initialAppToolsEnabled,
  );
  const [reasoningEffort, setReasoningEffortState] =
    useState<ReasoningEffort>("off");
  const [screen, dispatchScreen] = useReducer(
    reduceScreen,
    "landing" as ScreenState,
  );
  const [loadKickoff, setLoadKickoff] = useState(0);
  const [loadingText, setLoadingText] = useState<string | null>(null);
  const [errorBanner, setErrorBannerState] = useState<string | null>(null);
  const [telemetryStats, setTelemetryStats] = useState<ProfileLikeStats | null>(
    null,
  );
  const [decodeTokensPerSec, setDecodeTokensPerSec] = useState<number | null>(
    null,
  );
  const [modelLine, setModelLine] = useState<string>(
    `${DEFAULT_MODEL_LABEL} · bf16`,
  );
  const appToolsEnabledRef = useRef(initialAppToolsEnabled);
  const initialMaxOutputTokens = Math.min(
    MAX_BROWSER_OUTPUT_TOKENS,
    Math.max(
      1,
      Number.parseInt(
        initialUrlParams.get("max_new_tokens") ??
          initialUrlParams.get("maxOutputTokens") ??
          `${DEFAULT_BROWSER_OUTPUT_TOKENS}`,
        10,
      ) || DEFAULT_BROWSER_OUTPUT_TOKENS,
    ),
  );
  const parsedInitialTemperature = Number.parseFloat(
    initialUrlParams.get("temperature") ??
      initialUrlParams.get("temp") ??
      `${DEFAULT_BROWSER_TEMPERATURE}`,
  );
  const initialTemperature = Number.isFinite(parsedInitialTemperature)
    ? Math.min(2, Math.max(0, parsedInitialTemperature))
    : DEFAULT_BROWSER_TEMPERATURE;

  useEffect(() => {
    if (loadKickoff === 0) {
      return;
    }
    const statusEl = statusRef.current!;
    const workspaceGrid = workspaceGridRef.current!;
    const chatEl = chatRef.current!;
    // Legacy preview surface refs are dead after Task 9; route create_app_preview
    // through renderInlinePreview() into the assistant bubble instead. We keep
    // detached DOM stubs here so the legacy setPreviewStatus / setEmptyPreview /
    // executeToolCall paths remain harmless until Task 11 removes them.
    const previewSurface =
      previewSurfaceRef.current ?? document.createElement("div");
    const previewFrame =
      previewFrameRef.current ?? document.createElement("iframe");
    const previewTitle =
      previewTitleRef.current ?? document.createElement("div");
    const previewMeta =
      previewMetaRef.current ?? document.createElement("div");
    const promptEl = promptRef.current!;
    const sendBtn = sendRef.current!;
    const imageBtn = imageButtonRef.current!;
    const imageInput = imageInputRef.current!;
    const modelDirInput = modelDirInputRef.current!;
    const composerModelLabel = composerModelLabelRef.current!;
    const temperatureInput = temperatureInputRef.current!;
    const maxOutputTokensInput = maxOutputTokensInputRef.current!;

    if (
      !statusEl ||
      !workspaceGrid ||
      !chatEl ||
      !promptEl ||
      !sendBtn ||
      !imageBtn ||
      !imageInput ||
      !modelDirInput ||
      !composerModelLabel ||
      !maxOutputTokensInput
    ) {
      return;
    }
    modelDirInput.setAttribute("webkitdirectory", "");
    modelDirInput.setAttribute("directory", "");
    let activeModelLabel = DEFAULT_MODEL_LABEL;

    if (navigator.storage?.persist) {
      void navigator.storage
        .persist()
        .then((persisted) => {
          log(
            persisted
              ? "Browser storage persistence granted."
              : "Browser storage persistence unavailable; cached models may be evicted under storage pressure.",
          );
        })
        .catch((error) => {
          log(`Browser storage persistence request failed: ${String(error)}`);
        });
    }

    function setStatus(text: string, state: StatusState = "info") {
      statusEl.textContent = text;
      statusEl.className = `status-pill ${state}`;
      setLoadingText(text);
      if (state === "ready") {
        dispatchScreen({ type: "model_ready" });
      } else if (state === "error") {
        setErrorBannerState(text);
        dispatchScreen({ type: "model_error" });
      }
    }

    function setPreviewStatus(state: PreviewState, text: string) {
      previewSurface.dataset.previewState = state;
      previewMeta.textContent = text;
    }

    function setAppToolsEnabled(enabled: boolean) {
      appToolsEnabledRef.current = enabled;
      setAppToolsEnabledState(enabled);
      workspaceGrid.dataset.appTools = enabled ? "on" : "off";
      previewSurface.hidden = !enabled;
      if (enabled) {
        setPreviewStatus(
          (previewSurface.dataset.previewState as PreviewState) || "idle",
          previewMeta.textContent || "Waiting for create_app_preview",
        );
      } else {
        setPreviewStatus("idle", "App preview disabled");
      }
    }

    function log(msg: string) {
      // eslint-disable-next-line no-console
      console.log(`[mlx] ${msg}`);
    }

    function compactModelLabel(label: string) {
      return label;
    }

    function readMaxOutputTokens() {
      const parsed = Number.parseInt(maxOutputTokensInput.value, 10);
      const clamped = Math.min(
        MAX_BROWSER_OUTPUT_TOKENS,
        Math.max(
          1,
          Number.isFinite(parsed) ? parsed : DEFAULT_BROWSER_OUTPUT_TOKENS,
        ),
      );
      if (`${clamped}` !== maxOutputTokensInput.value) {
        maxOutputTokensInput.value = `${clamped}`;
      }
      return clamped;
    }

    function readTemperature() {
      const parsed = Number.parseFloat(temperatureInput.value);
      const clamped = Math.min(
        2,
        Math.max(
          0,
          Number.isFinite(parsed) ? parsed : DEFAULT_BROWSER_TEMPERATURE,
        ),
      );
      const formatted = Number.isInteger(clamped)
        ? `${clamped}`
        : `${Math.round(clamped * 100) / 100}`;
      if (formatted !== temperatureInput.value) {
        temperatureInput.value = formatted;
      }
      return clamped;
    }

    let imageCapabilityKnown = false;
    let supportsImages = false;
    let pendingImage: Uint8Array | null = null;

    function setImageAttached(attached: boolean) {
      imageBtn.dataset.attached = attached ? "true" : "false";
      imageBtn.dataset.unsupported = supportsImages ? "false" : "true";
      imageBtn.title = supportsImages
        ? attached
          ? "Image attached. Click to replace"
          : "Attach image"
        : imageCapabilityKnown
          ? "Image input unavailable for this model"
          : "Image input available after model loads";
      imageBtn.setAttribute("aria-label", imageBtn.title);
    }

    function setImageCapability(enabled: boolean) {
      imageCapabilityKnown = true;
      supportsImages = enabled;
      imageBtn.disabled = !enabled;
      if (!enabled) {
        pendingImage = null;
        imageInput.value = "";
      }
      setImageAttached(pendingImage !== null);
    }

    function autosizePrompt() {
      promptEl.style.height = "auto";
      promptEl.style.height = `${Math.min(promptEl.scrollHeight, 132)}px`;
    }

    function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
      const copy = new Uint8Array(bytes.byteLength);
      copy.set(bytes);
      return copy.buffer;
    }

    function asRecord(value: unknown): Record<string, unknown> {
      if (typeof value === "string") {
        try {
          const parsed = JSON.parse(value);
          return asRecord(parsed);
        } catch {
          return {};
        }
      }
      if (value && typeof value === "object" && !Array.isArray(value)) {
        return value as Record<string, unknown>;
      }
      return {};
    }

    function stringArg(args: Record<string, unknown>, key: string) {
      const value = args[key];
      return typeof value === "string" ? value : "";
    }

    function stripRawToolMarkupForDisplay(value: string | null | undefined) {
      let input = value ?? "";
      let output = "";

      while (input.length > 0) {
        const openIndex = input.indexOf("<tool_call>");
        if (openIndex < 0) {
          output += input;
          break;
        }

        output += input.slice(0, openIndex);
        const bodyStart = openIndex + "<tool_call>".length;
        const closeIndex = input.indexOf("</tool_call>", bodyStart);
        if (closeIndex < 0) {
          break;
        }
        input = input.slice(closeIndex + "</tool_call>".length);
      }

      const orphanMarkers = ["<function=", "<parameter="];
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
      const text = value ?? "";
      return (
        text.includes("<tool_call>") ||
        text.includes("<function=") ||
        text.includes("<parameter=")
      );
    }

    function makeUnparsedToolCallCard(finishReason: string): ToolCallResult {
      return {
        id: "call_unparsed_app_preview",
        name: APP_PREVIEW_TOOL_NAME,
        arguments: {},
        status: "parse_error",
        rawContent: "",
        error:
          `Incomplete preview tool call: model stopped before emitting a complete ` +
          `<tool_call>...</tool_call> block (finish_reason=${finishReason}).`,
      };
    }

    function escapeHtmlText(text: string) {
      return text.replace(/[<>&"]/g, (char) => {
        switch (char) {
          case "<":
            return "&lt;";
          case ">":
            return "&gt;";
          case "&":
            return "&amp;";
          case '"':
            return "&quot;";
          default:
            return char;
        }
      });
    }

    function escapeStyleBlock(text: string) {
      return text.replace(/<\/style/gi, "<\\/style");
    }

    function escapeScriptBlock(text: string) {
      return text.replace(/<\/script/gi, "<\\/script");
    }

    function wrapPreviewScript(text: string) {
      return `(() => {\n${text}\n})();`;
    }

    function isClassicScript(script: HTMLScriptElement) {
      const type = script.getAttribute("type")?.trim().toLowerCase();
      return (
        !type || type === "text/javascript" || type === "application/javascript"
      );
    }

    function wrapInlinePreviewScripts(root: ParentNode) {
      root.querySelectorAll("script").forEach((script) => {
        if (
          !(script instanceof HTMLScriptElement) ||
          !isClassicScript(script)
        ) {
          return;
        }
        const source = script.textContent ?? "";
        if (!source.trim()) return;
        script.textContent = wrapPreviewScript(source);
      });
    }

    function sanitizePreviewHtmlFragment(html: string) {
      if (!/<script[\s>]/i.test(html)) return html;
      const template = document.createElement("template");
      template.innerHTML = html;
      wrapInlinePreviewScripts(template.content);
      return template.innerHTML;
    }

    function sanitizePreviewDocument(html: string) {
      if (!/<script[\s>]/i.test(html)) return html;
      const doc = new DOMParser().parseFromString(html, "text/html");
      wrapInlinePreviewScripts(doc);
      const doctype = doc.doctype
        ? `<!doctype ${doc.doctype.name}>`
        : "<!doctype html>";
      return `${doctype}\n${doc.documentElement.outerHTML}`;
    }

    function buildPreviewDocument(args: Record<string, unknown>) {
      const title = stringArg(args, "title") || "App preview";
      const rawHtml = stringArg(args, "html");
      const css = stringArg(args, "css");
      const js = stringArg(args, "js") || stringArg(args, "javascript");
      const isDocument = /<!doctype|<html[\s>]/i.test(rawHtml);
      const html = isDocument
        ? sanitizePreviewDocument(rawHtml)
        : sanitizePreviewHtmlFragment(rawHtml);
      const cssTag = css ? `<style>\n${escapeStyleBlock(css)}\n</style>` : "";
      const jsTag = js
        ? `<script>\n${escapeScriptBlock(wrapPreviewScript(js))}\n</script>`
        : "";
      if (isDocument) {
        let doc = html;
        if (cssTag) {
          doc = /<\/head>/i.test(doc)
            ? doc.replace(/<\/head>/i, `${cssTag}\n</head>`)
            : `${cssTag}\n${doc}`;
        }
        if (jsTag) {
          doc = /<\/body>/i.test(doc)
            ? doc.replace(/<\/body>/i, `${jsTag}\n</body>`)
            : `${doc}\n${jsTag}`;
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

    function renderInlinePreview(
      bubbleEl: HTMLElement,
      title: string,
      srcdoc: string,
    ) {
      // Create a child div in the bubble, mount React InlinePreviewCard into it
      let host = bubbleEl.querySelector<HTMLDivElement>(
        ":scope > .inline-preview-host",
      );
      if (!host) {
        host = document.createElement("div");
        host.className = "inline-preview-host";
        bubbleEl.appendChild(host);
      }
      // Reuse a previously-mounted root if any (avoid re-creating)
      const existing = (host as HTMLDivElement & { __reactRoot?: Root })
        .__reactRoot;
      const root = existing ?? createRoot(host);
      (host as HTMLDivElement & { __reactRoot?: Root }).__reactRoot = root;
      root.render(<InlinePreviewCard title={title} srcdoc={srcdoc} />);
    }

    function setEmptyPreview() {
      previewFrame.srcdoc = `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body {
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-items: center;
        background: #0f1218;
        color: #8f96a3;
        font: 14px/1.5 Inter, system-ui, sans-serif;
      }
    </style>
  </head>
  <body>App preview will appear here.</body>
</html>`;
      previewTitle.textContent = "Preview";
      setPreviewStatus("idle", "Waiting for create_app_preview");
    }

    function compactToolArguments(call: ToolCallResult) {
      const args = asRecord(call.arguments);
      const title = stringArg(args, "title");
      const summary = title
        ? { title }
        : Object.fromEntries(
            Object.keys(args)
              .slice(0, 3)
              .map((key) => [key, args[key]]),
          );
      return JSON.stringify(summary);
    }

    function appendToolCell(
      call: ToolCallResult,
      state: "calling" | "result" | "error",
    ) {
      const toolDiv = document.createElement("div");
      toolDiv.className = `message tool-card ${state}`;

      const header = document.createElement("div");
      header.className = "tool-card-header";

      const bullet = document.createElement("span");
      bullet.className = "tool-card-bullet";
      bullet.setAttribute("aria-hidden", "true");
      header.appendChild(bullet);

      const title = document.createElement("div");
      title.className = "tool-card-title";
      title.textContent = `${state === "calling" ? "Calling" : state === "result" ? "Called" : "Tool error"} ${call.name}`;
      header.appendChild(title);

      const meta = document.createElement("code");
      meta.className = "tool-card-meta";
      meta.textContent = compactToolArguments(call);
      header.appendChild(meta);

      const body = document.createElement("div");
      body.className = "tool-card-body";

      toolDiv.appendChild(header);
      toolDiv.appendChild(body);
      chatEl.appendChild(toolDiv);
      chatEl.scrollTop = chatEl.scrollHeight;
      return toolDiv;
    }

    function updateToolCell(
      toolDiv: HTMLElement,
      call: ToolCallResult,
      state: "result" | "error",
      text: string,
    ) {
      toolDiv.className = `message tool-card ${state}`;
      const title = toolDiv.querySelector(".tool-card-title");
      if (title) {
        title.textContent = `${state === "result" ? "Called" : "Tool error"} ${call.name}`;
      }
      const body = toolDiv.querySelector(".tool-card-body");
      if (body) {
        body.textContent = text;
      }
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    function toAssistantToolCalls(
      toolCalls: readonly ToolCallResult[],
    ): BrowserToolCall[] {
      return toolCalls
        .filter((call) => call.status === "ok")
        .map((call) => {
          const args = asRecord(call.arguments);
          const title = stringArg(args, "title");
          const compactArgs =
            call.name === APP_PREVIEW_TOOL_NAME
              ? JSON.stringify({
                  title: title || "App preview",
                  renderedInBrowserPreview: true,
                })
              : typeof call.arguments === "string"
                ? call.arguments
                : JSON.stringify(call.arguments ?? {});
          return {
            id: call.id,
            name: call.name,
            arguments: compactArgs,
          };
        });
    }

    function executeToolCall(call: ToolCallResult): BrowserChatMessage {
      if (call.status !== "ok") {
        setPreviewStatus(
          "error",
          call.error || `Tool call parse failed: ${call.status}`,
        );
        return {
          role: "tool",
          toolCallId: call.id,
          content: JSON.stringify({
            ok: false,
            error: call.error || `Tool call parse failed: ${call.status}`,
          }),
        };
      }

      if (call.name !== APP_PREVIEW_TOOL_NAME) {
        setPreviewStatus("error", `Unknown browser tool: ${call.name}`);
        return {
          role: "tool",
          toolCallId: call.id,
          content: JSON.stringify({
            ok: false,
            error: `Unknown browser tool: ${call.name}`,
          }),
        };
      }

      const args = asRecord(call.arguments);
      const title = stringArg(args, "title") || "App preview";
      const html = stringArg(args, "html");
      const css = stringArg(args, "css");
      const js = stringArg(args, "js") || stringArg(args, "javascript");
      if (!html && !css && !js) {
        setPreviewStatus("error", "create_app_preview missing content");
        return {
          role: "tool",
          toolCallId: call.id,
          content: JSON.stringify({
            ok: false,
            error: "create_app_preview requires html, css, or js content.",
          }),
        };
      }

      previewSequence++;
      setPreviewStatus("calling", `Rendering ${title}`);
      const srcdoc = buildPreviewDocument(args);
      previewFrame.srcdoc = srcdoc;
      previewTitle.textContent = title;
      // Mount the inline preview card under the assistant bubble that owns
      // this tool call. Falls back to appending under the chat scroll
      // container if there is no current assistant bubble (e.g. the assistant
      // turn has already finalized before the tool fires).
      const previewHost: HTMLElement = currentAssistantDiv ?? chatEl;
      renderInlinePreview(previewHost, title, srcdoc);
      setPreviewStatus("rendered", `Rendered preview #${previewSequence}`);
      log(`[TOOL] ${APP_PREVIEW_TOOL_NAME} rendered "${title}"`);

      return {
        role: "tool",
        toolCallId: call.id,
        content: JSON.stringify({
          ok: true,
          previewId: `preview-${previewSequence}`,
          title,
          message: "The app is rendered in the browser preview iframe.",
        }),
      };
    }

    setEmptyPreview();
    setAppToolsEnabled(appToolsEnabledRef.current);

    const messages: BrowserChatMessage[] = [
      {
        role: "system",
        content: "You are a helpful assistant. Be concise.",
      },
    ];

    let currentAssistantDiv: HTMLDivElement | null = null;
    let currentThinkingDiv: HTMLDetailsElement | null = null;
    let currentResponseDiv: HTMLDivElement | null = null;
    let currentToolCallIndicatorDiv: HTMLDivElement | null = null;
    let currentReasoningVisible = false;
    let isInThinking = false;
    let streamTokenCount = 0;
    let toolContinuationCount = 0;
    let previewSequence = 0;

    let reasoningBuffer = "";
    let contentQueue = "";
    let responseRenderText = "";
    let contentPrefixBuffer = "";
    let contentPrefixResolved = false;
    let currentUserPrompt = "";
    let rafHandle: number | null = null;
    let scrollDirty = false;
    let reasoningHasContent = false;
    let contentHasContent = false;
    let sharedWasmMemory: WebAssembly.Memory | null = null;
    let activeReaderAbort: AbortController | null = null;
    let streamT0 = 0;
    const markdownRoots = new Map<HTMLElement, ReturnType<typeof createRoot>>();
    const contentToolCallDisplayBuffer = new ToolCallDisplayBuffer();
    const reasoningToolCallDisplayBuffer = new ToolCallDisplayBuffer();

    function renderStreamdown(
      container: HTMLElement,
      text: string,
      isStreaming: boolean,
      autoScroll = false,
    ) {
      let root = markdownRoots.get(container);
      if (!root) {
        root = createRoot(container);
        markdownRoots.set(container, root);
      }
      root.render(
        <Streamdown
          mode={isStreaming ? "streaming" : "static"}
          className="streamdown-render"
          animated={false}
          isAnimating={isStreaming}
          parseIncompleteMarkdown={isStreaming}
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
      const assistantDiv = document.createElement("div");
      assistantDiv.className = "message assistant";

      const thinkingDiv = document.createElement("details");
      thinkingDiv.className = "thinking";
      const thinkingSummary = document.createElement("summary");
      thinkingSummary.textContent = "Thinking...";
      thinkingDiv.appendChild(thinkingSummary);
      const thinkingContent = document.createElement("div");
      thinkingContent.className = "thinking-content";
      thinkingDiv.appendChild(thinkingContent);
      assistantDiv.appendChild(thinkingDiv);

      const responseDiv = document.createElement("div");
      responseDiv.className = "response-content";
      assistantDiv.appendChild(responseDiv);

      const toolCallIndicator = document.createElement("div");
      toolCallIndicator.className = "streaming-tool-call";
      toolCallIndicator.hidden = true;
      const indicatorPulse = document.createElement("span");
      indicatorPulse.className = "streaming-tool-call-pulse";
      indicatorPulse.setAttribute("aria-hidden", "true");
      toolCallIndicator.appendChild(indicatorPulse);
      const indicatorLabel = document.createElement("span");
      indicatorLabel.className = "streaming-tool-call-label";
      indicatorLabel.textContent = "Creating app preview";
      toolCallIndicator.appendChild(indicatorLabel);
      const indicatorDots = document.createElement("span");
      indicatorDots.className = "streaming-tool-call-dots";
      indicatorDots.setAttribute("aria-hidden", "true");
      for (let i = 0; i < 3; i++) {
        indicatorDots.appendChild(document.createElement("i"));
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

      thinkingDiv.style.display = "none";
      reasoningBuffer = "";
      contentQueue = "";
      responseRenderText = "";
      contentPrefixBuffer = "";
      contentPrefixResolved = false;
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
        .replace(/^(?:\((?:user)\)|user)\s*:\s*/i, "")
        .toLowerCase();
      if (!normalizedBuffer) return true;
      if (normalizedPrompt.startsWith(normalizedBuffer)) return true;

      if (normalizedBuffer.startsWith(normalizedPrompt)) {
        const rest = normalizedBuffer.slice(normalizedPrompt.length);
        return !/[\r\n]/.test(rest);
      }

      return false;
    }

    function sanitizeThinkingText(
      thinking: string | null | undefined,
      latestUserText?: string,
    ) {
      const cleaned = sanitizeThinkingMarkup(
        splitAssistantThinking(thinking, latestUserText).thinking || thinking,
      );
      if (/^[A-Z]$/u.test(cleaned)) return "";
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
      const thinkingContentEl = currentThinkingDiv.querySelector(
        ".thinking-content",
      ) as HTMLElement | null;
      const summary = currentThinkingDiv.querySelector(
        "summary",
      ) as HTMLElement | null;
      if (summary) summary.textContent = "Thought process";
      if (thinkingContentEl)
        renderStreamdown(thinkingContentEl, cleaned, true, true);
      currentThinkingDiv.style.display = "";
      currentThinkingDiv.open = open;
      reasoningHasContent = true;
      scrollDirty = true;
      scheduleFlush();
    }

    function showStreamingToolCallIndicator() {
      if (!currentAssistantDiv || !currentToolCallIndicatorDiv) return;
      currentAssistantDiv.classList.add("tool-streaming");
      currentToolCallIndicatorDiv.hidden = false;
      scrollDirty = true;
      scheduleFlush();
    }

    function hideStreamingToolCallIndicator() {
      currentAssistantDiv?.classList.remove("tool-streaming");
      if (currentToolCallIndicatorDiv) {
        currentToolCallIndicatorDiv.hidden = true;
      }
    }

    function looksLikeReasoningLeak(text: string) {
      return /(?:<\/(?:think|longcat_think)>|here'?s a thinking process|(?:^|\n)\s*(?:\d+\.\s*)?\*\*(?:draft response|check against|final output|analy[sz]e|identify))/iu.test(
        text,
      );
    }

    function appendStreamedToken(deltaText: string, isReasoning: boolean) {
      if (!currentAssistantDiv || !currentThinkingDiv || !currentResponseDiv)
        return;
      if (!deltaText) return;

      streamTokenCount++;
      setStatus(`Generating... ${streamTokenCount} tokens`, "info");

      if (isReasoning) {
        const visibleReasoning = appToolsEnabledRef.current
          ? reasoningToolCallDisplayBuffer.push(deltaText)
          : deltaText;
        if (!currentReasoningVisible) {
          if (
            appToolsEnabledRef.current &&
            reasoningToolCallDisplayBuffer.isToolCallActive()
          ) {
            showStreamingToolCallIndicator();
          }
          return;
        }
        if (visibleReasoning.length === 0) {
          if (
            appToolsEnabledRef.current &&
            reasoningToolCallDisplayBuffer.isToolCallActive()
          ) {
            showStreamingToolCallIndicator();
          }
          return;
        }
        if (
          appToolsEnabledRef.current &&
          reasoningToolCallDisplayBuffer.isToolCallActive()
        ) {
          showStreamingToolCallIndicator();
        }
        reasoningBuffer += visibleReasoning;
        setStreamingThinkingText(reasoningBuffer, true);
        return;
      } else {
        let text = contentHasContent
          ? deltaText
          : deltaText.replace(/^\s+/, "");
        if (text.length === 0) return;
        if (!contentPrefixResolved && currentUserPrompt) {
          contentPrefixBuffer += text;
          if (
            currentReasoningVisible &&
            reasoningHasContent &&
            looksLikeReasoningLeak(contentPrefixBuffer)
          ) {
            const split = splitAssistantThinking(
              contentPrefixBuffer,
              currentUserPrompt,
            );
            if (split.thinking) {
              reasoningBuffer = mergeThinkingText(
                reasoningBuffer,
                split.thinking,
              );
              setStreamingThinkingText(reasoningBuffer, true);
              text = split.text;
              contentPrefixBuffer = "";
              contentPrefixResolved = true;
              if (text.length === 0) return;
            } else if (contentPrefixBuffer.length < 16_384) {
              setStreamingThinkingText(
                mergeThinkingText(reasoningBuffer, contentPrefixBuffer),
                true,
              );
              return;
            }
          }
          if (!contentPrefixResolved) {
            const cleaned = sanitizeAssistantText(
              contentPrefixBuffer,
              currentUserPrompt,
            );
            const shouldWait =
              cleaned === contentPrefixBuffer.trimStart() &&
              couldStillBePromptEchoPrefix(
                contentPrefixBuffer,
                currentUserPrompt,
              ) &&
              contentPrefixBuffer.length < currentUserPrompt.length + 24;
            if (shouldWait) return;
            text = cleaned;
            contentPrefixBuffer = "";
            contentPrefixResolved = true;
            if (text.length === 0) return;
          }
        }
        const visibleText = appToolsEnabledRef.current
          ? contentToolCallDisplayBuffer.push(text)
          : text;
        if (visibleText.length === 0) {
          if (
            appToolsEnabledRef.current &&
            contentToolCallDisplayBuffer.isToolCallActive()
          ) {
            showStreamingToolCallIndicator();
          }
          return;
        }
        if (
          appToolsEnabledRef.current &&
          contentToolCallDisplayBuffer.isToolCallActive()
        ) {
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
        contentQueue = "";
        return;
      }

      if (contentQueue.length > 0) {
        const reveal = Math.max(1, Math.ceil(contentQueue.length / 20));
        const slice = contentQueue.slice(0, reveal);
        contentQueue = contentQueue.slice(reveal);

        if (isInThinking) {
          isInThinking = false;
          const summary = currentThinkingDiv.querySelector(
            "summary",
          ) as HTMLElement | null;
          if (summary) summary.textContent = "Thought process";
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
            const summary = currentThinkingDiv.querySelector(
              "summary",
            ) as HTMLElement | null;
            if (summary) summary.textContent = "Thought process";
            currentThinkingDiv.open = false;
          }
          responseRenderText += contentQueue;
          renderStreamdown(currentResponseDiv, responseRenderText, true);
        }
      }
      reasoningBuffer = "";
      contentQueue = "";
      contentPrefixBuffer = "";
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
      if (!currentAssistantDiv || !currentThinkingDiv || !currentResponseDiv)
        return;

      drainQueuesSync();

      const trimmedThinking = thinking?.trim() || "";
      const shouldShowThinking =
        currentReasoningVisible && trimmedThinking.length > 3;
      if (shouldShowThinking) {
        const thinkingContentEl = currentThinkingDiv.querySelector(
          ".thinking-content",
        ) as HTMLElement | null;
        if (thinkingContentEl)
          renderStreamdown(thinkingContentEl, trimmedThinking, false);
        const summary = currentThinkingDiv.querySelector(
          "summary",
        ) as HTMLElement | null;
        if (summary) summary.textContent = "Thought process";
        currentThinkingDiv.open = false;
        currentThinkingDiv.style.display = "";
      } else {
        currentThinkingDiv.style.display = "none";
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
        responseRenderText = "";
        renderStreamdown(currentResponseDiv, "", false);
      }
      currentAssistantDiv.classList.add("done");
      chatEl.scrollTop = chatEl.scrollHeight;

      currentAssistantDiv = null;
      currentThinkingDiv = null;
      currentResponseDiv = null;
      currentToolCallIndicatorDiv = null;
    }

    let worker = new Worker(new URL("../src/mlx-worker.ts", import.meta.url), {
      type: "module",
    });

    function postChatRequest() {
      const toolsEnabled = appToolsEnabledRef.current;
      const requestReasoningEffort: ReasoningEffort =
        reasoningEffortRef.current;
      const outboundMessages = messages.map((message, index) =>
        index === 0 && message.role === "system"
          ? {
              ...message,
              content: toolsEnabled
                ? "You are a helpful assistant. Be concise. When the user asks you to build, write, prototype, or preview a web app, call create_app_preview with complete self-contained HTML, CSS, and JavaScript instead of only describing the app. If reasoning is enabled, you may call create_app_preview during reasoning. After the tool result, briefly summarize what you built."
                : "You are a helpful assistant. Be concise.",
            }
          : message,
      );
      const config: Record<string, unknown> = {
        maxNewTokens: readMaxOutputTokens(),
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
        type: "chat",
        messages: outboundMessages,
        config,
        useSab,
        mode: modeParam ?? undefined,
        reasoningEffort: requestReasoningEffort,
      });
    }

    function finishChatTurn() {
      toolContinuationCount = 0;
      setStatus(`${compactModelLabel(activeModelLabel)} - Ready`, "ready");
      sendBtn.disabled = false;
      promptEl.disabled = false;
      promptEl.focus();
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    function finalizeFromResult(result: ChatResult) {
      const latestUserText =
        [...messages].reverse().find((message) => message.role === "user")
          ?.content ?? currentUserPrompt;
      const textParts = splitAssistantThinking(result.text, latestUserText);
      const rawParts = splitAssistantThinking(result.rawText, latestUserText);
      const toolsEnabled = appToolsEnabledRef.current;
      const finishReason = result.finishReason || "unknown";
      const rawToolMarkup =
        toolsEnabled &&
        (hasRawToolMarkup(result.text) ||
          hasRawToolMarkup(result.rawText) ||
          hasRawToolMarkup(result.thinking) ||
          hasRawToolMarkup(reasoningBuffer));
      let text =
        textParts.text ||
        sanitizeAssistantText(result.text, latestUserText) ||
        (toolsEnabled ? "" : rawParts.text);
      let thinking = sanitizeThinkingText(
        result.thinking ?? reasoningBuffer,
        latestUserText,
      );
      thinking = mergeThinkingText(thinking, textParts.thinking);
      if (!toolsEnabled) {
        thinking = mergeThinkingText(thinking, rawParts.thinking);
      } else {
        text = stripRawToolMarkupForDisplay(text);
        thinking = stripRawToolMarkupForDisplay(thinking);
      }
      const toolCalls = (result.toolCalls ?? []).filter(Boolean);
      const okToolCalls = toolsEnabled
        ? toolCalls.filter((call) => call.status === "ok")
        : [];
      const suppressedToolCalls = toolsEnabled
        ? []
        : toolCalls.filter((call) => call.status === "ok");
      const failedToolCalls = toolCalls.filter((call) => call.status !== "ok");
      const displayText =
        text ||
        (okToolCalls.length > 0
          ? ""
          : suppressedToolCalls.length > 0
            ? "Tool call skipped: app preview is disabled."
            : "");
      finalizeAssistantMessage(displayText, thinking || null);
      messages.push({
        role: "assistant",
        content: text,
        toolCalls:
          okToolCalls.length > 0
            ? toAssistantToolCalls(okToolCalls)
            : undefined,
      });

      if (result.performance) {
        const prefill =
          Number.isFinite(result.performance.prefillTokensPerSecond) &&
          result.performance.prefillTokensPerSecond > 0
            ? ` | Prefill ${result.performance.prefillTokensPerSecond.toFixed(1)} tok/s`
            : "";
        log(
          `${result.numTokens} tokens | finish ${finishReason} | TTFT ${result.performance.ttftMs.toFixed(0)}ms${prefill} | Decode ${result.performance.decodeTokensPerSecond.toFixed(1)} tok/s`,
        );
        setDecodeTokensPerSec(
          result.performance?.decodeTokensPerSecond ?? null,
        );
      } else {
        log(`${result.numTokens ?? 0} tokens | finish ${finishReason}`);
      }

      for (const call of failedToolCalls) {
        const toolCell = appendToolCell(call, "error");
        updateToolCell(
          toolCell,
          call,
          "error",
          call.error || `Tool call parse failed: ${call.status}`,
        );
      }
      if (suppressedToolCalls.length > 0) {
        for (const call of suppressedToolCalls) {
          const toolCell = appendToolCell(call, "error");
          updateToolCell(
            toolCell,
            call,
            "error",
            "Tool call ignored: app preview is disabled.",
          );
        }
      }
      if (rawToolMarkup && okToolCalls.length === 0) {
        const call = makeUnparsedToolCallCard(finishReason);
        const toolCell = appendToolCell(call, "error");
        updateToolCell(toolCell, call, "error", call.error || "");
      }

      if (okToolCalls.length > 0) {
        for (const call of okToolCalls) {
          const toolCell = appendToolCell(call, "calling");
          setPreviewStatus("calling", `Calling ${call.name}`);
          const toolMessage = executeToolCall(call);
          const toolResult = asRecord(toolMessage.content);
          const ok = toolResult.ok === true;
          updateToolCell(
            toolCell,
            call,
            ok ? "result" : "error",
            ok
              ? `Preview updated: ${stringArg(toolResult, "title") || call.name}`
              : stringArg(toolResult, "error") || `Tool failed: ${call.name}`,
          );
          messages.push(toolMessage);
        }
        if (!AUTO_CONTINUE_AFTER_APP_PREVIEW) {
          finishChatTurn();
          return;
        }
        if (toolContinuationCount < MAX_TOOL_CONTINUATIONS) {
          toolContinuationCount++;
          setStatus("Tool complete - generating follow-up...", "info");
          currentReasoningVisible = reasoningEffortRef.current !== "off";
          createAssistantMessage();
          postChatRequest();
          return;
        }
        const call = okToolCalls[okToolCalls.length - 1]!;
        const toolCell = appendToolCell(call, "error");
        updateToolCell(
          toolCell,
          call,
          "error",
          "Tool loop stopped: max continuations reached.",
        );
      }

      finishChatTurn();
    }

    function resetStreamingUi() {
      reasoningBuffer = "";
      contentQueue = "";
      responseRenderText = "";
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
        case "log":
          console.log(data.message);
          break;

        case "progress":
          log(data.message);
          if (data.step === "chat") {
            break;
          } else if (data.step === "download" && data.pct != null) {
            setStatus(`Downloading weights... ${data.pct}%`, "info");
          } else {
            setStatus(data.message, "info");
          }
          break;

        case "ready":
          if (typeof data.modelLabel === "string" && data.modelLabel) {
            activeModelLabel = data.modelLabel;
            composerModelLabel.textContent =
              compactModelLabel(activeModelLabel);
          }
          setModelLine(`${activeModelLabel} · bf16`);
          log("Model ready!");
          setStatus(`${compactModelLabel(activeModelLabel)} - Ready`, "ready");
          sharedWasmMemory =
            (data as { sharedMemory?: WebAssembly.Memory }).sharedMemory ??
            null;
          promptEl.disabled = false;
          sendBtn.disabled = false;
          setImageCapability(
            (data as { supportsImages?: boolean }).supportsImages === true,
          );
          break;

        case "stream-sab-open": {
          if (!sharedWasmMemory) {
            log("Error: received stream-sab-open before shared WASM memory");
            break;
          }
          streamT0 = performance.now();
          const ringOffset = (data as { ringOffset: number }).ringOffset;
          const size = (data as { size: number }).size;
          const ring = createSabRingOverHeap(
            sharedWasmMemory,
            ringOffset,
            size,
          );
          activeReaderAbort = ring.reader(
            (chunk: ChatStreamChunk) => {
              if (chunk.done) {
                const elapsed = ((performance.now() - streamT0) / 1000).toFixed(
                  1,
                );
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
                  type: "stream-finalize",
                  numTokens: chunk.numTokens ?? 0,
                });
              } else {
                appendStreamedToken(chunk.text, chunk.isReasoning ?? false);
              }
            },
            (err: Error) => {
              log(`Stream error: ${err.message}`);
              resetStreamingUi();
              if (currentResponseDiv) {
                renderStreamdown(
                  currentResponseDiv,
                  `Error: ${err.message}`,
                  false,
                );
              }
              setStatus("Error", "error");
              sendBtn.disabled = false;
              promptEl.disabled = false;
              currentAssistantDiv = null;
              currentThinkingDiv = null;
              currentResponseDiv = null;
              currentToolCallIndicatorDiv = null;
              activeReaderAbort?.abort();
              activeReaderAbort = null;
              worker.postMessage({ type: "stream-finalize", numTokens: 0 });
            },
          );
          break;
        }

        case "result":
          finalizeFromResult(data as ChatResult);
          break;

        case "profile":
          logProfile((data as { stats: ProfileStats }).stats);
          break;

        case "error":
          log(`Error: ${data.message}`);
          logStack(data.stack);
          resetStreamingUi();
          if (currentResponseDiv) {
            renderStreamdown(
              currentResponseDiv,
              `Error: ${data.message}`,
              false,
            );
          }
          setStatus("Error", "error");
          sendBtn.disabled = false;
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

        case "bridge-poisoned": {
          const reason = (data as { reason?: string }).reason ?? "unknown";
          log(
            `Bridge poisoned (${reason}). The GPU worker is unresponsive - reload the page to recover.`,
          );
          setStatus("Bridge poisoned - reload required", "error");
          sendBtn.disabled = true;
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
        .map(
          ({ op, count }) =>
            `${rpcFnName(op)}(${count}, ${(count / n).toFixed(1)}/t)`,
        )
        .join(", ");

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
      const phRate = pt > 0 ? ((ph / pt) * 100).toFixed(1) : "0.0";
      log(
        `[profile] pool: hits=${ph} (${(ph / n).toFixed(1)}/tok) misses=${pm} (${(pm / n).toFixed(1)}/tok) hitRate=${phRate}%`,
      );

      const gph = s.gpuPoolHits ?? 0;
      const gpm = s.gpuPoolMisses ?? 0;
      const gpt = gph + gpm;
      const gphRate = gpt > 0 ? ((gph / gpt) * 100).toFixed(1) : "0.0";
      log(
        `[profile] gpu-pool: hits=${gph} (${(gph / n).toFixed(1)}/tok) misses=${gpm} (${(gpm / n).toFixed(1)}/tok) hitRate=${gphRate}%`,
      );

      if (s.bindGroupCacheHits != null && s.bindGroupCacheMisses != null) {
        const bgh = s.bindGroupCacheHits;
        const bgm = s.bindGroupCacheMisses;
        const bgt = bgh + bgm;
        const bghRate = bgt > 0 ? ((bgh / bgt) * 100).toFixed(1) : "0.0";
        log(
          `[profile] bg-cache: hits=${bgh} (${(bgh / n).toFixed(1)}/tok) misses=${bgm} (${(bgm / n).toFixed(1)}/tok) hitRate=${bghRate}%`,
        );
      }

      if (s.uniformHotHits != null && s.uniformHotMisses != null) {
        const uhh = s.uniformHotHits;
        const uhm = s.uniformHotMisses;
        const uht = uhh + uhm;
        const uhhRate = uht > 0 ? ((uhh / uht) * 100).toFixed(1) : "0.0";
        log(
          `[profile] uniform-hot: hits=${uhh} (${(uhh / n).toFixed(1)}/tok) misses=${uhm} (${(uhm / n).toFixed(1)}/tok) hitRate=${uhhRate}%`,
        );
      }

      if (s.spinHits != null && s.spinMisses != null) {
        const sh = s.spinHits;
        const sm = s.spinMisses;
        const st = sh + sm;
        const shRate = st > 0 ? ((sh / st) * 100).toFixed(1) : "0.0";
        const sb = s.spinBudget ?? 0;
        log(
          `[profile] spin: hits=${sh} misses=${sm} hitRate=${shRate}% budget=${sb}`,
        );
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
      for (const line of stack.split("\n").slice(0, 12)) {
        if (line.trim()) log(`  ${line}`);
      }
    }

    const handleWorkerError = (e: ErrorEvent) => {
      e.preventDefault();
      const inner = e.error instanceof Error ? e.error.message : "";
      log(`Worker error: ${e.message || inner || String(e)}`);
      log(`  file: ${e.filename}`);
      log(`  line: ${e.lineno}, col: ${e.colno}`);
      if (e.error instanceof Error) logStack(e.error.stack);
      setStatus("Worker error", "error");
    };
    worker.onerror = handleWorkerError;

    const onMessageError = (e: MessageEvent) => {
      log(`Worker messageerror: ${e}`);
    };

    worker.addEventListener("messageerror", onMessageError);

    const urlParams = new URLSearchParams(location.search);
    const packBf16 = urlParams.get("pack_bf16") === "1";
    const sdpaFallback = urlParams.get("sdpa_fallback") === "1";
    const profile = urlParams.get("profile") === "1";
    const dispatchBatch = urlParams.get("dispatch_batch") !== "0";
    const compileMlp = urlParams.get("compile_mlp") !== "0";
    const compileGdnPre = urlParams.get("compile_gdn_pre") !== "0";
    const compileGdnPost = urlParams.get("compile_gdn_post") !== "0";
    const compileGdnG = urlParams.get("compile_gdn_g") !== "0";
    const enableVlm = urlParams.get("disable_vlm") !== "1";
    const fuseDispatch = urlParams.get("fuse_dispatch") !== "0";
    const useSab = urlParams.get("stream_sab") !== "0";
    const weightUploadBatchMb = Number.parseFloat(
      urlParams.get("weight_upload_batch_mb") ?? "",
    );
    const weightUploadBatchBytes =
      Number.isFinite(weightUploadBatchMb) && weightUploadBatchMb > 0
        ? Math.round(weightUploadBatchMb * 1024 * 1024)
        : undefined;
    const modeParam = urlParams.get("mode") as
      | "sab"
      | "tsfn"
      | "baseline"
      | null;
    const flagBadges =
      (packBf16 ? " (pack_bf16=1)" : "") +
      (sdpaFallback ? " (sdpa_fallback=1)" : "") +
      (profile ? " (profile=1)" : "") +
      (dispatchBatch ? " (dispatch_batch=1)" : "") +
      (compileMlp ? " (compile_mlp=1)" : "") +
      (compileGdnPre ? " (compile_gdn_pre=1)" : "") +
      (compileGdnPost ? " (compile_gdn_post=1)" : "") +
      (compileGdnG ? " (compile_gdn_g=1)" : "") +
      (fuseDispatch ? "" : " (fuse_dispatch=0)") +
      (enableVlm ? "" : " (disable_vlm=1)") +
      (useSab ? "" : " (stream_sab=0)") +
      (weightUploadBatchBytes
        ? ` (weight_upload_batch_mb=${Math.round(weightUploadBatchBytes / 1024 / 1024)})`
        : "") +
      (modeParam ? ` (mode=${modeParam})` : "");

    function resetForModelLoad(label?: string) {
      activeModelLabel = label ?? DEFAULT_MODEL_LABEL;
      composerModelLabel.textContent = compactModelLabel(activeModelLabel);
      setModelLine(`${activeModelLabel} · bf16`);
      activeReaderAbort?.abort();
      activeReaderAbort = null;
      sharedWasmMemory = null;
      imageCapabilityKnown = false;
      supportsImages = false;
      pendingImage = null;
      imageInput.value = "";
      setImageAttached(false);
      resetStreamingUi();
      unmountAllStreamdown();
      currentAssistantDiv = null;
      currentThinkingDiv = null;
      currentResponseDiv = null;
      currentToolCallIndicatorDiv = null;
      messages.splice(1);
      chatEl.replaceChildren();
      promptEl.disabled = true;
      sendBtn.disabled = true;
      imageBtn.disabled = true;
      setStatus("Initializing...", "info");
      log(`Starting MLX Worker${flagBadges}${label ? ` (${label})` : ""}...`);
    }

    function startWorker(
      source: {
        modelFiles?: File[];
        label?: string;
      } = {},
    ) {
      resetForModelLoad(source.label);
      worker.postMessage({
        type: "init",
        wasmUrl: new URL(`/mlx-core.opt.wasm?v=${Date.now()}`, location.href)
          .href,
        modelUrl: "/model",
        modelLabel: source.label ?? DEFAULT_MODEL_LABEL,
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

    function restartWorker(source: { modelFiles?: File[]; label?: string }) {
      worker.removeEventListener("messageerror", onMessageError);
      worker.terminate();
      worker = new Worker(new URL("../src/mlx-worker.ts", import.meta.url), {
        type: "module",
      });
      worker.onmessage = handleWorkerMessage;
      worker.onerror = handleWorkerError;
      worker.addEventListener("messageerror", onMessageError);
      startWorker(source);
    }

    startWorker();

    function handleSend() {
      const text = promptEl.value.trim();
      if (!text) return;

      promptEl.value = "";
      autosizePrompt();
      sendBtn.disabled = true;
      promptEl.disabled = true;

      if (pendingImage && !supportsImages) {
        log("Image removed: the loaded model does not support vision input.");
        pendingImage = null;
        imageInput.value = "";
        setImageAttached(false);
      }

      const userDiv = document.createElement("div");
      userDiv.className = "message user";
      userDiv.textContent = text;
      if (pendingImage) {
        const img = document.createElement("img");
        img.src = URL.createObjectURL(new Blob([toArrayBuffer(pendingImage)]));
        img.className = "attached-image";
        userDiv.appendChild(img);
      }
      chatEl.appendChild(userDiv);

      const msg: BrowserChatMessage = {
        role: "user",
        content: text,
      };
      if (pendingImage) {
        msg.images = [pendingImage];
        pendingImage = null;
        imageInput.value = "";
        setImageAttached(false);
      }
      messages.push(msg);

      setStatus("Generating...", "info");
      streamTokenCount = 0;
      toolContinuationCount = 0;
      isInThinking = false;
      currentUserPrompt = text;
      currentReasoningVisible = reasoningEffortRef.current !== "off";
      createAssistantMessage();

      postChatRequest();
    }

    const onPromptKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    };

    const onImageClick = () => {
      if (!supportsImages) {
        setStatus("Image input unavailable", "info");
        log("Image input unavailable for this model.");
        return;
      }
      imageInput.click();
    };
    const onImageChange = async () => {
      const file = imageInput.files?.[0];
      if (!file) return;
      if (!supportsImages) {
        imageInput.value = "";
        setStatus("Image input unavailable", "info");
        log("Image ignored: the loaded model does not support vision input.");
        return;
      }
      pendingImage = new Uint8Array(await file.arrayBuffer());
      imageInput.value = "";
      log(
        `Image attached: ${file.name} (${(pendingImage.length / 1024).toFixed(0)} KB)`,
      );
      setImageAttached(true);
    };

    const onModelDirChange = () => {
      const files = Array.from(modelDirInput.files ?? []);
      if (files.length === 0) return;
      const firstPath =
        (files[0] as File & { webkitRelativePath?: string })
          .webkitRelativePath || files[0]!.name;
      const label = firstPath.split("/")[0] || "local model";
      modelDirInput.value = "";
      restartWorker({ modelFiles: files, label });
    };

    const onPaste = async (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          if (!supportsImages) {
            setStatus("Image input unavailable", "info");
            log(
              "Image paste ignored: the loaded model does not support vision input.",
            );
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

    sendBtn.addEventListener("click", handleSend);
    promptEl.addEventListener("keydown", onPromptKeyDown);
    promptEl.addEventListener("input", autosizePrompt);
    imageBtn.addEventListener("click", onImageClick);
    imageInput.addEventListener("change", onImageChange);
    modelDirInput.addEventListener("change", onModelDirChange);
    document.addEventListener("paste", onPaste);

    // Expose a global reset hook for the new ChatHeader's "Reset Chat" button.
    // Clears the chat DOM, resets the local message history (keeping the
    // system prompt), and tears down any in-flight streaming UI state. The
    // underlying worker session will get a fresh start on the next chat
    // message because messages[] now carries only the system prompt.
    (
      window as unknown as { __mlxResetChat?: () => void }
    ).__mlxResetChat = () => {
      activeReaderAbort?.abort();
      activeReaderAbort = null;
      resetStreamingUi();
      unmountAllStreamdown();
      currentAssistantDiv = null;
      currentThinkingDiv = null;
      currentResponseDiv = null;
      currentToolCallIndicatorDiv = null;
      messages.splice(1);
      chatEl.replaceChildren();
    };

    return () => {
      sendBtn.removeEventListener("click", handleSend);
      promptEl.removeEventListener("keydown", onPromptKeyDown);
      promptEl.removeEventListener("input", autosizePrompt);
      imageBtn.removeEventListener("click", onImageClick);
      imageInput.removeEventListener("change", onImageChange);
      modelDirInput.removeEventListener("change", onModelDirChange);
      document.removeEventListener("paste", onPaste);
      worker.removeEventListener("messageerror", onMessageError);
      activeReaderAbort?.abort();
      if (rafHandle != null) cancelAnimationFrame(rafHandle);
      unmountAllStreamdown();
      worker.terminate();
      delete (window as unknown as { __mlxResetChat?: () => void })
        .__mlxResetChat;
    };
  }, [loadKickoff]);

  return (
    <div className="app-root">
      <div
        className={`chat-layer ${screen === "chat" ? "visible" : ""}`}
      >
    <div className="app-shell">
      <ChatHeader
        onReset={() => {
          // The existing useEffect installs a global reset hook on
          // window.__mlxResetChat when the chat session is established. If
          // it's set, call it; the chat DOM will clear and the underlying
          // chat session will be reset.
          if (
            (window as unknown as { __mlxResetChat?: () => void })
              .__mlxResetChat
          ) {
            (
              window as unknown as { __mlxResetChat: () => void }
            ).__mlxResetChat();
          }
          dispatchScreen({ type: "reset_chat" });
        }}
      />
      {/*
        Hidden ref-only status element. The legacy app-header rendered the
        Initializing status pill; we kept the ref so the existing useEffect's
        statusEl.textContent / statusEl.className mutations remain safe. The
        new ChatHeader shows its own static "Ready on WebGPU" status.
      */}
      <span
        ref={statusRef}
        id="status"
        style={{ display: "none" }}
        aria-hidden="true"
      />


      <main
        ref={workspaceGridRef}
        className="workspace-grid"
        data-app-tools={initialAppToolsEnabled ? "on" : "off"}
      >
        <Card className="surface-card">
          <CardHeader className="surface-header">
            <div className="surface-title-row">
              <div>
                <CardTitle className="surface-title">Chat</CardTitle>
                <CardDescription className="surface-meta">
                  Streaming response
                </CardDescription>
              </div>
              <CardAction>
                <Cpu />
              </CardAction>
            </div>
          </CardHeader>
          <CardContent className="surface-content">
            <div id="chat" ref={chatRef} />
          </CardContent>
        </Card>

      </main>

      <TelemetryStrip
        stats={telemetryStats}
        decodeTokensPerSecond={decodeTokensPerSec}
        modelLine={modelLine}
      />

      <footer className="composer-bar">
        <div className="composer-shell">
          <Textarea
            id="prompt"
            ref={promptRef}
            rows={1}
            placeholder="Message the model..."
            disabled
            className="composer-input"
          />
          <div className="composer-actions">
            <div className="composer-actions-left">
              <Button
                id="image-btn"
                ref={imageButtonRef}
                type="button"
                variant="ghost"
                size="icon"
                className="composer-icon-btn"
                disabled
              >
                <ImagePlus data-icon="inline-start" />
                <span className="sr-only">Attach image</span>
              </Button>
            </div>
            <div className="composer-actions-right">
              <label
                className="composer-tool-toggle"
                title="Enable app-preview tool calls"
              >
                <span className="composer-tool-toggle-label">Preview</span>
                <Switch
                  size="sm"
                  checked={appToolsEnabled}
                  aria-label="App preview tool calls"
                  onCheckedChange={(checked) => {
                    const enabled = checked === true;
                    setAppToolsEnabledState(enabled);
                    appToolsEnabledRef.current = enabled;
                    if (workspaceGridRef.current) {
                      workspaceGridRef.current.dataset.appTools = enabled
                        ? "on"
                        : "off";
                    }
                    if (previewSurfaceRef.current) {
                      previewSurfaceRef.current.hidden = !enabled;
                    }
                    if (previewMetaRef.current) {
                      if (enabled) {
                        if (
                          previewMetaRef.current.textContent ===
                          "App preview disabled"
                        ) {
                          previewMetaRef.current.textContent =
                            "Waiting for create_app_preview";
                        }
                      } else {
                        previewMetaRef.current.textContent =
                          "App preview disabled";
                      }
                    }
                  }}
                />
              </label>
              <span className="composer-status-dot" aria-hidden="true" />
              <span
                ref={composerModelLabelRef}
                className="composer-model-label"
              >
                {DEFAULT_MODEL_LABEL}
              </span>
              <input
                ref={temperatureInputRef}
                className="composer-temperature-input"
                type="number"
                min={0}
                max={2}
                step={0.1}
                defaultValue={initialTemperature}
                aria-label="Temperature"
                title="Temperature (0-2)"
              />
              <input
                ref={maxOutputTokensInputRef}
                className="composer-max-output-input"
                type="number"
                min={1}
                max={MAX_BROWSER_OUTPUT_TOKENS}
                step={1}
                defaultValue={initialMaxOutputTokens}
                aria-label="Max output tokens"
                title={`Max output tokens (1-${MAX_BROWSER_OUTPUT_TOKENS})`}
              />
              <Select
                value={reasoningEffort}
                onValueChange={(value) => {
                  reasoningEffortRef.current = value as ReasoningEffort;
                  setReasoningEffortState(value as ReasoningEffort);
                }}
              >
                <SelectTrigger
                  id="reasoning-effort"
                  size="sm"
                  className="composer-reasoning-trigger"
                  aria-label="Reasoning effort"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectItem value="off">Off</SelectItem>
                    <SelectItem value="low">Low</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="high">High</SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="composer-icon-btn"
                aria-label="Voice input"
              >
                <Mic data-icon="inline-start" />
              </Button>
              <Button
                id="send"
                ref={sendRef}
                type="button"
                size="icon"
                className="composer-send-btn"
                disabled
              >
                <ArrowUp data-icon="inline-start" />
                <span className="sr-only">Send</span>
              </Button>
            </div>
          </div>
        </div>
        <input
          id="image-input"
          ref={imageInputRef}
          type="file"
          accept="image/*"
          className="hidden"
        />
      </footer>
    </div>
      </div>

      {/*
        Always-mounted hidden file input for picking a local model directory.
        Lives at the App root (outside the screen-conditional Landing/Loading
        overlays) so that the `useEffect` keyed on `loadKickoff` can attach
        `webkitdirectory` + `change` listeners regardless of which screen is
        active when the load is kicked off.
      */}
      <input
        id="model-dir-input"
        ref={modelDirInputRef}
        type="file"
        multiple
        className="hidden"
      />

      {screen === "landing" && (
        <Landing
          onLoad={() => {
            setErrorBannerState(null);
            setLoadKickoff((k) => k + 1);
            dispatchScreen({ type: "load_kickoff" });
          }}
          onLocalModel={() => modelDirInputRef.current?.click()}
          modelDirInputRef={modelDirInputRef}
          errorBanner={errorBanner}
        />
      )}
      {screen === "loading" && <Loading status={loadingText} />}
    </div>
  );
}

function rpcFnName(fn: number): string {
  switch (fn) {
    case 1:
      return "CREATE_INSTANCE";
    case 2:
      return "INSTANCE_REQUEST_ADAPTER";
    case 3:
      return "INSTANCE_RELEASE";
    case 4:
      return "ADAPTER_REQUEST_DEVICE";
    case 5:
      return "ADAPTER_RELEASE";
    case 6:
      return "ADAPTER_GET_PROPERTIES";
    case 10:
      return "DEVICE_CREATE_BUFFER";
    case 11:
      return "DEVICE_CREATE_SHADER_MODULE";
    case 12:
      return "DEVICE_CREATE_COMPUTE_PIPELINE";
    case 13:
      return "DEVICE_CREATE_BIND_GROUP";
    case 14:
      return "DEVICE_CREATE_COMMAND_ENCODER";
    case 15:
      return "DEVICE_GET_QUEUE";
    case 16:
      return "DEVICE_GET_LIMITS";
    case 17:
      return "DEVICE_SET_ERROR_CALLBACK";
    case 18:
      return "DEVICE_SET_LOST_CALLBACK";
    case 19:
      return "DEVICE_RELEASE";
    case 20:
      return "QUEUE_SUBMIT";
    case 21:
      return "QUEUE_WRITE_BUFFER";
    case 22:
      return "QUEUE_ON_SUBMITTED_WORK_DONE";
    case 23:
      return "QUEUE_RELEASE";
    case 30:
      return "CMD_ENCODER_BEGIN_COMPUTE_PASS";
    case 31:
      return "CMD_ENCODER_COPY_BUFFER";
    case 32:
      return "CMD_ENCODER_FINISH";
    case 33:
      return "CMD_ENCODER_RELEASE";
    case 34:
      return "CMD_BUFFER_RELEASE";
    case 40:
      return "COMPUTE_PASS_SET_PIPELINE";
    case 41:
      return "COMPUTE_PASS_SET_BIND_GROUP";
    case 42:
      return "COMPUTE_PASS_DISPATCH";
    case 43:
      return "COMPUTE_PASS_END";
    case 44:
      return "COMPUTE_PASS_RELEASE";
    case 50:
      return "BUFFER_GET_SIZE";
    case 51:
      return "BUFFER_GET_MAPPED_RANGE";
    case 52:
      return "BUFFER_GET_CONST_MAPPED_RANGE";
    case 53:
      return "BUFFER_UNMAP";
    case 54:
      return "BUFFER_MAP_ASYNC";
    case 55:
      return "BUFFER_DESTROY";
    case 56:
      return "BUFFER_RELEASE";
    case 60:
      return "PIPELINE_GET_BIND_GROUP_LAYOUT";
    case 61:
      return "PIPELINE_RELEASE";
    case 70:
      return "BIND_GROUP_RELEASE";
    case 71:
      return "BIND_GROUP_LAYOUT_RELEASE";
    case 72:
      return "SHADER_MODULE_RELEASE";
    case 80:
      return "POLL";
    case 90:
      return "ADD_GPU_BUFFER";
    case 91:
      return "FUSED_DISPATCH";
    case 92:
      return "FUSED_DISPATCH_2BG";
    case 93:
      return "FUSED_SUBMIT";
    case 94:
      return "FUSED_BG_DISPATCH";
    case 95:
      return "CREATE_BUFFER_FROM_DATA";
    case 96:
      return "FUSED_FULL_DISPATCH";
    case 97:
      return "FUSED_DISPATCH_WITH_UNIFORM";
    case 98:
      return "FUSED_COPY_BUFFER";
    case 99:
      return "GET_STATS";
    case 102:
      return "BUFFER_RELEASE_BATCH";
    case 103:
      return "DISPATCH_BATCH";
    default:
      return `fn${fn}`;
  }
}

createRoot(document.getElementById("root")!).render(<App />);
