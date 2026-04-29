import type { ChatStreamChunk } from "@mlx-node/core";

import {
  ArrowUp,
  Cpu,
  Download,
  FolderOpen,
  ImagePlus,
  Mic,
  TerminalSquare,
} from "lucide-react";
import { useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";

import { createSabRingOverHeap } from "../src/chat-stream-sab.js";
import { Badge } from "./components/ui/badge";
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
import { Textarea } from "./components/ui/textarea";
import {
  sanitizeAssistantText,
  sanitizeThinkingText as sanitizeThinkingMarkup,
  splitAssistantThinking,
} from "../src/generated-text.js";
import "./styles.css";

type StatusState = "info" | "ready" | "error";
type ReasoningEffort = "off" | "low" | "medium" | "high";
const DEFAULT_MODEL_LABEL = "Qwen3.6-35b-a3b-UD-Q3_K_XL-mlx";
const MAX_BROWSER_OUTPUT_TOKENS = 36864;
const DEFAULT_BROWSER_OUTPUT_TOKENS = 1024;
const DEFAULT_BROWSER_TEMPERATURE = 0.6;

type ChatResult = {
  text?: string;
  rawText?: string;
  thinking?: string | null;
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
  const logRef = useRef<HTMLDivElement>(null);
  const chatRef = useRef<HTMLDivElement>(null);
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const sendRef = useRef<HTMLButtonElement>(null);
  const imageButtonRef = useRef<HTMLButtonElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const modelDirInputRef = useRef<HTMLInputElement>(null);
  const hfFormRef = useRef<HTMLFormElement>(null);
  const hfRepoInputRef = useRef<HTMLInputElement>(null);
  const hfRevisionInputRef = useRef<HTMLInputElement>(null);
  const hfLoadButtonRef = useRef<HTMLButtonElement>(null);
  const composerModelLabelRef = useRef<HTMLSpanElement>(null);
  const temperatureInputRef = useRef<HTMLInputElement>(null);
  const maxOutputTokensInputRef = useRef<HTMLInputElement>(null);
  const reasoningEffortRef = useRef<ReasoningEffort>("off");
  const initialUrlParams = new URLSearchParams(location.search);
  const initialHfRepo = initialUrlParams.get("hf") ?? "";
  const initialHfRevision = initialUrlParams.get("revision") ?? "main";
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
    const statusEl = statusRef.current!;
    const logEl = logRef.current!;
    const chatEl = chatRef.current!;
    const promptEl = promptRef.current!;
    const sendBtn = sendRef.current!;
    const imageBtn = imageButtonRef.current!;
    const imageInput = imageInputRef.current!;
    const modelDirInput = modelDirInputRef.current!;
    const hfForm = hfFormRef.current!;
    const hfRepoInput = hfRepoInputRef.current!;
    const hfRevisionInput = hfRevisionInputRef.current!;
    const hfLoadBtn = hfLoadButtonRef.current!;
    const composerModelLabel = composerModelLabelRef.current!;
    const temperatureInput = temperatureInputRef.current!;
    const maxOutputTokensInput = maxOutputTokensInputRef.current!;

    if (
      !statusEl ||
      !logEl ||
      !chatEl ||
      !promptEl ||
      !sendBtn ||
      !imageBtn ||
      !imageInput ||
      !modelDirInput ||
      !hfForm ||
      !hfRepoInput ||
      !hfRevisionInput ||
      !hfLoadBtn ||
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
    }

    function log(msg: string) {
      const line = document.createElement("div");
      line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
      logEl.appendChild(line);
      logEl.scrollTop = logEl.scrollHeight;
    }

    function setHfControlsDisabled(disabled: boolean) {
      hfRepoInput.disabled = disabled;
      hfRevisionInput.disabled = disabled;
      hfLoadBtn.disabled = disabled;
    }

    function normalizeHfRepoInput(value: string) {
      return value
        .trim()
        .replace(/^https:\/\/huggingface\.co\//i, "")
        .replace(/^hf:\/\//i, "")
        .replace(/^models\//i, "")
        .replace(/\/(?:tree|resolve)\/.*$/i, "")
        .replace(/^\/+|\/+$/g, "");
    }

    function compactModelLabel(label: string) {
      if (!label.startsWith("HF ")) return label;
      const full = label.slice(3);
      const [repoId, revision] = full.split("@");
      const name = repoId?.split("/").pop() || repoId || full;
      return revision ? `${name}@${revision}` : name;
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

    const messages: Array<{
      role: string;
      content: string;
      images?: Uint8Array[];
    }> = [
      { role: "system", content: "You are a helpful assistant. Be concise." },
    ];

    let currentAssistantDiv: HTMLDivElement | null = null;
    let currentThinkingDiv: HTMLDetailsElement | null = null;
    let currentResponseDiv: HTMLDivElement | null = null;
    let currentReasoningVisible = false;
    let isInThinking = false;
    let streamTokenCount = 0;

    let reasoningBuffer = "";
    let contentQueue = "";
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

      chatEl.appendChild(assistantDiv);
      chatEl.scrollTop = chatEl.scrollHeight;

      currentAssistantDiv = assistantDiv;
      currentThinkingDiv = thinkingDiv;
      currentResponseDiv = responseDiv;
      isInThinking = false;

      thinkingDiv.style.display = "none";
      reasoningBuffer = "";
      contentQueue = "";
      contentPrefixBuffer = "";
      contentPrefixResolved = false;
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
      if (thinkingContentEl) thinkingContentEl.textContent = cleaned;
      currentThinkingDiv.style.display = "";
      currentThinkingDiv.open = open;
      reasoningHasContent = true;
      scrollDirty = true;
      scheduleFlush();
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
        if (!currentReasoningVisible) return;
        reasoningBuffer += deltaText;
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
        contentHasContent = true;
        contentQueue += text;
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
        currentResponseDiv.textContent += slice;
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
          if (isInThinking) {
            isInThinking = false;
            const summary = currentThinkingDiv.querySelector(
              "summary",
            ) as HTMLElement | null;
            if (summary) summary.textContent = "Thought process";
            currentThinkingDiv.open = false;
          }
          currentResponseDiv.textContent += contentQueue;
        }
      }
      reasoningBuffer = "";
      contentQueue = "";
      contentPrefixBuffer = "";
      contentPrefixResolved = false;
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
      if (currentReasoningVisible && trimmedThinking.length > 3) {
        const thinkingContentEl = currentThinkingDiv.querySelector(
          ".thinking-content",
        ) as HTMLElement | null;
        if (thinkingContentEl) thinkingContentEl.textContent = trimmedThinking;
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
        currentResponseDiv.textContent = text;
      }
      currentAssistantDiv.classList.add("done");
      chatEl.scrollTop = chatEl.scrollHeight;

      currentAssistantDiv = null;
      currentThinkingDiv = null;
      currentResponseDiv = null;
    }

    let worker = new Worker(new URL("../src/mlx-worker.ts", import.meta.url), {
      type: "module",
    });

    function finalizeFromResult(result: ChatResult) {
      const latestUserText =
        [...messages].reverse().find((message) => message.role === "user")
          ?.content ?? currentUserPrompt;
      const textParts = splitAssistantThinking(result.text, latestUserText);
      const rawParts = splitAssistantThinking(result.rawText, latestUserText);
      let text =
        rawParts.text ||
        textParts.text ||
        sanitizeAssistantText(result.text, latestUserText);
      let thinking = sanitizeThinkingText(
        result.thinking ?? reasoningBuffer,
        latestUserText,
      );
      thinking = mergeThinkingText(thinking, textParts.thinking);
      thinking = mergeThinkingText(thinking, rawParts.thinking);
      const rawText =
        text || sanitizeAssistantText(result.rawText, latestUserText);
      finalizeAssistantMessage(text, thinking || null);
      messages.push({ role: "assistant", content: rawText });

      if (result.performance) {
        log(
          `${result.numTokens} tokens | TTFT ${result.performance.ttftMs.toFixed(0)}ms | Decode ${result.performance.decodeTokensPerSecond.toFixed(1)} tok/s`,
        );
      }
      setStatus(`${compactModelLabel(activeModelLabel)} - Ready`, "ready");
      sendBtn.disabled = false;
      promptEl.disabled = false;
      setHfControlsDisabled(false);
      promptEl.focus();
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    function resetStreamingUi() {
      reasoningBuffer = "";
      contentQueue = "";
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
          log("Model ready!");
          setStatus(`${compactModelLabel(activeModelLabel)} - Ready`, "ready");
          sharedWasmMemory =
            (data as { sharedMemory?: WebAssembly.Memory }).sharedMemory ??
            null;
          promptEl.disabled = false;
          sendBtn.disabled = false;
          setHfControlsDisabled(false);
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
                currentResponseDiv.textContent = `Error: ${err.message}`;
              }
              setStatus("Error", "error");
              sendBtn.disabled = false;
              promptEl.disabled = false;
              setHfControlsDisabled(false);
              currentAssistantDiv = null;
              currentThinkingDiv = null;
              currentResponseDiv = null;
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
            currentResponseDiv.textContent = `Error: ${data.message}`;
          }
          setStatus("Error", "error");
          sendBtn.disabled = false;
          promptEl.disabled = false;
          setHfControlsDisabled(false);
          currentAssistantDiv = null;
          currentThinkingDiv = null;
          currentResponseDiv = null;
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
          setHfControlsDisabled(false);
          break;
        }
      }
    };
    worker.onmessage = handleWorkerMessage;

    function logProfile(s: ProfileStats) {
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
      setHfControlsDisabled(false);
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
      (modeParam ? ` (mode=${modeParam})` : "");

    function resetForModelLoad(label?: string) {
      activeModelLabel = label ?? DEFAULT_MODEL_LABEL;
      composerModelLabel.textContent = compactModelLabel(activeModelLabel);
      activeReaderAbort?.abort();
      activeReaderAbort = null;
      sharedWasmMemory = null;
      imageCapabilityKnown = false;
      supportsImages = false;
      pendingImage = null;
      imageInput.value = "";
      setImageAttached(false);
      resetStreamingUi();
      currentAssistantDiv = null;
      currentThinkingDiv = null;
      currentResponseDiv = null;
      messages.splice(1);
      chatEl.replaceChildren();
      promptEl.disabled = true;
      sendBtn.disabled = true;
      imageBtn.disabled = true;
      setHfControlsDisabled(true);
      setStatus("Initializing...", "info");
      log(`Starting MLX Worker${flagBadges}${label ? ` (${label})` : ""}...`);
    }

    function startWorker(
      source: {
        modelFiles?: File[];
        hfModel?: { repoId: string; revision?: string };
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
        hfModel: source.hfModel,
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
      });
    }

    function restartWorker(source: {
      modelFiles?: File[];
      hfModel?: { repoId: string; revision?: string };
      label?: string;
    }) {
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

    const initialRepoId = normalizeHfRepoInput(urlParams.get("hf") ?? "");
    if (initialRepoId) {
      const revision = urlParams.get("revision")?.trim() || "main";
      hfRepoInput.value = initialRepoId;
      hfRevisionInput.value = revision;
      startWorker({
        hfModel: { repoId: initialRepoId, revision },
        label: `HF ${initialRepoId}@${revision}`,
      });
    } else {
      startWorker();
    }

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

      const msg: { role: string; content: string; images?: Uint8Array[] } = {
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
      isInThinking = false;
      currentUserPrompt = text;
      currentReasoningVisible = reasoningEffortRef.current !== "off";
      createAssistantMessage();

      worker.postMessage({
        type: "chat",
        messages: [...messages],
        config: {
          maxNewTokens: readMaxOutputTokens(),
          temperature: readTemperature(),
          reportPerformance: true,
        },
        useSab,
        mode: modeParam ?? undefined,
        reasoningEffort: reasoningEffortRef.current,
      });
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

    const onHfSubmit = (event: Event) => {
      event.preventDefault();
      const repoId = normalizeHfRepoInput(hfRepoInput.value);
      const revision = hfRevisionInput.value.trim() || "main";
      if (!repoId) {
        setStatus("Enter a Hugging Face repo", "error");
        hfRepoInput.focus();
        return;
      }
      hfRepoInput.value = repoId;
      hfRevisionInput.value = revision;
      restartWorker({
        hfModel: { repoId, revision },
        label: `HF ${repoId}@${revision}`,
      });
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
    hfForm.addEventListener("submit", onHfSubmit);
    document.addEventListener("paste", onPaste);

    return () => {
      sendBtn.removeEventListener("click", handleSend);
      promptEl.removeEventListener("keydown", onPromptKeyDown);
      promptEl.removeEventListener("input", autosizePrompt);
      imageBtn.removeEventListener("click", onImageClick);
      imageInput.removeEventListener("change", onImageChange);
      modelDirInput.removeEventListener("change", onModelDirChange);
      hfForm.removeEventListener("submit", onHfSubmit);
      document.removeEventListener("paste", onPaste);
      worker.removeEventListener("messageerror", onMessageError);
      activeReaderAbort?.abort();
      if (rafHandle != null) cancelAnimationFrame(rafHandle);
      worker.terminate();
    };
  }, []);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-mark">MLX</div>
        <div className="brand-copy">
          <h1>MLX Browser</h1>
          <p>WebGPU model playground</p>
        </div>
        <div className="header-actions">
          <form ref={hfFormRef} className="header-hf-form">
            <input
              ref={hfRepoInputRef}
              className="hf-input hf-repo-input"
              type="text"
              placeholder="Hugging Face repo"
              defaultValue={initialHfRepo}
              spellCheck={false}
              autoCapitalize="none"
              aria-label="Hugging Face model repository"
            />
            <input
              ref={hfRevisionInputRef}
              className="hf-input hf-revision-input"
              type="text"
              placeholder="main"
              defaultValue={initialHfRevision}
              spellCheck={false}
              autoCapitalize="none"
              aria-label="Hugging Face model revision"
            />
            <Button
              ref={hfLoadButtonRef}
              type="submit"
              variant="outline"
              size="sm"
              className="header-hf-btn"
            >
              <Download data-icon="inline-start" />
              <span>Load HF</span>
            </Button>
          </form>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="header-local-btn"
            onClick={() => modelDirInputRef.current?.click()}
          >
            <FolderOpen data-icon="inline-start" />
            <span>Local model</span>
          </Button>
          <Badge ref={statusRef} id="status" className="status-pill info">
            Initializing...
          </Badge>
        </div>
      </header>

      <main className="workspace-grid">
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

        <Card className="surface-card">
          <CardHeader className="surface-header">
            <div className="surface-title-row">
              <div>
                <CardTitle className="surface-title">Telemetry</CardTitle>
                <CardDescription className="surface-meta">
                  Worker and decode log
                </CardDescription>
              </div>
              <CardAction>
                <TerminalSquare />
              </CardAction>
            </div>
          </CardHeader>
          <CardContent className="surface-content">
            <div id="log" ref={logRef} />
          </CardContent>
        </Card>
      </main>

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
                defaultValue="off"
                onValueChange={(value) => {
                  reasoningEffortRef.current = value as ReasoningEffort;
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
        <input
          id="model-dir-input"
          ref={modelDirInputRef}
          type="file"
          multiple
          className="hidden"
        />
      </footer>
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
