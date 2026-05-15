/**
 * MLX Inference Worker
 *
 * Runs the entire MLX pipeline on a dedicated Web Worker:
 * - Creates its own GPUDevice (navigator.gpu works in workers)
 * - Loads WASM with real WebGPU bridge
 * - Fetches + loads model weights directly into GPU buffers
 * - Runs inference (chat/generate) without blocking the main thread
 *
 * The main thread communicates via postMessage only.
 */
import { Buffer } from "buffer";
(globalThis as any).Buffer = Buffer;

// Revision marker: changes worker asset URLs after enabling COOP/COEP headers
// on Void, avoiding stale immutable edge-cache entries without those headers.
(globalThis as any).__MLX_VOID_COEP_WORKER_ASSET_REV = "2026-05-15";

// Patch TextDecoder to handle SharedArrayBuffer views.
// WASM memory is a SharedArrayBuffer (for threads support), but TextDecoder
// rejects shared views. Copy to a non-shared buffer before decoding.
const _origDecode = TextDecoder.prototype.decode;
TextDecoder.prototype.decode = function (
  input?: BufferSource | null,
  options?: TextDecodeOptions,
) {
  if (
    input &&
    ArrayBuffer.isView(input) &&
    (input.buffer as any) instanceof SharedArrayBuffer
  ) {
    input = (input as Uint8Array).slice();
  }
  return _origDecode.call(this, input, options);
};

import {
  instantiateNapiModule,
  getDefaultContext,
  WASI,
} from "@napi-rs/wasm-runtime";

import {
  CMD_OFFSET,
  READBACK_BUFFER_SIZE,
  DISPATCH_BATCH_BUFFER_SIZE,
  STATS_BUFFER_SIZE,
} from "./rpc-protocol.js";
import {
  parseSafeTensorsHeaderBytes,
  dtypeToCode,
  type TensorInfo,
} from "./safetensors.js";
import gpuWorkerUrl from "./gpu-worker.ts?worker&url";
import webgpuWorkerUrl from "./webgpu-worker.mjs?worker&url";
import {
  createBridgeStub,
  POOL_STATS_SIZE_BYTES,
  BUFFER_METADATA_SIZE_BYTES,
  type BridgeStub,
} from "./webgpu-bridge-stub.js";
import { workerAssetUrl } from "./worker-asset-url.js";

let model: any = null;
let mlxExports: any = null;
let wasmInst: WebAssembly.Instance | null = null;
let cppTag: any = null;
let wasmMemory: WebAssembly.Memory | null = null;
let wasmMalloc: ((size: number) => number) | null = null;
let wasmFree: ((ptr: number) => void) | null = null;
let modelSupportsImages = false;
let bridgeOptimizationControl: Int32Array | null = null;
let configuredFusionEnabled = true;
let configuredDispatchBatchEnabled = false;
let modelGenerationDefaults: Record<string, number | boolean> = {};

// GPU-buffer import currently wraps JS-created WGPUBuffer handles before Rust
// builds Linear::weight_t transpose views. Until packed-bf16 metadata is proven
// to survive that path end-to-end, keep JS-side packed weight upload disabled:
// otherwise raw bf16 pairs can be read by normal f32 kernels and corrupt logits.
const PACKED_GPU_WEIGHT_UPLOAD_ENABLED = false;

type UploadTensorInfo = TensorInfo & { fromMerged?: boolean };
type ReasoningEffort = "off" | "low" | "medium" | "high";
type LocalModelFile = File & { webkitRelativePath?: string };
type HuggingFaceModelConfig = {
  repoId: string;
  revision?: string;
};

function normalizeGenerationConfig(
  raw: unknown,
): Record<string, number | boolean> {
  if (!raw || typeof raw !== "object") return {};
  const cfg = raw as Record<string, unknown>;
  const out: Record<string, number | boolean> = {};
  const assignNumber = (src: string, dst: string = src) => {
    const value = cfg[src];
    if (typeof value === "number" && Number.isFinite(value)) {
      out[dst] = value;
    }
  };
  assignNumber("temperature");
  assignNumber("top_k", "topK");
  assignNumber("top_p", "topP");
  assignNumber("min_p", "minP");
  assignNumber("repetition_penalty", "repetitionPenalty");
  if (typeof cfg.do_sample === "boolean") out.doSample = cfg.do_sample;
  return out;
}

const VERY_LOW_MEMORY_WEIGHT_UPLOAD_BATCH_BYTES = 64 * 1024 * 1024;
const LOW_MEMORY_WEIGHT_UPLOAD_BATCH_BYTES = 128 * 1024 * 1024;
const DEFAULT_WEIGHT_UPLOAD_BATCH_BYTES = 256 * 1024 * 1024;
const HIGH_MEMORY_WEIGHT_UPLOAD_BATCH_BYTES = 384 * 1024 * 1024;
const VERY_HIGH_MEMORY_WEIGHT_UPLOAD_BATCH_BYTES = 512 * 1024 * 1024;
const WORKSTATION_MEMORY_WEIGHT_UPLOAD_BATCH_BYTES = 640 * 1024 * 1024;
const LARGE_WORKSTATION_WEIGHT_UPLOAD_BATCH_BYTES = 768 * 1024 * 1024;
const HUGE_WORKSTATION_WEIGHT_UPLOAD_BATCH_BYTES = 896 * 1024 * 1024;
const MAX_MEMORY_WEIGHT_UPLOAD_BATCH_BYTES = 1024 * 1024 * 1024;
const MIN_WEIGHT_UPLOAD_BATCH_BYTES = 32 * 1024 * 1024;
const MAX_WEIGHT_UPLOAD_BATCH_BYTES = MAX_MEMORY_WEIGHT_UPLOAD_BATCH_BYTES;
const LARGE_TENSOR_SOLO_BATCH_BYTES = 384 * 1024 * 1024;
const HF_CACHE_NAME = "mlx-browser-huggingface-models-v1";
const HF_CACHE_ORIGIN = "https://mlx-node-browser.local";

type ModelSource =
  | {
      kind: "remote";
      baseUrl: string;
      label: string;
      fileCache: Map<string, Uint8Array>;
    }
  | { kind: "local"; files: Map<string, LocalModelFile>; label: string }
  | {
      kind: "huggingface";
      repoId: string;
      revision: string;
      resolvedRevision?: string;
      label: string;
      blobCache: Map<string, Blob>;
      files?: string[];
    };

function formatModelLabel(source: ModelSource, config: any): string {
  const textConfig = config?.text_config ?? config;
  const rawModelType = String(
    textConfig?.model_type ?? config?.model_type ?? "model",
  );
  let family = rawModelType
    .replace(/_text$/i, "")
    .replace(/qwen3_5_moe/i, "Qwen3.5 MoE")
    .replace(/qwen3_5/i, "Qwen3.5")
    .replace(/qwen3_6/i, "Qwen3.6")
    .replace(/_/g, " ");
  family = family
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => (part.toLowerCase() === "qwen" ? "Qwen" : part))
    .join(" ");
  const layers = Number(textConfig?.num_hidden_layers ?? 0) || 0;
  const hidden = Number(textConfig?.hidden_size ?? 0) || 0;
  const experts = Number(textConfig?.num_experts ?? 0) || 0;
  const bits = Number(config?.quantization?.bits ?? 0) || 0;
  const details = [
    layers > 0 ? `${layers}L` : "",
    hidden > 0 ? `${hidden}h` : "",
    experts > 0 ? `${experts}e` : "",
    bits > 0 ? `Q${bits}` : "",
  ].filter(Boolean);
  const configLabel =
    details.length > 0 ? `${family} ${details.join(" ")}` : family;
  if (source.kind === "remote" && source.label && source.label !== "/model") {
    return source.label;
  }
  if (source.kind === "remote") return configLabel || source.label;
  if (source.label && source.label !== "local model") return source.label;
  return configLabel || source.label;
}

type GpuTensorDescriptor = {
  name: string;
  handle: number;
  dtypeCode: number;
  shape: number[];
  byteSize: number;
  packedBf16: boolean;
};

type UploadWeightsResult = {
  handles: number[];
  uploadedDtypes: string[];
  uploadedByteSizes: number[];
  packedBf16Flags: boolean[];
  debugPackedTotal?: number;
  debugUnpackedBf16Total?: number;
};

const REASONING_TOKEN_BUDGET: Record<
  Exclude<ReasoningEffort, "off">,
  number
> = {
  low: 32,
  medium: 128,
  high: 256,
};

function normalizeReasoningEffort(
  value: unknown,
  enableThinking?: boolean,
): ReasoningEffort {
  if (value === "off" || value === "none") return "off";
  if (value === "low" || value === "medium" || value === "high") return value;
  return enableThinking === true ? "high" : "off";
}

function buildChatConfig(data: {
  messages?: any[];
  config?: any;
  enableThinking?: boolean;
  reasoningEffort?: unknown;
}) {
  const baseConfig = {
    ...modelGenerationDefaults,
    ...(data.config ?? {}),
  };
  const requestedEffort = normalizeReasoningEffort(
    data.reasoningEffort,
    data.enableThinking,
  );
  const hasImages =
    Array.isArray(data.messages) && hasMessageImages(data.messages);
  const hasTools =
    Array.isArray(baseConfig.tools) && baseConfig.tools.length > 0;
  const effort =
    requestedEffort === "off" && hasImages && !hasTools
      ? "low"
      : requestedEffort;

  if (effort === "off") {
    return {
      ...baseConfig,
      reasoningEffort: "none",
      includeReasoning: false,
      reportPerformance: true,
    };
  }

  return {
    ...baseConfig,
    // Qwen3.5 maps `low` to no-thinking internally, so the browser uses the
    // thinking template with a low token cap to provide a true low effort.
    reasoningEffort: effort === "low" ? "medium" : effort,
    thinkingTokenBudget: REASONING_TOKEN_BUDGET[effort],
    includeReasoning: requestedEffort !== "off",
    reportPerformance: true,
  };
}

function isVisionTensorName(name: string): boolean {
  return (
    name.startsWith("vision_tower.") ||
    name.startsWith("visual.") ||
    name.startsWith("model.visual.")
  );
}

function hasMessageImages(messages: any[]): boolean {
  return messages.some(
    (message) => Array.isArray(message?.images) && message.images.length > 0,
  );
}

function rejectUnsupportedImages(messages: any[]): boolean {
  if (modelSupportsImages || !hasMessageImages(messages)) return false;
  post({
    type: "error",
    message: "Image input is unavailable for the loaded text-only model.",
  });
  return true;
}

function setBridgeOptimizations(
  fusionEnabled: boolean,
  passCachingEnabled: boolean,
  dispatchBatchEnabled: boolean,
) {
  if (!bridgeOptimizationControl) return;
  Atomics.store(bridgeOptimizationControl, 0, fusionEnabled ? 1 : 0);
  Atomics.store(bridgeOptimizationControl, 1, passCachingEnabled ? 1 : 0);
  Atomics.store(bridgeOptimizationControl, 2, dispatchBatchEnabled ? 1 : 0);
}

async function withBridgeModeForMessages<T>(
  messages: any[],
  run: () => Promise<T>,
): Promise<T> {
  const imageRequest = hasMessageImages(messages);
  if (!bridgeOptimizationControl) return run();

  const previousFusion = Atomics.load(bridgeOptimizationControl, 0) !== 0;
  const previousPassCaching = Atomics.load(bridgeOptimizationControl, 1) !== 0;
  const previousDispatchBatch =
    Atomics.load(bridgeOptimizationControl, 2) !== 0;
  if (imageRequest) {
    // VLM prefill still needs the conservative bridge path because it has
    // cross-worker pass ownership and copy/restart patterns that are not yet
    // safe with the text decode fusion stack.
    setBridgeOptimizations(false, false, false);
  } else {
    setBridgeOptimizations(
      configuredFusionEnabled,
      true,
      configuredDispatchBatchEnabled,
    );
  }

  try {
    return await run();
  } finally {
    setBridgeOptimizations(
      previousFusion,
      previousPassCaching,
      previousDispatchBatch,
    );
  }
}

type LinearAttentionMergePlan = {
  left: TensorInfo;
  right: TensorInfo;
  tensor: UploadTensorInfo;
};

function planMergedLinearAttentionTensors(tensors: TensorInfo[]): {
  tensors: TensorInfo[];
  mergePlans: LinearAttentionMergePlan[];
  mergedCount: number;
} {
  // Keep browser model assembly aligned with native: Rust sanitize_weights()
  // owns linear-attention projection merging for dense and quantized tensors.
  // A JS raw-byte concat here can diverge from MLX concatenate semantics when
  // tensor storage/layout rules change.
  return { tensors, mergePlans: [], mergedCount: 0 };
}

function uploadWeightsToGpu(
  gpuWorker: Worker,
  weightsBuffer: ArrayBuffer | SharedArrayBuffer,
  dataOffset: number,
  tensors: UploadTensorInfo[],
  mergedBuffer: ArrayBuffer | SharedArrayBuffer | undefined,
  packBf16: boolean,
): Promise<UploadWeightsResult> {
  return new Promise((resolve, reject) => {
    const onMessage = (ev: MessageEvent) => {
      const msg = ev.data;
      if (msg?.type === "weights_uploaded") {
        cleanup();
        resolve(msg);
      } else if (
        msg?.type === "error" ||
        msg?.type === "rpc-error" ||
        msg?.type === "gpu-error"
      ) {
        cleanup();
        reject(new Error(msg.message || "GPU weight upload failed"));
      }
    };
    const onError = (ev: ErrorEvent) => {
      cleanup();
      reject(new Error(ev.message || "GPU worker error during weight upload"));
    };
    const cleanup = () => {
      gpuWorker.removeEventListener("message", onMessage);
      gpuWorker.removeEventListener("error", onError);
    };
    gpuWorker.addEventListener("message", onMessage);
    gpuWorker.addEventListener("error", onError);
    const transfer: Transferable[] = [];
    if (weightsBuffer instanceof ArrayBuffer) transfer.push(weightsBuffer);
    if (mergedBuffer instanceof ArrayBuffer) transfer.push(mergedBuffer);
    gpuWorker.postMessage(
      {
        type: "upload_weights",
        weightsBuffer,
        mergedBuffer,
        dataOffset,
        tensors,
        packBf16,
      },
      transfer,
    );
  });
}

type UploadItem =
  | { kind: "tensor"; tensor: TensorInfo }
  | { kind: "merge"; plan: LinearAttentionMergePlan };

function uploadItemByteSize(item: UploadItem): number {
  return item.kind === "tensor"
    ? item.tensor.byteSize
    : item.plan.tensor.byteSize;
}

function defaultWeightUploadBatchBytes(): number {
  const deviceMemory = (globalThis.navigator as
    | (Navigator & { deviceMemory?: unknown })
    | undefined)?.deviceMemory;
  if (typeof deviceMemory !== "number" || !Number.isFinite(deviceMemory)) {
    return DEFAULT_WEIGHT_UPLOAD_BATCH_BYTES;
  }
  if (deviceMemory >= 64) return MAX_MEMORY_WEIGHT_UPLOAD_BATCH_BYTES;
  if (deviceMemory >= 48) return HUGE_WORKSTATION_WEIGHT_UPLOAD_BATCH_BYTES;
  if (deviceMemory >= 32) return LARGE_WORKSTATION_WEIGHT_UPLOAD_BATCH_BYTES;
  if (deviceMemory >= 24) return WORKSTATION_MEMORY_WEIGHT_UPLOAD_BATCH_BYTES;
  if (deviceMemory >= 16) return VERY_HIGH_MEMORY_WEIGHT_UPLOAD_BATCH_BYTES;
  if (deviceMemory >= 8) return HIGH_MEMORY_WEIGHT_UPLOAD_BATCH_BYTES;
  if (deviceMemory >= 4) return DEFAULT_WEIGHT_UPLOAD_BATCH_BYTES;
  if (deviceMemory >= 2) return LOW_MEMORY_WEIGHT_UPLOAD_BATCH_BYTES;
  return VERY_LOW_MEMORY_WEIGHT_UPLOAD_BATCH_BYTES;
}

function normalizeWeightUploadBatchBytes(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return defaultWeightUploadBatchBytes();
  }
  return Math.min(
    MAX_WEIGHT_UPLOAD_BATCH_BYTES,
    Math.max(MIN_WEIGHT_UPLOAD_BATCH_BYTES, Math.floor(value)),
  );
}

function describeUploadItem(item: UploadItem): UploadTensorInfo {
  if (item.kind === "tensor") {
    return { ...item.tensor };
  }
  return { ...item.plan.tensor };
}

async function copyTensorBytesInto(
  source: ModelSource,
  weightFile: string,
  dataOffset: number,
  tensor: TensorInfo,
  dst: Uint8Array,
  dstOffset: number,
) {
  const bytes = await readSourceSlice(
    source,
    weightFile,
    dataOffset + tensor.byteOffset,
    dataOffset + tensor.byteOffset + tensor.byteSize,
  );
  dst.set(bytes, dstOffset);
}

async function uploadPreparedWeightItems(
  source: ModelSource,
  weightFile: string,
  dataOffset: number,
  items: UploadItem[],
  gpuWorker: Worker,
  uploadPackedBf16: boolean,
  weightUploadBatchBytes: number,
  onProgress: (
    uploaded: UploadWeightsResult,
    tensors: UploadTensorInfo[],
  ) => void,
): Promise<{ debugPackedTotal: number; debugUnpackedBf16Total: number }> {
  let debugPackedTotal = 0;
  let debugUnpackedBf16Total = 0;
  let uploadedItemCount = 0;
  let activeBatchBytes = weightUploadBatchBytes;

  for (let startIndex = 0; startIndex < items.length; ) {
    const batchStartIndex = startIndex;
    const batchItems: UploadItem[] = [];
    let batchBytes = 0;

    while (startIndex < items.length) {
      const item = items[startIndex]!;
      const itemBytes = uploadItemByteSize(item);
      if (
        batchItems.length > 0 &&
        batchBytes + itemBytes > activeBatchBytes
      ) {
        break;
      }
      batchItems.push(item);
      batchBytes += itemBytes;
      startIndex++;
      if (batchItems.length === 1 && itemBytes >= LARGE_TENSOR_SOLO_BATCH_BYTES)
        break;
      if (batchBytes >= activeBatchBytes) break;
    }

    let batchBuffer: ArrayBuffer;
    try {
      batchBuffer = new ArrayBuffer(batchBytes);
    } catch (error) {
      if (
        batchItems.length > 1 &&
        activeBatchBytes > MIN_WEIGHT_UPLOAD_BATCH_BYTES
      ) {
        activeBatchBytes = Math.max(
          MIN_WEIGHT_UPLOAD_BATCH_BYTES,
          Math.floor(activeBatchBytes / 2),
        );
        startIndex = batchStartIndex;
        post({
          type: "progress",
          step: "init_model",
          message:
            `Upload batch allocation failed; retrying ${weightFile} with ` +
            `${Math.round(activeBatchBytes / 1024 / 1024)} MB batches...`,
        });
        continue;
      }
      throw error;
    }
    const batchView = new Uint8Array(batchBuffer);
    const batchTensors: UploadTensorInfo[] = [];
    let cursor = 0;

    for (const item of batchItems) {
      const uploadTensor = describeUploadItem(item);
      uploadTensor.byteOffset = cursor;
      uploadTensor.fromMerged = false;

      if (item.kind === "tensor") {
        await copyTensorBytesInto(
          source,
          weightFile,
          dataOffset,
          item.tensor,
          batchView,
          cursor,
        );
      } else {
        const { left, right } = item.plan;
        await copyTensorBytesInto(
          source,
          weightFile,
          dataOffset,
          left,
          batchView,
          cursor,
        );
        await copyTensorBytesInto(
          source,
          weightFile,
          dataOffset,
          right,
          batchView,
          cursor + left.byteSize,
        );
      }

      batchTensors.push(uploadTensor);
      cursor += uploadTensor.byteSize;
    }

    post({
      type: "progress",
      step: "init_model",
      message:
        `Uploading ${weightFile}: ${uploadedItemCount + batchItems.length}/${items.length}` +
        ` tensors (${(batchBytes / 1024 / 1024).toFixed(0)} MB batch)...`,
    });

    const uploaded = await uploadWeightsToGpu(
      gpuWorker,
      batchBuffer,
      0,
      batchTensors,
      undefined,
      uploadPackedBf16,
    );
    if (uploaded.handles.length !== batchTensors.length) {
      throw new Error(
        `GPU upload returned ${uploaded.handles.length} handles for ${batchTensors.length} tensors in ${weightFile}`,
      );
    }

    uploadedItemCount += batchItems.length;
    debugPackedTotal += uploaded.debugPackedTotal ?? 0;
    debugUnpackedBf16Total += uploaded.debugUnpackedBf16Total ?? 0;
    onProgress(uploaded, batchTensors);
  }

  return { debugPackedTotal, debugUnpackedBf16Total };
}

function normalizeModelPath(path: string): string {
  return path.replace(/\\/g, "/").replace(/^\/+/, "");
}

function normalizeHfRepoId(value: string): string {
  return value
    .trim()
    .replace(/^https:\/\/huggingface\.co\//i, "")
    .replace(/^hf:\/\//i, "")
    .replace(/^models\//i, "")
    .replace(/\/(?:tree|resolve)\/.*$/i, "")
    .replace(/^\/+|\/+$/g, "");
}

function encodePathParts(value: string): string {
  return value.split("/").map(encodeURIComponent).join("/");
}

function hfResolveUrl(
  source: Extract<ModelSource, { kind: "huggingface" }>,
  path: string,
): string {
  const repoId = encodePathParts(source.repoId);
  const revision = encodeURIComponent(
    source.resolvedRevision ?? source.revision,
  );
  return `https://huggingface.co/${repoId}/resolve/${revision}/${encodePathParts(path)}`;
}

function hfTreeUrl(
  source: Extract<ModelSource, { kind: "huggingface" }>,
): string {
  const repoId = encodePathParts(source.repoId);
  const revision = encodeURIComponent(
    source.resolvedRevision ?? source.revision,
  );
  return `https://huggingface.co/api/models/${repoId}/tree/${revision}?recursive=1`;
}

function isPinnedHfRevision(revision: string): boolean {
  return /^[0-9a-f]{40}$/i.test(revision);
}

function hfCacheRequest(
  source: Extract<ModelSource, { kind: "huggingface" }>,
  path: string,
): Request {
  const revision = source.resolvedRevision ?? source.revision;
  const key =
    `${HF_CACHE_ORIGIN}/huggingface/` +
    `${encodeURIComponent(source.repoId)}/` +
    `${encodeURIComponent(revision)}/` +
    encodePathParts(path);
  return new Request(key);
}

async function openHfCache(): Promise<Cache | null> {
  if (typeof caches === "undefined") return null;
  try {
    return await caches.open(HF_CACHE_NAME);
  } catch (e) {
    post({
      type: "log",
      message: `[HF] Cache Storage unavailable: ${String(e)}`,
    });
    return null;
  }
}

async function fetchHfNetworkResponse(
  source: Extract<ModelSource, { kind: "huggingface" }>,
  path: string,
  optional = false,
): Promise<Response | undefined> {
  const normalizedPath = normalizeModelPath(path);
  post({
    type: "progress",
    step: "download",
    message: `Downloading ${normalizedPath} from Hugging Face...`,
  });
  const resp = await fetch(hfResolveUrl(source, normalizedPath), {
    headers: { Accept: "application/octet-stream" },
  });
  if (!resp.ok) {
    if (optional && resp.status === 404) return undefined;
    throw new Error(
      `Failed to fetch ${normalizedPath} from ${source.repoId}@${source.revision}: HTTP ${resp.status} ${resp.statusText}`,
    );
  }

  const commit = resp.headers.get("x-repo-commit");
  if (commit && source.resolvedRevision !== commit) {
    source.resolvedRevision = commit;
    post({
      type: "log",
      message: `[HF] ${source.repoId}@${source.revision} resolved to ${commit.slice(0, 12)}`,
    });
  }
  return resp;
}

async function fetchHfFileResponse(
  source: Extract<ModelSource, { kind: "huggingface" }>,
  path: string,
  optional = false,
): Promise<Response | undefined> {
  const normalizedPath = normalizeModelPath(path);
  const cache = await openHfCache();
  const shouldResolveRevision =
    source.resolvedRevision == null &&
    normalizedPath === "config.json" &&
    !isPinnedHfRevision(source.revision);

  if (cache && !shouldResolveRevision) {
    const cached = await cache.match(hfCacheRequest(source, normalizedPath));
    if (cached) return cached;
  }

  const resp = await fetchHfNetworkResponse(source, normalizedPath, optional);
  if (!resp) return undefined;

  if (cache) {
    const request = hfCacheRequest(source, normalizedPath);
    try {
      await cache.put(request, resp.clone());
      const cached = await cache.match(request);
      if (cached) return cached;
    } catch (e) {
      post({
        type: "log",
        message: `[HF] Could not persist ${normalizedPath}; continuing without browser cache: ${String(e)}`,
      });
    }
  }

  return resp;
}

async function getHfOpfsRevisionDirectory(
  source: Extract<ModelSource, { kind: "huggingface" }>,
  create: boolean,
): Promise<any | null> {
  const storage = (navigator as any).storage;
  if (!storage?.getDirectory) return null;

  try {
    let dir = await storage.getDirectory();
    dir = await dir.getDirectoryHandle(HF_CACHE_NAME, { create });
    dir = await dir.getDirectoryHandle(encodeURIComponent(source.repoId), {
      create,
    });
    dir = await dir.getDirectoryHandle(
      encodeURIComponent(source.resolvedRevision ?? source.revision),
      { create },
    );
    return dir;
  } catch (e) {
    if (!create) return null;
    post({
      type: "log",
      message: `[HF] OPFS cache unavailable: ${String(e)}`,
    });
    return null;
  }
}

async function getHfOpfsFileHandle(
  source: Extract<ModelSource, { kind: "huggingface" }>,
  path: string,
  create: boolean,
): Promise<any | null> {
  let dir = await getHfOpfsRevisionDirectory(source, create);
  if (!dir) return null;

  const parts = normalizeModelPath(path).split("/").filter(Boolean);
  const fileName = parts.pop();
  if (!fileName) return null;

  try {
    for (const part of parts) {
      dir = await dir.getDirectoryHandle(encodeURIComponent(part), { create });
    }
    return await dir.getFileHandle(encodeURIComponent(fileName), { create });
  } catch (e) {
    if (!create) return null;
    throw e;
  }
}

async function readHfOpfsFileBlob(
  source: Extract<ModelSource, { kind: "huggingface" }>,
  path: string,
): Promise<Blob | undefined> {
  const handle = await getHfOpfsFileHandle(source, path, false);
  if (!handle) return undefined;
  const file = await handle.getFile();
  return file.size > 0 ? file : undefined;
}

function maybePostHfDownloadProgress(
  path: string,
  loaded: number,
  total: number,
  lastPct: number,
): number {
  if (total > 0) {
    const pct = Math.min(100, Math.floor((loaded / total) * 100));
    if (pct === 100 || pct >= lastPct + 5) {
      post({
        type: "progress",
        step: "download",
        pct,
        message: `Caching ${path} from Hugging Face... ${pct}%`,
      });
      return pct;
    }
    return lastPct;
  }

  const loadedMb = Math.floor(loaded / 1024 / 1024);
  if (loadedMb >= lastPct + 128) {
    post({
      type: "progress",
      step: "download",
      message: `Caching ${path} from Hugging Face... ${loadedMb} MB`,
    });
    return loadedMb;
  }
  return lastPct;
}

async function downloadHfFileToOpfs(
  source: Extract<ModelSource, { kind: "huggingface" }>,
  path: string,
): Promise<Blob | undefined> {
  const normalizedPath = normalizeModelPath(path);
  if (!(navigator as any).storage?.getDirectory) return undefined;

  const resp = await fetchHfNetworkResponse(source, normalizedPath);
  if (!resp) return undefined;

  const handle = await getHfOpfsFileHandle(source, normalizedPath, true);
  if (!handle) return undefined;

  const writable = await handle.createWritable();
  try {
    const total = Number(resp.headers.get("content-length") ?? "0") || 0;
    let loaded = 0;
    let lastProgress = total > 0 ? -5 : -128;

    if (resp.body) {
      const reader = resp.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value) continue;
        await writable.write(value);
        loaded += value.byteLength;
        lastProgress = maybePostHfDownloadProgress(
          normalizedPath,
          loaded,
          total,
          lastProgress,
        );
      }
    } else {
      const blob = await resp.blob();
      await writable.write(blob);
      loaded = blob.size;
      maybePostHfDownloadProgress(normalizedPath, loaded, loaded, -5);
    }

    await writable.close();
    return await handle.getFile();
  } catch (e) {
    try {
      await writable.abort?.();
    } catch {
      // Ignore abort errors; the original download/write error is more useful.
    }
    throw e;
  }
}

async function readHfFileBlob(
  source: Extract<ModelSource, { kind: "huggingface" }>,
  path: string,
): Promise<Blob> {
  const normalizedPath = normalizeModelPath(path);
  const cached = source.blobCache.get(normalizedPath);
  if (cached) return cached;

  const opfsBlob = await readHfOpfsFileBlob(source, normalizedPath);
  if (opfsBlob) {
    source.blobCache.set(normalizedPath, opfsBlob);
    return opfsBlob;
  }

  const downloadedBlob = await downloadHfFileToOpfs(source, normalizedPath);
  if (downloadedBlob) {
    source.blobCache.set(normalizedPath, downloadedBlob);
    return downloadedBlob;
  }

  const resp = await fetchHfFileResponse(source, normalizedPath);
  if (!resp) throw new Error(`Hugging Face model is missing ${normalizedPath}`);
  const blob = await resp.blob();
  source.blobCache.set(normalizedPath, blob);
  return blob;
}

async function listHfRepoFiles(
  source: Extract<ModelSource, { kind: "huggingface" }>,
): Promise<string[]> {
  if (source.files) return source.files;

  post({
    type: "progress",
    step: "download",
    message: `Listing files from ${source.label}...`,
  });
  const resp = await fetch(hfTreeUrl(source), {
    headers: { Accept: "application/json" },
  });
  if (!resp.ok) {
    throw new Error(
      `Failed to list ${source.label}: HTTP ${resp.status} ${resp.statusText}`,
    );
  }

  const tree = (await resp.json()) as Array<{ path?: string; type?: string }>;
  source.files = tree
    .filter(
      (entry) => entry.path && (entry.type == null || entry.type === "file"),
    )
    .map((entry) => normalizeModelPath(entry.path!));
  return source.files;
}

function createModelSource(
  modelUrl: string,
  modelFiles?: LocalModelFile[],
  hfModel?: HuggingFaceModelConfig,
  modelLabel?: string,
): ModelSource {
  if (modelFiles && modelFiles.length > 0) {
    const rawPaths = modelFiles.map((file) =>
      normalizeModelPath(file.webkitRelativePath || file.name),
    );
    const firstRoot = rawPaths[0]?.split("/")[0] ?? "";
    const stripRoot =
      firstRoot.length > 0 &&
      rawPaths.every(
        (path) => path === firstRoot || path.startsWith(`${firstRoot}/`),
      );

    const files = new Map<string, LocalModelFile>();
    for (let i = 0; i < modelFiles.length; i++) {
      const file = modelFiles[i]!;
      const rawPath = rawPaths[i] || file.name;
      const stripped = stripRoot
        ? rawPath.split("/").slice(1).join("/") || file.name
        : rawPath;
      files.set(stripped, file);
      files.set(rawPath, file);
      if (!files.has(file.name)) files.set(file.name, file);
    }

    return {
      kind: "local",
      files,
      label: stripRoot ? firstRoot : "local model",
    };
  }

  const repoId = hfModel?.repoId ? normalizeHfRepoId(hfModel.repoId) : "";
  if (repoId) {
    const revision = hfModel?.revision?.trim() || "main";
    return {
      kind: "huggingface",
      repoId,
      revision,
      label: `hf:${repoId}@${revision}`,
      blobCache: new Map(),
    };
  }

  return {
    kind: "remote",
    baseUrl: modelUrl.replace(/\/+$/, ""),
    label: modelLabel?.trim() || modelUrl,
    fileCache: new Map(),
  };
}

async function readSourceText(
  source: ModelSource,
  path: string,
  optional = false,
): Promise<string | undefined> {
  const normalizedPath = normalizeModelPath(path);
  if (source.kind === "local") {
    const file = source.files.get(normalizedPath);
    if (!file) {
      if (optional) return undefined;
      throw new Error(`Local model is missing ${normalizedPath}`);
    }
    return file.text();
  }

  if (source.kind === "huggingface") {
    const resp = await fetchHfFileResponse(source, normalizedPath, optional);
    if (!resp) return undefined;
    const text = await resp.text();
    const contentType = resp.headers.get("content-type") ?? "";
    if (
      optional &&
      contentType.includes("text/html") &&
      text.trimStart().startsWith("<!doctype")
    ) {
      return undefined;
    }
    return text;
  }

  const resp = await fetch(`${source.baseUrl}/${normalizedPath}`);
  if (!resp.ok) {
    if (optional && resp.status === 404) return undefined;
    throw new Error(
      `Failed to fetch ${normalizedPath}: HTTP ${resp.status} ${resp.statusText}`,
    );
  }
  const text = await resp.text();
  const contentType = resp.headers.get("content-type") ?? "";
  if (
    contentType.includes("text/html") &&
    text.trimStart().startsWith("<!doctype")
  ) {
    if (optional) return undefined;
    throw new Error(
      `Model source ${source.baseUrl} served the app HTML for ${normalizedPath}; the hosted /model files are not deployed. Choose a local model directory or deploy the model assets separately.`,
    );
  }
  return text;
}

async function readSourceSlice(
  source: ModelSource,
  path: string,
  start: number,
  end: number,
): Promise<Uint8Array> {
  const normalizedPath = normalizeModelPath(path);
  if (end < start) {
    throw new Error(
      `Invalid byte range for ${normalizedPath}: ${start}-${end}`,
    );
  }
  if (end === start) return new Uint8Array(0);

  if (source.kind === "local") {
    const file = source.files.get(normalizedPath);
    if (!file) throw new Error(`Local model is missing ${normalizedPath}`);
    if (end > file.size) {
      throw new Error(
        `Local range for ${normalizedPath} exceeds file size: ${end}/${file.size}`,
      );
    }
    return new Uint8Array(await file.slice(start, end).arrayBuffer());
  }

  if (source.kind === "huggingface") {
    const blob = await readHfFileBlob(source, normalizedPath);
    if (end > blob.size) {
      throw new Error(
        `Hugging Face range for ${normalizedPath} exceeds file size: ${end}/${blob.size}`,
      );
    }
    return new Uint8Array(await blob.slice(start, end).arrayBuffer());
  }

  const cached = source.fileCache.get(normalizedPath);
  if (cached) {
    if (end > cached.byteLength) {
      throw new Error(
        `Cached range for ${normalizedPath} exceeds file size: ${end}/${cached.byteLength}`,
      );
    }
    return cached.subarray(start, end);
  }

  const resp = await fetch(`${source.baseUrl}/${normalizedPath}`, {
    headers: { Range: `bytes=${start}-${end - 1}` },
  });
  if (!resp.ok) {
    throw new Error(
      `Failed to fetch ${normalizedPath}: HTTP ${resp.status} ${resp.statusText}`,
    );
  }
  const bytes = new Uint8Array(await resp.arrayBuffer());
  if (resp.status === 206) {
    if (bytes.byteLength !== end - start) {
      throw new Error(
        `Range fetch for ${normalizedPath} returned ${bytes.byteLength} bytes, expected ${end - start}`,
      );
    }
    return bytes;
  }

  if (resp.status === 200 && start === 0 && bytes.byteLength >= end) {
    source.fileCache.set(normalizedPath, bytes);
    return bytes.subarray(start, end);
  }

  throw new Error(
    `Server did not honor byte range for ${normalizedPath}; use a local model directory or a server with Range support`,
  );
}

async function readSafeTensorsHeader(
  source: ModelSource,
  path: string,
): Promise<{ tensors: TensorInfo[]; dataOffset: number }> {
  const prefix = await readSourceSlice(source, path, 0, 8);
  const headerLen = Number(
    new DataView(
      prefix.buffer,
      prefix.byteOffset,
      prefix.byteLength,
    ).getBigUint64(0, true),
  );
  if (!Number.isSafeInteger(headerLen) || headerLen <= 0) {
    throw new Error(
      `Invalid safetensors header length in ${path}: ${headerLen}`,
    );
  }
  const dataOffset = 8 + headerLen;
  const headerBytes = await readSourceSlice(source, path, 8, dataOffset);
  return parseSafeTensorsHeaderBytes(headerBytes, dataOffset);
}

async function discoverWeightFiles(source: ModelSource): Promise<string[]> {
  const indexJson = await readSourceText(
    source,
    "model.safetensors.index.json",
    true,
  );
  if (indexJson) {
    try {
      const parsed = JSON.parse(indexJson) as {
        weight_map?: Record<string, string>;
      };
      const files = [...new Set(Object.values(parsed.weight_map ?? {}))];
      if (files.length > 0) return files;
    } catch {
      post({
        type: "log",
        message: "[MODEL] Ignoring invalid model.safetensors.index.json",
      });
    }
  }

  if (source.kind === "local") {
    const files = [...source.files.keys()]
      .filter((path) => path.endsWith(".safetensors") && !path.includes("/"))
      .filter((path, index, array) => array.indexOf(path) === index)
      .sort((a, b) => {
        if (a === "model.safetensors") return -1;
        if (b === "model.safetensors") return 1;
        return a.localeCompare(b, undefined, { numeric: true });
      });
    if (files.length > 0) return files;
  }

  if (source.kind === "huggingface") {
    try {
      const files = (await listHfRepoFiles(source))
        .filter((path) => path.endsWith(".safetensors") && !path.includes("/"))
        .sort((a, b) => {
          if (a === "model.safetensors") return -1;
          if (b === "model.safetensors") return 1;
          return a.localeCompare(b, undefined, { numeric: true });
        });
      if (files.length > 0) return files;
    } catch (e) {
      post({
        type: "log",
        message: `[HF] Could not list repository files; falling back to model.safetensors: ${String(e)}`,
      });
    }
  }

  return ["model.safetensors"];
}

// Phase 0 (?profile=1) state. When enabled, handleChat* wraps each
// generation with a reset + readback of:
//   - wasm-worker rpcCall histogram (bridge.getBridgeStats)
//   - gpu-worker rpcCount histogram (bridge.fetchGpuWorkerStats)
//   - C++ WebGPU dispatch counters (mlxExports.wgpuGetDispatchStats)
// Then posts a {type:'profile', stats:...} message to the main thread.
let profileEnabled = false;
let bridgeRef: BridgeStub | null = null;
// Per-worker DISPATCH_BATCH index counter. Child workers (created by emnapi's
// asyncWorkPoolSize) receive batch buffer IDs 1, 2, 3, ...
let nextWorkerBatchIndex = 1;

function post(msg: any) {
  (self as any).postMessage(msg);
}

function formatUnknownError(e: unknown): { message: string; stack?: string } {
  if (e instanceof Error) {
    return { message: e.message || String(e), stack: e.stack };
  }

  if (typeof ErrorEvent !== "undefined" && e instanceof ErrorEvent) {
    const inner = formatUnknownError(e.error);
    const location =
      e.filename || e.lineno || e.colno
        ? `${e.filename}:${e.lineno}:${e.colno}`
        : undefined;
    return {
      message: e.message || inner.message || "Worker error",
      stack: inner.stack || location,
    };
  }

  return { message: typeof e === "string" ? e : String(e) };
}

async function handleInit(data: {
  wasmUrl: string;
  modelUrl: string;
  modelLabel?: string;
  modelFiles?: LocalModelFile[];
  hfModel?: HuggingFaceModelConfig;
  packBf16?: boolean;
  sdpaFallback?: boolean;
  profile?: boolean;
  dispatchBatch?: boolean;
  compileMlp?: boolean;
  compileGdnPre?: boolean;
  compileGdnPost?: boolean;
  compileGdnG?: boolean;
  enableVlm?: boolean;
  fuseDispatch?: boolean;
  weightUploadBatchBytes?: number;
}) {
  const packBf16 = data.packBf16 === true;
  const sdpaFallback = data.sdpaFallback === true;
  profileEnabled = data.profile === true;
  const dispatchBatch = data.dispatchBatch === true;
  // Phase 6b: SwiGLU MLP compile fast path. Default ON; disable via
  // ?compile_mlp=0 on the demo URL. Plumbed through handleInit so the
  // backend flag flips BEFORE any model forward pass runs.
  const compileMlp = data.compileMlp !== false;
  // Phase 6c: GDN pre-recurrence compile fast path. Default ON; disable
  // via ?compile_gdn_pre=0 on the demo URL. Same plumbing pattern as the
  // Phase 6b MLP flag — process-wide atomic flipped before any model
  // forward pass runs.
  const compileGdnPre = data.compileGdnPre !== false;
  // Phase 6d: GDN post-recurrence compile fast path. Default ON; disable
  // via ?compile_gdn_post=0 on the demo URL. Same pattern as 6b/6c —
  // process-wide atomic flipped before any forward pass runs.
  const compileGdnPost = data.compileGdnPost !== false;
  // Phase 6e: GDN decay-gate (compute_g) compile fast path. Default ON;
  // disable via ?compile_gdn_g=0 on the demo URL. Same pattern as
  // 6b/6c/6d — process-wide atomic flipped before any forward pass runs.
  const compileGdnG = data.compileGdnG !== false;
  const enableVlm = data.enableVlm !== false;
  const fuseDispatch = data.fuseDispatch !== false;
  const weightUploadBatchBytes = normalizeWeightUploadBatchBytes(
    data.weightUploadBatchBytes,
  );
  configuredFusionEnabled = fuseDispatch;
  configuredDispatchBatchEnabled = dispatchBatch;
  try {
    // 1. Spawn gpu-worker (owns GPUDevice, event loop free for GPU callbacks)
    post({ type: "progress", step: "gpu", message: "Initializing WebGPU..." });
    const cmdBuffer = new SharedArrayBuffer(CMD_OFFSET.TOTAL);
    const readbackBuffer = new SharedArrayBuffer(READBACK_BUFFER_SIZE);
    const poolStatsBuffer = profileEnabled
      ? new SharedArrayBuffer(POOL_STATS_SIZE_BYTES)
      : undefined;
    nextWorkerBatchIndex = 1;
    // Phase 2: dedicated dispatch-batch SABs — one per worker (main + up to
    // asyncWorkPoolSize child workers). Each worker gets its own SAB so
    // DISPATCH_BATCH can be safely enabled on child pthread workers without
    // races on a shared batch cursor. The gpu-worker receives all buffers in
    // init and looks up the right one by batchBufferId.
    const ASYNC_WORK_POOL_SIZE = 4;
    const NUM_BATCH_BUFFERS = ASYNC_WORK_POOL_SIZE + 1; // 0 = main
    const batchBuffers: SharedArrayBuffer[] = [];
    for (let i = 0; i < NUM_BATCH_BUFFERS; i++) {
      batchBuffers.push(new SharedArrayBuffer(DISPATCH_BATCH_BUFFER_SIZE));
    }
    const dispatchBatchBuffer = batchBuffers[0]!;
    // Task 3: shared buffer-metadata SAB. Every bridge stub spawned against
    // this wasmMemory (main + child pthread workers created by emnapi's
    // asyncWorkPoolSize) reads and writes the same (size, usage) table,
    // fixing the `unknownHandle=100%` pathology where handles created on
    // one stub were released from another and the release-side metadata
    // lookup came up empty. Sized for ~35 back-to-back Qwen3.5-0.8B
    // decodes; see webgpu-bridge-stub.ts for the capacity calculation.
    const bufferMetadataBuffer = new SharedArrayBuffer(
      BUFFER_METADATA_SIZE_BYTES,
    );
    // JS-F010: dedicated 1 KiB stats SAB for the gpu-worker's per-opcode
    // RPC histogram. Replaces the old scheme that striped the histogram
    // across the cmd SAB's CALLBACK_BASE / UNIFORM_DATA / reserved regions
    // (which was "safe by serialization" — one loosening of cmd-SAB
    // single-slot and a fused dispatch would silently lose its UNIFORM_DATA
    // bytes). Plumbed into both the gpu-worker init message and every
    // createBridgeStub call so main + child workers share the same view.
    const statsBuffer = profileEnabled
      ? new SharedArrayBuffer(STATS_BUFFER_SIZE)
      : undefined;
    const optimizationControlBuffer = new SharedArrayBuffer(12);
    bridgeOptimizationControl = new Int32Array(optimizationControlBuffer);
    setBridgeOptimizations(
      configuredFusionEnabled,
      false,
      configuredDispatchBatchEnabled,
    );
    // WASM module requires min 1002 pages (~66 MB). We use 4096 (~268 MB)
    // for headroom during WASM init (thread stacks, emnapi, etc.) and let
    // memory.grow expand as needed — keeps total well under the 2 GB JS
    // pointer limit (i32 > 2^31 wraps negative in TypedArray constructors).
    // Old value of 30000 pages (~1.97 GB) left almost no room for model data.
    const sharedMemory = new WebAssembly.Memory({
      initial: 4096,
      maximum: 65536,
      shared: true,
    });
    wasmMemory = sharedMemory;

    const gpuWorker = new Worker(workerAssetUrl(gpuWorkerUrl), {
      type: "module",
    });

    // Wait for gpu-worker to create GPUDevice and be ready
    const gpuReady = await new Promise<any>((resolve, reject) => {
      gpuWorker.onmessage = (e) => {
        if (e.data.type === "ready") resolve(e.data);
        else if (e.data.type === "error") reject(new Error(e.data.message));
      };
      gpuWorker.postMessage({
        type: "init",
        cmdBuffer,
        readbackBuffer,
        wasmMemory: sharedMemory, // Send Memory object, not .buffer — .buffer getter always returns current byteLength after grow()
        batchBuffers,
        statsBuffer,
        bufferMetadataBuffer,
      });
    });
    post({ type: "progress", step: "gpu", message: "WebGPU ready" });
    gpuWorker.addEventListener("message", (event: MessageEvent) => {
      const msg = event.data;
      if (msg?.type === "gpu-error" || msg?.type === "rpc-error") {
        const context = msg.context ? ` (${msg.context})` : "";
        const detail = `[GPU] ${msg.type}${context}: ${msg.message}`;
        post({ type: "log", message: detail });
        if (msg.stack) post({ type: "log", message: msg.stack });
        if (msg.type === "gpu-error") {
          post({ type: "error", message: detail });
        }
      }
    });

    // 2. Create bridge stub (RPC via Atomics.wait to gpu-worker)
    const bridge = createBridgeStub(
      cmdBuffer,
      sharedMemory,
      {
        instanceHandle: gpuReady.instanceHandle,
        adapterHandle: gpuReady.adapterHandle,
        deviceHandle: gpuReady.deviceHandle,
        queueHandle: gpuReady.queueHandle,
      },
      readbackBuffer,
      gpuReady.features,
      poolStatsBuffer,
      dispatchBatchBuffer,
      dispatchBatch,
      0, // batchBufferId = 0 for main worker
      bufferMetadataBuffer,
      statsBuffer,
      {
        fusionEnabled: configuredFusionEnabled,
        passCachingEnabled: true,
        optimizationControlBuffer,
      },
    );
    // Retain for ?profile=1 readback (resetBridgeStats / fetchGpuWorkerStats
    // need the same BridgeStub instance that owns the SAB cmdBuffer view).
    bridgeRef = bridge;

    // 3. Load WASM with bridge stub
    post({ type: "progress", step: "wasm", message: "Loading WASM module..." });
    // Forward lines starting with [MLX-KERNEL] to the main thread so we can
    // verify which matmul kernel variants fire. The WASM stderr lands on
    // the mlx-worker's own devtools target which isn't readable from the
    // main frame console.
    const wasi = new WASI({
      version: "preview1",
      print: function (...args: unknown[]) {
        const line = args.map(String).join(" ");
        if (line.includes("[MLX-KERNEL]")) {
          post({ type: "log", message: line });
        } else {
          console.log(...args);
        }
      },
      printErr: function (...args: unknown[]) {
        const line = args.map(String).join(" ");
        if (line.includes("[MLX-KERNEL]")) {
          post({ type: "log", message: line });
        } else {
          console.error(...args);
        }
      },
    } as any);
    const context = getDefaultContext();
    const wasmFile = await fetch(data.wasmUrl).then((r) => r.arrayBuffer());

    const cppExceptionTag = new WebAssembly.Tag({ parameters: ["i32"] });
    cppTag = cppExceptionTag;

    // Task 4's SabSink declares extern "C" fn __wasm_i32_atomic_wait /
    // __wasm_atomic_notify expecting LLVM compiler-rt builtins, but wasm-ld
    // emits them as host imports on wasm32-wasip1-threads. Provide JS stubs
    // that wrap Atomics.wait / Atomics.notify over the shared memory.
    // Cache the Int32Array view once (shared memory never detaches, so the
    // view stays valid for the lifetime of the worker). Avoids per-call
    // TypedArray construction on the hot atomic notify path.
    const sharedI32 = new Int32Array(sharedMemory.buffer);
    const cxxStubs = {
      __cpp_exception: cppExceptionTag,
      _ZN3mlx4core3gpu4initEv: () => {},
      __wasm_i32_atomic_wait: (
        ptr: number,
        expected: number,
        timeoutNs: bigint,
      ): number => {
        const index = ptr >>> 2;
        const timeoutMs =
          timeoutNs === -1n ? Infinity : Number(timeoutNs / 1_000_000n);
        const result = Atomics.wait(sharedI32, index, expected, timeoutMs);
        return result === "ok" ? 0 : result === "not-equal" ? 1 : 2;
      },
      __wasm_atomic_notify: (ptr: number, count: number): number => {
        return Atomics.notify(sharedI32, ptr >>> 2, count);
      },
    };

    const { napiModule } = await instantiateNapiModule(wasmFile, {
      context,
      asyncWorkPoolSize: ASYNC_WORK_POOL_SIZE,
      wasi,
      onCreateWorker() {
        // Child workers (model thread) share the gpu-worker's GPUDevice via
        // the same RPC bridge as the main WASM thread. The direct bridge
        // (own GPUDevice) deadlocks because wgpuBufferMapAsync registers a
        // JS Promise .then() callback, but raw_ptr()'s polling loop calls
        // poll_instance() which is a no-op on WASM — the event loop never
        // runs, the callback never fires, infinite loop.
        const w = new Worker(workerAssetUrl(webgpuWorkerUrl), {
          type: "module",
        });
        w.addEventListener("message", (event: MessageEvent) => {
          const msg = event.data;
          if (msg?.type === "__mlx_child_error") {
            post({
              type: "error",
              message: msg.message,
              stack: msg.stack,
            });
          }
        });
        w.addEventListener("error", (event: ErrorEvent) => {
          post({
            type: "error",
            message: `WASM child worker error: ${event.message}`,
            stack:
              event.error instanceof Error
                ? event.error.stack
                : `${event.filename}:${event.lineno}:${event.colno}`,
          });
        });
        // Child pthreads can outlive the compute-pass lifetime observed by the
        // main bridge stub. Their bridge view cannot safely cache/fuse pass
        // handles because another stub may end/release the pass first. Keep
        // the aggressive bridge path on the main worker only.
        const workerBatchIndex = nextWorkerBatchIndex++;
        // Child pthreads share the optimization-control SAB with the main
        // worker, so text decode keeps the fast bridge path while VLM image
        // prefill can temporarily force immediate bridge calls.
        const workerDispatchBatch =
          dispatchBatch && workerBatchIndex < batchBuffers.length;
        const workerBatchBuffer = workerDispatchBatch
          ? batchBuffers[workerBatchIndex]
          : undefined;
        w.postMessage({
          type: "__mlx_rpc_config",
          cmdBuffer,
          readbackBuffer,
          poolStatsBuffer,
          dispatchBatchBuffer: workerBatchBuffer,
          batchBufferId: workerBatchIndex,
          dispatchBatch: workerDispatchBatch,
          fusionEnabled: configuredFusionEnabled,
          passCachingEnabled: false,
          optimizationControlBuffer,
          bufferMetadataBuffer,
          statsBuffer,
          handles: {
            instanceHandle: gpuReady.instanceHandle,
            adapterHandle: gpuReady.adapterHandle,
            deviceHandle: gpuReady.deviceHandle,
            queueHandle: gpuReady.queueHandle,
          },
          features: gpuReady.features,
        });
        return w;
      },
      // getTable runs AFTER WebAssembly.instantiate but BEFORE _initialize.
      // The C++ Device constructor runs during _initialize and busy-waits for
      // adapter/device callbacks that need wasmTable to resolve function pointers.
      getTable(exports: WebAssembly.Exports) {
        const table = exports.__indirect_function_table as WebAssembly.Table;
        bridge.setInstance({ exports } as unknown as WebAssembly.Instance);
        return table;
      },
      overwriteImports(importObject: Record<string, Record<string, unknown>>) {
        importObject.env = {
          ...importObject.env,
          ...importObject.napi,
          ...importObject.emnapi,
          memory: sharedMemory,
          ...cxxStubs,
          ...bridge.imports,
        };
        return importObject;
      },
      beforeInit({ instance }) {
        wasmInst = instance;
        bridge.setInstance(instance);
        for (const name of Object.keys(instance.exports)) {
          if (name.startsWith("__napi_register__")) {
            (instance.exports[name] as Function)();
          }
        }
      },
    });

    mlxExports = napiModule.exports;
    // Flip the backend's packed-bf16 flag BEFORE any model/weight work runs.
    // wgpu_buffer() checks this flag on first upload to decide whether an
    // eligible bf16 weight gets flipped into StorageMode::PackedBf16. Toggling
    // after the fact is a no-op for already-uploaded buffers.
    if (typeof mlxExports.wgpuSetPackedBf16Enabled === "function") {
      mlxExports.wgpuSetPackedBf16Enabled(packBf16);
    }
    // Flip the SDPA fallback kill-switch. When true, the WebGPU backend's
    // fast/tile SDPA kernels are bypassed in favor of the decomposed
    // matmul→softmax→matmul path — used by the demo to A/B the fused
    // kernels against the baseline without a rebuild.
    if (typeof mlxExports.wgpuSetSdpaFallbackForced === "function") {
      mlxExports.wgpuSetSdpaFallbackForced(sdpaFallback);
    }
    // Phase 6b SwiGLU MLP compile fast path. The setter flips a process-wide
    // std::atomic<bool> in mlx_fused_ops.cpp so the next call to
    // mlx_swiglu_mlp_forward routes the element-wise tail through
    // mlx::core::compile. Default ON; disable via ?compile_mlp=0.
    if (typeof mlxExports.wgpuSetSwigluCompileEnabled === "function") {
      mlxExports.wgpuSetSwigluCompileEnabled(compileMlp);
    }
    // Phase 6c GDN pre-recurrence compile fast path. Same pattern as Phase
    // 6b — process-wide atomic flag, flipped before any model forward pass
    // runs. Default ON; disable via ?compile_gdn_pre=0.
    if (typeof mlxExports.wgpuSetGdnPreCompileEnabled === "function") {
      mlxExports.wgpuSetGdnPreCompileEnabled(compileGdnPre);
    }
    // Phase 6d GDN post-recurrence compile fast path. Same pattern as
    // Phase 6b/6c — process-wide atomic flag, flipped before any forward
    // pass runs. Default ON; disable via ?compile_gdn_post=0.
    if (typeof mlxExports.wgpuSetGdnPostCompileEnabled === "function") {
      mlxExports.wgpuSetGdnPostCompileEnabled(compileGdnPost);
    }
    // Phase 6e GDN decay-gate (compute_g) compile fast path. Same pattern
    // as 6b/6c/6d — process-wide atomic flag, flipped before any forward
    // pass runs. Default ON; disable via ?compile_gdn_g=0.
    if (typeof mlxExports.wgpuSetGdnGCompileEnabled === "function") {
      mlxExports.wgpuSetGdnGCompileEnabled(compileGdnG);
    }
    const flagSuffix =
      (packBf16 ? " pack_bf16=1" : "") +
      (sdpaFallback ? " sdpa_fallback=1" : "") +
      (compileMlp ? " compile_mlp=1" : "") +
      (compileGdnPre ? " compile_gdn_pre=1" : "") +
      (compileGdnPost ? " compile_gdn_post=1" : "") +
      (compileGdnG ? " compile_gdn_g=1" : "") +
      ` weight_upload_batch=${Math.round(weightUploadBatchBytes / 1024 / 1024)}MB` +
      (fuseDispatch ? "" : " fuse_dispatch=0");
    post({
      type: "progress",
      step: "wasm",
      message: `WASM loaded${flagSuffix ? " (" + flagSuffix.trim() + ")" : ""}`,
    });

    // 3. Load model files. The source can be either /model over HTTP or a
    // directory selected by the user in the browser. We process safetensors
    // one shard at a time so large local checkpoints never need one giant JS
    // ArrayBuffer.
    const modelSource = createModelSource(
      data.modelUrl,
      data.modelFiles,
      data.hfModel,
      data.modelLabel,
    );
    post({
      type: "progress",
      step: "model",
      message: `Loading config from ${modelSource.label}...`,
    });
    const configJson = await readSourceText(modelSource, "config.json");
    if (!configJson) throw new Error("config.json is empty");
    const config = JSON.parse(configJson) as {
      model_type?: string;
      text_config?: { model_type?: string };
    };
    const modelType = String(
      config.model_type ?? config.text_config?.model_type ?? "",
    );
    const modelLabel = formatModelLabel(modelSource, config);

    post({ type: "progress", step: "model", message: "Loading tokenizer..." });
    const tokenizerJson = await readSourceText(modelSource, "tokenizer.json");
    if (!tokenizerJson) throw new Error("tokenizer.json is empty");
    const tokenizerConfigJson = await readSourceText(
      modelSource,
      "tokenizer_config.json",
      true,
    );
    const generationConfigJson = await readSourceText(
      modelSource,
      "generation_config.json",
      true,
    );
    modelGenerationDefaults = generationConfigJson
      ? normalizeGenerationConfig(JSON.parse(generationConfigJson))
      : {};
    if (Object.keys(modelGenerationDefaults).length > 0) {
      post({
        type: "log",
        message: `[MODEL] generation defaults ${JSON.stringify(modelGenerationDefaults)}`,
      });
    }

    let processorConfigJson: string | undefined;
    for (const file of ["preprocessor_config.json", "processor_config.json"]) {
      processorConfigJson = await readSourceText(modelSource, file, true);
      if (processorConfigJson) break;
    }

    const weightFiles = await discoverWeightFiles(modelSource);
    post({
      type: "log",
      message: `[MODEL] ${modelSource.label}: model_type=${modelType || "unknown"}, safetensors=${weightFiles.length}`,
    });

    // 4. Parse safetensors and build the model. Prefer the GPU-buffer path:
    // JS uploads raw weights directly into WebGPU buffers, then Rust wraps the
    // handles with loadFromGpuBuffers. This is the only active load path that
    // can preserve PackedBf16 storage for decode weights.
    wasmMalloc = wasmInst!.exports.malloc as (size: number) => number;
    wasmFree = wasmInst!.exports.free as (ptr: number) => void;

    const DenseModel = mlxExports.Qwen35Model || mlxExports.Qwen3_5Model;
    const MoeModel = mlxExports.Qwen35MoeModel || mlxExports.Qwen3_5MoeModel;
    const Qwen35Model = modelType.includes("moe") ? MoeModel : DenseModel;
    if (!Qwen35Model) {
      throw new Error(
        `No WASM model class exported for model_type=${modelType}`,
      );
    }
    if (typeof Qwen35Model.loadFromGpuBuffers !== "function") {
      throw new Error(
        `${modelType || "Qwen3.5"} does not export loadFromGpuBuffers in this WASM build`,
      );
    }

    const uploadPackedBf16 = packBf16 && PACKED_GPU_WEIGHT_UPLOAD_ENABLED;
    if (packBf16 && !uploadPackedBf16) {
      post({
        type: "log",
        message:
          "[GPU] Packed bf16 GPU-buffer upload disabled for correctness; using f32-expanded weights",
      });
    }

    const gpuTensors: GpuTensorDescriptor[] = [];
    let totalTensorCount = 0;
    let totalUploadedCount = 0;
    let totalVisionTensorCount = 0;
    let totalMergedCount = 0;
    let debugPackedTotal = 0;
    let debugUnpackedBf16Total = 0;

    for (let shardIndex = 0; shardIndex < weightFiles.length; shardIndex++) {
      const weightFile = weightFiles[shardIndex]!;
      post({
        type: "progress",
        step: "download",
        message: `Loading weights ${shardIndex + 1}/${weightFiles.length}: ${weightFile}`,
      });
      post({
        type: "progress",
        step: "init_model",
        message: `Parsing ${weightFile}...`,
      });
      const { tensors, dataOffset } = await readSafeTensorsHeader(
        modelSource,
        weightFile,
      );
      totalTensorCount += tensors.length;
      const visionTensorCount = tensors.filter((tensor) =>
        isVisionTensorName(tensor.name),
      ).length;
      totalVisionTensorCount += visionTensorCount;
      const tensorsForModel = enableVlm
        ? tensors
        : tensors.filter((tensor) => !isVisionTensorName(tensor.name));

      const prepared = planMergedLinearAttentionTensors(tensorsForModel);
      totalMergedCount += prepared.mergedCount;
      const uploadItems: UploadItem[] = [
        ...prepared.tensors.map((tensor) => ({
          kind: "tensor" as const,
          tensor,
        })),
        ...prepared.mergePlans.map((plan) => ({
          kind: "merge" as const,
          plan,
        })),
      ];
      post({
        type: "progress",
        step: "init_model",
        message: `Uploading ${weightFile}: ${uploadItems.length} GPU tensors${prepared.mergedCount ? ` (${prepared.mergedCount} merged)` : ""}...`,
      });

      const uploadDebug = await uploadPreparedWeightItems(
        modelSource,
        weightFile,
        dataOffset,
        uploadItems,
        gpuWorker,
        uploadPackedBf16,
        weightUploadBatchBytes,
        (uploaded, batchTensors) => {
          for (let i = 0; i < batchTensors.length; i++) {
            const tensor = batchTensors[i]!;
            gpuTensors.push({
              name: tensor.name,
              handle: uploaded.handles[i]!,
              dtypeCode: dtypeToCode(
                uploaded.uploadedDtypes[i] ?? tensor.dtype,
              ),
              shape: tensor.shape,
              byteSize: uploaded.uploadedByteSizes[i] ?? tensor.byteSize,
              packedBf16: uploaded.packedBf16Flags[i] === true,
            });
          }
          totalUploadedCount += batchTensors.length;
        },
      );
      debugPackedTotal += uploadDebug.debugPackedTotal;
      debugUnpackedBf16Total += uploadDebug.debugUnpackedBf16Total;
    }

    modelSupportsImages = enableVlm && totalVisionTensorCount > 0;
    post({
      type: "log",
      message: modelSupportsImages
        ? `[MODEL] Vision input enabled (${totalVisionTensorCount} vision tensors)`
        : totalVisionTensorCount > 0
          ? `[MODEL] Vision tensors present (${totalVisionTensorCount}) but VLM input is disabled by ?disable_vlm=1`
          : "[MODEL] Vision input unavailable for this model",
    });
    post({
      type: "log",
      message:
        `[GPU] Uploaded ${totalUploadedCount}/${totalTensorCount} tensors` +
        ` merged=${totalMergedCount}` +
        ` packed=${debugPackedTotal}` +
        ` unpacked_bf16=${debugUnpackedBf16Total}`,
    });

    post({
      type: "progress",
      step: "init_model",
      message: "Building model from GPU buffers...",
    });
    const t1 = performance.now();
    model = await Qwen35Model.loadFromGpuBuffers(
      configJson,
      gpuTensors,
      tokenizerJson,
      tokenizerConfigJson ?? null,
      processorConfigJson ?? null,
    );
    post({
      type: "progress",
      step: "init_model",
      message: `Model ready (${(performance.now() - t1).toFixed(0)}ms)`,
    });

    // 6. Pipeline warmup — first inference warms GPU pipelines + shader compilation.
    // The VLM-capable Qwen3.5 checkpoint uses a vision-aware chat template; a
    // text-only warmup can wedge the worker before the UI becomes usable. Let
    // the first real image request warm the VLM path instead.
    if (modelSupportsImages) {
      post({
        type: "log",
        message: "[WARMUP] skipped for VLM-capable model",
      });
      post({ type: "progress", step: "warmup", message: "Warmup skipped" });
    } else if (typeof model.chat === "function") {
      post({ type: "progress", step: "warmup", message: "Warming up..." });
      const warmupResult = await model.chat([{ role: "user", content: "hi" }], {
        maxNewTokens: 2,
        temperature: 0,
        reasoningEffort: "none",
        includeReasoning: false,
        reuseCache: false,
      });
      post({
        type: "log",
        message: `[WARMUP] rawText=${warmupResult.rawText} text=${warmupResult.text} finish=${warmupResult.finishReason}`,
      });
      post({ type: "progress", step: "warmup", message: "Warmup complete" });
    } else {
      post({
        type: "log",
        message:
          "[WARMUP] skipped because chat() is not exported by this WASM build",
      });
      post({ type: "progress", step: "warmup", message: "Warmup skipped" });
    }

    // Ship the shared WASM memory to the main thread so it can mount the
    // SAB chat-stream reader directly over the WASM heap. A shared
    // WebAssembly.Memory is structured-cloneable and the clone refers to
    // the SAME underlying SharedArrayBuffer, so main-thread Int32Array
    // views over memory.buffer see every byte the Rust SabSink writes.
    // This is required to avoid delivering tokens in a single burst at
    // end-of-decode: WASM-side chatStreamSab synchronously blocks this
    // mlx-worker on Atomics.wait round-trips to gpu-worker, so a reader
    // whose Atomics.waitAsync promise resolves on this thread cannot
    // actually run its microtask until chatStreamSab returns. The main
    // thread is never blocked during decode, so the reader drains live.
    post({
      type: "ready",
      sharedMemory,
      supportsImages: modelSupportsImages,
      modelLabel,
    });
  } catch (e) {
    post({
      type: "error",
      message: String(e),
      stack: e instanceof Error ? e.stack : undefined,
    });
  }
}

const SAB_RING_SIZE = 262_144;

// Phase 0 (?profile=1) helpers. These are no-ops unless profileEnabled is
// true — each chat/chatStream handler calls resetProfileCounters() before
// generation and postProfileSnapshot() after. See the 'profile' message
// handler in demo/app.ts for the display side.
function resetProfileCounters(): void {
  if (!profileEnabled) return;
  try {
    bridgeRef?.resetBridgeStats();
  } catch (e) {
    console.warn("[mlx-worker] resetBridgeStats failed:", e);
  }
  try {
    if (mlxExports && typeof mlxExports.wgpuResetDispatchStats === "function") {
      mlxExports.wgpuResetDispatchStats();
    }
  } catch (e) {
    console.warn("[mlx-worker] wgpuResetDispatchStats failed:", e);
  }
  try {
    // Zero the gpu-worker histogram too (reset=1 on the RPC).
    bridgeRef?.fetchGpuWorkerStats(true);
  } catch (e) {
    console.warn("[mlx-worker] fetchGpuWorkerStats(reset=true) failed:", e);
  }
}

function postProfileSnapshot(numTokens: number): void {
  if (!profileEnabled) return;
  try {
    const bridgeStats =
      bridgeRef?.getBridgeStats() ??
      ({
        rpcCount: 0,
        byFn: {},
        poolHits: 0,
        poolMisses: 0,
        diagCreateAll: 0,
        diagCreateMappedCopyDst: 0,
        diagCreateMappedNoCopyDst: 0,
        diagReleaseAll: 0,
        diagReleaseUnknownHandle: 0,
        diagReleaseUnpoolable: 0,
        diagPoolEvictions: 0,
        poisonedRpcCount: 0,
      } as any);
    const gpuStats = bridgeRef?.fetchGpuWorkerStats(false) ?? {
      totalRpcs: 0,
      byFn: {},
      gpuPoolHits: 0,
      gpuPoolMisses: 0,
      bindGroupCacheHits: 0,
      bindGroupCacheMisses: 0,
      uniformHotHits: 0,
      uniformHotMisses: 0,
      spinHits: 0,
      spinMisses: 0,
      spinBudget: 0,
    };
    let totalDispatches = 0;
    let totalPassEnds = 0;
    if (mlxExports && typeof mlxExports.wgpuGetDispatchStats === "function") {
      const arr = mlxExports.wgpuGetDispatchStats() as number[];
      totalDispatches = arr[0] ?? 0;
      totalPassEnds = arr[1] ?? 0;
    }
    post({
      type: "profile",
      stats: {
        numTokens,
        totalDispatches,
        totalPassEnds,
        bridgeRpcCount: bridgeStats.rpcCount,
        bridgeByFn: bridgeStats.byFn,
        gpuRpcCount: gpuStats.totalRpcs,
        gpuByFn: gpuStats.byFn,
        gpuPoolHits: gpuStats.gpuPoolHits,
        gpuPoolMisses: gpuStats.gpuPoolMisses,
        bindGroupCacheHits: gpuStats.bindGroupCacheHits,
        bindGroupCacheMisses: gpuStats.bindGroupCacheMisses,
        uniformHotHits: gpuStats.uniformHotHits,
        uniformHotMisses: gpuStats.uniformHotMisses,
        spinHits: gpuStats.spinHits,
        spinMisses: gpuStats.spinMisses,
        spinBudget: gpuStats.spinBudget,
        poolHits: bridgeStats.poolHits,
        poolMisses: bridgeStats.poolMisses,
        diagCreateAll: (bridgeStats as any).diagCreateAll ?? 0,
        diagCreateMappedCopyDst:
          (bridgeStats as any).diagCreateMappedCopyDst ?? 0,
        diagCreateMappedNoCopyDst:
          (bridgeStats as any).diagCreateMappedNoCopyDst ?? 0,
        diagReleaseAll: (bridgeStats as any).diagReleaseAll ?? 0,
        diagReleaseUnknownHandle:
          (bridgeStats as any).diagReleaseUnknownHandle ?? 0,
        diagReleaseUnpoolable: (bridgeStats as any).diagReleaseUnpoolable ?? 0,
        diagPoolEvictions: (bridgeStats as any).diagPoolEvictions ?? 0,
        diagBatchAttempt: (bridgeStats as any).diagBatchAttempt ?? 0,
        diagBatchStaged: (bridgeStats as any).diagBatchStaged ?? 0,
        diagBatchDeferredBlock:
          (bridgeStats as any).diagBatchDeferredBlock ?? 0,
        diagBatchStageRefused: (bridgeStats as any).diagBatchStageRefused ?? 0,
        // JS-F008: RPCs dropped because the bridge was poisoned by an
        // earlier BUFFER_RELEASE_BATCH F&F drain timeout. Non-zero here
        // means the demo should treat the bridge as dead and reload.
        poisonedRpcCount: (bridgeStats as any).poisonedRpcCount ?? 0,
      },
    });
  } catch (e) {
    console.warn("[mlx-worker] postProfileSnapshot failed:", e);
  }
}

// Pending stream-finalize resolver. The main-thread SAB reader posts
// {type:'stream-finalize', numTokens} when it has seen the done-chunk;
// handleChatStreamSab awaits this so it can call postProfileSnapshot with
// the correct token count before freeing the ring.
let streamFinalizeResolve: ((numTokens: number) => void) | null = null;

async function handleChat(data: {
  messages: any[];
  config?: any;
  useSab?: boolean;
  mode?: "sab" | "tsfn" | "baseline";
  enableThinking?: boolean;
  reasoningEffort?: ReasoningEffort;
}) {
  if (!wasmMemory || !wasmMalloc || !wasmFree) {
    post({
      type: "error",
      message: "handleChat called before WASM init complete",
    });
    return;
  }
  if (rejectUnsupportedImages(data.messages)) return;
  const mode = data.mode ?? (data.useSab === false ? "tsfn" : "sab");

  if (mode === "tsfn") {
    await handleChatTsfn(data);
    return;
  }
  if (mode === "baseline") {
    await handleChatBaseline(data);
    return;
  }
  const memory = wasmMemory;
  const mallocFn = wasmMalloc;
  const freeFn = wasmFree;

  // Allocate the ring region INSIDE the WASM heap so that the Buffer we hand to
  // napi-rs shares storage with the Rust SabSink. If we allocated a separate
  // SharedArrayBuffer and wrapped it in Buffer.from, emnapi/napi-rs would copy
  // the bytes into the WASM heap on every call, so Rust would write to a heap
  // copy that this JS reader would never see.
  const ringOffset = mallocFn(SAB_RING_SIZE);
  if (ringOffset === 0) {
    post({
      type: "error",
      message: `Failed to malloc ${SAB_RING_SIZE} bytes for SAB ring`,
    });
    return;
  }

  let freed = false;
  const freeRing = () => {
    if (freed) return;
    freed = true;
    freeFn(ringOffset);
  };

  // Register the finalize-resolver BEFORE we tell main to start reading so the
  // main-thread reader's 'stream-finalize' message can't land before we listen.
  const finalizePromise = new Promise<number>((resolve) => {
    streamFinalizeResolve = resolve;
  });

  try {
    const chatConfig = buildChatConfig(data);

    // Buffer.from(arrayBuffer, offset, length) returns a zero-copy VIEW over
    // the WASM heap. Because the underlying ArrayBuffer IS the WASM memory,
    // napi-rs can pass the pointer through without copying — Rust's SabSink
    // and the JS reader see the same bytes.
    const sabBuf = Buffer.from(memory.buffer, ringOffset, SAB_RING_SIZE);

    // Zero the 32-byte SAB header so seq/write_cur/read_cur/cancelled start
    // at 0 for this stream. Previously createSabRingOverHeap did this as a
    // side-effect of its constructor; now the reader runs on the main
    // thread so we must zero it ourselves before the producer writes.
    new Uint8Array(memory.buffer, ringOffset, 32).fill(0);

    const t0 = performance.now();

    // Phase 0: zero all counters before the generation starts. We reset
    // here rather than at the top of handleChat so the SAB-ring alloc
    // doesn't skew the numbers.
    resetProfileCounters();

    // Tell the main thread where the SAB ring lives. The main thread mounts
    // a reader over the WASM heap at this offset and decodes records as
    // SabSink writes them. Main thread NEVER blocks during WASM decode (this
    // worker does), so its Atomics.waitAsync microtasks run immediately and
    // tokens render live. See the 'stream-finalize' handler below.
    post({ type: "stream-sab-open", ringOffset, size: SAB_RING_SIZE });

    await withBridgeModeForMessages(data.messages, async () => {
      // chatStreamSab writes directly into the heap-backed ring — no TSFN on the
      // hot decode path. It returns a handle once the stream is running and
      // resolves when the producer has written the done record.
      const handle = await model.chatStreamSab(
        data.messages,
        chatConfig,
        sabBuf,
      );
      void handle;

      // Wait for the main-thread reader to drain the done-chunk and send us
      // back its numTokens so we can emit the profile snapshot with correct
      // ratios. This also serialises the ring free — we don't free until the
      // reader has consumed everything.
      const numTokens = await finalizePromise;
      const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
      post({
        type: "progress",
        step: "chat",
        message: `chatStream completed in ${elapsed}s`,
      });
      postProfileSnapshot(numTokens);
    });
  } catch (e) {
    let message = String(e);
    let stack: string | undefined;
    if (e instanceof Error) {
      message = e.message;
      stack = e.stack;
    }
    post({ type: "error", message, stack });
  } finally {
    streamFinalizeResolve = null;
    freeRing();
  }
}

async function handleChatBaseline(data: {
  messages: any[];
  config?: any;
  enableThinking?: boolean;
  reasoningEffort?: ReasoningEffort;
}) {
  try {
    if (rejectUnsupportedImages(data.messages)) return;
    const chatConfig = buildChatConfig(data);
    resetProfileCounters();
    const t0 = performance.now();
    const result = await withBridgeModeForMessages(data.messages, () =>
      model.chat(data.messages, chatConfig),
    );
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    post({
      type: "progress",
      step: "chat",
      message: `chat (non-stream) completed in ${elapsed}s`,
    });
    postProfileSnapshot(result.numTokens ?? 0);
    post({
      type: "result",
      text: result.text,
      rawText: result.rawText,
      numTokens: result.numTokens,
      finishReason: result.finishReason,
      toolCalls: result.toolCalls,
      thinking: result.thinking,
      performance: result.performance
        ? {
            ttftMs: result.performance.ttftMs,
            prefillTokensPerSecond: result.performance.prefillTokensPerSecond,
            decodeTokensPerSecond: result.performance.decodeTokensPerSecond,
          }
        : null,
    });
  } catch (e) {
    let message = String(e);
    let stack: string | undefined;
    if (e instanceof Error) {
      message = e.message;
      stack = e.stack;
    }
    post({ type: "error", message, stack });
  }
}

async function handleChatTsfn(data: {
  messages: any[];
  config?: any;
  enableThinking?: boolean;
  reasoningEffort?: ReasoningEffort;
}) {
  try {
    if (rejectUnsupportedImages(data.messages)) return;
    const chatConfig = buildChatConfig(data);
    resetProfileCounters();
    const t0 = performance.now();
    let doneResolve: (() => void) | null = null;
    const doneP = new Promise<void>((r) => {
      doneResolve = r;
    });
    await withBridgeModeForMessages(data.messages, async () => {
      const handle = await model.chatStream(
        data.messages,
        chatConfig,
        (err: Error | null, chunk: any) => {
          if (err) {
            post({ type: "error", message: err.message, stack: err.stack });
            doneResolve?.();
            return;
          }
          if (chunk.done) {
            const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
            post({
              type: "progress",
              step: "chat",
              message: `chatStream completed in ${elapsed}s`,
            });
            postProfileSnapshot(chunk.numTokens ?? 0);
            post({
              type: "result",
              text: chunk.text,
              rawText: chunk.rawText,
              numTokens: chunk.numTokens,
              finishReason: chunk.finishReason,
              toolCalls: chunk.toolCalls,
              thinking: chunk.thinking,
              performance: chunk.performance
                ? {
                    ttftMs: chunk.performance.ttftMs,
                    prefillTokensPerSecond:
                      chunk.performance.prefillTokensPerSecond,
                    decodeTokensPerSecond:
                      chunk.performance.decodeTokensPerSecond,
                  }
                : null,
            });
            doneResolve?.();
          } else {
            post({
              type: "chunk",
              text: chunk.text,
              isReasoning: chunk.isReasoning ?? false,
            });
          }
        },
      );
      void handle;
      await doneP;
    });
  } catch (e) {
    let message = String(e);
    let stack: string | undefined;
    if (e instanceof Error) {
      message = e.message;
      stack = e.stack;
    }
    post({ type: "error", message, stack });
  }
}

// Catch unhandled errors/rejections on this worker
self.addEventListener("error", (e) => {
  e.preventDefault();
  const details = formatUnknownError((e as ErrorEvent).error ?? e);
  const location =
    (e as ErrorEvent).filename ||
    (e as ErrorEvent).lineno ||
    (e as ErrorEvent).colno
      ? `${(e as ErrorEvent).filename}:${(e as ErrorEvent).lineno}:${(e as ErrorEvent).colno}`
      : undefined;
  post({
    type: "error",
    message: `Worker error: ${(e as ErrorEvent).message || details.message}`,
    stack: details.stack || location,
  });
});
self.addEventListener("unhandledrejection", (e: PromiseRejectionEvent) => {
  e.preventDefault();
  const details = formatUnknownError(e.reason);
  post({
    type: "error",
    message: `Unhandled rejection: ${details.message}`,
    stack: details.stack,
  });
});

self.onmessage = (e: MessageEvent) => {
  switch (e.data.type) {
    case "init":
      handleInit(e.data);
      break;
    case "chat":
      handleChat(e.data);
      break;
    case "stream-finalize":
      // Main-thread SAB reader saw the done-chunk and is handing us the
      // token count so we can emit the profile snapshot with correct
      // ratios before freeing the ring. See handleChat above.
      if (streamFinalizeResolve) {
        streamFinalizeResolve(e.data.numTokens ?? 0);
      }
      break;
  }
};
