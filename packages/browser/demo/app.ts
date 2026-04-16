/**
 * MLX Browser Demo — Streaming chat with thinking rendering
 *
 * All heavy work (WASM, WebGPU, model, inference) runs on a dedicated worker.
 * Main thread handles UI, postMessage communication, and streaming token rendering.
 */

const statusEl = document.getElementById('status')!;
const logEl = document.getElementById('log')!;
const chatEl = document.getElementById('chat')!;
const promptEl = document.getElementById('prompt') as HTMLTextAreaElement;
const sendBtn = document.getElementById('send') as HTMLButtonElement;
const imageBtn = document.getElementById('image-btn') as HTMLButtonElement;
const imageInput = document.getElementById('image-input') as HTMLInputElement;

function setStatus(text: string, state: 'info' | 'ready' | 'error' = 'info') {
  statusEl.textContent = text;
  statusEl.className = state;
}

function log(msg: string) {
  const line = document.createElement('div');
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

let pendingImage: Uint8Array | null = null;
const messages: Array<{ role: string; content: string; images?: Uint8Array[] }> = [
  { role: 'system', content: 'You are a helpful assistant. Be concise.' },
];

// Streaming state
let currentAssistantDiv: HTMLDivElement | null = null;
let currentThinkingDiv: HTMLDivElement | null = null;
let currentResponseDiv: HTMLDivElement | null = null;
let isInThinking = false;
let streamedText = ''; // Accumulates all streamed tokens
let streamTokenCount = 0;

function createAssistantMessage(): HTMLDivElement {
  const assistantDiv = document.createElement('div');
  assistantDiv.className = 'message assistant';

  // Thinking section (collapsible)
  const thinkingDiv = document.createElement('details');
  thinkingDiv.className = 'thinking';
  const thinkingSummary = document.createElement('summary');
  thinkingSummary.textContent = 'Thinking...';
  thinkingDiv.appendChild(thinkingSummary);
  const thinkingContent = document.createElement('div');
  thinkingContent.className = 'thinking-content';
  thinkingDiv.appendChild(thinkingContent);
  assistantDiv.appendChild(thinkingDiv);

  // Response section
  const responseDiv = document.createElement('div');
  responseDiv.className = 'response-content';
  assistantDiv.appendChild(responseDiv);

  chatEl.appendChild(assistantDiv);
  chatEl.scrollTop = chatEl.scrollHeight;

  currentAssistantDiv = assistantDiv;
  currentThinkingDiv = thinkingDiv;
  currentResponseDiv = responseDiv;
  isInThinking = false; // Will be set true when we see <think> in the stream

  // Hide thinking section initially — only show if model produces thinking
  thinkingDiv.style.display = 'none';

  return assistantDiv;
}

function appendStreamedToken(deltaText: string, isReasoning: boolean) {
  if (!currentAssistantDiv || !currentThinkingDiv || !currentResponseDiv) return;

  streamedText += deltaText;
  streamTokenCount++;
  setStatus(`Generating... ${streamTokenCount} tokens`, 'info');

  if (isReasoning) {
    // Model is in thinking mode — show thinking content
    isInThinking = true;
    currentThinkingDiv.style.display = '';
    currentThinkingDiv.open = true;
    const thinkingContentEl = currentThinkingDiv.querySelector('.thinking-content') as HTMLElement;
    if (thinkingContentEl) thinkingContentEl.textContent += deltaText;
    const summary = currentThinkingDiv.querySelector('summary') as HTMLElement;
    if (summary) summary.textContent = 'Thinking...';
  } else {
    // Transition from thinking → response
    if (isInThinking) {
      isInThinking = false;
      const summary = currentThinkingDiv.querySelector('summary') as HTMLElement;
      if (summary) summary.textContent = 'Thought process';
      currentThinkingDiv.open = false;
    }
    currentResponseDiv.textContent += deltaText;
  }

  chatEl.scrollTop = chatEl.scrollHeight;
}

function finalizeAssistantMessage(text: string, thinking: string | null) {
  if (!currentAssistantDiv || !currentThinkingDiv || !currentResponseDiv) return;

  const trimmedThinking = thinking?.trim() || '';
  // Only show thinking section if it contains substantial content (not just punctuation)
  if (trimmedThinking.length > 3) {
    // Show thinking (only if substantial)
    const thinkingContentEl = currentThinkingDiv.querySelector('.thinking-content') as HTMLElement;
    if (thinkingContentEl) thinkingContentEl.textContent = trimmedThinking;
    const summary = currentThinkingDiv.querySelector('summary') as HTMLElement;
    if (summary) summary.textContent = 'Thought process';
    currentThinkingDiv.open = false;
    currentThinkingDiv.style.display = '';
  } else {
    // No or trivial thinking — hide the section
    currentThinkingDiv.style.display = 'none';
  }

  currentResponseDiv.textContent = text || '';
  currentAssistantDiv.classList.add('done'); // Remove cursor animation
  chatEl.scrollTop = chatEl.scrollHeight;

  currentAssistantDiv = null;
  currentThinkingDiv = null;
  currentResponseDiv = null;
}

// Create the MLX Worker
const worker = new Worker(new URL('../src/mlx-worker.ts', import.meta.url), { type: 'module' });

// Handle messages from worker
worker.onmessage = (e) => {
  const { type, ...data } = e.data;

  switch (type) {
    case 'chunk':
      // Per-token streaming chunk from chatStream callback
      appendStreamedToken(data.text, data.isReasoning);
      break;

    case 'log':
      // Forwarded log from the worker (e.g. PACKBF16 debug from gpu-worker).
      // Routed through the main-thread console so DevTools / MCP capture it.
      console.log(data.message);
      break;

    case 'progress':
      log(data.message);
      if (data.step === 'download') {
        setStatus(`Downloading weights... ${data.pct}%`, 'info');
      } else {
        setStatus(data.message, 'info');
      }
      break;

    case 'ready':
      log('Model ready!');
      setStatus('Qwen 3.5 0.8B — Ready', 'ready');
      promptEl.disabled = false;
      sendBtn.disabled = false;
      imageBtn.disabled = false;
      break;

    case 'result': {
      finalizeAssistantMessage(data.text, data.thinking);
      messages.push({ role: 'assistant', content: data.rawText });

      if (data.performance) {
        log(
          `${data.numTokens} tokens | TTFT ${data.performance.ttftMs.toFixed(0)}ms | Decode ${data.performance.decodeTokensPerSecond.toFixed(1)} tok/s`,
        );
      }
      setStatus('Qwen 3.5 0.8B — Ready', 'ready');
      sendBtn.disabled = false;
      promptEl.disabled = false;
      promptEl.focus();
      chatEl.scrollTop = chatEl.scrollHeight;
      break;
    }

    case 'profile': {
      // Phase 0 observability readout. The payload is a cumulative snapshot
      // taken right after chat/chatStream completes. We format a single
      // summary line into the log panel so we can eyeball dispatches/token
      // without plumbing it through perf.
      const s = data.stats as {
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
        diagCreateAll?: number;
        diagCreateMappedCopyDst?: number;
        diagCreateMappedNoCopyDst?: number;
        diagReleaseAll?: number;
        diagReleaseUnknownHandle?: number;
        diagReleaseUnpoolable?: number;
        diagBatchAttempt?: number;
        diagBatchStaged?: number;
        diagBatchDeferredBlock?: number;
        diagBatchStageRefused?: number;
      };
      const n = Math.max(1, s.numTokens);
      const dpt = s.totalDispatches / n;
      const rpt = s.bridgeRpcCount / n;
      const ept = s.totalPassEnds / n;
      const gptRpc = s.gpuRpcCount / n;
      // Emit every non-zero opcode from the gpu-worker histogram, sorted
      // descending. We need the full picture to diagnose whether the
      // bottleneck is compute-pass overhead or buffer-lifecycle churn.
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
      log(
        `[profile] diag: createAll=${s.diagCreateAll ?? 0} (mappedCopyDst=${s.diagCreateMappedCopyDst ?? 0}, mappedNoCopyDst=${s.diagCreateMappedNoCopyDst ?? 0}) | releaseAll=${s.diagReleaseAll ?? 0} (unknownHandle=${s.diagReleaseUnknownHandle ?? 0}, unpoolable=${s.diagReleaseUnpoolable ?? 0})`,
      );
      const ba = s.diagBatchAttempt ?? 0;
      const bs = s.diagBatchStaged ?? 0;
      const bdb = s.diagBatchDeferredBlock ?? 0;
      const bsr = s.diagBatchStageRefused ?? 0;
      log(
        `[profile] batch: attempt=${ba} staged=${bs} deferredBlock=${bdb} stageRefused=${bsr}`,
      );
      break;
    }

    case 'error':
      log(`Error: ${data.message}`);
      if (currentResponseDiv) {
        currentResponseDiv.textContent = `Error: ${data.message}`;
      }
      setStatus('Error', 'error');
      sendBtn.disabled = false;
      promptEl.disabled = false;
      currentAssistantDiv = null;
      currentThinkingDiv = null;
      currentResponseDiv = null;
      break;
  }
};

// Phase 0: minimal opcode → name mapping for the profile line. Keeps the
// demo free of a runtime import of the const-enum RpcFn from rpc-protocol.ts
// (which is inlined by tsdown and wouldn't produce a reverse mapping).
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
    default:
      return `fn${fn}`;
  }
}

worker.onerror = (e) => {
  log(`Worker error: ${e.message || e}`);
  if (e instanceof ErrorEvent) {
    log(`  file: ${e.filename}`);
    log(`  line: ${e.lineno}, col: ${e.colno}`);
  }
  setStatus('Worker error', 'error');
};

worker.addEventListener('messageerror', (e) => {
  log(`Worker messageerror: ${e}`);
});

// Initialize
// ?pack_bf16=1 opts into the packed-bf16 weight storage path for the WebGPU
// backend. This is a runtime A/B toggle — no rebuild required. The flag is
// forwarded to the WASM worker which calls wgpuSetPackedBf16Enabled on the
// native binding before the model is loaded.
//
// ?sdpa_fallback=1 forces the WebGPU SDPA primitive onto the decomposed
// matmul→softmax→matmul path, bypassing the fused vector / tile kernels.
// Used for A/B-ing TTFT against the fused tile kernel without a rebuild.
//
// ?profile=1 enables Phase 0 dispatch-stats observability. When set, the
// mlx-worker wraps each chat/chatStream call with a reset + readback of
// the C++ dispatch counters, the wasm-worker RPC histogram, and the
// gpu-worker RPC histogram. The result lands in the log panel as a
// single summary line so we can verify the decode dispatches-per-token
// bottleneck without a rebuild.
//
// ?compile_mlp=1 enables the Phase 6b SwiGLU MLP `mlx::core::compile`
// fast path. The compile pass collapses the sigmoid + 2 multiplies between
// the three matmuls into a single fused element-wise kernel (saving 2
// dispatches per layer per token). Default OFF — toggle here for an A/B.
//
// ?compile_gdn_pre=1 enables the Phase 6c GDN pre-recurrence
// `mlx::core::compile` fast path. The compile pass collapses the ~10
// element-wise ops that surround the matmuls / slices / rms_norms in the
// GDN pre-recurrence body into fused Compiled kernels — the single
// biggest dispatch-count lever in the Phase 6 plan. Default OFF.
//
// ?compile_gdn_post=1 enables the Phase 6d GDN post-recurrence
// `mlx::core::compile` fast path. The compile pass collapses the sigmoid
// + two multiplies between fast::rms_norm and the out-proj matmul into a
// single fused Compiled kernel (~3 dispatches saved per linear-attention
// layer per token). Smaller win than Phase 6c but same mechanism.
// Default OFF.
//
// ?compile_gdn_g=1 enables the Phase 6e GDN decay-gate (compute_g)
// `mlx::core::compile` fast path. The compile pass collapses the 9-op
// softplus/exp chain (-exp(a_log) * softplus(a + dt_bias)) into a single
// fused Compiled kernel — the biggest remaining dispatch lever at ~8
// dispatches saved per linear-attention layer per token (~144 per decode
// token across all 18 layers). Default OFF.
const urlParams = new URLSearchParams(location.search);
const packBf16 = urlParams.get('pack_bf16') !== '0'; // enabled by default, ?pack_bf16=0 to disable
const sdpaFallback = urlParams.get('sdpa_fallback') === '1';
const profile = urlParams.get('profile') === '1';
// Phase 2: DISPATCH_BATCH gating. Enabled by default — batches up to 16
// FUSED_*_DISPATCH RPCs into a single DISPATCH_BATCH RPC, cutting ~765
// blocking Atomics.wait round-trips per token.  Disable with
// ?dispatch_batch=0 for A/B comparison.
const dispatchBatch = urlParams.get('dispatch_batch') !== '0';
const compileMlp = urlParams.get('compile_mlp') === '1';
const compileGdnPre = urlParams.get('compile_gdn_pre') === '1';
const compileGdnPost = urlParams.get('compile_gdn_post') === '1';
const compileGdnG = urlParams.get('compile_gdn_g') === '1';
const useSab = urlParams.get('stream_sab') !== '0'; // enabled by default, ?stream_sab=0 to fall back to TSFN
// ?mode=baseline | sab | tsfn — diagnostic A/B for the streaming path.
// baseline = non-streaming chat() (ground-truth compute, no sink overhead).
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
  (useSab ? '' : ' (stream_sab=0)') +
  (modeParam ? ` (mode=${modeParam})` : '');
log(`Starting MLX Worker${flagBadges}...`);
setStatus('Initializing...', 'info');
worker.postMessage({
  type: 'init',
  wasmUrl: new URL(`/mlx-core.opt.wasm?v=${Date.now()}`, location.href).href,
  modelUrl: '/model',
  packBf16,
  sdpaFallback,
  profile,
  dispatchBatch,
  compileMlp,
  compileGdnPre,
  compileGdnPost,
  compileGdnG,
});

// Chat handler
function handleSend() {
  const text = promptEl.value.trim();
  if (!text) return;

  promptEl.value = '';
  sendBtn.disabled = true;
  promptEl.disabled = true;

  // Show user message
  const userDiv = document.createElement('div');
  userDiv.className = 'message user';
  userDiv.textContent = text;
  if (pendingImage) {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(new Blob([pendingImage]));
    img.style.cssText = 'max-width:200px;max-height:150px;border-radius:6px;margin-top:8px;display:block';
    userDiv.appendChild(img);
  }
  chatEl.appendChild(userDiv);

  const msg: { role: string; content: string; images?: Uint8Array[] } = { role: 'user', content: text };
  if (pendingImage) {
    msg.images = [pendingImage];
    pendingImage = null;
    imageBtn.textContent = '+';
  }
  messages.push(msg);

  // Show pending response with thinking/response structure
  setStatus('Generating...', 'info');
  streamedText = '';
  streamTokenCount = 0;
  isInThinking = false;
  createAssistantMessage();

  // Send to worker
  worker.postMessage({
    type: 'chat',
    messages: [...messages],
    config: { maxNewTokens: 512, temperature: 0, reportPerformance: true },
    useSab,
    mode: modeParam ?? undefined,
  });
}

sendBtn.addEventListener('click', handleSend);
promptEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

// Image handling
imageBtn.addEventListener('click', () => imageInput.click());
imageInput.addEventListener('change', async () => {
  const file = imageInput.files?.[0];
  if (!file) return;
  pendingImage = new Uint8Array(await file.arrayBuffer());
  log(`Image attached: ${file.name} (${(pendingImage.length / 1024).toFixed(0)} KB)`);
  imageBtn.textContent = '🖼';
});

document.addEventListener('paste', async (e) => {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      const file = item.getAsFile();
      if (!file) continue;
      pendingImage = new Uint8Array(await file.arrayBuffer());
      log(`Image pasted (${(pendingImage.length / 1024).toFixed(0)} KB)`);
      imageBtn.textContent = '🖼';
      break;
    }
  }
});
