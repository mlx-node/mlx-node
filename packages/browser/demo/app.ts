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
const urlParams = new URLSearchParams(location.search);
const packBf16 = urlParams.get('pack_bf16') !== '0'; // enabled by default, ?pack_bf16=0 to disable
const sdpaFallback = urlParams.get('sdpa_fallback') === '1';
const flagBadges =
  (packBf16 ? ' (pack_bf16=1)' : '') +
  (sdpaFallback ? ' (sdpa_fallback=1)' : '');
log(`Starting MLX Worker${flagBadges}...`);
setStatus('Initializing...', 'info');
worker.postMessage({
  type: 'init',
  wasmUrl: new URL(`/mlx-core.opt.wasm?v=${Date.now()}`, location.href).href,
  modelUrl: '/model',
  packBf16,
  sdpaFallback,
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
