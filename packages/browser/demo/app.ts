/**
 * MLX Browser Demo — Thin UI shell
 *
 * All heavy work (WASM, WebGPU, model, inference) runs on a dedicated worker.
 * Main thread only handles UI and postMessage communication.
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

// Create the MLX Worker
const worker = new Worker(
  new URL('../src/mlx-worker.ts', import.meta.url),
  { type: 'module' },
);

// Handle messages from worker
worker.onmessage = (e) => {
  const { type, ...data } = e.data;

  switch (type) {
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
      const assistantDiv = chatEl.querySelector('.message.assistant:last-child');
      if (assistantDiv) {
        assistantDiv.textContent = data.text;
      }
      messages.push({ role: 'assistant', content: data.rawText });

      if (data.performance) {
        log(`${data.numTokens} tokens | TTFT ${data.performance.ttftMs.toFixed(0)}ms | Decode ${data.performance.decodeTokensPerSecond.toFixed(1)} tok/s`);
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
      const errDiv = chatEl.querySelector('.message.assistant:last-child');
      if (errDiv) errDiv.textContent = `Error: ${data.message}`;
      setStatus('Error', 'error');
      sendBtn.disabled = false;
      promptEl.disabled = false;
      break;
  }
};

worker.onerror = (e) => {
  log(`Worker error: ${e.message || e}`);
  if (e instanceof ErrorEvent) {
    log(`  file: ${e.filename}`);
    log(`  line: ${e.lineno}, col: ${e.colno}`);
    if (e.error) {
      log(`  error type: ${e.error?.constructor?.name}`);
      log(`  error: ${String(e.error)}`);
      if (e.error.stack) log(`  stack: ${e.error.stack}`);
    }
  }
  setStatus('Worker error', 'error');
};

worker.addEventListener('messageerror', (e) => {
  log(`Worker messageerror: ${e}`);
});

// Initialize
log('Starting MLX Worker...');
setStatus('Initializing...', 'info');
worker.postMessage({
  type: 'init',
  wasmUrl: new URL('/mlx-core.opt.wasm', location.href).href,
  modelUrl: '/model',
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

  // Show pending response
  setStatus('Generating...', 'info');
  const assistantDiv = document.createElement('div');
  assistantDiv.className = 'message assistant';
  assistantDiv.textContent = '...';
  chatEl.appendChild(assistantDiv);
  chatEl.scrollTop = chatEl.scrollHeight;

  // Send to worker
  worker.postMessage({
    type: 'chat',
    messages: [...messages],
    config: { maxNewTokens: 512, temperature: 0.7, reportPerformance: true },
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
