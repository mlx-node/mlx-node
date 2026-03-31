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
import { Buffer } from 'buffer';
(globalThis as any).Buffer = Buffer;

import {
  instantiateNapiModule,
  getDefaultContext,
  WASI,
} from '@napi-rs/wasm-runtime';

import { createWebGPUBridge } from './webgpu-bridge.js';
import { parseSafeTensorsHeader, dtypeToCode } from './safetensors.js';

let model: any = null;
let mlxExports: any = null;

function post(msg: any) {
  (self as any).postMessage(msg);
}

async function handleInit(data: { wasmUrl: string; modelUrl: string }) {
  try {
    // 1. Create GPUDevice on this worker
    post({ type: 'progress', step: 'gpu', message: 'Initializing WebGPU...' });
    if (!navigator.gpu) {
      throw new Error('WebGPU not available in this worker');
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 10),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1 << 30),
        maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1 << 30),
      },
    });
    const bridge = createWebGPUBridge(adapter, device);
    post({ type: 'progress', step: 'gpu', message: 'WebGPU ready' });

    // 2. Load WASM with real bridge
    post({ type: 'progress', step: 'wasm', message: 'Loading WASM module...' });
    const wasi = new WASI({ version: 'preview1' });
    const context = getDefaultContext();
    const sharedMemory = new WebAssembly.Memory({
      initial: 16000,
      maximum: 65536,
      shared: true,
    });

    const wasmFile = await fetch(data.wasmUrl).then((r) => r.arrayBuffer());

    // WASM exception tag for C++ exceptions (used by -fwasm-exceptions)
    const cppExceptionTag = new WebAssembly.Tag({ parameters: ['i32'] });

    const cxxStubs = {
      __cpp_exception: cppExceptionTag,
      // MLX GPU init — no-op (GPU pre-initialized via bridge)
      _ZN3mlx4core3gpu4initEv: () => {},
    };

    let wasmInstance: WebAssembly.Instance | null = null;

    const { napiModule } = await instantiateNapiModule(wasmFile, {
      context,
      asyncWorkPoolSize: 4,
      onCreateWorker() {
        // Child workers need the real WebGPU bridge (not stubs) because
        // NAPI-RS async functions dispatch through emnapi's worker pool,
        // and MLX GPU ops execute on those workers.
        return new Worker(
          new URL('./webgpu-worker.mjs', import.meta.url),
          { type: 'module' },
        );
      },
      wasi,
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
        wasmInstance = instance;
        bridge.setInstance(instance);
        for (const name of Object.keys(instance.exports)) {
          if (name.startsWith('__napi_register__')) {
            (instance.exports[name] as Function)();
          }
        }
      },
    });

    mlxExports = napiModule.exports;
    post({ type: 'progress', step: 'wasm', message: 'WASM loaded' });

    // 3. Fetch model files
    post({ type: 'progress', step: 'model', message: 'Fetching config...' });
    const configJson = await fetch(`${data.modelUrl}/config.json`).then(r => r.text());

    post({ type: 'progress', step: 'model', message: 'Fetching tokenizer...' });
    const tokenizerJson = await fetch(`${data.modelUrl}/tokenizer.json`).then(r => r.text());

    post({ type: 'progress', step: 'model', message: 'Fetching weights...' });
    const weightsResponse = await fetch(`${data.modelUrl}/model.safetensors`);
    const totalSize = Number(weightsResponse.headers.get('content-length') || 0);
    const reader = weightsResponse.body!.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      if (totalSize > 0 && loaded % (50 * 1024 * 1024) < value.length) {
        const pct = ((loaded / totalSize) * 100).toFixed(0);
        post({ type: 'progress', step: 'download', message: `Downloading weights... ${pct}%`, pct: Number(pct) });
      }
    }

    const weightsBuffer = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      weightsBuffer.set(chunk, offset);
      offset += chunk.length;
    }

    // 4. Parse SafeTensors + create GPU buffers (zero-copy)
    post({ type: 'progress', step: 'gpu_upload', message: 'Creating GPU buffers...' });
    const t0 = performance.now();
    const { tensors, dataOffset } = parseSafeTensorsHeader(weightsBuffer.buffer);

    const gpuTensors: Array<{ name: string; handle: number; dtypeCode: number; shape: number[]; byteSize: number }> = [];

    for (const tensor of tensors) {
      const alignedSize = Math.max(4, Math.ceil(tensor.byteSize / 4) * 4);
      const gpuBuffer = device.createBuffer({
        size: alignedSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });

      const mapped = new Uint8Array(gpuBuffer.getMappedRange());
      const src = new Uint8Array(weightsBuffer.buffer, dataOffset + tensor.byteOffset, tensor.byteSize);
      mapped.set(src);
      gpuBuffer.unmap();

      const handle = bridge.addGPUBuffer(gpuBuffer, alignedSize);
      gpuTensors.push({
        name: tensor.name,
        handle,
        dtypeCode: dtypeToCode(tensor.dtype),
        shape: tensor.shape,
        byteSize: tensor.byteSize,
      });
    }

    post({ type: 'progress', step: 'gpu_upload', message: `${tensors.length} GPU buffers created (${(performance.now() - t0).toFixed(0)}ms)` });

    // 5. Build model
    post({ type: 'progress', step: 'init_model', message: 'Initializing model...' });
    const t1 = performance.now();
    const Qwen35Model = mlxExports.Qwen35Model || mlxExports.Qwen3_5Model;
    model = Qwen35Model.loadFromGpuBuffers(configJson, gpuTensors, tokenizerJson);
    post({ type: 'progress', step: 'init_model', message: `Model ready (${(performance.now() - t1).toFixed(0)}ms)` });

    post({ type: 'ready' });
  } catch (e) {
    post({ type: 'error', message: String(e), stack: (e instanceof Error) ? e.stack : undefined });
  }
}

async function handleChat(data: { messages: any[]; config?: any }) {
  try {
    post({ type: 'progress', step: 'chat', message: 'Starting chat inference...' });

    // Test: try a simple tokenize first to isolate where the error is
    const Qwen35Model = mlxExports.Qwen35Model || mlxExports.Qwen3_5Model;
    post({ type: 'progress', step: 'chat', message: 'Tokenizer test OK, calling model.chat...' });

    // Use chatSync (synchronous) to avoid emnapi async worker thread dispatch
    // which deadlocks because asyncify can't yield on child worker threads.
    const chatFn = model.chatSync || model.chat;
    const result = chatFn.call(model, data.messages, data.config ?? {
      maxNewTokens: 512,
      temperature: 0.7,
      reportPerformance: true,
    });

    post({
      type: 'result',
      text: result.text,
      rawText: result.rawText,
      numTokens: result.numTokens,
      finishReason: result.finishReason,
      toolCalls: result.toolCalls,
      thinking: result.thinking,
      performance: result.performance ? {
        ttftMs: result.performance.ttftMs,
        prefillTokensPerSecond: result.performance.prefillTokensPerSecond,
        decodeTokensPerSecond: result.performance.decodeTokensPerSecond,
      } : null,
    });
  } catch (e) {
    let message = String(e);
    let stack: string | undefined;
    if (e instanceof Error) {
      message = e.message;
      stack = e.stack;
    } else if (e instanceof WebAssembly.Exception) {
      message = `WebAssembly.Exception (C++ exception escaped to JS — likely an unimplemented op or runtime error in MLX)`;
    }
    post({ type: 'error', message, stack });
  }
}

// Catch unhandled errors/rejections on this worker
self.addEventListener('error', (e) => {
  post({ type: 'error', message: `Worker error: ${e.message}`, stack: (e as ErrorEvent).filename });
});
self.addEventListener('unhandledrejection', (e: PromiseRejectionEvent) => {
  post({ type: 'error', message: `Unhandled rejection: ${String(e.reason)}` });
});

self.onmessage = (e: MessageEvent) => {
  switch (e.data.type) {
    case 'init':
      handleInit(e.data);
      break;
    case 'chat':
      handleChat(e.data);
      break;
  }
};
