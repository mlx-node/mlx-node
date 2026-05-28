/**
 * Test bridge: singleton that manages the test-worker lifecycle for Vitest.
 *
 * Spawns test-worker.ts as a Web Worker in 'init-vitest' mode, then provides
 * runTest(name) to execute individual tests on demand. The worker handles all
 * WASM + WebGPU initialization and owns the GPUDevice via its child gpu-worker.
 */

export interface TestResult {
  passed: boolean;
  error?: string;
  info?: unknown;
}

class TestBridge {
  private worker: Worker;
  private pending = new Map<string, { resolve: (r: TestResult) => void }>();
  testNames: string[];

  private constructor(worker: Worker, testNames: string[]) {
    this.worker = worker;
    this.testNames = testNames;

    worker.onmessage = (e: MessageEvent) => {
      const { type, name, passed, error, info } = e.data;
      if (type === 'result') {
        const p = this.pending.get(name);
        if (p) {
          this.pending.delete(name);
          p.resolve({ passed, error, info });
        }
      }
    };
  }

  static async create(): Promise<TestBridge> {
    const worker = new Worker(new URL('../test-worker.ts', import.meta.url), {
      type: 'module',
    });

    const testNames = await new Promise<string[]>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Test worker init timeout (60s)')), 60_000);

      worker.onerror = (err) => {
        clearTimeout(timer);
        reject(new Error(`Test worker error: ${err.message}`));
      };

      worker.onmessage = (e: MessageEvent) => {
        const { type } = e.data;
        if (type === 'ready') {
          clearTimeout(timer);
          resolve(e.data.tests as string[]);
        } else if (type === 'error') {
          clearTimeout(timer);
          reject(new Error(e.data.message));
        }
        // Ignore 'status' messages during init
      };

      worker.postMessage({
        type: 'init-vitest',
        wasmUrl: '/mlx-core.opt.wasm',
      });
    });

    return new TestBridge(worker, testNames);
  }

  runTest(name: string): Promise<TestResult> {
    return new Promise((resolve) => {
      this.pending.set(name, { resolve });
      this.worker.postMessage({ type: 'run', name });
    });
  }

  terminate(): void {
    for (const pending of this.pending.values()) {
      pending.resolve({ passed: false, error: 'Test worker terminated' });
    }
    this.pending.clear();
    this.worker.terminate();
  }
}

// Singleton — shared across all test files when isolate: false
let instance: TestBridge | null = null;
let initPromise: Promise<TestBridge> | null = null;

export async function getBridge(): Promise<TestBridge> {
  if (instance) return instance;
  if (!initPromise) {
    initPromise = TestBridge.create().then((b) => {
      instance = b;
      return b;
    });
  }
  return initPromise;
}

export async function runTest(name: string): Promise<TestResult> {
  const bridge = await getBridge();
  return bridge.runTest(name);
}

export async function getTestNames(): Promise<string[]> {
  const bridge = await getBridge();
  return bridge.testNames;
}

export function terminateBridge(): void {
  instance?.terminate();
  instance = null;
  initPromise = null;
}

if (typeof globalThis.addEventListener === 'function') {
  const cleanup = () => terminateBridge();
  globalThis.addEventListener('pagehide', cleanup, { once: true });
  globalThis.addEventListener('beforeunload', cleanup, { once: true });
}
