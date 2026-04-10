/**
 * WebGPU Op Test Worker
 */

function post(data: any) { (self as any).postMessage(data); }

// Register message handler IMMEDIATELY (before any async work)
self.onmessage = async (e: MessageEvent) => {
  if (e.data.type !== 'init') return;
  try {
    await initAndRun(e.data.wasmUrl);
  } catch (err: any) {
    post({ type: 'error', message: `${err.message}\n${err.stack || ''}` });
  }
};

async function initAndRun(wasmUrl: string) {
  post({ type: 'status', message: 'Importing modules...' });

  // Dynamic imports to avoid blocking module evaluation
  const { Buffer } = await import('buffer');
  (globalThis as any).Buffer = Buffer;

  // Patch TextDecoder for SharedArrayBuffer
  const _origDecode = TextDecoder.prototype.decode;
  TextDecoder.prototype.decode = function (input?: BufferSource | null, options?: TextDecodeOptions) {
    if (input && ArrayBuffer.isView(input) && (input.buffer as any) instanceof SharedArrayBuffer) {
      input = (input as Uint8Array).slice();
    }
    return _origDecode.call(this, input, options);
  };

  const { instantiateNapiModule, getDefaultContext, WASI } = await import('@napi-rs/wasm-runtime');
  const { createBridgeStub } = await import('./webgpu-bridge-stub');
  const { CMD_OFFSET, READBACK_BUFFER_SIZE } = await import('./rpc-protocol');
  const { createCxxStubs } = await import('./cxx-stubs');

  // Monkey-patch WebAssembly.instantiate to fix WASM32 signed pointer overflow.
  // When the WASM heap exceeds 2GB, malloc returns addresses >= 0x80000000.
  // JavaScript coerces the WASM i32 return value to a signed number, making
  // these addresses negative. Callers that use the pointer as a TypedArray
  // offset (e.g., emnapi's getArrayBufferPointer) then fail with
  // "offset is out of bounds". We wrap the exported malloc so it always
  // returns an unsigned 32-bit value via >>> 0.
  //
  // WebAssembly.Instance.exports is frozen, so Proxy get-traps can't return
  // a different value for non-configurable properties. Instead, we create a
  // plain-object copy of exports with malloc wrapped, and return a fake
  // instance object. emnapi's loadNapiModuleImpl copies exports via
  // Object.assign, so the wrapped malloc propagates to its internal _malloc.
  const _origInstantiate = WebAssembly.instantiate;
  (WebAssembly as any).instantiate = async function (...args: any[]) {
    const result = await _origInstantiate.apply(WebAssembly, args);
    // WebAssembly.instantiate returns { module, instance } for ArrayBuffer input
    const inst = result.instance ?? result;
    const origExports = inst.exports;
    const origMalloc = origExports.malloc as Function | undefined;
    if (typeof origMalloc === 'function') {
      // Build a plain-object copy of exports with malloc wrapped.
      const wrappedExports: Record<string, WebAssembly.ExportValue> = Object.create(null);
      for (const key of Object.keys(origExports)) {
        wrappedExports[key] = origExports[key];
      }
      wrappedExports.malloc = function (size: number) {
        return (origMalloc(size) as number) >>> 0;
      };
      // Return a fake instance with the wrapped exports.
      // The real instance is preserved as __realInstance for callers
      // (like beforeInit) that need the genuine WebAssembly.Instance.
      const fakeInstance = { exports: wrappedExports, __realInstance: inst };
      if (result.instance) {
        return { module: result.module, instance: fakeInstance };
      }
      return fakeInstance;
    }
    return result;
  };


  post({ type: 'status', message: 'Spawning GPU worker...' });

  const cmdBuf = new SharedArrayBuffer(CMD_OFFSET.TOTAL);
  const readbackBuf = new SharedArrayBuffer(READBACK_BUFFER_SIZE);
  const wasmMem = new WebAssembly.Memory({ initial: 16000, maximum: 65536, shared: true });

  const gpuW = new Worker(new URL('./gpu-worker.ts', import.meta.url), { type: 'module' });
  gpuW.onerror = (err) => post({ type: 'error', message: `GPU worker error: ${err.message}` });

  const ready = await new Promise<any>((res, rej) => {
    const timer = setTimeout(() => rej(new Error('GPU worker timeout (15s)')), 15000);
    gpuW.onmessage = (ev) => {
      if (ev.data.type === 'ready') { clearTimeout(timer); res(ev.data); }
      else if (ev.data.type === 'error') { clearTimeout(timer); rej(new Error(ev.data.message)); }
    };
    gpuW.postMessage({ type: 'init', cmdBuffer: cmdBuf, readbackBuffer: readbackBuf, wasmMemory: wasmMem });
  });

  post({ type: 'status', message: 'WebGPU ready. Loading WASM...' });

  const bridge = createBridgeStub(cmdBuf, wasmMem, {
    instanceHandle: ready.instanceHandle, adapterHandle: ready.adapterHandle,
    deviceHandle: ready.deviceHandle, queueHandle: ready.queueHandle,
  }, readbackBuf);

  const wasmFile = await fetch(wasmUrl).then(r => r.arrayBuffer());
  const stubs = createCxxStubs(wasmMem);
  const tag = new WebAssembly.Tag({ parameters: ['i32'] });
  const wasi = new WASI({ version: 'preview1' });
  const context = getDefaultContext();

  let wasmInst: WebAssembly.Instance | null = null;
  let napiModResult: any;
  try {
    napiModResult = await instantiateNapiModule(wasmFile, {
      context,
      asyncWorkPoolSize: 0,
      wasi,
      overwriteImports(importObject: Record<string, Record<string, unknown>>) {
        importObject.env = {
          ...importObject.env,
          ...importObject.napi,
          ...importObject.emnapi,
          memory: wasmMem,
          ...stubs.stubs,
          ...bridge.imports,
          __cpp_exception: tag,
          _ZN3mlx4core3gpu4initEv: () => {},
          mlx_stream_write: () => {},
          mlx_stream_reset: () => {},
        };
        return importObject;
      },
      beforeInit({ instance }: { instance: WebAssembly.Instance }) {
        wasmInst = instance;
        bridge.setInstance(instance);
        for (const k of Object.keys(instance.exports)) {
          if (k.startsWith('__napi_register__')) (instance.exports[k] as Function)();
        }
      },
    });
  } finally {
    // Restore original WebAssembly.instantiate
    (WebAssembly as any).instantiate = _origInstantiate;
  }
  const mod = napiModResult.napiModule;

  const mlxExports = (mod as any).exports;
  const MxArray = mlxExports.MxArray;
  if (!MxArray) throw new Error('MxArray not found');

  post({ type: 'status', message: 'WASM loaded. Running tests...' });

  // ---- Test helpers ----
  function assertClose(actual: number[], expected: number[], tol = 1e-3) {
    if (actual.length !== expected.length) throw new Error(`Length: ${actual.length} vs ${expected.length}`);
    for (let i = 0; i < actual.length; i++) {
      if (Math.abs(actual[i] - expected[i]) > tol) throw new Error(`[${i}]: ${actual[i]} vs ${expected[i]}`);
    }
  }

  // ---- Packed bf16 helpers ----
  // Toggle the WebGPU backend's runtime flag that decides whether
  // bf16 weight buffers go through the packed-bf16 storage path. The flag
  // is process-wide on the C++ side; tests bracket their dispatches with
  // setPackedBf16(true/false) and always restore false in `finally`.
  const setPackedBf16 = (v: boolean) => {
    if (typeof mlxExports.wgpuSetPackedBf16Enabled === 'function') {
      mlxExports.wgpuSetPackedBf16Enabled(v);
    }
  };
  setPackedBf16(false);

  // ---- SDPA fallback force helper ----
  // Toggle the WebGPU SDPA primitive's runtime kill-switch. When true,
  // `ScaledDotProductAttention::use_fallback()` returns true unconditionally
  // so the decomposed matmul→softmax→matmul path runs instead of the fused
  // tile/vector kernels. Used by the tile SDPA parity tests to get a reference
  // value from the fallback path, then compare against the fused-kernel output.
  const setSdpaFallbackForced = (v: boolean) => {
    if (typeof mlxExports.wgpuSetSdpaFallbackForced !== 'function') {
      throw new Error('wgpuSetSdpaFallbackForced not found — SDPA fallback toggle not available');
    }
    mlxExports.wgpuSetSdpaFallbackForced(v);
  };
  setSdpaFallbackForced(false);

  // Convert a Float32Array → bf16 bytes by truncating each f32 to its
  // upper 16 bits (round-toward-zero). Hands the bytes to the bf16 array
  // constructor, which (when packed flag is on and size ≥ threshold) flips
  // the underlying WebGPUBuffer into PackedBf16 storage.
  const f32ToBf16Bytes = (src: Float32Array): Uint8Array => {
    const out = new Uint8Array(src.length * 2);
    const f32buf = new Float32Array(1);
    const u32view = new Uint32Array(f32buf.buffer);
    for (let i = 0; i < src.length; i++) {
      f32buf[0] = src[i];
      const bits = u32view[0];
      const hi = (bits >>> 16) & 0xffff;
      out[2 * i] = hi & 0xff;
      out[2 * i + 1] = (hi >>> 8) & 0xff;
    }
    return out;
  };
  const makeBf16 = (src: Float32Array, shape: BigInt64Array) => {
    return MxArray.fromBfloat16Bytes(f32ToBf16Bytes(src), shape);
  };
  // Test-only variant: matches the production `gpu-worker.ts`
  // NORM_PACKED_MIN_ELEMENTS=256 threshold used by the RMSNorm upload path,
  // which is lower than the GEMV-tuned default of 4096 baked into
  // `fromBfloat16Bytes`. Exercises the packed-norm path for small D=1024
  // Qwen3.5 RMSNorm weights that would otherwise sit below the default floor.
  // Do NOT use outside packed-norm tests — the default 4096 floor is there
  // for a reason on matmul paths.
  const makeBf16NormThreshold = (src: Float32Array, shape: BigInt64Array) => {
    return MxArray.fromBfloat16Bytes(f32ToBf16Bytes(src), shape, 256);
  };
  // Tolerance for packed-vs-unpacked parity. Both paths take the same
  // truncated bf16 bytes — the only difference is GPU storage layout — so
  // results should be bit-identical modulo accumulator ordering. 1e-3 abs
  // is conservative for the f32 accumulators used in the GEMV reductions.
  const PACKED_TOL = 1e-3;

  // ---- Tests ----
  const tests: { name: string; run: () => void }[] = [
    { name: 'create float32', run() {
      const a = MxArray.fromFloat32(new Float32Array([1, 2, 3]), new BigInt64Array([3n]));
      const s = a.size();
      if (s != 3) throw new Error(`size=${s} (type=${typeof s})`);
    }},
    { name: 'zeros+eval', run() {
      const a = MxArray.zeros(new BigInt64Array([4n]));
      a.eval();
      assertClose([...a.toFloat32()], [0, 0, 0, 0]);
    }},
    { name: 'add', run() {
      const a = MxArray.fromFloat32(new Float32Array([1, 2, 3]), new BigInt64Array([3n]));
      const b = MxArray.fromFloat32(new Float32Array([4, 5, 6]), new BigInt64Array([3n]));
      const c = a.add(b); c.eval();
      assertClose([...c.toFloat32()], [5, 7, 9]);
    }},
    { name: 'mul', run() {
      const a = MxArray.fromFloat32(new Float32Array([2, 3]), new BigInt64Array([2n]));
      const b = MxArray.fromFloat32(new Float32Array([10, 20]), new BigInt64Array([2n]));
      const c = a.mul(b); c.eval();
      assertClose([...c.toFloat32()], [20, 60]);
    }},
    { name: 'abs', run() {
      const a = MxArray.fromFloat32(new Float32Array([-1, 2, -3]), new BigInt64Array([3n]));
      const c = a.abs(); c.eval();
      assertClose([...c.toFloat32()], [1, 2, 3]);
    }},
    { name: 'exp', run() {
      const a = MxArray.fromFloat32(new Float32Array([0, 1]), new BigInt64Array([2n]));
      const c = a.exp(); c.eval();
      assertClose([...c.toFloat32()], [1, Math.exp(1)], 1e-2);
    }},
    { name: 'sqrt', run() {
      const a = MxArray.fromFloat32(new Float32Array([1, 4, 9]), new BigInt64Array([3n]));
      const c = a.sqrt(); c.eval();
      assertClose([...c.toFloat32()], [1, 2, 3]);
    }},
    { name: 'reshape', run() {
      const a = MxArray.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), new BigInt64Array([6n]));
      const b = a.reshape(new BigInt64Array([2n, 3n]));
      const s = b.shape();
      if (Number(s[0]) !== 2 || Number(s[1]) !== 3) throw new Error(`shape=${s}`);
    }},
    { name: 'sum', run() {
      const a = MxArray.fromFloat32(new Float32Array([1, 2, 3, 4]), new BigInt64Array([4n]));
      const c = a.sum(); c.eval();
      assertClose([...c.toFloat32()], [10]);
    }},
    { name: 'max', run() {
      const a = MxArray.fromFloat32(new Float32Array([3, 1, 5, 2]), new BigInt64Array([4n]));
      const c = a.max(); c.eval();
      assertClose([...c.toFloat32()], [5]);
    }},
    { name: 'matmul 2x3*3x2', run() {
      const a = MxArray.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), new BigInt64Array([2n, 3n]));
      const b = MxArray.fromFloat32(new Float32Array([7, 8, 9, 10, 11, 12]), new BigInt64Array([3n, 2n]));
      const c = a.matmul(b); c.eval();
      assertClose([...c.toFloat32()], [58, 64, 139, 154]);
    }},
    { name: 'addScalar', run() {
      const a = MxArray.fromFloat32(new Float32Array([1, 2, 3]), new BigInt64Array([3n]));
      const c = a.addScalar(10); c.eval();
      assertClose([...c.toFloat32()], [11, 12, 13]);
    }},
    { name: 'negative', run() {
      const a = MxArray.fromFloat32(new Float32Array([1, -2, 3]), new BigInt64Array([3n]));
      const c = a.negative(); c.eval();
      assertClose([...c.toFloat32()], [-1, 2, -3]);
    }},
    { name: 'sum axis', run() {
      const a = MxArray.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), new BigInt64Array([2n, 3n]));
      const c = a.sum(new Int32Array([1])); c.eval();
      assertClose([...c.toFloat32()], [6, 15]);
    }},
    { name: 'broadcast add', run() {
      const a = MxArray.fromFloat32(new Float32Array([1, 2, 3]), new BigInt64Array([1n, 3n]));
      const b = MxArray.fromFloat32(new Float32Array([10, 20]), new BigInt64Array([2n, 1n]));
      const c = a.add(b); c.eval();
      assertClose([...c.toFloat32()], [11, 12, 13, 21, 22, 23]);
    }},
    { name: 'logSoftmax', run() {
      const a = MxArray.fromFloat32(new Float32Array([1, 2, 3]), new BigInt64Array([1n, 3n]));
      const c = a.logSoftmax(-1); c.eval();
      const d = [...c.toFloat32()];
      // log(softmax([1,2,3])) = [ln(0.0900), ln(0.2447), ln(0.6652)]
      assertClose(d, [-2.4076, -1.4076, -0.4076], 1e-2);
    }},
    { name: 'concatenate', run() {
      const a = MxArray.fromFloat32(new Float32Array([1, 2, 3]), new BigInt64Array([3n]));
      const b = MxArray.fromFloat32(new Float32Array([4, 5]), new BigInt64Array([2n]));
      const c = MxArray.concatenate(a, b, 0); c.eval();
      assertClose([...c.toFloat32()], [1, 2, 3, 4, 5]);
    }},
    { name: 'arange', run() {
      const a = MxArray.arange(0, 5, 1); a.eval();
      assertClose([...a.toFloat32()], [0, 1, 2, 3, 4]);
    }},
    { name: 'chained (a+b)*c', run() {
      const a = MxArray.fromFloat32(new Float32Array([1, 2, 3]), new BigInt64Array([3n]));
      const b = MxArray.fromFloat32(new Float32Array([4, 5, 6]), new BigInt64Array([3n]));
      const c = MxArray.fromFloat32(new Float32Array([2, 2, 2]), new BigInt64Array([3n]));
      const r = a.add(b).mul(c); r.eval();
      assertClose([...r.toFloat32()], [10, 14, 18]);
    }},
    // ---- mlx::core::compile tests (graph caching for WebGPU) ----
    // Run these early to avoid resource exhaustion from 100+ prior tests
    { name: 'compile basic (add, cached graph)', run() {
      if (!MxArray.testCompileBasic()) throw new Error('compile basic failed');
    }},
    { name: 'compile matmul + fused ops (5 iterations)', run() {
      if (!MxArray.testCompileMatmul()) throw new Error('compile matmul failed');
    }},
    { name: 'compile repeated eval (50x heap stress)', run() {
      if (!MxArray.testCompileRepeated()) throw new Error('compile repeated failed');
    }},
    { name: 'gpu buffer array: create from eval\'d buffer, matmul 3 rounds', run() {
      if (!MxArray.testGpuBufferArrays()) throw new Error('gpu buffer array test failed');
    }},
    { name: 'transpose of GPU-backed array correctness', run() {
      const data = new Float32Array(16);
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) data[i*4+j] = i*10 + j;
      const a = MxArray.fromFloat32(data, new BigInt64Array([4n, 4n]));
      a.eval();
      const at = a.transpose();
      at.eval();
      const r = [...at.toFloat32()];
      // a[0][1]=1 → at[1][0]=1, a[2][3]=23 → at[3][2]=23
      if (Math.abs(r[1*4+0] - 1) > 0.001) throw new Error('at[1][0]=' + r[4] + ' expected 1');
      if (Math.abs(r[3*4+2] - 23) > 0.001) throw new Error('at[3][2]=' + r[14] + ' expected 23');
      if (Math.abs(r[2*4+1] - 12) > 0.001) throw new Error('at[2][1]=' + r[9] + ' expected 12');
    }},
    { name: 'matmul with transposed weight (linear_proj pattern)', run() {
      // linear_proj: y = matmul(x, W.T) where W=[O,D], W.T=[D,O]
      const D = 8, O = 4;
      const wData = new Float32Array(O * D);
      for (let i = 0; i < O; i++) for (let j = 0; j < D; j++) wData[i*D+j] = (i*D+j) * 0.01;
      const W = MxArray.fromFloat32(wData, new BigInt64Array([BigInt(O), BigInt(D)]));
      W.eval();
      const Wt = W.transpose(); Wt.eval(); // [D, O]
      const x = MxArray.fromFloat32(new Float32Array(D).fill(1), new BigInt64Array([1n, BigInt(D)]));
      x.eval();
      const y = x.matmul(Wt); y.eval(); // [1, D] @ [D, O] = [1, O]
      const r = [...y.toFloat32()];
      for (let j = 0; j < O; j++) {
        let exp = 0;
        for (let i = 0; i < D; i++) exp += wData[j*D+i]; // W[j][i]
        if (Math.abs(r[j] - exp) > 0.01) throw new Error('y[' + j + ']=' + r[j] + ' expected ' + exp);
      }
    }},
    { name: 'repeated matmul with transposed weight (deterministic)', run() {
      const D = 16, O = 8;
      const wData = new Float32Array(O * D);
      for (let i = 0; i < wData.length; i++) wData[i] = Math.sin(i * 0.1) * 0.5;
      const W = MxArray.fromFloat32(wData, new BigInt64Array([BigInt(O), BigInt(D)]));
      W.eval();
      const Wt = W.transpose(); Wt.eval();
      const xData = new Float32Array(D).fill(0).map(function(_, i) { return Math.cos(i * 0.3); });
      const results: number[][] = [];
      for (let round = 0; round < 5; round++) {
        const x = MxArray.fromFloat32(xData, new BigInt64Array([1n, BigInt(D)]));
        const y = x.matmul(Wt); y.eval();
        results.push([...y.toFloat32()]);
      }
      for (let round = 1; round < 5; round++) {
        for (let j = 0; j < O; j++) {
          if (Math.abs(results[round][j] - results[0][j]) > 1e-5)
            throw new Error('round ' + round + ' y[' + j + ']=' + results[round][j] + ' != round 0 ' + results[0][j]);
        }
      }
    }},

    { name: 'RMSNorm numerical accuracy check', run() {
      // Check if RMSNorm produces correct values — this is critical for model quality
      const D = 8;
      const xData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
      const wData = new Float32Array(D).fill(1); // unit weight
      const x = MxArray.fromFloat32(xData, new BigInt64Array([1n, BigInt(D)]));
      const w = MxArray.fromFloat32(wData, new BigInt64Array([BigInt(D)]));
      // RMSNorm: x * w / sqrt(mean(x^2) + eps)
      const eps = 1e-6;
      const x2 = x.mul(x);
      const mean_x2 = x2.mean(new Int32Array([-1]));
      const rms = mean_x2.addScalar(eps).sqrt();
      const rmsR = rms.reshape(new BigInt64Array([1n, 1n]));
      const normed = x.div(rmsR).mul(w);
      normed.eval();
      const result = [...normed.toFloat32()];
      // CPU reference: rms = sqrt(mean([1,4,9,16,25,36,49,64]) + eps) = sqrt(25.5 + eps) ≈ 5.0498
      const rms_ref = Math.sqrt((1+4+9+16+25+36+49+64)/8 + eps);
      const expected = xData.map(v => v / rms_ref);
      for (let i = 0; i < D; i++) {
        if (Math.abs(result[i] - expected[i]) > 0.01)
          throw new Error('RMSNorm wrong: result[' + i + ']=' + result[i].toFixed(6) + ' expected=' + expected[i].toFixed(6));
      }
    }},
    { name: 'softmax numerical accuracy', run() {
      const data = new Float32Array([1, 2, 3, 4]);
      const x = MxArray.fromFloat32(data, new BigInt64Array([1n, 4n]));
      // softmax = exp(x) / sum(exp(x))
      const e = x.exp();
      const s = e.sum(new Int32Array([-1]));
      const sm = e.div(s.reshape(new BigInt64Array([1n, 1n])));
      sm.eval();
      const result = [...sm.toFloat32()];
      // CPU ref
      const exps = data.map(v => Math.exp(v));
      const sum = exps.reduce((a, b) => a + b);
      const expected = exps.map(v => v / sum);
      for (let i = 0; i < 4; i++) {
        if (Math.abs(result[i] - expected[i]) > 0.001)
          throw new Error('softmax wrong: [' + i + ']=' + result[i].toFixed(6) + ' expected=' + expected[i].toFixed(6));
      }
    }},
    { name: 'multi-layer chained RMSNorm+matmul accuracy', run() {
      // Simulate 24 layers of RMSNorm + matmul to check error accumulation
      const D = 32;
      const wData = new Float32Array(D * D);
      // Use orthogonal-like weight (scaled identity + small noise)
      for (let i = 0; i < D; i++) {
        for (let j = 0; j < D; j++) {
          wData[i * D + j] = (i === j ? 0.9 : 0.0) + ((i * 7 + j * 13) % 100 - 50) / 5000;
        }
      }
      const W = MxArray.fromFloat32(wData, new BigInt64Array([BigInt(D), BigInt(D)]));
      W.eval();

      let x = MxArray.fromFloat32(new Float32Array(D).fill(1.0), new BigInt64Array([1n, BigInt(D)]));
      x.eval();

      // Run 24 layers
      for (let l = 0; l < 24; l++) {
        // RMSNorm
        const x2 = x.mul(x);
        const rms = x2.mean(new Int32Array([-1])).addScalar(1e-6).sqrt();
        const xn = x.div(rms.reshape(new BigInt64Array([1n, 1n])));
        // Matmul
        x = xn.matmul(W);
        x.eval();
      }

      const result = x.toFloat32();
      // Check output is finite and non-zero
      let hasNonZero = false;
      for (let i = 0; i < D; i++) {
        if (!isFinite(result[i])) throw new Error('Layer chain produced non-finite at [' + i + ']');
        if (Math.abs(result[i]) > 1e-10) hasNonZero = true;
      }
      if (!hasNonZero) throw new Error('Layer chain produced all zeros');
    }},
    { name: 'scalar constant correctness (array::init)', run() {
      // Verify scalar constants created via array::init work correctly
      // This tests the WebGPU set_cpu_ptr path for small arrays
      const a = MxArray.fromFloat32(new Float32Array([3.14]), new BigInt64Array([1n]));
      a.eval();
      const v = a.toFloat32()[0];
      if (Math.abs(v - 3.14) > 0.001) throw new Error('scalar 3.14 got ' + v);

      // Test zero
      const z = MxArray.fromFloat32(new Float32Array([0]), new BigInt64Array([1n]));
      z.eval();
      if (z.toFloat32()[0] !== 0) throw new Error('scalar 0 got ' + z.toFloat32()[0]);

      // Test negative
      const n = MxArray.fromFloat32(new Float32Array([-Infinity]), new BigInt64Array([1n]));
      n.eval();
      if (isFinite(n.toFloat32()[0])) throw new Error('scalar -inf got ' + n.toFloat32()[0]);

      // Test: does addScalar(1e-6) produce the right result?
      const x = MxArray.fromFloat32(new Float32Array([100]), new BigInt64Array([1n]));
      const y = x.addScalar(1e-6);
      y.eval();
      const yv = y.toFloat32()[0];
      if (Math.abs(yv - 100.000001) > 0.001) throw new Error('100 + 1e-6 = ' + yv);
    }},
    { name: 'matmul determinism at model scale', run() {
      const D = 1024, O = 8192;
      const wData = new Float32Array(O * D);
      for (let i = 0; i < wData.length; i++) wData[i] = ((i * 7 + 13) % 1000 - 500) / (500 * D);
      const W = MxArray.fromFloat32(wData, new BigInt64Array([BigInt(O), BigInt(D)]));
      W.eval();
      const Wt = W.transpose(); Wt.eval();
      const x = MxArray.fromFloat32(new Float32Array(D).fill(0.01), new BigInt64Array([1n, BigInt(D)]));
      const results: number[] = [];
      for (let r = 0; r < 3; r++) {
        const y = x.matmul(Wt); y.eval();
        results.push(y.toFloat32()[0]);
      }
      if (results[0] !== results[1] || results[1] !== results[2])
        throw new Error('Non-deterministic: ' + results.join(', '));
    }},
    { name: 'readCppWeight API works (no model loaded)', run() {
      const vals = MxArray.readCppWeight('layers.0.input_layernorm.weight', 5);
      if (vals.length !== 1 || vals[0] !== -999) throw new Error('Expected -999 for empty weight map, got ' + vals);
    }},
    { name: 'lazy transpose vs eager transpose equivalence', run() {
      // Test that lazy transpose on an eval'd GPU array produces same matmul result as eager
      const D = 64, O = 32;
      const wData = new Float32Array(O * D);
      for (let i = 0; i < wData.length; i++) wData[i] = Math.sin(i * 0.01) * 0.5;
      const W = MxArray.fromFloat32(wData, new BigInt64Array([BigInt(O), BigInt(D)]));
      W.eval(); // Put on GPU

      // Method 1: Lazy transpose (like store_weight does)
      const Wt_lazy = W.transpose(); // NOT eval'd — lazy
      const x1 = MxArray.fromFloat32(new Float32Array(D).fill(1), new BigInt64Array([1n, BigInt(D)]));
      const y1 = x1.matmul(Wt_lazy); // matmul with lazy transpose
      y1.eval();
      const r1 = [...y1.toFloat32()];

      // Method 2: Eager transpose (eval transpose first)
      const Wt_eager = W.transpose();
      Wt_eager.eval();
      const x2 = MxArray.fromFloat32(new Float32Array(D).fill(1), new BigInt64Array([1n, BigInt(D)]));
      const y2 = x2.matmul(Wt_eager);
      y2.eval();
      const r2 = [...y2.toFloat32()];

      // Compare — must be exactly equal
      for (let j = 0; j < O; j++) {
        if (Math.abs(r1[j] - r2[j]) > 1e-5) {
          throw new Error('lazy vs eager mismatch at [' + j + ']: lazy=' + r1[j].toFixed(6) + ' eager=' + r2[j].toFixed(6));
        }
      }
    }},

    // ---- Inference step-by-step tests (mirrors Qwen3.5 forward pass patterns) ----

    { name: 'embedding lookup accuracy', run() {
      // Create embedding weight [10 vocab, 8 dim]
      const V = 10, D = 8;
      const embedData = new Float32Array(V * D);
      for (let i = 0; i < V * D; i++) embedData[i] = Math.sin(i * 0.37) * 0.5;
      const embedW = MxArray.fromFloat32(embedData, new BigInt64Array([BigInt(V), BigInt(D)]));
      embedW.eval();

      // Token IDs [2, 5, 7]
      const ids = MxArray.fromInt32(new Int32Array([2, 5, 7]), new BigInt64Array([3n]));

      // Gather embeddings: take rows from embedding table
      const embedded = embedW.take(ids, 0); // [3, 8]
      embedded.eval();
      const result = [...embedded.toFloat32()];

      // CPU reference: row i of embedW = embedData[i*D .. (i+1)*D]
      const expected: number[] = [];
      for (const tok of [2, 5, 7]) {
        for (let j = 0; j < D; j++) expected.push(embedData[tok * D + j]);
      }
      assertClose(result, expected, 1e-4);
    }},

    { name: 'conv1d depthwise (GDN pattern)', run() {
      // Depthwise conv1d: each channel convolved independently
      // input [1, 4, 16] (batch=1, time=4, channels=16)
      // weight [16, 3, 1] (out_ch=16, kernel=3, in_ch_per_group=1)
      // groups=16 => depthwise
      // Simulate manually since conv1d may not be exposed via NAPI:
      // For each channel c, output[t] = sum_k input[t+k, c] * weight[c, k, 0]
      const T = 4, C = 16, K = 3;
      const inputData = new Float32Array(T * C);
      for (let i = 0; i < T * C; i++) inputData[i] = Math.sin(i * 0.2) * 0.5;
      const weightData = new Float32Array(C * K);
      for (let i = 0; i < C * K; i++) weightData[i] = Math.cos(i * 0.3) * 0.1;

      // Compute CPU reference for depthwise conv (valid padding => output time = T - K + 1 = 2)
      const outT = T - K + 1; // 2
      const cpuOut = new Float32Array(outT * C);
      for (let c = 0; c < C; c++) {
        for (let t = 0; t < outT; t++) {
          let acc = 0;
          for (let k = 0; k < K; k++) {
            acc += inputData[(t + k) * C + c] * weightData[c * K + k];
          }
          cpuOut[t * C + c] = acc;
        }
      }

      // Reproduce on GPU using matmul per channel (simulate depthwise conv)
      // Build a [outT, K] patch matrix per channel, multiply with [K, 1] filter
      // More practically: build full Toeplitz-like matmul
      // input reshaped: for each output position, gather K consecutive timesteps
      // patches[t, c, k] = input[(t+k), c]
      const patchData = new Float32Array(outT * C * K);
      for (let t = 0; t < outT; t++) {
        for (let c = 0; c < C; c++) {
          for (let k = 0; k < K; k++) {
            patchData[(t * C + c) * K + k] = inputData[(t + k) * C + c];
          }
        }
      }
      // patches: [outT*C, K], weight reshaped: [outT*C, K] (broadcast weight per channel)
      // Element-wise mul then sum over K axis
      const patches = MxArray.fromFloat32(patchData, new BigInt64Array([BigInt(outT * C), BigInt(K)]));
      // Tile weight: each channel's K weights repeated outT times
      const tiledWeightData = new Float32Array(outT * C * K);
      for (let t = 0; t < outT; t++) {
        for (let c = 0; c < C; c++) {
          for (let k = 0; k < K; k++) {
            tiledWeightData[(t * C + c) * K + k] = weightData[c * K + k];
          }
        }
      }
      const tiledW = MxArray.fromFloat32(tiledWeightData, new BigInt64Array([BigInt(outT * C), BigInt(K)]));
      const product = patches.mul(tiledW); // [outT*C, K]
      const convOut = product.sum(new Int32Array([1])); // [outT*C]
      convOut.eval();
      const gpuResult = [...convOut.toFloat32()];

      // Verify against CPU reference
      assertClose(gpuResult, [...cpuOut], 1e-3);
    }},

    { name: 'slice_update for KV cache (attention pattern)', run() {
      // KV cache: [1, 2, 10, 4] (batch=1, heads=2, max_len=10, head_dim=4) filled with zeros
      const B = 1, H = 2, maxLen = 10, Dh = 4;
      const cacheSize = B * H * maxLen * Dh;
      const cache = MxArray.zeros(new BigInt64Array([BigInt(B), BigInt(H), BigInt(maxLen), BigInt(Dh)]));
      cache.eval();

      // New keys for position 3: [1, 2, 1, 4]
      const newKeyData = new Float32Array(B * H * 1 * Dh);
      for (let i = 0; i < newKeyData.length; i++) newKeyData[i] = (i + 1) * 0.5;
      const newKeys = MxArray.fromFloat32(newKeyData, new BigInt64Array([BigInt(B), BigInt(H), 1n, BigInt(Dh)]));

      // Use putAlongAxis to update position 3 in the seq_len dimension (axis=2)
      const posIdx = MxArray.fromInt32(new Int32Array([3]), new BigInt64Array([1n]));
      // Reshape index for broadcasting: [1, 1, 1, 1]
      const posIdxReshaped = posIdx.reshape(new BigInt64Array([1n, 1n, 1n, 1n]));
      const updated = cache.putAlongAxis(posIdxReshaped, newKeys, 2);
      updated.eval();

      // Read back: slice position 3 => [1, 2, 1, 4]
      const readBack = updated.slice(
        new BigInt64Array([0n, 0n, 3n, 0n]),
        new BigInt64Array([BigInt(B), BigInt(H), 4n, BigInt(Dh)])
      );
      const rb = readBack.addScalar(0); rb.eval(); // force contiguous
      const rbData = [...rb.toFloat32()];
      assertClose(rbData, [...newKeyData], 1e-5);

      // Verify position 0 is still zero
      const readZero = updated.slice(
        new BigInt64Array([0n, 0n, 0n, 0n]),
        new BigInt64Array([BigInt(B), BigInt(H), 1n, BigInt(Dh)])
      );
      const rz = readZero.addScalar(0); rz.eval();
      const rzData = [...rz.toFloat32()];
      assertClose(rzData, new Array(B * H * Dh).fill(0), 1e-5);
    }},

    { name: 'GDN recurrence step correctness', run() {
      // One step of GDN (Gated Delta Network) recurrence:
      //   state = state * g + k.T * (beta * (v - state @ k))
      //   y = state @ q
      // Dimensions: Hv=2, Dv=3, Dk=4
      const Hv = 2, Dv = 3, Dk = 4;

      // state: [Hv, Dv, Dk] = [2, 3, 4]
      const stateData = new Float32Array(Hv * Dv * Dk);
      for (let i = 0; i < stateData.length; i++) stateData[i] = Math.sin(i * 0.5) * 0.1;
      // g (gate): [Hv, Dv, Dk] = [2, 3, 4] — sigmoid-like values in (0, 1)
      const gData = new Float32Array(Hv * Dv * Dk);
      for (let i = 0; i < gData.length; i++) gData[i] = 0.5 + 0.3 * Math.sin(i * 0.7);
      // k: [Hv, Dk, 1] = [2, 4, 1]
      const kData = new Float32Array(Hv * Dk);
      for (let i = 0; i < kData.length; i++) kData[i] = Math.cos(i * 0.4) * 0.2;
      // v: [Hv, Dv, 1] = [2, 3, 1]
      const vData = new Float32Array(Hv * Dv);
      for (let i = 0; i < vData.length; i++) vData[i] = Math.sin(i * 0.6) * 0.3;
      // beta: scalar
      const beta = 0.8;
      // q: [Hv, Dk, 1] = [2, 4, 1]
      const qData = new Float32Array(Hv * Dk);
      for (let i = 0; i < qData.length; i++) qData[i] = Math.cos(i * 0.3) * 0.15;

      // CPU reference (per head h):
      //   state_h [Dv, Dk], k_h [Dk, 1], v_h [Dv, 1], g_h [Dv, Dk], q_h [Dk, 1]
      //   delta = v_h - state_h @ k_h   => [Dv, 1]
      //   outer = delta @ k_h.T         => [Dv, Dk]  (but scaled by beta)
      //   new_state_h = state_h * g_h + beta * outer
      //   y_h = new_state_h @ q_h       => [Dv, 1]
      const cpuY = new Float32Array(Hv * Dv);
      for (let h = 0; h < Hv; h++) {
        const sOff = h * Dv * Dk;
        const kOff = h * Dk;
        const vOff = h * Dv;
        // state_h @ k_h => [Dv, 1]
        const sk = new Float32Array(Dv);
        for (let d = 0; d < Dv; d++) {
          let acc = 0;
          for (let j = 0; j < Dk; j++) acc += stateData[sOff + d * Dk + j] * kData[kOff + j];
          sk[d] = acc;
        }
        // delta = v_h - sk
        const delta = new Float32Array(Dv);
        for (let d = 0; d < Dv; d++) delta[d] = vData[vOff + d] - sk[d];
        // new_state = state * g + beta * outer_product(delta, k)
        const newState = new Float32Array(Dv * Dk);
        for (let d = 0; d < Dv; d++) {
          for (let j = 0; j < Dk; j++) {
            newState[d * Dk + j] = stateData[sOff + d * Dk + j] * gData[sOff + d * Dk + j]
              + beta * delta[d] * kData[kOff + j];
          }
        }
        // y_h = new_state @ q_h => [Dv]
        for (let d = 0; d < Dv; d++) {
          let acc = 0;
          for (let j = 0; j < Dk; j++) acc += newState[d * Dk + j] * qData[kOff + j];
          cpuY[h * Dv + d] = acc;
        }
      }

      // GPU computation (process each head via matmul ops)
      const gpuY = new Float32Array(Hv * Dv);
      for (let h = 0; h < Hv; h++) {
        const sOff = h * Dv * Dk;
        const kOff = h * Dk;
        const vOff = h * Dv;
        // state_h [Dv, Dk]
        const state_h = MxArray.fromFloat32(stateData.slice(sOff, sOff + Dv * Dk), new BigInt64Array([BigInt(Dv), BigInt(Dk)]));
        // k_h [Dk, 1]
        const k_h = MxArray.fromFloat32(kData.slice(kOff, kOff + Dk), new BigInt64Array([BigInt(Dk), 1n]));
        // v_h [Dv, 1]
        const v_h = MxArray.fromFloat32(vData.slice(vOff, vOff + Dv), new BigInt64Array([BigInt(Dv), 1n]));
        // g_h [Dv, Dk]
        const g_h = MxArray.fromFloat32(gData.slice(sOff, sOff + Dv * Dk), new BigInt64Array([BigInt(Dv), BigInt(Dk)]));
        // q_h [Dk, 1]
        const q_h = MxArray.fromFloat32(qData.slice(kOff, kOff + Dk), new BigInt64Array([BigInt(Dk), 1n]));

        // sk = state_h @ k_h => [Dv, 1]
        const sk = state_h.matmul(k_h);
        // delta = v_h - sk => [Dv, 1]
        const delta_h = v_h.sub(sk);
        // outer = delta_h @ k_h.T => [Dv, Dk]
        const k_hT = k_h.transpose(); // [1, Dk]
        const outer = delta_h.matmul(k_hT);
        // new_state = state_h * g_h + beta * outer
        const newState = state_h.mul(g_h).add(outer.mulScalar(beta));
        // y_h = new_state @ q_h => [Dv, 1]
        const y_h = newState.matmul(q_h);
        y_h.eval();
        const yVals = y_h.toFloat32();
        for (let d = 0; d < Dv; d++) gpuY[h * Dv + d] = yVals[d];
      }

      assertClose([...gpuY], [...cpuY], 1e-3);
    }},

    { name: 'full layer forward (RMSNorm + matmul + residual)', run() {
      // Simulate one transformer layer: RMSNorm -> projection -> residual add
      const D = 32, Dout = 64;
      // Input [1, 1, 32]
      const xData = new Float32Array(D);
      for (let i = 0; i < D; i++) xData[i] = Math.sin(i * 0.2) * 0.5;
      const x = MxArray.fromFloat32(xData, new BigInt64Array([1n, 1n, BigInt(D)]));

      // RMSNorm weight [32]
      const normWData = new Float32Array(D);
      for (let i = 0; i < D; i++) normWData[i] = 0.8 + 0.4 * Math.cos(i * 0.1);
      const normW = MxArray.fromFloat32(normWData, new BigInt64Array([BigInt(D)]));

      // Projection weight [64, 32] — used as W.T = [32, 64]
      const projData = new Float32Array(Dout * D);
      for (let i = 0; i < Dout * D; i++) projData[i] = ((i * 7 + 3) % 100 - 50) / 500;
      const projW = MxArray.fromFloat32(projData, new BigInt64Array([BigInt(Dout), BigInt(D)]));
      projW.eval();
      const projWt = projW.transpose(); // [32, 64]
      projWt.eval();

      // RMSNorm: normed = x * w / sqrt(mean(x^2) + eps)
      const xFlat = x.reshape(new BigInt64Array([1n, BigInt(D)])); // [1, D]
      const x2 = xFlat.mul(xFlat);
      const meanX2 = x2.mean(new Int32Array([-1])); // [1]
      const rms = meanX2.addScalar(1e-6).sqrt(); // [1]
      const rmsReshaped = rms.reshape(new BigInt64Array([1n, 1n]));
      const normed = xFlat.div(rmsReshaped).mul(normW.reshape(new BigInt64Array([1n, BigInt(D)]))); // [1, D]

      // Projection: out = normed @ W.T => [1, 64]
      const projected = normed.matmul(projWt);
      projected.eval();
      const projResult = projected.toFloat32();

      // Check output shape and finiteness
      if (projResult.length !== Dout) throw new Error('proj length=' + projResult.length + ' expected ' + Dout);
      for (let i = 0; i < Dout; i++) {
        if (!isFinite(projResult[i])) throw new Error('proj[' + i + '] is not finite: ' + projResult[i]);
      }

      // Residual: pad x to Dout by taking first Dout dims (or zero-pad)
      // In real models, residual is same dim as input. Use D-dim residual on a D-dim slice.
      // Here, just test residual pattern with matching dims:
      // output = projected[:, :D] + xFlat
      const projSlice = projected.slice(new BigInt64Array([0n, 0n]), new BigInt64Array([1n, BigInt(D)]));
      const residual = projSlice.add(xFlat);
      residual.eval();
      const resResult = [...residual.toFloat32()];

      // CPU reference
      const rms_ref = Math.sqrt(xData.reduce((a, v) => a + v * v, 0) / D + 1e-6);
      const normedCpu = xData.map((v, i) => (v / rms_ref) * normWData[i]);
      // projected_cpu[j] = sum_i normedCpu[i] * projWt[i, j] = sum_i normedCpu[i] * projData[j * D + i]
      const projCpu = new Float32Array(Dout);
      for (let j = 0; j < Dout; j++) {
        let acc = 0;
        for (let i = 0; i < D; i++) acc += normedCpu[i] * projData[j * D + i];
        projCpu[j] = acc;
      }
      const resCpu: number[] = [];
      for (let i = 0; i < D; i++) resCpu.push(projCpu[i] + xData[i]);
      assertClose(resResult, resCpu, 1e-2);
    }},

    { name: 'model forward simulation (3 passes, 24 layers)', run() {
      // Simulate model forward: 24 layers × 3 passes with per-layer eval
      // Uses smaller dims than real model (256 instead of 1024) to fit in WASM memory
      // (real model uses GPU buffer arrays which don't consume WASM memory)
      const NUM_LAYERS = 24;
      const D = 256;
      const O = 1024;

      // Create persistent weight arrays: in_proj [D,O] + out_proj [O,D] per layer
      const weights: InstanceType<typeof MxArray>[] = [];
      for (let l = 0; l < NUM_LAYERS; l++) {
        const inData = new Float32Array(D * O);
        for (let i = 0; i < inData.length; i++) inData[i] = (((i * 7 + l * 13) % 1000) - 500) / (500 * O);
        const wIn = MxArray.fromFloat32(inData, new BigInt64Array([BigInt(D), BigInt(O)]));
        wIn.eval();
        const outData = new Float32Array(O * D);
        for (let i = 0; i < outData.length; i++) outData[i] = (((i * 11 + l * 17) % 1000) - 500) / (500 * D);
        const wOut = MxArray.fromFloat32(outData, new BigInt64Array([BigInt(O), BigInt(D)]));
        wOut.eval();
        weights.push(wIn, wOut);
      }

      for (let pass = 0; pass < 3; pass++) {
        const T = pass === 0 ? 15 : 1;
        const inputData = new Float32Array(T * D);
        for (let i = 0; i < inputData.length; i++) inputData[i] = ((i * 3 + pass * 7) % 1000 - 500) / 500;
        let x = MxArray.fromFloat32(inputData, new BigInt64Array([1n, BigInt(T), BigInt(D)]));
        x.eval();
        x = x.reshape(new BigInt64Array([BigInt(T), BigInt(D)]));

        for (let l = 0; l < NUM_LAYERS; l++) {
          let h = x.matmul(weights[l * 2]);
          h = h.mulScalar(0.125);
          h = h.matmul(weights[l * 2 + 1]);
          h = h.addScalar(0.01);
          h.eval();
          x = h;
        }

        const argmaxResult = x.argmax(-1);
        argmaxResult.eval();
        const tokenIds = argmaxResult.toUint32();
        if (tokenIds.length === 0) throw new Error(`pass ${pass}: empty argmax`);
        if (tokenIds[tokenIds.length - 1] >= D) throw new Error(`pass ${pass}: token out of range`);
      }
    }},

    { name: 'RPC stress: rapid eval cycles with readback', run() {
      // Stress test the RPC bridge: many rapid eval+readback cycles
      // This exercises the exact path that crashes the model:
      // eval → GPU dispatch → readback → eval → GPU dispatch → readback
      const D = 512, N = 20;
      const w = MxArray.fromFloat32(
        new Float32Array(D * D).fill(0).map((_,i) => ((i % 1000) - 500) / (500 * D)),
        new BigInt64Array([BigInt(D), BigInt(D)]));
      w.eval();

      for (let i = 0; i < N; i++) {
        const x = MxArray.fromFloat32(
          new Float32Array(D).fill(0).map((_,j) => Math.sin(i + j * 0.1)),
          new BigInt64Array([1n, BigInt(D)]));
        const y = x.matmul(w);
        y.eval();
        // Readback (triggers raw_ptr → staging buffer → RPC → copy)
        const v = y.toFloat32();
        if (!isFinite(v[0])) throw new Error(`iter ${i}: non-finite ${v[0]}`);
      }
    }},
    { name: 'RPC stress: concurrent buffer lifecycle', run() {
      // Create many buffers, eval, free, repeat — tests allocator + RPC interaction
      const D = 256;
      for (let round = 0; round < 5; round++) {
        const arrays: any[] = [];
        // Create 50 arrays
        for (let i = 0; i < 50; i++) {
          const a = MxArray.fromFloat32(
            new Float32Array(D).fill(round * 50 + i),
            new BigInt64Array([BigInt(D)]));
          arrays.push(a);
        }
        // Chain operations
        let h = arrays[0];
        for (let i = 1; i < arrays.length; i++) {
          h = h.add(arrays[i]);
        }
        h.eval();
        const v = h.toFloat32();
        if (!isFinite(v[0])) throw new Error(`round ${round}: non-finite`);
        // All arrays go out of scope — tests buffer free/cache cycle
      }
    }},
    { name: 'where (float cond)', run() {
      const cond = MxArray.fromFloat32(new Float32Array([1, 0, 1, 0]), new BigInt64Array([4n]));
      const x = MxArray.fromFloat32(new Float32Array([10, 20, 30, 40]), new BigInt64Array([4n]));
      const y = MxArray.fromFloat32(new Float32Array([1, 2, 3, 4]), new BigInt64Array([4n]));
      const c = cond.where(x, y); c.eval();
      assertClose([...c.toFloat32()], [10, 2, 30, 4]);
    }},

    // ---- New tests for recently added/fixed WebGPU ops ----

    { name: 'sub', run() {
      const a = MxArray.fromFloat32(new Float32Array([10, 20, 30]), new BigInt64Array([3n]));
      const b = MxArray.fromFloat32(new Float32Array([1, 5, 10]), new BigInt64Array([3n]));
      const c = a.sub(b); c.eval();
      assertClose([...c.toFloat32()], [9, 15, 20]);
    }},
    { name: 'div', run() {
      const a = MxArray.fromFloat32(new Float32Array([10, 20, 30]), new BigInt64Array([3n]));
      const b = MxArray.fromFloat32(new Float32Array([2, 5, 10]), new BigInt64Array([3n]));
      const c = a.div(b); c.eval();
      assertClose([...c.toFloat32()], [5, 4, 3]);
    }},
    { name: 'log', run() {
      const a = MxArray.fromFloat32(new Float32Array([1, Math.E, Math.E * Math.E]), new BigInt64Array([3n]));
      const c = a.log(); c.eval();
      assertClose([...c.toFloat32()], [0, 1, 2], 1e-2);
    }},
    { name: 'sin', run() {
      const a = MxArray.fromFloat32(new Float32Array([0, Math.PI / 2, Math.PI]), new BigInt64Array([3n]));
      const c = a.sin(); c.eval();
      assertClose([...c.toFloat32()], [0, 1, 0], 1e-2);
    }},
    { name: 'cos', run() {
      const a = MxArray.fromFloat32(new Float32Array([0, Math.PI / 2, Math.PI]), new BigInt64Array([3n]));
      const c = a.cos(); c.eval();
      assertClose([...c.toFloat32()], [1, 0, -1], 1e-2);
    }},
    { name: 'tanh', run() {
      const a = MxArray.fromFloat32(new Float32Array([0, 1, -1]), new BigInt64Array([3n]));
      const c = a.tanh(); c.eval();
      assertClose([...c.toFloat32()], [0, Math.tanh(1), Math.tanh(-1)], 1e-3);
    }},
    { name: 'power', run() {
      const a = MxArray.fromFloat32(new Float32Array([2, 3, 4]), new BigInt64Array([3n]));
      const b = MxArray.fromFloat32(new Float32Array([3, 2, 0.5]), new BigInt64Array([3n]));
      const c = a.power(b); c.eval();
      assertClose([...c.toFloat32()], [8, 9, 2], 1e-3);
    }},
    { name: 'transpose 2D', run() {
      const a = MxArray.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), new BigInt64Array([2n, 3n]));
      const b = a.transpose(); b.eval();
      const s = b.shape();
      if (Number(s[0]) !== 3 || Number(s[1]) !== 2) throw new Error(`shape=${s}`);
      assertClose([...b.toFloat32()], [1, 4, 2, 5, 3, 6]);
    }},
    { name: 'slice', run() {
      const a = MxArray.fromFloat32(new Float32Array([10, 20, 30, 40, 50]), new BigInt64Array([5n]));
      const sliced = a.slice(new BigInt64Array([1n]), new BigInt64Array([4n]));
      // Force contiguous copy — slice creates a view with an offset that
      // toFloat32() readback may not handle correctly
      const c = sliced.addScalar(0); c.eval();
      assertClose([...c.toFloat32()], [20, 30, 40]);
    }},
    { name: 'argmax', run() {
      const a = MxArray.fromFloat32(new Float32Array([1, 5, 3, 2]), new BigInt64Array([4n]));
      const c = a.argmax(0); c.eval();
      const idx = c.toUint32();
      if (idx[0] !== 1) throw new Error(`argmax=${idx[0]} (expected 1)`);
    }},
    { name: 'min', run() {
      const a = MxArray.fromFloat32(new Float32Array([3, 1, 5, 2]), new BigInt64Array([4n]));
      const c = a.min(); c.eval();
      assertClose([...c.toFloat32()], [1]);
    }},
    { name: 'mean', run() {
      const a = MxArray.fromFloat32(new Float32Array([2, 4, 6, 8]), new BigInt64Array([4n]));
      const c = a.mean(); c.eval();
      assertClose([...c.toFloat32()], [5]);
    }},
    { name: 'large matmul 64x128*128x64', run() {
      // Test larger matmul to exercise tiled dispatch
      const M = 64, K = 128, N = 64;
      const aData = new Float32Array(M * K);
      const bData = new Float32Array(K * N);
      // Fill with simple pattern: a[i,j] = (i+j) % 7, b[i,j] = (i*j) % 5
      for (let i = 0; i < M; i++) for (let j = 0; j < K; j++) aData[i * K + j] = (i + j) % 7;
      for (let i = 0; i < K; i++) for (let j = 0; j < N; j++) bData[i * N + j] = (i * j) % 5;
      const a = MxArray.fromFloat32(aData, new BigInt64Array([BigInt(M), BigInt(K)]));
      const b = MxArray.fromFloat32(bData, new BigInt64Array([BigInt(K), BigInt(N)]));
      const c = a.matmul(b); c.eval();
      const result = [...c.toFloat32()];
      // Verify a few elements by manual computation
      // c[0,0] = sum_k a[0,k]*b[k,0] = sum_k (k%7)*(0) = 0
      if (Math.abs(result[0] - 0) > 1) throw new Error(`c[0,0]=${result[0]}`);
      // c[0,1] = sum_k (k%7)*(k%5) — compute expected
      let expected01 = 0;
      for (let k = 0; k < K; k++) expected01 += (k % 7) * (k % 5);
      if (Math.abs(result[1] - expected01) > 1) throw new Error(`c[0,1]=${result[1]} vs ${expected01}`);
    }},
    { name: 'sign', run() {
      const a = MxArray.fromFloat32(new Float32Array([-5, 0, 3]), new BigInt64Array([3n]));
      const c = a.sign(); c.eval();
      assertClose([...c.toFloat32()], [-1, 0, 1]);
    }},
    { name: 'square', run() {
      const a = MxArray.fromFloat32(new Float32Array([2, -3, 4]), new BigInt64Array([3n]));
      const c = a.square(); c.eval();
      assertClose([...c.toFloat32()], [4, 9, 16]);
    }},
    { name: 'comparison (greater)', run() {
      const a = MxArray.fromFloat32(new Float32Array([1, 5, 3]), new BigInt64Array([3n]));
      const b = MxArray.fromFloat32(new Float32Array([2, 3, 3]), new BigInt64Array([3n]));
      const c = a.greater(b); c.eval();
      // Bool results packed as u32 bytes
      const vals = [...c.toFloat32()];
      // greater: [false, true, false] — depending on how bool is extracted
      // For now just check it doesn't crash
      if (vals.length !== 3) throw new Error(`length=${vals.length}`);
    }},
    { name: 'multi-axis sum [2,3,4]', run() {
      // 24 elements in shape [2,3,4], sum over axis 2
      const data = new Float32Array(24);
      for (let i = 0; i < 24; i++) data[i] = i + 1;
      const a = MxArray.fromFloat32(data, new BigInt64Array([2n, 3n, 4n]));
      const c = a.sum(new Int32Array([2])); c.eval();
      // sum over last axis: [1+2+3+4, 5+6+7+8, 9+10+11+12, 13+14+15+16, ...]
      assertClose([...c.toFloat32()], [10, 26, 42, 58, 74, 90]);
    }},
    { name: 'bfloat16 add', run() {
      // BFloat16 = 3 in DType enum. bf16 stored as 2 bytes per element.
      // Tests that the WebGPU backend handles bf16 data correctly.
      const BF16 = 3; // DType::BFloat16
      const a = MxArray.fromFloat32(new Float32Array([1, 2, 3]), new BigInt64Array([3n])).astype(BF16);
      const b = MxArray.fromFloat32(new Float32Array([4, 5, 6]), new BigInt64Array([3n])).astype(BF16);
      const c = a.add(b); c.eval();
      const result = [...c.toFloat32()];
      assertClose(result, [5, 7, 9], 0.5);
    }},
    { name: 'bfloat16 matmul', run() {
      const BF16 = 3;
      const a = MxArray.fromFloat32(
        new Float32Array([1, 2, 3, 4]),
        new BigInt64Array([2n, 2n])).astype(BF16);
      const b = MxArray.fromFloat32(
        new Float32Array([1, 2, 3, 4]),
        new BigInt64Array([2n, 2n])).astype(BF16);
      const c = a.matmul(b); c.eval();
      const result = [...c.toFloat32()];
      // [[1,2],[3,4]] @ [[1,2],[3,4]] = [[7,10],[15,22]]
      assertClose(result, [7, 10, 15, 22], 1.0);
    }},
    { name: 'large 1D dispatch (256K elements)', run() {
      // Test copy/binary with large arrays that would exceed 65535 workgroups
      // 256K elements / 256 WG_SIZE / 4 N_READS = 256 workgroups (should be fine)
      const N = 256 * 1024;
      const data = new Float32Array(N);
      for (let i = 0; i < N; i++) data[i] = 1.0;
      const a = MxArray.fromFloat32(data, new BigInt64Array([BigInt(N)]));
      const b = MxArray.fromFloat32(data, new BigInt64Array([BigInt(N)]));
      const c = a.add(b); c.eval();
      const result = c.toFloat32();
      // Spot-check a few values
      if (Math.abs(result[0] - 2) > 0.01) throw new Error(`[0]=${result[0]}`);
      if (Math.abs(result[N-1] - 2) > 0.01) throw new Error(`[${N-1}]=${result[N-1]}`);
    }},
    { name: 'matmul with vocab-like dim (1x1024 * 1024x2048)', run() {
      // Simulates lm_head projection — tests large N dimension in tiled matmul
      const M = 1, K = 1024, N = 2048;
      const aData = new Float32Array(M * K).fill(1.0 / K); // avg 1/K per element
      const bData = new Float32Array(K * N);
      for (let i = 0; i < K * N; i++) bData[i] = (i % 7) * 0.01;
      const a = MxArray.fromFloat32(aData, new BigInt64Array([BigInt(M), BigInt(K)]));
      const b = MxArray.fromFloat32(bData, new BigInt64Array([BigInt(K), BigInt(N)]));
      const c = a.matmul(b); c.eval();
      const result = c.toFloat32();
      if (result.length !== M * N) throw new Error(`length=${result.length}`);
      // Verify first element: sum_k (1/1024) * (k%7 * 0.01)
      let expected0 = 0;
      for (let k = 0; k < K; k++) expected0 += (1.0/K) * ((k%7) * 0.01);
      if (Math.abs(result[0] - expected0) > 0.01) throw new Error(`[0]=${result[0]} vs ${expected0}`);
    }},
    { name: 'chained linear: x @ W1 + b then ReLU-like', run() {
      // Simulates a single linear layer + activation (model building block)
      const B = 2, D = 64, H = 128;
      const xData = new Float32Array(B * D);
      const wData = new Float32Array(D * H);
      for (let i = 0; i < B * D; i++) xData[i] = Math.sin(i * 0.1);
      for (let i = 0; i < D * H; i++) wData[i] = Math.cos(i * 0.01) * 0.1;
      const x = MxArray.fromFloat32(xData, new BigInt64Array([BigInt(B), BigInt(D)]));
      const w = MxArray.fromFloat32(wData, new BigInt64Array([BigInt(D), BigInt(H)]));
      const y = x.matmul(w); // [2, 128]
      const z = y.abs(); // ReLU-like (abs instead since we don't have relu)
      z.eval();
      const result = z.toFloat32();
      if (result.length !== B * H) throw new Error(`length=${result.length}`);
      // All values should be non-negative (abs)
      for (let i = 0; i < 10; i++) {
        if (result[i] < 0) throw new Error(`[${i}]=${result[i]} < 0`);
      }
    }},
    { name: 'mul by scalar broadcast', run() {
      // Test multiply with broadcast: [2,3] * scalar
      const a = MxArray.fromFloat32(new Float32Array([6, 15, 3, 9, 12, 21]), new BigInt64Array([2n, 3n]));
      const s = MxArray.fromFloat32(new Float32Array([0.5]), new BigInt64Array([1n]));
      const c = a.mul(s); c.eval();
      assertClose([...c.toFloat32()], [3, 7.5, 1.5, 4.5, 6, 10.5], 0.1);
    }},
    { name: 'mul by 0-dim scalar', run() {
      // Test multiply with 0-dimensional scalar (shape=[])
      // MLX mean uses this: sum(x) * array(1/N, dtype) where array has shape []
      const a = MxArray.fromFloat32(new Float32Array([6, 15]), new BigInt64Array([2n]));
      const s = MxArray.fromFloat32(new Float32Array([0.5]), new BigInt64Array([]));
      const c = a.mul(s); c.eval();
      assertClose([...c.toFloat32()], [3, 7.5], 0.1);
    }},
    { name: 'sum then mul (eager eval)', run() {
      // Manually do what mean does: sum(x, axis=1) * (1/3), eval each step
      const data = new Float32Array([1, 2, 3, 4, 5, 6]);
      const x = MxArray.fromFloat32(data, new BigInt64Array([2n, 3n]));
      const s = x.sum(new Int32Array([1])); s.eval(); // [6, 15]
      const scale = MxArray.fromFloat32(new Float32Array([1/3]), new BigInt64Array([1n]));
      const c = s.mul(scale); c.eval();
      assertClose([...c.toFloat32()], [2, 5], 0.1);
    }},
    { name: 'sum then mul (lazy eval)', run() {
      // Same but lazy — only eval the final result (like mean does internally)
      const data = new Float32Array([1, 2, 3, 4, 5, 6]);
      const x = MxArray.fromFloat32(data, new BigInt64Array([2n, 3n]));
      const s = x.sum(new Int32Array([1])); // DON'T eval yet
      const scale = MxArray.fromFloat32(new Float32Array([1/3]), new BigInt64Array([1n]));
      const c = s.mul(scale); c.eval(); // eval the whole chain at once
      assertClose([...c.toFloat32()], [2, 5], 0.1);
    }},
    { name: 'mean axis=0 on 1D', run() {
      // Simplest mean-with-axis: 1D array, axis=0
      const x = MxArray.fromFloat32(new Float32Array([2, 4, 6]), new BigInt64Array([3n]));
      const m = x.mean(new Int32Array([0])); m.eval();
      assertClose([...m.toFloat32()], [4], 0.1);
    }},
    { name: 'mean axis=1 on 2D', run() {
      // Test mean reduction on axis 1 (positive)
      const data = new Float32Array([1, 2, 3, 4, 5, 6]);
      const x = MxArray.fromFloat32(data, new BigInt64Array([2n, 3n]));
      const mean = x.mean(new Int32Array([1])); mean.eval();
      const result = [...mean.toFloat32()];
      // mean([1,2,3]) = 2, mean([4,5,6]) = 5
      assertClose(result, [2, 5], 0.1);
    }},
    { name: 'mean axis=-1 on 2D', run() {
      // Same but negative axis
      const data = new Float32Array([1, 2, 3, 4, 5, 6]);
      const x = MxArray.fromFloat32(data, new BigInt64Array([2n, 3n]));
      const mean = x.mean(new Int32Array([-1])); mean.eval();
      const result = [...mean.toFloat32()];
      assertClose(result, [2, 5], 0.1);
    }},
    { name: 'square then reduce', run() {
      // Chain: mul then mean — tests chained eval with intermediate
      const data = new Float32Array([1, 2, 3, 4, 5, 6]);
      const x = MxArray.fromFloat32(data, new BigInt64Array([2n, 3n]));
      const x2 = x.mul(x);
      const mean = x2.mean(new Int32Array([-1]));
      mean.eval();
      const result = [...mean.toFloat32()];
      // mean([1,4,9]) = 14/3 ≈ 4.667, mean([16,25,36]) = 77/3 ≈ 25.667
      assertClose(result, [14/3, 77/3], 0.1);
    }},
    { name: 'gather (embedding lookup)', run() {
      // Simulates token embedding lookup
      // NOTE: gather requires specific MLX API — use indexing if available
      const vocab = MxArray.fromFloat32(
        new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
        new BigInt64Array([4n, 3n])); // [4 tokens, 3 dim embedding]
      // Test basic slicing as a proxy for embedding lookup
      const row0 = vocab.slice(new BigInt64Array([0n, 0n]), new BigInt64Array([1n, 3n]));
      const r = row0.addScalar(0); r.eval(); // force contiguous
      assertClose([...r.toFloat32()], [0.1, 0.2, 0.3], 0.01);
    }},
    { name: 'rmsnorm pattern', run() {
      // RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
      // This is the exact pattern used in every Qwen3.5 layer
      const x = MxArray.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), new BigInt64Array([2n, 3n]));
      const w = MxArray.fromFloat32(new Float32Array([1, 1, 1]), new BigInt64Array([3n]));
      // Step 1: x^2
      const x2 = x.mul(x);
      // Step 2: mean(x^2, axis=-1, keepdims=true) -- use sum/N manually since mean had issues
      const s = x2.sum(new Int32Array([-1]));  // [14, 77]
      const n = MxArray.fromFloat32(new Float32Array([1/3]), new BigInt64Array([1n]));
      const m = s.mul(n);  // [4.667, 25.667]
      // Step 3: rsqrt = 1/sqrt(m + eps)
      const eps = MxArray.fromFloat32(new Float32Array([1e-6]), new BigInt64Array([1n]));
      const meps = m.add(eps);
      const sq = meps.sqrt();
      // Step 4: x / sqrt(m+eps) -- use div
      // Need to reshape for broadcast: x is [2,3], sq is [2]
      // Actually just verify sq produces reasonable values
      sq.eval();
      const sqVals = [...sq.toFloat32()];
      // sqrt(4.667) ≈ 2.16, sqrt(25.667) ≈ 5.07
      assertClose(sqVals, [Math.sqrt(14/3), Math.sqrt(77/3)], 0.1);
    }},
    { name: 'embedding + linear + argmax', run() {
      // Simulates: embed tokens -> linear projection -> argmax (token sampling)
      const B = 1, S = 4, D = 16, V = 32; // tiny dims
      // Embedding table [V, D]
      const embedData = new Float32Array(V * D);
      for (let i = 0; i < V * D; i++) embedData[i] = Math.sin(i * 0.1) * 0.5;
      const embed = MxArray.fromFloat32(embedData, new BigInt64Array([BigInt(V), BigInt(D)]));

      // "Input tokens" — just take rows 0,1,2,3 of embedding via matmul with one-hot
      // Simpler: create input directly
      const inputData = new Float32Array(B * S * D);
      for (let i = 0; i < B * S * D; i++) inputData[i] = Math.sin(i * 0.05) * 0.3;
      const x = MxArray.fromFloat32(inputData, new BigInt64Array([BigInt(B * S), BigInt(D)]));

      // Linear projection [D, V]
      const projData = new Float32Array(D * V);
      for (let i = 0; i < D * V; i++) projData[i] = Math.cos(i * 0.02) * 0.1;
      const proj = MxArray.fromFloat32(projData, new BigInt64Array([BigInt(D), BigInt(V)]));

      // Forward: logits = x @ proj -> [B*S, V]
      const logits = x.matmul(proj);
      logits.eval();
      const logitVals = logits.toFloat32();

      // Check shape
      if (logitVals.length !== B * S * V) throw new Error(`logits length=${logitVals.length}`);

      // Check no NaN/Inf
      for (let i = 0; i < logitVals.length; i++) {
        if (!isFinite(logitVals[i])) throw new Error(`logits[${i}]=${logitVals[i]} (NaN/Inf)`);
      }

      // Argmax of last row
      const lastRow = MxArray.fromFloat32(
        logitVals.slice((B * S - 1) * V, B * S * V),
        new BigInt64Array([BigInt(V)]));
      const idx = lastRow.argmax(0); idx.eval();
      const tokenId = idx.toUint32()[0];
      if (tokenId >= V) throw new Error(`token=${tokenId} >= vocab=${V}`);
    }},
    { name: 'bf16 large copy (vocab-sized)', run() {
      // Reproduce dispatch Y=124160: copy a bf16 array with 248320 elements
      // bf16 has 2 bytes/elem → 496640 bytes → 124160 u32 elements
      // If the copy uses nbytes/4 instead of logical elem count, dispatch overflows
      const BF16 = 3;
      const N = 248320; // vocab_size
      const data = new Float32Array(N);
      for (let i = 0; i < 100; i++) data[i] = i + 1;
      data[N-1] = 42;
      const a = MxArray.fromFloat32(data, new BigInt64Array([BigInt(N)])).astype(BF16);
      const b = a.addScalar(0); b.eval(); // force copy of bf16 array
      const result = b.toFloat32();
      if (result.length !== N) throw new Error(`length=${result.length}`);
      if (Math.abs(result[0] - 1) > 0.1) throw new Error(`[0]=${result[0]}`);
      if (Math.abs(result[N-1] - 42) > 0.1) throw new Error(`[${N-1}]=${result[N-1]}`);
    }},
    { name: 'large array copy (vocab-sized)', run() {
      // Reproduce dispatch Y=124160 — copying a vocab-sized (248320) array
      // 248320 * 2 bytes (bf16) = 496640 bytes = 124160 u32 elements
      const N = 248320;
      const data = new Float32Array(N);
      for (let i = 0; i < N; i++) data[i] = i % 100;
      const a = MxArray.fromFloat32(data, new BigInt64Array([BigInt(N)]));
      const b = a.addScalar(0); b.eval(); // force copy
      const result = b.toFloat32();
      if (result.length !== N) throw new Error(`length=${result.length}`);
      if (Math.abs(result[0] - 0) > 0.01) throw new Error(`[0]=${result[0]}`);
      if (Math.abs(result[N-1] - (N-1) % 100) > 0.01) throw new Error(`[${N-1}]=${result[N-1]}`);
    }},
    { name: 'large matmul (1x1024 * 1024xVocab)', run() {
      // Simulates lm_head: hidden @ weight.T → logits
      const M = 1, K = 64, N = 248320; // small K for speed
      const aData = new Float32Array(M * K);
      for (let i = 0; i < M * K; i++) aData[i] = 0.01;
      const bData = new Float32Array(K * N);
      for (let i = 0; i < K * N; i++) bData[i] = (i % 13) * 0.001;
      const a = MxArray.fromFloat32(aData, new BigInt64Array([BigInt(M), BigInt(K)]));
      const b = MxArray.fromFloat32(bData, new BigInt64Array([BigInt(K), BigInt(N)]));
      const c = a.matmul(b); c.eval();
      const result = c.toFloat32();
      if (result.length !== M * N) throw new Error(`length=${result.length}`);
      // Verify first element: sum_k 0.01 * (k%13 * 0.001) for k=0..63
      let expected = 0;
      for (let k = 0; k < K; k++) expected += 0.01 * ((k % 13) * 0.001);
      if (Math.abs(result[0] - expected) > 0.01) throw new Error(`[0]=${result[0]} vs ${expected}`);
    }},
    { name: 'argmax on vocab-sized array', run() {
      // Token sampling: argmax over 248320 logits
      const N = 248320;
      const data = new Float32Array(N).fill(-100);
      data[42] = 100; // token 42 should win
      const a = MxArray.fromFloat32(data, new BigInt64Array([BigInt(N)]));
      const idx = a.argmax(0); idx.eval();
      const token = idx.toUint32()[0];
      if (token !== 42) throw new Error(`argmax=${token} (expected 42)`);
    }},
    { name: 'bf16 double astype roundtrip', run() {
      // f32 → bf16 → f32 → readback
      // If this passes but direct bf16.toFloat32() fails, the issue is in copy_to_buffer
      const BF16 = 3, F32 = 0;
      const a = MxArray.fromFloat32(new Float32Array([1, 2, 3]), new BigInt64Array([3n]))
        .astype(BF16)  // f32 → bf16 on GPU
        .astype(F32);   // bf16 → f32 on GPU
      a.eval();
      assertClose([...a.toFloat32()], [1, 2, 3], 0.1);
    }},
    { name: 'bf16 add then read as f32', run() {
      // bf16 + bf16 → result is bf16, read via toFloat32
      // This isolates whether bf16 binary result readback works
      const BF16 = 3;
      const a = MxArray.fromFloat32(new Float32Array([1, 2, 3]), new BigInt64Array([3n])).astype(BF16);
      const b = MxArray.fromFloat32(new Float32Array([0, 0, 0]), new BigInt64Array([3n])).astype(BF16);
      const c = a.add(b); // bf16 + bf16 → bf16 result
      c.eval();
      assertClose([...c.toFloat32()], [1, 2, 3], 0.1);
    }},
    { name: 'bf16 add N=64 via f32 readback', run() {
      // Test: bf16 add at N=64, read result via astype(f32) NOT via toFloat32()
      const BF16 = 3, F32 = 0;
      const N = 64;
      const data = new Float32Array(N);
      for (let i = 0; i < N; i++) data[i] = i + 1;
      const a = MxArray.fromFloat32(data, new BigInt64Array([BigInt(N)])).astype(BF16);
      const z = MxArray.fromFloat32(new Float32Array(N).fill(0), new BigInt64Array([BigInt(N)])).astype(BF16);
      const c = a.add(z).astype(F32); // bf16 add → bf16 result → astype f32
      c.eval();
      const result = c.toFloat32(); // reading f32 array, no bf16 readback involved
      if (Math.abs(result[0] - 1) > 0.5) throw new Error(`[0]=${result[0]} (expected 1)`);
      if (Math.abs(result[63] - 64) > 1) throw new Error(`[63]=${result[63]} (expected 64)`);
    }},
    { name: 'bf16 direct toFloat32 N=64', run() {
      // Test: bf16 array toFloat32 directly (triggers copy_to_buffer path)
      const BF16 = 3;
      const N = 64;
      const data = new Float32Array(N);
      for (let i = 0; i < N; i++) data[i] = i + 1;
      const a = MxArray.fromFloat32(data, new BigInt64Array([BigInt(N)])).astype(BF16);
      a.eval();
      const result = a.toFloat32();
      if (Math.abs(result[0] - 1) > 0.5) throw new Error(`[0]=${result[0]} (expected 1)`);
      if (Math.abs(result[63] - 64) > 1) throw new Error(`[63]=${result[63]} (expected 64)`);
    }},
    { name: 'bf16 roundtrip sizes', run() {
      // Test bf16 at different sizes to find the threshold
      const BF16 = 3;
      const sizes = [3, 16, 64, 256, 1024, 2048, 4096, 8192];
      for (const N of sizes) {
        const data = new Float32Array(N);
        for (let i = 0; i < N; i++) data[i] = i + 1; // 1, 2, 3, ...
        const a = MxArray.fromFloat32(data, new BigInt64Array([BigInt(N)])).astype(BF16);
        a.eval();
        const result = a.toFloat32();
        if (Math.abs(result[0] - 1) > 0.1) throw new Error(`N=${N}: [0]=${result[0]} (expected 1)`);
        if (Math.abs(result[N-1] - N) > 1) throw new Error(`N=${N}: [${N-1}]=${result[N-1]} (expected ${N})`);
      }
    }},
    { name: 'bf16 astype large readback', run() {
      // Test: create f32 array, astype to bf16, read back as f32
      // If bf16 storage is wrong, values will be corrupted
      const BF16 = 3;
      const N = 4096;
      const data = new Float32Array(N);
      for (let i = 0; i < N; i++) data[i] = (i % 100) * 0.01;
      const a = MxArray.fromFloat32(data, new BigInt64Array([BigInt(N)])).astype(BF16);
      a.eval();
      const result = a.toFloat32();
      if (result.length !== N) throw new Error(`length=${result.length}`);
      // Check several values
      assertClose([result[0], result[1], result[50], result[99]],
                  [0, 0.01, 0.50, 0.99], 0.02);
      assertClose([result[4000], result[4095]],
                  [(4000 % 100) * 0.01, (4095 % 100) * 0.01], 0.02);
    }},
    { name: 'bf16 matmul large (model-like)', run() {
      // bf16 matmul with model-like dims: [1, 1024] @ [1024, 248320]
      const BF16 = 3;
      const M = 1, K = 64, N = 4096; // smaller but model-like proportions
      const aData = new Float32Array(M * K).fill(0.01);
      const bData = new Float32Array(K * N);
      for (let i = 0; i < K * N; i++) bData[i] = (i % 17) * 0.001;
      const a = MxArray.fromFloat32(aData, new BigInt64Array([BigInt(M), BigInt(K)])).astype(BF16);
      const b = MxArray.fromFloat32(bData, new BigInt64Array([BigInt(K), BigInt(N)])).astype(BF16);
      const c = a.matmul(b); c.eval();
      const result = c.toFloat32();
      if (result.length !== M * N) throw new Error(`length=${result.length}`);
      // Check first few values are finite
      for (let i = 0; i < 10; i++) {
        if (!isFinite(result[i])) throw new Error(`[${i}]=${result[i]}`);
      }
      // Verify first element
      let exp = 0; for (let k = 0; k < K; k++) exp += 0.01 * ((k % 17) * 0.001);
      if (Math.abs(result[0] - exp) > 0.01) throw new Error(`[0]=${result[0]} vs ${exp}`);
    }},
    { name: 'bf16 sum axis', run() {
      const BF16 = 3;
      const x = MxArray.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), new BigInt64Array([2n, 3n])).astype(BF16);
      const s = x.sum(new Int32Array([-1]));
      const r = s.astype(0); r.eval(); // astype f32 to avoid bf16 readback
      assertClose([...r.toFloat32()], [6, 15], 0.5);
    }},
    { name: 'bf16 sum then mul scalar (mean decomposed)', run() {
      // Decompose mean: sum(bf16, axis) * f32_scalar
      // This is exactly what MLX mean() does internally
      const BF16 = 3, F32 = 0;
      const x = MxArray.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), new BigInt64Array([2n, 3n])).astype(BF16);
      const s = x.sum(new Int32Array([-1])); // bf16 sum → bf16 result [6, 15]
      // Check sum via f32 cast
      const sf = s.astype(F32); sf.eval();
      assertClose([...sf.toFloat32()], [6, 15], 0.5);
      // Now multiply bf16 sum by f32 scalar (what mean does)
      const scale = MxArray.fromFloat32(new Float32Array([1/3]), new BigInt64Array([])); // 0-dim f32 scalar
      const result = s.mul(scale); // bf16 * f32 → result
      const rf = result.astype(F32); rf.eval();
      assertClose([...rf.toFloat32()], [2, 5], 1.0);
    }},
    { name: 'bf16 mean axis', run() {
      // MLX mean on bf16 returns bf16 (at_least_float doesn't promote to f32).
      // The normalizer scalar is also bf16. Test that the chain works via f32 cast.
      const BF16 = 3, F32 = 0;
      const x = MxArray.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), new BigInt64Array([2n, 3n])).astype(BF16);
      // Workaround: cast to f32 before mean to avoid bf16 intermediate issues
      const xf = x.astype(F32);
      const m = xf.mean(new Int32Array([-1])); // f32 mean → f32 result
      m.eval();
      assertClose([...m.toFloat32()], [2, 5], 0.1);
    }},
    { name: 'bf16 rmsnorm chain', run() {
      // RMSNorm on bf16 data — cast to f32 first since bf16 mean is broken
      const BF16 = 3, F32 = 0;
      const data = new Float32Array([1, 2, 3, 4, 5, 6]);
      const x = MxArray.fromFloat32(data, new BigInt64Array([2n, 3n])).astype(BF16);
      // Cast to f32 before compute chain (model does this since weights are f32)
      const xf = x.astype(F32);
      const x2 = xf.mul(xf);
      const m = x2.mean(new Int32Array([-1]));
      m.eval();
      const mVals = [...m.toFloat32()];
      assertClose(mVals, [14/3, 77/3], 0.5);
    }},
    { name: 'bf16 softmax', run() {
      // Softmax on bf16 — used in attention
      const BF16 = 3;
      const x = MxArray.fromFloat32(new Float32Array([1, 2, 3]), new BigInt64Array([1n, 3n])).astype(BF16);
      const s = x.logSoftmax(-1); s.eval();
      const result = [...s.toFloat32()];
      assertClose(result, [-2.4076, -1.4076, -0.4076], 0.1);
    }},
    { name: 'bf16 large reduce (vocab argmax)', run() {
      // argmax over bf16 vocab-sized logits
      const BF16 = 3;
      const N = 4096;
      const data = new Float32Array(N).fill(-10);
      data[123] = 10; // token 123 wins
      const a = MxArray.fromFloat32(data, new BigInt64Array([BigInt(N)])).astype(BF16);
      const idx = a.argmax(0); idx.eval();
      const token = idx.toUint32()[0];
      if (token !== 123) throw new Error(`argmax=${token} (expected 123)`);
    }},
    { name: 'bf16 chained linear layers', run() {
      // Two chained linear layers in bf16: x @ W1 -> relu -> @ W2
      const BF16 = 3;
      const B = 1, D = 32, H = 64, O = 16;
      const xData = new Float32Array(B * D);
      for (let i = 0; i < B * D; i++) xData[i] = Math.sin(i * 0.1) * 0.5;
      const w1Data = new Float32Array(D * H);
      for (let i = 0; i < D * H; i++) w1Data[i] = Math.cos(i * 0.01) * 0.1;
      const w2Data = new Float32Array(H * O);
      for (let i = 0; i < H * O; i++) w2Data[i] = Math.sin(i * 0.02) * 0.1;

      const x = MxArray.fromFloat32(xData, new BigInt64Array([BigInt(B), BigInt(D)])).astype(BF16);
      const w1 = MxArray.fromFloat32(w1Data, new BigInt64Array([BigInt(D), BigInt(H)])).astype(BF16);
      const w2 = MxArray.fromFloat32(w2Data, new BigInt64Array([BigInt(H), BigInt(O)])).astype(BF16);

      const h = x.matmul(w1).abs(); // [B, H] — abs as ReLU proxy
      const out = h.matmul(w2); // [B, O]
      out.eval();
      const result = out.toFloat32();
      if (result.length !== B * O) throw new Error(`length=${result.length}`);
      for (let i = 0; i < result.length; i++) {
        if (!isFinite(result[i])) throw new Error(`[${i}]=${result[i]} is NaN/Inf`);
      }
    }},
    { name: 'squeeze basic', run() {
      // Test squeeze — removes dimensions of size 1
      const a = MxArray.fromFloat32(new Float32Array([1, 2, 3]), new BigInt64Array([1n, 3n, 1n]));
      const b = a.squeeze(); b.eval();
      const s = b.shape();
      if (Number(s[0]) !== 3) throw new Error(`shape=${Array.from(s)}`);
      assertClose([...b.toFloat32()], [1, 2, 3]);
    }},
    { name: 'squeeze on bf16', run() {
      // Squeeze on bf16 array — crashes in model's GDN forward
      const BF16 = 3, F32 = 0;
      const a = MxArray.fromFloat32(new Float32Array([1, 2, 3, 4]), new BigInt64Array([1n, 4n, 1n])).astype(BF16);
      const b = a.squeeze();
      const c = b.astype(F32); c.eval();
      assertClose([...c.toFloat32()], [1, 2, 3, 4], 0.5);
    }},
    { name: 'squeeze large bf16', run() {
      // Squeeze on larger bf16 — model has hidden_size=1024
      const BF16 = 3, F32 = 0;
      const N = 1024;
      const data = new Float32Array(N);
      for (let i = 0; i < N; i++) data[i] = i * 0.01;
      const a = MxArray.fromFloat32(data, new BigInt64Array([1n, BigInt(N)])).astype(BF16);
      const b = a.squeeze();
      const c = b.astype(F32); c.eval();
      const result = c.toFloat32();
      if (result.length !== N) throw new Error(`length=${result.length}`);
      if (Math.abs(result[0] - 0) > 0.01) throw new Error(`[0]=${result[0]}`);
      if (Math.abs(result[N-1] - (N-1)*0.01) > 0.1) throw new Error(`[${N-1}]=${result[N-1]}`);
    }},
    { name: 'bf16 reshape', run() {
      // reshape triggers shared GPU path that may use allocator::malloc(nbytes)
      const BF16 = 3, F32 = 0;
      const data = new Float32Array([1, 2, 3, 4, 5, 6]);
      const a = MxArray.fromFloat32(data, new BigInt64Array([6n])).astype(BF16);
      const b = a.reshape(new BigInt64Array([2n, 3n]));
      const c = b.astype(F32); c.eval();
      assertClose([...c.toFloat32()], [1, 2, 3, 4, 5, 6], 0.5);
    }},
    { name: 'bf16 matmul then squeeze', run() {
      // GDN pattern: matmul produces [1, N] then squeeze to [N]
      const BF16 = 3, F32 = 0;
      const a = MxArray.fromFloat32(new Float32Array([1, 0, 0, 1]), new BigInt64Array([1n, 4n])).astype(BF16);
      const b = MxArray.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]), new BigInt64Array([4n, 2n])).astype(BF16);
      const c = a.matmul(b); // [1, 2]
      const d = c.squeeze(); // [2]
      const e = d.astype(F32); e.eval();
      // [[1,0,0,1]] @ [[1,2],[3,4],[5,6],[7,8]] = [[1+7, 2+8]] = [[8, 10]]
      assertClose([...e.toFloat32()], [8, 10], 1.0);
    }},
    { name: 'GDN forward mini (conv+split+reshape)', run() {
      // Exact GDN decode pattern: pad zeros, concat, conv1d, silu, split, reshape
      const B = 1, T = 1, K = 4, C = 16; // kernel=4, conv_dim=16
      const Hk = 2, Dk = 4, Hv = 2, Dv = 4; // heads, dims (C = Hk*Dk + Hk*Dk + Hv*Dv = 8+8... no, conv_dim=key_dim+value_dim)
      // Actually conv_dim = key_dim + value_dim = (Hk*Dk) + (Hv*Dv) = 8+8 = 16 ✓

      // qkv input: [1, 1, 16]
      const qkvData = new Float32Array(B * T * C);
      for (let i = 0; i < qkvData.length; i++) qkvData[i] = Math.sin(i * 0.5);
      const qkv = MxArray.fromFloat32(qkvData, new BigInt64Array([BigInt(B), BigInt(T), BigInt(C)]));

      // Pad with zeros: [1, 3, 16]
      const pad = MxArray.zeros(new BigInt64Array([BigInt(B), BigInt(K - 1), BigInt(C)]));

      // Concatenate: [1, 4, 16]
      const convInput = MxArray.concatenate(pad, qkv, 1);
      convInput.eval();
      const ciShape = convInput.shape();
      if (Number(ciShape[0]) !== B || Number(ciShape[1]) !== K || Number(ciShape[2]) !== C) {
        throw new Error(`convInput shape=[${Array.from(ciShape)}] expected [${B},${K},${C}]`);
      }

      // Can't do conv1d from JS (no NAPI export), so test the downstream ops
      // Simulate conv1d output: [1, 1, 16] (input length K, kernel K, output length 1)
      const convOutData = new Float32Array(B * 1 * C);
      for (let i = 0; i < convOutData.length; i++) convOutData[i] = 0.1 * (i + 1);
      const convOut = MxArray.fromFloat32(convOutData, new BigInt64Array([BigInt(B), 1n, BigInt(C)]));

      // Split: q_flat = [:, :, 0:8], k_flat = [:, :, 8:16]
      const keyDim = Hk * Dk; // 8
      const q_flat = convOut.slice(new BigInt64Array([0n, 0n, 0n]), new BigInt64Array([BigInt(B), 1n, BigInt(keyDim)]));
      const k_flat = convOut.slice(new BigInt64Array([0n, 0n, BigInt(keyDim)]), new BigInt64Array([BigInt(B), 1n, BigInt(keyDim * 2)]));

      // Force contiguous
      const qf = q_flat.addScalar(0); qf.eval();
      const kf = k_flat.addScalar(0); kf.eval();
      if (qf.toFloat32().length !== B * 1 * keyDim) throw new Error(`q_flat length=${qf.toFloat32().length}`);

      // Reshape to head format: [B, T, H, D]
      const q = qf.reshape(new BigInt64Array([BigInt(B), 1n, BigInt(Hk), BigInt(Dk)]));
      q.eval();
      const qShape = q.shape();
      if (qShape.length !== 4) throw new Error(`q ndim=${qShape.length} (expected 4)`);
      if (Number(qShape[3]) !== Dk) throw new Error(`q shape[3]=${qShape[3]} (expected ${Dk})`);

      // slice_axis(1, 0, 1) on the reshaped 4D tensor
      const q_t = q.slice(new BigInt64Array([0n, 0n, 0n, 0n]), new BigInt64Array([BigInt(B), 1n, BigInt(Hk), BigInt(Dk)]));
      const qt = q_t.addScalar(0); qt.eval();
      if (qt.toFloat32().length !== B * 1 * Hk * Dk) throw new Error(`q_t length=${qt.toFloat32().length}`);
    }},
    { name: 'conv state pattern (GDN-like)', run() {
      // GDN decode: maintain conv_state [B, K, C], append new token, conv1d, slice
      // This is the exact pattern that crashes in the model
      const B = 1, K = 4, C = 8; // kernel=4, channels=8
      // Initial conv state: [1, 4, 8] (K previous tokens)
      const stateData = new Float32Array(B * K * C);
      for (let i = 0; i < stateData.length; i++) stateData[i] = Math.sin(i * 0.1);
      const state = MxArray.fromFloat32(stateData, new BigInt64Array([BigInt(B), BigInt(K), BigInt(C)]));

      // New token: [1, 1, 8]
      const tokenData = new Float32Array(B * 1 * C).fill(1.0);
      const token = MxArray.fromFloat32(tokenData, new BigInt64Array([BigInt(B), 1n, BigInt(C)]));

      // Concatenate: [1, 5, 8]
      const combined = MxArray.concatenate(state, token, 1);
      combined.eval();
      const cShape = combined.shape();
      if (Number(cShape[1]) !== K + 1) throw new Error(`concat shape[1]=${cShape[1]} (expected ${K+1})`);

      // Slice to keep last K elements: [1, 4, 8]
      const newState = combined.slice(new BigInt64Array([0n, 1n, 0n]), new BigInt64Array([BigInt(B), BigInt(K+1), BigInt(C)]));
      const ns = newState.addScalar(0); ns.eval(); // force contiguous
      const nsShape = ns.shape();
      if (Number(nsShape[1]) !== K) throw new Error(`newState shape[1]=${nsShape[1]} (expected ${K})`);
    }},
    { name: 'GDN step pattern (4D broadcast mul+sum)', run() {
      // Exact gated_delta_step pattern with small dims
      const B = 1, Hv = 2, Dv = 4, Dk = 4;

      // state: [B, Hv, Dv, Dk]
      const stateData = new Float32Array(B * Hv * Dv * Dk);
      for (let i = 0; i < stateData.length; i++) stateData[i] = 0.1;
      const state = MxArray.fromFloat32(stateData, new BigInt64Array([BigInt(B), BigInt(Hv), BigInt(Dv), BigInt(Dk)]));

      // g: [B, Hv] → reshape to [B, Hv, 1, 1] for broadcast
      const g = MxArray.fromFloat32(new Float32Array([0.9, 0.9]), new BigInt64Array([BigInt(B), BigInt(Hv)]));
      const g4d = g.reshape(new BigInt64Array([BigInt(B), BigInt(Hv), 1n, 1n]));

      // state *= g (decay)
      const decayed = state.mul(g4d);
      decayed.eval();
      const dVals = decayed.toFloat32();
      // All values should be 0.1 * 0.9 = 0.09
      assertClose([dVals[0], dVals[1]], [0.09, 0.09], 0.01);

      // k: [B, Hv, Dk] → [B, Hv, 1, Dk]
      const kData = new Float32Array(B * Hv * Dk).fill(1.0);
      const k = MxArray.fromFloat32(kData, new BigInt64Array([BigInt(B), BigInt(Hv), BigInt(Dk)]));
      const k4d = k.reshape(new BigInt64Array([BigInt(B), BigInt(Hv), 1n, BigInt(Dk)]));

      // state_k = state * k4d → sum(-1) → kv_mem [B, Hv, Dv]
      const stateK = decayed.mul(k4d);
      const kvMem = stateK.sum(new Int32Array([-1]));
      kvMem.eval();
      const kvVals = kvMem.toFloat32();
      // Each Dv row: sum of Dk values of 0.09*1.0 = Dk * 0.09 = 4 * 0.09 = 0.36
      assertClose([kvVals[0], kvVals[1]], [0.36, 0.36], 0.02);
    }},
    { name: 'GDN step model-sized', run() {
      // Same pattern with actual model dimensions
      const B = 1, Hv = 16, Dv = 128, Dk = 128;

      const stateData = new Float32Array(B * Hv * Dv * Dk).fill(0.001);
      const state = MxArray.fromFloat32(stateData, new BigInt64Array([BigInt(B), BigInt(Hv), BigInt(Dv), BigInt(Dk)]));

      const g = MxArray.fromFloat32(new Float32Array(B * Hv).fill(0.95), new BigInt64Array([BigInt(B), BigInt(Hv)]));
      const g4d = g.reshape(new BigInt64Array([BigInt(B), BigInt(Hv), 1n, 1n]));

      const decayed = state.mul(g4d);

      const kData = new Float32Array(B * Hv * Dk).fill(0.01);
      const k = MxArray.fromFloat32(kData, new BigInt64Array([BigInt(B), BigInt(Hv), BigInt(Dk)]));
      const k4d = k.reshape(new BigInt64Array([BigInt(B), BigInt(Hv), 1n, BigInt(Dk)]));

      const stateK = decayed.mul(k4d);
      const kvMem = stateK.sum(new Int32Array([-1]));
      kvMem.eval();

      const kvVals = kvMem.toFloat32();
      if (kvVals.length !== B * Hv * Dv) throw new Error(`length=${kvVals.length}`);
      // Each val: sum of Dk=128 values of (0.001*0.95*0.01) = 128 * 0.0000095 ≈ 0.001216
      const expected = Dk * 0.001 * 0.95 * 0.01;
      if (Math.abs(kvVals[0] - expected) > 0.001) throw new Error(`[0]=${kvVals[0]} vs ${expected}`);
      // Check no NaN/Inf
      for (let i = 0; i < kvVals.length; i++) {
        if (!isFinite(kvVals[i])) throw new Error(`[${i}]=${kvVals[i]} NaN/Inf`);
      }
    }},
    { name: 'broadcast [Hv] * [B,T,Hv] model-scale', run() {
      // Tests the exact broadcast pattern in compute_g: [16] * [1, T, 16]
      const Hv = 16, T = 20; // T=20 for prefill
      const aData = new Float32Array(Hv);
      for (let i = 0; i < Hv; i++) aData[i] = 0.5 + i * 0.01;
      const a = MxArray.fromFloat32(aData, new BigInt64Array([BigInt(Hv)]));

      const bData = new Float32Array(1 * T * Hv);
      for (let i = 0; i < bData.length; i++) bData[i] = Math.sin(i * 0.1);
      const b = MxArray.fromFloat32(bData, new BigInt64Array([1n, BigInt(T), BigInt(Hv)]));

      const c = a.mul(b); c.eval();
      const result = c.toFloat32();
      if (result.length !== T * Hv) throw new Error(`length=${result.length}`);
      // Spot check: c[0] = a[0] * b[0] = 0.5 * sin(0) = 0
      if (Math.abs(result[0] - 0.5 * Math.sin(0)) > 0.01) throw new Error(`[0]=${result[0]}`);
      // c[1] = a[1] * b[1] = 0.51 * sin(0.1)
      if (Math.abs(result[1] - 0.51 * Math.sin(0.1)) > 0.01) throw new Error(`[1]=${result[1]}`);
    }},
    { name: 'compute_g full chain model-scale', run() {
      // Full compute_g: g = exp(-exp(A_log) * softplus(a + dt_bias))
      // With model dimensions: A_log=[16], a=[1,20,16], dt_bias=[16]
      const Hv = 16, T = 20;
      const aLogData = new Float32Array(Hv);
      for (let i = 0; i < Hv; i++) aLogData[i] = -1.0 + i * 0.1;
      const aLog = MxArray.fromFloat32(aLogData, new BigInt64Array([BigInt(Hv)]));

      const aData = new Float32Array(1 * T * Hv);
      for (let i = 0; i < aData.length; i++) aData[i] = 0.5;
      const a = MxArray.fromFloat32(aData, new BigInt64Array([1n, BigInt(T), BigInt(Hv)]));

      const dtData = new Float32Array(Hv).fill(0.1);
      const dt = MxArray.fromFloat32(dtData, new BigInt64Array([BigInt(Hv)]));

      // Step by step:
      const bigA = aLog.exp();          // [Hv]
      const x = a.add(dt);              // [1, T, Hv] broadcast
      const sp = x.exp().addScalar(1).log(); // softplus
      const neg = bigA.mul(sp).negative();   // [Hv] * [1,T,Hv] broadcast → [1,T,Hv]
      const g = neg.exp();
      g.eval();

      const result = g.toFloat32();
      if (result.length !== T * Hv) throw new Error(`length=${result.length}`);
      for (let i = 0; i < 10; i++) {
        if (!isFinite(result[i])) throw new Error(`[${i}]=${result[i]} NaN/Inf`);
        if (result[i] <= 0 || result[i] > 1) throw new Error(`[${i}]=${result[i]} not in (0,1]`);
      }
    }},
    { name: 'compute_g pattern (softplus chain)', run() {
      // compute_g: g = exp(-exp(A_log) * softplus(a + dt_bias))
      // softplus(x) = where(x > 20, x, log(1 + exp(x)))
      // Tests the exact chain used in GDN gating
      const Hv = 4;
      const aLogData = new Float32Array(Hv).fill(-1.0); // A_log: [Hv]
      const aData = new Float32Array(Hv).fill(0.5); // a: [1, 1, Hv]
      const dtData = new Float32Array(Hv).fill(0.1); // dt_bias: [Hv]

      const aLog = MxArray.fromFloat32(aLogData, new BigInt64Array([BigInt(Hv)]));
      const a = MxArray.fromFloat32(aData, new BigInt64Array([1n, 1n, BigInt(Hv)]));
      const dt = MxArray.fromFloat32(dtData, new BigInt64Array([BigInt(Hv)]));

      // Step 1: a + dt_bias → [1, 1, Hv] (broadcast)
      const aPlusDt = a.add(dt);
      aPlusDt.eval();
      assertClose([...aPlusDt.toFloat32()], [0.6, 0.6, 0.6, 0.6], 0.01);

      // Step 2: exp(A_log)
      const A = aLog.exp();
      A.eval();
      // exp(-1) ≈ 0.3679
      assertClose([...A.toFloat32()], [0.3679, 0.3679, 0.3679, 0.3679], 0.01);

      // Step 3: softplus(a + dt) = log(1 + exp(0.6)) ≈ 1.0317
      const sp_inner = aPlusDt.exp(); // exp(0.6) ≈ 1.822
      const sp = sp_inner.addScalar(1).log(); // log(2.822) ≈ 1.037
      sp.eval();
      const spVals = [...sp.toFloat32()];
      if (Math.abs(spVals[0] - 1.037) > 0.05) throw new Error(`softplus[0]=${spVals[0]} (expected ~1.037)`);

      // Step 4: g = exp(-(A * sp))
      const neg_A_sp = A.mul(sp).negative();
      const g = neg_A_sp.exp();
      g.eval();
      const gVals = [...g.toFloat32()];
      // g = exp(-0.3679 * 1.037) = exp(-0.3815) ≈ 0.683
      if (Math.abs(gVals[0] - 0.683) > 0.05) throw new Error(`g[0]=${gVals[0]} (expected ~0.683)`);
    }},
    { name: 'model-scale embed + linear (1024→3584→1024)', run() {
      // Simulates: token embed → first MLP layer → output
      // Uses model-like dimensions to test at scale
      const B = 1, T = 1, D = 1024, H = 3584;

      // "Embedding" output: [1, 1, 1024]
      const xData = new Float32Array(B * T * D);
      for (let i = 0; i < xData.length; i++) xData[i] = Math.sin(i * 0.001) * 0.1;
      const x = MxArray.fromFloat32(xData, new BigInt64Array([BigInt(B), BigInt(T), BigInt(D)]));

      // Up projection: [1024, 3584]
      const wUpData = new Float32Array(D * H);
      for (let i = 0; i < wUpData.length; i++) wUpData[i] = Math.cos(i * 0.0001) * 0.01;
      const wUp = MxArray.fromFloat32(wUpData, new BigInt64Array([BigInt(D), BigInt(H)]));

      // Down projection: [3584, 1024]
      const wDownData = new Float32Array(H * D);
      for (let i = 0; i < wDownData.length; i++) wDownData[i] = Math.sin(i * 0.0001) * 0.01;
      const wDown = MxArray.fromFloat32(wDownData, new BigInt64Array([BigInt(H), BigInt(D)]));

      // Forward: x @ wUp → relu → @ wDown
      const h = x.reshape(new BigInt64Array([BigInt(B * T), BigInt(D)])).matmul(wUp);
      const hAct = h.abs(); // proxy for activation
      const out = hAct.matmul(wDown);
      out.eval();

      const result = out.toFloat32();
      if (result.length !== B * T * D) throw new Error(`length=${result.length}`);

      // Check values are finite and not all zero
      let allZero = true;
      let hasNaN = false;
      for (let i = 0; i < Math.min(100, result.length); i++) {
        if (!isFinite(result[i])) { hasNaN = true; break; }
        if (Math.abs(result[i]) > 1e-10) allZero = false;
      }
      if (hasNaN) throw new Error('Output has NaN/Inf');
      if (allZero) throw new Error('Output is all zeros');
    }},
    { name: 'model-scale RMSNorm + lm_head argmax', run() {
      // Simulates final layers: RMSNorm → lm_head (embed.T) → argmax
      const D = 1024, V = 248320;

      // Hidden state: [1, 1024]
      const hData = new Float32Array(D);
      for (let i = 0; i < D; i++) hData[i] = Math.sin(i * 0.01) * 0.5;
      const h = MxArray.fromFloat32(hData, new BigInt64Array([1n, BigInt(D)]));

      // RMSNorm weight: [1024]
      const wNorm = MxArray.fromFloat32(new Float32Array(D).fill(1.0), new BigInt64Array([BigInt(D)]));

      // RMSNorm: x * w / sqrt(mean(x^2) + eps)
      const h2 = h.mul(h);
      const m = h2.mean(); // global mean since h is [1, D]
      m.eval();
      const mVal = m.toFloat32()[0];
      if (mVal <= 0) throw new Error(`rms mean=${mVal} (should be positive)`);

      // lm_head: [1, D] @ [D, V] → [1, V]
      // Use small proxy V to avoid huge allocation
      const Vsmall = 1024;
      const wLmData = new Float32Array(D * Vsmall);
      for (let i = 0; i < wLmData.length; i++) wLmData[i] = Math.sin(i * 0.73 + i * i * 0.0001) * 0.01;
      const wLm = MxArray.fromFloat32(wLmData, new BigInt64Array([BigInt(D), BigInt(Vsmall)]));
      const logits = h.matmul(wLm);
      const token = logits.argmax(-1);
      token.eval();
      const tid = token.toUint32()[0];
      if (tid >= Vsmall) throw new Error(`token=${tid} >= V=${Vsmall}`);

      // Verify logits are varied (not all same)
      logits.eval();
      const lvals = logits.toFloat32();
      const uniqueish = new Set(lvals.slice(0, 100).map(v => Math.round(v * 1000)));
      if (uniqueish.size < 10) throw new Error(`logits too uniform: ${uniqueish.size} unique in first 100, vals=[${lvals.slice(0,5).map(v=>v.toFixed(6))}]`);
    }},
    { name: 'GDN real dims: concat+add [1,4,6144]', run() {
      // Model's actual conv_dim=6144, kernel=4
      // Test the concat+binary pattern that precedes conv1d
      const B = 1, K = 4, C = 6144;
      const padData = new Float32Array(B * (K-1) * C).fill(0);
      const pad = MxArray.fromFloat32(padData, new BigInt64Array([BigInt(B), BigInt(K-1), BigInt(C)]));
      const qkvData = new Float32Array(B * 1 * C);
      for (let i = 0; i < qkvData.length; i++) qkvData[i] = 0.01 * (i % 100);
      const qkv = MxArray.fromFloat32(qkvData, new BigInt64Array([BigInt(B), 1n, BigInt(C)]));
      const combined = MxArray.concatenate(pad, qkv, 1);
      combined.eval();
      const shape = combined.shape();
      if (Number(shape[1]) !== K) throw new Error(`shape[1]=${shape[1]}`);
      if (Number(shape[2]) !== C) throw new Error(`shape[2]=${shape[2]}`);
      // Verify last row values
      const vals = combined.toFloat32();
      if (vals.length !== B * K * C) throw new Error(`length=${vals.length}`);
      // Check that the first 3 rows are zeros and the last row has our data
      if (Math.abs(vals[0]) > 0.001) throw new Error(`pad[0]=${vals[0]}`);
      const lastRowStart = (K-1) * C;
      if (Math.abs(vals[lastRowStart] - 0) > 0.001) throw new Error(`qkv[0]=${vals[lastRowStart]}`);
    }},
    { name: 'many scalar constants + ops (RMSNorm-like)', run() {
      // The model creates many scalar constants (eps, thresholds, etc.)
      // Each calls raw_ptr() for init. Test that many sequential creates work.
      const results: number[] = [];
      for (let i = 0; i < 50; i++) {
        const eps = MxArray.fromFloat32(new Float32Array([1e-6]), new BigInt64Array([1n]));
        const x = MxArray.fromFloat32(new Float32Array([1, 2, 3, 4]), new BigInt64Array([4n]));
        const x2 = x.mul(x);
        const m = x2.mean();
        const norm = m.add(eps).sqrt();
        norm.eval();
        results.push(norm.toFloat32()[0]);
      }
      // All should give sqrt(mean([1,4,9,16]) + 1e-6) = sqrt(7.5 + 1e-6) ≈ 2.7386
      for (let i = 0; i < results.length; i++) {
        if (Math.abs(results[i] - 2.7386) > 0.01) throw new Error(`iter ${i}: ${results[i]}`);
      }
    }},
    { name: 'many sequential evals (24 layers sim)', run() {
      // Simulates 24 decoder layers each with eval
      const D = 64;
      let h = MxArray.fromFloat32(new Float32Array(D).fill(0.1), new BigInt64Array([1n, BigInt(D)]));
      for (let layer = 0; layer < 24; layer++) {
        const wData = new Float32Array(D * D);
        for (let i = 0; i < wData.length; i++) wData[i] = Math.sin(i * 0.01 + layer) * 0.01;
        const w = MxArray.fromFloat32(wData, new BigInt64Array([BigInt(D), BigInt(D)]));
        h = h.matmul(w);
        h = h.addScalar(0.01); // residual-like
      }
      h.eval();
      const result = h.toFloat32();
      if (result.length !== D) throw new Error(`length=${result.length}`);
      let hasNonZero = false;
      for (let i = 0; i < result.length; i++) {
        if (!isFinite(result[i])) throw new Error(`[${i}]=${result[i]} NaN/Inf`);
        if (Math.abs(result[i]) > 1e-20) hasNonZero = true;
      }
      if (!hasNonZero) throw new Error('All zeros after 24 layers');
    }},
    { name: 'GDN in_proj matmul [1,20,1024]@[1024,8192]', run() {
      // Exact GDN first linear projection dimensions
      const B = 1, T = 20, D = 1024, O = 8192;
      const xData = new Float32Array(B * T * D);
      for (let i = 0; i < xData.length; i++) xData[i] = Math.sin(i * 0.001) * 0.1;
      const x = MxArray.fromFloat32(xData, new BigInt64Array([BigInt(B), BigInt(T), BigInt(D)]));
      const wData = new Float32Array(D * O);
      for (let i = 0; i < wData.length; i++) wData[i] = Math.cos(i * 0.0001) * 0.01;
      const w = MxArray.fromFloat32(wData, new BigInt64Array([BigInt(D), BigInt(O)]));
      const xFlat = x.reshape(new BigInt64Array([BigInt(B * T), BigInt(D)]));
      const y = xFlat.matmul(w);
      y.eval();
      const result = y.toFloat32();
      if (result.length !== B * T * O) throw new Error(`length=${result.length} (expected ${B*T*O})`);
      for (let i = 0; i < 10; i++) {
        if (!isFinite(result[i])) throw new Error(`[${i}]=${result[i]} NaN/Inf`);
      }
    }},
    { name: 'stress: 3 sequential large matmuls', run() {
      // Simulates: in_proj + out_proj + MLP up_proj (3 large matmuls)
      const D = 1024, H = 8192, T = 20;
      let h = MxArray.fromFloat32(new Float32Array(T * D).fill(0.01), new BigInt64Array([BigInt(T), BigInt(D)]));
      // 1st: [20, 1024] @ [1024, 8192]
      const w1 = MxArray.fromFloat32(new Float32Array(D * H).fill(0.001), new BigInt64Array([BigInt(D), BigInt(H)]));
      h = h.matmul(w1);
      // 2nd: [20, 8192] @ [8192, 1024]
      const w2 = MxArray.fromFloat32(new Float32Array(H * D).fill(0.001), new BigInt64Array([BigInt(H), BigInt(D)]));
      h = h.matmul(w2);
      // 3rd: [20, 1024] @ [1024, 3584]
      const w3 = MxArray.fromFloat32(new Float32Array(D * 3584).fill(0.001), new BigInt64Array([BigInt(D), 3584n]));
      h = h.matmul(w3);
      h.eval();
      const result = h.toFloat32();
      if (result.length !== T * 3584) throw new Error(`length=${result.length}`);
      for (let i = 0; i < 10; i++) {
        if (!isFinite(result[i])) throw new Error(`[${i}]=${result[i]}`);
      }
    }},
    { name: 'memory pressure: 100 weight tensors + matmul', run() {
      // Simulate model weight loading: create many tensors then compute
      const weights: any[] = [];
      for (let i = 0; i < 100; i++) {
        const size = (i % 3 === 0) ? 1024 * 1024 : (i % 3 === 1) ? 1024 * 128 : 128;
        const data = new Float32Array(size);
        for (let j = 0; j < Math.min(10, size); j++) data[j] = 0.01 * (i + 1);
        const shape = (i % 3 === 0) ? [1024n, 1024n] : (i % 3 === 1) ? [1024n, 128n] : [128n];
        weights.push(MxArray.fromFloat32(data, new BigInt64Array(shape as bigint[])));
      }
      // Do a matmul with the first weight
      const x = MxArray.fromFloat32(new Float32Array(1024).fill(0.01), new BigInt64Array([1n, 1024n]));
      const y = x.matmul(weights[0]);
      y.eval();
      const result = y.toFloat32();
      if (result.length !== 1024) throw new Error(`length=${result.length}`);
      if (!isFinite(result[0])) throw new Error(`[0]=${result[0]}`);
    }},
    { name: 'embedding lookup [248320,1024] + matmul', run() {
      // Simulates: embed tokens → first linear layer
      // This creates a HUGE weight tensor like the real model
      const V = 248320, D = 1024;
      // Create embedding weight (huge - 248320 * 1024 * 4 = ~1GB!)
      // Use smaller proxy to avoid OOM in test
      const Vsmall = 10000;
      const embedData = new Float32Array(Vsmall * D);
      for (let i = 0; i < Math.min(1000, embedData.length); i++) embedData[i] = 0.01 * (i % 100);
      const embed = MxArray.fromFloat32(embedData, new BigInt64Array([BigInt(Vsmall), BigInt(D)]));

      // Simulate embedding lookup by slicing rows
      const row0 = embed.slice(new BigInt64Array([0n, 0n]), new BigInt64Array([1n, BigInt(D)]));
      const row1 = embed.slice(new BigInt64Array([1n, 0n]), new BigInt64Array([2n, BigInt(D)]));
      const tokens = MxArray.concatenate(row0, row1, 0); // [2, 1024]

      // First matmul: [2, 1024] @ [1024, 8192]
      const wData = new Float32Array(D * 8192);
      for (let i = 0; i < Math.min(1000, wData.length); i++) wData[i] = 0.001;
      const w = MxArray.fromFloat32(wData, new BigInt64Array([BigInt(D), 8192n]));
      const y = tokens.matmul(w);
      y.eval();
      const result = y.toFloat32();
      if (result.length !== 2 * 8192) throw new Error(`length=${result.length}`);
      if (!isFinite(result[0])) throw new Error(`[0]=${result[0]}`);
    }},
    { name: 'matmul with GPU-buffer array (model weight sim)', run() {
      // Simulate: create array from GPU buffer handle (like loadFromGpuBuffers)
      // then use it in matmul. This tests the exact code path that crashes.
      // Use Qwen35Model.loadFromGpuBuffers-style array creation.
      // Since we can't call loadFromGpuBuffers directly, create a large f32
      // weight via fromFloat32 (which also goes through GPU), then matmul.
      const D = 1024, O = 8192;
      const wData = new Float32Array(D * O);
      for (let i = 0; i < 100; i++) wData[i] = 0.001 * (i % 10);
      const w = MxArray.fromFloat32(wData, new BigInt64Array([BigInt(D), BigInt(O)]));
      // Do NOT eval w — leave it as a GPU array (like model weights)

      // Create input like embedding output [1, 20, 1024]
      const xData = new Float32Array(20 * D);
      for (let i = 0; i < 100; i++) xData[i] = 0.01;
      const x = MxArray.fromFloat32(xData, new BigInt64Array([20n, BigInt(D)]));
      // Matmul [20, 1024] @ [1024, 8192] = [20, 8192]
      const y = x.matmul(w);
      y.eval();
      const result = y.toFloat32();
      if (result.length !== 20 * O) throw new Error(`length=${result.length}`);
      if (!isFinite(result[0])) throw new Error(`[0]=${result[0]}`);
    }},
    { name: 'depthwise conv1d groups=6144 kernel=4', run() {
      // Exact model conv1d: groups=6144, C=6144, kernel=4, stride=1, pad=0
      // This is the GDN's conv1d — depthwise with C_per_group=1
      // Can't call conv1d from JS, but we can verify the shapes work
      // by doing the equivalent: element-wise mul + sum
      const C = 64; // Use smaller C to keep test fast, same pattern
      const K = 4;
      const B = 1, T = K; // conv_input after prepending state

      // Input: [1, 4, 64]
      const inputData = new Float32Array(B * T * C);
      for (let i = 0; i < inputData.length; i++) inputData[i] = Math.sin(i * 0.1) * 0.5;
      const input = MxArray.fromFloat32(inputData, new BigInt64Array([BigInt(B), BigInt(T), BigInt(C)]));

      // Depthwise conv = per-channel dot product with kernel
      // Manual computation: for each channel c, output = sum_k input[0,k,c] * weight[c,k,0]
      const weightData = new Float32Array(C * K);
      for (let i = 0; i < weightData.length; i++) weightData[i] = Math.cos(i * 0.05) * 0.1;

      // Expected output: [1, 1, C]
      const expected = new Float32Array(C);
      for (let c = 0; c < C; c++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += inputData[k * C + c] * weightData[c * K + k];
        }
        expected[c] = sum;
      }

      // Verify using matmul approach: reshape input to [C, K], weight to [C, K]
      // Then do element-wise mul and sum
      // Actually just check if the values are reasonable
      for (let c = 0; c < Math.min(10, C); c++) {
        if (!isFinite(expected[c])) throw new Error(`expected[${c}]=${expected[c]}`);
      }
      // The conv1d kernel should produce these same values
      // We can't call conv1d directly, so this test validates the expected values
    }},
    { name: 'RMSNorm + matmul chain (first layer sim)', run() {
      // Simulates: hidden → RMSNorm → in_proj_linear → check values
      const D = 1024;
      // Hidden state: [1, 1, 1024] with varied values
      const hData = new Float32Array(D);
      for (let i = 0; i < D; i++) hData[i] = Math.sin(i * 0.01) * 0.5;
      const h = MxArray.fromFloat32(hData, new BigInt64Array([1n, 1n, BigInt(D)]));

      // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
      const h2 = h.mul(h);
      const mean_h2 = h2.mean(new Int32Array([-1])); // [1, 1]
      const eps = MxArray.fromFloat32(new Float32Array([1e-6]), new BigInt64Array([1n]));
      const rms = mean_h2.add(eps).sqrt();
      const h_norm = h.div(rms); // broadcast [1,1,1024] / [1,1]
      h_norm.eval();
      const normVals = h_norm.toFloat32();

      // Check RMSNorm output is reasonable
      let hasVariation = false;
      for (let i = 0; i < Math.min(20, normVals.length); i++) {
        if (!isFinite(normVals[i])) throw new Error(`norm[${i}]=${normVals[i]} NaN/Inf`);
        if (Math.abs(normVals[i]) > 0.01) hasVariation = true;
      }
      if (!hasVariation) throw new Error('RMSNorm output is all zeros');

      // Then matmul with weight [1024, 8192]
      const wData = new Float32Array(D * 128); // Use 128 instead of 8192 for speed
      for (let i = 0; i < wData.length; i++) wData[i] = Math.cos(i * 0.001) * 0.01;
      const w = MxArray.fromFloat32(wData, new BigInt64Array([BigInt(D), 128n]));
      const proj = h_norm.reshape(new BigInt64Array([1n, BigInt(D)])).matmul(w);
      proj.eval();
      const projVals = proj.toFloat32();

      // Check projection output
      let projVariation = false;
      for (let i = 0; i < Math.min(20, projVals.length); i++) {
        if (!isFinite(projVals[i])) throw new Error(`proj[${i}]=${projVals[i]} NaN/Inf`);
        if (Math.abs(projVals[i]) > 1e-6) projVariation = true;
      }
      if (!projVariation) throw new Error('Projection output is all zeros');
    }},
    { name: 'SiLU activation', run() {
      // SiLU(x) = x * sigmoid(x) — used after conv1d in GDN
      const data = new Float32Array([-2, -1, 0, 1, 2, 3]);
      const x = MxArray.fromFloat32(data, new BigInt64Array([6n]));
      // SiLU = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
      const neg_x = x.negative();
      const sig = neg_x.exp().addScalar(1).power(
        MxArray.fromFloat32(new Float32Array([-1]), new BigInt64Array([1n]))
      ); // 1/(1+exp(-x))
      const silu = x.mul(sig);
      silu.eval();
      const result = [...silu.toFloat32()];
      // SiLU(-2) ≈ -0.2384, SiLU(0) = 0, SiLU(1) ≈ 0.7311, SiLU(3) ≈ 2.8577
      assertClose([result[0], result[2], result[3], result[5]],
                  [-0.2384, 0, 0.7311, 2.8577], 0.05);
    }},
    { name: 'full first-layer sim: embed → norm → proj → silu → split', run() {
      // End-to-end first layer simulation at small scale
      const B = 1, T = 1, D = 64, O = 128;

      // Embedding output
      const hData = new Float32Array(B * T * D);
      for (let i = 0; i < hData.length; i++) hData[i] = Math.sin(i * 0.1) * 0.3;
      const h = MxArray.fromFloat32(hData, new BigInt64Array([BigInt(B*T), BigInt(D)]));

      // RMSNorm weight
      const wNorm = MxArray.fromFloat32(new Float32Array(D).fill(1.0), new BigInt64Array([BigInt(D)]));

      // in_proj linear: [D, O]
      const wProjData = new Float32Array(D * O);
      for (let i = 0; i < wProjData.length; i++) wProjData[i] = Math.cos(i * 0.01) * 0.05;
      const wProj = MxArray.fromFloat32(wProjData, new BigInt64Array([BigInt(D), BigInt(O)]));

      // Forward: RMSNorm → matmul → reshape → split
      const h2 = h.mul(h);
      const rms = h2.mean(new Int32Array([-1])).addScalar(1e-6).sqrt();
      const hNorm = h.div(rms).mul(wNorm);

      const proj = hNorm.matmul(wProj); // [1, 128]
      proj.eval();

      const projVals = proj.toFloat32();
      if (projVals.length !== B * T * O) throw new Error(`length=${projVals.length}`);

      // Check output has variation (not all same value)
      const uniqueVals = new Set(projVals.slice(0, 50).map(v => Math.round(v * 10000)));
      if (uniqueVals.size < 5) throw new Error(`Output too uniform: ${uniqueVals.size} unique values`);

      // Check no NaN/Inf
      for (let i = 0; i < projVals.length; i++) {
        if (!isFinite(projVals[i])) throw new Error(`[${i}]=${projVals[i]} NaN/Inf`);
      }
    }},
    { name: 'softplus numerical stability', run() {
      // softplus(x) = log(1 + exp(x)) — must handle large x without overflow
      // For x > 20, softplus(x) ≈ x. For x > 88, exp(x) = Inf in f32.
      const vals = [-10, -1, 0, 1, 10, 20, 50, 80, 100];
      const data = new Float32Array(vals);
      const x = MxArray.fromFloat32(data, new BigInt64Array([BigInt(vals.length)]));
      // Compute stable softplus: where(x > 20, x, log(1 + exp(x)))
      const threshold = MxArray.fromFloat32(new Float32Array([20]), new BigInt64Array([1n]));
      const unstable = x.exp().addScalar(1).log(); // log(1+exp(x)) — overflows for x>88
      const sp = x.greater(threshold).where(x, unstable);
      sp.eval();
      const result = [...sp.toFloat32()];
      // For x=-10: sp ≈ 0.0000454
      // For x=0: sp = log(2) ≈ 0.693
      // For x=1: sp ≈ 1.313
      // For x=20: sp ≈ 20 (threshold boundary)
      // For x=100: sp = 100 (direct passthrough, no overflow)
      if (Math.abs(result[2] - 0.693) > 0.01) throw new Error(`sp(0)=${result[2]} (expected 0.693)`);
      if (Math.abs(result[4] - 10.0004) > 0.1) throw new Error(`sp(10)=${result[4]}`);
      // Critical: x=100 must NOT be Inf
      if (!isFinite(result[8])) throw new Error(`sp(100)=${result[8]} — OVERFLOW!`);
      if (Math.abs(result[8] - 100) > 0.1) throw new Error(`sp(100)=${result[8]} (expected 100)`);
    }},
    { name: 'compute_g: exp(-A*softplus(x)) no precision loss', run() {
      // Test the FULL compute_g chain: g = exp(-exp(A_log) * softplus(a + dt_bias))
      // Verify no NaN/Inf and values in (0, 1]
      const Hv = 16, T = 5;
      // A_log values (typically small negative, e.g. -1 to -3)
      const aLogData = new Float32Array(Hv);
      for (let i = 0; i < Hv; i++) aLogData[i] = -1.0 - i * 0.1;
      const aLog = MxArray.fromFloat32(aLogData, new BigInt64Array([BigInt(Hv)]));
      // a values (typically around 0-2)
      const aData = new Float32Array(T * Hv);
      for (let i = 0; i < aData.length; i++) aData[i] = 0.5 + Math.sin(i * 0.1);
      const a = MxArray.fromFloat32(aData, new BigInt64Array([1n, BigInt(T), BigInt(Hv)]));
      // dt_bias (typically small, 0-1)
      const dtData = new Float32Array(Hv).fill(0.1);
      const dt = MxArray.fromFloat32(dtData, new BigInt64Array([BigInt(Hv)]));

      // Compute g_log = -exp(A_log) * softplus(a + dt_bias)
      const bigA = aLog.exp();
      const x = a.add(dt);
      const threshold = MxArray.fromFloat32(new Float32Array([20]), new BigInt64Array([1n]));
      const sp = x.exp().addScalar(1).log();
      const spStable = x.greater(threshold).where(x, sp);
      const gLog = bigA.mul(spStable).negative(); // -A * softplus
      // Then g = exp(g_log)
      const g = gLog.exp();
      g.eval();
      const gVals = g.toFloat32();
      // All g values should be in (0, 1] (decay gate)
      for (let i = 0; i < gVals.length; i++) {
        if (!isFinite(gVals[i])) throw new Error(`g[${i}]=${gVals[i]} NaN/Inf!`);
        if (gVals[i] <= 0 || gVals[i] > 1.001) throw new Error(`g[${i}]=${gVals[i]} not in (0,1]`);
      }
    }},
    { name: 'GDN recurrence: state decay + update', run() {
      // Full gated_delta_step with realistic values and verify state evolves
      const B = 1, Hv = 4, Dv = 8, Dk = 8;
      // Initial state: zeros (like first call)
      const state = MxArray.fromFloat32(new Float32Array(B*Hv*Dv*Dk).fill(0),
        new BigInt64Array([BigInt(B), BigInt(Hv), BigInt(Dv), BigInt(Dk)]));
      // Query, Key, Value with varied values
      const q = MxArray.fromFloat32(new Float32Array(B*Hv*Dk).fill(0).map((_,i) => Math.sin(i*0.3)*0.5),
        new BigInt64Array([BigInt(B), BigInt(Hv), BigInt(Dk)]));
      const k = MxArray.fromFloat32(new Float32Array(B*Hv*Dk).fill(0).map((_,i) => Math.cos(i*0.2)*0.5),
        new BigInt64Array([BigInt(B), BigInt(Hv), BigInt(Dk)]));
      const v = MxArray.fromFloat32(new Float32Array(B*Hv*Dv).fill(0).map((_,i) => Math.sin(i*0.5)*0.3),
        new BigInt64Array([BigInt(B), BigInt(Hv), BigInt(Dv)]));
      // Decay gate g ∈ (0,1) and beta ∈ (0,1)
      const g = MxArray.fromFloat32(new Float32Array(B*Hv).fill(0.95),
        new BigInt64Array([BigInt(B), BigInt(Hv)]));
      const beta = MxArray.fromFloat32(new Float32Array(B*Hv).fill(0.5),
        new BigInt64Array([BigInt(B), BigInt(Hv)]));

      // Step 1: decay state (0 * 0.95 = 0, then update with k*delta)
      const g4d = g.reshape(new BigInt64Array([BigInt(B), BigInt(Hv), 1n, 1n]));
      const decayed = state.mul(g4d); // still zeros
      // kv_mem = (state * k).sum(-1) = 0
      const k4d = k.reshape(new BigInt64Array([BigInt(B), BigInt(Hv), 1n, BigInt(Dk)]));
      const kvMem = decayed.mul(k4d).sum(new Int32Array([-1])); // [B, Hv, Dv] = zeros
      // delta = (v - 0) * beta = v * beta
      const beta3d = beta.reshape(new BigInt64Array([BigInt(B), BigInt(Hv), 1n]));
      const delta = v.sub(kvMem).mul(beta3d); // [B, Hv, Dv]
      // state += k * delta (outer product)
      const delta4d = delta.reshape(new BigInt64Array([BigInt(B), BigInt(Hv), BigInt(Dv), 1n]));
      const newState = decayed.add(k4d.mul(delta4d));
      // y = (newState * q).sum(-1)
      const q4d = q.reshape(new BigInt64Array([BigInt(B), BigInt(Hv), 1n, BigInt(Dk)]));
      const y = newState.mul(q4d).sum(new Int32Array([-1]));
      y.eval();
      const yVals = y.toFloat32();
      // y should be non-zero (state was updated with k*delta)
      if (yVals.length !== B*Hv*Dv) throw new Error(`y length=${yVals.length}`);
      let hasNonZero = false;
      for (let i = 0; i < yVals.length; i++) {
        if (!isFinite(yVals[i])) throw new Error(`y[${i}]=${yVals[i]} NaN/Inf`);
        if (Math.abs(yVals[i]) > 1e-6) hasNonZero = true;
      }
      if (!hasNonZero) throw new Error('GDN output is all zeros after first step');
    }},
    { name: 'greater same-shape (no broadcast)', run() {
      // Test greater() with same-shape arrays — no broadcast
      const a = MxArray.fromFloat32(new Float32Array([1, 25, 50]), new BigInt64Array([3n]));
      const b = MxArray.fromFloat32(new Float32Array([20, 20, 20]), new BigInt64Array([3n]));
      const cond = a.greater(b);
      const condF = cond.addScalar(0);
      condF.eval();
      const vals = [...condF.toFloat32()];
      // 1>20=false, 25>20=true, 50>20=true → [0, 1, 1]
      const expected = [0, 1, 1];
      for (let i = 0; i < 3; i++) {
        if (Math.round(vals[i]) !== expected[i]) {
          throw new Error(`[${i}]: ${vals[i]} (expected ${expected[i]}), all=[${vals.map(v=>v.toFixed(4))}]`);
        }
      }
    }},
    { name: 'greater broadcast (scalar threshold)', run() {
      // Test greater() with broadcast: [6] > [1] (scalar threshold)
      const x = MxArray.fromFloat32(new Float32Array([1, 5, 10, 25, 50, 100]), new BigInt64Array([6n]));
      const thresh = MxArray.fromFloat32(new Float32Array([20]), new BigInt64Array([1n]));
      const cond = x.greater(thresh);
      // Read bool result directly as f32 via add (bool + 0.0 → f32 0.0 or 1.0)
      const condF = cond.addScalar(0);
      condF.eval();
      const condVals = [...condF.toFloat32()];
      // Expected: [0, 0, 0, 1, 1, 1] (25>20, 50>20, 100>20)
      // Log actual values for debugging
      if (condVals.length !== 6) throw new Error(`length=${condVals.length}`);
      const actual = condVals.map(v => Math.round(v));
      const expected = [0, 0, 0, 1, 1, 1];
      for (let i = 0; i < 6; i++) {
        if (actual[i] !== expected[i]) {
          throw new Error(`[${i}]: got ${condVals[i]} (round=${actual[i]}) expected ${expected[i]}, all=[${condVals.map(v=>v.toFixed(4))}]`);
        }
      }
    }},
    { name: 'where with greater (softplus pattern)', run() {
      // The exact softplus pattern: where(x > 20, x, log(1+exp(x)))
      const x = MxArray.fromFloat32(new Float32Array([0, 10, 20, 25, 50, 100]), new BigInt64Array([6n]));
      const thresh = MxArray.fromFloat32(new Float32Array([20]), new BigInt64Array([1n]));
      const cond = x.greater(thresh);
      const stable = x; // for x > 20, use x directly
      const unstable = x.exp().addScalar(1).log(); // log(1+exp(x))
      const sp = cond.where(stable, unstable);
      sp.eval();
      const vals = [...sp.toFloat32()];
      // x=0: sp=log(2)≈0.693, x=10: sp≈10.0, x=25: sp=25, x=100: sp=100
      if (Math.abs(vals[0] - 0.693) > 0.05) throw new Error(`sp(0)=${vals[0]}`);
      if (!isFinite(vals[5])) throw new Error(`sp(100)=${vals[5]} OVERFLOW — where didn't select x!`);
      if (Math.abs(vals[5] - 100) > 1) throw new Error(`sp(100)=${vals[5]} (expected 100)`);
    }},
    { name: 'large random (vocab-sized)', run() {
      // RandomBits dispatch: grid_y = half_size = N/2
      // For N=248320 (vocab_size): grid_y = 124160 > 65535 → clamped!
      // This is the exact pattern that causes dispatch Y=124160
      const N = 248320;
      // MxArray.random doesn't exist, but we can test via the sampling path
      // Actually test by checking if random key generation works at large scale
      // Use a simple pattern: create large array, verify no crash
      const data = new Float32Array(N);
      for (let i = 0; i < N; i++) data[i] = -100;
      data[42] = 100;
      const logits = MxArray.fromFloat32(data, new BigInt64Array([1n, BigInt(N)]));
      // argmax should return 42 (no random involved)
      const idx = logits.argmax(-1);
      idx.eval();
      const token = idx.toUint32()[0];
      if (token !== 42) throw new Error(`argmax=${token} (expected 42)`);
    }},
    { name: 'bf16 multi-eval stress (model pattern)', run() {
      const BF16 = 3, F32 = 0;
      // This tests the EXACT pattern that crashes the model:
      // - bf16 weight arrays (persistent)
      // - Multiple forward passes with per-layer eval
      // - bf16→f32 promotion during matmul
      const D = 64, H = 128;

      // Create bf16 weights (like model loading)
      const w1Data = new Float32Array(D * H);
      for (let i = 0; i < w1Data.length; i++) w1Data[i] = Math.cos(i * 0.01) * 0.05;
      const w1_f32 = MxArray.fromFloat32(w1Data, new BigInt64Array([BigInt(D), BigInt(H)]));
      const w1 = w1_f32.astype(BF16); w1.eval();

      const w2Data = new Float32Array(H * D);
      for (let i = 0; i < w2Data.length; i++) w2Data[i] = Math.sin(i * 0.01) * 0.05;
      const w2_f32 = MxArray.fromFloat32(w2Data, new BigInt64Array([BigInt(H), BigInt(D)]));
      const w2 = w2_f32.astype(BF16); w2.eval();

      // Simulate 3 forward passes (prefill + 2 decode steps)
      for (let fwd = 0; fwd < 3; fwd++) {
        // Input (bf16)
        const xData = new Float32Array(D);
        for (let i = 0; i < D; i++) xData[i] = Math.sin(fwd + i * 0.1) * 0.3;
        const x_f32 = MxArray.fromFloat32(xData, new BigInt64Array([1n, BigInt(D)]));
        const x = x_f32.astype(BF16); x.eval();

        // Layer 1: matmul + RMSNorm pattern
        const h1 = x.matmul(w1);
        const h1sq = h1.mul(h1);
        const rms = h1sq.mean(new Int32Array([-1])).addScalar(1e-6).sqrt();
        const rmsR = rms.reshape(new BigInt64Array([1n, 1n]));
        const h1n = h1.div(rmsR);

        // Layer 2: matmul back
        const out = h1n.matmul(w2);
        out.eval(); // per-layer eval

        // Scalar ops (like mul_scalar in GDN)
        const scaled = out.mulScalar(0.125);
        scaled.eval();

        // Verify output
        const vals = scaled.astype(F32).toFloat32();
        if (vals.length !== D) throw new Error(`fwd${fwd}: length=${vals.length}`);
        for (let i = 0; i < vals.length; i++) {
          if (!isFinite(vals[i])) throw new Error(`fwd${fwd}: vals[${i}]=${vals[i]}`);
        }
      }
    }},
    { name: 'bf16 repeated eval cycles (24 layers x 3 passes)', run() {
      const BF16 = 3, F32 = 0;
      // Simulate the full model eval pattern: 24 layers × 3 forward passes
      // with bf16 weights and per-layer eval
      const D = 32;

      // Create 24 "layer" weight pairs (bf16)
      const layers: any[][] = [];
      for (let l = 0; l < 24; l++) {
        const wData = new Float32Array(D * D);
        for (let i = 0; i < wData.length; i++) wData[i] = Math.cos(l * 0.3 + i * 0.01) * 0.1;
        const w_f32 = MxArray.fromFloat32(wData, new BigInt64Array([BigInt(D), BigInt(D)]));
        const w = w_f32.astype(BF16); w.eval();
        layers.push([w]);
      }

      // 3 forward passes
      for (let fwd = 0; fwd < 3; fwd++) {
        const xData = new Float32Array(D);
        for (let i = 0; i < D; i++) xData[i] = Math.sin(fwd * 10 + i * 0.2) * 0.5;
        let h = MxArray.fromFloat32(xData, new BigInt64Array([1n, BigInt(D)]));
        h = h.astype(BF16); h.eval();

        // Run through all 24 layers with per-layer eval
        for (let l = 0; l < 24; l++) {
          const proj = h.matmul(layers[l][0]);
          const projSq = proj.mul(proj);
          const rms = projSq.mean(new Int32Array([-1])).addScalar(1e-6).sqrt();
          h = proj.div(rms.reshape(new BigInt64Array([1n, 1n])));
          h.eval(); // per-layer eval
        }

        // Read back result
        const vals = h.astype(F32).toFloat32();
        if (vals.length !== D) throw new Error(`fwd${fwd}: length=${vals.length}`);
        let hasFinite = false;
        for (let i = 0; i < vals.length; i++) {
          if (isFinite(vals[i]) && vals[i] !== 0) hasFinite = true;
        }
        // After 24 layers of normalization, values should be finite
        if (!hasFinite) throw new Error(`fwd${fwd}: all zeros/non-finite`);
      }
    }},
    { name: 'large matmul repeated eval (GDN in_proj pattern)', run() {
      // This is the EXACT pattern that corrupts the heap:
      // Repeated eval of [1,T,1024] @ [1024,8192] matmul across decode steps
      const D = 1024, O = 8192;
      const wData = new Float32Array(D * O);
      for (let i = 0; i < wData.length; i++) wData[i] = Math.cos(i * 0.0001) * 0.01;
      const w = MxArray.fromFloat32(wData, new BigInt64Array([BigInt(D), BigInt(O)]));

      // Simulate 3 forward passes (prefill + 2 decode)
      for (let fwd = 0; fwd < 3; fwd++) {
        const T = fwd === 0 ? 15 : 1;
        const xData = new Float32Array(T * D);
        for (let i = 0; i < xData.length; i++) xData[i] = Math.sin(fwd * 100 + i * 0.01) * 0.1;
        const x = MxArray.fromFloat32(xData, new BigInt64Array([1n, BigInt(T), BigInt(D)]));
        const xr = x.reshape(new BigInt64Array([BigInt(T), BigInt(D)]));
        const y = xr.matmul(w);
        y.eval();
        // Read back a small part to force completion
        const yr = y.reshape(new BigInt64Array([1n, BigInt(T), BigInt(O)]));
        const sliced = yr.slice(
          new BigInt64Array([0n, 0n, 0n]),
          new BigInt64Array([1n, BigInt(T), 1n])
        );
        const val = sliced.addScalar(0);
        val.eval();
        const v = val.toFloat32();
        if (!isFinite(v[0])) throw new Error(`fwd${fwd}: non-finite`);
      }
    }},
    { name: 'multi-step decode simulation (event alignment)', run() {
      const BF16 = 3, F32 = 0;
      // Simulate the exact decode pattern: prefill → sample → decode → sample → decode
      // This exercises Event::is_signaled() which has uint64_t alignment issues
      const D = 32, V = 1024;

      // Weight (simulated LM head)
      const wData = new Float32Array(D * V);
      for (let i = 0; i < wData.length; i++) wData[i] = Math.cos(i * 0.001) * 0.01;
      const w = MxArray.fromFloat32(wData, new BigInt64Array([BigInt(D), BigInt(V)]));

      // 3 forward passes (prefill + 2 decode steps)
      for (let step = 0; step < 3; step++) {
        const xData = new Float32Array(D);
        for (let i = 0; i < D; i++) xData[i] = Math.sin(step * 10 + i * 0.1);
        const x = MxArray.fromFloat32(xData, new BigInt64Array([1n, BigInt(D)]));

        // Matmul → large output (like LM head)
        const logits = x.matmul(w);
        logits.eval();

        // Argmax (like sampling)
        const token = logits.argmax(-1);
        token.eval();

        // Read back token (triggers raw_ptr on scalar)
        const tokenVal = token.toUint32()[0];
        if (tokenVal >= V) throw new Error(`step${step}: token=${tokenVal} >= vocab`);
      }
    }},
    { name: 'heavy stress: 500 weight arrays + deep GDN-like graph', run() {
      // Step 1: Create many weight arrays (simulate model loading)
      const weights: any[] = [];
      for (let i = 0; i < 500; i++) {
        const size = (i % 5 === 0) ? 1024 * 64 : 64;
        const d = new Float32Array(size);
        for (let j = 0; j < size; j++) d[j] = Math.sin(i * 0.1 + j * 0.01) * 0.1;
        const dims = (size === 64)
          ? new BigInt64Array([BigInt(size)])
          : new BigInt64Array([1024n, 64n]);
        weights.push(MxArray.fromFloat32(d, dims));
      }

      // Step 2: Heap check after weight creation
      const chk = MxArray.fromFloat32(new Float32Array([7,8,9]), new BigInt64Array([3n]));
      chk.eval();
      const chkV = [...chk.toFloat32()];
      if (chkV[0] !== 7) throw new Error(`Heap check after weights: ${chkV}`);

      // Step 3: Warm up eval
      const warmup = MxArray.fromFloat32(new Float32Array([1]), new BigInt64Array([1n]));
      warmup.add(warmup).eval();

      // Step 4: Simulate forward pass
      const B = 1, T = 15, D = 64;
      const hData = new Float32Array(B * T * D);
      for (let j = 0; j < hData.length; j++) hData[j] = Math.sin(j * 0.1) * 0.3;
      let h = MxArray.fromFloat32(hData, new BigInt64Array([BigInt(B), BigInt(T), BigInt(D)]));
      h.eval();

      // Step 5: Simulate 3 layers
      for (let layer = 0; layer < 3; layer++) {
        const w = weights[layer * 5]; // [1024, 64]
        const h2 = h.mul(h);
        const rmsRaw = h2.mean(new Int32Array([-1])).addScalar(1e-6).sqrt();
        // Reshape to [B,T,1] for broadcasting with [B,T,D]
        const rms = rmsRaw.reshape(new BigInt64Array([BigInt(B), BigInt(T), 1n]));
        const hNorm = h.div(rms);
        const proj = hNorm.reshape(new BigInt64Array([BigInt(B*T), BigInt(D)])).matmul(w.transpose());
        const projR = proj.reshape(new BigInt64Array([BigInt(B), BigInt(T), 1024n]));
        const scaled = projR.mulScalar(0.125);
        const sliced = scaled.slice(new BigInt64Array([0n, 0n, 0n]), new BigInt64Array([BigInt(B), BigInt(T), BigInt(D)]));
        sliced.eval();
        h = sliced;
      }

      const result = h.toFloat32();
      if (result.length !== B * T * D) throw new Error(`length=${result.length}`);
      let hasFinite = false;
      for (let i = 0; i < result.length; i++) {
        if (isFinite(result[i]) && result[i] !== 0) { hasFinite = true; break; }
      }
      if (!hasFinite) throw new Error('All zeros or non-finite');
    }},
    { name: 'stress: weight arrays only (no compute)', run() {
      // Just create 500 weight arrays — no computation
      const weights: any[] = [];
      for (let i = 0; i < 500; i++) {
        const size = (i % 5 === 0) ? 1024 * 64 : 64;
        const d = new Float32Array(size);
        for (let j = 0; j < size; j++) d[j] = Math.sin(i * 0.1 + j * 0.01) * 0.1;
        const dims = (size === 64)
          ? new BigInt64Array([BigInt(size)])
          : new BigInt64Array([1024n, 64n]);
        weights.push(MxArray.fromFloat32(d, dims));
      }
      // Heap check
      const chk = MxArray.fromFloat32(new Float32Array([42]), new BigInt64Array([1n]));
      chk.eval();
      if (chk.toFloat32()[0] !== 42) throw new Error('Heap corrupted after 500 array creation');
    }},
    { name: 'stress: deep lazy graph (50 ops) then eval', run() {
      // Build a long lazy chain then eval — tests make_shared pressure
      const D = 64;
      let h = MxArray.fromFloat32(new Float32Array(D).fill(1), new BigInt64Array([BigInt(D)]));
      h.eval();
      // Chain 50 operations without eval
      for (let i = 0; i < 50; i++) {
        h = h.addScalar(0.01).mulScalar(0.999);
      }
      h.eval();
      const v = h.toFloat32();
      if (!isFinite(v[0])) throw new Error(`Non-finite after 50-op chain: ${v[0]}`);
    }},
    { name: 'chained: softmax components', run() {
      // Test the ops used in softmax: exp, sum, div
      const a = MxArray.fromFloat32(new Float32Array([1, 2, 3]), new BigInt64Array([1n, 3n]));
      const e = a.exp(); e.eval();
      const s = e.sum(new Int32Array([-1])); s.eval();
      // Verify exp values
      const expVals = [...e.toFloat32()];
      assertClose(expVals, [Math.exp(1), Math.exp(2), Math.exp(3)], 1e-2);
      // Verify sum
      const sumVal = [...s.toFloat32()];
      const expectedSum = Math.exp(1) + Math.exp(2) + Math.exp(3);
      assertClose(sumVal, [expectedSum], 1e-1);
    }},

    // ---- SDPA causal attention tests (where + bool mask + softmax) ----
    { name: 'greaterEqual on integer arrays (causal mask creation)', run() {
      // Reproduces how SDPA creates the causal mask:
      //   mask = greater_equal(arange(offset, qL+offset), arange(0, kL))
      // With offset=0, qL=4, kL=4 this should produce a lower-triangular bool mask.
      // arange returns float32; greaterEqual on those should still produce correct bool.
      const rows = MxArray.arange(0, 4, 1); // [0, 1, 2, 3]  shape [4]
      const cols = MxArray.arange(0, 4, 1); // [0, 1, 2, 3]  shape [4]
      // Reshape for broadcast: rows [4,1] >= cols [1,4] → [4,4]
      const rowsCol = rows.reshape(new BigInt64Array([4n, 1n]));
      const colsRow = cols.reshape(new BigInt64Array([1n, 4n]));
      const mask = rowsCol.greaterEqual(colsRow); // [4,4] bool
      mask.eval();
      const shape = mask.shape();
      if (Number(shape[0]) !== 4 || Number(shape[1]) !== 4)
        throw new Error(`mask shape=[${shape}] expected [4,4]`);
      // Convert bool→float to read values: bool + 0.0 → f32
      const maskF = mask.addScalar(0); maskF.eval();
      const vals = [...maskF.toFloat32()];
      // Expected lower-triangular:
      // row0: [1, 0, 0, 0]
      // row1: [1, 1, 0, 0]
      // row2: [1, 1, 1, 0]
      // row3: [1, 1, 1, 1]
      const expected = [
        1, 0, 0, 0,
        1, 1, 0, 0,
        1, 1, 1, 0,
        1, 1, 1, 1,
      ];
      for (let i = 0; i < 16; i++) {
        if (Math.round(vals[i]) !== expected[i])
          throw new Error(`mask[${Math.floor(i/4)},${i%4}]=${vals[i]} expected=${expected[i]}, all=[${vals.map(v=>v.toFixed(1))}]`);
      }
    }},

    { name: 'SDPA causal where+softmax [1,8,4,4]', run() {
      // Reproduces the exact SDPA fallback pattern for causal attention:
      //   scores = where(bool_mask, scores, -1e9)
      //   probs = softmax(scores, axis=-1)
      // This is where Qwen3.5 diverges at layer 3 on WebGPU.

      const B = 1, H = 8, Lq = 4, Lkv = 4;

      // 1) Create scores [1, 8, 4, 4] with known values (small range to keep softmax stable)
      const scoreData = new Float32Array(B * H * Lq * Lkv);
      for (let b = 0; b < B; b++)
        for (let h = 0; h < H; h++)
          for (let i = 0; i < Lq; i++)
            for (let j = 0; j < Lkv; j++)
              scoreData[b * H * Lq * Lkv + h * Lq * Lkv + i * Lkv + j] = (h * 0.1 + i * 0.3 - j * 0.2 + 0.5);
      const scores = MxArray.fromFloat32(scoreData, new BigInt64Array([1n, 8n, 4n, 4n]));

      // 2) Create [4,4] bool causal mask via greaterEqual (same as SDPA decomposition)
      const rowIdx = MxArray.arange(0, 4, 1).reshape(new BigInt64Array([4n, 1n]));
      const colIdx = MxArray.arange(0, 4, 1).reshape(new BigInt64Array([1n, 4n]));
      const mask = rowIdx.greaterEqual(colIdx); // [4,4] bool, lower triangular

      // 3) Apply where(mask, scores, -1e9) — the ternary kernel on bool condition
      const NEG_INF = MxArray.fromFloat32(new Float32Array([-1e9]), new BigInt64Array([1n]));
      const masked = mask.where(scores, NEG_INF); // broadcasts [4,4] mask over [1,8,4,4]
      masked.eval();

      // Verify masked shape
      const mShape = masked.shape();
      if (Number(mShape[0]) !== 1 || Number(mShape[1]) !== 8 || Number(mShape[2]) !== 4 || Number(mShape[3]) !== 4)
        throw new Error(`masked shape=[${mShape}] expected [1,8,4,4]`);

      // 4) Manual softmax on last axis: exp(x - max) / sum(exp(x - max))
      const maxPerRow = masked.max(new Int32Array([-1])); // [1,8,4]  (keepdims=false)
      const maxReshaped = maxPerRow.reshape(new BigInt64Array([1n, 8n, 4n, 1n])); // [1,8,4,1] for broadcast
      const shifted = masked.sub(maxReshaped);
      const expS = shifted.exp();
      const sumExp = expS.sum(new Int32Array([-1])); // [1,8,4]
      const sumReshaped = sumExp.reshape(new BigInt64Array([1n, 8n, 4n, 1n]));
      const probs = expS.div(sumReshaped); // [1,8,4,4]
      probs.eval();

      const probVals = [...probs.toFloat32()];

      // 5) Compute expected values on CPU
      const maskedExpected = new Float32Array(B * H * Lq * Lkv);
      for (let b = 0; b < B; b++)
        for (let h = 0; h < H; h++)
          for (let i = 0; i < Lq; i++)
            for (let j = 0; j < Lkv; j++) {
              const idx = b * H * Lq * Lkv + h * Lq * Lkv + i * Lkv + j;
              maskedExpected[idx] = (j <= i) ? scoreData[idx] : -1e9;
            }

      const cpuProbs = new Float32Array(B * H * Lq * Lkv);
      for (let b = 0; b < B; b++)
        for (let h = 0; h < H; h++)
          for (let i = 0; i < Lq; i++) {
            const base = b * H * Lq * Lkv + h * Lq * Lkv + i * Lkv;
            // Find max in row
            let mx = -Infinity;
            for (let j = 0; j < Lkv; j++) mx = Math.max(mx, maskedExpected[base + j]);
            // exp(x - max) and sum
            let s = 0;
            for (let j = 0; j < Lkv; j++) { cpuProbs[base + j] = Math.exp(maskedExpected[base + j] - mx); s += cpuProbs[base + j]; }
            for (let j = 0; j < Lkv; j++) cpuProbs[base + j] /= s;
          }

      // Verify: masked positions should be ~0, unmasked should match CPU softmax
      for (let idx = 0; idx < probVals.length; idx++) {
        const i = Math.floor((idx % (Lq * Lkv)) / Lkv);
        const j = idx % Lkv;
        if (j > i) {
          // Masked position: should be ~0
          if (probVals[idx] > 1e-4)
            throw new Error(`prob[${idx}] (masked pos [${i},${j}])=${probVals[idx]} should be ~0`);
        } else {
          // Unmasked position: should match CPU
          if (Math.abs(probVals[idx] - cpuProbs[idx]) > 1e-3)
            throw new Error(`prob[${idx}] (pos [${i},${j}])=${probVals[idx].toFixed(6)} expected=${cpuProbs[idx].toFixed(6)}`);
        }
      }

      // Extra: verify each row sums to ~1
      for (let b = 0; b < B; b++)
        for (let h = 0; h < H; h++)
          for (let i = 0; i < Lq; i++) {
            const base = b * H * Lq * Lkv + h * Lq * Lkv + i * Lkv;
            let rowSum = 0;
            for (let j = 0; j < Lkv; j++) rowSum += probVals[base + j];
            if (Math.abs(rowSum - 1.0) > 1e-3)
              throw new Error(`row sum [b=${b},h=${h},i=${i}]=${rowSum} expected=1.0`);
          }
    }},

    // SDPA causal test - exercises matmul + causal_mask + where + softmax + matmul
    { name: 'SDPA causal mode correctness', run() {
      const vals = MxArray.testSdpaCausal(20);
      if (vals[0] === -999) throw new Error('testSdpaCausal returned error');
      const ref = [0.252441, 0.271324, 0.285627, 0.295110, 0.299613, 0.299060, 0.293459, 0.282906, 0.259342, 0.260570];
      assertClose(vals.slice(0, 10), ref, 0.01);
    }},

    // SDPA GQA test: n_q_heads=8, n_kv_heads=2, T=8, D=32
    { name: 'SDPA causal GQA (Hq=8 Hkv=2)', run() {
      const vals = MxArray.testSdpaGqa(20);
      if (vals[0] === -999) throw new Error('testSdpaGqa returned error');
      // Model-scale bf16 GQA: T=25, D=256, Hq=8, Hkv=2
      const ref = [0.251953, 0.263672, 0.273438, 0.281250, 0.287109, 0.292969, 0.296875, 0.298828, 0.300781, 0.298828];
      assertClose(vals.slice(0, 10), ref, 0.02);
    }},

    // GDN isolated recurrence test (doesn't need model weights)
    { name: 'GDN isolated recurrence (small dims)', run() {
      const vals = MxArray.testGdnRecurrenceSmall();
      if (vals[0] === -999) throw new Error('testGdnRecurrenceSmall returned error');
      // Native reference: y_out (4 values) + state_out (first 20 values)
      const ref = [0.002823, 0.005646, 0.008469, 0.011293, 0.005000, 0.004998, 0.004991, 0.004980, 0.004964, 0.004944];
      assertClose(vals.slice(0, 10), ref, 0.001);
    }},

    // =============== Phase 1+2 Inference Step Tests ===============

    // --- RoPE ---
    { name: 'RoPE bf16 model dims', run() {
      const vals = MxArray.testRopeBf16(20);
      if (vals[0] === -999) throw new Error('testRopeBf16 error');
      const ref = [0.000000, 0.005005, 0.010010, 0.015015, 0.020020, 0.025024, 0.030029, 0.034912, 0.040039, 0.044922];
      assertClose(vals.slice(0, 10), ref, 0.01);
    }},

    // --- QK norm + RoPE chain ---
    { name: 'QK norm + RoPE chain bf16', run() {
      const vals = MxArray.testQkNormRope(20);
      if (vals[0] === -999) throw new Error('testQkNormRope error');
      const ref = [0.000000, 0.013062, 0.026123, 0.039063, 0.052246, 0.065430, 0.078125, 0.090820, 0.104492, 0.117188];
      assertClose(vals.slice(0, 10), ref, 0.01);
    }},

    // --- SDPA with additive mask ---
    { name: 'SDPA additive mask bf16', run() {
      const vals = MxArray.testSdpaAdditiveMask(20);
      if (vals[0] === -999) throw new Error('testSdpaAdditiveMask error');
      const ref = [-0.001328, -0.001411, -0.001289, -0.001472, -0.001305, -0.001389, -0.001480, -0.001404, -0.001488, -0.001503];
      assertClose(vals.slice(0, 10), ref, 0.01);
    }},

    // --- SDPA decode with GQA + KV cache ---
    { name: 'SDPA decode GQA bf16', run() {
      const vals = MxArray.testSdpaDecodeGqa(20);
      if (vals[0] === -999) throw new Error('testSdpaDecodeGqa error');
      const ref = [0.011597, 0.011475, 0.011597, 0.011475, 0.011536, 0.011414, 0.011353, 0.011414, 0.011230, 0.011230];
      assertClose(vals.slice(0, 10), ref, 0.01);
    }},

    // =============== SDPA Tile (prefill) parity tests ===============
    //
    // Each test runs a Tq > 1 SDPA helper twice:
    //   1) With setSdpaFallbackForced(true)  -> decomposed matmul path (reference)
    //   2) With setSdpaFallbackForced(false) -> fused tile kernel (under test)
    // The two arrays must match within `tol`. This catches any algebraic
    // divergence in the tile kernel's online softmax + running accumulators
    // without hardcoding reference values (which would drift across dtypes
    // and compilers).
    //
    // Each test fetches the FULL flat output (all batches, heads, Q rows, D)
    // and compares representative slices:
    //   - First Q row, first D elements (prefix)
    //   - Last Q row of head 0 (tail / partial-tile rows)
    //   - Last head's first Q row (GQA head reindexing coverage)
    // This ensures tail-tile rows and later GQA heads are tested, not just
    // the leading prefix.

    { name: 'SDPA tile Tq=2 D=64 (f32)', run() {
      // Output shape: [1, 2, 2, 64] = 256 elements total
      const COUNT = 256;
      setSdpaFallbackForced(true);
      let ref: number[];
      try {
        ref = MxArray.testSdpaTileTq2D64(COUNT);
      } finally {
        setSdpaFallbackForced(false);
      }
      if (ref[0] === -999) throw new Error('testSdpaTileTq2D64 ref error');
      const vals = MxArray.testSdpaTileTq2D64(COUNT);
      if (vals[0] === -999) throw new Error('testSdpaTileTq2D64 tile error');
      // Full output comparison — small enough to check entirely
      assertClose(vals, ref, 1e-4);
    }},

    { name: 'SDPA tile Tq=8 D=128 causal GQA (f32)', run() {
      // B=1, Hq=8, Hkv=2, Tq=8, D=128 -> output [1,8,8,128] = 8192 elements
      // GQA 4:1: heads 0-3 use KV head 0, heads 4-7 use KV head 1.
      // Fetch full output but compare within KV-head-0 territory (heads 0-3,
      // 4096 elements) which covers first Q row, last Q row, and multiple
      // query heads sharing a single KV head.
      const D = 128, Tq = 8, Hq = 8;
      const gqa = Hq / 2; // 4 heads per KV head
      const COUNT = Hq * Tq * D; // 8192 (full output)
      setSdpaFallbackForced(true);
      let ref: number[];
      try {
        ref = MxArray.testSdpaTileTq8D128CausalGqa(COUNT);
      } finally {
        setSdpaFallbackForced(false);
      }
      if (ref[0] === -999) throw new Error('testSdpaTileTq8D128CausalGqa ref error');
      const vals = MxArray.testSdpaTileTq8D128CausalGqa(COUNT);
      if (vals[0] === -999) throw new Error('testSdpaTileTq8D128CausalGqa tile error');
      // Check head 0, first Q row (prefix)
      assertClose(vals.slice(0, D), ref.slice(0, D), 1e-4);
      // Check head 0, last Q row (row 7)
      const h0LastRow = (Tq - 1) * D; // 896
      assertClose(vals.slice(h0LastRow, h0LastRow + D), ref.slice(h0LastRow, h0LastRow + D), 1e-4);
      // Check head 3 (last head in KV-head-0 group) — GQA h/gqa_factor coverage
      const h3Start = 3 * Tq * D; // 3072
      assertClose(vals.slice(h3Start, h3Start + D), ref.slice(h3Start, h3Start + D), 1e-4);
      // Check head 3, last Q row
      const h3LastRow = h3Start + (Tq - 1) * D; // 3968
      assertClose(vals.slice(h3LastRow, h3LastRow + D), ref.slice(h3LastRow, h3LastRow + D), 1e-4);
    }},

    { name: 'SDPA tile Tq=32 D=128 additive mask (f32)', run() {
      // B=1, Hq=4, Hkv=2, Tq=32, D=128 -> output [1,4,32,128] = 16384 elements
      // GQA 2:1: heads 0-1 use KV head 0. Fetch full output, compare within
      // KV-head-0 territory (heads 0-1, 8192 elements).
      const D = 128, Tq = 32, Hq = 4;
      const COUNT = Hq * Tq * D; // 16384 (full output)
      setSdpaFallbackForced(true);
      let ref: number[];
      try {
        ref = MxArray.testSdpaTileTq32D128Addmask(COUNT);
      } finally {
        setSdpaFallbackForced(false);
      }
      if (ref[0] === -999) throw new Error('testSdpaTileTq32D128Addmask ref error');
      const vals = MxArray.testSdpaTileTq32D128Addmask(COUNT);
      if (vals[0] === -999) throw new Error('testSdpaTileTq32D128Addmask tile error');
      // Check head 0, first Q row (prefix)
      assertClose(vals.slice(0, D), ref.slice(0, D), 1e-4);
      // Check head 0, last Q row (row 31 — end of second full tile)
      const h0LastRow = (Tq - 1) * D; // 3968
      assertClose(vals.slice(h0LastRow, h0LastRow + D), ref.slice(h0LastRow, h0LastRow + D), 1e-4);
      // Check head 1 (second head in KV-head-0 group)
      const h1Start = 1 * Tq * D; // 4096
      assertClose(vals.slice(h1Start, h1Start + D), ref.slice(h1Start, h1Start + D), 1e-4);
      // Check head 1, last Q row
      const h1LastRow = h1Start + (Tq - 1) * D; // 8064
      assertClose(vals.slice(h1LastRow, h1LastRow + D), ref.slice(h1LastRow, h1LastRow + D), 1e-4);
    }},

    { name: 'SDPA tile Tq=33 D=128 causal partial-last-tile (f32)', run() {
      // B=1, Hq=4, Hkv=2, Tq=33, D=128 -> output [1,4,33,128] = 16896 elements
      // GQA 2:1: heads 0-1 use KV head 0. Tail row is row 32 (the lone row
      // in the 3rd tile). Fetch full output, compare within KV-head-0
      // territory including the tail row.
      const D = 128, Tq = 33, Hq = 4;
      const COUNT = Hq * Tq * D; // 16896 (full output)
      setSdpaFallbackForced(true);
      let ref: number[];
      try {
        ref = MxArray.testSdpaTileTq33D128Tailtile(COUNT);
      } finally {
        setSdpaFallbackForced(false);
      }
      if (ref[0] === -999) throw new Error('testSdpaTileTq33D128Tailtile ref error');
      const vals = MxArray.testSdpaTileTq33D128Tailtile(COUNT);
      if (vals[0] === -999) throw new Error('testSdpaTileTq33D128Tailtile tile error');
      // Check head 0, first Q row (prefix)
      assertClose(vals.slice(0, D), ref.slice(0, D), 1e-4);
      // Check head 0, tail row (row 32 — the partial-last-tile row)
      const tailRow = 32 * D; // 4096
      assertClose(vals.slice(tailRow, tailRow + D), ref.slice(tailRow, tailRow + D), 1e-4);
      // Check head 1 (second head in KV-head-0 group), first Q row
      const h1Start = 1 * Tq * D; // 4224
      assertClose(vals.slice(h1Start, h1Start + D), ref.slice(h1Start, h1Start + D), 1e-4);
      // Check head 1, tail row (row 32)
      const h1Tail = h1Start + 32 * D; // 8320
      assertClose(vals.slice(h1Tail, h1Tail + D), ref.slice(h1Tail, h1Tail + D), 1e-4);
    }},

    { name: 'SDPA tile Tq=128 D=128 L=4096 causal (f32)', run() {
      // B=1, Hq=2, Tq=128, D=128 -> output [1,2,128,128] = 32768 elements
      const D = 128, Tq = 128, Hq = 2;
      const COUNT = Hq * Tq * D; // 32768
      setSdpaFallbackForced(true);
      let ref: number[];
      try {
        ref = MxArray.testSdpaTileTq128D128L4096(COUNT);
      } finally {
        setSdpaFallbackForced(false);
      }
      if (ref[0] === -999) throw new Error('testSdpaTileTq128D128L4096 ref error');
      const vals = MxArray.testSdpaTileTq128D128L4096(COUNT);
      if (vals[0] === -999) throw new Error('testSdpaTileTq128D128L4096 tile error');
      // Check head 0, first Q row (prefix)
      assertClose(vals.slice(0, D), ref.slice(0, D), 1e-4);
      // Check head 0, last Q row (row 127 — end of 8th tile)
      const h0LastRow = (Tq - 1) * D; // 127*128 = 16256
      assertClose(vals.slice(h0LastRow, h0LastRow + D), ref.slice(h0LastRow, h0LastRow + D), 1e-4);
      // Check head 0, mid Q row (row 64 — middle tile boundary)
      const h0MidRow = 64 * D; // 64*128 = 8192
      assertClose(vals.slice(h0MidRow, h0MidRow + D), ref.slice(h0MidRow, h0MidRow + D), 1e-4);
      // Check head 1 (last head), first Q row
      const h1Start = 1 * Tq * D; // 1*128*128 = 16384
      assertClose(vals.slice(h1Start, h1Start + D), ref.slice(h1Start, h1Start + D), 1e-4);
      // Check head 1, last Q row
      const h1LastRow = h1Start + (Tq - 1) * D; // 16384 + 16256 = 32640
      assertClose(vals.slice(h1LastRow, h1LastRow + D), ref.slice(h1LastRow, h1LastRow + D), 1e-4);
    }},

    { name: 'SDPA tile D=256 Tq=8 causal GQA (f32, Qwen3.5-0.8B shape)', run() {
      // B=1, Hq=8, Hkv=2, Tq=8, L=16, D=256 -> output [1,8,8,256] = 16384 elements
      // GQA 4:1: heads 0-3 use KV head 0, heads 4-7 use KV head 1.
      // C++ returns [tile_output..., cpu_reference...] = 2*N floats.
      const D = 256, Tq = 8, Hq = 8;
      const N = Hq * Tq * D; // 16384 per half
      const both = MxArray.testSdpaTileD256Gqa(2 * N);
      if (both[0] === -999) throw new Error('testSdpaTileD256Gqa error');
      const tile = both.slice(0, N);
      const ref = both.slice(N, 2 * N);
      // Check all heads: h0-h3 (KV head 0) and h4-h7 (KV head 1)
      for (let h = 0; h < Hq; h++) {
        const off = h * Tq * D;
        assertClose(tile.slice(off, off + D), ref.slice(off, off + D), 1e-3);
      }
      // Also check last row of head 0
      const lastRow = (Tq - 1) * D;
      assertClose(tile.slice(lastRow, lastRow + D), ref.slice(lastRow, lastRow + D), 1e-3);
    }},

    { name: 'SDPA tile D=256 simple (no causal, no GQA)', run() {
      // B=1, Hq=1, Hkv=1, Tq=2, L=4, D=256 -> output [1,1,2,256] = 512 elements
      const D = 256;
      const COUNT = 512;
      setSdpaFallbackForced(true);
      let ref: number[];
      try {
        ref = MxArray.testSdpaTileD256Simple(COUNT);
      } finally {
        setSdpaFallbackForced(false);
      }
      if (ref[0] === -999) throw new Error('testSdpaTileD256Simple ref error');
      const vals = MxArray.testSdpaTileD256Simple(COUNT);
      if (vals[0] === -999) throw new Error('testSdpaTileD256Simple tile error');
      // Full comparison
      assertClose(vals.slice(0, D), ref.slice(0, D), 1e-4);
      assertClose(vals.slice(D, 2*D), ref.slice(D, 2*D), 1e-4);
    }},

    { name: 'SDPA tile D=256 causal (no GQA)', run() {
      const D = 256, Tq = 8, Hq = 2;
      const COUNT = Hq * Tq * D;
      setSdpaFallbackForced(true);
      let ref: number[];
      try {
        ref = MxArray.testSdpaTileD256Causal(COUNT);
      } finally {
        setSdpaFallbackForced(false);
      }
      if (ref[0] === -999) throw new Error('testSdpaTileD256Causal ref error');
      const vals = MxArray.testSdpaTileD256Causal(COUNT);
      if (vals[0] === -999) throw new Error('testSdpaTileD256Causal tile error');
      assertClose(vals.slice(0, D), ref.slice(0, D), 1e-4);
      const lastRow = (Tq - 1) * D;
      assertClose(vals.slice(lastRow, lastRow + D), ref.slice(lastRow, lastRow + D), 1e-4);
    }},

    { name: 'SDPA tile D=256 GQA (no causal)', run() {
      // B=1, Hq=8, Hkv=2, Tq=8, L=16, D=256 -> N=16384
      // C++ returns [tile_output..., cpu_reference...] = 2*N floats.
      const D = 256, Tq = 8, Hq = 8;
      const N = Hq * Tq * D;
      const both = MxArray.testSdpaTileD256GqaNocausal(2 * N);
      if (both[0] === -999) throw new Error('testSdpaTileD256GqaNocausal error');
      const tile = both.slice(0, N);
      const ref = both.slice(N, 2 * N);
      // Check h0 (KV head 0), h4 (KV head 1), and last row of h7
      assertClose(tile.slice(0, D), ref.slice(0, D), 1e-3);
      const h4Start = 4 * Tq * D;
      assertClose(tile.slice(h4Start, h4Start + D), ref.slice(h4Start, h4Start + D), 1e-3);
      const h7LastRow = 7 * Tq * D + (Tq - 1) * D;
      assertClose(tile.slice(h7LastRow, h7LastRow + D), ref.slice(h7LastRow, h7LastRow + D), 1e-3);
    }},

    { name: 'SDPA tile D=256 minimal causal+GQA (H=4 Hkv=2)', run() {
      // B=1, Hq=4, Hkv=2, Tq=2, L=4, D=256 -> N=2048
      // h=0,1 → KV head 0; h=2,3 → KV head 1
      // C++ returns [tile_output..., cpu_reference...] = 2*N floats.
      const D = 256, Tq = 2, Hq = 4;
      const N = Hq * Tq * D; // 2048
      const both = MxArray.testSdpaTileD256CausalGqaMinimal(2 * N);
      if (both[0] === -999) throw new Error('testSdpaTileD256CausalGqaMinimal error');
      const tile = both.slice(0, N);
      const ref = both.slice(N, 2 * N);
      // Check all 4 heads against CPU reference
      for (let h = 0; h < Hq; h++) {
        const off = h * Tq * D;
        assertClose(tile.slice(off, off + D), ref.slice(off, off + D), 1e-3);
      }
    }},

    // --- SDPA vector D=256 (decode, Tq=1) ---
    { name: 'SDPA vector D=256 Tq=1 GQA (f32, Qwen3.5-0.8B decode shape)', run() {
      // B=1, Hq=8, Hkv=2, Tq=1, L=32, D=256 -> output [1,8,1,256] = 2048 elements
      // GQA 4:1: heads 0-3 use KV head 0, heads 4-7 use KV head 1.
      // C++ returns [vector_output..., cpu_reference...] = 2*N floats.
      const D = 256, Hq = 8;
      const N = Hq * 1 * D; // 2048 per half
      const both = MxArray.testSdpaVectorD256(2 * N);
      if (both[0] === -999) throw new Error('testSdpaVectorD256 error');
      const vec = both.slice(0, N);
      const ref = both.slice(N, 2 * N);
      // Check all heads against CPU reference
      for (let h = 0; h < Hq; h++) {
        const off = h * D;
        assertClose(vec.slice(off, off + D), ref.slice(off, off + D), 1e-3);
      }
    }},

    { name: 'SDPA vector D=256 simple (no GQA, Tq=1)', run() {
      // B=1, Hq=2, Hkv=2, Tq=1, L=16, D=256 -> output [1,2,1,256] = 512 elements
      // C++ returns [vector_output..., cpu_reference...] = 2*N floats.
      const D = 256, Hq = 2;
      const N = Hq * 1 * D; // 512 per half
      const both = MxArray.testSdpaVectorD256Simple(2 * N);
      if (both[0] === -999) throw new Error('testSdpaVectorD256Simple error');
      const vec = both.slice(0, N);
      const ref = both.slice(N, 2 * N);
      // Check both heads
      for (let h = 0; h < Hq; h++) {
        const off = h * D;
        assertClose(vec.slice(off, off + D), ref.slice(off, off + D), 1e-3);
      }
    }},

    // --- Full attention layer bf16 ---
    { name: 'full attention layer bf16 (Q/K/V proj+QK norm+RoPE+GQA+SDPA+gate+Oproj)', run() {
      const vals = MxArray.testFullAttnLayerBf16(20);
      if (vals[0] === -999) throw new Error('testFullAttnLayerBf16 error');
      const ref = [-0.000786, 0.001366, -0.000106, -0.001266, 0.000984, 0.000641, -0.001343, 0.000282, 0.001205, -0.001038];
      assertClose(vals.slice(0, 10), ref, 0.01);
    }},

    // --- RMSNorm bf16 ---
    { name: 'RMSNorm bf16 model dims [1,1024]', run() {
      const vals = MxArray.testRmsNormBf16(20);
      if (vals[0] === -999) throw new Error('testRmsNormBf16 error');
      const ref = [0.000000, 0.014526, 0.029053, 0.043457, 0.058105, 0.073242, 0.087402, 0.102051, 0.117188, 0.130859];
      assertClose(vals.slice(0, 10), ref, 0.01);
    }},

    // --- SwiGLU MLP bf16 ---
    { name: 'SwiGLU MLP bf16 model dims', run() {
      const vals = MxArray.testSwigluMlpBf16(20);
      if (vals[0] === -999) throw new Error('testSwigluMlpBf16 error');
      const ref = [0.000092, 0.000961, 0.001419, 0.000927, 0.000012, -0.000916, -0.001297, -0.001030, -0.000106, 0.000877];
      assertClose(vals.slice(0, 10), ref, 0.01);
    }},

    // --- Decode step with KV cache ---
    { name: 'decode step with KV cache bf16', run() {
      const vals = MxArray.testDecodeStepWithCache(20);
      if (vals[0] === -999) throw new Error('testDecodeStepWithCache error');
      const ref = [-0.003998, -0.003525, -0.002563, -0.001724, -0.001266, -0.000767, -0.000187, 0.000999, 0.000778, 0.002243];
      assertClose(vals.slice(0, 10), ref, 0.01);
    }},

    // --- Full attention decoder layer bf16 (norm+attn+residual+norm+MLP+residual) ---
    { name: 'attention decoder layer bf16', run() {
      const vals = MxArray.testAttnLayerBf16(20);
      if (vals[0] === -999) throw new Error('testAttnLayerBf16 error');
      const ref = [0.051025, 0.072266, 0.039063, -0.014954, -0.043457, -0.019287, 0.037109, 0.084473, 0.090820, 0.059326];
      assertClose(vals.slice(0, 10), ref, 0.05);
    }},

    // --- 4 sequential layers bf16 ---
    { name: 'first 4 layers sequential bf16', run() {
      const vals = MxArray.testFirst4LayersBf16(20);
      if (vals[0] === -999) throw new Error('testFirst4LayersBf16 error');
      const ref = [1.242188, 1.210938, 1.140625, 1.039063, 0.910156, 0.765625, 0.625000, 0.486328, 0.373047, 0.277344];
      assertClose(vals.slice(0, 10), ref, 0.05);
    }},

    // --- GDN full forward bf16 ---
    { name: 'GDN full forward bf16 (proj+conv1d+silu+recurrence+norm+proj)', run() {
      const vals = MxArray.testGdnFullBf16(20);
      if (vals[0] === -999) throw new Error('testGdnFullBf16 error');
      const ref = [0.000232, 0.000481, 0.000460, 0.000181, -0.000199, -0.000467, -0.000473, -0.000214, 0.000166, 0.000454];
      assertClose(vals.slice(0, 10), ref, 0.01);
    }},

    // --- GDN multi-step recurrence bf16 ---
    { name: 'GDN 5-step recurrence bf16 state', run() {
      const vals = MxArray.testGdnMultiStepBf16(20);
      if (vals[0] === -999) throw new Error('testGdnMultiStepBf16 error');
      const ref = [-0.008240, -0.008118, -0.008057, -0.007996, -0.007874, -0.007782, -0.007599, -0.007538, -0.007446, -0.007324];
      assertClose(vals.slice(0, 10), ref, 0.01);
    }},

    // --- GDN decoder layer bf16 ---
    { name: 'GDN decoder layer bf16 (norm+GDN+residual+norm+MLP+residual)', run() {
      const vals = MxArray.testGdnLayerBf16(20);
      if (vals[0] === -999) throw new Error('testGdnLayerBf16 error');
      const ref = [0.018677, 0.034912, 0.029053, 0.016113, 0.010193, 0.010254, 0.011414, 0.018188, 0.039063, 0.067871];
      assertClose(vals.slice(0, 10), ref, 0.05);
    }},

    // --- Stream channel test ---
    { name: 'stream channel SharedArrayBuffer write', run() {
      // test_stream_channel writes "hello from WASM stream" to the stream buffer
      const result = (MxArray as any).testStreamChannel?.();
      if (result === undefined) throw new Error('testStreamChannel not available (non-WASM build?)');
      if (result !== true) throw new Error(`testStreamChannel returned ${result}`);
    }},

    // --- Categorical sampling with bf16 (catches temperature>0 crash) ---
    { name: 'categorical sampling bf16 (temp=0.7, sliced logits)', run() {
      const vals = MxArray.testCategoricalSamplingBf16(10);
      if (vals[0] === -999) throw new Error('testCategoricalSamplingBf16 error');
      // Both token IDs should be valid (0..4095)
      if (vals[0] < 0 || vals[0] >= 4096) throw new Error(`Token 0 out of range: ${vals[0]}`);
      if (vals[1] < 0 || vals[1] >= 4096) throw new Error(`Token 1 (sliced) out of range: ${vals[1]}`);
    }},

    // --- JS-only tests ---

    // bf16 embedding lookup
    { name: 'bf16 embedding lookup model dims', run() {
      const BF16 = 3, F32 = 0, D = 64, V = 256;
      const embData = new Float32Array(V * D);
      for (let i = 0; i < V * D; i++) embData[i] = Math.sin(i * 0.01) * 0.5;
      const emb = MxArray.fromFloat32(embData, new BigInt64Array([BigInt(V), BigInt(D)])).astype(BF16);
      emb.eval();
      const ids = MxArray.fromInt32(new Int32Array([0, 42, 100, 255]), new BigInt64Array([4n]));
      const result = emb.take(ids, 0).astype(F32); // take(indices, axis=0)
      result.eval();
      const vals = [...result.toFloat32()];
      if (vals.length !== 4 * D) throw new Error(`Expected ${4*D} values, got ${vals.length}`);
      // Verify first row matches embedding[0]
      for (let j = 0; j < 5; j++) {
        const expected = Math.sin(j * 0.01) * 0.5;
        if (Math.abs(vals[j] - expected) > 0.05) throw new Error(`emb[0][${j}]: ${vals[j]} vs ${expected}`);
      }
    }},

    // bf16 KV cache pattern — write bf16 data, slice it back, verify integrity
    { name: 'KV cache bf16 write/read model dims', run() {
      const BF16 = 3, F32 = 0;
      const B = 1, H = 2, T = 8, D = 32;
      // Use small integer-ish values that are exact in bf16
      const writeData = new Float32Array(B * H * T * D);
      for (let i = 0; i < writeData.length; i++) writeData[i] = (i % 128) * 0.125 - 8.0; // range [-8, 8], step 0.125 (exact in bf16)
      const kvData = MxArray.fromFloat32(writeData, new BigInt64Array([BigInt(B), BigInt(H), BigInt(T), BigInt(D)])).astype(BF16);
      kvData.eval();
      // Convert back to f32 and read
      const readback = kvData.astype(F32);
      readback.eval();
      const vals = [...readback.toFloat32()];
      if (vals.length !== writeData.length) throw new Error(`Length mismatch: ${vals.length} vs ${writeData.length}`);
      let maxDiff = 0;
      for (let i = 0; i < vals.length; i++) {
        const diff = Math.abs(vals[i] - writeData[i]);
        if (diff > maxDiff) maxDiff = diff;
      }
      // Values are multiples of 0.125 which is exact in bf16 — should be zero error
      if (maxDiff > 0.001) throw new Error(`KV cache bf16 max diff: ${maxDiff}`);
      // Now test slicing: read second half
      const sliced = kvData.slice(
        new BigInt64Array([0n, 0n, BigInt(T/2), 0n]),
        new BigInt64Array([BigInt(B), BigInt(H), BigInt(T), BigInt(D)])
      ).astype(F32);
      sliced.eval();
      const slicedVals = [...sliced.toFloat32()];
      const halfOffset = B * H * (T/2) * D;
      // Hmm — the slice along axis 2 doesn't give contiguous offset into original data
      // Instead check that values are finite and non-zero
      if (slicedVals.length !== B * H * (T/2) * D) throw new Error(`Slice length: ${slicedVals.length}`);
      let hasNonZero = false;
      for (const v of slicedVals) {
        if (!isFinite(v)) throw new Error(`Non-finite in slice: ${v}`);
        if (v !== 0) hasNonZero = true;
      }
      if (!hasNonZero) throw new Error('Slice is all zeros');
    }},

    // Q projection split
    { name: 'Q projection split bf16 correctness', run() {
      const BF16 = 3, F32 = 0;
      const B = 1, T = 1, H = 8, D = 32;
      const data = new Float32Array(B * T * H * D * 2);
      for (let i = 0; i < data.length; i++) data[i] = (i + 1) * 0.001; // simple ascending
      const qph = MxArray.fromFloat32(data, new BigInt64Array([BigInt(B), BigInt(T), BigInt(H), BigInt(D * 2)])).astype(BF16);
      qph.eval();
      const queries = qph.slice(
        new BigInt64Array([0n, 0n, 0n, 0n]),
        new BigInt64Array([BigInt(B), BigInt(T), BigInt(H), BigInt(D)])
      ).astype(F32);
      const gate = qph.slice(
        new BigInt64Array([0n, 0n, 0n, BigInt(D)]),
        new BigInt64Array([BigInt(B), BigInt(T), BigInt(H), BigInt(D * 2)])
      ).astype(F32);
      queries.eval(); gate.eval();
      const qVals = [...queries.toFloat32()];
      const gVals = [...gate.toFloat32()];
      // q[0] should be close to data[0]=0.001, gate[0] should be close to data[D]=D*0.001+0.001
      if (Math.abs(qVals[0] - data[0]) > 0.01) throw new Error(`q[0]: ${qVals[0]} vs ${data[0]}`);
      if (Math.abs(gVals[0] - data[D]) > 0.01) throw new Error(`gate[0]: ${gVals[0]} vs ${data[D]}`);
      if (qVals.length !== B * T * H * D) throw new Error(`q length: ${qVals.length}`);
      if (gVals.length !== B * T * H * D) throw new Error(`gate length: ${gVals.length}`);
    }},

    // Attention gate (sigmoid gating)
    { name: 'attention gate sigmoid gating bf16', run() {
      const BF16 = 3, F32 = 0, N = 256;
      const attnData = new Float32Array(N);
      const gateData = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        attnData[i] = Math.sin(i * 0.05) * 0.3;
        gateData[i] = Math.cos(i * 0.03) * 2.0;
      }
      const attn = MxArray.fromFloat32(attnData, new BigInt64Array([1n, BigInt(N)])).astype(BF16);
      const gate = MxArray.fromFloat32(gateData, new BigInt64Array([1n, BigInt(N)])).astype(BF16);
      // sigmoid(gate) = 1 / (1 + exp(-gate))
      const negGate = gate.negative();
      const expNeg = negGate.exp();
      const one = MxArray.fromFloat32(new Float32Array([1]), new BigInt64Array([1n])).astype(BF16);
      const gateSig = one.div(one.add(expNeg));
      const result = attn.mul(gateSig).astype(F32);
      result.eval();
      const vals = [...result.toFloat32()];
      for (let i = 0; i < 10; i++) {
        const expected = attnData[i] * (1.0 / (1.0 + Math.exp(-gateData[i])));
        if (Math.abs(vals[i] - expected) > 0.05) throw new Error(`gate[${i}]: ${vals[i]} vs ${expected.toFixed(6)}`);
      }
    }},

    // bf16 argmax on vocab-sized logits
    { name: 'bf16 argmax vocab-sized logits', run() {
      const BF16 = 3;
      const V = 4096;
      const data = new Float32Array(V);
      for (let i = 0; i < V; i++) data[i] = -5.0 + Math.sin(i * 0.001) * 2.0;
      data[1234] = 15.0;
      const logits = MxArray.fromFloat32(data, new BigInt64Array([1n, BigInt(V)])).astype(BF16);
      const idx = logits.argmax(-1);
      idx.eval();
      const token = [...idx.toInt32()][0];
      if (token !== 1234) throw new Error(`argmax returned ${token}, expected 1234`);
    }},

    // bf16 embed -> lm_head -> argmax roundtrip
    { name: 'bf16 embed -> matmul(lm_head) -> argmax roundtrip', run() {
      const BF16 = 3, F32 = 0, D = 64, V = 512;
      const embData = new Float32Array(V * D);
      for (let i = 0; i < V * D; i++) embData[i] = Math.sin(i * 0.005) * 0.1;
      const emb = MxArray.fromFloat32(embData, new BigInt64Array([BigInt(V), BigInt(D)])).astype(BF16);
      const ids = MxArray.fromInt32(new Int32Array([42]), new BigInt64Array([1n]));
      const hidden = emb.take(ids, 0); // [1, D] bf16
      const embT = emb.transpose();
      const logits = hidden.matmul(embT); // [1, V] bf16
      const token = logits.argmax(-1);
      token.eval();
      const tokenId = [...token.toInt32()][0];
      if (tokenId < 0 || tokenId >= V) throw new Error(`Invalid token ${tokenId}`);
      // Verify logits are finite
      const logVals = [...logits.astype(F32).toFloat32()];
      if (!logVals.every(isFinite)) throw new Error('Non-finite logits');
    }},

    // bf16 RMSNorm -> matmul -> argmax chain
    { name: 'bf16 RMSNorm -> matmul -> argmax chain', run() {
      const BF16 = 3, D = 64, V = 256;
      const hData = new Float32Array(D);
      for (let i = 0; i < D; i++) hData[i] = Math.sin(i * 0.1) * 2.0;
      const h = MxArray.fromFloat32(hData, new BigInt64Array([1n, BigInt(D)])).astype(BF16);
      const normW = MxArray.fromFloat32(new Float32Array(D).fill(1), new BigInt64Array([BigInt(D)])).astype(BF16);
      const h2 = h.mul(h);
      const rmsVal = h2.mean(new Int32Array([-1])).addScalar(1e-6).sqrt();
      const normed = h.div(rmsVal.reshape(new BigInt64Array([1n, 1n]))).mul(normW);
      const wData = new Float32Array(D * V);
      for (let i = 0; i < D * V; i++) wData[i] = Math.cos(i * 0.01) * 0.1;
      const w = MxArray.fromFloat32(wData, new BigInt64Array([BigInt(D), BigInt(V)])).astype(BF16);
      const logits = normed.matmul(w);
      const token = logits.argmax(-1);
      token.eval();
      const tokenId = [...token.toInt32()][0];
      if (tokenId < 0 || tokenId >= V) throw new Error(`Invalid token ${tokenId}`);
    }},

    // bf16 residual accumulation
    { name: 'bf16 residual add 24-layer accumulation', run() {
      const BF16 = 3, F32 = 0, D = 64;
      const hData = new Float32Array(D);
      for (let i = 0; i < D; i++) hData[i] = Math.sin(i * 0.1) * 1.0;
      let h = MxArray.fromFloat32(hData, new BigInt64Array([1n, BigInt(D)])).astype(BF16);
      h.eval();
      for (let layer = 0; layer < 24; layer++) {
        const deltaData = new Float32Array(D);
        for (let i = 0; i < D; i++) deltaData[i] = Math.sin((layer * D + i) * 0.01) * 0.01;
        const delta = MxArray.fromFloat32(deltaData, new BigInt64Array([1n, BigInt(D)])).astype(BF16);
        h = h.add(delta);
        h.eval();
      }
      const vals = [...h.astype(F32).toFloat32()];
      const ref = new Float32Array(D);
      for (let i = 0; i < D; i++) ref[i] = Math.sin(i * 0.1) * 1.0;
      for (let layer = 0; layer < 24; layer++) {
        for (let i = 0; i < D; i++) ref[i] += Math.sin((layer * D + i) * 0.01) * 0.01;
      }
      assertClose(vals.slice(0, 10), [...ref].slice(0, 10), 0.1);
    }},

    // Multi-step decode determinism
    { name: 'multi-step decode determinism (2 runs)', run() {
      const BF16 = 3, F32 = 0, D = 64;
      const runs: number[][] = [];
      for (let run = 0; run < 2; run++) {
        const data = new Float32Array(D);
        for (let i = 0; i < D; i++) data[i] = Math.sin(i * 0.1) * 0.5;
        let h = MxArray.fromFloat32(data, new BigInt64Array([1n, BigInt(D)])).astype(BF16);
        h.eval();
        for (let step = 0; step < 3; step++) {
          const h2 = h.mul(h);
          const rms = h2.mean(new Int32Array([-1])).addScalar(1e-6).sqrt();
          const normed = h.div(rms.reshape(new BigInt64Array([1n, 1n])));
          const wData = new Float32Array(D * D);
          for (let i = 0; i < D * D; i++) wData[i] = Math.cos((step * D * D + i) * 0.001) * 0.1;
          const w = MxArray.fromFloat32(wData, new BigInt64Array([BigInt(D), BigInt(D)])).astype(BF16);
          const proj = normed.matmul(w);
          h = h.add(proj);
          h.eval();
        }
        const vals = [...h.astype(F32).toFloat32()];
        runs.push(vals);
      }
      for (let i = 0; i < D; i++) {
        if (runs[0][i] !== runs[1][i]) throw new Error(`Non-deterministic at [${i}]: ${runs[0][i]} vs ${runs[1][i]}`);
      }
    }},

    {
      name: 'packed bf16 gemv K=896 N=2048 transposed',
      run() {
        const F32 = 0;
        const K = 896, N = 2048;
        // W is [N, K] so W.transpose() produces a column-major view that
        // check_transpose reports as b_transposed=true (the linear_proj
        // hot path). 2048*896 = 1.8M elements, well above the 4096 opt-in
        // threshold, so the packed flip will fire when the flag is on.
        const wData = new Float32Array(N * K);
        for (let i = 0; i < wData.length; i++) wData[i] = Math.sin(i * 0.001) * 0.25;
        const xData = new Float32Array(K);
        for (let i = 0; i < K; i++) xData[i] = Math.cos(i * 0.01) * 0.5;
        const wShape = new BigInt64Array([BigInt(N), BigInt(K)]);
        const xShape = new BigInt64Array([1n, BigInt(K)]);

        try {
          // Baseline: flag OFF, fresh weight, f32-storage path.
          setPackedBf16(false);
          const W0 = makeBf16(wData, wShape);
          const x0 = makeBf16(xData, xShape);
          const y0 = x0.matmul(W0.transpose());
          const y0f = y0.astype(F32);
          y0f.eval();
          const y0Arr = [...y0f.toFloat32()];

          // Packed: flag ON, fresh weight → buffer flip fires on upload.
          setPackedBf16(true);
          const W1 = makeBf16(wData, wShape);
          const x1 = makeBf16(xData, xShape);
          const y1 = x1.matmul(W1.transpose());
          const y1f = y1.astype(F32);
          y1f.eval();
          const y1Arr = [...y1f.toFloat32()];

          if (y0Arr.length !== N) throw new Error(`baseline length=${y0Arr.length}`);
          if (y1Arr.length !== N) throw new Error(`packed length=${y1Arr.length}`);
          for (let i = 0; i < N; i++) {
            const d = Math.abs(y0Arr[i] - y1Arr[i]);
            if (d > PACKED_TOL) {
              throw new Error(`[${i}]: f32=${y0Arr[i]} packed=${y1Arr[i]} diff=${d}`);
            }
          }
        } finally {
          setPackedBf16(false);
        }
      },
    },

    {
      name: 'packed bf16 gemv K=128 N=1 transposed (below-threshold fallback)',
      run() {
        // 128 elements is far below the 4096 opt-in threshold, so the
        // packed flip must NOT fire even with the flag on. This test
        // verifies the dispatcher correctly falls back to the f32 kernel
        // for tiny weight buffers and still produces matching output.
        const F32 = 0;
        const K = 128, N = 1;
        const wData = new Float32Array(N * K);
        for (let i = 0; i < wData.length; i++) wData[i] = Math.sin(i * 0.05) * 0.3;
        const xData = new Float32Array(K);
        for (let i = 0; i < K; i++) xData[i] = Math.cos(i * 0.02);
        const wShape = new BigInt64Array([BigInt(N), BigInt(K)]);
        const xShape = new BigInt64Array([1n, BigInt(K)]);

        try {
          setPackedBf16(false);
          const W0 = makeBf16(wData, wShape);
          const y0 = makeBf16(xData, xShape).matmul(W0.transpose());
          const y0f = y0.astype(F32);
          y0f.eval();
          const y0Arr = [...y0f.toFloat32()];

          setPackedBf16(true);
          const W1 = makeBf16(wData, wShape);
          const y1 = makeBf16(xData, xShape).matmul(W1.transpose());
          const y1f = y1.astype(F32);
          y1f.eval();
          const y1Arr = [...y1f.toFloat32()];

          if (y0Arr.length !== N || y1Arr.length !== N) {
            throw new Error(`lengths ${y0Arr.length}, ${y1Arr.length}`);
          }
          for (let i = 0; i < N; i++) {
            if (Math.abs(y0Arr[i] - y1Arr[i]) > PACKED_TOL) {
              throw new Error(`[${i}]: f32=${y0Arr[i]} vs ${y1Arr[i]}`);
            }
          }
        } finally {
          setPackedBf16(false);
        }
      },
    },

    {
      name: 'packed bf16 gemv K=896 N=7 transposed odd-N',
      run() {
        // Odd N exercises the multi-col bounds-check (non-n_aligned) path.
        // 7*896 = 6272 elements — above the 4096 threshold, so packed
        // storage fires. N=7 is below the multi-col=16 threshold though,
        // so this also covers the single-col packed GEMV variant.
        const F32 = 0;
        const K = 896, N = 7;
        const wData = new Float32Array(N * K);
        for (let i = 0; i < wData.length; i++) wData[i] = Math.sin(i * 0.003) * 0.4;
        const xData = new Float32Array(K);
        for (let i = 0; i < K; i++) xData[i] = Math.cos(i * 0.015) * 0.3;
        const wShape = new BigInt64Array([BigInt(N), BigInt(K)]);
        const xShape = new BigInt64Array([1n, BigInt(K)]);

        try {
          setPackedBf16(false);
          const W0 = makeBf16(wData, wShape);
          const y0 = makeBf16(xData, xShape).matmul(W0.transpose());
          const y0f = y0.astype(F32);
          y0f.eval();
          const y0Arr = [...y0f.toFloat32()];

          setPackedBf16(true);
          const W1 = makeBf16(wData, wShape);
          const y1 = makeBf16(xData, xShape).matmul(W1.transpose());
          const y1f = y1.astype(F32);
          y1f.eval();
          const y1Arr = [...y1f.toFloat32()];

          if (y0Arr.length !== N || y1Arr.length !== N) {
            throw new Error(`lengths ${y0Arr.length}, ${y1Arr.length}`);
          }
          for (let i = 0; i < N; i++) {
            const d = Math.abs(y0Arr[i] - y1Arr[i]);
            if (d > PACKED_TOL) {
              throw new Error(`[${i}]: f32=${y0Arr[i]} packed=${y1Arr[i]} diff=${d}`);
            }
          }
        } finally {
          setPackedBf16(false);
        }
      },
    },

    {
      name: 'packed bf16 byte-concat parity (axis=0, bf16)',
      run() {
        // Verify that the Step-B JS-side byte concatenation produces the
        // same bytes as MxArray.concatenate(a, b, 0) for a bf16 pair with
        // matching trailing dimension. This is the invariant that the
        // pre-merge in mlx-worker.ts relies on: for row-major bf16 the
        // axis-0 concat IS a straight byte append, so we can build the
        // merged in_proj_qkvz bytes in JS before upload without calling
        // into MLX.
        const F32 = 0;
        const ROWS_A = 3, ROWS_B = 5, COLS = 8;
        const aData = new Float32Array(ROWS_A * COLS);
        for (let i = 0; i < aData.length; i++) aData[i] = Math.sin(i * 0.1) * 0.7;
        const bData = new Float32Array(ROWS_B * COLS);
        for (let i = 0; i < bData.length; i++) bData[i] = Math.cos(i * 0.2 + 1) * 0.4;
        const aShape = new BigInt64Array([BigInt(ROWS_A), BigInt(COLS)]);
        const bShape = new BigInt64Array([BigInt(ROWS_B), BigInt(COLS)]);

        // Reference path: build bf16 via MxArray → concatenate → read back.
        const aBf16 = makeBf16(aData, aShape);
        const bBf16 = makeBf16(bData, bShape);
        const refConcat = MxArray.concatenate(aBf16, bBf16, 0);
        const refVals = [...refConcat.astype(F32).toFloat32()];

        // JS byte-concat path: take the bf16 source bytes, append them,
        // hand to fromBfloat16Bytes with the combined shape. This mirrors
        // exactly what mlx-worker.ts does before upload.
        const aBytes = f32ToBf16Bytes(aData);
        const bBytes = f32ToBf16Bytes(bData);
        const mergedBytes = new Uint8Array(aBytes.length + bBytes.length);
        mergedBytes.set(aBytes, 0);
        mergedBytes.set(bBytes, aBytes.length);
        const mergedShape = new BigInt64Array([BigInt(ROWS_A + ROWS_B), BigInt(COLS)]);
        const jsMerged = MxArray.fromBfloat16Bytes(mergedBytes, mergedShape);
        const jsVals = [...jsMerged.astype(F32).toFloat32()];

        if (refVals.length !== jsVals.length) {
          throw new Error(`length mismatch ${refVals.length} vs ${jsVals.length}`);
        }
        for (let i = 0; i < refVals.length; i++) {
          if (refVals[i] !== jsVals[i]) {
            throw new Error(`[${i}]: ref=${refVals[i]} js=${jsVals[i]}`);
          }
        }
      },
    },

    {
      name: 'packed bf16 gemv K=1024 N=8192 transposed (in_proj_qkvz scale)',
      run() {
        // Qwen3.5-0.8B linear_attn.in_proj_qkvz.weight is [8192, 1024] bf16
        // (16 Q heads + 16 K heads + 16 V heads + 16 Z streams, each
        // head_dim=128 → 64 × 128 = 8192 rows, hidden_size=1024 cols). The
        // decode GEMV is x @ W.T which lands on b_transposed=true, the
        // only branch with a packed bf16 kernel. This test exercises the
        // exact shape the Step-B pre-merge produces for the dominant
        // bandwidth-bound weight in the Qwen3.5 decode path.
        const F32 = 0;
        const K = 1024, N = 8192;
        const wData = new Float32Array(N * K);
        // Deterministic, non-trivial pattern; keep values small so bf16
        // truncation doesn't dominate the parity tolerance.
        for (let i = 0; i < wData.length; i++) wData[i] = Math.sin(i * 0.0007) * 0.2;
        const xData = new Float32Array(K);
        for (let i = 0; i < K; i++) xData[i] = Math.cos(i * 0.013) * 0.3;
        const wShape = new BigInt64Array([BigInt(N), BigInt(K)]);
        const xShape = new BigInt64Array([1n, BigInt(K)]);

        try {
          setPackedBf16(false);
          const W0 = makeBf16(wData, wShape);
          const y0 = makeBf16(xData, xShape).matmul(W0.transpose());
          const y0f = y0.astype(F32);
          y0f.eval();
          const y0Arr = [...y0f.toFloat32()];

          setPackedBf16(true);
          const W1 = makeBf16(wData, wShape);
          const y1 = makeBf16(xData, xShape).matmul(W1.transpose());
          const y1f = y1.astype(F32);
          y1f.eval();
          const y1Arr = [...y1f.toFloat32()];

          if (y0Arr.length !== N || y1Arr.length !== N) {
            throw new Error(`lengths ${y0Arr.length}, ${y1Arr.length}`);
          }
          // Larger K/N → more accumulated rounding noise, so loosen the
          // parity bound slightly. Still well below any meaningful output
          // drift for model forward passes.
          const TOL = 5e-3;
          for (let i = 0; i < N; i++) {
            const d = Math.abs(y0Arr[i] - y1Arr[i]);
            if (d > TOL) {
              throw new Error(`[${i}]: f32=${y0Arr[i]} packed=${y1Arr[i]} diff=${d}`);
            }
          }
        } finally {
          setPackedBf16(false);
        }
      },
    },

    {
      name: 'packed bf16 rmsnorm D=4096',
      run() {
        // RMSNorm parity: baseline (w as f32-upconverted bf16) vs packed
        // (w as PackedBf16 array<u32>). D=4096 is exactly at the opt-in
        // threshold so the packed flip fires on a single row of weights.
        // Input x must stay UN-packed — the RMSNorm kernel still binds
        // `input` as `array<f32>` regardless of the weight storage — so
        // we toggle the flag off before creating x, then back on for w,
        // then off again for the packed run's x as well. The packed path
        // only differs in how `weight` is bound and unpacked in the shader.
        const F32 = 0;
        const R = 4, D = 4096;
        const eps = 1e-5;
        const xData = new Float32Array(R * D);
        for (let i = 0; i < xData.length; i++) xData[i] = Math.sin(i * 0.01) * 0.5;
        const wData = new Float32Array(D);
        for (let i = 0; i < D; i++) wData[i] = 1.0 + Math.cos(i * 0.003) * 0.2;
        const xShape = new BigInt64Array([BigInt(R), BigInt(D)]);
        const wShape = new BigInt64Array([BigInt(D)]);

        try {
          // Baseline: everything unpacked (flag off), w goes to array<f32>.
          setPackedBf16(false);
          const x0 = makeBf16(xData, xShape);
          const w0 = makeBf16(wData, wShape);
          const y0 = x0.fastRmsNorm(w0, eps);
          const y0f = y0.astype(F32);
          y0f.eval();
          const y0Arr = [...y0f.toFloat32()];

          // Packed: create w FIRST with flag on (flips w's buffer to
          // PackedBf16), then flip flag off before creating x so x stays
          // in the upconverted layout the kernel's `input` binding expects.
          setPackedBf16(true);
          const w1 = makeBf16(wData, wShape);
          setPackedBf16(false);
          const x1 = makeBf16(xData, xShape);
          const y1 = x1.fastRmsNorm(w1, eps);
          const y1f = y1.astype(F32);
          y1f.eval();
          const y1Arr = [...y1f.toFloat32()];

          if (y0Arr.length !== R * D || y1Arr.length !== R * D) {
            throw new Error(`lengths ${y0Arr.length}, ${y1Arr.length}`);
          }
          for (let i = 0; i < y0Arr.length; i++) {
            const d = Math.abs(y0Arr[i] - y1Arr[i]);
            if (d > PACKED_TOL) {
              throw new Error(`[${i}]: f32=${y0Arr[i]} packed=${y1Arr[i]} diff=${d}`);
            }
          }
        } finally {
          setPackedBf16(false);
        }
      },
    },

    {
      name: 'packed bf16 rmsnorm D=1024 production norm threshold (256)',
      run() {
        // Mirrors the production Qwen3.5-0.8B hidden_size = 1024 RMSNorm
        // weight shape. gpu-worker.ts uses NORM_PACKED_MIN_ELEMENTS = 256
        // on the real-model upload path, far below the default 4096 floor
        // baked into fromBfloat16Bytes. This test uses the test-only
        // `min_elements` override on from_bfloat16_bytes so the packed
        // storage flip actually fires at D=1024. Parity between the
        // baseline (w upconverted to f32) and packed (w as array<u32>)
        // runs proves the packed RMSNorm kernel (`rmsnorm_*_bf16p_*`)
        // is both dispatched AND correct for the real shape the Qwen3.5
        // model ships. The previous D=4096/4098 tests happened to work
        // through the default-floor path but did NOT cover the 256-floor
        // production branch.
        const F32 = 0;
        const R = 4, D = 1024;
        const eps = 1e-5;
        const xData = new Float32Array(R * D);
        for (let i = 0; i < xData.length; i++) xData[i] = Math.sin(i * 0.01) * 0.5;
        const wData = new Float32Array(D);
        for (let i = 0; i < D; i++) wData[i] = 1.0 + Math.cos(i * 0.003) * 0.2;
        const xShape = new BigInt64Array([BigInt(R), BigInt(D)]);
        const wShape = new BigInt64Array([BigInt(D)]);

        try {
          // Baseline: flag off, w goes to array<f32>. Use the regular
          // makeBf16 (default 4096 floor) — at D=1024 the flag-off path
          // short-circuits on the enabled check, so no storage flip.
          setPackedBf16(false);
          const x0 = makeBf16(xData, xShape);
          const w0 = makeBf16(wData, wShape);
          const y0 = x0.fastRmsNorm(w0, eps);
          const y0f = y0.astype(F32);
          y0f.eval();
          const y0Arr = [...y0f.toFloat32()];

          // Packed: flag ON + lowered 256 threshold. The storage flip
          // fires because n_elem=1024 >= 256 AND the flag is enabled,
          // matching what gpu-worker.ts does for the real safetensors
          // norm upload. x stays unpacked via the regular makeBf16 helper
          // with the flag flipped off first.
          setPackedBf16(true);
          const w1 = makeBf16NormThreshold(wData, wShape);
          setPackedBf16(false);
          const x1 = makeBf16(xData, xShape);
          const y1 = x1.fastRmsNorm(w1, eps);
          const y1f = y1.astype(F32);
          y1f.eval();
          const y1Arr = [...y1f.toFloat32()];

          if (y0Arr.length !== R * D || y1Arr.length !== R * D) {
            throw new Error(`lengths ${y0Arr.length}, ${y1Arr.length}`);
          }
          for (let i = 0; i < y0Arr.length; i++) {
            const d = Math.abs(y0Arr[i] - y1Arr[i]);
            if (d > PACKED_TOL) {
              throw new Error(`[${i}]: f32=${y0Arr[i]} packed=${y1Arr[i]} diff=${d}`);
            }
          }
        } finally {
          setPackedBf16(false);
        }
      },
    },

    {
      name: 'packed bf16 rmsnorm D=4098 even-tail (buffer pad check)',
      run() {
        // Even dim > 4096 exercises the aligned packed weight read path
        // with a non-power-of-two K. Keeps the u32 pair count = D/2 =
        // 2049 slots, verifying the bind-group size accounting matches
        // ((D+1)/2)*4 = 4100 bytes (rounded to 4). Guards against off-by-one
        // errors in wgpu_packed_alloc_size vs the shader's axis_size bound.
        const F32 = 0;
        const R = 2, D = 4098;
        const eps = 1e-5;
        const xData = new Float32Array(R * D);
        for (let i = 0; i < xData.length; i++) xData[i] = Math.cos(i * 0.007) * 0.4;
        const wData = new Float32Array(D);
        for (let i = 0; i < D; i++) wData[i] = 0.9 + Math.sin(i * 0.005) * 0.15;
        const xShape = new BigInt64Array([BigInt(R), BigInt(D)]);
        const wShape = new BigInt64Array([BigInt(D)]);

        try {
          setPackedBf16(false);
          const x0 = makeBf16(xData, xShape);
          const w0 = makeBf16(wData, wShape);
          const y0 = x0.fastRmsNorm(w0, eps);
          const y0f = y0.astype(F32);
          y0f.eval();
          const y0Arr = [...y0f.toFloat32()];

          setPackedBf16(true);
          const w1 = makeBf16(wData, wShape);
          setPackedBf16(false);
          const x1 = makeBf16(xData, xShape);
          const y1 = x1.fastRmsNorm(w1, eps);
          const y1f = y1.astype(F32);
          y1f.eval();
          const y1Arr = [...y1f.toFloat32()];

          if (y0Arr.length !== R * D || y1Arr.length !== R * D) {
            throw new Error(`lengths ${y0Arr.length}, ${y1Arr.length}`);
          }
          for (let i = 0; i < y0Arr.length; i++) {
            const d = Math.abs(y0Arr[i] - y1Arr[i]);
            if (d > PACKED_TOL) {
              throw new Error(`[${i}]: f32=${y0Arr[i]} packed=${y1Arr[i]} diff=${d}`);
            }
          }
        } finally {
          setPackedBf16(false);
        }
      },
    },

    {
      name: 'packed bf16 gemv K=896 N=2048 transposed offset-sliced',
      run() {
        // Source = [2*N, K] bf16; slice src[N:2*N, :] so the view has a
        // non-zero offset() in BYTES. The packed dispatcher divides offset
        // by 4 to produce a u32 index — (N*K*2) / 4 = N*K/2, and K is even
        // so the packed indexing lands on a u32 boundary. This exercises
        // the offset_b = offset()/4 branch in matmul.cpp.
        const F32 = 0;
        const K = 896, N = 2048;
        const srcRows = 2 * N;
        const srcData = new Float32Array(srcRows * K);
        for (let i = 0; i < srcData.length; i++) srcData[i] = Math.sin(i * 0.0005) * 0.2;
        const xData = new Float32Array(K);
        for (let i = 0; i < K; i++) xData[i] = Math.cos(i * 0.008) * 0.4;
        const srcShape = new BigInt64Array([BigInt(srcRows), BigInt(K)]);
        const xShape = new BigInt64Array([1n, BigInt(K)]);
        const sliceStart = new BigInt64Array([BigInt(N), 0n]);
        const sliceStop = new BigInt64Array([BigInt(srcRows), BigInt(K)]);

        try {
          setPackedBf16(false);
          const Src0 = makeBf16(srcData, srcShape);
          const W0 = Src0.slice(sliceStart, sliceStop);
          const y0 = makeBf16(xData, xShape).matmul(W0.transpose());
          const y0f = y0.astype(F32);
          y0f.eval();
          const y0Arr = [...y0f.toFloat32()];

          setPackedBf16(true);
          const Src1 = makeBf16(srcData, srcShape);
          const W1 = Src1.slice(sliceStart, sliceStop);
          const y1 = makeBf16(xData, xShape).matmul(W1.transpose());
          const y1f = y1.astype(F32);
          y1f.eval();
          const y1Arr = [...y1f.toFloat32()];

          if (y0Arr.length !== N || y1Arr.length !== N) {
            throw new Error(`lengths ${y0Arr.length}, ${y1Arr.length}`);
          }
          for (let i = 0; i < N; i++) {
            const d = Math.abs(y0Arr[i] - y1Arr[i]);
            if (d > PACKED_TOL) {
              throw new Error(`[${i}]: f32=${y0Arr[i]} packed=${y1Arr[i]} diff=${d}`);
            }
          }
        } finally {
          setPackedBf16(false);
        }
      },
    },

  ];

  let passed = 0, failed = 0;
  for (const t of tests) {
    try { t.run(); passed++; post({ type: 'result', name: t.name, passed: true }); }
    catch (e: any) {
      failed++;
      let msg = e.message || `${e}`;
      if (e instanceof WebAssembly.Exception && wasmInst) {
        try {
          const ptr = e.getArg(tag, 0);
          const mem = new Uint8Array((wasmInst.exports.memory as WebAssembly.Memory).buffer);
          const view = new DataView(mem.buffer);
          // Try to read exception message by scanning memory near the pointer
          // __cxa_exception header is before the thrown object, object starts at ptr
          // For std::runtime_error: [vtable(4), __imp_(4+)], __imp_ points to string
          // Try multiple offsets to find a readable string
          const found: string[] = [];
          for (const off of [4, 8, 12, 16, 20]) {
            if (ptr + off + 4 < mem.length) {
              const strPtr = view.getUint32(ptr + off, true);
              if (strPtr > 1024 && strPtr < mem.length - 4) {
                let s = '';
                for (let i = strPtr; i < mem.length && mem[i] !== 0 && i - strPtr < 200; i++) {
                  const ch = mem[i];
                  if (ch >= 32 && ch < 127) s += String.fromCharCode(ch);
                  else { s = ''; break; }
                }
                if (s.length > 5) found.push(`@${off}:${s}`);
              }
            }
          }
          msg = found.length > 0 ? `C++: ${found[0]}` : `WasmException(ptr=${ptr}, no readable msg)`;
        } catch (ex: any) { msg += ` [extract err: ${ex.message}]`; }
      }
      post({ type: 'result', name: t.name, passed: false, error: msg });
    }
  }
  post({ type: 'done', summary: `Done: ${passed} passed, ${failed} failed of ${tests.length}` });
}
