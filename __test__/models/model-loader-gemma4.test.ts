import type { Gemma4Config } from '@mlx-node/core';
import { Gemma4Model as Gemma4ModelNative } from '@mlx-node/core';
import { Gemma4Model } from '@mlx-node/lm';
import { describe, expect, it } from 'vite-plus/test';

/**
 * Regression guard for the bug where `packages/lm/src/models/model-loader.ts`
 * imported `Gemma4Model` from `@mlx-node/core` (the raw native class with
 * callback-based streaming) instead of from `../stream.js` (the wrapper
 * with `AsyncGenerator<ChatStreamEvent>` overrides). That broke
 * `ChatSession` streaming for Gemma4 because the native methods resolve
 * to a `ChatStreamHandle` — not an `AsyncGenerator` — so
 * `for await (const event of model.chatStreamSessionStart(...))` failed
 * structurally at the `SessionCapableModel` boundary.
 *
 * These tests assert the prototype shape of the `Gemma4Model` re-exported
 * from `@mlx-node/lm`: the three `chatStream*` entry points must be async
 * generator functions (not plain async methods returning promises), and
 * the wrapper class must be a subclass of the native one. No model load
 * required — purely structural checks to keep the test lightweight and
 * GPU-independent.
 */
describe('Gemma4Model re-export from @mlx-node/lm', () => {
  it('is a subclass of the native @mlx-node/core Gemma4Model', () => {
    expect(Object.getPrototypeOf(Gemma4Model)).toBe(Gemma4ModelNative);
  });

  it('overrides chatStreamSessionStart with an async generator function', () => {
    const proto = Gemma4Model.prototype as unknown as Record<string, unknown>;
    const nativeProto = Gemma4ModelNative.prototype as unknown as Record<string, unknown>;
    expect(typeof proto.chatStreamSessionStart).toBe('function');
    // The wrapper override must NOT be the same reference as the native
    // callback-based method — that is the exact footgun the regression
    // test is guarding against.
    expect(proto.chatStreamSessionStart).not.toBe(nativeProto.chatStreamSessionStart);
    // Async generator functions have their own well-known constructor
    // name — plain async methods report `AsyncFunction`.
    const fn = proto.chatStreamSessionStart as (...args: unknown[]) => unknown;
    expect(fn.constructor.name).toBe('AsyncGeneratorFunction');
  });

  it('overrides chatStreamSessionContinue with an async generator function', () => {
    const proto = Gemma4Model.prototype as unknown as Record<string, unknown>;
    const nativeProto = Gemma4ModelNative.prototype as unknown as Record<string, unknown>;
    expect(typeof proto.chatStreamSessionContinue).toBe('function');
    expect(proto.chatStreamSessionContinue).not.toBe(nativeProto.chatStreamSessionContinue);
    const fn = proto.chatStreamSessionContinue as (...args: unknown[]) => unknown;
    expect(fn.constructor.name).toBe('AsyncGeneratorFunction');
  });

  it('overrides chatStreamSessionContinueTool with an async generator function', () => {
    const proto = Gemma4Model.prototype as unknown as Record<string, unknown>;
    const nativeProto = Gemma4ModelNative.prototype as unknown as Record<string, unknown>;
    expect(typeof proto.chatStreamSessionContinueTool).toBe('function');
    expect(proto.chatStreamSessionContinueTool).not.toBe(nativeProto.chatStreamSessionContinueTool);
    const fn = proto.chatStreamSessionContinueTool as (...args: unknown[]) => unknown;
    expect(fn.constructor.name).toBe('AsyncGeneratorFunction');
  });
});

/**
 * A minimal `Gemma4Config` that satisfies the NAPI-derived struct so
 * `new Gemma4ModelNative(cfg)` accepts it. Values are NOT meaningful —
 * the stub constructor never materializes weights or a tokenizer, so
 * only the shape matters. Kept inside the test file rather than a
 * shared helper because the fields are documented fully by
 * `packages/core/index.d.cts` and any drift is caught by the existing
 * typecheck pass on the test suite.
 */
function stubConfig(overrides: Partial<Gemma4Config> = {}): Gemma4Config {
  return {
    vocabSize: 256,
    hiddenSize: 8,
    numHiddenLayers: 1,
    numAttentionHeads: 1,
    numKeyValueHeads: 1,
    headDim: 8,
    intermediateSize: 16,
    rmsNormEps: 1e-6,
    tieWordEmbeddings: false,
    maxPositionEmbeddings: 128,
    slidingWindow: 64,
    layerTypes: ['full_attention'],
    ropeTheta: 1_000_000,
    ropeLocalBaseFreq: 10_000,
    partialRotaryFactor: 0.25,
    attentionKEqV: false,
    perLayerInputEmbeds: false,
    padTokenId: 0,
    eosTokenIds: [1],
    bosTokenId: 2,
    attentionBias: false,
    useDoubleWideMlp: false,
    enableMoeBlock: false,
    ...overrides,
  };
}

/**
 * Round-5 Finding B regression coverage.
 *
 * `new Gemma4Model(config)` was a runnable entry point before the
 * cache-limit coordinator work landed. It is now a deliberate
 * config-only stub (matches `VLModel::new(config)` /
 * `QianfanOCRModel::new(config)`) because a no-op `new(config)` would
 * have registered an empty coordinator delta and broken the
 * deterministic-weight-bytes baseline.
 *
 * The Rust side uses an exact error message (see
 * `crates/mlx-core/src/models/gemma4/model.rs`:
 * `"Model not initialized. Call Gemma4Model.load() first."`) so this
 * test asserts on a fragment (`"not initialized"`) to keep the probe
 * robust against punctuation tweaks without letting the wrong error
 * pass.
 *
 * We exercise the NATIVE class directly, not the `@mlx-node/lm`
 * wrapper, because the wrapper adds async-generator streaming that
 * masks the native rejection behind the generator protocol — the
 * native surface is the one the handler dispatches on.
 */
describe('Gemma4Model(config) stub (round-5 Finding B)', () => {
  it('returns an object with isInitialized=false and a numeric modelId', () => {
    const stub = new Gemma4ModelNative(stubConfig());
    expect(stub.isInitialized).toBe(false);
    expect(typeof stub.modelId()).toBe('number');
  });

  it('rejects chatSessionStart with a "not initialized" error', async () => {
    const stub = new Gemma4ModelNative(stubConfig());
    await expect(stub.chatSessionStart([{ role: 'user', content: 'hi' }])).rejects.toThrow(/not initialized/i);
  });

  it('rejects chatSessionContinue with a "not initialized" error', async () => {
    const stub = new Gemma4ModelNative(stubConfig());
    await expect(stub.chatSessionContinue('hi', null, null)).rejects.toThrow(/not initialized/i);
  });

  it('rejects chatSessionContinueTool with a "not initialized" error', async () => {
    const stub = new Gemma4ModelNative(stubConfig());
    await expect(stub.chatSessionContinueTool('tool_123', '{"ok":true}')).rejects.toThrow(/not initialized/i);
  });

  it('rejects chatStreamSessionStart with a "not initialized" error', async () => {
    const stub = new Gemma4ModelNative(stubConfig());
    // Streaming methods still resolve on the NAPI boundary — they
    // hand back a `ChatStreamHandle` — but the precondition check
    // runs BEFORE the callback is ever invoked, so the promise must
    // reject synchronously with the not-initialized message.
    await expect(stub.chatStreamSessionStart([{ role: 'user', content: 'hi' }], null, () => {})).rejects.toThrow(
      /not initialized/i,
    );
  });

  it('rejects chatStreamSessionContinue with a "not initialized" error', async () => {
    const stub = new Gemma4ModelNative(stubConfig());
    await expect(stub.chatStreamSessionContinue('hi', null, null, () => {})).rejects.toThrow(/not initialized/i);
  });

  it('rejects chatStreamSessionContinueTool with a "not initialized" error', async () => {
    const stub = new Gemma4ModelNative(stubConfig());
    await expect(stub.chatStreamSessionContinueTool('tool_123', '{"ok":true}', null, () => {})).rejects.toThrow(
      /not initialized/i,
    );
  });

  it('resetCaches is a silent no-op on the stub', () => {
    const stub = new Gemma4ModelNative(stubConfig());
    // Matches the documented contract on the Rust impl: uninitialized
    // stub returns Ok(()) so `ChatSession.reset()` is idempotent
    // across stub + loaded instances.
    expect(() => stub.resetCaches()).not.toThrow();
  });
});

/**
 * Round-5 Finding B also asks for a positive-path assertion that
 * `Gemma4Model.load(validPath)` returns a runnable model. Real weights
 * are not available in CI — they are multi-gigabyte HuggingFace
 * downloads — so we assert the SHAPE of the class instead: `load` is a
 * static async function, the stub produced by `new(config)` is an
 * instance of the same class a `load()` call would return, and the
 * runnable surface (`chatSessionStart` et al) is present on the
 * prototype so a loaded instance would dispatch correctly. A
 * full-weight end-to-end load is covered by the integration runs in
 * `examples/` when real weights are available locally.
 */
describe('Gemma4Model.load() shape (round-5 Finding B)', () => {
  it('exposes load as a static promise-returning function on the class', () => {
    expect(typeof Gemma4ModelNative.load).toBe('function');
    // NAPI-RS emits a plain function whose body dispatches to a
    // native tokio task and returns a thenable — it is NOT a native
    // JS `async function` (constructor name would be
    // `AsyncFunction`), so we verify the return shape instead. A
    // refactor that accidentally made `load()` sync would return an
    // instance of `Gemma4ModelNative`, not a thenable, and this
    // probe would catch it. We pass an intentionally-invalid path so
    // no real disk I/O happens — the returned promise rejects, and
    // we only care about the `then` shape on the returned value.
    const ret = Gemma4ModelNative.load('/dev/null/__does_not_exist__');
    expect(ret).toBeDefined();
    expect(typeof (ret as { then?: unknown }).then).toBe('function');
    // Swallow the eventual rejection so vitest does not flag an
    // unhandled rejection on shutdown.
    ret.then(
      () => undefined,
      () => undefined,
    );
  });

  it('exposes the full session surface on the prototype', () => {
    const proto = Gemma4ModelNative.prototype as unknown as Record<string, unknown>;
    expect(typeof proto.chatSessionStart).toBe('function');
    expect(typeof proto.chatSessionContinue).toBe('function');
    expect(typeof proto.chatSessionContinueTool).toBe('function');
    expect(typeof proto.chatStreamSessionStart).toBe('function');
    expect(typeof proto.chatStreamSessionContinue).toBe('function');
    expect(typeof proto.chatStreamSessionContinueTool).toBe('function');
    expect(typeof proto.resetCaches).toBe('function');
  });
});
