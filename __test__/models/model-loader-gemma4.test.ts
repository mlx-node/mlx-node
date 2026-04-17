import { Gemma4Model as Gemma4ModelNative } from '@mlx-node/core';
import { Gemma4Model } from '@mlx-node/lm';
import { describe, it, expect } from 'vite-plus/test';

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
