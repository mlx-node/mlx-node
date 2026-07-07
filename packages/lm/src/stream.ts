import { join } from 'node:path';

import {
  Gemma4Model as Gemma4ModelNative,
  Lfm2Model as Lfm2ModelNative,
  Qwen3Tokenizer,
  Qwen3Model as Qwen3ModelNative,
  Qwen35Model as Qwen35ModelNative,
  Qwen35MoeModel as Qwen35MoeModelNative,
} from '@mlx-node/core';
import type {
  ChatConfig,
  ChatMessage,
  ChatStreamChunk,
  ChatStreamHandle,
  PerformanceMetrics,
  ToolDefinition,
  ToolCallResult,
} from '@mlx-node/core';

import type { SessionCapableModel } from './chat-session.js';

export interface ChatStreamDelta {
  text: string;
  done: false;
  isReasoning?: boolean;
}

export interface ChatStreamFinal {
  text: string;
  done: true;
  finishReason: string;
  toolCalls: ToolCallResult[];
  thinking: string | null;
  numTokens: number;
  promptTokens: number;
  reasoningTokens: number;
  rawText: string;
  /**
   * Number of prompt tokens served from the reused KV-cache prefix on
   * this turn. Mirrors the `cachedTokens` field on the non-streaming
   * `ChatResult` so session-aware streaming consumers can observe
   * prefix-cache reuse without round-tripping to the non-streaming
   * path.
   *
   * The native `ChatStreamChunk` surfaces `cachedTokens` on the
   * terminal (`done == true`) chunk for every streaming entry point
   * (Qwen3, Qwen3.5 Dense / MoE, LFM2, Gemma4, QianfanOCR) — start-path
   * chunks carry the matched prefix length from
   * `verify_cache_prefix_direct`, delta-path chunks carry the reused
   * prior-history length. Non-terminal deltas carry `None` /
   * `undefined` (only the terminal chunk is authoritative).
   *
   * This field remains OPTIONAL because the bridge-level mock tests
   * (and any future in-process driver that constructs its own
   * `ChatStreamChunk`) may legitimately omit it. Consumers SHOULD
   * treat `undefined` distinctly from `0` (e.g. skip emitting
   * `X-Cached-Tokens` rather than reporting `0`); a numeric value is
   * always authoritative.
   */
  cachedTokens?: number;
  performance?: PerformanceMetrics;
}

export type ChatStreamEvent = ChatStreamDelta | ChatStreamFinal;

const modelPathsForTokenizers = new WeakMap<object, string>();
const tokenizerPromises = new WeakMap<object, Promise<Qwen3Tokenizer>>();

function getNativeIsReasoning(chunk: ChatStreamChunk): boolean | undefined {
  return typeof chunk.isReasoning === 'boolean' ? chunk.isReasoning : undefined;
}

function rememberModelPath(model: object, modelPath: string): void {
  modelPathsForTokenizers.set(model, modelPath);
}

async function applyChatTemplateFromModelPath(
  model: object,
  messages: ChatMessage[],
  addGenerationPrompt?: boolean | null,
  tools?: ToolDefinition[] | null,
  enableThinking?: boolean | null,
): Promise<Uint32Array> {
  const modelPath = modelPathsForTokenizers.get(model);
  if (modelPath == null) {
    throw new Error('applyChatTemplate unavailable: model path was not recorded when this model was loaded');
  }
  let tokenizerPromise = tokenizerPromises.get(model);
  if (tokenizerPromise == null) {
    tokenizerPromise = Qwen3Tokenizer.fromPretrained(join(modelPath, 'tokenizer.json'));
    tokenizerPromises.set(model, tokenizerPromise);
  }
  const tokenizer = await tokenizerPromise;
  return tokenizer.applyChatTemplate(messages, addGenerationPrompt, tools, enableThinking);
}

/**
 * Shared AsyncGenerator adapter for callback-based native streaming methods.
 *
 * Takes a `startCall` closure that, given the JS-side callback, dispatches
 * the underlying native stream (whatever method signature that is — the
 * closure captures `messages` / `config` / `userMessage` etc) and resolves
 * with a `ChatStreamHandle`. The generator pumps the resulting chunk queue,
 * transforms each chunk into a `ChatStreamEvent`, and calls `handle.cancel()`
 * in a `finally` block so early termination (user `break`, exception) still
 * cleans up native state.
 *
 * ## Signal-driven fast-abort
 *
 * When an optional `AbortSignal` is supplied and fires, the adapter:
 *   1. Calls `handle.cancel()` immediately so the native decode stops
 *      on the next safepoint rather than running to completion.
 *   2. Wakes the pending `waitForItem()` await by pushing a synthetic
 *      "aborted" marker into the queue and calling `notify()`. Without
 *      this wake-up the generator would stay parked on the `await`
 *      until the next native chunk arrived, which on a fast-abort
 *      path (client disconnect before first token) never happens.
 *   3. The generator sees the marker, breaks out of its loop, and the
 *      finally block runs `cancelOnce()` — which is a no-op because
 *      `triggerAbort` already flipped the `cancelled` flag. Some
 *      backends throw on double-cancel, so routing every cancel site
 *      through `cancelOnce` keeps abort behavior deterministic.
 *
 * The finally block is also the landing site for the consumer calling
 * `.return()` on the outer generator — the existing `yield` cleanup
 * covers that case. Signal-driven abort covers the window where the
 * consumer cannot reach `.return()` because they are blocked waiting
 * for the very `yield` that `waitForItem()` is gating.
 *
 * @internal Exported so the VLM wrapper (`@mlx-node/vlm`) can reuse the
 * exact same bridge without duplicating the plumbing. Not part of the
 * public API — may change without notice.
 */
export async function* _runChatStream(
  startCall: (callback: (err: Error | null, chunk: ChatStreamChunk) => void) => Promise<ChatStreamHandle>,
  signal?: AbortSignal,
): AsyncGenerator<ChatStreamEvent> {
  const queue: Array<{ chunk?: ChatStreamChunk; error?: Error; aborted?: boolean }> = [];
  let resolve: (() => void) | null = null;

  const waitForItem = () =>
    queue.length > 0
      ? Promise.resolve()
      : new Promise<void>((r) => {
          resolve = r;
        });

  const notify = () => {
    if (resolve) {
      const r = resolve;
      resolve = null;
      r();
    }
  };

  const callback = (err: Error | null, chunk: ChatStreamChunk) => {
    queue.push(err ? { error: err } : { chunk });
    notify();
  };

  const handle = await startCall(callback);

  // Guard against double-cancel. Some native backends throw on a
  // second `cancel()`; we route every cancel site through this
  // helper so the abort path (via `triggerAbort`) and the unwind
  // path (via the `finally` block) don't cancel twice and so any
  // backend that does throw is swallowed rather than escaping as
  // an error out of an otherwise-clean early termination.
  let cancelled = false;
  const cancelOnce = (): void => {
    if (cancelled) return;
    cancelled = true;
    try {
      handle.cancel();
    } catch {
      // Native backend threw on cancel — nothing actionable here.
      // Swallow so aborted streams still surface as a clean early
      // termination via the synthetic `aborted` marker rather than
      // as an unexpected error out of the generator.
    }
  };

  // Signal-driven fast-abort. If the signal is already aborted at
  // attach time we still arm the listener so the synchronous abort
  // dispatch path runs below (calling `handle.cancel()` after the
  // handle is in hand, not before — there is nothing to cancel
  // pre-start). For an already-fired signal Node delivers the
  // `'abort'` event on the next microtask.
  let onAbort: (() => void) | null = null;
  if (signal != null) {
    const triggerAbort = (): void => {
      // Cancel the native side first so any work-in-flight winds
      // down ASAP. `cancelOnce` is idempotent — the finally block
      // will invoke it again after the generator unwinds, but the
      // second call becomes a no-op.
      cancelOnce();
      // Push a synthetic abort marker so the consumer-visible
      // generator breaks out of its loop at the next iteration
      // rather than waiting for a native chunk that will never
      // arrive (e.g. client disconnect before first token).
      queue.push({ aborted: true });
      notify();
    };
    if (signal.aborted) {
      // Signal already fired before we could attach — dispatch
      // synchronously so the waitForItem below resolves immediately.
      triggerAbort();
    } else {
      onAbort = triggerAbort;
      signal.addEventListener('abort', onAbort, { once: true });
    }
  }

  try {
    loop: while (true) {
      await waitForItem();
      while (queue.length > 0) {
        const item = queue.shift()!;
        if (item.aborted) {
          // Consumer asked us to stop before the next chunk landed.
          // Break out so the finally block runs `handle.cancel()`
          // (idempotent — `triggerAbort` already called it) and the
          // consumer-side `for await` unblocks cleanly. We do NOT
          // throw an AbortError: callers (e.g. server endpoints that
          // flag client disconnect) have already decided this is not
          // an error from their perspective, just an early stop.
          break loop;
        }
        if (item.error) throw item.error;
        const chunk = item.chunk!;
        if (chunk.done) {
          // The native `ChatStreamChunk` carries `cachedTokens` on the
          // terminal (`done == true`) chunk for every streaming entry
          // point. Emit it on the final event verbatim — undefined means
          // the native dispatch did not populate it (e.g. a bridge-level
          // mock or an in-process driver), in which case downstream
          // consumers treat the absence as "unknown / not plumbed" and
          // skip emitting e.g. `X-Cached-Tokens` rather than reporting a
          // fabricated `0`.
          const chunkWithCached = chunk as ChatStreamChunk & { cachedTokens?: number };
          const finalEvent: ChatStreamFinal = {
            text: chunk.text,
            done: true,
            finishReason: chunk.finishReason!,
            toolCalls: chunk.toolCalls ?? [],
            thinking: chunk.thinking ?? null,
            numTokens: chunk.numTokens!,
            promptTokens: chunk.promptTokens ?? 0,
            reasoningTokens: chunk.reasoningTokens ?? 0,
            rawText: chunk.rawText!,
            performance: chunk.performance ?? undefined,
          };
          if (typeof chunkWithCached.cachedTokens === 'number') {
            finalEvent.cachedTokens = chunkWithCached.cachedTokens;
          }
          yield finalEvent;
          return;
        }
        const delta: ChatStreamDelta = { text: chunk.text, done: false };
        const isReasoning = getNativeIsReasoning(chunk);
        if (isReasoning !== undefined) {
          delta.isReasoning = isReasoning;
        }
        yield delta;
      }
    }
  } finally {
    if (signal != null && onAbort != null) {
      try {
        signal.removeEventListener('abort', onAbort);
      } catch {
        // removeEventListener shouldn't throw, but stay defensive —
        // a misbehaving signal must not leak out of the finally.
      }
    }
    cancelOnce();
  }
}

// -------------------------------------------------------------------
// Generic streaming-model factory
// -------------------------------------------------------------------
//
// Every generative family (Qwen3, Qwen3.5 dense / MoE, LFM2, Gemma4,
// and the QianfanOCR VLM in `@mlx-node/vlm`) wraps its native class
// identically: capture the three callback-based session-streaming
// methods, re-expose them as `AsyncGenerator<ChatStreamEvent>`, set the
// subclass prototype in `static load`, and (for path-recording families)
// add `applyChatTemplate`. `makeStreamingModel` builds that subclass
// once so each family becomes a one-line `extends` declaration.

/**
 * The three callback-based session-streaming methods every native chat
 * class carries on its prototype (and, structurally, on its instances).
 * Used both as the native `prototype` shape and as the constructed
 * instance type so `InstanceType<NativeStreamingCtor>` resolves to the
 * full native instance surface (`generate`, `saveModel`,
 * `numParameters`, `hasMtpWeights`, …) — see {@link NativeStreamingCtor}.
 */
interface NativeStreamingInstance {
  chatStreamSessionStart: (...args: never[]) => Promise<ChatStreamHandle>;
  chatStreamSessionContinue: (...args: never[]) => Promise<ChatStreamHandle>;
  chatStreamSessionContinueTool: (...args: never[]) => Promise<ChatStreamHandle>;
}

/**
 * Minimal structural shape of a native chat model constructor that the
 * factory needs: a real `new (...)` signature (so `InstanceType<C>`
 * resolves to the native instance surface and the factory return type
 * can preserve `generate`/`saveModel`/`numParameters`/… on the public
 * subclass), a `static load(path)`, and the three callback-based
 * session-streaming methods on its prototype. The native NAPI classes
 * (`Qwen35ModelNative` etc.) all satisfy this — the concrete generic
 * `C` passed at each call site carries the full per-family instance
 * type, which `InstanceType<C>` recovers.
 */
interface NativeStreamingCtor {
  // A real constructor signature so `InstanceType<C>` resolves to the
  // concrete native instance type at each call site. The native classes
  // are NAPI-constructed (no public `new`), but structurally they satisfy
  // this and the factory never actually invokes `new` on them.
  new (...args: never[]): NativeStreamingInstance;
  // The native classes resolve `load` to their own concrete instance
  // type; the factory only needs it to be an object, and the public
  // subclass return type re-narrows via `InstanceType<C>` + the
  // `SessionCapableModel` streaming overrides.
  load(modelPath: string): Promise<object>;
  prototype: NativeStreamingInstance;
}

/** Tuning knobs for {@link makeStreamingModel}. */
interface StreamingModelOptions {
  /**
   * When `true`, `static load` records the on-disk model path so the
   * generated subclass can serve `applyChatTemplate` from a lazily
   * constructed tokenizer (see {@link applyChatTemplateFromModelPath}).
   * When `false` (QianfanOCR) the path is not recorded and
   * `applyChatTemplate` is omitted.
   */
  recordModelPath: boolean;
  /**
   * Whether to attach an `applyChatTemplate` method. Defaults to
   * `recordModelPath` because the method can only work when a path was
   * recorded. Qwen3 (first-gen) records its path but exposes no
   * `applyChatTemplate`; pass `applyTemplate: false` to suppress the
   * method while still recording the path.
   */
  applyTemplate?: boolean;
}

/**
 * Shared base type produced by the factory: a `SessionCapableModel`
 * whose static surface still exposes `load`. Concrete families extend
 * the returned class with an empty body so they inherit everything and
 * pick up the correct `.name` (and working `instanceof`) for free.
 */
export type StreamingModel = SessionCapableModel;

/**
 * The effective `applyTemplate` flag resolved from the options literal:
 * an explicit `applyTemplate` wins, otherwise it defaults to `recordModelPath`
 * — mirroring the runtime `opts.applyTemplate ?? recordPath`. Requires the
 * options to be inferred as a literal (the `const` type parameter below), so
 * `{ recordModelPath: true }` yields `true`, not `boolean`.
 */
type ResolvedApplyTemplate<O extends StreamingModelOptions> = O extends {
  applyTemplate: boolean;
}
  ? O['applyTemplate']
  : O['recordModelPath'];

/**
 * Instance surface of a generated streaming wrapper: the native instance (minus
 * its native callback chat methods) plus the `SessionCapableModel` generator
 * overrides. When the wrapper installs `applyChatTemplate` (the templating
 * variants — `applyTemplate` resolves truthy), it is re-added as a REQUIRED
 * member instead of the optional one `SessionCapableModel` declares, so
 * `model.applyChatTemplate(...)` is not a possibly-undefined call after
 * `load()` in strict TS.
 */
type StreamingInstance<C extends NativeStreamingCtor, O extends StreamingModelOptions> = Omit<
  InstanceType<C>,
  keyof SessionCapableModel
> &
  SessionCapableModel &
  (ResolvedApplyTemplate<O> extends true ? Required<Pick<SessionCapableModel, 'applyChatTemplate'>> : object);

/**
 * Build the streaming-model subclass for a native chat model class.
 *
 * The returned class:
 *   - captures the three native callback-based session-streaming methods
 *     from `NativeClass.prototype`,
 *   - overrides them as `async *` generators delegating to
 *     {@link _runChatStream} with identical argument plumbing (including
 *     `config ?? null`, `images`, `isError ?? null`, and the `signal`),
 *   - overrides `static load` to re-prototype the native instance onto
 *     the concrete subclass (`this`) and optionally record the path,
 *   - exposes `applyChatTemplate` when `opts.applyTemplate` (defaulting
 *     to `opts.recordModelPath`).
 *
 * @internal Exported so the VLM wrapper (`@mlx-node/vlm`) builds its
 * `QianfanOCRModel` from the same factory. Not part of the public API.
 */
export function makeStreamingModel<C extends NativeStreamingCtor, const O extends StreamingModelOptions>(
  NativeClass: C,
  opts: O,
): {
  // Preserve the native instance surface (`generate`, `batchGenerate`,
  // `saveModel`, `numParameters`, `hasMtpWeights`, …) by re-deriving it
  // from `InstanceType<C>`, while letting the `SessionCapableModel`
  // streaming overrides (AsyncGenerator chat methods + `resetCaches`,
  // etc.) win. `Omit<…, keyof SessionCapableModel>` drops the native
  // callback-style chat methods so the re-added `SessionCapableModel`
  // generator signatures take precedence. `applyChatTemplate` is required on
  // templating variants (see `StreamingInstance`).
  //
  // `ConstructorParameters<C>` (not `never[]`) keeps each native config
  // constructor — e.g. `new Gemma4Model(config)` / `new QianfanOCRModel(config)`
  // — visible on the generated wrapper for TypeScript consumers. Likewise,
  // `Parameters<C['load']>` keeps each family's native load signature —
  // `[modelPath]` for most, `[modelPath, options?]` for Gemma4
  // (`Gemma4LoadOptions.draftModelPath` — external DSpark or Google
  // assistant draft).
  new (...args: ConstructorParameters<C>): StreamingInstance<C, O>;
  load(...args: Parameters<C['load']>): Promise<StreamingInstance<C, O>>;
} {
  const recordPath = opts.recordModelPath;
  const applyTemplate = opts.applyTemplate ?? recordPath;

  // Capture the native callback-based methods before the subclass
  // overrides below shadow them on the prototype.
  const nativeStart = NativeClass.prototype.chatStreamSessionStart;
  const nativeContinue = NativeClass.prototype.chatStreamSessionContinue;
  const nativeContinueTool = NativeClass.prototype.chatStreamSessionContinueTool;

  // `NativeClass` is structurally a constructor; cast to a concrete
  // constructor type so `class extends` accepts it. Runtime behavior is
  // unchanged — we extend the real native class.
  const Base = NativeClass as unknown as new (...args: never[]) => SessionCapableModel;

  class StreamingModelImpl extends Base {
    static async load(modelPath: string, ...rest: unknown[]): Promise<StreamingModel> {
      // Forward any trailing family-specific load options verbatim (e.g.
      // Gemma4's `Gemma4LoadOptions` with `draftModelPath`); families whose
      // native `load` takes only the path receive no extras. The public
      // signature is re-narrowed per family via `Parameters<C['load']>` in
      // the factory return type below.
      const instance = await (NativeClass.load as (...args: unknown[]) => Promise<object>)(modelPath, ...rest);
      // Use `this.prototype` (not `StreamingModelImpl.prototype`) so the
      // concrete subclass declared per family supplies the prototype and
      // `instanceof ConcreteSubclass` holds.
      Object.setPrototypeOf(instance, this.prototype);
      if (recordPath) rememberModelPath(instance, modelPath);
      return instance as unknown as StreamingModel;
    }

    // The native methods are callback-based, but `Base` is typed as a
    // `SessionCapableModel` constructor (whose streaming methods already
    // return `AsyncGenerator<ChatStreamEvent>`), so these overrides are
    // type-compatible and need no `@ts-expect-error` suppression. The
    // callback bridging happens at runtime via the captured natives.
    async *chatStreamSessionStart(
      messages: ChatMessage[],
      config?: ChatConfig | null,
      signal?: AbortSignal,
    ): AsyncGenerator<ChatStreamEvent> {
      yield* _runChatStream(
        (callback) => nativeStart.call(this, messages as never, (config ?? null) as never, callback as never),
        signal,
      );
    }

    async *chatStreamSessionContinue(
      userMessage: string,
      images: Uint8Array[] | null,
      audio: Uint8Array[] | null,
      config?: ChatConfig | null,
      signal?: AbortSignal,
    ): AsyncGenerator<ChatStreamEvent> {
      yield* _runChatStream(
        (callback) =>
          nativeContinue.call(
            this,
            userMessage as never,
            images as never,
            audio as never,
            (config ?? null) as never,
            callback as never,
          ),
        signal,
      );
    }

    async *chatStreamSessionContinueTool(
      toolCallId: string,
      content: string,
      config?: ChatConfig | null,
      signal?: AbortSignal,
      isError?: boolean | null,
    ): AsyncGenerator<ChatStreamEvent> {
      yield* _runChatStream(
        (callback) =>
          nativeContinueTool.call(
            this,
            toolCallId as never,
            content as never,
            (config ?? null) as never,
            callback as never,
            (isError ?? null) as never,
          ),
        signal,
      );
    }
  }

  if (applyTemplate) {
    Object.defineProperty(StreamingModelImpl.prototype, 'applyChatTemplate', {
      configurable: true,
      writable: true,
      value(
        this: object,
        messages: ChatMessage[],
        addGenerationPrompt?: boolean | null,
        tools?: ToolDefinition[] | null,
        enableThinking?: boolean | null,
      ): Promise<Uint32Array> {
        return applyChatTemplateFromModelPath(this, messages, addGenerationPrompt, tools, enableThinking);
      },
    });
  }

  return StreamingModelImpl as unknown as {
    new (...args: ConstructorParameters<C>): StreamingInstance<C, O>;
    load(...args: Parameters<C['load']>): Promise<StreamingInstance<C, O>>;
  };
}

/**
 * Qwen3.5 dense model with AsyncGenerator-based session streaming.
 *
 * The empty `extends` inherits the factory's streaming overrides,
 * `static load`, and `applyChatTemplate`, and supplies the concrete
 * `.name === 'Qwen35Model'` and a working `instanceof`. Records its
 * model path so `applyChatTemplate` can serve a lazily built tokenizer.
 */
export class Qwen35Model extends makeStreamingModel(Qwen35ModelNative, { recordModelPath: true }) {}

/** Qwen3.5 MoE model — see {@link Qwen35Model} for the wrapper shape. */
export class Qwen35MoeModel extends makeStreamingModel(Qwen35MoeModelNative, { recordModelPath: true }) {}

/** LFM2 model (text-only) — see {@link Qwen35Model} for the wrapper shape. */
export class Lfm2Model extends makeStreamingModel(Lfm2ModelNative, { recordModelPath: true }) {}

/** Gemma4 model (text-only) — see {@link Qwen35Model} for the wrapper shape. */
export class Gemma4Model extends makeStreamingModel(Gemma4ModelNative, { recordModelPath: true }) {}

/**
 * Qwen3 (first-gen, text-only) model.
 *
 * Records its model path (so prototype-set + path-recording match the
 * other families) but exposes NO `applyChatTemplate` —
 * `applyTemplate: false` suppresses that method.
 */
export class Qwen3Model extends makeStreamingModel(Qwen3ModelNative, {
  recordModelPath: true,
  applyTemplate: false,
}) {}

// -------------------------------------------------------------------
// Compile-time conformance check
// -------------------------------------------------------------------
//
// Ensures each family class structurally satisfies
// `SessionCapableModel` so `ChatSession<XxxModel>` type-checks in
// downstream code. Compile-only — the `null as unknown as T`
// placeholder never runs. If a factory override signature drifts away
// from the interface, this block fails to compile.
function _assertSessionCapable(): void {
  const _qwen35: SessionCapableModel = null as unknown as Qwen35Model;
  const _moe: SessionCapableModel = null as unknown as Qwen35MoeModel;
  const _lfm2: SessionCapableModel = null as unknown as Lfm2Model;
  const _gemma4: SessionCapableModel = null as unknown as Gemma4Model;
  const _qwen3: SessionCapableModel = null as unknown as Qwen3Model;
  void _qwen35;
  void _moe;
  void _lfm2;
  void _gemma4;
  void _qwen3;
}
void _assertSessionCapable;
