// model-loader-state.ts — pure, React-free state derivation for the model
// loader. Kept dependency-free (no React, no worker, no inspector imports) so
// it can be unit-tested in isolation and so importing it never drags the model
// runtime into the prose/render path.

export type ModelLoaderStatus = 'idle' | 'loading' | 'ready' | 'error';

/**
 * Inputs that determine the model loader's externally-visible status. These
 * mirror the App-level state in app.tsx (errorBanner, modelReady, loadKickoff).
 */
export type ModelLoaderStatusInputs = {
  /** Non-null when a load attempt surfaced an error. */
  errorBanner: string | null;
  /** True once the full model is loaded and the worker is ready to infer. */
  modelReady: boolean;
  /** Monotonic counter bumped each time a load (full or device-only) STARTS. */
  loadKickoff: number;
  /**
   * True when the in-flight (or most recent) bring-up is a DEVICE-ONLY init —
   * it brings the WebGPU device + WASM runtime up but does NOT fetch the model.
   * A device-only bring-up bumps `loadKickoff` too, so without this flag the
   * derivation would report `loading` forever after a device-only init (e.g.
   * the Training chapter) even though no full-model download is in flight. The
   * status is MODEL-centric: a device-only load is not a model load, so it must
   * read as `idle` (the model still hasn't been requested).
   */
  deviceOnly: boolean;
};

/**
 * Derive the MODEL-centric loader status from raw App state.
 *
 * Precedence: error > ready > loading > idle.
 *
 * Crucially, status is `idle` until a FULL-model load actually STARTS. Two
 * things must NOT flip it to `loading`:
 *   - Merely *detecting* a hosted model (hostedModelAvailable === true) — nothing
 *     has downloaded yet.
 *   - A device-only bring-up (`deviceOnly`) — it bumps `loadKickoff` but never
 *     fetches the model, so model-facing surfaces (model-mode consent panels,
 *     the chat overlay) must fall back to their `idle` "Load model" prompt
 *     rather than a CTA-less spinner.
 */
export function deriveModelLoaderStatus({
  errorBanner,
  modelReady,
  loadKickoff,
  deviceOnly,
}: ModelLoaderStatusInputs): ModelLoaderStatus {
  if (errorBanner != null) return 'error';
  if (modelReady) return 'ready';
  if (loadKickoff > 0 && !deviceOnly) return 'loading';
  return 'idle';
}

/**
 * The consent layer covers either a full-model demo or a device-only (WebGPU,
 * no download) demo. The resource each waits on differs.
 */
export type ConsentMode = 'model' | 'device';

/**
 * Which visual state the consent layer should render.
 *
 *   ready   — the demo's resource is up; render the live demo (children).
 *   prompt  — pre-download "frozen" panel with the consent CTA.
 *   loading — spinner + progress; CTA disabled.
 *   error   — error message + Retry CTA.
 */
export type ConsentLayerState = 'ready' | 'prompt' | 'loading' | 'error';

export type ConsentLayerInputs = {
  mode: ConsentMode;
  status: ModelLoaderStatus;
  /** True once the WebGPU device + WASM runtime are up (device-only chapters). */
  deviceReady: boolean;
  /**
   * Monotonic load-kickoff counter. Needed by the DEVICE mode: a device-only
   * bring-up reads as model-status `idle` (the model isn't being fetched), so a
   * device demo can't use `status` alone to know its device init is still in
   * flight — it falls back to "a load has started" (`loadKickoff > 0`).
   */
  loadKickoff: number;
};

/**
 * Select the consent-layer state, per resource mode.
 *
 *   device — needs only the WebGPU device. Ready once `deviceReady` (a
 *            device-only bring-up) or a full model is ready; `deviceReady`
 *            short-circuits even a model error (a device demo doesn't depend on
 *            the model). Loading while the device phase of a full load is in
 *            flight (`status === 'loading'`) OR any load has started
 *            (`loadKickoff > 0`, which covers a device-only init that reads as
 *            model-status `idle`).
 *   model  — gates strictly on the full model via the MODEL-centric `status`. A
 *            device-only init leaves `status` `idle`, so this correctly shows
 *            the "Load model" prompt (not a stuck spinner) once the device is up.
 *
 * `error` takes precedence over `loading` so a surfaced load failure always
 * shows the retry affordance rather than a stuck spinner.
 */
export function selectConsentLayerState({
  mode,
  status,
  deviceReady,
  loadKickoff,
}: ConsentLayerInputs): ConsentLayerState {
  if (mode === 'device') {
    if (deviceReady || status === 'ready') return 'ready';
    if (status === 'error') return 'error';
    if (status === 'loading' || loadKickoff > 0) return 'loading';
    return 'prompt';
  }
  if (status === 'ready') return 'ready';
  if (status === 'error') return 'error';
  if (status === 'loading') return 'loading';
  return 'prompt';
}

/**
 * Whether clearLoadError (run on navigation to a fresh surface) should clear a
 * surfaced error and reset the loader.
 *
 * Only a FAILED LOAD error — an error surfaced while `modelReady === false` — is
 * clearable: clearing it tears the dead worker down and returns the loader to a
 * neutral `idle` prompt for the next surface.
 *
 * A POST-READY error (`modelReady === true` — a runtime failure on an
 * already-loaded model, e.g. a poisoned GPU bridge / "reload required", a worker
 * crash, or a session error) is NOT clearable. Silently clearing it on
 * navigation would hide the failure AND let deriveModelLoaderStatus fall back to
 * `ready`, mounting live demos / dropping the chat gate against a
 * dead-or-poisoned worker. Those errors stay `error` (with a Retry affordance)
 * until the user explicitly retries or reloads.
 */
export function isClearableLoadError({
  errorBanner,
  modelReady,
}: {
  errorBanner: string | null;
  modelReady: boolean;
}): boolean {
  return errorBanner != null && !modelReady;
}
