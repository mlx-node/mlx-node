// Pure unit tests for the model-loader state derivation. These exercise the
// React-free helpers in demo/lib/model-loader-state.ts — no worker, no DOM, no
// React. They lock in the core invariants of Phase 1 of the "decouple model
// loading from the UI" work:
//   - status is `idle` until a FULL-model load actually STARTS; merely detecting
//     a hosted model, OR a device-only bring-up, must NOT read as `loading`.
//   - the consent layer only reports `ready` (and thus mounts the live demo)
//     when its mode-specific resource is up.

import { describe, expect, test } from 'vitest';

import {
  type ConsentMode,
  deriveModelLoaderStatus,
  isClearableLoadError,
  type ModelLoaderStatus,
  selectConsentLayerState,
} from '../../demo/lib/model-loader-state';

describe('deriveModelLoaderStatus', () => {
  test('is idle before any load starts, even when a hosted model is detected', () => {
    // The regression this guards: hostedModelAvailable === true must NOT flip
    // the status to loading. Nothing has downloaded yet.
    expect(deriveModelLoaderStatus({ errorBanner: null, modelReady: false, loadKickoff: 0, deviceOnly: false })).toBe(
      'idle',
    );
  });

  test('is loading only once a FULL-model load has been kicked off', () => {
    expect(deriveModelLoaderStatus({ errorBanner: null, modelReady: false, loadKickoff: 1, deviceOnly: false })).toBe(
      'loading',
    );
    expect(deriveModelLoaderStatus({ errorBanner: null, modelReady: false, loadKickoff: 5, deviceOnly: false })).toBe(
      'loading',
    );
  });

  test('a device-only bring-up does NOT read as loading (stays idle)', () => {
    // The Finding-1 regression: a device-only init (Training chapter) bumps
    // loadKickoff but never fetches the model. Model-facing surfaces must fall
    // back to their `idle` "Load model" prompt, not a CTA-less spinner.
    expect(deriveModelLoaderStatus({ errorBanner: null, modelReady: false, loadKickoff: 1, deviceOnly: true })).toBe(
      'idle',
    );
    expect(deriveModelLoaderStatus({ errorBanner: null, modelReady: false, loadKickoff: 9, deviceOnly: true })).toBe(
      'idle',
    );
  });

  test('is ready once the model is loaded', () => {
    expect(deriveModelLoaderStatus({ errorBanner: null, modelReady: true, loadKickoff: 1, deviceOnly: false })).toBe(
      'ready',
    );
  });

  test('error takes precedence over everything', () => {
    expect(deriveModelLoaderStatus({ errorBanner: 'boom', modelReady: false, loadKickoff: 0, deviceOnly: false })).toBe(
      'error',
    );
    expect(deriveModelLoaderStatus({ errorBanner: 'boom', modelReady: true, loadKickoff: 3, deviceOnly: true })).toBe(
      'error',
    );
  });

  test('ready takes precedence over loading', () => {
    // modelReady wins over a non-zero kickoff counter.
    expect(deriveModelLoaderStatus({ errorBanner: null, modelReady: true, loadKickoff: 2, deviceOnly: false })).toBe(
      'ready',
    );
  });

  test('a retry after a post-ready error reports loading, not ready', () => {
    // Finding (round 3): on Retry, kickoffLoad resets modelReady to false
    // SYNCHRONOUSLY in the same batch that clears errorBanner and bumps
    // loadKickoff. That is the input state asserted here -> it must derive
    // 'loading' (the reload is in flight), never a one-render 'ready' against the
    // worker being torn down. (If modelReady were left stale-true, the precedence
    // rule above would wrongly yield 'ready'.)
    expect(deriveModelLoaderStatus({ errorBanner: null, modelReady: false, loadKickoff: 7, deviceOnly: false })).toBe(
      'loading',
    );
  });
});

describe('selectConsentLayerState', () => {
  const cases: {
    mode: ConsentMode;
    status: ModelLoaderStatus;
    deviceReady: boolean;
    loadKickoff: number;
    expected: ReturnType<typeof selectConsentLayerState>;
  }[] = [
    // model mode: ready only when status === 'ready'
    { mode: 'model', status: 'idle', deviceReady: false, loadKickoff: 0, expected: 'prompt' },
    { mode: 'model', status: 'loading', deviceReady: false, loadKickoff: 1, expected: 'loading' },
    { mode: 'model', status: 'error', deviceReady: false, loadKickoff: 1, expected: 'error' },
    { mode: 'model', status: 'ready', deviceReady: false, loadKickoff: 1, expected: 'ready' },
    // model mode ignores deviceReady — the device being up is not enough.
    { mode: 'model', status: 'idle', deviceReady: true, loadKickoff: 0, expected: 'prompt' },
    // Finding-1 lock: after a device-only init (device up, status idle, kickoff
    // bumped) a MODEL demo shows its "Load model" prompt, NOT a stuck spinner.
    { mode: 'model', status: 'idle', deviceReady: true, loadKickoff: 1, expected: 'prompt' },

    // device mode: ready when deviceReady OR a full model is ready
    { mode: 'device', status: 'idle', deviceReady: false, loadKickoff: 0, expected: 'prompt' },
    { mode: 'device', status: 'idle', deviceReady: true, loadKickoff: 1, expected: 'ready' },
    { mode: 'device', status: 'ready', deviceReady: false, loadKickoff: 1, expected: 'ready' },
    { mode: 'device', status: 'loading', deviceReady: false, loadKickoff: 1, expected: 'loading' },
    { mode: 'device', status: 'error', deviceReady: false, loadKickoff: 1, expected: 'error' },
    // A device-only bring-up in flight reads as model-status `idle`, but the
    // device demo must still show its spinner via the raw kickoff counter.
    { mode: 'device', status: 'idle', deviceReady: false, loadKickoff: 1, expected: 'loading' },
  ];

  for (const { mode, status, deviceReady, loadKickoff, expected } of cases) {
    test(`mode=${mode} status=${status} deviceReady=${deviceReady} loadKickoff=${loadKickoff} -> ${expected}`, () => {
      expect(selectConsentLayerState({ mode, status, deviceReady, loadKickoff })).toBe(expected);
    });
  }

  test('error takes precedence over loading when not ready (model mode)', () => {
    // A surfaced load failure should always show the retry affordance, never a
    // stuck spinner.
    expect(selectConsentLayerState({ mode: 'model', status: 'error', deviceReady: false, loadKickoff: 1 })).toBe(
      'error',
    );
  });

  test('device demo with the device already up is ready even mid model error', () => {
    // deviceReady short-circuits to ready before the error branch — a
    // device-only demo (Training) does not depend on the model, so a model
    // load error must not block it once the device is up.
    expect(selectConsentLayerState({ mode: 'device', status: 'error', deviceReady: true, loadKickoff: 1 })).toBe(
      'ready',
    );
  });
});

describe('isClearableLoadError', () => {
  test('clears a FAILED-load error (error while the model never became ready)', () => {
    expect(isClearableLoadError({ errorBanner: 'boom', modelReady: false })).toBe(true);
  });

  test('does NOT clear a POST-READY error (keeps it surfaced for retry)', () => {
    // The Finding (round 2) lock: a poisoned-bridge / worker / session error
    // surfaced AFTER the model was ready must stay 'error'. Silently clearing it
    // on navigation would falsely report 'ready' and mount live demos / drop the
    // chat gate against a dead-or-poisoned worker.
    expect(isClearableLoadError({ errorBanner: 'Bridge poisoned - reload required', modelReady: true })).toBe(false);
  });

  test('is a no-op when there is no error', () => {
    expect(isClearableLoadError({ errorBanner: null, modelReady: false })).toBe(false);
    expect(isClearableLoadError({ errorBanner: null, modelReady: true })).toBe(false);
  });
});
