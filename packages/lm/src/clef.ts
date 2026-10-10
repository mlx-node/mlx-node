import { ClefModel as NativeClefModel, ClefCancellation } from '@mlx-node/core';

import {
  formatClefResult,
  validateClefRequest,
  type ClefNativeResult,
  type ClefRequest,
  type ClefResult,
  type DecisionModel,
} from './clef-types.js';

/** CLEF/CLEF Flash text inference. All questions share one native backbone pass. */
export class ClefModel implements DecisionModel {
  readonly kind = 'decision' as const;
  /** @internal Use ClefModel.load(). */
  constructor(private readonly native: NativeClefModel) {}
  static async load(path: string): Promise<ClefModel> {
    return new ClefModel(await NativeClefModel.load(path));
  }
  async decide(request: ClefRequest, options?: { signal?: AbortSignal }): Promise<ClefResult> {
    return this.decideRaw(JSON.stringify(request), options);
  }
  async decideRaw(raw: string, options?: { signal?: AbortSignal }): Promise<ClefResult> {
    const request: unknown = JSON.parse(raw);
    validateClefRequest(request);
    const signal = options?.signal;
    signal?.throwIfAborted();
    const cancellation = new ClefCancellation();
    const abort = () => cancellation.cancel();
    signal?.addEventListener('abort', abort, { once: true });
    try {
      const result = await this.native.decideJson(raw, cancellation);
      signal?.throwIfAborted();
      return formatClefResult(JSON.parse(result) as ClefNativeResult, request);
    } finally {
      signal?.removeEventListener('abort', abort);
    }
  }
}
