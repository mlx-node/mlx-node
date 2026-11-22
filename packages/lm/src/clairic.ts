import { ClairicCancellation, ClairicModel as NativeClairicModel } from '@mlx-node/core';

import { renderClairicDocuments } from './clairic-pdf.js';
import {
  isClairicModel,
  type ClairicExtractionOptions,
  type ClairicRequest,
  type ClairicResult,
  type ExtractionModel,
  type RenderedPdfPage,
} from './clairic-types.js';

export { isClairicModel };

/** Clairic (System One: Extract) structured extraction over Qwen3.5-VL. */
export class ClairicModel implements ExtractionModel {
  readonly kind = 'decision' as const;
  readonly mode = 'extract' as const;

  /** @internal Use ClairicModel.load(). */
  constructor(private readonly native: NativeClairicModel) {}

  static async load(modelPath: string): Promise<ClairicModel> {
    return new ClairicModel(await NativeClairicModel.load(modelPath));
  }

  get modelId(): string {
    return this.native.modelId;
  }

  get maxLength(): number {
    return this.native.maxLength;
  }

  async extract(request: ClairicRequest, options?: ClairicExtractionOptions): Promise<ClairicResult> {
    return this.extractRaw(JSON.stringify(request), options);
  }

  async extractRaw(raw: string, options?: ClairicExtractionOptions): Promise<ClairicResult> {
    const signal = options?.signal;
    signal?.throwIfAborted();

    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch (error) {
      throw new Error(`Invalid Clairic request: ${error instanceof Error ? error.message : 'malformed JSON'}`, {
        cause: error,
      });
    }
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('Invalid Clairic request: request body must be a JSON object');
    }

    let pdfPages: RenderedPdfPage[] | undefined;
    if ('documents' in parsed && (parsed as Record<string, unknown>).documents !== undefined) {
      try {
        pdfPages = await renderClairicDocuments(parsed as Record<string, unknown>, { signal });
      } catch (error) {
        if (signal?.aborted || (error instanceof Error && error.name === 'AbortError')) {
          throw error;
        }
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(
          message.startsWith('Invalid Clairic request:') ? message : `Invalid Clairic request: ${message}`,
          { cause: error },
        );
      }
    }
    signal?.throwIfAborted();

    const cancellation = new ClairicCancellation();
    const abort = () => cancellation.cancel();
    signal?.addEventListener('abort', abort, { once: true });
    try {
      const result = await this.native.extractJson(raw, pdfPages, cancellation);
      signal?.throwIfAborted();
      return JSON.parse(result) as ClairicResult;
    } finally {
      signal?.removeEventListener('abort', abort);
    }
  }
}
