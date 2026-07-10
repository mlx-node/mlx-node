// revive.ts — rehydrate a baked J-lens frame back into a live `LensReadoutRun`.
//
// The offline bake (scripts/jlens/bake.mts) serializes each curated prompt to a
// JSON-safe envelope: every typed array in a `LensReadoutRun`
// (`cells[].topKLogits` / `cells[].topKProbs` : Float32Array, `pinned[].ranks` :
// Int32Array) is flattened to a plain `number[]` so it survives JSON. This
// module is the exact inverse: it walks a `SerializedRun` back into a real
// `LensReadoutRun` with the typed arrays restored, so the SAME `buildLensSlice`
// + jlens children that render a LIVE worker run render the baked frame
// UNCHANGED (design decision D11: one `LensSliceData` type consumed by BOTH the
// baked JSON and the live `lensReadout` RPC).
//
// The serialized shapes below MIRROR scripts/jlens/bake.mts:88-108 byte for byte
// — keep them in sync if the bake schema ever changes.

import type { LensCell, LensPinned, LensReadoutRun, TokenInfo } from '../../../../src/inspector-types';

/** JSON-safe mirror of {@link LensCell}: the two Float32Arrays are `number[]`. */
export type SerializedCell = {
  layer: number;
  position: number;
  argmaxId: number;
  topKIds: number[];
  topKLogits: number[];
  topKProbs: number[];
  topKTexts: string[];
};

/** JSON-safe mirror of {@link LensPinned}: the Int32Array `ranks` is `number[]`. */
export type SerializedPinned = { tokenId: number; tokenText: string; ranks: number[] };

/** JSON-safe mirror of {@link LensReadoutRun} (typed arrays → `number[]`). */
export type SerializedRun = {
  promptLen: number;
  topK: number;
  useJacobian: boolean;
  jacobianApplied: boolean;
  layers: number[];
  tokens: { id: number; text: string }[];
  cells: SerializedCell[];
  pinned: SerializedPinned[];
};

/** Provenance sidecar baked alongside each frame (never user-facing verbatim). */
export type BakedMeta = {
  model: string;
  lensVersion: string;
  nPrompts: number;
  bakedDate: string;
  method: string;
};

/** One committed `baked/<slug>.json` file: BOTH a logit and a jacobian run per
 *  curated prompt, so the widget's default view needs no model and no pack. */
export type BakedFile = {
  slug: string;
  prompt: string;
  concepts: string[];
  /** Index-aligned with `concepts`/`logit.pinned`: `true` where a concept
   *  tokenized to more than one piece and only its first token is tracked. */
  partialFlags: boolean[];
  layers: number[];
  meta: BakedMeta;
  /** `useJacobian:false` run — the plain logit lens. `jacobianApplied:false`. */
  logit: SerializedRun;
  /** `useJacobian:true` run — the fitted Jacobian lens. `jacobianApplied:true`. */
  jacobian: SerializedRun;
};

/**
 * Rebuild a real {@link LensReadoutRun} from its serialized (baked) form: the
 * inverse of `serializeRun` in scripts/jlens/bake.mts. Only the flattened typed
 * arrays are reconstructed (`cells[].topKLogits`/`topKProbs` → `Float32Array`,
 * `pinned[].ranks` → `Int32Array`); every other field passes through unchanged.
 * The `useJacobian` / `jacobianApplied` flags are preserved verbatim so the
 * widget can label the frame honestly (never "Jacobian applied" unless the run
 * actually set `jacobianApplied === true`).
 */
export function reviveRun(s: SerializedRun): LensReadoutRun {
  const cells: LensCell[] = s.cells.map((c) => ({
    layer: c.layer,
    position: c.position,
    argmaxId: c.argmaxId,
    topKIds: c.topKIds,
    topKLogits: Float32Array.from(c.topKLogits),
    topKProbs: Float32Array.from(c.topKProbs),
    topKTexts: c.topKTexts,
  }));

  const pinned: LensPinned[] = s.pinned.map((p) => ({
    tokenId: p.tokenId,
    tokenText: p.tokenText,
    ranks: Int32Array.from(p.ranks),
  }));

  const tokens: TokenInfo[] = s.tokens.map((t) => ({ id: t.id, text: t.text }));

  return {
    promptLen: s.promptLen,
    topK: s.topK,
    useJacobian: s.useJacobian,
    jacobianApplied: s.jacobianApplied,
    layers: s.layers,
    tokens,
    cells,
    pinned,
  };
}
