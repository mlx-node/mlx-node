// Canonical data contract for the Education App "Inspector" runs.
//
// The frontend (chapter components and inspector visualizations under
// demo/learn/) imports these types. The backend (Rust NAPI in crates/mlx-core)
// must serialize its inspector payloads into shapes that match these types
// after NAPI / SAB transport.
//
// Until the backend hook lands, chapter components may construct mock
// AttentionRun values for development. The mock shape must validate against
// these types so swap-in is mechanical.

export type TokenInfo = {
  /** Token id from the Qwen3 tokenizer. */
  id: number;
  /** Decoded text for this token. May contain leading whitespace or special
   *  markers (e.g. "Ġthe", "<|im_start|>"). The frontend renders this verbatim. */
  text: string;
};

export type AttentionLayerKind = 'full' | 'linear';

export type AttentionLayer = {
  /** 0-based index into the model's full layer stack (not the index among
   *  full-attention layers only). */
  layerIndex: number;
  /** Qwen3.5 is hybrid: only 'full' layers have classic softmax attention
   *  scores. 'linear' layers (GatedDeltaNet) appear in the layer list for
   *  context but their `scores` field is empty. */
  kind: AttentionLayerKind;
  /** Number of query heads. */
  numHeads: number;
  /** Number of KV heads (GQA: numKvHeads <= numHeads, evenly divides numHeads). */
  numKvHeads: number;
  /** Softmaxed attention scores. Length === numHeads * seqLen * seqLen.
   *  Layout: [head_index][query_token_index][key_token_index], row-major.
   *  For 'linear' layers this is a zero-length Float32Array. */
  scores: Float32Array;
};

export type ModelMeta = {
  /** Human-readable model name (e.g. "Qwen3.5-0.8B"). */
  name: string;
  /** Total number of layers in the model. */
  numLayers: number;
  /** Indices (into the full 0..numLayers-1 range) of layers that use full
   *  attention. The inspector returns scores only for these. */
  fullAttentionLayerIndices: number[];
};

export type AttentionRun = {
  /** The prompt the user typed. */
  prompt: string;
  /** Prompt tokens after tokenization. Length === seqLen. */
  tokens: TokenInfo[];
  /** The single token generated for this inspector run. */
  generatedToken: TokenInfo;
  /** One entry per layer captured. Order matches `modelMeta.fullAttentionLayerIndices`. */
  attention: AttentionLayer[];
  /** Model identification + which layers carry attention scores. */
  modelMeta: ModelMeta;
};

// -----------------------------------------------------------------------------
// Worker-bridge protocol (extends the existing mlx-worker.ts request/response).
// -----------------------------------------------------------------------------

export type InspectorRequest = {
  type: 'runForInspector';
  /** Caller-supplied correlation id. The worker echoes this in the response. */
  id: string;
  prompt: string;
  /** Capture attention scores. Default: false. */
  attention?: boolean;
  /** Restrict to specific full-attention layer indices. Default: all. */
  attentionLayers?: number[];
  /** How many new tokens to generate before returning. Default: 1. */
  maxNewTokens?: number;
};

export type InspectorResult =
  | { type: 'inspectorResult'; id: string; result: AttentionRun }
  | { type: 'inspectorError'; id: string; error: string };

// -----------------------------------------------------------------------------
// Protocol string constants
// -----------------------------------------------------------------------------
//
// The worker (`mlx-worker.ts`) and the main-thread bridge
// (`demo/lib/inspector-client.ts`) both need to write/read these `type` fields
// on `postMessage` payloads. Exporting them here keeps the single source of
// truth in this file so the two ends can't drift on a typo. The TypeScript
// literal unions above (`type: 'runForInspector'` etc.) are intentionally kept
// as plain string literals for readability; the constants are the *runtime*
// values both sides use.

export const INSPECTOR_REQUEST_TYPE = 'runForInspector' as const;
export const INSPECTOR_RESULT_TYPE = 'inspectorResult' as const;
export const INSPECTOR_ERROR_TYPE = 'inspectorError' as const;

// -----------------------------------------------------------------------------
// Notes for the backend implementer (crates/mlx-core, Rust).
// -----------------------------------------------------------------------------
//
// The Rust side should expose a NAPI method on the model class (or a new
// dedicated handle) with this signature:
//
//   #[napi]
//   pub fn run_for_inspector(
//     &self,
//     prompt: String,
//     opts: InspectorRunOptions,
//   ) -> napi::Result<AttentionRunNapi>;
//
// Where:
//
//   #[napi(object)]
//   pub struct InspectorRunOptions {
//     pub attention: Option<bool>,
//     pub attention_layers: Option<Vec<i32>>,
//     pub max_new_tokens: Option<i32>,
//   }
//
//   #[napi(object)]
//   pub struct AttentionRunNapi {
//     pub prompt: String,
//     pub tokens: Vec<TokenInfoNapi>,
//     pub generated_token: TokenInfoNapi,
//     pub attention: Vec<AttentionLayerNapi>,
//     pub model_meta: ModelMetaNapi,
//   }
//
//   #[napi(object)]
//   pub struct AttentionLayerNapi {
//     pub layer_index: i32,
//     pub kind: String, // "full" | "linear"
//     pub num_heads: i32,
//     pub num_kv_heads: i32,
//     pub scores: napi::bindgen_prelude::Float32Array,
//   }
//
// NAPI-RS converts snake_case Rust field names to camelCase JS field names
// automatically; the generated index.d.cts will match the TypeScript types
// above. Verify with `yarn build` and a `tsc --noEmit` smoke check.
//
// During the forward pass in qwen3_5/attention.rs, after computing
// `softmax(QK^T / sqrt(d_head))` and BEFORE the matmul with V, if the
// inspector toggle is on, copy the score tensor (shape: [num_heads, seq_q, seq_k])
// out of GPU memory into a CPU-side Vec<f32>. This is the layout consumed
// above (row-major, head-major).
//
// Performance: this hook is off by default. With it on, expect ~1.5-2x decode
// slowdown depending on captured layer count. Document this in the lesson copy.
