//! K2-Horizon (IFM `K2HorizonForCausalLM`) — dense decoder-only LM.
//!
//! GQA + grouped RMSNorm (`layernorm_num_groups=4`) + SwiGLU + full RoPE,
//! untied `lm_head`. Loads compressed-tensors block-FP8 checkpoints
//! (dequantized to the target dtype at ingest) or MLX mxfp8-converted
//! checkpoints. Pure standard-KV: flat `KVCache` path, block-paged
//! prefix cache, and the continuous-batching scheduler — no
//! conv/recurrent sidecar, no MTP.

pub(crate) mod attention;
pub(crate) mod config;
pub(crate) mod finalize;
pub(crate) mod layer;
pub(crate) mod model;
pub(crate) mod persistence;
