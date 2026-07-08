/**
 * Models Module
 *
 * Contains all model implementations.
 */
pub(crate) mod chat_napi;
// Families not constructed by the browser (Qwen3.5-0.8B dense only) that pull
// in native-only `crate::engine` / `crate::moe` machinery are excluded from the
// wasm build (task #68 wasm-source parity). Native is unaffected.
#[cfg(not(target_family = "wasm"))]
pub mod gemma4;
#[cfg(not(target_family = "wasm"))]
pub mod harrier;
#[cfg(not(target_family = "wasm"))]
pub mod lfm2;
pub mod mtp_drafter;
pub mod paddleocr_vl;
pub mod pp_doc_ori;
pub mod pp_doc_unwarp;
pub mod pp_doclayout_v3;
pub mod pp_text_det;
pub mod pp_text_rec;
#[cfg(not(target_family = "wasm"))]
pub mod privacy_filter;
#[cfg(not(target_family = "wasm"))]
pub mod qianfan_ocr;
pub mod quant_dispatch;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_5_moe;
