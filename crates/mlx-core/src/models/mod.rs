/**
 * Models Module
 *
 * Contains all model implementations.
 */
pub(crate) mod attention_core;
pub(crate) mod chat_napi;
pub(crate) mod forward;
pub mod gemma4;
pub mod harrier;
pub mod int8_gemm;
pub mod k2_horizon;
pub mod lfm2;
pub mod mtp_drafter;
pub mod muse_glimmer;
pub mod nemotron_h;
pub mod paddleocr_vl;
pub mod paged_config;
pub mod pp_doc_ori;
pub mod pp_doc_unwarp;
pub mod pp_doclayout_v3;
pub mod pp_text_det;
pub mod pp_text_rec;
pub mod privacy_filter;
pub mod qianfan_ocr;
pub mod quant_dispatch;
pub mod quantized_linear;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_5_moe;
pub mod qwen3_asr;
pub mod qwen4_exp;
