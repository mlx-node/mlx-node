//! Muse-Glimmer-30B: windowed vision tower + 3-stage projector + hybrid text
//! decoder ([slide,slide,slide,full] x13, NoPE on the full layers).

pub mod attention;
pub(crate) mod cold_sidecar;
pub mod config;
pub mod decoder_layer;
pub(crate) mod dflash;
pub(crate) mod dflash_decode;
pub mod kv_cache;
pub mod mlp;
pub mod model;
pub mod output_parser;
pub mod persistence;
pub mod stream_guard;

pub use model::MuseGlimmerModel;
