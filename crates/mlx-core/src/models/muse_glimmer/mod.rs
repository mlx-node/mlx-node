//! Muse-Glimmer-30B: windowed vision tower + 3-stage projector + hybrid text
//! decoder ([slide,slide,slide,full] x13, NoPE on the full layers).

pub mod config;
pub mod output_parser;
pub mod stream_guard;
