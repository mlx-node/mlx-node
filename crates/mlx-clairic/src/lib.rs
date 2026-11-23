//! Pure-Rust Clairic structured document extraction pipeline and constrained
//! grammar decoding engine (zero `mlx-sys`, zero `napi`).

#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::doc_nested_refdefs)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

pub mod ar;
pub mod completion;
pub mod contracts;
pub mod crops;
pub mod decimal_util;
pub mod embedded;
pub mod errors;
pub mod grammar;
pub mod grounding;
pub mod head_config;
pub mod html_rows;
pub mod identifier_spelling;
pub mod image_rows;
pub mod json_validate;
pub mod lists;
pub mod media;
pub mod media_cache;
pub mod media_state_prompt;
pub mod orchestrator;
pub mod orientation;
pub mod printed_numbers;
pub mod printed_temporal;
pub mod profile;
pub mod receipt;
pub mod row_omission;
pub mod schema;
pub mod scopes;
pub mod settings;
pub mod source_dates;
pub mod source_rows;
pub mod table_rows;
pub mod typed_decode;
pub mod typed_prompt;
pub mod types;

pub use errors::{ClairicError, Result};
pub use head_config::{ClairicHeadConfig, PredictionViews};
pub use profile::ClairicProfile;
pub use types::{ArGenOutput, ArGenTask, FinishReason};
