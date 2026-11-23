//! Pure-Rust configuration, layout constants, and index views for `ClairicHead`.
//!
//! Shared between the MLX typed head (`mlx-core`) and the pure-Rust prompt
//! encoding, profile selection, and decoding pipeline (`mlx-clairic`).

use std::path::Path;

use serde::Deserialize;

use crate::errors::{ClairicError, Result};

pub const MAX_SPAN: i64 = 64;
pub const TRIM_CLASSES: i64 = 8;
pub const BINS: i64 = 256;
pub const POINTER_WIDTH: i64 = 256;
pub const NUMBER_DIGITS: i64 = 12;
pub const MIN_EXPONENT: i64 = -12;
pub const MAX_EXPONENT: i64 = 15;
pub const EXPONENT_CLASSES: i64 = MAX_EXPONENT - MIN_EXPONENT + 2; // 29
pub const YEAR_DIGITS: i64 = 4;
pub const YEAR_LOW: i32 = 1800;
pub const YEAR_HIGH: i32 = 2199;
pub const YEAR_OUTSIDE: i64 = (YEAR_HIGH - YEAR_LOW + 1) as i64; // 400
pub const YEAR_CLASSES: i64 = YEAR_OUTSIDE + 1; // 401
pub const LEXICAL_DIGITS: i64 = 4;
pub const ESTIMATE_SPREAD_LIMIT: f64 = 2.0;
pub const NUMBER_WIDTH: i64 = 2 + EXPONENT_CLASSES + NUMBER_DIGITS * 10; // 151
pub const GROUP_DIGITS: i64 = 3;
pub const GROUP_CLASSES: i64 = 1000;
pub const GROUP_LOW: i64 = -4; // MIN_EXPONENT / GROUP_DIGITS
pub const GROUP_HIGH: i64 = 5; // MAX_EXPONENT / GROUP_DIGITS
pub const NUMBER_GROUPS: i64 = GROUP_HIGH - GROUP_LOW + 1; // 10
pub const GROUP_RANK: i64 = 256;
pub const DATE_WIDTH: i64 = YEAR_DIGITS * 10 + YEAR_CLASSES + 12 + 31; // 484
pub const TIME_WIDTH: i64 = 24 + 60 + 60; // 144

pub const DIGIT_SLOTS: i64 = NUMBER_DIGITS + YEAR_DIGITS + 2 + 2 + 2 + 2; // 24
pub const YEAR_SLOT: i64 = NUMBER_DIGITS; // 12
pub const DAY_SLOT: i64 = NUMBER_DIGITS + YEAR_DIGITS; // 16
pub const HOUR_SLOT: i64 = NUMBER_DIGITS + YEAR_DIGITS + 2; // 18
pub const MINUTE_SLOT: i64 = NUMBER_DIGITS + YEAR_DIGITS + 4; // 20
pub const SECOND_SLOT: i64 = NUMBER_DIGITS + YEAR_DIGITS + 6; // 22

pub const ROUTE_KINDS: [&str; 8] = [
    "text",
    "tags",
    "number",
    "date",
    "time",
    "estimate",
    "number_list",
    "date_list",
];
pub const LIST_KINDS: [&str; 2] = ["number_list", "date_list"];
pub const ROUTE_KIND_COUNT: i64 = 8;
pub const LIST_KIND_COUNT: i64 = 2;
pub const KIND_EMBEDDING_ROWS: i64 = ROUTE_KIND_COUNT + LIST_KIND_COUNT; // 10

/// Configuration deserialized from `clair_head_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ClairicHeadConfig {
    pub hidden_size: i64,
    pub bins: i64,
    pub pointer_width: i64,
    pub readout: String,
    pub route_width: i64,
    pub route_layers: usize,
    pub route_heads: i64,
    pub slot_rank: i64,
    #[serde(default)]
    pub estimate_lexical: bool,
    #[serde(default)]
    pub span_trim: bool,
    #[serde(default)]
    pub loop_feedback: bool,
    #[serde(default)]
    pub answer_slots: bool,
    #[serde(default)]
    pub answer_slot_routing: bool,
    #[serde(default)]
    pub number_groups: bool,
    #[serde(default)]
    pub number_pointer: bool,
}

impl ClairicHeadConfig {
    /// Load `clair_head_config.json` from a checkpoint directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(dir.join("clair_head_config.json")).map_err(|e| {
            ClairicError::from_reason(format!("Failed to read clair_head_config.json: {e}"))
        })?;
        serde_json::from_str(&raw)
            .map_err(|e| ClairicError::from_reason(format!("Invalid clair_head_config.json: {e}")))
    }

    /// Validate that the configuration matches the supported lexical typed head contract
    /// (`mlx_engine.py::TypedHead.REQUIRED`) and expected backbone hidden size.
    pub fn validate(&self, expected_hidden_size: i64) -> Result<()> {
        if self.readout != "lexical" {
            return Err(ClairicError::from_reason(
                "unsupported typed head: readout must be 'lexical'",
            ));
        }
        for (name, flag) in [
            ("estimate_lexical", self.estimate_lexical),
            ("span_trim", self.span_trim),
            ("answer_slots", self.answer_slots),
            ("answer_slot_routing", self.answer_slot_routing),
            ("number_groups", self.number_groups),
            ("number_pointer", self.number_pointer),
        ] {
            if !flag {
                return Err(ClairicError::from_reason(format!(
                    "unsupported typed head: {name} must be true"
                )));
            }
        }
        if self.loop_feedback {
            return Err(ClairicError::from_reason(
                "unsupported typed head: loop_feedback is not supported",
            ));
        }
        if self.hidden_size != expected_hidden_size
            || self.hidden_size <= 0
            || self.bins <= 0
            || self.pointer_width <= 0
            || self.route_width <= 0
            || self.route_heads <= 0
            || self.route_width % self.route_heads != 0
            || self.route_layers == 0
            || self.route_layers > 32
            || self.slot_rank <= 0
        {
            return Err(ClairicError::from_reason(
                "Invalid Clairic head configuration or backbone hidden size",
            ));
        }
        Ok(())
    }
}

/// Index views describing which query positions and list-item slots belong to each readout group.
#[derive(Debug, Clone, Default)]
pub struct PredictionViews {
    /// Token positions in `hidden` for each field query (`[Q]`).
    pub query_positions: Vec<i64>,
    /// Row indices into `query_positions` for `text` span fields.
    pub text_rows: Vec<i64>,
    /// Row indices into `query_positions` for `tags` (`text_list`) BIO fields.
    pub tags_rows: Vec<i64>,
    /// Row indices into `query_positions` for `number` fields.
    pub number_rows: Vec<i64>,
    /// Row indices into `query_positions` for `date` fields.
    pub date_rows: Vec<i64>,
    /// Row indices into `query_positions` for `time` fields.
    pub time_rows: Vec<i64>,
    /// Row indices into `query_positions` for `estimate` fields.
    pub estimate_rows: Vec<i64>,
    /// Row indices into `query_positions` for `number_list` fields.
    pub number_list_rows: Vec<i64>,
    /// Row indices into `query_positions` for `date_list` fields.
    pub date_list_rows: Vec<i64>,
    /// Token positions in `hidden` for `number_list` item cues (`[number_list_rows.len(), list_slots]`).
    pub number_list_items: Vec<Vec<i64>>,
    /// Token positions in `hidden` for `date_list` item cues (`[date_list_rows.len(), list_slots]`).
    pub date_list_items: Vec<Vec<i64>>,
}
