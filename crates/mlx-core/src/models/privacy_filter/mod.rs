//! OpenAI Privacy Filter — token-classification PII detector.
pub mod config;
pub mod spans;
pub mod viterbi;
pub use config::{PrivacyFilterConfig, RopeParameters};
pub use spans::{Entity, extract_spans};
pub use viterbi::{Calibration, build_transition_matrix, label_id, viterbi_decode};
