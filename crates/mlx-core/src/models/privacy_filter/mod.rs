//! OpenAI Privacy Filter — token-classification PII detector.
pub mod config;
pub mod persistence;
pub mod spans;
pub mod viterbi;
pub use config::{PrivacyFilterConfig, RopeParameters};
pub use persistence::{
    AttnWeights, LayerWeights, LoadedModel, MlpWeights, ModelWeights, load_from_directory,
};
pub use spans::{Entity, extract_spans};
pub use viterbi::{Calibration, build_transition_matrix, label_id, viterbi_decode};
