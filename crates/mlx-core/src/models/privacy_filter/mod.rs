//! OpenAI Privacy Filter — token-classification PII detector.
pub mod attention;
pub mod config;
pub mod persistence;
pub mod spans;
pub mod viterbi;
pub mod yarn;
pub use attention::AttentionLayer;
pub use config::{PrivacyFilterConfig, RopeParameters};
pub use persistence::{
    AttnWeights, LayerWeights, LoadedModel, MlpWeights, ModelWeights, load_from_directory,
};
pub use spans::{Entity, extract_spans};
pub use viterbi::{Calibration, build_transition_matrix, label_id, viterbi_decode};
pub use yarn::compute_yarn_freqs;
