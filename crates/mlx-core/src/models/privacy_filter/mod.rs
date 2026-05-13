//! OpenAI Privacy Filter — token-classification PII detector.
pub mod attention;
pub mod classifier;
pub mod config;
pub mod experts;
pub mod forward;
pub mod persistence;
pub mod spans;
pub mod transformer;
pub mod viterbi;
pub mod yarn;
pub use attention::AttentionLayer;
pub use classifier::classifier_forward;
pub use config::{PrivacyFilterConfig, RopeParameters};
pub use experts::GptOssMlp;
pub use forward::PrivacyFilterModel;
pub use persistence::{
    AttnWeights, LayerWeights, LoadedModel, MlpWeights, ModelWeights, load_from_directory,
};
pub use spans::{Entity, extract_spans};
pub use transformer::Block;
pub use viterbi::{Calibration, build_transition_matrix, label_id, viterbi_decode};
pub use yarn::compute_yarn_freqs;
