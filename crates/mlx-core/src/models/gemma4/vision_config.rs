/// Vision encoder configuration for Gemma4 multimodal models.
///
/// Parsed from the `vision_config` sub-dict in config.json.
#[napi_derive::napi(object)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Gemma4VisionConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f64,
    pub patch_size: i32,
    pub position_embedding_size: i32,
    pub default_output_length: i32,
    pub pooling_kernel_size: i32,
    pub use_clipped_linears: bool,
    pub rope_theta: f64,
    pub standardize: bool,
}

impl Gemma4VisionConfig {
    /// Parse from the `vision_config` sub-dict of config.json.
    pub fn from_json(vision_cfg: &serde_json::Value) -> Self {
        let get_i32 = |key: &str, default: i32| -> i32 {
            vision_cfg
                .get(key)
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
                .unwrap_or(default)
        };

        let get_f64 = |key: &str, default: f64| -> f64 {
            vision_cfg
                .get(key)
                .and_then(|v| v.as_f64())
                .unwrap_or(default)
        };

        let get_bool = |key: &str, default: bool| -> bool {
            vision_cfg
                .get(key)
                .and_then(|v| v.as_bool())
                .unwrap_or(default)
        };

        // rope_theta lives inside rope_parameters.rope_theta
        let rope_theta = vision_cfg
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(100.0);

        Self {
            hidden_size: get_i32("hidden_size", 768),
            intermediate_size: get_i32("intermediate_size", 3072),
            num_hidden_layers: get_i32("num_hidden_layers", 16),
            num_attention_heads: get_i32("num_attention_heads", 12),
            num_key_value_heads: get_i32("num_key_value_heads", 12),
            head_dim: get_i32("head_dim", 64),
            rms_norm_eps: get_f64("rms_norm_eps", 1e-6),
            patch_size: get_i32("patch_size", 16),
            position_embedding_size: get_i32("position_embedding_size", 10240),
            default_output_length: get_i32("default_output_length", 280),
            pooling_kernel_size: get_i32("pooling_kernel_size", 3),
            use_clipped_linears: get_bool("use_clipped_linears", false),
            rope_theta,
            standardize: get_bool("standardize", false),
        }
    }
}
