//! Training state that lives on the model thread.
//!
//! When training is active, each model's Inner struct holds an `Option<ModelThreadTrainingState>`.
//! This stores optimizer state, gradient accumulation buffers, and cached generation results
//! (as MxArrays) that are reused between the generate and train phases of GRPO.

use std::collections::HashMap;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::optimizers::AdamW;

/// Training state owned by the model thread.
///
/// Created when `InitTraining` command is received, destroyed when training ends.
/// All MxArray state lives here — never crosses the thread boundary.
pub(crate) struct ModelThreadTrainingState {
    // === Optimizer ===
    pub optimizer: Option<AdamW>,

    // === Gradient accumulation ===
    pub accumulated_gradients: Option<HashMap<String, MxArray>>,
    pub micro_step: i32,
    pub grad_accumulation_steps: i32,

    // === Step tracking ===
    pub step: i64,

    // === NaN tracking ===
    pub nan_gradient_count: u64,
    pub consecutive_nan_count: u32,

    // === Cached generation results (MxArrays reused by TrainStep) ===
    /// Prompt token arrays cached from GenerateForTraining, consumed by TrainStepGRPO.
    pub cached_prompt_tokens: Option<Vec<MxArray>>,
    /// Completion token arrays cached from GenerateForTraining.
    pub cached_completion_tokens: Option<Vec<MxArray>>,
    /// Completion logprob arrays cached from GenerateForTraining.
    pub cached_completion_logprobs: Option<Vec<MxArray>>,

    // === Config (copied from engine config on init) ===
    pub learning_rate: f64,
    pub gradient_clip_norm: Option<f64>,
    pub gradient_clip_value: Option<f64>,
    pub max_nan_gradients: i64,
    pub emergency_save_threshold: i32,
    pub verbose_nan_detection: bool,
    pub gradient_checkpointing: bool,
}

impl ModelThreadTrainingState {
    /// Create a new training state from engine configuration values.
    pub fn new(
        learning_rate: f64,
        grad_accumulation_steps: i32,
        gradient_clip_norm: Option<f64>,
        gradient_clip_value: Option<f64>,
        max_nan_gradients: i64,
        emergency_save_threshold: i32,
        verbose_nan_detection: bool,
        gradient_checkpointing: bool,
        optimizer: Option<AdamW>,
    ) -> Self {
        Self {
            optimizer,
            accumulated_gradients: None,
            micro_step: 0,
            grad_accumulation_steps,
            step: 0,
            nan_gradient_count: 0,
            consecutive_nan_count: 0,
            cached_prompt_tokens: None,
            cached_completion_tokens: None,
            cached_completion_logprobs: None,
            learning_rate,
            gradient_clip_norm,
            gradient_clip_value,
            max_nan_gradients,
            emergency_save_threshold,
            verbose_nan_detection,
            gradient_checkpointing,
        }
    }

    /// Clear cached generation results (called after training step consumes them).
    pub fn clear_generation_cache(&mut self) {
        self.cached_prompt_tokens = None;
        self.cached_completion_tokens = None;
        self.cached_completion_logprobs = None;
    }

    /// Serialize AdamW moment tensors + step to a SafeTensors file.
    ///
    /// Must run on the model thread — the MxArrays in optimizer state belong
    /// to the thread that created them.
    ///
    /// Format:
    /// - metadata.step: i64 as string
    /// - metadata.format: "adamw_optimizer_state"
    /// - tensor "{param}.m": first moment
    /// - tensor "{param}.v": second moment
    ///
    /// SGD (no optimizer) or empty state → no-op, returns Ok(()).
    pub(crate) fn save_optimizer_state_sync(&self, path: &str) -> Result<()> {
        let Some(opt) = self.optimizer.as_ref() else {
            return Ok(());
        };
        let step = opt.get_step();
        let keys = opt.get_state_keys();
        if keys.is_empty() {
            return Ok(());
        }
        let mut tensors: HashMap<String, MxArray> = HashMap::new();
        for key in &keys {
            if let Some(m) = opt.get_first_moment(key.clone()) {
                tensors.insert(format!("{}.m", key), m);
            }
            if let Some(v) = opt.get_second_moment(key.clone()) {
                tensors.insert(format!("{}.v", key), v);
            }
        }
        let metadata = serde_json::json!({
            "step": step.to_string(),
            "format": "adamw_optimizer_state",
        });
        crate::utils::safetensors::save_safetensors(path, &tensors, Some(metadata))
    }

    /// Restore AdamW moment tensors + step from a SafeTensors file.
    ///
    /// SGD (no optimizer) → no-op.
    ///
    /// Validation invariants (all fail loudly with actionable errors):
    /// 1. Metadata block must be present.
    /// 2. `format` metadata field must equal `"adamw_optimizer_state"`.
    /// 3. `step` metadata field must be present and parseable as i64.
    /// 4. Every tensor key must end in `.m` or `.v`.
    /// 5. Every param name must have both `.m` and `.v` tensors.
    /// 6. No tensor may contain NaN or Inf values.
    pub(crate) fn load_optimizer_state_sync(&mut self, path: &str) -> Result<()> {
        let Some(opt) = self.optimizer.as_mut() else {
            return Ok(());
        };

        let st_file = crate::utils::safetensors::SafeTensorsFile::load(path)?;

        // --- Invariant 1: metadata block required ---
        let metadata = st_file.metadata.as_ref().ok_or_else(|| {
            Error::from_reason(format!(
                "'{path}' is not a recognized AdamW optimizer state file: missing __metadata__ block"
            ))
        })?;

        // --- Invariant 2: `format` field required and must be "adamw_optimizer_state" ---
        let fmt = metadata.get("format").ok_or_else(|| {
            Error::from_reason(format!(
                "'{path}': optimizer state file is missing required metadata field 'format'"
            ))
        })?;
        let fmt = fmt.as_str().ok_or_else(|| {
            Error::from_reason(format!(
                "'{path}': metadata field 'format' must be a string, got {fmt}"
            ))
        })?;
        if fmt != "adamw_optimizer_state" {
            return Err(Error::from_reason(format!(
                "'{path}': expected format=adamw_optimizer_state, got format={fmt}"
            )));
        }

        // --- Invariant 3: `step` field required and parseable as i64 ---
        let step_val = metadata.get("step").ok_or_else(|| {
            Error::from_reason(format!(
                "'{path}': optimizer state file is missing required metadata field 'step'"
            ))
        })?;
        let step_str = step_val.as_str().ok_or_else(|| {
            Error::from_reason(format!(
                "'{path}': metadata field 'step' must be a string, got {step_val}"
            ))
        })?;
        let step = step_str.parse::<i64>().map_err(|_| {
            Error::from_reason(format!(
                "'{path}': metadata field 'step' is not a valid i64 integer: {step_str:?}"
            ))
        })?;

        // Load tensor data now that metadata is validated.
        let tensors = st_file.load_tensors(path)?;

        // --- Invariant 4: every tensor key must end in `.m` or `.v` and have a non-empty param name ---
        for tensor_key in tensors.keys() {
            let stripped = tensor_key
                .strip_suffix(".m")
                .or_else(|| tensor_key.strip_suffix(".v"));
            match stripped {
                None => {
                    return Err(Error::from_reason(format!(
                        "'{path}': unexpected tensor key {tensor_key:?} — all keys must end in '.m' or '.v' (is this a model.safetensors file?)"
                    )));
                }
                Some("") => {
                    return Err(Error::from_reason(format!(
                        "'{path}': tensor key {tensor_key:?} has empty param name"
                    )));
                }
                Some(_) => {}
            }
        }

        // --- Invariant 5: every param must have both `.m` and `.v` ---
        let m_params: std::collections::HashSet<&str> = tensors
            .keys()
            .filter_map(|k| k.strip_suffix(".m"))
            .collect();
        let v_params: std::collections::HashSet<&str> = tensors
            .keys()
            .filter_map(|k| k.strip_suffix(".v"))
            .collect();

        let missing_v: Vec<&str> = m_params.difference(&v_params).copied().collect();
        if !missing_v.is_empty() {
            let mut keys = missing_v;
            keys.sort_unstable();
            return Err(Error::from_reason(format!(
                "'{path}': params have '.m' but no '.v' moment: {}",
                keys.join(", ")
            )));
        }

        let missing_m: Vec<&str> = v_params.difference(&m_params).copied().collect();
        if !missing_m.is_empty() {
            let mut keys = missing_m;
            keys.sort_unstable();
            return Err(Error::from_reason(format!(
                "'{path}': params have '.v' but no '.m' moment: {}",
                keys.join(", ")
            )));
        }

        // --- Invariant 6: no NaN/Inf in moment tensors (checked before mutating optimizer) ---
        for (tensor_key, array) in &tensors {
            if array.has_nan_or_inf()? {
                return Err(Error::from_reason(format!(
                    "'{path}': tensor {tensor_key:?} contains NaN or Inf — checkpoint is corrupt"
                )));
            }
        }

        // All invariants passed — now mutate optimizer state.
        opt.set_step(step);
        for (tensor_key, array) in &tensors {
            if let Some(param_name) = tensor_key.strip_suffix(".m") {
                opt.set_first_moment(param_name.to_string(), array)?;
            } else if let Some(param_name) = tensor_key.strip_suffix(".v") {
                opt.set_second_moment(param_name.to_string(), array)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::AdamW;

    #[test]
    fn new_initializes_counters_to_zero_and_caches_to_none() {
        let state = ModelThreadTrainingState::new(1e-4, 1, None, None, 100, 5, false, true, None);
        assert_eq!(state.step, 0);
        assert_eq!(state.micro_step, 0);
        assert_eq!(state.nan_gradient_count, 0);
        assert_eq!(state.consecutive_nan_count, 0);
        assert!(state.accumulated_gradients.is_none());
        assert!(state.cached_prompt_tokens.is_none());
        assert!(state.cached_completion_tokens.is_none());
        assert!(state.cached_completion_logprobs.is_none());
        assert!(state.optimizer.is_none());
        assert_eq!(state.learning_rate, 1e-4);
        assert_eq!(state.grad_accumulation_steps, 1);
        assert_eq!(state.max_nan_gradients, 100);
        assert_eq!(state.emergency_save_threshold, 5);
        assert!(!state.verbose_nan_detection);
        assert!(state.gradient_checkpointing);
    }

    #[test]
    fn clear_generation_cache_drops_all_three_caches() {
        let mut state =
            ModelThreadTrainingState::new(1e-4, 1, None, None, 100, 5, false, true, None);
        state.cached_prompt_tokens = Some(vec![]);
        state.cached_completion_tokens = Some(vec![]);
        state.cached_completion_logprobs = Some(vec![]);
        state.clear_generation_cache();
        assert!(state.cached_prompt_tokens.is_none());
        assert!(state.cached_completion_tokens.is_none());
        assert!(state.cached_completion_logprobs.is_none());
    }

    #[test]
    fn bump_skipped_step_increments_step_and_clears_cache() {
        // Mirrors the BumpSkippedStep command handler logic.
        let mut state =
            ModelThreadTrainingState::new(1e-4, 1, None, None, 100, 5, false, true, None);
        state.step = 3;
        state.cached_prompt_tokens = Some(vec![]);
        state.cached_completion_tokens = Some(vec![]);
        state.cached_completion_logprobs = Some(vec![]);

        // BumpSkippedStep: clear_generation_cache + step += 1
        state.clear_generation_cache();
        state.step += 1;

        assert_eq!(state.step, 4);
        assert!(state.cached_prompt_tokens.is_none());
        assert!(state.cached_completion_tokens.is_none());
        assert!(state.cached_completion_logprobs.is_none());
    }

    #[test]
    fn set_training_step_overwrites_step() {
        // Mirrors the SetTrainingStep command handler logic.
        let mut state =
            ModelThreadTrainingState::new(1e-4, 1, None, None, 100, 5, false, true, None);
        state.step = 7;
        // SetTrainingStep: ts.step = new_step
        state.step = 42;
        assert_eq!(state.step, 42);
    }

    #[test]
    fn new_accepts_adamw_optimizer() {
        let adamw = AdamW::new(
            Some(1e-4),
            Some(0.9),
            Some(0.999),
            Some(1e-8),
            Some(0.01),
            Some(true),
        );
        let state =
            ModelThreadTrainingState::new(1e-4, 1, None, None, 100, 5, false, true, Some(adamw));
        assert!(state.optimizer.is_some());
        assert_eq!(state.optimizer.as_ref().unwrap().get_step(), 0);
    }

    // =========================================================================
    // Helpers shared by save/load round-trip tests
    // =========================================================================

    fn make_adamw() -> AdamW {
        AdamW::new(
            Some(1e-4),
            Some(0.9),
            Some(0.999),
            Some(1e-8),
            Some(0.01),
            Some(false),
        )
    }

    fn make_training_state_with_adamw() -> ModelThreadTrainingState {
        let adamw = make_adamw();
        ModelThreadTrainingState::new(1e-4, 1, None, None, 100, 5, false, false, Some(adamw))
    }

    /// Create a unique temp path under /tmp for test files.
    fn tmp_path(name: &str) -> String {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        format!("/tmp/mlx_test_{}_{}.safetensors", name, unique)
    }

    /// Write a minimal safetensors file for negative tests.
    ///
    /// `metadata` is the JSON value for `__metadata__` (may be `None`).
    /// `tensors` are written as F32 scalars keyed by name.
    fn write_test_safetensors(
        path: &str,
        metadata: Option<serde_json::Value>,
        tensors: &[(&str, f32)],
    ) {
        let mut tensor_map = std::collections::HashMap::new();
        for (key, val) in tensors {
            let arr = MxArray::from_float32(&[*val], &[1]).unwrap();
            tensor_map.insert(key.to_string(), arr);
        }
        crate::utils::safetensors::save_safetensors(path, &tensor_map, metadata).unwrap();
    }

    // =========================================================================
    // Test 1: Happy path round-trip
    // =========================================================================

    #[test]
    fn save_load_round_trip_restores_step_and_moments() {
        use std::collections::HashSet;

        let path_str = tmp_path("round_trip");
        let path_str = path_str.as_str();

        // Build a state with TWO updated parameters so the key set is
        // non-trivial and we can catch silent re-ordering / dropped keys.
        let mut ts = make_training_state_with_adamw();
        let param_w = MxArray::from_float32(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let grad_w = MxArray::from_float32(&[0.1f32, 0.2, 0.3], &[3]).unwrap();
        let param_b = MxArray::from_float32(&[0.5f32, -0.5], &[2]).unwrap();
        let grad_b = MxArray::from_float32(&[0.4f32, -0.4], &[2]).unwrap();
        let opt = ts.optimizer.as_mut().unwrap();
        let _updated = opt
            .update_batch(
                vec!["weight".to_string(), "bias".to_string()],
                vec![&param_w, &param_b],
                vec![&grad_w, &grad_b],
            )
            .unwrap();
        let saved_step = opt.get_step();
        assert_eq!(saved_step, 1);

        // Capture the exact saved moment values so we can compare post-load.
        let saved_keys: HashSet<String> = opt.get_state_keys().into_iter().collect();
        assert_eq!(
            saved_keys,
            ["weight".to_string(), "bias".to_string()]
                .into_iter()
                .collect::<HashSet<_>>()
        );

        fn moment_values(arr: &MxArray, len: usize) -> Vec<f32> {
            arr.eval();
            (0..len).map(|i| arr.item_at_float32(i).unwrap()).collect()
        }
        let saved_m_w = moment_values(&opt.get_first_moment("weight".into()).unwrap(), 3);
        let saved_v_w = moment_values(&opt.get_second_moment("weight".into()).unwrap(), 3);
        let saved_m_b = moment_values(&opt.get_first_moment("bias".into()).unwrap(), 2);
        let saved_v_b = moment_values(&opt.get_second_moment("bias".into()).unwrap(), 2);

        // Sanity: first moment = (1-β1)*grad = 0.1 * grad, so should be non-zero
        // wherever grad is non-zero. Guards against a regression that silently
        // saves all-zero tensors.
        assert!(saved_m_w.iter().any(|x| x.abs() > 1e-6));
        assert!(saved_v_w.iter().any(|x| x.abs() > 1e-9));

        // Save.
        ts.save_optimizer_state_sync(path_str).unwrap();

        // Load into a fresh state.
        let mut ts2 = make_training_state_with_adamw();
        ts2.load_optimizer_state_sync(path_str).unwrap();

        let opt2 = ts2.optimizer.as_ref().unwrap();

        // (a) step matches exactly.
        assert_eq!(opt2.get_step(), saved_step);

        // (b) exact key-set equality — no extras, no drops.
        let loaded_keys: HashSet<String> = opt2.get_state_keys().into_iter().collect();
        assert_eq!(
            loaded_keys, saved_keys,
            "loaded optimizer key set must match saved set"
        );

        // (c) first moment `m` values match exactly (both params).
        let loaded_m_w = moment_values(&opt2.get_first_moment("weight".into()).unwrap(), 3);
        let loaded_m_b = moment_values(&opt2.get_first_moment("bias".into()).unwrap(), 2);
        assert_eq!(loaded_m_w, saved_m_w, "first moment `m` for weight");
        assert_eq!(loaded_m_b, saved_m_b, "first moment `m` for bias");

        // (d) second moment `v` values match exactly (both params).
        let loaded_v_w = moment_values(&opt2.get_second_moment("weight".into()).unwrap(), 3);
        let loaded_v_b = moment_values(&opt2.get_second_moment("bias".into()).unwrap(), 2);
        assert_eq!(loaded_v_w, saved_v_w, "second moment `v` for weight");
        assert_eq!(loaded_v_b, saved_v_b, "second moment `v` for bias");
    }

    // =========================================================================
    // Test 2: Missing metadata rejected
    // =========================================================================

    #[test]
    fn load_missing_metadata_returns_error() {
        let path_str = tmp_path("missing_meta");
        let path_str = path_str.as_str();

        write_test_safetensors(path_str, None, &[("weight.m", 1.0), ("weight.v", 1.0)]);

        let mut ts = make_training_state_with_adamw();
        let err = ts.load_optimizer_state_sync(path_str).unwrap_err();
        let msg = err.reason.to_lowercase();
        assert!(
            msg.contains("metadata") || msg.contains("not a recognized"),
            "expected 'metadata' in error, got: {msg}"
        );
    }

    // =========================================================================
    // Test 3: Wrong format metadata rejected
    // =========================================================================

    #[test]
    fn load_wrong_format_returns_error() {
        let path_str = tmp_path("wrong_format");
        let path_str = path_str.as_str();

        write_test_safetensors(
            path_str,
            Some(serde_json::json!({"format": "mlx", "step": "5"})),
            &[("weight.m", 1.0), ("weight.v", 1.0)],
        );

        let mut ts = make_training_state_with_adamw();
        let err = ts.load_optimizer_state_sync(path_str).unwrap_err();
        let msg = err.reason.to_lowercase();
        assert!(
            msg.contains("format"),
            "expected 'format' in error, got: {msg}"
        );
    }

    // =========================================================================
    // Test 4: Missing `step` metadata rejected
    // =========================================================================

    #[test]
    fn load_missing_step_returns_error() {
        let path_str = tmp_path("missing_step");
        let path_str = path_str.as_str();

        write_test_safetensors(
            path_str,
            Some(serde_json::json!({"format": "adamw_optimizer_state"})),
            &[("weight.m", 1.0), ("weight.v", 1.0)],
        );

        let mut ts = make_training_state_with_adamw();
        let err = ts.load_optimizer_state_sync(path_str).unwrap_err();
        let msg = err.reason.to_lowercase();
        assert!(msg.contains("step"), "expected 'step' in error, got: {msg}");
    }

    // =========================================================================
    // Test 5: Unparseable `step` metadata rejected
    // =========================================================================

    #[test]
    fn load_unparseable_step_returns_error() {
        let path_str = tmp_path("bad_step");
        let path_str = path_str.as_str();

        write_test_safetensors(
            path_str,
            Some(serde_json::json!({"format": "adamw_optimizer_state", "step": "not-an-integer"})),
            &[("weight.m", 1.0), ("weight.v", 1.0)],
        );

        let mut ts = make_training_state_with_adamw();
        let err = ts.load_optimizer_state_sync(path_str).unwrap_err();
        let msg = err.reason.to_lowercase();
        assert!(
            msg.contains("step") || msg.contains("integer") || msg.contains("valid"),
            "expected step parse error, got: {msg}"
        );
    }

    // =========================================================================
    // Test 6: Orphan tensor (no .m/.v suffix) rejected
    // =========================================================================

    #[test]
    fn load_orphan_tensor_key_returns_error() {
        let path_str = tmp_path("orphan_key");
        let path_str = path_str.as_str();

        write_test_safetensors(
            path_str,
            Some(serde_json::json!({"format": "adamw_optimizer_state", "step": "1"})),
            &[("weight.m", 1.0), ("weight.v", 1.0), ("foo.bar", 1.0)],
        );

        let mut ts = make_training_state_with_adamw();
        let err = ts.load_optimizer_state_sync(path_str).unwrap_err();
        let msg = err.reason.to_lowercase();
        assert!(
            msg.contains("foo.bar")
                || msg.contains("unexpected")
                || msg.contains(".m")
                || msg.contains(".v"),
            "expected orphan key error mentioning 'foo.bar', got: {msg}"
        );
    }

    // =========================================================================
    // Test 7: Unpaired .m without .v rejected
    // =========================================================================

    #[test]
    fn load_unpaired_m_without_v_returns_error() {
        let path_str = tmp_path("unpaired_m");
        let path_str = path_str.as_str();

        write_test_safetensors(
            path_str,
            Some(serde_json::json!({"format": "adamw_optimizer_state", "step": "1"})),
            &[("weight.m", 1.0)], // missing weight.v
        );

        let mut ts = make_training_state_with_adamw();
        let err = ts.load_optimizer_state_sync(path_str).unwrap_err();
        let msg = err.reason.to_lowercase();
        assert!(
            msg.contains("weight")
                || msg.contains(".v")
                || msg.contains("pair")
                || msg.contains("missing"),
            "expected unpaired moment error, got: {msg}"
        );
    }

    // =========================================================================
    // Test 8: NaN in loaded moment rejected
    // =========================================================================

    #[test]
    fn load_nan_moment_returns_error() {
        let path_str = tmp_path("nan_moment");
        let path_str = path_str.as_str();

        write_test_safetensors(
            path_str,
            Some(serde_json::json!({"format": "adamw_optimizer_state", "step": "1"})),
            &[("weight.m", f32::NAN), ("weight.v", 1.0)],
        );

        let mut ts = make_training_state_with_adamw();
        let err = ts.load_optimizer_state_sync(path_str).unwrap_err();
        let msg = err.reason.to_lowercase();
        assert!(
            msg.contains("nan") || msg.contains("inf") || msg.contains("corrupt"),
            "expected NaN/Inf error, got: {msg}"
        );
    }

    // =========================================================================
    // Test 8b: Empty param name (literal ".m" key) rejected
    // =========================================================================

    #[test]
    fn load_empty_param_name_returns_error() {
        // A tensor keyed literally ".m" strips to an empty param name. Even
        // though it has the right suffix, accepting it would create an
        // optimizer state entry with an empty name, which can never match
        // any real parameter.
        let path_str = tmp_path("empty_param");
        let path_str = path_str.as_str();

        write_test_safetensors(
            path_str,
            Some(serde_json::json!({"format": "adamw_optimizer_state", "step": "1"})),
            &[(".m", 1.0), (".v", 1.0)],
        );

        let mut ts = make_training_state_with_adamw();
        let err = ts.load_optimizer_state_sync(path_str).unwrap_err();
        let msg = err.reason.to_lowercase();
        assert!(
            msg.contains("empty") && msg.contains("param"),
            "expected empty-param-name error, got: {msg}"
        );
    }

    // =========================================================================
    // Test 8c: Wrong JSON type for `format` / `step` metadata rejected
    // =========================================================================

    #[test]
    fn load_non_string_format_returns_error() {
        // Metadata field `format` exists but is not a string. The error must
        // distinguish "wrong type" from "missing", or the user wastes time
        // debugging the wrong thing.
        let path_str = tmp_path("non_string_format");
        let path_str = path_str.as_str();

        write_test_safetensors(
            path_str,
            Some(serde_json::json!({"format": 42, "step": "1"})),
            &[("weight.m", 1.0), ("weight.v", 1.0)],
        );

        let mut ts = make_training_state_with_adamw();
        let err = ts.load_optimizer_state_sync(path_str).unwrap_err();
        let msg = err.reason.to_lowercase();
        // Must report wrong-type explicitly, not the old missing-field wording.
        assert!(
            msg.contains("must be a string") && msg.contains("format"),
            "expected wrong-type error for 'format' mentioning 'must be a string', got: {msg}"
        );
        assert!(
            !msg.contains("missing required"),
            "should not report 'missing required' for a present-but-wrong-type field, got: {msg}"
        );
    }

    #[test]
    fn load_non_string_step_returns_error() {
        // Metadata field `step` exists but is a JSON number rather than the
        // string we serialize. Must report wrong-type, not "missing".
        let path_str = tmp_path("non_string_step");
        let path_str = path_str.as_str();

        write_test_safetensors(
            path_str,
            Some(serde_json::json!({"format": "adamw_optimizer_state", "step": 1})),
            &[("weight.m", 1.0), ("weight.v", 1.0)],
        );

        let mut ts = make_training_state_with_adamw();
        let err = ts.load_optimizer_state_sync(path_str).unwrap_err();
        let msg = err.reason.to_lowercase();
        // Must report wrong-type explicitly, not the old missing-field or
        // unparseable-integer wording.
        assert!(
            msg.contains("must be a string") && msg.contains("step"),
            "expected wrong-type error for 'step' mentioning 'must be a string', got: {msg}"
        );
        assert!(
            !msg.contains("missing required"),
            "should not report 'missing required' for a present-but-wrong-type field, got: {msg}"
        );
        assert!(
            !msg.contains("not a valid i64"),
            "should not report 'not a valid i64' before the type check rejects the field, got: {msg}"
        );
    }

    // =========================================================================
    // Test 9: SGD (optimizer=None) is a no-op even if file doesn't exist
    // =========================================================================

    #[test]
    fn load_sgd_no_op_returns_ok() {
        // ModelThreadTrainingState with no optimizer — load_optimizer_state_sync
        // must return Ok(()) without touching the (non-existent) file.
        let mut ts = ModelThreadTrainingState::new(1e-4, 1, None, None, 100, 5, false, false, None);
        let result =
            ts.load_optimizer_state_sync("/nonexistent/path/that/does_not_exist.safetensors");
        assert!(
            result.is_ok(),
            "SGD no-op should return Ok(()), got: {result:?}"
        );
    }
}
