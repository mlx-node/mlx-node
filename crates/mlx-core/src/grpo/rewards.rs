/// GRPO Reward System - Built-in Rust Rewards + JS Callback Support
///
/// This module provides a flexible reward system that can:
/// 1. Use fast, built-in Rust reward functions (no FFI overhead)
/// 2. Call JavaScript reward functions via ThreadsafeFunction (when needed)
/// 3. Combine multiple reward functions with weights
///
/// ## Built-in Rewards
/// - `ToolUseReward`: Validates tool call format and structure
/// - `XMLFormatReward`: Validates XML structure and required tags
/// - `LengthReward`: Scores based on completion length
/// - `JsonSchemaReward`: Validates JSON against expected structure
///
/// ## Usage
/// ```no_run
/// use mlx_core::grpo::{RewardRegistry, ToolUseReward, LengthReward};
///
/// let mut registry = RewardRegistry::new();
/// registry.register_builtin("tool_use", ToolUseReward::new(&["search", "calculate"], true), 1.0);
/// registry.register_builtin("length", LengthReward::new(100, 500, true), 1.0);
///
/// let score = registry.score("prompt", "completion");
/// ```
use std::collections::HashMap;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Trait for reward functions that can score completions
pub trait RewardFunction: Send + Sync {
    /// Score a completion given its prompt
    ///
    /// # Arguments
    /// * `prompt` - The input prompt text
    /// * `completion` - The model's completion text
    ///
    /// # Returns
    /// * Score value (typically in range [-1, 1] or [0, 1])
    fn score(&self, prompt: &str, completion: &str) -> f64;

    /// Get the name of this reward function
    fn name(&self) -> &str;
}

/// Registry for managing multiple reward functions
pub struct RewardRegistry {
    /// Built-in Rust reward functions
    builtin: HashMap<String, Arc<dyn RewardFunction>>,
    /// Weights for each reward function
    weights: HashMap<String, f64>,
    /// Whether to normalize the final score
    normalize: bool,
}

impl Default for RewardRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RewardRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            builtin: HashMap::new(),
            weights: HashMap::new(),
            normalize: true,
        }
    }

    /// Register a built-in reward function
    pub fn register_builtin<R: RewardFunction + 'static>(
        &mut self,
        name: &str,
        reward_fn: R,
        weight: f64,
    ) {
        self.builtin.insert(name.to_string(), Arc::new(reward_fn));
        self.weights.insert(name.to_string(), weight);
    }

    /// Set whether to normalize scores by weight sum
    pub fn set_normalize(&mut self, normalize: bool) {
        self.normalize = normalize;
    }

    /// Score a completion using all registered reward functions
    pub fn score(&self, prompt: &str, completion: &str) -> f64 {
        if self.builtin.is_empty() {
            return 0.0;
        }

        let mut total_score = 0.0;
        let mut total_weight = 0.0;

        for (name, reward_fn) in &self.builtin {
            let weight = self.weights.get(name).copied().unwrap_or(1.0);
            let score = reward_fn.score(prompt, completion);
            total_score += score * weight;
            total_weight += weight;
        }

        if self.normalize && total_weight > 0.0 {
            total_score / total_weight
        } else {
            total_score
        }
    }

    /// Score a batch of completions
    pub fn score_batch(&self, prompts: &[String], completions: &[String]) -> Vec<f64> {
        prompts
            .iter()
            .zip(completions.iter())
            .map(|(p, c)| self.score(p, c))
            .collect()
    }

    /// Check if registry has any reward functions
    pub fn is_empty(&self) -> bool {
        self.builtin.is_empty()
    }

    /// Get registered reward function names
    pub fn names(&self) -> Vec<String> {
        self.builtin.keys().cloned().collect()
    }
}

// =============================================================================
// Built-in Reward Functions
// =============================================================================

/// Tool Use Reward - Validates tool call format
///
/// Checks if completions contain valid tool calls with:
/// - Proper XML structure (`<tool_call>...</tool_call>`)
/// - Valid tool names from allowed list
/// - Required parameters present
pub struct ToolUseReward {
    /// Allowed tool names
    allowed_tools: Vec<String>,
    /// Whether tool call is required (vs optional)
    required: bool,
    /// Penalty for invalid tool calls
    invalid_penalty: f64,
}

impl ToolUseReward {
    /// Create a new tool use reward
    ///
    /// # Arguments
    /// * `allowed_tools` - List of valid tool names
    /// * `required` - If true, missing tool calls get 0 score
    pub fn new(allowed_tools: &[&str], required: bool) -> Self {
        Self {
            allowed_tools: allowed_tools.iter().map(|s| s.to_string()).collect(),
            required,
            invalid_penalty: -0.5,
        }
    }

    /// Check if a tool call is valid
    fn validate_tool_call(&self, tool_content: &str) -> (bool, f64) {
        // Extract tool name from <name>...</name>
        let name_start = tool_content.find("<name>");
        let name_end = tool_content.find("</name>");

        match (name_start, name_end) {
            (Some(start), Some(end)) if end > start + 6 => {
                let tool_name = &tool_content[start + 6..end].trim();

                // Check if tool is in allowed list
                if self.allowed_tools.iter().any(|t| t == tool_name) {
                    // Valid tool, check for arguments
                    let has_args = tool_content.contains("<arguments>")
                        && tool_content.contains("</arguments>");

                    if has_args {
                        (true, 1.0) // Perfect: valid tool with arguments
                    } else {
                        (true, 0.7) // Valid tool but missing arguments
                    }
                } else {
                    (false, self.invalid_penalty) // Unknown tool
                }
            }
            _ => (false, self.invalid_penalty), // Malformed tool call
        }
    }
}

impl RewardFunction for ToolUseReward {
    fn score(&self, _prompt: &str, completion: &str) -> f64 {
        // Find tool call tags
        let tool_start = completion.find("<tool_call>");
        let tool_end = completion.find("</tool_call>");

        match (tool_start, tool_end) {
            (Some(start), Some(end)) if end > start + 11 => {
                let tool_content = &completion[start + 11..end];
                let (_valid, score) = self.validate_tool_call(tool_content);
                score
            }
            (Some(_), Some(_)) => {
                // Tags found but in wrong order
                self.invalid_penalty
            }
            (Some(_), None) | (None, Some(_)) => {
                // Unbalanced tags
                self.invalid_penalty
            }
            (None, None) => {
                // No tool call found
                if self.required {
                    0.0 // Required but missing
                } else {
                    0.5 // Optional, neutral score
                }
            }
        }
    }

    fn name(&self) -> &str {
        "tool_use"
    }
}

/// XML Format Reward - Validates XML structure
///
/// Checks for proper XML formatting with configurable required tags.
pub struct XMLFormatReward {
    /// Required tag names (without angle brackets)
    required_tags: Vec<String>,
    /// Whether to check for balanced tags (reserved for future use)
    _check_balanced: bool,
}

impl XMLFormatReward {
    /// Create a new XML format reward
    ///
    /// # Arguments
    /// * `required_tags` - Tags that must be present (e.g., ["thinking", "answer"])
    pub fn new(required_tags: &[&str]) -> Self {
        Self {
            required_tags: required_tags.iter().map(|s| s.to_string()).collect(),
            _check_balanced: true,
        }
    }

    /// Count balanced tag pairs
    fn count_balanced_tags(&self, text: &str) -> (usize, usize) {
        let mut balanced = 0;
        let mut unbalanced = 0;

        for tag in &self.required_tags {
            let open_tag = format!("<{}>", tag);
            let close_tag = format!("</{}>", tag);

            let open_count = text.matches(&open_tag).count();
            let close_count = text.matches(&close_tag).count();

            if open_count == close_count && open_count > 0 {
                balanced += 1;
            } else if open_count != close_count {
                unbalanced += 1;
            }
        }

        (balanced, unbalanced)
    }
}

impl RewardFunction for XMLFormatReward {
    fn score(&self, _prompt: &str, completion: &str) -> f64 {
        if self.required_tags.is_empty() {
            return 1.0;
        }

        let (balanced, unbalanced) = self.count_balanced_tags(completion);
        let total_required = self.required_tags.len();

        // Penalize unbalanced tags heavily
        let unbalanced_penalty = unbalanced as f64 * 0.3;

        // Score based on balanced required tags
        let balanced_score = balanced as f64 / total_required as f64;

        (balanced_score - unbalanced_penalty).clamp(0.0, 1.0)
    }

    fn name(&self) -> &str {
        "xml_format"
    }
}

/// Length Reward - Scores based on completion length
///
/// Rewards completions within a target length range.
pub struct LengthReward {
    /// Minimum desired length
    min_length: usize,
    /// Maximum desired length
    max_length: usize,
    /// Use character count (true) or word count (false)
    use_chars: bool,
}

impl LengthReward {
    /// Create a new length reward
    ///
    /// # Arguments
    /// * `min_length` - Minimum target length
    /// * `max_length` - Maximum target length
    /// * `use_chars` - Count characters if true, words if false
    pub fn new(min_length: usize, max_length: usize, use_chars: bool) -> Self {
        Self {
            min_length,
            max_length,
            use_chars,
        }
    }
}

impl RewardFunction for LengthReward {
    fn score(&self, _prompt: &str, completion: &str) -> f64 {
        let length = if self.use_chars {
            completion.chars().count()
        } else {
            completion.split_whitespace().count()
        };

        if length < self.min_length {
            // Too short - linear penalty
            length as f64 / self.min_length as f64
        } else if length > self.max_length {
            // Too long - gradual penalty
            let overage = length - self.max_length;
            let penalty = (overage as f64 / self.max_length as f64).min(1.0);
            1.0 - penalty * 0.5
        } else {
            // In range - full score
            1.0
        }
    }

    fn name(&self) -> &str {
        "length"
    }
}

/// JSON Schema Reward - Validates JSON structure
///
/// Checks if completion contains valid JSON with expected fields.
pub struct JsonSchemaReward {
    /// Required top-level fields
    required_fields: Vec<String>,
    /// Whether JSON must be parseable (reserved for future use)
    _must_parse: bool,
}

impl JsonSchemaReward {
    /// Create a new JSON schema reward
    ///
    /// # Arguments
    /// * `required_fields` - Fields that must be present in JSON
    pub fn new(required_fields: &[&str]) -> Self {
        Self {
            required_fields: required_fields.iter().map(|s| s.to_string()).collect(),
            _must_parse: true,
        }
    }

    /// Extract JSON from text (finds first {...} or [...])
    fn extract_json<'a>(&self, text: &'a str) -> Option<&'a str> {
        // Find first { or [
        let obj_start = text.find('{');
        let arr_start = text.find('[');

        let (start, end_char) = match (obj_start, arr_start) {
            (Some(o), Some(a)) if o < a => (o, '}'),
            (Some(o), Some(_)) => (o, '}'),
            (Some(o), None) => (o, '}'),
            (None, Some(a)) => (a, ']'),
            (None, None) => return None,
        };

        // Find matching end bracket (simple approach - finds last occurrence)
        let remaining = &text[start..];
        let end = remaining.rfind(end_char)?;

        Some(&remaining[..=end])
    }
}

impl RewardFunction for JsonSchemaReward {
    fn score(&self, _prompt: &str, completion: &str) -> f64 {
        // Try to extract JSON
        let json_str = match self.extract_json(completion) {
            Some(s) => s,
            None => return 0.0, // No JSON found
        };

        // Try to parse (basic validation)
        // Note: We're not pulling in serde_json here for simplicity
        // Just check for balanced braces and required fields as strings
        let open_braces = json_str.matches('{').count();
        let close_braces = json_str.matches('}').count();

        if open_braces != close_braces {
            return 0.2; // Unbalanced braces
        }

        // Check for required fields (simple string search)
        let mut found_fields = 0;
        for field in &self.required_fields {
            let field_pattern = format!("\"{}\"", field);
            if json_str.contains(&field_pattern) {
                found_fields += 1;
            }
        }

        if self.required_fields.is_empty() {
            1.0 // No required fields, just valid-ish JSON
        } else {
            found_fields as f64 / self.required_fields.len() as f64
        }
    }

    fn name(&self) -> &str {
        "json_schema"
    }
}

// =============================================================================
// NAPI Exports for JavaScript integration
// =============================================================================

/// Built-in reward function types
#[napi(string_enum)]
#[derive(Clone, Debug)]
pub enum BuiltinRewardType {
    /// Tool use validation
    ToolUse,
    /// XML format validation
    XmlFormat,
    /// Length-based scoring
    Length,
    /// JSON schema validation
    JsonSchema,
}

/// Configuration for built-in rewards
#[napi(object)]
#[derive(Clone)]
pub struct BuiltinRewardConfig {
    /// Type of reward function
    pub reward_type: BuiltinRewardType,
    /// Weight for this reward (default 1.0)
    pub weight: Option<f64>,
    /// Allowed tool names (for ToolUse)
    pub allowed_tools: Option<Vec<String>>,
    /// Required tags (for XmlFormat)
    pub required_tags: Option<Vec<String>>,
    /// Minimum length (for Length)
    pub min_length: Option<u32>,
    /// Maximum length (for Length)
    pub max_length: Option<u32>,
    /// Use character count vs word count (for Length)
    pub use_chars: Option<bool>,
    /// Required JSON fields (for JsonSchema)
    pub required_fields: Option<Vec<String>>,
    /// Whether tool call is required (for ToolUse)
    pub required: Option<bool>,
}

/// NAPI-exported reward registry wrapper
#[napi]
pub struct NativeRewardRegistry {
    inner: RewardRegistry,
}

#[napi]
impl NativeRewardRegistry {
    /// Create a new reward registry
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: RewardRegistry::new(),
        }
    }

    /// Register a built-in reward function
    #[napi]
    pub fn register(&mut self, config: BuiltinRewardConfig) -> Result<()> {
        let weight = config.weight.unwrap_or(1.0);

        match config.reward_type {
            BuiltinRewardType::ToolUse => {
                let tools: Vec<&str> = config
                    .allowed_tools
                    .as_ref()
                    .map(|v| v.iter().map(|s| s.as_str()).collect())
                    .unwrap_or_else(|| vec!["search", "calculate", "code"]);
                let required = config.required.unwrap_or(true);

                self.inner.register_builtin(
                    "tool_use",
                    ToolUseReward::new(&tools, required),
                    weight,
                );
            }
            BuiltinRewardType::XmlFormat => {
                let tags: Vec<&str> = config
                    .required_tags
                    .as_ref()
                    .map(|v| v.iter().map(|s| s.as_str()).collect())
                    .unwrap_or_else(|| vec!["thinking", "answer"]);

                self.inner
                    .register_builtin("xml_format", XMLFormatReward::new(&tags), weight);
            }
            BuiltinRewardType::Length => {
                let min = config.min_length.unwrap_or(50) as usize;
                let max = config.max_length.unwrap_or(500) as usize;
                let use_chars = config.use_chars.unwrap_or(true);

                self.inner.register_builtin(
                    "length",
                    LengthReward::new(min, max, use_chars),
                    weight,
                );
            }
            BuiltinRewardType::JsonSchema => {
                let fields: Vec<&str> = config
                    .required_fields
                    .as_ref()
                    .map(|v| v.iter().map(|s| s.as_str()).collect())
                    .unwrap_or_default();

                self.inner
                    .register_builtin("json_schema", JsonSchemaReward::new(&fields), weight);
            }
        }

        Ok(())
    }

    /// Score a single completion
    #[napi]
    pub fn score(&self, prompt: String, completion: String) -> f64 {
        self.inner.score(&prompt, &completion)
    }

    /// Score a batch of completions
    #[napi]
    pub fn score_batch(&self, prompts: Vec<String>, completions: Vec<String>) -> Vec<f64> {
        self.inner.score_batch(&prompts, &completions)
    }

    /// Check if registry is empty
    #[napi(getter)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get registered reward names
    #[napi(getter)]
    pub fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    /// Set whether to normalize scores
    #[napi]
    pub fn set_normalize(&mut self, normalize: bool) {
        self.inner.set_normalize(normalize);
    }
}

impl Default for NativeRewardRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_use_reward_valid() {
        let reward = ToolUseReward::new(&["search", "calculate"], true);

        let completion =
            r#"<tool_call><name>search</name><arguments>{"query": "test"}</arguments></tool_call>"#;
        let score = reward.score("prompt", completion);
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_tool_use_reward_missing() {
        let reward = ToolUseReward::new(&["search"], true);

        let completion = "No tool call here";
        let score = reward.score("prompt", completion);
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_tool_use_reward_invalid_tool() {
        let reward = ToolUseReward::new(&["search"], true);

        let completion = r#"<tool_call><name>unknown</name></tool_call>"#;
        let score = reward.score("prompt", completion);
        assert!(score < 0.0); // Penalty
    }

    #[test]
    fn test_xml_format_reward() {
        let reward = XMLFormatReward::new(&["thinking", "answer"]);

        let good = "<thinking>Let me think</thinking><answer>42</answer>";
        let score = reward.score("prompt", good);
        assert!((score - 1.0).abs() < 0.001);

        let partial = "<thinking>Let me think</thinking>";
        let score = reward.score("prompt", partial);
        assert!((score - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_length_reward() {
        let reward = LengthReward::new(10, 50, true);

        // Perfect length
        let good = "This is about twenty five characters";
        let score = reward.score("prompt", good);
        assert!((score - 1.0).abs() < 0.001);

        // Too short
        let short = "Hi";
        let score = reward.score("prompt", short);
        assert!(score < 0.5);

        // Too long - 60 chars with max=50 gives penalty of 0.2, score ~0.9
        let long = "a".repeat(60);
        let score = reward.score("prompt", &long);
        assert!(
            (0.5..1.0).contains(&score),
            "Too long score {} should be in [0.5, 1.0)",
            score
        );
    }

    #[test]
    fn test_json_schema_reward() {
        let reward = JsonSchemaReward::new(&["name", "value"]);

        let valid = r#"{"name": "test", "value": 42}"#;
        let score = reward.score("prompt", valid);
        assert!((score - 1.0).abs() < 0.001);

        let partial = r#"{"name": "test"}"#;
        let score = reward.score("prompt", partial);
        assert!((score - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_reward_registry() {
        let mut registry = RewardRegistry::new();
        registry.register_builtin("length", LengthReward::new(10, 100, true), 1.0);
        registry.register_builtin("xml", XMLFormatReward::new(&["answer"]), 1.0);

        let completion = "<answer>This is a good answer with proper length</answer>";
        let score = registry.score("prompt", completion);
        assert!(score > 0.8);
    }

    #[test]
    fn test_combined_rewards_with_weights() {
        let mut registry = RewardRegistry::new();

        // Register tool use with weight 0.6
        registry.register_builtin("tool_use", ToolUseReward::new(&["search"], true), 0.6);

        // Register XML format with weight 0.4
        registry.register_builtin("xml", XMLFormatReward::new(&["result"]), 0.4);

        // Completion with both valid tool call and XML
        let completion = r#"<tool_call><name>search</name><arguments>{}</arguments></tool_call><result>Found it!</result>"#;
        let score = registry.score("prompt", completion);

        // Should be high score (both rewards satisfied)
        assert!(score > 0.8, "Score {} should be > 0.8", score);
    }

    #[test]
    fn test_batch_scoring() {
        let mut registry = RewardRegistry::new();
        registry.register_builtin("length", LengthReward::new(5, 50, true), 1.0);

        let prompts: Vec<String> = vec!["p1".to_string(), "p2".to_string(), "p3".to_string()];
        let completions: Vec<String> = vec![
            "good length text".to_string(),
            "ab".to_string(),
            "also good completion here".to_string(),
        ];
        let scores = registry.score_batch(&prompts, &completions);

        assert_eq!(scores.len(), 3);
        assert!(
            (scores[0] - 1.0).abs() < 0.001,
            "Score[0] {} should be ~1.0",
            scores[0]
        ); // good length
        assert!(
            scores[1] < 1.0,
            "Score[1] {} should be < 1.0 (too short)",
            scores[1]
        ); // too short
        assert!(
            (scores[2] - 1.0).abs() < 0.001,
            "Score[2] {} should be ~1.0",
            scores[2]
        ); // good length
    }

    #[test]
    fn test_partial_weight_combination() {
        let mut registry = RewardRegistry::new();

        // Only register one reward with weight 1.0
        registry.register_builtin("xml", XMLFormatReward::new(&["thinking", "answer"]), 1.0);

        // Partial match (only thinking tag)
        let partial = "<thinking>Let me think</thinking>";
        let score = registry.score("prompt", partial);
        assert!(
            (score - 0.5).abs() < 0.001,
            "Partial XML score {} should be ~0.5",
            score
        );
    }

    #[test]
    fn test_registry_names() {
        let mut registry = RewardRegistry::new();
        registry.register_builtin("length", LengthReward::new(10, 100, true), 1.0);
        registry.register_builtin("xml", XMLFormatReward::new(&["answer"]), 1.0);

        let names = registry.names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"length".to_string()));
        assert!(names.contains(&"xml".to_string()));
    }

    #[test]
    fn test_empty_registry() {
        let registry = RewardRegistry::new();
        assert!(registry.is_empty());

        // Scoring with empty registry should return 0
        let score = registry.score("prompt", "completion");
        assert!((score - 0.0).abs() < 0.001);
    }
}
