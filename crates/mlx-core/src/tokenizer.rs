//! # Tokenizer Module
//!
//! Provides fast, production-ready tokenization for Qwen3 models with:
//! - BPE encoding/decoding
//! - Special token handling (EOS, BOS, PAD, etc.)
//! - ChatML format support
//! - Batch processing
//! - Tool/function calling support with Jinja2 template rendering
//!
//! ## Security Model
//!
//! The tokenizer loads configuration files (`tokenizer.json`, `tokenizer_config.json`)
//! from the model directory. **These files are assumed to be trusted.**
//!
//! Specifically:
//! - `tokenizer.json` - Defines vocabulary and tokenization rules
//! - `tokenizer_config.json` - May contain Jinja2 chat templates
//!
//! ### Warning: Untrusted Sources
//!
//! Loading tokenizer files from untrusted sources could pose security risks:
//!
//! - **Malicious templates**: While minijinja sandboxes execution (no file access,
//!   no arbitrary code execution), a malicious template could cause denial of service
//!   through excessive loops or memory allocation.
//!
//! - **Data extraction**: A malicious template could potentially extract sensitive
//!   data from the context (messages, tool definitions) in unexpected ways.
//!
//! - **Vocabulary manipulation**: Malicious vocabulary could affect model behavior
//!   in unexpected ways or enable prompt injection attacks.
//!
//! ### Recommended Sources
//!
//! Always use tokenizer files from trusted sources:
//! - Official Hugging Face Hub repositories
//! - Your own trained/fine-tuned models
//! - Verified model providers
//!
//! **Do NOT load tokenizer files from:**
//! - Random internet downloads
//! - User-uploaded files without verification
//! - Untrusted third-party sources
use minijinja::{Environment, context};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokenizers::{EncodeInput, Encoding, Tokenizer};
use tracing::warn;

/// Special token IDs for Qwen3 models
const ENDOFTEXT_TOKEN_ID: u32 = 151643;
const IM_END_TOKEN_ID: u32 = 151645;

/// Valid roles for ChatML format (prevents role injection attacks)
const VALID_CHATML_ROLES: &[&str] = &["system", "user", "assistant", "tool", "developer"];

/// Tool call made by an assistant
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Optional unique identifier for the tool call
    pub id: Option<String>,
    /// Name of the tool/function to call
    pub name: String,
    /// JSON string of arguments to pass to the tool
    pub arguments: String,
}

/// Function parameters schema (JSON Schema subset)
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionParameters {
    /// Type (usually "object")
    #[serde(rename = "type")]
    pub r#type: String,
    /// JSON string of property definitions
    pub properties: Option<String>,
    /// List of required parameter names
    pub required: Option<Vec<String>>,
}

/// Function definition for tool calling
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Name of the function
    pub name: String,
    /// Description of what the function does
    pub description: Option<String>,
    /// Parameter schema
    pub parameters: Option<FunctionParameters>,
}

/// OpenAI-compatible tool definition
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool type (currently only "function" is supported)
    #[serde(rename = "type")]
    pub r#type: String,
    /// Function definition
    pub function: FunctionDefinition,
}

/// Chat message with tool calling support
#[napi(object)]
#[derive(Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role: "system", "user", "assistant", or "tool"
    #[napi(ts_type = "'system' | 'user' | 'assistant' | 'tool' | (string & {})")]
    pub role: String,
    /// Message content
    pub content: String,
    /// Tool calls made by the assistant (for assistant messages)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID this message is responding to (for tool messages)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Reasoning content for thinking mode (used with <think> tags)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Image data for VLM models (encoded image bytes: PNG/JPEG, passed as Uint8Array/Buffer)
    #[napi(ts_type = "Array<Uint8Array> | undefined")]
    #[serde(skip)]
    pub images: Option<Vec<Uint8Array>>,
}

impl std::fmt::Debug for ChatMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatMessage")
            .field("role", &self.role)
            .field("content", &self.content)
            .field("tool_calls", &self.tool_calls)
            .field("tool_call_id", &self.tool_call_id)
            .field("reasoning_content", &self.reasoning_content)
            .field(
                "images",
                &self
                    .images
                    .as_ref()
                    .map(|imgs| imgs.iter().map(|i| i.len()).collect::<Vec<_>>()),
            )
            .finish()
    }
}

/// Qwen3 Tokenizer class with NAPI bindings
#[napi]
pub struct Qwen3Tokenizer {
    tokenizer: Arc<Tokenizer>,
    pad_token_id: u32,
    eos_token_id: u32,
    bos_token_id: Option<u32>,
    /// Jinja2 chat template loaded from tokenizer_config.json
    chat_template: Option<String>,
    /// Token ID for `</think>` or `</longcat_think>` (None if not in vocabulary).
    think_end_id: Option<u32>,
    /// The actual think-end string (e.g., `"</think>"` or `"</longcat_think>"`).
    think_end_str: Option<String>,
}

#[napi]
impl Qwen3Tokenizer {
    /// Load tokenizer from tokenizer.json file
    ///
    /// # Arguments
    /// * `path` - Path to tokenizer.json file (default: "../.cache/assets/tokenizers/qwen3_tokenizer.json")
    ///
    /// # Example
    /// ```typescript
    /// const tokenizer = Qwen3Tokenizer.fromPretrained();
    /// const tokens = tokenizer.encode("Hello, world!");
    /// ```
    #[napi]
    pub fn from_pretrained(
        env: &Env,
        tokenizer_path: String,
    ) -> Result<PromiseRaw<'_, Qwen3Tokenizer>> {
        env.spawn_future(async move {
            napi::bindgen_prelude::spawn_blocking(move || {
                let tokenizer = Tokenizer::from_file(&tokenizer_path)
                    .map_err(|e| Error::from_reason(format!("Failed to load tokenizer: {}", e)))?;

                // Load chat template from tokenizer_config.json (in same directory)
                let chat_template = Self::load_chat_template(&tokenizer_path);

                let (think_end_id, think_end_str) = Self::detect_think_end(&tokenizer);

                // Read special token IDs from tokenizer_config.json if available.
                // Falls back to Qwen defaults (pad=151643, eos=151645, bos=None)
                // for backward compatibility with Qwen3/3.5 models.
                let tokenizer_path_ref = Path::new(&tokenizer_path);
                let (pad_token_id, eos_token_id, bos_token_id) =
                    Self::resolve_special_tokens(&tokenizer, tokenizer_path_ref);

                Ok(Self {
                    tokenizer: Arc::new(tokenizer),
                    pad_token_id,
                    eos_token_id,
                    bos_token_id,
                    chat_template,
                    think_end_id,
                    think_end_str,
                })
            })
            .await
            .map_err(|join_err| {
                Error::new(
                    Status::GenericFailure,
                    format!("Failed to load tokenizer: {join_err}"),
                )
            })?
        })
    }

    /// Load chat template from tokenizer_config.json file.
    ///
    /// # Security Considerations
    ///
    /// The chat template loaded from `tokenizer_config.json` is a Jinja2 template
    /// that will be rendered with user-provided message content. This function
    /// assumes that the `tokenizer_config.json` file comes from a **trusted source**
    /// (e.g., Hugging Face Hub, local model files you control).
    ///
    /// While minijinja provides sandboxed execution (no file system access, no
    /// arbitrary code execution), loading templates from untrusted sources could:
    /// - Cause denial of service through excessive template loops
    /// - Extract/expose data from the template context unexpectedly
    ///
    /// **Do NOT use tokenizer files from untrusted sources.**
    ///
    /// # Arguments
    /// * `tokenizer_path` - Path to the tokenizer.json file. The function looks for
    ///   `tokenizer_config.json` in the same directory.
    ///
    /// # Returns
    /// The chat template string if found and valid, `None` otherwise.
    fn load_chat_template(tokenizer_path: &str) -> Option<String> {
        let path = Path::new(tokenizer_path);
        let dir = path.parent()?;

        // First: try tokenizer_config.json (embedded template)
        let config_path = dir.join("tokenizer_config.json");
        if config_path.exists()
            && let Ok(config_content) = std::fs::read_to_string(&config_path)
            && let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_content)
            && let Some(template) = config.get("chat_template").and_then(|v| v.as_str())
        {
            // Basic template safety validation
            if let Err(warning) = Self::validate_template_safety(template) {
                // Log warning but don't fail - the template may still work
                #[cfg(debug_assertions)]
                eprintln!("Warning: {}", warning);
                let _ = warning; // Suppress unused warning in release builds
            }
            return Some(Self::patch_preserve_thinking(template));
        }

        // Second: try standalone chat_template.jinja file (used by Gemma4 HF snapshots)
        let jinja_path = dir.join("chat_template.jinja");
        if jinja_path.exists()
            && let Ok(template) = std::fs::read_to_string(&jinja_path)
        {
            if let Err(warning) = Self::validate_template_safety(&template) {
                #[cfg(debug_assertions)]
                eprintln!("Warning: {}", warning);
                let _ = warning;
            }
            return Some(Self::patch_preserve_thinking(&template));
        }

        None
    }

    /// Rewrite the Qwen3.5 chat-template reasoning gate so our `preserve_thinking=true`
    /// context variable takes effect.
    ///
    /// The stock Qwen3.5 template gates `<think>…</think>` rendering on prior assistant
    /// turns with `{%- if loop.index0 > ns.last_query_index %}`. `ns.last_query_index`
    /// jumps forward when a fresh top-level user message arrives, so the moment a user
    /// appends a follow-up, every prior assistant turn flips into the else branch and
    /// silently drops its `<think>` block on re-render. That breaks the token-level
    /// prefix equality that `verify_cache_prefix_direct` needs for tier-2 warm-session
    /// reuse — a 19-turn agent session observed a 151s / 180s cold re-prefill on each
    /// such boundary (see `.logging/requests.ndjson`, turns 11 and 16).
    ///
    /// We already pass `preserve_thinking=true` into the Jinja context
    /// (`render_chat_template_jinja2`), but the shipped template never reads the
    /// variable. Rather than fork the template per model, patch the gate at load time
    /// so `preserve_thinking` wins regardless of `last_query_index`:
    ///
    ///   `loop.index0 > ns.last_query_index`
    ///     → `preserve_thinking or loop.index0 > ns.last_query_index`
    ///
    /// Idempotent: if the template already references `preserve_thinking` (future
    /// upstream templates, or our own re-patched string) the replacement becomes a
    /// no-op. For templates that don't contain the Qwen3.5 gate at all (e.g. Gemma4,
    /// LFM2, legacy ChatML) this is a silent pass-through.
    fn patch_preserve_thinking(template: &str) -> String {
        if template.contains("preserve_thinking") {
            return template.to_string();
        }
        template.replace(
            "loop.index0 > ns.last_query_index",
            "preserve_thinking or loop.index0 > ns.last_query_index",
        )
    }

    /// Load tokenizer from file synchronously (for internal use)
    ///
    /// # Arguments
    /// * `tokenizer_path` - Path to the tokenizer.json file
    ///
    /// # Returns
    /// Qwen3Tokenizer instance or error
    pub fn from_file(tokenizer_path: &Path) -> std::result::Result<Self, String> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        // Load chat template from tokenizer_config.json (in same directory)
        let chat_template = Self::load_chat_template(tokenizer_path.to_string_lossy().as_ref());

        let (think_end_id, think_end_str) = Self::detect_think_end(&tokenizer);

        // Read special token IDs from tokenizer_config.json if available.
        // Falls back to Qwen defaults (pad=151643, eos=151645, bos=None)
        // for backward compatibility with Qwen3/3.5 models.
        let (pad_token_id, eos_token_id, bos_token_id) =
            Self::resolve_special_tokens(&tokenizer, tokenizer_path);

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            pad_token_id,
            eos_token_id,
            bos_token_id,
            chat_template,
            think_end_id,
            think_end_str,
        })
    }

    /// Resolve special token IDs from tokenizer_config.json.
    /// Returns (pad_token_id, eos_token_id, bos_token_id).
    fn resolve_special_tokens(
        tokenizer: &Tokenizer,
        tokenizer_path: &Path,
    ) -> (u32, u32, Option<u32>) {
        let config_path = tokenizer_path
            .parent()
            .map(|p| p.join("tokenizer_config.json"));

        let config: Option<serde_json::Value> = config_path
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str(&s).ok());

        let resolve = |key: &str| -> Option<u32> {
            config
                .as_ref()
                .and_then(|c| c.get(key))
                .and_then(|v| {
                    v.as_str()
                        .or_else(|| v.get("content").and_then(|c| c.as_str()))
                })
                .and_then(|token_str| tokenizer.token_to_id(token_str))
        };

        let pad = resolve("pad_token").unwrap_or(ENDOFTEXT_TOKEN_ID);
        let eos = resolve("eos_token").unwrap_or(IM_END_TOKEN_ID);
        let bos = resolve("bos_token");

        (pad, eos, bos)
    }

    /// Validates a chat template for suspicious patterns that could indicate
    /// denial of service risks.
    ///
    /// This is a defense-in-depth measure. Even if validation passes, templates
    /// should only be loaded from trusted sources.
    ///
    /// # Arguments
    /// * `template` - The Jinja2 template string to validate
    ///
    /// # Returns
    /// `Ok(())` if the template passes basic safety checks, `Err(warning)` with
    /// a description of the concern otherwise.
    fn validate_template_safety(template: &str) -> std::result::Result<(), String> {
        // Check for extremely long templates that might cause issues
        const MAX_TEMPLATE_LENGTH: usize = 100_000;
        if template.len() > MAX_TEMPLATE_LENGTH {
            return Err(format!(
                "Chat template exceeds maximum length ({} > {} bytes)",
                template.len(),
                MAX_TEMPLATE_LENGTH
            ));
        }

        // Check for excessive loop nesting (potential DoS risk)
        const MAX_LOOPS: usize = 20;
        let loop_count = template.matches("{% for").count();
        if loop_count > MAX_LOOPS {
            return Err(format!(
                "Chat template has {} loop constructs (max: {}), which may affect performance",
                loop_count, MAX_LOOPS
            ));
        }

        // Check for recursive macro definitions (potential infinite recursion)
        let macro_count = template.matches("{% macro").count();
        let call_count = template.matches("{% call").count();
        if macro_count > 10 && call_count > macro_count * 2 {
            return Err(format!(
                "Chat template has {} macros with {} calls, potential recursion risk",
                macro_count, call_count
            ));
        }

        Ok(())
    }

    /// Encode text to token IDs
    ///
    /// # Arguments
    /// * `text` - Text to encode
    /// * `add_special_tokens` - Whether to add special tokens (default: true)
    ///
    /// # Returns
    /// Array of token IDs as Int32Array
    ///
    /// # Example
    /// ```typescript
    /// const tokens = tokenizer.encode("Hello, world!");
    /// console.log(tokens); // Int32Array [9906, 11, 1879, 0]
    /// ```
    #[napi]
    pub fn encode<'env>(
        &self,
        env: &'env Env,
        text: String,
        add_special_tokens: Option<bool>,
    ) -> Result<PromiseRaw<'env, Uint32ArraySlice<'env>>> {
        let tokenizer = self.tokenizer.clone();
        env.spawn_future_with_callback(
            async move {
                napi::bindgen_prelude::spawn_blocking(move || {
                    Self::encode_internal(&tokenizer, text, add_special_tokens)
                })
                .await
                .map_err(|join_error| {
                    Error::new(
                        Status::GenericFailure,
                        format!("Spawn tokenizer::encode failed: {join_error}"),
                    )
                })?
            },
            encoding_to_uint32_array,
        )
    }

    fn encode_internal<'s, E>(
        tokenizer: &Arc<Tokenizer>,
        text: E,
        add_special_tokens: Option<bool>,
    ) -> Result<Encoding>
    where
        E: Into<EncodeInput<'s>>,
    {
        let add_special = add_special_tokens.unwrap_or(true);
        tokenizer
            .encode(text, add_special)
            .map_err(|e| Error::new(Status::InvalidArg, format!("Encoding failed: {}", e)))
    }

    /// Encode multiple texts in batch
    ///
    /// # Arguments
    /// * `texts` - Array of texts to encode
    /// * `add_special_tokens` - Whether to add special tokens (default: true)
    ///
    /// # Returns
    /// Array of Int32Arrays, one for each text
    #[napi]
    pub fn encode_batch<'env>(
        &self,
        env: &'env Env,
        texts: Vec<String>,
        add_special_tokens: Option<bool>,
    ) -> Result<PromiseRaw<'env, Vec<Uint32ArraySlice<'env>>>> {
        let add_special = add_special_tokens.unwrap_or(true);

        let tokenizer = self.tokenizer.clone();

        env.spawn_future_with_callback(
            async move {
                napi::bindgen_prelude::spawn_blocking(move || {
                    tokenizer.encode_batch(texts, add_special).map_err(|e| {
                        Error::new(Status::InvalidArg, format!("Batch encoding failed: {}", e))
                    })
                })
                .await
                .map_err(|join_error| {
                    Error::new(
                        Status::GenericFailure,
                        format!("Spawn tokenizer::encode_batch failed: {join_error}"),
                    )
                })?
            },
            |env, encodings| {
                encodings
                    .into_iter()
                    .map(|encoding| encoding_to_uint32_array(env, encoding))
                    .collect()
            },
        )
    }

    /// Decode token IDs to text
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs to decode
    /// * `skip_special_tokens` - Whether to skip special tokens (default: true)
    ///
    /// # Returns
    /// Decoded text string
    ///
    /// # Example
    /// ```typescript
    /// const text = tokenizer.decode(new Int32Array([9906, 11, 1879, 0]));
    /// console.log(text); // "Hello, world!"
    /// ```
    #[napi]
    pub fn decode<'env>(
        &self,
        env: &'env Env,
        token_ids: Uint32Array,
        skip_special_tokens: Option<bool>,
    ) -> Result<PromiseRaw<'env, String>> {
        let skip_special = skip_special_tokens.unwrap_or(true);
        let tokenizer = self.tokenizer.clone();

        env.spawn_future(async move {
            napi::bindgen_prelude::spawn_blocking(move || {
                tokenizer
                    .decode(&token_ids, skip_special)
                    .map_err(|e| Error::from_reason(format!("Decoding failed: {}", e)))
            })
            .await
            .map_err(|join_error| {
                Error::new(
                    Status::GenericFailure,
                    format!("Spawn tokenizer::decode failed: {join_error}"),
                )
            })?
        })
    }

    /// Decode multiple token sequences in batch
    ///
    /// # Arguments
    /// * `token_ids_batch` - Array of token ID arrays to decode
    /// * `skip_special_tokens` - Whether to skip special tokens (default: true)
    ///
    /// # Returns
    /// Array of decoded text strings
    #[napi]
    pub fn decode_batch<'env>(
        &self,
        env: &'env Env,
        token_ids_batch: Vec<Uint32Array>,
        skip_special_tokens: Option<bool>,
    ) -> Result<PromiseRaw<'env, Vec<String>>> {
        let skip_special = skip_special_tokens.unwrap_or(true);
        let tokenizer = self.tokenizer.clone();

        env.spawn_future(async move {
            napi::bindgen_prelude::spawn_blocking(move || {
                let token_ids_vec: Vec<&[u32]> =
                    token_ids_batch.iter().map(|arr| arr.as_ref()).collect();
                tokenizer
                    .decode_batch(&token_ids_vec, skip_special)
                    .map_err(|e| Error::from_reason(format!("Batch decoding failed: {}", e)))
            })
            .await
            .map_err(|join_error| {
                Error::new(
                    Status::GenericFailure,
                    format!("Spawn tokenizer::decode_batch failed: {join_error}"),
                )
            })?
        })
    }

    /// Apply chat template to messages and encode
    ///
    /// Supports both simple ChatML format and full Jinja2 template rendering with tools.
    /// When tools are provided or a chat template exists, uses Jinja2 rendering.
    /// Otherwise falls back to simple ChatML format.
    ///
    /// # Arguments
    /// * `messages` - Array of chat messages
    /// * `add_generation_prompt` - Whether to add assistant prompt at end (default: true)
    /// * `tools` - Optional array of tool definitions for function calling
    /// * `enable_thinking` - Optional flag to enable thinking mode (<think> tags)
    ///
    /// # Returns
    /// Encoded token IDs ready for model input
    ///
    /// # Example
    /// ```typescript
    /// const messages = [
    ///   { role: "system", content: "You are a helpful assistant." },
    ///   { role: "user", content: "What is 2+2?" }
    /// ];
    /// const tokens = tokenizer.applyChatTemplate(messages, true);
    ///
    /// // With tools
    /// const tools = [{
    ///   type: "function",
    ///   function: { name: "get_weather", description: "Get weather info" }
    /// }];
    /// const tokens = tokenizer.applyChatTemplate(messages, true, tools);
    /// ```
    #[napi]
    pub fn apply_chat_template<'env>(
        &self,
        env: &'env Env,
        messages: Vec<ChatMessage>,
        add_generation_prompt: Option<bool>,
        tools: Option<Vec<ToolDefinition>>,
        enable_thinking: Option<bool>,
    ) -> Result<PromiseRaw<'env, Uint32ArraySlice<'env>>> {
        let add_prompt = add_generation_prompt.unwrap_or(true);
        let tokenizer = self.tokenizer.clone();
        let chat_template = self.chat_template.clone();
        let bos_str = self
            .bos_token_id
            .and_then(|id| self.tokenizer.id_to_token(id))
            .unwrap_or_default();
        let eos_str = self
            .tokenizer
            .id_to_token(self.eos_token_id)
            .unwrap_or_default();

        env.spawn_future_with_callback(
            async move {
                napi::bindgen_prelude::spawn_blocking(move || {
                    // Sanitize messages before formatting (prevents injection in all paths)
                    let sanitized: Vec<ChatMessage> = Self::sanitize_messages(&messages);

                    // Use Jinja2 rendering if template exists, fallback to ChatML otherwise
                    let formatted = if let Some(chat_template) = chat_template {
                        Self::render_chat_template_jinja2(
                            &chat_template,
                            &sanitized,
                            tools.as_deref(),
                            add_prompt,
                            enable_thinking,
                            &bos_str,
                            &eos_str,
                        )
                        .map_err(Error::from_reason)?
                    } else {
                        // Fallback to simple ChatML when no template in tokenizer_config.json
                        Self::format_chatml_presanitized(&sanitized, add_prompt)
                    };

                    Self::encode_internal(&tokenizer, formatted, Some(false)) // Don't add extra special tokens
                })
                .await
                .map_err(|join_error| {
                    Error::new(
                        Status::GenericFailure,
                        format!("Spawn tokenizer::encode failed: {join_error}"),
                    )
                })?
            },
            |env, encoding| {
                let ids = encoding.get_ids();
                unsafe {
                    Uint32ArraySlice::from_external(
                        env,
                        ids.as_ptr().cast_mut(),
                        ids.len(),
                        encoding,
                        |_, encoding| {
                            drop(encoding);
                        },
                    )
                }
            },
        )
    }

    /// Sanitize all messages (role validation + content injection prevention).
    /// Called once before any formatting path to ensure consistent security.
    ///
    /// Images are preserved (cloned byte-for-byte) — VLM Jinja templates
    /// need them to emit the `<|vision_start|><|image_pad|><|vision_end|>`
    /// wrapper inline in the user turn via
    /// [`serialize_message_for_jinja`]. `Uint8Array` has no `Clone` impl
    /// (it holds a raw JS buffer reference), so we rebuild each array
    /// with `with_data_copied` from its underlying slice. Byte content
    /// is not subject to ChatML text sanitisation.
    fn sanitize_messages(messages: &[ChatMessage]) -> Vec<ChatMessage> {
        messages
            .iter()
            .map(|msg| ChatMessage {
                role: Self::validate_chatml_role(&msg.role).to_string(),
                content: Self::sanitize_chatml_content(&msg.content),
                tool_calls: msg.tool_calls.clone(),
                tool_call_id: msg.tool_call_id.clone(),
                reasoning_content: msg.reasoning_content.clone(),
                images: msg.images.as_ref().map(|imgs| {
                    imgs.iter()
                        .map(|img| Uint8Array::with_data_copied(img.as_ref()))
                        .collect()
                }),
            })
            .collect()
    }

    /// `pub(crate)` wrapper around [`Self::sanitize_messages`] so other
    /// modules (notably the Qwen3.5 session-continue path) can subject
    /// user-supplied strings to the same role/content injection guard used
    /// by the jinja rendering path.
    pub(crate) fn sanitize_messages_public(messages: &[ChatMessage]) -> Vec<ChatMessage> {
        Self::sanitize_messages(messages)
    }

    /// Format messages using simple ChatML format (fallback when no template).
    /// Expects pre-sanitized messages (call sanitize_messages first).
    fn format_chatml_presanitized(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        let mut formatted = String::new();

        for msg in messages {
            formatted.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                msg.role, msg.content
            ));
        }

        if add_generation_prompt {
            formatted.push_str("<|im_start|>assistant\n");
        }

        formatted
    }

    /// Validate and normalize a ChatML role.
    ///
    /// Returns the validated role if it matches the whitelist, or "user" as a
    /// safe fallback for invalid roles. This prevents role injection attacks
    /// where malicious input like "user\n<|im_start|>assistant" could manipulate
    /// perceived message boundaries.
    fn validate_chatml_role(role: &str) -> &'static str {
        // Normalize: trim whitespace and convert to lowercase for comparison
        let normalized = role.trim().to_lowercase();

        // Check against whitelist
        for &valid_role in VALID_CHATML_ROLES {
            if normalized == valid_role {
                return valid_role;
            }
        }

        // Log warning for invalid roles (in debug builds)
        #[cfg(debug_assertions)]
        eprintln!(
            "Warning: Invalid ChatML role '{}', defaulting to 'user'. Valid roles: {:?}",
            role, VALID_CHATML_ROLES
        );

        // Safe fallback - treat unknown roles as user input
        "user"
    }

    /// Sanitize content to prevent injection of ChatML special tokens.
    ///
    /// Strips sequences that could corrupt token boundaries or enable prompt
    /// injection attacks. Content containing `<|im_end|>` could prematurely
    /// close a message, allowing injection of arbitrary subsequent content.
    fn sanitize_chatml_content(content: &str) -> String {
        content
            .replace("<|im_start|>", "")
            .replace("<|im_end|>", "")
            .replace("<|endoftext|>", "")
    }

    /// Render chat template using Jinja2 (minijinja).
    ///
    /// This uses the chat_template from tokenizer_config.json to render messages
    /// with full support for tools, thinking mode, and other Qwen3 features.
    ///
    /// # Security Considerations
    ///
    /// This function assumes that the `template_str` (loaded from `tokenizer_config.json`)
    /// comes from a **trusted source** (e.g., Hugging Face Hub, local model files you control).
    ///
    /// ## Why Trust Matters
    ///
    /// While minijinja is designed for safe template rendering and sandboxes execution:
    /// - No file system access from templates
    /// - No arbitrary code execution
    /// - No access to Rust internals
    ///
    /// A malicious template from an untrusted source could still:
    /// - **Cause excessive resource usage**: Deep loops or recursion could consume
    ///   CPU/memory, causing denial of service.
    /// - **Extract context data unexpectedly**: The template has access to the full
    ///   context (messages, tools), and could potentially format/expose this data
    ///   in unexpected ways.
    ///
    /// ## Recommendations
    ///
    /// - **DO** use tokenizer files from official Hugging Face repositories
    /// - **DO** use your own trained/fine-tuned model files
    /// - **DO NOT** load `tokenizer_config.json` from untrusted sources
    /// - **DO NOT** allow user-uploaded tokenizer configurations without verification
    ///
    /// # Arguments
    /// * `template_str` - The Jinja2 template string (from tokenizer_config.json)
    /// * `messages` - Chat messages to format (content is escaped by the template engine)
    /// * `tools` - Optional tool definitions for function calling
    /// * `add_generation_prompt` - Whether to add the assistant prompt prefix
    /// * `enable_thinking` - Whether to enable thinking mode (`<think>` tags)
    ///
    /// # Returns
    /// Rendered template string ready for tokenization, or an error description.
    fn render_chat_template_jinja2(
        template_str: &str,
        messages: &[ChatMessage],
        tools: Option<&[ToolDefinition]>,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
        bos_token: &str,
        eos_token: &str,
    ) -> std::result::Result<String, String> {
        let mut env = Environment::new();

        // Add the tojson filter that Qwen3's template uses
        env.add_filter("tojson", |value: minijinja::Value| -> String {
            serde_json::to_string(&value).unwrap_or_else(|_| "null".to_string())
        });

        // Add Python-compatible string methods that Qwen3's template uses
        // These are called as methods on strings: content.startswith('prefix')
        //
        // Also bridges Python-dict `.get(key[, default])` on mappings,
        // which Gemma4's chat_template.jinja relies on
        // (`message.get('reasoning_content')`, `message.get('tool_calls')`)
        // — miniJinja only exposes bracket access `map[key]` out of the
        // box, and a missing `.get` aborts template rendering with
        // `unknown method: map has no method named get`.
        env.set_unknown_method_callback(|_state, value, method, args| {
            // String methods (Qwen3.5 / LFM2 / Gemma4 all use these)
            if let Some(s) = value.as_str() {
                match method {
                    "startswith" => {
                        if let Some(prefix) = args.first().and_then(|v| v.as_str()) {
                            return Ok(minijinja::Value::from(s.starts_with(prefix)));
                        }
                        return Err(minijinja::Error::new(
                            minijinja::ErrorKind::InvalidOperation,
                            "startswith requires a string argument",
                        ));
                    }
                    "endswith" => {
                        if let Some(suffix) = args.first().and_then(|v| v.as_str()) {
                            return Ok(minijinja::Value::from(s.ends_with(suffix)));
                        }
                        return Err(minijinja::Error::new(
                            minijinja::ErrorKind::InvalidOperation,
                            "endswith requires a string argument",
                        ));
                    }
                    "strip" => {
                        if let Some(chars) = args.first().and_then(|v| v.as_str()) {
                            return Ok(minijinja::Value::from(
                                s.trim_matches(|c| chars.contains(c)),
                            ));
                        }
                        return Ok(minijinja::Value::from(s.trim()));
                    }
                    "lstrip" => {
                        if let Some(chars) = args.first().and_then(|v| v.as_str()) {
                            return Ok(minijinja::Value::from(
                                s.trim_start_matches(|c| chars.contains(c)),
                            ));
                        }
                        return Ok(minijinja::Value::from(s.trim_start()));
                    }
                    "rstrip" => {
                        if let Some(chars) = args.first().and_then(|v| v.as_str()) {
                            return Ok(minijinja::Value::from(
                                s.trim_end_matches(|c| chars.contains(c)),
                            ));
                        }
                        return Ok(minijinja::Value::from(s.trim_end()));
                    }
                    "split" => {
                        let delim = args.first().and_then(|v| v.as_str());
                        let maxsplit = args
                            .get(1)
                            .and_then(|v| i64::try_from(v.clone()).ok())
                            .filter(|&n| n >= 0);
                        let parts: Vec<&str> = match (delim, maxsplit) {
                            (Some(d), Some(n)) => s.splitn(n as usize + 1, d).collect(),
                            (Some(d), None) => s.split(d).collect(),
                            (None, Some(n)) => {
                                s.splitn(n as usize + 1, char::is_whitespace).collect()
                            }
                            (None, None) => s.split_whitespace().collect(),
                        };
                        return Ok(minijinja::Value::from(
                            parts
                                .into_iter()
                                .map(minijinja::Value::from)
                                .collect::<Vec<_>>(),
                        ));
                    }
                    _ => {
                        return Err(minijinja::Error::new(
                            minijinja::ErrorKind::UnknownMethod,
                            format!("string has no method named {}", method),
                        ));
                    }
                }
            }

            // Map/dict methods (Gemma4 uses `.get(key[, default])`)
            if value.kind() == minijinja::value::ValueKind::Map {
                match method {
                    "get" => {
                        let key = args.first().ok_or_else(|| {
                            minijinja::Error::new(
                                minijinja::ErrorKind::InvalidOperation,
                                "get requires a key argument",
                            )
                        })?;
                        let key_str = key.as_str().ok_or_else(|| {
                            minijinja::Error::new(
                                minijinja::ErrorKind::InvalidOperation,
                                "get key must be a string",
                            )
                        })?;
                        let default = args.get(1).cloned().unwrap_or(minijinja::Value::UNDEFINED);
                        match value.get_attr(key_str) {
                            Ok(v) if !v.is_undefined() => Ok(v),
                            _ => Ok(default),
                        }
                    }
                    _ => Err(minijinja::Error::new(
                        minijinja::ErrorKind::UnknownMethod,
                        format!("map has no method named {}", method),
                    )),
                }
            } else {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::UnknownMethod,
                    format!("{} has no method named {}", value.kind(), method),
                ))
            }
        });

        // Register raise_exception (used by official Qwen3.5 VLM template for validation)
        env.add_function(
            "raise_exception",
            |msg: String| -> std::result::Result<minijinja::Value, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );

        env.add_template("chat", template_str)
            .map_err(|e| format!("Template parse error: {}", e))?;

        let tmpl = env
            .get_template("chat")
            .map_err(|e| format!("Template not found: {}", e))?;

        // Convert tools to JSON-serializable format for minijinja
        let tools_value: Option<Vec<serde_json::Value>> = tools.map(|t| {
            t.iter()
                .map(|tool| {
                    let mut obj = serde_json::Map::new();
                    obj.insert("type".to_string(), serde_json::json!(tool.r#type));

                    let mut func = serde_json::Map::new();
                    func.insert("name".to_string(), serde_json::json!(tool.function.name));
                    if let Some(desc) = &tool.function.description {
                        func.insert("description".to_string(), serde_json::json!(desc));
                    }
                    if let Some(params) = &tool.function.parameters {
                        let mut params_obj = serde_json::Map::new();
                        params_obj.insert("type".to_string(), serde_json::json!(params.r#type));
                        if let Some(props) = &params.properties {
                            // Parse the JSON string to include it properly
                            match serde_json::from_str::<serde_json::Value>(props) {
                                Ok(props_val) => {
                                    params_obj.insert("properties".to_string(), props_val);
                                }
                                Err(e) => {
                                    warn!("Failed to parse tool properties JSON: {}", e);
                                }
                            }
                        }
                        if let Some(req) = &params.required {
                            params_obj.insert("required".to_string(), serde_json::json!(req));
                        }
                        func.insert(
                            "parameters".to_string(),
                            serde_json::Value::Object(params_obj),
                        );
                    }

                    obj.insert("function".to_string(), serde_json::Value::Object(func));
                    serde_json::Value::Object(obj)
                })
                .collect()
        });

        // Convert messages to JSON-serializable format (already sanitized by caller)
        let messages_value: Vec<serde_json::Value> =
            messages.iter().map(serialize_message_for_jinja).collect();

        // Build context for Jinja2 template
        // Note: enable_thinking defaults to true to allow model to think naturally.
        // Setting to false adds empty <think></think> tags which DISABLES thinking.
        // bos_token/eos_token: used by Gemma4 and other templates ({{ bos_token }}).
        //
        // `preserve_thinking=true` keeps `reasoning_content` rendered on
        // EVERY prior assistant turn, not just on the most recent one after
        // the last user query. Qwen3.5/3.6's template gate is
        //   `preserve_thinking or loop.index0 > ns.last_query_index`
        // which means when a NEW user message arrives mid-session,
        // `last_query_index` jumps forward and all earlier assistant turns
        // silently drop their `<think>…</think>` blocks on re-render. That
        // flips the token prefix at the first reasoning boundary, so the
        // server's tier-2 KV cache misses entirely and the next turn cold-
        // prefills the full conversation. Pinning `preserve_thinking=true`
        // keeps the rendered prompt byte-stable turn over turn so
        // `verify_cache_prefix_direct` can reuse the prior cached prefix.
        //
        // Templates that don't read `preserve_thinking` (e.g. Qwen3
        // non-thinking, LFM2, Gemma4) ignore the extra key — minijinja
        // treats unknown variables in `context!` as a no-op on access.
        let ctx = context! {
            messages => messages_value,
            tools => tools_value,
            add_generation_prompt => add_generation_prompt,
            enable_thinking => enable_thinking.unwrap_or(true),
            preserve_thinking => true,
            bos_token => bos_token,
            eos_token => eos_token,
        };

        tmpl.render(ctx)
            .map_err(|e| format!("Template render error: {}", e))
    }

    /// Get vocabulary size
    #[napi]
    pub fn vocab_size(&self) -> u32 {
        self.tokenizer.get_vocab_size(true) as u32
    }

    /// Get PAD token ID
    #[napi]
    pub fn get_pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    /// Get EOS token ID
    #[napi]
    pub fn get_eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get BOS token ID (if exists)
    #[napi]
    pub fn get_bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Convert token ID to string
    #[napi]
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    /// Convert token string to ID
    #[napi]
    pub fn token_to_id(&self, token: String) -> Option<u32> {
        self.tokenizer.token_to_id(&token)
    }

    /// Get the special token for IM_START
    #[napi]
    pub fn get_im_start_token(&self) -> String {
        "<|im_start|>".to_string()
    }

    /// Get the special token for IM_END
    #[napi]
    pub fn get_im_end_token(&self) -> String {
        "<|im_end|>".to_string()
    }

    /// Get the special token for ENDOFTEXT (used as PAD)
    #[napi]
    pub fn get_endoftext_token(&self) -> String {
        "<|endoftext|>".to_string()
    }

    /// Load tokenizer from file synchronously (for internal use)
    ///
    /// This is used by load() to load the tokenizer without async overhead.
    pub(crate) fn load_from_file_sync(tokenizer_path: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::from_reason(format!("Failed to load tokenizer: {}", e)))?;

        // Load chat template from tokenizer_config.json (in same directory)
        let chat_template = Self::load_chat_template(tokenizer_path);

        let (think_end_id, think_end_str) = Self::detect_think_end(&tokenizer);

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            pad_token_id: ENDOFTEXT_TOKEN_ID,
            eos_token_id: IM_END_TOKEN_ID,
            bos_token_id: None,
            chat_template,
            think_end_id,
            think_end_str,
        })
    }

    /// Returns true if the tokenizer has a chat template loaded.
    ///
    /// Used by models (e.g. Gemma4) to decide whether to use the template or
    /// fall back to a model-specific manual prompt format.
    pub(crate) fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    /// Encode text synchronously (for internal use by generate())
    pub(crate) fn encode_sync(
        &self,
        text: &str,
        add_special_tokens: Option<bool>,
    ) -> Result<Vec<u32>> {
        let encoding = Self::encode_internal(&self.tokenizer, text, add_special_tokens)?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs synchronously (for internal use by generate())
    pub(crate) fn decode_sync(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| Error::from_reason(format!("Failed to decode tokens: {}", e)))
    }

    /// Get a reference to the inner tokenizer for creating a DecodeStream.
    pub(crate) fn inner(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Step the decode stream with error recovery. On InvalidPrefix,
    /// recreates the stream, replays all generated tokens, and returns
    /// the delta since `streamed_text_len`.
    pub(crate) fn step_decode_stream<'a>(
        decode_stream: &mut tokenizers::DecodeStream<
            'a,
            tokenizers::ModelWrapper,
            tokenizers::NormalizerWrapper,
            tokenizers::PreTokenizerWrapper,
            tokenizers::PostProcessorWrapper,
            tokenizers::DecoderWrapper,
        >,
        tokenizer: &'a tokenizers::Tokenizer,
        token_id: u32,
        generated_tokens: &[u32],
        streamed_text_len: usize,
    ) -> String {
        match decode_stream.step(token_id) {
            Ok(Some(text)) => text,
            Ok(None) => String::new(),
            Err(_) => {
                // Recreate stream and replay all tokens to recover state
                let mut new_ds = tokenizer.decode_stream(true);
                let mut replayed = String::new();
                for &tid in generated_tokens {
                    if let Ok(Some(t)) = new_ds.step(tid) {
                        replayed.push_str(&t);
                    }
                }
                *decode_stream = new_ds;
                if replayed.len() > streamed_text_len {
                    replayed[streamed_text_len..].to_string()
                } else {
                    String::new()
                }
            }
        }
    }

    /// Apply chat template synchronously (for internal use by chat())
    ///
    /// This is a synchronous version of apply_chat_template for use in blocking tasks.
    pub(crate) fn apply_chat_template_sync(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: Option<bool>,
        tools: Option<&[ToolDefinition]>,
        enable_thinking: Option<bool>,
    ) -> Result<Vec<u32>> {
        let add_prompt = add_generation_prompt.unwrap_or(true);

        // Sanitize messages before formatting (prevents injection in all paths)
        let sanitized: Vec<ChatMessage> = Self::sanitize_messages(messages);

        // Use Jinja2 rendering if template exists, fallback to ChatML otherwise
        let bos_str = self
            .bos_token_id
            .and_then(|id| self.tokenizer.id_to_token(id))
            .unwrap_or_default();
        let eos_str = self
            .tokenizer
            .id_to_token(self.eos_token_id)
            .unwrap_or_default();
        let formatted = if let Some(chat_template) = &self.chat_template {
            Self::render_chat_template_jinja2(
                chat_template,
                &sanitized,
                tools,
                add_prompt,
                enable_thinking,
                &bos_str,
                &eos_str,
            )
            .map_err(Error::from_reason)?
        } else {
            // Fallback to simple ChatML when no template in tokenizer_config.json
            Self::format_chatml_presanitized(&sanitized, add_prompt)
        };

        // Encode the formatted text (don't add extra special tokens)
        let encoding = Self::encode_internal(&self.tokenizer, formatted, Some(false))?;
        Ok(encoding.get_ids().to_vec())
    }
}

impl Clone for Qwen3Tokenizer {
    fn clone(&self) -> Self {
        Self {
            tokenizer: self.tokenizer.clone(),
            pad_token_id: self.pad_token_id,
            eos_token_id: self.eos_token_id,
            bos_token_id: self.bos_token_id,
            chat_template: self.chat_template.clone(),
            think_end_id: self.think_end_id,
            think_end_str: self.think_end_str.clone(),
        }
    }
}

impl Qwen3Tokenizer {
    /// Detect think-end token from tokenizer vocabulary.
    /// Returns (token_id, token_string) for whichever variant is found.
    fn detect_think_end(tokenizer: &Tokenizer) -> (Option<u32>, Option<String>) {
        let vocab = tokenizer.get_vocab(true);
        for tag in &["</think>", "</longcat_think>"] {
            if let Some(&id) = vocab.get(*tag) {
                return (Some(id), Some(tag.to_string()));
            }
        }
        (None, None)
    }

    /// Get the think-end token ID, if the tokenizer has thinking support.
    pub fn think_end_id(&self) -> Option<u32> {
        self.think_end_id
    }

    /// Get the think-end string (e.g., `"</think>"` or `"</longcat_think>"`).
    pub fn think_end_str(&self) -> Option<&str> {
        self.think_end_str.as_deref()
    }

    /// Get the `<|im_end|>` token ID, if the tokenizer has it in its vocab.
    ///
    /// This is the "turn end" sentinel for ChatML-style templates. It's
    /// preferable to `config.json:eos_token_id` for session-based chat
    /// because it yields clean cache boundaries: cached history ends at
    /// `<|im_end|>`, and the next turn's delta starts with
    /// `\n<|im_start|>user\n...`. Using the raw `eos_token_id` from
    /// `config.json` (which may be `<|endoftext|>` for Qwen3.5) wastes
    /// decode tokens and makes clean template continuation impossible.
    pub fn im_end_id(&self) -> Option<u32> {
        self.tokenizer.token_to_id("<|im_end|>")
    }
}

/// Serialize a single `ChatMessage` into the shape Jinja chat templates
/// expect.
///
/// Mirrors the Python reference `mlx-vlm/mlx_vlm/prompt_utils.py`'s
/// `_format_list_with_image`: when a `user` message carries one or more
/// images, `content` is rendered as a content-parts array
/// `[{type:"text", text:...}, {type:"image"}, ...]` so VLM Jinja templates
/// (Qwen3/3.5/3.6 VL, Gemma4, etc.) emit the `<|vision_start|>
/// <|image_pad|><|vision_end|>` wrapper inline in the user turn.
/// Otherwise `content` stays a plain string — preserving byte-for-byte
/// parity with every text-only template path.
///
/// `msg.images` is `#[serde(skip)]` so a direct `serde_json::to_value(msg)`
/// would drop images entirely, which is why this helper exists.
pub(crate) fn serialize_message_for_jinja(msg: &ChatMessage) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("role".to_string(), serde_json::json!(msg.role));

    let has_images = msg.images.as_ref().is_some_and(|imgs| !imgs.is_empty());

    if has_images && msg.role == "user" {
        let mut parts: Vec<serde_json::Value> = Vec::new();
        if !msg.content.is_empty() {
            parts.push(serde_json::json!({ "type": "text", "text": msg.content }));
        }
        // SAFETY: `is_some_and` above ensured this is Some and non-empty.
        for _ in msg.images.as_ref().unwrap() {
            parts.push(serde_json::json!({ "type": "image" }));
        }
        obj.insert("content".to_string(), serde_json::Value::Array(parts));
    } else {
        obj.insert("content".to_string(), serde_json::json!(msg.content));
    }

    if let Some(tool_calls) = &msg.tool_calls {
        let calls: Vec<serde_json::Value> = tool_calls
            .iter()
            .map(|tc| {
                let mut call_obj = serde_json::Map::new();
                if let Some(id) = &tc.id {
                    call_obj.insert("id".to_string(), serde_json::json!(id));
                }
                // Flat format (backward compat with some templates)
                call_obj.insert("name".to_string(), serde_json::json!(tc.name));
                // Parse arguments
                let args_value = serde_json::from_str::<serde_json::Value>(&tc.arguments)
                    .unwrap_or_else(|_| serde_json::json!(tc.arguments));
                call_obj.insert("arguments".to_string(), args_value.clone());
                // Wrapped format (Gemma4/OpenAI standard: tool_call.function.name)
                call_obj.insert(
                    "function".to_string(),
                    serde_json::json!({
                        "name": tc.name,
                        "arguments": args_value,
                    }),
                );
                serde_json::Value::Object(call_obj)
            })
            .collect();
        obj.insert("tool_calls".to_string(), serde_json::json!(calls));
    }

    if let Some(tool_call_id) = &msg.tool_call_id {
        obj.insert("tool_call_id".to_string(), serde_json::json!(tool_call_id));
    }

    if let Some(reasoning) = &msg.reasoning_content {
        obj.insert(
            "reasoning_content".to_string(),
            serde_json::json!(reasoning),
        );
    }

    serde_json::Value::Object(obj)
}

fn encoding_to_uint32_array<'env>(
    env: &'env Env,
    encoding: Encoding,
) -> Result<Uint32ArraySlice<'env>> {
    let ids = encoding.get_ids();
    unsafe {
        Uint32ArraySlice::from_external(
            env,
            ids.as_ptr().cast_mut(),
            ids.len(),
            encoding,
            |_, encoding| {
                drop(encoding);
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minijinja::{Environment, context};

    fn user_msg(content: &str, num_images: usize) -> ChatMessage {
        let images = if num_images > 0 {
            Some(
                (0..num_images)
                    .map(|i| Uint8Array::new(vec![i as u8; 4]))
                    .collect(),
            )
        } else {
            None
        };
        ChatMessage {
            role: "user".to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            images,
        }
    }

    #[test]
    fn text_only_user_renders_content_as_string() {
        // Preserves the existing shape for every text-only template path —
        // any change here would fork the byte-for-byte parity the
        // text-only suite (Qwen3, Qwen3.5, LFM2, Gemma4) relies on.
        let msg = user_msg("Hello", 0);
        let v = serialize_message_for_jinja(&msg);
        assert_eq!(v["role"], "user");
        assert!(v["content"].is_string());
        assert_eq!(v["content"], "Hello");
    }

    #[test]
    fn user_with_one_image_emits_content_array_with_text_and_image() {
        let msg = user_msg("Describe this.", 1);
        let v = serialize_message_for_jinja(&msg);
        assert_eq!(v["role"], "user");
        let parts = v["content"].as_array().expect("content is an array");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "Describe this.");
        assert_eq!(parts[1]["type"], "image");
    }

    #[test]
    fn user_with_multiple_images_emits_one_image_part_per_image() {
        let msg = user_msg("Compare.", 3);
        let v = serialize_message_for_jinja(&msg);
        let parts = v["content"].as_array().unwrap();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0]["type"], "text");
        for (i, part) in parts.iter().enumerate().skip(1) {
            assert_eq!(part["type"], "image", "part {i} should be image");
        }
    }

    #[test]
    fn user_image_without_text_omits_text_part() {
        // Empty content + one image → just the image part, no empty text
        // block. Matches mlx-vlm's `_format_list_with_image` output.
        let msg = user_msg("", 1);
        let v = serialize_message_for_jinja(&msg);
        let parts = v["content"].as_array().unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["type"], "image");
    }

    #[test]
    fn non_user_role_with_images_keeps_content_as_string() {
        // Only the user turn should ever ship images in practice; system /
        // assistant / tool keep their flat `content: string` shape so
        // templates that don't expect arrays on those roles keep working.
        let mut msg = user_msg("A reply", 2);
        msg.role = "assistant".to_string();
        let v = serialize_message_for_jinja(&msg);
        assert!(v["content"].is_string());
        assert_eq!(v["content"], "A reply");
    }

    #[test]
    fn user_images_none_is_equivalent_to_text_only() {
        let mut msg = user_msg("Hi", 0);
        msg.images = None;
        let v = serialize_message_for_jinja(&msg);
        assert_eq!(v["content"], "Hi");
    }

    #[test]
    fn user_images_empty_vec_is_equivalent_to_text_only() {
        // `is_some_and(|imgs| !imgs.is_empty())` must reject Some([]) too,
        // or the array branch would emit a content-array with just the
        // text part and trip downstream Jinja templates that only branch
        // on string-vs-array.
        let mut msg = user_msg("Hi", 0);
        msg.images = Some(Vec::new());
        let v = serialize_message_for_jinja(&msg);
        assert_eq!(v["content"], "Hi");
    }

    /// Render a minimal Jinja template that mimics the relevant slice of
    /// the Qwen3.6 VL chat template (see
    /// `.cache/models/Qwen3.6-35b-a3b-UD-Q4_K_XL-mlx/chat_template.jinja`)
    /// to verify the content-array path actually produces the vision
    /// wrapper inline inside the user turn — not spliced after BOS by the
    /// `inject_image_placeholders` fallback.
    #[test]
    fn rendered_prompt_includes_vision_wrapper_for_user_image() {
        let template = r#"{%- for message in messages -%}
<|im_start|>{{ message.role }}
{%- if message.content is string -%}
{{ message.content }}
{%- else -%}
{%- for item in message.content -%}
{%- if 'image' in item or item.type == 'image' -%}
<|vision_start|><|image_pad|><|vision_end|>
{%- elif 'text' in item -%}
{{ item.text }}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
<|im_end|>
{% endfor -%}"#;

        let mut env = Environment::new();
        env.add_template("chat", template).unwrap();
        let tmpl = env.get_template("chat").unwrap();

        let msg = user_msg("What is this?", 1);
        let messages_value: Vec<serde_json::Value> = vec![serialize_message_for_jinja(&msg)];

        let rendered = tmpl
            .render(context! { messages => messages_value })
            .unwrap();

        assert!(
            rendered.contains("<|vision_start|><|image_pad|><|vision_end|>"),
            "rendered prompt missing vision wrapper:\n{rendered}",
        );
        // The wrapper must land INSIDE the user turn, after the text.
        let start_idx = rendered.find("<|im_start|>user").unwrap();
        let end_idx = rendered[start_idx..].find("<|im_end|>").unwrap() + start_idx;
        let user_turn = &rendered[start_idx..end_idx];
        assert!(
            user_turn.contains("<|vision_start|>"),
            "vision wrapper not inside user turn: {user_turn}",
        );
        assert!(
            user_turn.contains("What is this?"),
            "user text missing from user turn: {user_turn}",
        );
    }

    /// `sanitize_messages` sits between `apply_chat_template(_sync)` and
    /// `render_chat_template_jinja2` on every production path. If it
    /// zeroes `images`, `serialize_message_for_jinja` sees
    /// `msg.images: None` and the VLM content-array branch never fires,
    /// so the template falls back to the post-BOS `inject_image_placeholders`
    /// splice (vision tokens outside the user turn). Guard against that
    /// regression directly.
    #[test]
    fn sanitize_messages_preserves_user_images_byte_for_byte() {
        let original = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "describe these".to_string(),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
                images: Some(vec![
                    Uint8Array::new(vec![0x01, 0x02, 0x03, 0x04]),
                    Uint8Array::new(vec![0xaa, 0xbb, 0xcc]),
                ]),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "ok".to_string(),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
                images: None,
            },
        ];

        let sanitized = Qwen3Tokenizer::sanitize_messages_public(&original);

        assert_eq!(sanitized.len(), 2);
        let user = &sanitized[0];
        assert_eq!(user.role, "user");
        let imgs = user
            .images
            .as_ref()
            .expect("user images must survive sanitise");
        assert_eq!(imgs.len(), 2);
        assert_eq!(imgs[0].as_ref(), &[0x01, 0x02, 0x03, 0x04]);
        assert_eq!(imgs[1].as_ref(), &[0xaa, 0xbb, 0xcc]);

        // assistant path unchanged: still None.
        assert!(sanitized[1].images.is_none());
    }

    /// End-to-end: sanitize → serialize → Jinja render. Covers the exact
    /// composition production runs every turn. The direct-serialize test
    /// above only proves the helper itself is correct — this one proves
    /// the production chain is correct.
    #[test]
    fn sanitize_then_render_emits_vision_wrapper_in_user_turn() {
        let template = r#"{%- for message in messages -%}
<|im_start|>{{ message.role }}
{%- if message.content is string -%}
{{ message.content }}
{%- else -%}
{%- for item in message.content -%}
{%- if 'image' in item or item.type == 'image' -%}
<|vision_start|><|image_pad|><|vision_end|>
{%- elif 'text' in item -%}
{{ item.text }}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
<|im_end|>
{% endfor -%}"#;

        let mut env = Environment::new();
        env.add_template("chat", template).unwrap();
        let tmpl = env.get_template("chat").unwrap();

        let msgs = vec![ChatMessage {
            role: "user".to_string(),
            content: "What is this?".to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            images: Some(vec![Uint8Array::new(vec![0; 4])]),
        }];

        let sanitized = Qwen3Tokenizer::sanitize_messages_public(&msgs);
        let messages_value: Vec<serde_json::Value> =
            sanitized.iter().map(serialize_message_for_jinja).collect();

        let rendered = tmpl
            .render(context! { messages => messages_value })
            .unwrap();

        let start_idx = rendered.find("<|im_start|>user").unwrap();
        let end_idx = rendered[start_idx..].find("<|im_end|>").unwrap() + start_idx;
        let user_turn = &rendered[start_idx..end_idx];
        assert!(
            user_turn.contains("<|vision_start|><|image_pad|><|vision_end|>"),
            "vision wrapper not inside user turn after sanitize: {user_turn}",
        );
        assert!(
            user_turn.contains("What is this?"),
            "user text missing from user turn after sanitize: {user_turn}",
        );
    }

    /// Minimal slice of the stock Qwen3.5 chat template — just the
    /// last-query-index scan and the assistant `<think>` gate. Used by
    /// the preserve-thinking regression tests to verify the fix does what
    /// we say it does on the exact expression we rewrite at load time,
    /// without the noise of the full 7 KB template.
    ///
    /// Simplification vs. the shipped template: the tool-response check
    /// `content.startswith('<tool_response>')` is dropped — miniJinja's
    /// string-method bridge lives in `render_chat_template_jinja2` and we
    /// want these tests self-contained. All test fixtures use plain
    /// user text that never matches that branch anyway.
    const QWEN35_GATE_SLICE: &str = "{%- set ns = namespace(last_query_index=-1) %}\n{%- for message in messages %}\n    {%- if message.role == \"user\" %}\n        {%- set ns.last_query_index = loop.index0 %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- set content = message.content|trim %}\n    {%- if message.role == \"user\" %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- endif %}\n        {%- set reasoning_content = reasoning_content|trim %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content + '<|im_end|>\\n' }}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}";

    /// Build the Jinja env the same way `render_chat_template_jinja2` does
    /// (minus the string-method bridge, which this slice doesn't need).
    /// Keeps the fix tests rendering through the SAME `tojson` filter
    /// production uses, so any future filter drift trips these tests too.
    fn jinja_env() -> Environment<'static> {
        let mut env = Environment::new();
        env.add_filter("tojson", |value: minijinja::Value| -> String {
            serde_json::to_string(&value).unwrap_or_else(|_| "null".to_string())
        });
        env
    }

    #[test]
    fn patch_preserve_thinking_rewrites_qwen35_gate() {
        let patched = Qwen3Tokenizer::patch_preserve_thinking(QWEN35_GATE_SLICE);
        assert!(
            patched.contains("preserve_thinking or loop.index0 > ns.last_query_index"),
            "patched template missing preserve_thinking clause:\n{patched}",
        );
        // The stock gate (without the new disjunct) must be gone — any
        // leftover copy would still drop <think> on old assistants.
        let stock_occurrences = patched.matches("loop.index0 > ns.last_query_index").count();
        let preserve_occurrences = patched.matches("preserve_thinking or").count();
        assert_eq!(
            stock_occurrences, preserve_occurrences,
            "every `loop.index0 > ns.last_query_index` must be prefixed with `preserve_thinking or`",
        );
    }

    #[test]
    fn patch_preserve_thinking_is_idempotent() {
        let once = Qwen3Tokenizer::patch_preserve_thinking(QWEN35_GATE_SLICE);
        let twice = Qwen3Tokenizer::patch_preserve_thinking(&once);
        assert_eq!(
            once, twice,
            "patching twice must be a no-op so `preserve_thinking` never gets nested",
        );
    }

    #[test]
    fn patch_preserve_thinking_passthrough_on_unrelated_templates() {
        // Templates that don't carry the Qwen3.5 gate (Gemma4 / LFM2 /
        // minimal ChatML) must survive verbatim — no `replace` call is
        // allowed to silently corrupt their control flow.
        let gemma4 = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}";
        assert_eq!(Qwen3Tokenizer::patch_preserve_thinking(gemma4), gemma4);
    }

    /// Reproduces the turn-15 → turn-16 divergence from
    /// `.logging/requests.ndjson`: same assistant ChatMessage, rendered
    /// once as the end-of-turn cache state (no new user yet) and once as
    /// the next-turn echo (new user appended). The stock Qwen3.5 gate
    /// drops the prior assistant's `<think>` on the second render,
    /// breaking the byte-equal prefix `verify_cache_prefix_direct`
    /// expects. The patch must restore parity.
    #[test]
    fn preserve_thinking_keeps_think_block_when_new_user_turn_appended() {
        let env = jinja_env();

        let patched = Qwen3Tokenizer::patch_preserve_thinking(QWEN35_GATE_SLICE);
        let build = |messages: Vec<serde_json::Value>, preserve: bool| {
            let mut env = env.clone();
            env.add_template("chat", &patched).unwrap();
            let tmpl = env.get_template("chat").unwrap();
            tmpl.render(context! {
                messages => messages,
                preserve_thinking => preserve,
            })
            .unwrap()
        };

        let assistant = serde_json::json!({
            "role": "assistant",
            "content": "Done.",
            "reasoning_content": "step-by-step reasoning",
        });
        let msgs_end_of_turn = vec![
            serde_json::json!({ "role": "user", "content": "Hi" }),
            assistant.clone(),
        ];
        let msgs_new_user_appended = vec![
            serde_json::json!({ "role": "user", "content": "Hi" }),
            assistant.clone(),
            serde_json::json!({ "role": "user", "content": "Follow-up?" }),
        ];

        // Patched template + preserve_thinking=true: the assistant turn
        // must render identically in both contexts, so the warm cache
        // carries over to the follow-up turn byte-for-byte.
        let r_before = build(msgs_end_of_turn.clone(), true);
        let r_after = build(msgs_new_user_appended.clone(), true);

        let extract_assistant = |rendered: &str| -> String {
            let start = rendered.find("<|im_start|>assistant").unwrap();
            let end_rel = rendered[start..].find("<|im_end|>").unwrap();
            rendered[start..start + end_rel + "<|im_end|>\n".len()].to_string()
        };

        assert_eq!(
            extract_assistant(&r_before),
            extract_assistant(&r_after),
            "with preserve_thinking=true the echoed assistant turn must match the end-of-turn render",
        );
        assert!(
            extract_assistant(&r_after).contains("<think>\nstep-by-step reasoning\n</think>"),
            "echoed assistant turn must keep the <think> block intact",
        );

        // Control: with preserve_thinking=false (the stock behavior we
        // used to ship before the patch propagated) the `<think>` block
        // gets dropped when a new user arrives — which is exactly the
        // miss we observed on turn 16.
        let r_after_stock = build(msgs_new_user_appended, false);
        assert!(
            !extract_assistant(&r_after_stock).contains("<think>"),
            "stock gate (preserve_thinking=false) should drop <think> when a new user turn appends — sanity check on the control",
        );
    }

    /// Cache-reuse regression: the same assistant ChatMessage shape that
    /// turn 10 emitted (reasoning + content + two function_calls with
    /// schema-declared arg order `[path, edits]`) must re-render
    /// byte-for-byte across two consecutive tool-loop turns. Before we
    /// enabled serde_json's `preserve_order`, the BTreeMap default
    /// alphabetised `path`+`edits` into `edits`+`path`, swapping two
    /// `<parameter=…>` blocks and zeroing the cache at turn 11.
    /// End-to-end render of the stock Gemma4 chat_template.jinja
    /// through the production `render_chat_template_jinja2` entry
    /// point. `#[ignore]`-gated because the template lives in
    /// `.cache/` and tests run without network; opt in locally with
    /// `cargo test gemma4_full_template_renders -- --include-ignored`.
    ///
    /// This guards against future template features (Python idioms,
    /// new filters) we haven't bridged — the Jinja engine aborts
    /// rendering with `unknown method: … has no method named X` the
    /// moment it meets one it doesn't know.
    #[test]
    #[ignore]
    fn gemma4_full_template_renders_without_missing_methods() {
        let path = "/Users/brooklyn/workspace/github/mlx-node/.cache/models/gemma-4-26b-a4b-it-UD-Q8_K_XL-mlx/chat_template.jinja";
        let Ok(tmpl) = std::fs::read_to_string(path) else {
            // Skip silently when the fixture isn't checked out locally.
            return;
        };
        let msgs = vec![ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            images: None,
        }];
        let tools = vec![ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: "read".to_string(),
                description: Some("Read a file".to_string()),
                parameters: Some(FunctionParameters {
                    r#type: "object".to_string(),
                    properties: Some(
                        r#"{"path":{"type":"string","description":"file path"}}"#.to_string(),
                    ),
                    required: Some(vec!["path".to_string()]),
                }),
            },
        }];
        let rendered = Qwen3Tokenizer::render_chat_template_jinja2(
            &tmpl,
            &msgs,
            Some(&tools),
            true,
            Some(true),
            "<bos>",
            "<eos>",
        )
        .unwrap_or_else(|e| {
            panic!("Gemma4 template render failed: {e}");
        });
        assert!(
            rendered.contains("<|turn>user"),
            "rendered prompt missing user turn marker:\n{rendered}",
        );
        assert!(
            rendered.contains("<|tool>"),
            "rendered prompt missing tool declaration block:\n{rendered}",
        );
    }

    /// Gemma4's chat_template.jinja leans on Python's dict `.get()`
    /// idiom (`message.get('reasoning_content')`,
    /// `message.get('tool_calls')`, etc.) to avoid UndefinedError when
    /// an optional key is absent. miniJinja only ships bracket access
    /// out of the box, so we bridge `.get` ourselves in
    /// `render_chat_template_jinja2`. If this test fails, any Gemma4
    /// request aborts at template render time with
    /// `unknown method: map has no method named get`.
    #[test]
    fn map_get_bridge_mirrors_python_dict_get() {
        // Reuse the production Jinja setup — the bridge lives inside
        // `render_chat_template_jinja2`, so driving a real ChatMessage
        // through it exercises exactly the call site the shipped
        // template hits.
        let msg = ChatMessage {
            role: "assistant".to_string(),
            content: "hello".to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: Some("because".to_string()),
            images: None,
        };
        // Minimal template that drives the fixture map through `.get()`
        // three different ways — hit, miss (no default), miss (with
        // default). Any drift in the bridge trips this test.
        let template = "{% set m = messages[0] %}{{ m.get('role') }}|{{ m.get('missing') }}|{{ m.get('missing', 'fallback') }}|{{ m.get('reasoning_content') }}";
        let rendered = Qwen3Tokenizer::render_chat_template_jinja2(
            template,
            std::slice::from_ref(&msg),
            None,
            false,
            None,
            "<bos>",
            "<eos>",
        )
        .unwrap();
        // Undefined keys render to the empty string by default —
        // matching Python-Jinja behaviour for `dict.get(missing)`.
        assert_eq!(rendered, "assistant||fallback|because");
    }

    #[test]
    fn function_call_arg_order_survives_jinja_round_trip() {
        let mut env = jinja_env();
        let patched = Qwen3Tokenizer::patch_preserve_thinking(QWEN35_GATE_SLICE);
        env.add_template("chat", &patched).unwrap();

        // Args string exactly as pi-mono would echo it back — note
        // `path` first, matching the tool schema's `required` order
        // and whatever the model emitted on the prior turn.
        let args = r#"{"path":"/a.json","edits":[{"oldText":"x","newText":"y"}]}"#;
        let call = ToolCall {
            id: Some("call_1".to_string()),
            name: "edit".to_string(),
            arguments: args.to_string(),
        };
        let msg = ChatMessage {
            role: "assistant".to_string(),
            content: "Making the edit.".to_string(),
            tool_calls: Some(vec![call]),
            tool_call_id: None,
            reasoning_content: Some("think".to_string()),
            images: None,
        };
        let v = serialize_message_for_jinja(&msg);

        // The parsed arguments object, as the template sees it, must
        // iterate in the echoed order. Without `preserve_order` this
        // would come back alphabetised.
        let parsed_args = &v["tool_calls"][0]["arguments"];
        let keys: Vec<&str> = parsed_args
            .as_object()
            .expect("arguments parsed into an object")
            .keys()
            .map(|k| k.as_str())
            .collect();
        assert_eq!(
            keys,
            vec!["path", "edits"],
            "arg-key order must match echoed JSON (preserve_order feature must be on)",
        );

        // miniJinja's `|items` has to iterate in insertion order so
        // the template's `<parameter=…>` blocks come out in echoed
        // order. This is orthogonal from serde_json's `preserve_order`
        // — miniJinja has its OWN `preserve_order` feature flag, and
        // without it the `serde_json::Value → minijinja::Value`
        // conversion still alphabetises. Both flags must be on.
        let it_tmpl = "{%- for k, _v in args|items -%}{{ k }}|{%- endfor -%}";
        let mut dbg_env = Environment::new();
        dbg_env.add_template("d", it_tmpl).unwrap();
        let dbg_out = dbg_env
            .get_template("d")
            .unwrap()
            .render(context! { args => parsed_args.clone() })
            .unwrap();
        assert_eq!(
            dbg_out, "path|edits|",
            "miniJinja must iterate args in insertion order (requires the `preserve_order` feature on the `minijinja` dependency)",
        );

        // Round-trip through the minimal gate slice + tojson: the
        // rendered prompt has to embed the args in that same order.
        // We wrap serialize_message_for_jinja in a throwaway template
        // that exercises the same `| tojson` that the real assistant
        // block uses for array-typed parameter values.
        let test_template = "{%- for msg in messages -%}\n{%- if msg.role == 'assistant' and msg.tool_calls -%}\n{%- for tc in msg.tool_calls -%}\n<function={{ tc.name }}>\n{%- for name, value in tc.arguments|items -%}\n<parameter={{ name }}>{% if value is mapping or (value is sequence and value is not string) %}{{ value | tojson }}{% else %}{{ value }}{% endif %}</parameter>\n{%- endfor -%}\n</function>\n{%- endfor -%}\n{%- endif -%}\n{%- endfor -%}";
        let mut rt = Environment::new();
        rt.add_filter("tojson", |value: minijinja::Value| -> String {
            serde_json::to_string(&value).unwrap_or_else(|_| "null".to_string())
        });
        rt.add_template("t", test_template).unwrap();
        let messages_value = vec![v.clone()];
        let rendered = rt
            .get_template("t")
            .unwrap()
            .render(context! { messages => messages_value })
            .unwrap();

        let path_idx = rendered.find("<parameter=path>").expect("path rendered");
        let edits_idx = rendered.find("<parameter=edits>").expect("edits rendered");
        assert!(
            path_idx < edits_idx,
            "path must render before edits (got path={path_idx}, edits={edits_idx}):\n{rendered}",
        );
    }

    /// Gemma4 template echoes `reasoning_content` inside a
    /// `<|channel>thought\n{thinking_text}\n<channel|>` block (see
    /// `.cache/models/gemma-4-*-mlx/chat_template.jinja` line 238). The
    /// label `thought\n` is hardcoded by the template, which means
    /// `reasoning_content` MUST carry only the body — NOT the label.
    ///
    /// Our Gemma4 output parser historically stored the full body incl.
    /// the `thought\n` prefix inside `thinking`. When pi-mono echoed
    /// that back verbatim as `reasoning_summary.text` → mapper
    /// coalesced it into `reasoning_content`, the template re-emitted
    /// `<|channel>thought\nthought\n{body}\n<channel|>` — a byte-level
    /// divergence from the cached prefix, zeroing `verify_cache_prefix`
    /// on every turn. Fix: strip the leading `thought\n` in the parser
    /// before saving to `thinking`. Guard that invariant here.
    ///
    /// Regression test for Gemma4 cache-reuse (always `cached_tokens=0`
    /// under pi-mono) — see `.logging-gemma/requests.ndjson` turns 2-7.
    #[test]
    fn gemma4_reasoning_echo_renders_byte_for_byte_with_model_generation() {
        let path = "/Users/brooklyn/workspace/github/mlx-node/.cache/models/gemma-4-26b-a4b-it-UD-Q8_K_XL-mlx/chat_template.jinja";
        let Ok(tmpl) = std::fs::read_to_string(path) else {
            // Skip when the fixture isn't checked out locally.
            return;
        };

        // What the MODEL originally emitted on turn 1 between the two
        // channel markers. This is the slice that ends up inside the
        // cache's KV state after the decode loop.
        let model_channel_body = "The user wants me to run ls.";
        let model_generated = format!(
            "<|channel>thought\n{model_channel_body}\n<channel|><|tool_call>call:bash{{command:<|\"|>ls<|\"|>}}<tool_call|>"
        );

        // Turn 2 echoes the parsed output back through the Responses
        // mapper. Simulate the coalesced ChatMessage shape it produces:
        // reasoning_content is whatever the parser returned, which
        // (after the fix) must be the body WITHOUT the `thought\n`
        // label — so when the Gemma4 template re-renders, it emits
        // exactly what the model originally generated.
        //
        // We test both directions: the bug shape (with `thought\n`
        // preserved) must produce a divergent render, and the fixed
        // shape (body only) must produce a byte-equal render.
        let parsed_via_bug = format!("thought\n{model_channel_body}");
        let parsed_via_fix = model_channel_body.to_string();

        let build_messages = |reasoning: &str| {
            vec![
                ChatMessage {
                    role: "user".to_string(),
                    content: "Run ls.".to_string(),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                    images: None,
                },
                ChatMessage {
                    role: "assistant".to_string(),
                    content: String::new(),
                    tool_calls: Some(vec![ToolCall {
                        id: Some("call_1".to_string()),
                        name: "bash".to_string(),
                        arguments: r#"{"command":"ls"}"#.to_string(),
                    }]),
                    tool_call_id: None,
                    reasoning_content: Some(reasoning.to_string()),
                    images: None,
                },
            ]
        };

        let render = |reasoning: &str| {
            Qwen3Tokenizer::render_chat_template_jinja2(
                &tmpl,
                &build_messages(reasoning),
                None,
                /*add_generation_prompt=*/ true,
                /*enable_thinking=*/ Some(true),
                "<bos>",
                "<eos>",
            )
            .unwrap()
        };

        let rendered_bug = render(&parsed_via_bug);
        let rendered_fix = render(&parsed_via_fix);

        // Bug shape: the template re-emits `thought\n` before the
        // echoed reasoning, producing DOUBLED `thought\n` in the
        // rendered channel block — NOT what the model generated.
        assert!(
            rendered_bug.contains("<|channel>thought\nthought\nThe user wants"),
            "bug shape should produce doubled `thought\\n` in rendered prompt:\n{rendered_bug}"
        );

        // Fixed shape: the template re-emits `thought\n` exactly once,
        // matching what the model generated during turn 1 decode. This
        // is the byte sequence that was saved to `cached_token_history`
        // (post-tokenization) and the byte sequence turn 2 must
        // re-produce in order for `verify_cache_prefix` to succeed.
        assert!(
            rendered_fix.contains(model_generated.as_str()),
            "fixed shape must re-render the model-generated slice byte-for-byte; \
             model generated:\n  {model_generated:?}\nrendered prompt was:\n{rendered_fix}"
        );
        assert!(
            !rendered_fix.contains("thought\nthought\n"),
            "fixed shape must NOT double `thought\\n`:\n{rendered_fix}"
        );
    }
}
