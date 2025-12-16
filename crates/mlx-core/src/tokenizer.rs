/**
 * Qwen3 Tokenizer Implementation using HuggingFace tokenizers
 *
 * Provides fast, production-ready tokenization for Qwen3 models with:
 * - BPE encoding/decoding
 * - Special token handling (EOS, BOS, PAD, etc.)
 * - ChatML format support
 * - Batch processing
 * - Tool/function calling support with Jinja2 template rendering
 */
use minijinja::{Environment, context};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokenizers::{EncodeInput, Encoding, Tokenizer};

/// Special token IDs for Qwen3 models
const ENDOFTEXT_TOKEN_ID: u32 = 151643;
#[allow(dead_code)] // Reserved for future use (e.g., get_im_start_token_id())
const IM_START_TOKEN_ID: u32 = 151644;
const IM_END_TOKEN_ID: u32 = 151645;

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role: "system", "user", "assistant", or "tool"
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

                Ok(Self {
                    tokenizer: Arc::new(tokenizer),
                    pad_token_id: ENDOFTEXT_TOKEN_ID,
                    eos_token_id: IM_END_TOKEN_ID,
                    bos_token_id: None, // Qwen3 doesn't use BOS by default
                    chat_template,
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

    /// Load chat template from tokenizer_config.json file
    fn load_chat_template(tokenizer_path: &str) -> Option<String> {
        let path = Path::new(tokenizer_path);
        let config_path = path.parent()?.join("tokenizer_config.json");

        if !config_path.exists() {
            return None;
        }

        let config_content = std::fs::read_to_string(&config_path).ok()?;
        let config: serde_json::Value = serde_json::from_str(&config_content).ok()?;

        config
            .get("chat_template")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
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

        env.spawn_future_with_callback(
            async move {
                napi::bindgen_prelude::spawn_blocking(move || {
                    // Use Jinja2 rendering if tools provided or template exists
                    let formatted = if tools.is_some()
                        && let Some(chat_template) = chat_template
                    {
                        Self::render_chat_template_jinja2(
                            &chat_template,
                            &messages,
                            tools.as_deref(),
                            add_prompt,
                            enable_thinking,
                        )
                        .map_err(Error::from_reason)?
                    } else {
                        // Fallback to simple ChatML for backward compatibility
                        Self::format_chatml(&messages, add_prompt)
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

    /// Format messages using simple ChatML format (fallback when no template/tools)
    fn format_chatml(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        let mut formatted: String = messages
            .iter()
            .map(|msg| format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content))
            .collect();

        if add_generation_prompt {
            formatted.push_str("<|im_start|>assistant\n");
        }

        formatted
    }

    /// Render chat template using Jinja2 (minijinja)
    ///
    /// This uses the chat_template from tokenizer_config.json to render messages
    /// with full support for tools, thinking mode, and other Qwen3 features.
    fn render_chat_template_jinja2(
        template_str: &str,
        messages: &[ChatMessage],
        tools: Option<&[ToolDefinition]>,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> std::result::Result<String, String> {
        let mut env = Environment::new();

        // Add the tojson filter that Qwen3's template uses
        env.add_filter("tojson", |value: minijinja::Value| -> String {
            serde_json::to_string(&value).unwrap_or_else(|_| "null".to_string())
        });

        // Add Python-compatible string methods that Qwen3's template uses
        // These are called as methods on strings: content.startswith('prefix')
        env.set_unknown_method_callback(|_state, value, method, args| {
            // Only handle string methods
            if let Some(s) = value.as_str() {
                match method {
                    "startswith" => {
                        if let Some(prefix) = args.first().and_then(|v| v.as_str()) {
                            return Ok(minijinja::Value::from(s.starts_with(prefix)));
                        }
                        Err(minijinja::Error::new(
                            minijinja::ErrorKind::InvalidOperation,
                            "startswith requires a string argument",
                        ))
                    }
                    "endswith" => {
                        if let Some(suffix) = args.first().and_then(|v| v.as_str()) {
                            return Ok(minijinja::Value::from(s.ends_with(suffix)));
                        }
                        Err(minijinja::Error::new(
                            minijinja::ErrorKind::InvalidOperation,
                            "endswith requires a string argument",
                        ))
                    }
                    "strip" => {
                        // Python's strip() with optional chars argument
                        if let Some(chars) = args.first().and_then(|v| v.as_str()) {
                            Ok(minijinja::Value::from(
                                s.trim_matches(|c| chars.contains(c)),
                            ))
                        } else {
                            Ok(minijinja::Value::from(s.trim()))
                        }
                    }
                    "lstrip" => {
                        if let Some(chars) = args.first().and_then(|v| v.as_str()) {
                            Ok(minijinja::Value::from(
                                s.trim_start_matches(|c| chars.contains(c)),
                            ))
                        } else {
                            Ok(minijinja::Value::from(s.trim_start()))
                        }
                    }
                    "rstrip" => {
                        if let Some(chars) = args.first().and_then(|v| v.as_str()) {
                            Ok(minijinja::Value::from(
                                s.trim_end_matches(|c| chars.contains(c)),
                            ))
                        } else {
                            Ok(minijinja::Value::from(s.trim_end()))
                        }
                    }
                    "split" => {
                        // Python's split() - split by whitespace or delimiter
                        let parts: Vec<&str> =
                            if let Some(delim) = args.first().and_then(|v| v.as_str()) {
                                s.split(delim).collect()
                            } else {
                                s.split_whitespace().collect()
                            };
                        Ok(minijinja::Value::from(
                            parts
                                .into_iter()
                                .map(minijinja::Value::from)
                                .collect::<Vec<_>>(),
                        ))
                    }
                    _ => Err(minijinja::Error::new(
                        minijinja::ErrorKind::UnknownMethod,
                        format!("string has no method named {}", method),
                    )),
                }
            } else {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::UnknownMethod,
                    format!("{} has no method named {}", value.kind(), method),
                ))
            }
        });

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
                            if let Ok(props_val) = serde_json::from_str::<serde_json::Value>(props)
                            {
                                params_obj.insert("properties".to_string(), props_val);
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

        // Convert messages to JSON-serializable format
        let messages_value: Vec<serde_json::Value> = messages
            .iter()
            .map(|msg| {
                let mut obj = serde_json::Map::new();
                obj.insert("role".to_string(), serde_json::json!(msg.role));
                obj.insert("content".to_string(), serde_json::json!(msg.content));

                if let Some(tool_calls) = &msg.tool_calls {
                    let calls: Vec<serde_json::Value> = tool_calls
                        .iter()
                        .map(|tc| {
                            let mut call_obj = serde_json::Map::new();
                            if let Some(id) = &tc.id {
                                call_obj.insert("id".to_string(), serde_json::json!(id));
                            }
                            call_obj.insert("name".to_string(), serde_json::json!(tc.name));
                            call_obj
                                .insert("arguments".to_string(), serde_json::json!(tc.arguments));
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
            })
            .collect();

        // Build context for Jinja2 template
        // Note: enable_thinking defaults to true to allow model to think naturally.
        // Setting to false adds empty <think></think> tags which DISABLES thinking.
        let ctx = context! {
            messages => messages_value,
            tools => tools_value,
            add_generation_prompt => add_generation_prompt,
            enable_thinking => enable_thinking.unwrap_or(true),
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
    /// This is used by load_pretrained to load the tokenizer without async overhead.
    pub(crate) fn load_from_file_sync(tokenizer_path: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::from_reason(format!("Failed to load tokenizer: {}", e)))?;

        // Load chat template from tokenizer_config.json (in same directory)
        let chat_template = Self::load_chat_template(tokenizer_path);

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            pad_token_id: ENDOFTEXT_TOKEN_ID,
            eos_token_id: IM_END_TOKEN_ID,
            bos_token_id: None, // Qwen3 doesn't use BOS by default
            chat_template,
        })
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

        // Use Jinja2 rendering if tools provided or template exists
        let formatted = if tools.is_some()
            && let Some(chat_template) = &self.chat_template
        {
            Self::render_chat_template_jinja2(
                chat_template,
                messages,
                tools,
                add_prompt,
                enable_thinking,
            )
            .map_err(Error::from_reason)?
        } else {
            // Fallback to simple ChatML for backward compatibility
            Self::format_chatml(messages, add_prompt)
        };

        // Encode the formatted text (don't add extra special tokens)
        let encoding = Self::encode_internal(&self.tokenizer, formatted, Some(false))?;
        Ok(encoding.get_ids().to_vec())
    }
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
