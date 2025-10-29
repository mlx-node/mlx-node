/**
 * Qwen3 Tokenizer Implementation using HuggingFace tokenizers
 *
 * Provides fast, production-ready tokenization for Qwen3 models with:
 * - BPE encoding/decoding
 * - Special token handling (EOS, BOS, PAD, etc.)
 * - ChatML format support
 * - Batch processing
 */
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Arc;
use tokenizers::{EncodeInput, Encoding, Tokenizer};

/// Special token IDs for Qwen3 models
const ENDOFTEXT_TOKEN_ID: u32 = 151643;
#[allow(dead_code)] // Reserved for future use (e.g., get_im_start_token_id())
const IM_START_TOKEN_ID: u32 = 151644;
const IM_END_TOKEN_ID: u32 = 151645;

/// Chat message role
#[napi(object)]
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Role: "system", "user", or "assistant"
    pub role: String,
    /// Message content
    pub content: String,
}

/// Qwen3 Tokenizer class with NAPI bindings
#[napi]
pub struct Qwen3Tokenizer {
    tokenizer: Arc<Tokenizer>,
    pad_token_id: u32,
    eos_token_id: u32,
    bos_token_id: Option<u32>,
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
                Ok(Self {
                    tokenizer: Arc::new(tokenizer),
                    pad_token_id: ENDOFTEXT_TOKEN_ID,
                    eos_token_id: IM_END_TOKEN_ID,
                    bos_token_id: None, // Qwen3 doesn't use BOS by default
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
    /// Formats messages using ChatML format:
    /// <|im_start|>role\ncontent<|im_end|>
    ///
    /// # Arguments
    /// * `messages` - Array of chat messages
    /// * `add_generation_prompt` - Whether to add assistant prompt at end (default: true)
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
    /// ```
    #[napi]
    pub fn apply_chat_template<'env>(
        &self,
        env: &'env Env,
        messages: Vec<ChatMessage>,
        add_generation_prompt: Option<bool>,
    ) -> Result<PromiseRaw<'env, Uint32ArraySlice<'env>>> {
        let add_prompt = add_generation_prompt.unwrap_or(true);
        let tokenizer = self.tokenizer.clone();

        env.spawn_future_with_callback(
            async move {
                napi::bindgen_prelude::spawn_blocking(move || {
                    // Build ChatML formatted string
                    let mut formatted: String = messages
                        .iter()
                        .map(|msg| format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content))
                        .collect();
                    // Add generation prompt for assistant
                    if add_prompt {
                        formatted.push_str("<|im_start|>assistant\n");
                    }

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
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            pad_token_id: ENDOFTEXT_TOKEN_ID,
            eos_token_id: IM_END_TOKEN_ID,
            bos_token_id: None, // Qwen3 doesn't use BOS by default
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
