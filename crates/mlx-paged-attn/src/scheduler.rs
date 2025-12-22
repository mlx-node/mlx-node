//! Continuous Batching Scheduler
//!
//! Enables dynamic batch composition for efficient LLM inference.
//! Sequences can be added and removed at any step, maximizing GPU utilization.
//!
//! ## Key Concepts
//!
//! - **Waiting Queue**: Requests waiting for memory allocation
//! - **Running List**: Active sequences being generated
//! - **Prefill vs Decode**: Prefill processes entire prompt, decode generates one token
//!
//! ## Example
//!
//! ```typescript
//! const scheduler = new ContinuousBatchingScheduler(cache, { maxBatchSize: 32 });
//!
//! // Add requests
//! scheduler.addRequest({ requestId: "1", promptTokens: [...], maxNewTokens: 100 });
//!
//! // Generation loop
//! while (!scheduler.isEmpty()) {
//!     const batch = scheduler.scheduleStep();
//!     if (!batch) break;
//!
//!     // Forward pass and sampling
//!     const outputs = model.forward(batch);
//!     scheduler.processOutputs(outputs);
//! }
//! ```

use std::collections::{HashMap, VecDeque};

use crate::paged_kv_cache::PagedKVCache;

/// A pending request waiting to be scheduled
#[derive(Clone, Debug)]
pub struct PendingRequest {
    /// Unique request identifier
    pub request_id: String,
    /// Prompt token IDs
    pub prompt_tokens: Vec<u32>,
    /// Maximum new tokens to generate
    pub max_new_tokens: u32,
    /// Optional: priority (higher = scheduled first)
    pub priority: Option<i32>,
}

/// An actively running sequence
#[derive(Clone)]
pub struct ActiveSequence {
    /// Sequence ID in the PagedKVCache
    pub seq_id: u32,
    /// Original request ID
    pub request_id: String,
    /// Tokens generated so far
    pub generated_tokens: Vec<u32>,
    /// Maximum tokens to generate
    pub max_new_tokens: u32,
    /// Number of prompt tokens
    pub prompt_len: u32,
    /// Whether this sequence is in prefill phase
    pub is_prefill: bool,
    /// Current position in generation
    pub position: u32,
}

/// Output from scheduling a step
#[derive(Clone, Debug)]
pub struct ScheduledBatch {
    /// Sequence IDs in this batch
    pub seq_ids: Vec<u32>,
    /// Request IDs corresponding to each sequence
    pub request_ids: Vec<String>,
    /// Input token IDs for each sequence (prompt for prefill, last token for decode)
    pub input_tokens: Vec<Vec<u32>>,
    /// Whether each sequence is in prefill phase
    pub is_prefill: Vec<bool>,
    /// Context lengths for each sequence
    pub context_lens: Vec<u32>,
    /// Total number of tokens in this batch (for memory planning)
    pub total_tokens: u32,
    /// Number of prefill sequences
    pub num_prefill: u32,
    /// Number of decode sequences
    pub num_decode: u32,
}

/// Result for a completed sequence
#[derive(Clone, Debug)]
pub struct CompletedSequence {
    /// Original request ID
    pub request_id: String,
    /// All generated tokens
    pub generated_tokens: Vec<u32>,
    /// Finish reason: "stop", "length", or "error"
    pub finish_reason: String,
    /// Total tokens (prompt + generated)
    pub total_tokens: u32,
}

/// Token output for processing
#[derive(Clone, Debug)]
pub struct TokenOutput {
    /// Sequence ID
    pub seq_id: u32,
    /// Generated token
    pub token: u32,
    /// Whether this token is an EOS token
    pub is_eos: bool,
}

/// Scheduler configuration
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    /// Maximum sequences in a batch
    pub max_batch_size: u32,
    /// Maximum tokens per scheduling step (prefill + decode combined)
    pub max_tokens_per_step: Option<u32>,
    /// Maximum number of prefill sequences per step (to balance latency)
    pub max_prefill_per_step: Option<u32>,
    /// Whether to prioritize decode over prefill (reduces latency for active sequences)
    pub prioritize_decode: Option<bool>,
    /// EOS token ID for stopping generation
    pub eos_token_id: Option<u32>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_tokens_per_step: Some(4096),
            max_prefill_per_step: Some(1),
            prioritize_decode: Some(true),
            eos_token_id: Some(151645), // Qwen3 default
        }
    }
}

/// Continuous Batching Scheduler
///
/// Manages dynamic batch composition for efficient LLM serving.
/// Handles request queuing, memory allocation, and sequence lifecycle.
pub struct ContinuousBatchingScheduler {
    /// Requests waiting for memory
    waiting: VecDeque<PendingRequest>,
    /// Active sequences
    running: HashMap<u32, ActiveSequence>,
    /// Completed sequences ready for retrieval
    completed: Vec<CompletedSequence>,
    /// Scheduler configuration
    config: SchedulerConfig,
    /// Block size for memory calculations
    block_size: u32,
    // NOTE: We no longer track next_seq_id here.
    // The seq_id is assigned by PagedKVCache::add_sequence() and returned to us.
    // This ensures the scheduler and cache seq_ids are always in sync.
}

impl ContinuousBatchingScheduler {
    /// Create a new scheduler
    ///
    /// # Arguments
    /// * `block_size` - Block size from PagedKVCache config
    /// * `config` - Scheduler configuration
    pub fn new(block_size: u32, config: Option<SchedulerConfig>) -> Self {
        let config = config.unwrap_or_default();
        Self {
            waiting: VecDeque::new(),
            running: HashMap::new(),
            completed: Vec::new(),
            config,
            block_size,
        }
    }

    /// Add a new request to the waiting queue
    pub fn add_request(&mut self, request: PendingRequest) {
        // Insert based on priority (higher priority first)
        let priority = request.priority.unwrap_or(0);
        let insert_pos = self
            .waiting
            .iter()
            .position(|r| r.priority.unwrap_or(0) < priority)
            .unwrap_or(self.waiting.len());
        self.waiting.insert(insert_pos, request);
    }

    /// Schedule the next step
    ///
    /// Returns a batch of sequences to process, or None if nothing to do.
    /// Allocates memory for new sequences from the waiting queue.
    ///
    /// # Arguments
    /// * `cache` - PagedKVCache for memory allocation
    pub fn schedule_step(&mut self, cache: &mut PagedKVCache) -> Option<ScheduledBatch> {
        // If nothing running and nothing waiting, we're done
        if self.running.is_empty() && self.waiting.is_empty() {
            return None;
        }

        let max_batch_size = self.config.max_batch_size as usize;
        let max_tokens = self.config.max_tokens_per_step.unwrap_or(4096) as usize;
        let max_prefill = self.config.max_prefill_per_step.unwrap_or(1) as usize;
        let prioritize_decode = self.config.prioritize_decode.unwrap_or(true);

        let mut batch_seq_ids = Vec::new();
        let mut batch_request_ids = Vec::new();
        let mut batch_input_tokens = Vec::new();
        let mut batch_is_prefill = Vec::new();
        let mut batch_context_lens = Vec::new();
        let mut total_tokens = 0usize;
        let mut num_prefill = 0u32;
        let mut num_decode = 0u32;

        // Step 1: Add decode sequences (if prioritizing decode)
        if prioritize_decode {
            for (seq_id, seq) in &self.running {
                if batch_seq_ids.len() >= max_batch_size {
                    break;
                }
                if !seq.is_prefill {
                    // Decode: single token input
                    if total_tokens < max_tokens {
                        batch_seq_ids.push(*seq_id);
                        batch_request_ids.push(seq.request_id.clone());
                        batch_input_tokens
                            .push(vec![seq.generated_tokens.last().copied().unwrap_or(0)]);
                        batch_is_prefill.push(false);
                        batch_context_lens.push(seq.position);
                        total_tokens += 1;
                        num_decode += 1;
                    }
                }
            }
        }

        // NOTE: We removed the "continuing prefill from running" path because:
        // 1. Prefill is now handled atomically in a single step by step_paged_generation
        // 2. After prefill completes, process_outputs() transitions is_prefill to false
        // 3. There's no scenario where a sequence stays in is_prefill=true across steps
        // If chunked prefill is needed in the future, implement it properly with
        // remaining_prompt_tokens tracking, not empty vec![] placeholders.

        // Step 2: Admit new requests from waiting queue
        while batch_seq_ids.len() < max_batch_size
            && (num_prefill as usize) < max_prefill
            && !self.waiting.is_empty()
        {
            let request = self.waiting.front().unwrap();
            let prompt_len = request.prompt_tokens.len() as u32;
            let blocks_needed = prompt_len.div_ceil(self.block_size);

            // Check if we have enough tokens budget
            if total_tokens + prompt_len as usize > max_tokens {
                break;
            }

            // Check if we can allocate memory
            if !cache.can_allocate(blocks_needed) {
                break;
            }

            // Allocate and add to running
            let request = self.waiting.pop_front().unwrap();

            // Use the seq_id returned by the cache to ensure scheduler and cache are in sync
            match cache.add_sequence(prompt_len) {
                Ok(seq_id) => {
                    let seq = ActiveSequence {
                        seq_id, // Use cache's seq_id, not our own counter
                        request_id: request.request_id.clone(),
                        generated_tokens: Vec::new(),
                        max_new_tokens: request.max_new_tokens,
                        prompt_len,
                        is_prefill: true,
                        position: prompt_len,
                    };

                    batch_seq_ids.push(seq_id);
                    batch_request_ids.push(request.request_id);
                    batch_input_tokens.push(request.prompt_tokens);
                    batch_is_prefill.push(true);
                    batch_context_lens.push(prompt_len);
                    total_tokens += prompt_len as usize;
                    num_prefill += 1;

                    self.running.insert(seq_id, seq);
                }
                Err(_) => {
                    // Cache allocation failed - put request back and stop admitting
                    self.waiting.push_front(request);
                    break;
                }
            }
        }

        // Step 3: Add remaining decode sequences (if not prioritizing decode)
        if !prioritize_decode {
            for (seq_id, seq) in &self.running {
                if batch_seq_ids.len() >= max_batch_size {
                    break;
                }
                if !seq.is_prefill && !batch_seq_ids.contains(seq_id) && total_tokens < max_tokens {
                    batch_seq_ids.push(*seq_id);
                    batch_request_ids.push(seq.request_id.clone());
                    batch_input_tokens
                        .push(vec![seq.generated_tokens.last().copied().unwrap_or(0)]);
                    batch_is_prefill.push(false);
                    batch_context_lens.push(seq.position);
                    total_tokens += 1;
                    num_decode += 1;
                }
            }
        }

        if batch_seq_ids.is_empty() {
            return None;
        }

        Some(ScheduledBatch {
            seq_ids: batch_seq_ids,
            request_ids: batch_request_ids,
            input_tokens: batch_input_tokens,
            is_prefill: batch_is_prefill,
            context_lens: batch_context_lens,
            total_tokens: total_tokens as u32,
            num_prefill,
            num_decode,
        })
    }

    /// Process token outputs from a forward pass
    ///
    /// Updates sequence state and handles completion.
    ///
    /// # Arguments
    /// * `outputs` - Token outputs for each sequence
    /// * `cache` - PagedKVCache for memory management
    pub fn process_outputs(
        &mut self,
        outputs: Vec<TokenOutput>,
        cache: &mut PagedKVCache,
    ) -> Result<(), String> {
        let eos_token = self.config.eos_token_id.unwrap_or(151645);

        for output in outputs {
            if let Some(seq) = self.running.get_mut(&output.seq_id) {
                // Transition from prefill to decode
                if seq.is_prefill {
                    seq.is_prefill = false;
                }

                // Add generated token
                seq.generated_tokens.push(output.token);
                seq.position += 1;

                // Extend cache for the new token
                cache.extend_sequence(output.seq_id, 1)?;

                // Check for completion
                let should_stop = output.is_eos
                    || output.token == eos_token
                    || seq.generated_tokens.len() >= seq.max_new_tokens as usize;

                if should_stop {
                    let finish_reason = if output.is_eos || output.token == eos_token {
                        "stop"
                    } else {
                        "length"
                    };

                    // Move to completed
                    let seq = self.running.remove(&output.seq_id).unwrap();
                    let gen_len = seq.generated_tokens.len() as u32;
                    self.completed.push(CompletedSequence {
                        request_id: seq.request_id,
                        generated_tokens: seq.generated_tokens,
                        finish_reason: finish_reason.to_string(),
                        total_tokens: seq.prompt_len + gen_len,
                    });

                    // Free memory
                    cache.remove_sequence(output.seq_id)?;
                }
            }
        }

        Ok(())
    }

    /// Get and clear completed sequences
    pub fn get_completed(&mut self) -> Vec<CompletedSequence> {
        std::mem::take(&mut self.completed)
    }

    /// Check if there's any work to do
    pub fn is_empty(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    /// Check if there are completed sequences to retrieve
    pub fn has_completed(&self) -> bool {
        !self.completed.is_empty()
    }

    /// Get number of waiting requests
    pub fn num_waiting(&self) -> u32 {
        self.waiting.len() as u32
    }

    /// Get number of running sequences
    pub fn num_running(&self) -> u32 {
        self.running.len() as u32
    }

    /// Get number of completed sequences
    pub fn num_completed(&self) -> u32 {
        self.completed.len() as u32
    }

    /// Abort a request by ID
    ///
    /// Removes from waiting queue or running list and frees memory.
    pub fn abort_request(
        &mut self,
        request_id: String,
        cache: &mut PagedKVCache,
    ) -> Result<bool, String> {
        // Check waiting queue
        if let Some(pos) = self.waiting.iter().position(|r| r.request_id == request_id) {
            self.waiting.remove(pos);
            return Ok(true);
        }

        // Check running list
        let seq_id = self
            .running
            .iter()
            .find(|(_, seq)| seq.request_id == request_id)
            .map(|(id, _)| *id);

        if let Some(seq_id) = seq_id {
            self.running.remove(&seq_id);
            cache.remove_sequence(seq_id)?;
            return Ok(true);
        }

        Ok(false)
    }

    /// Get statistics about current scheduler state
    pub fn get_stats(&self) -> SchedulerStats {
        let total_running_tokens: u32 = self.running.values().map(|seq| seq.position).sum();

        let prefill_count = self.running.values().filter(|seq| seq.is_prefill).count() as u32;

        SchedulerStats {
            num_waiting: self.waiting.len() as u32,
            num_running: self.running.len() as u32,
            num_completed: self.completed.len() as u32,
            num_prefill: prefill_count,
            num_decode: self.running.len() as u32 - prefill_count,
            total_running_tokens,
        }
    }
}

/// Scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    /// Number of requests in waiting queue
    pub num_waiting: u32,
    /// Number of running sequences
    pub num_running: u32,
    /// Number of completed sequences
    pub num_completed: u32,
    /// Number of sequences in prefill phase
    pub num_prefill: u32,
    /// Number of sequences in decode phase
    pub num_decode: u32,
    /// Total tokens across all running sequences
    pub total_running_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PagedAttentionConfig;

    fn create_test_cache() -> PagedKVCache {
        let config = PagedAttentionConfig {
            block_size: 16,
            gpu_memory_mb: 1024,
            head_size: 128,
            num_kv_heads: 4,
            num_layers: 2,
            ..Default::default()
        };
        PagedKVCache::new(config).unwrap()
    }

    #[test]
    fn test_scheduler_creation() {
        let scheduler = ContinuousBatchingScheduler::new(16, None);
        assert!(scheduler.is_empty());
        assert_eq!(scheduler.num_waiting(), 0);
        assert_eq!(scheduler.num_running(), 0);
    }

    #[test]
    fn test_add_request() {
        let mut scheduler = ContinuousBatchingScheduler::new(16, None);

        scheduler.add_request(PendingRequest {
            request_id: "req1".to_string(),
            prompt_tokens: vec![1, 2, 3, 4],
            max_new_tokens: 10,
            priority: None,
        });

        assert_eq!(scheduler.num_waiting(), 1);
        assert!(!scheduler.is_empty());
    }

    #[test]
    fn test_priority_ordering() {
        let mut scheduler = ContinuousBatchingScheduler::new(16, None);

        scheduler.add_request(PendingRequest {
            request_id: "low".to_string(),
            prompt_tokens: vec![1],
            max_new_tokens: 10,
            priority: Some(0),
        });

        scheduler.add_request(PendingRequest {
            request_id: "high".to_string(),
            prompt_tokens: vec![2],
            max_new_tokens: 10,
            priority: Some(10),
        });

        // High priority should be first
        let first = scheduler.waiting.front().unwrap();
        assert_eq!(first.request_id, "high");
    }

    #[test]
    fn test_schedule_step() {
        let mut cache = create_test_cache();
        let mut scheduler = ContinuousBatchingScheduler::new(16, None);

        scheduler.add_request(PendingRequest {
            request_id: "req1".to_string(),
            prompt_tokens: vec![1, 2, 3, 4],
            max_new_tokens: 10,
            priority: None,
        });

        let batch = scheduler.schedule_step(&mut cache);
        assert!(batch.is_some());

        let batch = batch.unwrap();
        assert_eq!(batch.seq_ids.len(), 1);
        assert_eq!(batch.num_prefill, 1);
        assert_eq!(batch.num_decode, 0);
        assert!(batch.is_prefill[0]);

        // Request should now be running
        assert_eq!(scheduler.num_waiting(), 0);
        assert_eq!(scheduler.num_running(), 1);
    }

    #[test]
    fn test_process_outputs() {
        let mut cache = create_test_cache();
        let mut scheduler = ContinuousBatchingScheduler::new(16, None);

        scheduler.add_request(PendingRequest {
            request_id: "req1".to_string(),
            prompt_tokens: vec![1, 2, 3, 4],
            max_new_tokens: 3,
            priority: None,
        });

        // Schedule and get batch
        let batch = scheduler.schedule_step(&mut cache).unwrap();
        let seq_id = batch.seq_ids[0];

        // Process first token (not EOS)
        scheduler
            .process_outputs(
                vec![TokenOutput {
                    seq_id,
                    token: 100,
                    is_eos: false,
                }],
                &mut cache,
            )
            .unwrap();

        assert_eq!(scheduler.num_running(), 1);
        assert!(!scheduler.running.get(&seq_id).unwrap().is_prefill);

        // Process more tokens until max
        scheduler
            .process_outputs(
                vec![TokenOutput {
                    seq_id,
                    token: 101,
                    is_eos: false,
                }],
                &mut cache,
            )
            .unwrap();

        scheduler
            .process_outputs(
                vec![TokenOutput {
                    seq_id,
                    token: 102,
                    is_eos: false,
                }],
                &mut cache,
            )
            .unwrap();

        // Should be completed (max_new_tokens = 3)
        assert_eq!(scheduler.num_running(), 0);
        assert_eq!(scheduler.num_completed(), 1);

        let completed = scheduler.get_completed();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].request_id, "req1");
        assert_eq!(completed[0].finish_reason, "length");
        assert_eq!(completed[0].generated_tokens, vec![100, 101, 102]);
    }

    #[test]
    fn test_eos_stops_generation() {
        let mut cache = create_test_cache();
        let config = SchedulerConfig {
            eos_token_id: Some(999),
            ..Default::default()
        };
        let mut scheduler = ContinuousBatchingScheduler::new(16, Some(config));

        scheduler.add_request(PendingRequest {
            request_id: "req1".to_string(),
            prompt_tokens: vec![1, 2, 3, 4],
            max_new_tokens: 100,
            priority: None,
        });

        let batch = scheduler.schedule_step(&mut cache).unwrap();
        let seq_id = batch.seq_ids[0];

        // Send EOS token
        scheduler
            .process_outputs(
                vec![TokenOutput {
                    seq_id,
                    token: 999,
                    is_eos: true,
                }],
                &mut cache,
            )
            .unwrap();

        // Should be completed immediately
        assert_eq!(scheduler.num_running(), 0);
        let completed = scheduler.get_completed();
        assert_eq!(completed[0].finish_reason, "stop");
    }

    #[test]
    fn test_abort_request() {
        let mut cache = create_test_cache();
        let mut scheduler = ContinuousBatchingScheduler::new(16, None);

        // Add two requests
        scheduler.add_request(PendingRequest {
            request_id: "req1".to_string(),
            prompt_tokens: vec![1, 2],
            max_new_tokens: 10,
            priority: None,
        });
        scheduler.add_request(PendingRequest {
            request_id: "req2".to_string(),
            prompt_tokens: vec![3, 4],
            max_new_tokens: 10,
            priority: None,
        });

        // Abort waiting request
        let aborted = scheduler
            .abort_request("req2".to_string(), &mut cache)
            .unwrap();
        assert!(aborted);
        assert_eq!(scheduler.num_waiting(), 1);

        // Schedule remaining request
        scheduler.schedule_step(&mut cache);
        assert_eq!(scheduler.num_running(), 1);

        // Abort running request
        let aborted = scheduler
            .abort_request("req1".to_string(), &mut cache)
            .unwrap();
        assert!(aborted);
        assert_eq!(scheduler.num_running(), 0);

        // Abort non-existent
        let aborted = scheduler
            .abort_request("req999".to_string(), &mut cache)
            .unwrap();
        assert!(!aborted);
    }

    #[test]
    fn test_multiple_sequences() {
        let mut cache = create_test_cache();
        let config = SchedulerConfig {
            max_batch_size: 4,
            max_prefill_per_step: Some(2),
            ..Default::default()
        };
        let mut scheduler = ContinuousBatchingScheduler::new(16, Some(config));

        // Add multiple requests
        for i in 0..3 {
            scheduler.add_request(PendingRequest {
                request_id: format!("req{}", i),
                prompt_tokens: vec![1, 2, 3, 4],
                max_new_tokens: 5,
                priority: None,
            });
        }

        // First step: should admit up to max_prefill_per_step
        let batch = scheduler.schedule_step(&mut cache).unwrap();
        assert_eq!(batch.num_prefill, 2); // Limited by max_prefill_per_step
        assert_eq!(scheduler.num_running(), 2);
        assert_eq!(scheduler.num_waiting(), 1);

        // Process outputs to transition to decode
        for &seq_id in &batch.seq_ids {
            scheduler
                .process_outputs(
                    vec![TokenOutput {
                        seq_id,
                        token: 100,
                        is_eos: false,
                    }],
                    &mut cache,
                )
                .unwrap();
        }

        // Second step: should include decode + new prefill
        let batch = scheduler.schedule_step(&mut cache).unwrap();
        assert!(batch.num_decode >= 2); // At least the two from before
        assert!(batch.num_prefill >= 1); // The remaining waiting request
    }
}
