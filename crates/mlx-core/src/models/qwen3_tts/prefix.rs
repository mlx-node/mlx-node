//! Model-thread-owned immutable instruction snapshots. No generated speech is cached.
use super::transformer::{AttentionState, Decoder};
use crate::array::MxArray;
use napi::{Error, Result};
use serde::Deserialize;
use std::{
    collections::VecDeque,
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CachePolicy {
    pub enabled: bool,
    pub max_bytes: u64,
    pub max_entries: usize,
}
impl Default for CachePolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            max_bytes: 64 * 1024 * 1024,
            max_entries: 8,
        }
    }
}
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoadOptions {
    pub instruction_cache: CachePolicy,
}

struct Entry {
    // The key includes the adapter prompt format; model/precision are instance scoped.
    format: u32,
    tokens: Vec<i32>,
    _embeddings: MxArray,
    states: Vec<AttentionState>,
    bytes: u64,
}
pub struct PrefixCache {
    policy: CachePolicy,
    entries: VecDeque<Entry>,
    bytes: u64,
}
pub fn fork(states: &[AttentionState]) -> Result<Vec<AttentionState>> {
    states
        .iter()
        .map(|s| {
            Ok(AttentionState {
                kv: s.kv.fork()?,
                position: s.position,
            })
        })
        .collect()
}
fn arrays(states: &[AttentionState]) -> Vec<&MxArray> {
    states
        .iter()
        .flat_map(|s| [s.kv.keys_ref(), s.kv.values_ref()].into_iter().flatten())
        .collect()
}
impl PrefixCache {
    pub fn new(policy: CachePolicy) -> Self {
        Self {
            policy,
            entries: VecDeque::new(),
            bytes: 0,
        }
    }
    pub fn enabled(&self) -> bool {
        self.policy.enabled && self.policy.max_bytes > 0 && self.policy.max_entries > 0
    }
    #[cfg(test)]
    pub(super) fn usage(&self) -> (usize, u64) {
        (self.entries.len(), self.bytes)
    }
    pub fn prepare(
        &mut self,
        tokens: &[i32],
        decoder: &Decoder,
        embed: impl FnOnce() -> Result<MxArray>,
        cancelled: &AtomicBool,
    ) -> Result<Vec<AttentionState>> {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::from_reason(
                "TTS cancelled before instruction prefill",
            ));
        }
        const FORMAT: u32 = 1;
        if let Some(index) = self
            .entries
            .iter()
            .position(|e| e.format == FORMAT && e.tokens == tokens)
        {
            // Fork before changing LRU bookkeeping: allocation failure must not
            // lose the entry while retaining its byte charge.
            let entry = self
                .entries
                .get(index)
                .ok_or_else(|| Error::from_reason("Instruction cache entry missing"))?;
            let state = fork(&entry.states)?;
            let entry = self
                .entries
                .remove(index)
                .ok_or_else(|| Error::from_reason("Instruction cache entry missing"))?;
            self.entries.push_back(entry);
            return Ok(state);
        }
        let embeddings = embed()?;
        let mut states = decoder.state();
        decoder.forward(&embeddings, &mut states)?.eval();
        let mut tensors = arrays(&states);
        tensors.push(&embeddings);
        MxArray::eval_arrays(&tensors)?;
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::from_reason(
                "TTS cancelled during instruction prefill",
            ));
        }
        // Charge allocated capacity, including unused KV rows, rather than token count.
        let bytes = tensors
            .iter()
            .try_fold(std::mem::size_of_val(tokens) as u64, |n, x| {
                Ok::<_, Error>(n + x.size()? * x.dtype()?.byte_size() as u64)
            })?;
        if self.enabled() && bytes <= self.policy.max_bytes {
            while self.entries.len() >= self.policy.max_entries
                || self.bytes > self.policy.max_bytes - bytes
            {
                if let Some(old) = self.entries.pop_front() {
                    self.bytes -= old.bytes;
                } else {
                    break;
                }
            }
            let continuation = fork(&states)?;
            self.entries.push_back(Entry {
                format: FORMAT,
                tokens: tokens.to_vec(),
                _embeddings: embeddings,
                states,
                bytes,
            });
            self.bytes += bytes;
            Ok(continuation)
        } else {
            Ok(states)
        }
    }
}
