//! Shared Engram n-gram configuration and hasher (used by `deepseek_v41` and
//! `hy_v4_flash`): the reference `engram.py` `NgramHashState.forward` (same
//! math as llama.cpp PR #30817 `engram_hash`). Ids go through `token_map`,
//! look-back is blocked at the sequence start and at dead tokens, and blocked
//! slots hold `token_map[pad]`.

use napi::{Error, Result};

/// A raw id outside the token map hashes as a dead token (PR #30817).
const DEAD: u64 = u64::MAX;

#[derive(Debug, Clone, PartialEq)]
pub struct EngramConfig {
    pub layers: Vec<usize>,
    pub max_ngram: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub multipliers: Vec<u64>,
    /// Global row offset of each (layer, n-gram order, head) column in the one
    /// GGUF table; `(max_ngram - 1) * n_heads` columns per engram layer.
    pub head_offsets: Vec<u64>,
    pub head_vocab_sizes: Vec<u64>,
    /// Raw token id; the hash uses `token_map[pad_token_id]`.
    pub pad_token_id: usize,
    pub token_map: Vec<u32>,
    pub compressed_vocab_size: usize,
    pub vocab_size: Option<u64>,
}

impl EngramConfig {
    pub fn n_cols(&self) -> usize {
        (self.max_ngram - 1) * self.n_heads
    }

    /// Table rows of engram layer `e` (the reference's `engram_num_embeddings`).
    pub fn layer_rows(&self, e: usize) -> u64 {
        let n = self.n_cols();
        self.head_vocab_sizes[e * n..(e + 1) * n].iter().sum()
    }
}

pub struct EngramHasher {
    n_layers: usize,
    max_ngram: usize,
    n_heads: usize,
    multipliers: Vec<u64>,
    primes: Vec<u64>,
    offsets: Vec<u64>,
    pad: u64,
    token_map: Vec<u32>,
}

impl EngramHasher {
    pub fn new(cfg: &EngramConfig) -> Result<Self> {
        let pad = *cfg
            .token_map
            .get(cfg.pad_token_id)
            .ok_or_else(|| Error::from_reason("engram pad id outside the token map"))?;
        Ok(Self {
            n_layers: cfg.layers.len(),
            max_ngram: cfg.max_ngram,
            n_heads: cfg.n_heads,
            multipliers: cfg.multipliers.clone(),
            primes: cfg.head_vocab_sizes.clone(),
            offsets: cfg.head_offsets.clone(),
            pad: u64::from(pad),
            token_map: cfg.token_map.clone(),
        })
    }

    pub fn n_cols(&self) -> usize {
        (self.max_ngram - 1) * self.n_heads
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// First table row of engram layer `e` in the one GGUF table.
    pub fn layer_base(&self, e: usize) -> u64 {
        self.offsets[e * self.n_cols()]
    }

    fn compressed(&self, t: u32) -> u64 {
        self.token_map
            .get(t as usize)
            .map_or(DEAD, |&c| u64::from(c))
    }

    /// Table rows for positions `start..tokens.len()`, where `tokens` is the
    /// whole token history of the sequence. Layout `[pos][engram layer][col]`,
    /// col = (n-gram order - 2) * n_heads + head; rows are global (the
    /// reference's per-layer id plus `layer_base`).
    pub fn rows(&self, tokens: &[u32], start: usize) -> Vec<u32> {
        let (n, cols) = (self.max_ngram, self.n_cols());
        let mut out = Vec::with_capacity(tokens.len().saturating_sub(start) * self.n_layers * cols);
        let mut toks = vec![0u64; n];
        for p in start..tokens.len() {
            let mut blocked = false;
            for (s, slot) in toks.iter_mut().enumerate() {
                let c = if p < s {
                    DEAD
                } else {
                    self.compressed(tokens[p - s])
                };
                blocked = blocked || c == DEAD;
                *slot = if blocked { self.pad } else { c };
            }
            for e in 0..self.n_layers {
                let mult = &self.multipliers[e * n..(e + 1) * n];
                let mut rolling = toks[0].wrapping_mul(mult[0]);
                for o in 1..n {
                    rolling ^= toks[o].wrapping_mul(mult[o]);
                    for h in 0..self.n_heads {
                        let col = e * cols + (o - 1) * self.n_heads + h;
                        out.push((rolling % self.primes[col] + self.offsets[col]) as u32);
                    }
                }
            }
        }
        out
    }
}
