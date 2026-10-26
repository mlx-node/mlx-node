//! Engram n-gram hash: the reference `engram.py` `NgramHashState.forward`
//! (same math as llama.cpp PR #30817 `engram_hash`). Not the qwen4 hash: ids
//! go through `token_map`, look-back is blocked at the sequence start and at
//! dead tokens, and blocked slots hold `token_map[pad]`.

use napi::{Error, Result};

use super::config::EngramConfig;

/// A raw id outside the token map hashes as a dead token (PR #30817).
const DEAD: u64 = u64::MAX;

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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::Value;

    use super::*;
    use crate::models::deepseek_v41::DeepseekV41Config;
    use crate::utils::gguf::{parse_gguf, parse_gguf_reader};

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/deepseek-v41")
            .join(name)
    }

    fn u32s(v: &Value) -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    }

    fn u64s(v: &Value) -> Vec<u64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect()
    }

    /// `hash_ids` is `[pos][layer][col]` of per-layer ids; checks every
    /// prefill / decode split of the sequence against it.
    fn check_sequence(h: &EngramHasher, ids: &[u32], want: &Value, splits: &[usize]) {
        let want: Vec<u64> = want
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|pos| pos.as_array().unwrap().iter().flat_map(u64s))
            .collect();
        let per_pos = h.n_layers() * h.n_cols();
        assert_eq!(want.len(), ids.len() * per_pos);
        let local = |rows: &[u32]| -> Vec<u64> {
            rows.iter()
                .enumerate()
                .map(|(i, &r)| u64::from(r) - h.layer_base((i / h.n_cols()) % h.n_layers()))
                .collect()
        };
        assert_eq!(local(&h.rows(ids, 0)), want, "prefill");
        for &split in splits {
            let mut got = local(&h.rows(&ids[..split], 0));
            for p in split..ids.len() {
                got.extend(local(&h.rows(&ids[..=p], p)));
            }
            assert_eq!(got, want, "prefill {split} + decode");
        }
    }

    /// Tiny config: `ops.json` reference hash ids of the main and the pad
    /// sequence, whole prefill and prefill + decode splits.
    #[test]
    fn tiny_hash_ids_match_reference() {
        let g = parse_gguf(fixture("tiny-q8_0.gguf")).unwrap();
        let c = DeepseekV41Config::from_gguf(&g).unwrap();
        let h = EngramHasher::new(c.engram.as_ref().unwrap()).unwrap();
        let ops: Value =
            serde_json::from_slice(&std::fs::read(fixture("ops.json")).unwrap()).unwrap();
        for case in ["main", "pad"] {
            let ids = u32s(&ops["hash_ids"][case]["ids"]);
            check_sequence(&h, &ids, &ops["hash_ids"][case]["hash_ids"], &[1, 12, 23]);
        }
        // a raw id outside the token map hashes like a dead token: its own
        // position is all pad, and the next position's look-back stops there
        let ids = u32s(&ops["hash_ids"]["main"]["ids"]);
        let mut bad = ids[..6].to_vec();
        bad[3] = c.vocab_size as u32;
        let pad = c.engram.as_ref().unwrap().pad_token_id as u32;
        assert_eq!(h.rows(&bad[..4], 3), h.rows(&[pad], 0));
        assert_eq!(h.rows(&bad[..5], 4), h.rows(&[bad[4]], 0));
        assert_ne!(h.rows(&ids[..5], 4), h.rows(&[ids[4]], 0));
    }

    /// Real file, shard 1 KV only (F_NOCACHE): the engram constants equal the
    /// reference's, computed from HF 1017 `config.json` + `tokenizer.json`
    /// (`DSV41_ENGRAM_CONSTANTS`), and the hash ids of 3 real sequences match.
    #[test]
    #[ignore = "real DeepSeek-V4.1 shard 1 KV; DSV41_DIR, DSV41_ENGRAM_CONSTANTS"]
    fn real_constants() {
        let dir = PathBuf::from(std::env::var("DSV41_DIR").unwrap());
        let want: Value = serde_json::from_slice(
            &std::fs::read(std::env::var("DSV41_ENGRAM_CONSTANTS").unwrap()).unwrap(),
        )
        .unwrap();
        let path = dir.join("DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf");
        let file = crate::models::deepseek_v41::tests::open_nocache(&path);
        let g = parse_gguf_reader(std::io::BufReader::new(file)).unwrap();
        let c = DeepseekV41Config::from_gguf(&g).unwrap();
        let e = c.engram.as_ref().unwrap();
        assert_eq!(e.token_map, u32s(&want["token_map"]), "token_map");
        assert_eq!(
            e.compressed_vocab_size as u64,
            want["compressed_vocab_size"].as_u64().unwrap()
        );
        assert_eq!(e.pad_token_id as u64, want["pad_id"].as_u64().unwrap());
        let layers: Vec<usize> = u64s(&want["layer_ids"])
            .into_iter()
            .map(|x| x as usize)
            .collect();
        assert_eq!(e.layers, layers);
        assert_eq!(e.max_ngram as u64, want["max_ngram_size"].as_u64().unwrap());
        assert_eq!(e.n_heads as u64, want["n_heads"].as_u64().unwrap());
        assert_eq!(e.head_dim as u64, want["head_dim"].as_u64().unwrap());
        let flat =
            |k: &str| -> Vec<u64> { want[k].as_array().unwrap().iter().flat_map(u64s).collect() };
        assert_eq!(e.multipliers, flat("multipliers"), "multipliers");
        assert_eq!(e.head_vocab_sizes, flat("primes"), "primes");
        let rows = u64s(&want["num_embeddings"]);
        let mut base = 0u64;
        let mut offsets = Vec::new();
        for (l, local) in want["offsets"].as_array().unwrap().iter().enumerate() {
            offsets.extend(u64s(local).into_iter().map(|o| o + base));
            assert_eq!(e.layer_rows(l), rows[l], "layer {l} rows");
            base += rows[l];
        }
        assert_eq!(e.head_offsets, offsets, "offsets");
        let h = EngramHasher::new(e).unwrap();
        let seqs = want["sequences"].as_array().unwrap();
        for s in seqs {
            let ids = u32s(&s["ids"]);
            check_sequence(&h, &ids, &s["hash_ids"], &[1, ids.len() / 2]);
        }
        println!(
            "real_constants: token_map {} entries, compressed {}, {} primes, {} sequences",
            e.token_map.len(),
            e.compressed_vocab_size,
            e.head_vocab_sizes.len(),
            seqs.len()
        );
    }
}
