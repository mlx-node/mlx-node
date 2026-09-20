/// Prefix-cache identity supplied to shared lookup and publication walks.
///
/// Text-only callers use one uniform key vector, while multimodal callers use
/// image-aware keys for each block.
#[derive(Debug, Clone, Copy)]
pub enum PrefixKeys<'a> {
    Uniform(&'a [u64]),
    PerBlock(&'a [Vec<u64>]),
}

impl<'a> PrefixKeys<'a> {
    pub fn get(self, block_index: usize) -> Option<&'a [u64]> {
        match self {
            Self::Uniform(keys) => Some(keys),
            Self::PerBlock(keys) => keys.get(block_index).map(Vec::as_slice),
        }
    }

    pub fn chain_hashes(self, token_ids: &[u32], block_size: u32, cache_salt: u64) -> Vec<u64> {
        if block_size == 0 {
            return Vec::new();
        }
        let block_size_us = block_size as usize;
        let mut num_full_blocks = token_ids.len() / block_size_us;
        if let Self::PerBlock(keys) = self {
            num_full_blocks = num_full_blocks.min(keys.len());
        }
        let mut hashes = Vec::with_capacity(num_full_blocks);
        let mut parent_hash = 0;
        match self {
            Self::Uniform(keys) => {
                for (n, tokens) in token_ids
                    .chunks_exact(block_size_us)
                    .take(num_full_blocks)
                    .enumerate()
                {
                    let hash = hash_block(tokens, parent_hash, keys, cache_salt, n);
                    hashes.push(hash);
                    parent_hash = hash;
                }
            }
            Self::PerBlock(keys) => {
                // Zip stops at the shorter of the two inputs, which is exactly
                // the `min(num_full_blocks, keys.len())` bound computed above.
                for (n, (tokens, block_keys)) in token_ids
                    .chunks_exact(block_size_us)
                    .zip(keys.iter())
                    .enumerate()
                {
                    let hash = hash_block(tokens, parent_hash, block_keys, cache_salt, n);
                    hashes.push(hash);
                    parent_hash = hash;
                }
            }
        }
        hashes
    }
}

/// Hash function for token sequences (for prefix caching).
///
/// Computes a chained block hash in vLLM's style: feeds `parent_hash` first,
/// then each token id in order, then each entry of `extra_keys` in order.
///
/// `extra_keys` is reserved for per-block side-channel information that must
/// participate in cache identity — image content hashes, cache-salt, LoRA
/// names, etc. (see vLLM commit 269bf46d). Order matters: `[a, b]` and
/// `[b, a]` produce different hashes. Most callers should pass `&[]`.
///
/// Uses Rust's `DefaultHasher` (SipHash-1-3). vLLM uses xxhash/sha256 for
/// cross-process determinism, but our prefix cache is process-local — every
/// hash is computed and consumed in the same process — so SipHash's stronger
/// collision resistance is the better trade-off and we don't need stable
/// hashes across runs.
///
// FIXME: SipHash u64 is not cryptographically collision-resistant.
// `crate::BlockAllocator::find_longest_cache_hit` walks chained block hashes via
// `lookup_prefix`, so a mid-chain collision between two different token chains
// could cause a mixed-prefix lookup for entries registered without block
// identity metadata. `cache_full_blocks`/`cache_full_blocks_per_block` entries
// are verified on lookup and duplicate registration, but direct
// `register_prefix` callers still have no token/extra-key metadata. See the
// `block_allocator` module-level "SipHash collision limitation" docs.
pub fn hash_tokens(tokens: &[u32], parent_hash: u64, extra_keys: &[u64]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    parent_hash.hash(&mut hasher);
    for &token in tokens {
        token.hash(&mut hasher);
    }
    for &key in extra_keys {
        key.hash(&mut hasher);
    }
    hasher.finish()
}

/// Per-block hash helper shared by prefix registry lookup and publication.
///
/// Mixes `cache_salt` into block 0's hash only (when `cache_salt != 0` and
/// `block_index == 0`), matching vLLM's first-block-only `cache_salt`
/// composition (`vllm/v1/core/kv_cache_utils.py:521-531`):
///
/// ```text
/// cache_salt_keys = [cache_salt] if start_token_idx == 0 and cache_salt else []
/// extra_keys      = ... + cache_salt_keys + ...
/// ```
///
/// When `cache_salt == 0` OR `block_index > 0`, this collapses to
/// `hash_tokens(tokens, parent_hash, extra_keys)` byte-for-byte — no extra
/// allocation, no salt mixed in. The leading-block + non-zero-salt branch
/// drives the `Hasher` directly (`parent_hash`, every token, every
/// `extra_keys` entry, then `cache_salt`) instead of materializing an
/// `extra_keys` + `[cache_salt]` slice; the result is bit-equal to
/// `hash_tokens(tokens, parent_hash, &[extra_keys..., cache_salt])` but
/// avoids the heap allocation that path used to do. Ordering matches vLLM:
/// `cache_salt` is hashed AFTER the existing `extra_keys` entries.
#[inline]
pub(crate) fn hash_block(
    tokens: &[u32],
    parent_hash: u64,
    extra_keys: &[u64],
    cache_salt: u64,
    block_index: usize,
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    if cache_salt == 0 || block_index != 0 {
        return hash_tokens(tokens, parent_hash, extra_keys);
    }
    // First block + non-zero salt → drive the hasher directly. Bit-equal to
    // `hash_tokens(tokens, parent_hash, [extra_keys..., cache_salt])` because
    // `hash_tokens` writes the same prefix in the same order, then iterates
    // its `extra_keys` slice; appending `cache_salt` here matches that.
    let mut hasher = DefaultHasher::new();
    parent_hash.hash(&mut hasher);
    for &token in tokens {
        token.hash(&mut hasher);
    }
    for &key in extra_keys {
        key.hash(&mut hasher);
    }
    cache_salt.hash(&mut hasher);
    hasher.finish()
}

/// Chained per-block hot hashes for the leading full blocks of `token_ids`.
///
/// This is the single source for the per-block chain-hash walk shared by hot
/// lookup ([`crate::BlockAllocator::find_longest_cache_hit`]) and cold-tier
/// restore: block 0's parent hash seeds at `0`, every later block chains off its
/// predecessor's hash, and `cache_salt` is mixed into block 0 only (via
/// [`hash_block`]). Entry `n` is the hash the allocator looks up or registers
/// for full block `n`, so `chain_hashes(...)[n]` is the `hot_hash` and
/// `chain_hashes(...)[n - 1]` the `parent_hot_hash` a restore must supply.
///
/// Returns one hash per full block (`token_ids.len() / block_size`); an empty
/// vec when `block_size == 0` or fewer than one full block is present. Only
/// the leading `num_full_blocks * block_size` tokens participate; trailing
/// partial-block tokens are ignored.
pub fn chain_hashes(
    token_ids: &[u32],
    block_size: u32,
    extra_keys: &[u64],
    cache_salt: u64,
) -> Vec<u64> {
    PrefixKeys::Uniform(extra_keys).chain_hashes(token_ids, block_size, cache_salt)
}

/// Per-block-`extra_keys` variant of [`chain_hashes`].
///
/// Block `n` is hashed with `extra_keys_per_block[n]`; everything else matches
/// [`chain_hashes`] exactly (parent seeded at `0`, later blocks chain off their
/// predecessor, `cache_salt` mixed into block 0 only via [`hash_block`]).
///
/// The walk STOPS at the first block index without per-block keys, so the
/// returned vec has `min(token_ids.len() / block_size, extra_keys_per_block.len())`
/// entries. Callers must treat a short result as "no cache identity beyond this
/// point" — that is exactly what
/// [`crate::BlockAllocator::find_longest_cache_hit_per_block`] does (break →
/// miss), and what the cold-tier restore walk must do as well.
///
/// With an all-empty per-block vec of full length, the result is bit-equal to
/// `chain_hashes(token_ids, block_size, &[], cache_salt)`.
pub fn chain_hashes_per_block(
    token_ids: &[u32],
    block_size: u32,
    extra_keys_per_block: &[Vec<u64>],
    cache_salt: u64,
) -> Vec<u64> {
    PrefixKeys::PerBlock(extra_keys_per_block).chain_hashes(token_ids, block_size, cache_salt)
}
