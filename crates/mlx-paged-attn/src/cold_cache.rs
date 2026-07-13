//! Persistent SSD-backed cold tier for immutable PagedAttention prefix blocks.
//!
//! The hot allocator remains authoritative. This module stores only complete,
//! immutable blocks and restores them transactionally: bytes are validated and
//! uploaded into a reserved physical slot before the prefix is published.
//! Every I/O error is a cache miss, never an inference failure.

use std::collections::HashMap;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use safetensors::tensor::{Dtype, TensorView};
use safetensors::{SafeTensors, serialize};
use sha2::{Digest, Sha256};

use crate::{BlockAllocator, LayerKVPool, PhysicalBlock};

const CACHE_ABI: &str = "mlx-paged-v1";
const DEFAULT_QUEUE_DEPTH: usize = 8;
const GIB: u64 = 1024 * 1024 * 1024;
const MAX_DEFAULT_QUOTA: u64 = 100 * GIB;
const MIN_FREE_RESERVE: u64 = 5 * GIB;

/// Stable model/cache identity. Callers should hash exact weight shards plus
/// tokenizer/template, quantization, RoPE/MTP, and cache-layout components.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct ColdCacheFingerprint([u8; 32]);

impl ColdCacheFingerprint {
    /// Domain-separated SHA-256 over length-prefixed components.
    pub fn from_components<'a>(components: impl IntoIterator<Item = &'a [u8]>) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"mlx-node:cold-cache-fingerprint:v1\0");
        for component in components {
            hasher.update((component.len() as u64).to_le_bytes());
            hasher.update(component);
        }
        Self(hasher.finalize().into())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        hex_encode(&self.0)
    }
}

impl fmt::Debug for ColdCacheFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ColdCacheFingerprint")
            .field(&self.to_hex())
            .finish()
    }
}

/// Stable, collision-resistant chained key for one logical prefix block.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct ColdCacheKey([u8; 32]);

impl ColdCacheKey {
    /// Build a block key. `parent` is `None` for the first block and the
    /// preceding block key thereafter. Integer encoding is explicitly LE so
    /// the key is stable across processes and Rust versions.
    pub fn chain(
        fingerprint: ColdCacheFingerprint,
        parent: Option<Self>,
        tokens: &[u32],
        extra_keys: &[u64],
        cache_salt: u64,
        block_index: usize,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"mlx-node:cold-prefix-block:v1\0");
        hasher.update(fingerprint.as_bytes());
        hasher.update(parent.map_or([0u8; 32], |key| key.0));
        hasher.update((block_index as u64).to_le_bytes());
        hasher.update((tokens.len() as u64).to_le_bytes());
        for token in tokens {
            hasher.update(token.to_le_bytes());
        }
        hasher.update((extra_keys.len() as u64).to_le_bytes());
        for key in extra_keys {
            hasher.update(key.to_le_bytes());
        }
        // Match the hot-cache contract: salt isolates only block zero.
        hasher.update(if block_index == 0 { cache_salt } else { 0 }.to_le_bytes());
        Self(hasher.finalize().into())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        hex_encode(&self.0)
    }

    fn from_hex(value: &str) -> Option<Self> {
        let bytes = hex_decode_32(value)?;
        Some(Self(bytes))
    }
}

impl fmt::Debug for ColdCacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ColdCacheKey").field(&self.to_hex()).finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColdCacheLayout {
    pub block_size: u32,
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_size: u32,
    pub cache_dtype: String,
    pub key_bytes_per_layer: usize,
    pub value_bytes_per_layer: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColdLayerBlock {
    pub keys: Vec<u8>,
    pub values: Vec<u8>,
}

/// Owned host representation of one complete physical block across all layers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColdCacheBlock {
    pub key: ColdCacheKey,
    pub fingerprint: ColdCacheFingerprint,
    pub tokens: Vec<u32>,
    pub layout: ColdCacheLayout,
    pub layers: Vec<ColdLayerBlock>,
}

impl ColdCacheBlock {
    fn validate(&self) -> Result<(), String> {
        if self.tokens.len() != self.layout.block_size as usize {
            return Err("cold cache accepts immutable full blocks only".to_string());
        }
        if self.layers.len() != self.layout.num_layers as usize {
            return Err("cold-cache layer count does not match layout".to_string());
        }
        for layer in &self.layers {
            if layer.keys.len() != self.layout.key_bytes_per_layer
                || layer.values.len() != self.layout.value_bytes_per_layer
            {
                return Err("cold-cache layer byte length does not match layout".to_string());
            }
        }
        Ok(())
    }

    fn encoded_len(&self) -> u64 {
        self.layers
            .iter()
            .map(|layer| (layer.keys.len() + layer.values.len()) as u64)
            .sum::<u64>()
            + (self.tokens.len() * size_of::<u32>()) as u64
            + 4096
    }
}

#[derive(Clone, Debug)]
pub struct RestorePrefixIdentity {
    pub hot_hash: u64,
    pub tokens: Vec<u32>,
    pub parent_hot_hash: u64,
    pub extra_keys: Vec<u64>,
    pub cache_salt: u64,
    pub block_index: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ColdCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub enqueued: u64,
    pub queue_drops: u64,
    pub bytes_written: u64,
    pub bytes_restored: u64,
    pub evictions: u64,
    pub corruptions: u64,
}

#[derive(Default)]
struct AtomicStats {
    hits: AtomicU64,
    misses: AtomicU64,
    enqueued: AtomicU64,
    queue_drops: AtomicU64,
    bytes_written: AtomicU64,
    bytes_restored: AtomicU64,
    evictions: AtomicU64,
    corruptions: AtomicU64,
}

impl AtomicStats {
    fn snapshot(&self) -> ColdCacheStats {
        let load = |value: &AtomicU64| value.load(Ordering::Relaxed);
        ColdCacheStats {
            hits: load(&self.hits),
            misses: load(&self.misses),
            enqueued: load(&self.enqueued),
            queue_drops: load(&self.queue_drops),
            bytes_written: load(&self.bytes_written),
            bytes_restored: load(&self.bytes_restored),
            evictions: load(&self.evictions),
            corruptions: load(&self.corruptions),
        }
    }
}

#[derive(Clone, Debug)]
struct IndexEntry {
    path: PathBuf,
    size: u64,
    last_access: u128,
}

#[derive(Default)]
struct CacheIndex {
    entries: HashMap<ColdCacheKey, IndexEntry>,
    total_bytes: u64,
}

struct Shared {
    root: PathBuf,
    quota_bytes: u64,
    reserve_bytes: u64,
    index: Mutex<CacheIndex>,
    stats: AtomicStats,
}

struct WriteJob {
    block: ColdCacheBlock,
}

/// Bounded background SSD cache. Clones share one queue/index.
#[derive(Clone)]
pub struct ColdCacheManager {
    shared: Arc<Shared>,
    sender: SyncSender<WriteJob>,
}

impl ColdCacheManager {
    /// Open the automatic cache root (`~/.mlx-node/cache/paged/v1`) with a
    /// quota of 10% of filesystem capacity, capped at 100 GiB. At least 5%
    /// or 5 GiB (whichever is larger) remains reserved for the filesystem.
    pub fn open_default() -> Result<Self, String> {
        let home = std::env::var_os("HOME")
            .ok_or_else(|| "HOME is not set; cannot locate the paged cache".to_string())?;
        let root = PathBuf::from(home).join(".mlx-node/cache/paged/v1");
        fs::create_dir_all(&root).map_err(|e| format!("create cold-cache root: {e}"))?;
        set_private_dir_permissions(&root)?;
        let (total, _) = filesystem_space(&root)?;
        let quota = (total / 10).min(MAX_DEFAULT_QUOTA);
        let reserve = (total / 20).max(MIN_FREE_RESERVE);
        Self::open_at(root, quota, reserve, DEFAULT_QUEUE_DEPTH)
    }

    /// Explicit constructor used by tests and embedders with custom policy.
    pub fn open_at(
        root: PathBuf,
        quota_bytes: u64,
        reserve_bytes: u64,
        queue_depth: usize,
    ) -> Result<Self, String> {
        if quota_bytes == 0 || queue_depth == 0 {
            return Err("cold-cache quota and queue depth must be non-zero".to_string());
        }
        fs::create_dir_all(&root).map_err(|e| format!("create cold-cache root: {e}"))?;
        set_private_dir_permissions(&root)?;
        let index = rebuild_index(&root)?;
        let shared = Arc::new(Shared {
            root,
            quota_bytes,
            reserve_bytes,
            index: Mutex::new(index),
            stats: AtomicStats::default(),
        });
        let (sender, receiver) = mpsc::sync_channel::<WriteJob>(queue_depth);
        let worker_shared = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("mlx-paged-ssd-writer".to_string())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    // Fail-open: inference already has a valid hot block. A
                    // persistence error only means the next process recomputes.
                    let _ = persist_block(&worker_shared, &job.block);
                }
            })
            .map_err(|e| format!("spawn cold-cache writer: {e}"))?;
        Ok(Self { shared, sender })
    }

    pub fn root(&self) -> &Path {
        &self.shared.root
    }

    pub fn quota_bytes(&self) -> u64 {
        self.shared.quota_bytes
    }

    pub fn stats(&self) -> ColdCacheStats {
        self.shared.stats.snapshot()
    }

    /// Capture one pinned physical block from Metal, then enqueue only the
    /// owned host bytes. The writer thread never calls MLX/Metal and never
    /// holds the allocator lock.
    pub fn capture_and_enqueue(
        &self,
        pool: &LayerKVPool,
        block: &Arc<PhysicalBlock>,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
        tokens: &[u32],
    ) -> Result<bool, String> {
        if tokens.len() != pool.block_size() as usize {
            return Err("cold cache captures full blocks only".to_string());
        }

        // Logical pin prevents allocator eviction/reuse while Metal blits run.
        block.incref();
        let captured: Result<ColdCacheBlock, String> = (|| {
            let mut layers = Vec::with_capacity(pool.num_layers());
            for layer in 0..pool.num_layers() as u32 {
                let (keys, values) = pool.read_blocks_to_host(layer, &[block.block_id])?;
                layers.push(ColdLayerBlock { keys, values });
            }
            let first = layers
                .first()
                .ok_or_else(|| "cannot persist a pool with zero layers".to_string())?;
            let layout = ColdCacheLayout {
                block_size: pool.block_size(),
                num_layers: pool.num_layers() as u32,
                num_kv_heads: pool.config().num_kv_heads,
                head_size: pool.config().head_size,
                cache_dtype: format!("{:?}", pool.cache_dtype()),
                key_bytes_per_layer: first.keys.len(),
                value_bytes_per_layer: first.values.len(),
            };
            Ok(ColdCacheBlock {
                key,
                fingerprint,
                tokens: tokens.to_vec(),
                layout,
                layers,
            })
        })();
        let _ = block.decref();
        self.enqueue(captured?)
    }

    /// Non-blocking enqueue. A saturated queue deliberately drops the cold
    /// write so host buffers cannot grow without bound.
    pub fn enqueue(&self, block: ColdCacheBlock) -> Result<bool, String> {
        block.validate()?;
        match self.sender.try_send(WriteJob { block }) {
            Ok(()) => {
                self.shared.stats.enqueued.fetch_add(1, Ordering::Relaxed);
                Ok(true)
            }
            Err(TrySendError::Full(_)) => {
                self.shared
                    .stats
                    .queue_drops
                    .fetch_add(1, Ordering::Relaxed);
                Ok(false)
            }
            Err(TrySendError::Disconnected(_)) => Err("cold-cache writer stopped".to_string()),
        }
    }

    /// Load and validate a block. Corrupt/incompatible entries are removed
    /// and reported as misses.
    pub fn load(
        &self,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
    ) -> Option<ColdCacheBlock> {
        let path = self
            .shared
            .root
            .join(format!("{}.safetensors", key.to_hex()));
        let result = fs::read(&path)
            .map_err(|e| e.to_string())
            .and_then(|bytes| decode_block(&bytes, key, fingerprint));
        match result {
            Ok(block) => {
                self.shared.stats.hits.fetch_add(1, Ordering::Relaxed);
                self.shared
                    .stats
                    .bytes_restored
                    .fetch_add(block.encoded_len(), Ordering::Relaxed);
                // Startup rebuild derives recency from file mtime. Persist
                // every validated hit so a process restart preserves the
                // same LRU order instead of reverting to original write age.
                // Touch failure is deliberately fail-open: the block is
                // already validated and useful to inference; only future
                // eviction precision is affected.
                let touched_at = SystemTime::now();
                let _ = touch_file_recency(&path, touched_at);
                let touched_tick = touched_at
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos();
                if let Ok(mut index) = self.shared.index.lock()
                    && let Some(entry) = index.entries.get_mut(&key)
                {
                    entry.last_access = touched_tick;
                }
                Some(block)
            }
            Err(_) => {
                self.shared.stats.misses.fetch_add(1, Ordering::Relaxed);
                if path.exists() {
                    self.shared
                        .stats
                        .corruptions
                        .fetch_add(1, Ordering::Relaxed);
                    remove_indexed_file(&self.shared, key);
                }
                None
            }
        }
    }

    /// Restore one block transactionally. Returns `None` on every cold-tier
    /// failure so the caller can perform ordinary prefill.
    pub fn restore_block(
        &self,
        pool: &LayerKVPool,
        allocator: &Mutex<BlockAllocator>,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
        identity: &RestorePrefixIdentity,
    ) -> Option<Arc<PhysicalBlock>> {
        let cold = self.load(key, fingerprint)?;
        if cold.tokens != identity.tokens || !layout_matches_pool(&cold.layout, pool) {
            return None;
        }
        let block = allocator
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .allocate()?;

        for (layer_idx, layer) in cold.layers.iter().enumerate() {
            if pool
                .write_blocks_from_host(
                    layer_idx as u32,
                    &[block.block_id],
                    &layer.keys,
                    &layer.values,
                )
                .is_err()
            {
                allocator
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .free(Arc::clone(&block));
                return None;
            }
        }

        let published = allocator
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .publish_restored_prefix(
                Arc::clone(&block),
                identity.hot_hash,
                &identity.tokens,
                identity.parent_hot_hash,
                &identity.extra_keys,
                identity.cache_salt,
                identity.block_index,
            );
        match published {
            Ok(true) => Some(block),
            _ => {
                allocator
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .free(block);
                None
            }
        }
    }
}

fn layout_matches_pool(layout: &ColdCacheLayout, pool: &LayerKVPool) -> bool {
    layout.block_size == pool.block_size()
        && layout.num_layers as usize == pool.num_layers()
        && layout.num_kv_heads == pool.config().num_kv_heads
        && layout.head_size == pool.config().head_size
        && layout.cache_dtype == format!("{:?}", pool.cache_dtype())
}

fn persist_block(shared: &Shared, block: &ColdCacheBlock) -> Result<(), String> {
    block.validate()?;
    let bytes = encode_block(block)?;
    evict_for_write(shared, bytes.len() as u64)?;
    let destination = shared
        .root
        .join(format!("{}.safetensors", block.key.to_hex()));
    let temp = shared.root.join(format!(
        ".{}.{}.{}.tmp",
        block.key.to_hex(),
        std::process::id(),
        now_tick()
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temp)
        .map_err(|e| format!("create cold-cache temp file: {e}"))?;
    set_private_file_permissions(&temp)?;
    if let Err(error) = (|| -> Result<(), String> {
        file.write_all(&bytes)
            .map_err(|e| format!("write cold-cache file: {e}"))?;
        file.sync_all()
            .map_err(|e| format!("sync cold-cache file: {e}"))?;
        fs::rename(&temp, &destination).map_err(|e| format!("commit cold-cache file: {e}"))?;
        sync_directory(&shared.root)?;
        Ok(())
    })() {
        let _ = fs::remove_file(&temp);
        return Err(error);
    }

    let size = bytes.len() as u64;
    if let Ok(mut index) = shared.index.lock() {
        if let Some(old) = index.entries.insert(
            block.key,
            IndexEntry {
                path: destination,
                size,
                last_access: now_tick(),
            },
        ) {
            index.total_bytes = index.total_bytes.saturating_sub(old.size);
        }
        index.total_bytes = index.total_bytes.saturating_add(size);
    }
    shared
        .stats
        .bytes_written
        .fetch_add(size, Ordering::Relaxed);
    Ok(())
}

fn evict_for_write(shared: &Shared, incoming: u64) -> Result<(), String> {
    let (_, mut available) = filesystem_space(&shared.root)?;
    let mut index = shared
        .index
        .lock()
        .map_err(|_| "cold-cache index mutex poisoned".to_string())?;
    while index.total_bytes.saturating_add(incoming) > shared.quota_bytes
        || available < shared.reserve_bytes.saturating_add(incoming)
    {
        let Some((&key, _)) = index.entries.iter().min_by_key(|(_, e)| e.last_access) else {
            return Err("insufficient disk space for cold-cache write".to_string());
        };
        if let Some(entry) = index.entries.remove(&key) {
            let _ = fs::remove_file(&entry.path);
            index.total_bytes = index.total_bytes.saturating_sub(entry.size);
            available = available.saturating_add(entry.size);
            shared.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }
    Ok(())
}

fn remove_indexed_file(shared: &Shared, key: ColdCacheKey) {
    if let Ok(mut index) = shared.index.lock() {
        if let Some(entry) = index.entries.remove(&key) {
            let _ = fs::remove_file(&entry.path);
            index.total_bytes = index.total_bytes.saturating_sub(entry.size);
        } else {
            let _ = fs::remove_file(shared.root.join(format!("{}.safetensors", key.to_hex())));
        }
    }
}

fn encode_block(block: &ColdCacheBlock) -> Result<Vec<u8>, String> {
    let token_bytes: Vec<u8> = block.tokens.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut owned: Vec<(String, Vec<u8>)> = Vec::with_capacity(1 + block.layers.len() * 2);
    owned.push(("tokens".to_string(), token_bytes));
    for (i, layer) in block.layers.iter().enumerate() {
        owned.push((format!("layer.{i}.key"), layer.keys.clone()));
        owned.push((format!("layer.{i}.value"), layer.values.clone()));
    }
    let checksum = payload_checksum(&owned);
    let views: Result<Vec<_>, _> = owned
        .iter()
        .map(|(name, data)| {
            TensorView::new(Dtype::U8, vec![data.len()], data).map(|view| (name.as_str(), view))
        })
        .collect();
    let mut metadata = HashMap::new();
    metadata.insert("abi".to_string(), CACHE_ABI.to_string());
    metadata.insert("key".to_string(), block.key.to_hex());
    metadata.insert("fingerprint".to_string(), block.fingerprint.to_hex());
    metadata.insert("checksum".to_string(), checksum);
    metadata.insert(
        "block_size".to_string(),
        block.layout.block_size.to_string(),
    );
    metadata.insert(
        "num_layers".to_string(),
        block.layout.num_layers.to_string(),
    );
    metadata.insert(
        "num_kv_heads".to_string(),
        block.layout.num_kv_heads.to_string(),
    );
    metadata.insert("head_size".to_string(), block.layout.head_size.to_string());
    metadata.insert("cache_dtype".to_string(), block.layout.cache_dtype.clone());
    metadata.insert(
        "key_bytes".to_string(),
        block.layout.key_bytes_per_layer.to_string(),
    );
    metadata.insert(
        "value_bytes".to_string(),
        block.layout.value_bytes_per_layer.to_string(),
    );
    serialize(views.map_err(|e| e.to_string())?, Some(metadata)).map_err(|e| e.to_string())
}

fn decode_block(
    bytes: &[u8],
    expected_key: ColdCacheKey,
    expected_fingerprint: ColdCacheFingerprint,
) -> Result<ColdCacheBlock, String> {
    let (_, header) = SafeTensors::read_metadata(bytes).map_err(|e| e.to_string())?;
    let metadata = header
        .metadata()
        .as_ref()
        .ok_or_else(|| "cold-cache metadata missing".to_string())?;
    let tensors = SafeTensors::deserialize(bytes).map_err(|e| e.to_string())?;
    let get = |name: &str| {
        metadata
            .get(name)
            .cloned()
            .ok_or_else(|| format!("cold-cache metadata `{name}` missing"))
    };
    if get("abi")? != CACHE_ABI
        || get("key")? != expected_key.to_hex()
        || get("fingerprint")? != expected_fingerprint.to_hex()
    {
        return Err("cold-cache identity/ABI mismatch".to_string());
    }
    let parse = |name: &str| -> Result<u32, String> {
        get(name)?
            .parse::<u32>()
            .map_err(|_| format!("invalid cold-cache metadata `{name}`"))
    };
    let parse_usize = |name: &str| -> Result<usize, String> {
        get(name)?
            .parse::<usize>()
            .map_err(|_| format!("invalid cold-cache metadata `{name}`"))
    };
    let layout = ColdCacheLayout {
        block_size: parse("block_size")?,
        num_layers: parse("num_layers")?,
        num_kv_heads: parse("num_kv_heads")?,
        head_size: parse("head_size")?,
        cache_dtype: get("cache_dtype")?,
        key_bytes_per_layer: parse_usize("key_bytes")?,
        value_bytes_per_layer: parse_usize("value_bytes")?,
    };
    let token_data = tensors.tensor("tokens").map_err(|e| e.to_string())?;
    let token_bytes = token_data.data();
    if token_bytes.len() % 4 != 0 {
        return Err("cold-cache tokens have invalid byte length".to_string());
    }
    let tokens = token_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect();
    let mut layers = Vec::with_capacity(layout.num_layers as usize);
    for i in 0..layout.num_layers as usize {
        layers.push(ColdLayerBlock {
            keys: tensors
                .tensor(&format!("layer.{i}.key"))
                .map_err(|e| e.to_string())?
                .data()
                .to_vec(),
            values: tensors
                .tensor(&format!("layer.{i}.value"))
                .map_err(|e| e.to_string())?
                .data()
                .to_vec(),
        });
    }
    let block = ColdCacheBlock {
        key: expected_key,
        fingerprint: expected_fingerprint,
        tokens,
        layout,
        layers,
    };
    block.validate()?;

    let mut owned = Vec::with_capacity(1 + block.layers.len() * 2);
    owned.push((
        "tokens".to_string(),
        block.tokens.iter().flat_map(|v| v.to_le_bytes()).collect(),
    ));
    for (i, layer) in block.layers.iter().enumerate() {
        owned.push((format!("layer.{i}.key"), layer.keys.clone()));
        owned.push((format!("layer.{i}.value"), layer.values.clone()));
    }
    if payload_checksum(&owned) != get("checksum")? {
        return Err("cold-cache payload checksum mismatch".to_string());
    }
    Ok(block)
}

fn payload_checksum(tensors: &[(String, Vec<u8>)]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"mlx-node:cold-cache-payload:v1\0");
    for (name, data) in tensors {
        hasher.update((name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update((data.len() as u64).to_le_bytes());
        hasher.update(data);
    }
    hex_encode(&hasher.finalize())
}

fn rebuild_index(root: &Path) -> Result<CacheIndex, String> {
    let mut index = CacheIndex::default();
    for entry in fs::read_dir(root).map_err(|e| format!("scan cold-cache root: {e}"))? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.extension().and_then(|v| v.to_str()) != Some("safetensors") {
            if path.extension().and_then(|v| v.to_str()) == Some("tmp") {
                let _ = fs::remove_file(path);
            }
            continue;
        }
        let Some(key) = path
            .file_stem()
            .and_then(|v| v.to_str())
            .and_then(ColdCacheKey::from_hex)
        else {
            continue;
        };
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let size = metadata.len();
        let last_access = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |duration| duration.as_nanos());
        index.entries.insert(
            key,
            IndexEntry {
                path,
                size,
                last_access,
            },
        );
        index.total_bytes = index.total_bytes.saturating_add(size);
    }
    Ok(index)
}

#[cfg(unix)]
fn filesystem_space(path: &Path) -> Result<(u64, u64), String> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| "cold-cache path contains NUL".to_string())?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: c_path is NUL terminated and stats points to writable storage.
    if unsafe { libc::statvfs(c_path.as_ptr(), stats.as_mut_ptr()) } != 0 {
        return Err(format!(
            "statvfs cold-cache root: {}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: successful statvfs initialized the structure.
    let stats = unsafe { stats.assume_init() };
    let fragment = stats.f_frsize;
    Ok((
        (stats.f_blocks as u64).saturating_mul(fragment),
        (stats.f_bavail as u64).saturating_mul(fragment),
    ))
}

#[cfg(not(unix))]
fn filesystem_space(_path: &Path) -> Result<(u64, u64), String> {
    Err("automatic cold-cache quota requires a Unix statvfs implementation".to_string())
}

#[cfg(unix)]
fn set_private_dir_permissions(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .map_err(|e| format!("set cold-cache directory permissions: {e}"))
}

#[cfg(not(unix))]
fn set_private_dir_permissions(_path: &Path) -> Result<(), String> {
    Ok(())
}

#[cfg(unix)]
fn set_private_file_permissions(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
        .map_err(|e| format!("set cold-cache file permissions: {e}"))
}

#[cfg(not(unix))]
fn set_private_file_permissions(_path: &Path) -> Result<(), String> {
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|e| format!("sync cold-cache directory: {e}"))
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<(), String> {
    Ok(())
}

fn now_tick() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn touch_file_recency(path: &Path, modified: SystemTime) -> Result<(), String> {
    let file = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|e| format!("open cold-cache file for recency update: {e}"))?;
    file.set_times(std::fs::FileTimes::new().set_modified(modified))
        .map_err(|e| format!("persist cold-cache recency: {e}"))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn hex_decode_32(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64 {
        return None;
    }
    fn nibble(value: u8) -> Option<u8> {
        match value {
            b'0'..=b'9' => Some(value - b'0'),
            b'a'..=b'f' => Some(value - b'a' + 10),
            b'A'..=b'F' => Some(value - b'A' + 10),
            _ => None,
        }
    }
    let mut output = [0u8; 32];
    for (i, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        output[i] = nibble(pair[0])? << 4 | nibble(pair[1])?;
    }
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint() -> ColdCacheFingerprint {
        ColdCacheFingerprint::from_components([b"model".as_slice(), b"tokenizer".as_slice()])
    }

    fn block(key: ColdCacheKey) -> ColdCacheBlock {
        ColdCacheBlock {
            key,
            fingerprint: fingerprint(),
            tokens: vec![1, 2, 3, 4],
            layout: ColdCacheLayout {
                block_size: 4,
                num_layers: 2,
                num_kv_heads: 1,
                head_size: 2,
                cache_dtype: "BFloat16".to_string(),
                key_bytes_per_layer: 4,
                value_bytes_per_layer: 4,
            },
            layers: vec![
                ColdLayerBlock {
                    keys: vec![1, 2, 3, 4],
                    values: vec![5, 6, 7, 8],
                },
                ColdLayerBlock {
                    keys: vec![9, 10, 11, 12],
                    values: vec![13, 14, 15, 16],
                },
            ],
        }
    }

    fn temp_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "mlx-paged-cold-cache-{name}-{}-{}",
            std::process::id(),
            now_tick()
        ))
    }

    #[test]
    fn stable_chain_is_parent_and_fingerprint_sensitive() {
        let fp = fingerprint();
        let first = ColdCacheKey::chain(fp, None, &[1, 2, 3, 4], &[], 0, 0);
        assert_eq!(
            first,
            ColdCacheKey::chain(fp, None, &[1, 2, 3, 4], &[], 0, 0)
        );
        assert_ne!(
            first,
            ColdCacheKey::chain(fp, None, &[1, 2, 3, 5], &[], 0, 0)
        );
        assert_ne!(
            ColdCacheKey::chain(fp, Some(first), &[5, 6, 7, 8], &[], 0, 1),
            ColdCacheKey::chain(fp, None, &[5, 6, 7, 8], &[], 0, 1)
        );
    }

    #[test]
    fn safetensors_roundtrip_and_checksum() {
        let key = ColdCacheKey::chain(fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let original = block(key);
        let encoded = encode_block(&original).unwrap();
        let decoded = decode_block(&encoded, key, fingerprint()).unwrap();
        assert_eq!(decoded, original);

        let mut corrupt = encoded;
        *corrupt.last_mut().unwrap() ^= 0xff;
        assert!(decode_block(&corrupt, key, fingerprint()).is_err());
    }

    #[test]
    fn full_blocks_only() {
        let key = ColdCacheKey::chain(fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let mut partial = block(key);
        partial.tokens.pop();
        assert!(partial.validate().is_err());
    }

    #[test]
    fn writer_is_atomic_and_index_rebuilds() {
        let root = temp_root("roundtrip");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let key = ColdCacheKey::chain(fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let expected = block(key);
        assert!(manager.enqueue(expected.clone()).unwrap());

        let path = root.join(format!("{}.safetensors", key.to_hex()));
        for _ in 0..100 {
            if path.exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(manager.load(key, fingerprint()), Some(expected));
        drop(manager);

        let reopened = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        assert!(reopened.load(key, fingerprint()).is_some());
        assert_eq!(reopened.shared.index.lock().unwrap().entries.len(), 1);
        drop(reopened);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn restart_lru_uses_persisted_read_recency() {
        fn wait_for(path: &Path) {
            for _ in 0..200 {
                if path.exists() {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            panic!("timed out waiting for {}", path.display());
        }

        let root = temp_root("restart-lru");
        let fp = fingerprint();
        let key_a = ColdCacheKey::chain(fp, None, &[1, 2, 3, 4], &[1], 0, 0);
        let key_b = ColdCacheKey::chain(fp, None, &[1, 2, 3, 4], &[2], 0, 0);
        let key_c = ColdCacheKey::chain(fp, None, &[1, 2, 3, 4], &[3], 0, 0);
        let path_a = root.join(format!("{}.safetensors", key_a.to_hex()));
        let path_b = root.join(format!("{}.safetensors", key_b.to_hex()));
        let path_c = root.join(format!("{}.safetensors", key_c.to_hex()));

        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        manager.enqueue(block(key_a)).unwrap();
        wait_for(&path_a);
        // Keep write mtimes strictly ordered even on coarse filesystems.
        std::thread::sleep(std::time::Duration::from_millis(20));
        manager.enqueue(block(key_b)).unwrap();
        wait_for(&path_b);
        std::thread::sleep(std::time::Duration::from_millis(20));

        // A was written first but read last. The hit must persist that fact
        // in mtime so a new manager evicts B before A.
        assert!(manager.load(key_a, fp).is_some());
        let size_a = fs::metadata(&path_a).unwrap().len();
        let size_b = fs::metadata(&path_b).unwrap().len();
        drop(manager);
        std::thread::sleep(std::time::Duration::from_millis(10));

        let reopened = ColdCacheManager::open_at(root.clone(), size_a + size_b, 0, 1).unwrap();
        reopened.enqueue(block(key_c)).unwrap();
        wait_for(&path_c);
        // The writer updates the index immediately after rename; wait for the
        // old-file removal/index commit to be visible too.
        for _ in 0..200 {
            if path_a.exists() && !path_b.exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(path_a.exists(), "recently read A must survive restart LRU");
        assert!(
            !path_b.exists(),
            "older unread B must be evicted after restart"
        );
        assert!(path_c.exists());

        drop(reopened);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn corrupt_file_fails_open_and_is_removed() {
        let root = temp_root("corrupt");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 1).unwrap();
        let key = ColdCacheKey::chain(fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let path = root.join(format!("{}.safetensors", key.to_hex()));
        fs::write(&path, b"not a safetensors file").unwrap();
        assert!(manager.load(key, fingerprint()).is_none());
        assert!(!path.exists());
        assert_eq!(manager.stats().corruptions, 1);
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn transactional_restore_uploads_then_publishes() {
        use crate::metal::MetalDtype;
        use crate::{PagedAttentionConfig, hash_tokens};

        let config = PagedAttentionConfig {
            block_size: 8,
            gpu_memory_mb: 256,
            head_size: 64,
            num_kv_heads: 1,
            num_layers: 1,
            use_fp8_cache: Some(false),
            max_seq_len: Some(32),
            max_batch_size: Some(1),
        };
        let pool = match LayerKVPool::new(config, 2, MetalDtype::BFloat16) {
            Ok(pool) => pool,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping transactional_restore_uploads_then_publishes: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        let allocator = Mutex::new(BlockAllocator::new(2, 8));
        let source = allocator.lock().unwrap().allocate().unwrap();
        let bytes_per_side = 64 * 8 * 2;
        let keys: Vec<u8> = (0..bytes_per_side).map(|i| (i % 251) as u8).collect();
        let values: Vec<u8> = (0..bytes_per_side)
            .map(|i| (250 - (i % 251)) as u8)
            .collect();
        pool.write_blocks_from_host(0, &[source.block_id], &keys, &values)
            .unwrap();

        let root = temp_root("restore");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let key = ColdCacheKey::chain(fingerprint(), None, &tokens, &[], 0, 0);
        assert!(
            manager
                .capture_and_enqueue(&pool, &source, key, fingerprint(), &tokens)
                .unwrap()
        );
        let path = root.join(format!("{}.safetensors", key.to_hex()));
        for _ in 0..100 {
            if path.exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        allocator.lock().unwrap().free(source);

        let identity = RestorePrefixIdentity {
            hot_hash: hash_tokens(&tokens, 0, &[]),
            tokens: tokens.clone(),
            parent_hot_hash: 0,
            extra_keys: vec![],
            cache_salt: 0,
            block_index: 0,
        };
        let restored = manager
            .restore_block(&pool, &allocator, key, fingerprint(), &identity)
            .expect("cold block restore");
        let (restored_keys, restored_values) =
            pool.read_blocks_to_host(0, &[restored.block_id]).unwrap();
        assert_eq!(restored_keys, keys);
        assert_eq!(restored_values, values);

        let (hits, hit_tokens) =
            allocator
                .lock()
                .unwrap()
                .find_longest_cache_hit(&tokens, 8, &[], 0);
        assert_eq!(hit_tokens, 8, "publish must happen after complete upload");
        assert_eq!(hits[0].block_id, restored.block_id);
        {
            let mut allocator = allocator.lock().unwrap();
            allocator.free(restored);
            for hit in hits {
                allocator.free(hit);
            }
        }
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }
}
