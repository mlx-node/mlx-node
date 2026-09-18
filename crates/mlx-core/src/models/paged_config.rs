//! Shared block-paged KV cache options and raw `config.json` parsing.
//!
//! NAPI configs keep flat fields to preserve their TypeScript shape. Manual
//! loaders use the shared parsers without changing family-specific defaults;
//! Muse's serde-only `RawConfig` embeds this type with `#[serde(flatten)]`.

use serde_json::Value;

/// Default `paged_block_size` (tokens per block): every family's read site
/// resolves an unset field as `unwrap_or(16)`.
pub const DEFAULT_PAGED_BLOCK_SIZE: u32 = 16;

/// The historical fixed `paged_cache_memory_mb` default (2 GiB), used by the
/// qwen3 / lfm2 / k2_horizon `unwrap_or(2048)` read sites. Families with
/// context-aware auto sizing (qwen3_5 dense/MoE, gemma4, muse_glimmer,
/// nemotron_h) resolve their own per-checkpoint default instead.
pub const DEFAULT_PAGED_CACHE_MEMORY_MB: u32 = 2048;

/// The flat block-paged KV cache options shared by every paged-capable model
/// family's `config.json` surface.
///
/// All fields are optional at the serde layer and presence is load-bearing —
/// `None` means "not configured", not "default value" — so keep resolving
/// through each family's own defaults.
///
/// Serde notes for future `#[serde(flatten)]` adopters: every field carries
/// `#[serde(default)]`, so the flattened struct never fails on absent keys,
/// and `Serialize` emits the keys inline at the flatten position (including
/// `null`s for `None`, matching the per-family declarations).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PagedCacheConfig {
    /// GPU memory budget for the paged KV pool in megabytes. Only consumed
    /// when `use_block_paged_cache` resolves on.
    #[serde(default)]
    pub paged_cache_memory_mb: Option<u32>,

    /// Initial paged KV pool size in MiB for the grow-on-demand pool: the
    /// pool starts at this size and grows toward the max budget
    /// (`paged_cache_memory_mb` or the family auto default) on exhaustion.
    /// `None` keeps the historical initial == max behavior. Only the qwen3_5
    /// dense/MoE configs and loaders consume this key today; other families
    /// leave it `None`.
    #[serde(default)]
    pub paged_cache_initial_memory_mb: Option<u32>,

    /// Paged block size in tokens. Universally read as `unwrap_or(16)`
    /// ([`DEFAULT_PAGED_BLOCK_SIZE`]).
    #[serde(default)]
    pub paged_block_size: Option<u32>,

    /// Opt in/out of the block-paged KV adapter (`PagedKVCacheAdapter`).
    /// `None` defers to each family's load-time default (paged for every
    /// production family today); `Some(false)` selects the flat cache path.
    #[serde(default)]
    pub use_block_paged_cache: Option<bool>,

    /// Persist the family's paged cold-tier state (KV blocks plus per-family
    /// sidecars) to the SSD cold tier so warm prefixes survive restarts.
    /// Resolved against `MLX_PERSIST_PAGED_CACHE` by
    /// `crate::cold_tier::resolve_persist_cold`.
    #[serde(default)]
    pub persist_paged_cache: Option<bool>,
}

impl PagedCacheConfig {
    /// Shared `Some(explicit.unwrap_or(default))` policy behind each family's
    /// `resolve_use_block_paged_default`: an explicit `config.json` value
    /// always wins; absent resolves to the family default.
    pub fn resolve_use_paged_default(explicit: Option<bool>, default: bool) -> Option<bool> {
        Some(explicit.unwrap_or(default))
    }

    /// Parse the five keys from a raw `config.json` value.
    ///
    /// Mirrors the hand-rolled
    /// `raw.get(..).and_then(Value::as_u64).map(|v| v as u32)` / `as_bool`
    /// reads the manual-parse families use (gemma4, qwen3_5, qwen3_5_moe,
    /// nemotron_h): snake_case keys only, `as_u64` → `as u32` truncation on
    /// oversized numbers, `as_bool` strictness (non-bool → `None`).
    pub fn from_raw_json(raw: &Value) -> Self {
        Self {
            paged_cache_memory_mb: raw
                .get("paged_cache_memory_mb")
                .and_then(Value::as_u64)
                .map(|v| v as u32),
            paged_cache_initial_memory_mb: raw
                .get("paged_cache_initial_memory_mb")
                .and_then(Value::as_u64)
                .map(|v| v as u32),
            paged_block_size: raw
                .get("paged_block_size")
                .and_then(Value::as_u64)
                .map(|v| v as u32),
            use_block_paged_cache: raw.get("use_block_paged_cache").and_then(Value::as_bool),
            persist_paged_cache: raw.get("persist_paged_cache").and_then(Value::as_bool),
        }
    }

    /// qwen3's dual-name variant of [`Self::from_raw_json`]: the snake_case
    /// key wins, then the camelCase alias, and numbers read via `as_i64`
    /// (not `as_u64`) before the `as u32` cast. The two conventions are NOT
    /// interchangeable — `as_i64` maps `-1` to `u32::MAX` where `as_u64`
    /// yields `None` — so each manual-parse site keeps the variant it already
    /// used.
    pub fn from_raw_json_camel_aliases(raw: &Value) -> Self {
        let number = |snake: &str, camel: &str| {
            raw.get(snake)
                .and_then(Value::as_i64)
                .or_else(|| raw.get(camel).and_then(Value::as_i64))
                .map(|v| v as u32)
        };
        let boolean = |snake: &str, camel: &str| {
            raw.get(snake)
                .and_then(Value::as_bool)
                .or_else(|| raw.get(camel).and_then(Value::as_bool))
        };
        Self {
            paged_cache_memory_mb: number("paged_cache_memory_mb", "pagedCacheMemoryMb"),
            paged_cache_initial_memory_mb: number(
                "paged_cache_initial_memory_mb",
                "pagedCacheInitialMemoryMb",
            ),
            paged_block_size: number("paged_block_size", "pagedBlockSize"),
            use_block_paged_cache: boolean("use_block_paged_cache", "useBlockPagedCache"),
            persist_paged_cache: boolean("persist_paged_cache", "persistPagedCache"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Flattened into a surrounding struct, the five keys keep their flat
    /// top-level position and absent keys deserialize to `None` — the exact
    /// contract the per-family declarations have today.
    #[test]
    fn flatten_round_trips_inside_a_host_struct() {
        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct Host {
            required: i32,
            #[serde(default)]
            other: Option<String>,
            #[serde(flatten)]
            paged: PagedCacheConfig,
        }

        // Absent keys -> all None, required sibling fields untouched.
        let host: Host = serde_json::from_str(r#"{"required": 1}"#).unwrap();
        assert_eq!(host.paged, PagedCacheConfig::default());
        // Present keys land on the flattened struct; unknown keys still ignored.
        let host: Host = serde_json::from_str(
            r#"{"required": 1, "use_block_paged_cache": true, "paged_block_size": 32,
                "paged_cache_memory_mb": 256, "persist_paged_cache": false,
                "paged_cache_initial_memory_mb": 64, "unknown_key": [1,2,3]}"#,
        )
        .unwrap();
        assert_eq!(
            host.paged,
            PagedCacheConfig {
                paged_cache_memory_mb: Some(256),
                paged_cache_initial_memory_mb: Some(64),
                paged_block_size: Some(32),
                use_block_paged_cache: Some(true),
                persist_paged_cache: Some(false),
            }
        );
        // Explicit nulls read as None, same as `#[serde(default)]` fields.
        let host: Host =
            serde_json::from_str(r#"{"required": 1, "paged_block_size": null}"#).unwrap();
        assert_eq!(host.paged.paged_block_size, None);
        // Serialization emits the keys inline at the flatten position.
        let json = serde_json::to_string(&host).unwrap();
        assert!(json.contains("\"paged_block_size\":null"));
        assert!(json.contains("\"use_block_paged_cache\":null"));
        assert!(!json.contains("\"paged\""));
    }

    /// `from_raw_json` preserves the manual-parse semantics: `as_u64` for
    /// numbers (negatives and non-numbers -> `None`), `as_bool` for flags.
    #[test]
    fn from_raw_json_matches_manual_value_reads() {
        let raw = json!({
            "paged_cache_memory_mb": 512,
            "paged_block_size": 16,
            "use_block_paged_cache": false,
            "persist_paged_cache": true,
            "paged_cache_initial_memory_mb": 128,
        });
        let paged = PagedCacheConfig::from_raw_json(&raw);
        assert_eq!(paged.paged_cache_memory_mb, Some(512));
        assert_eq!(paged.paged_block_size, Some(16));
        assert_eq!(paged.use_block_paged_cache, Some(false));
        assert_eq!(paged.persist_paged_cache, Some(true));
        assert_eq!(paged.paged_cache_initial_memory_mb, Some(128));

        // Strict reads: wrong types and negatives land on `None`.
        let raw = json!({
            "paged_cache_memory_mb": -1,
            "paged_block_size": "16",
            "use_block_paged_cache": 1,
            "persist_paged_cache": "yes",
        });
        assert_eq!(
            PagedCacheConfig::from_raw_json(&raw),
            PagedCacheConfig::default()
        );
        assert_eq!(
            PagedCacheConfig::from_raw_json(&json!({})),
            PagedCacheConfig::default()
        );
    }

    /// The camel-alias variant preserves qwen3's snake-first precedence and
    /// its `as_i64` number reads (so `-1` wraps to `u32::MAX` as today).
    #[test]
    fn from_raw_json_camel_aliases_preserves_qwen3_convention() {
        let raw = json!({
            "use_block_paged_cache": false,
            "useBlockPagedCache": true,
            "pagedBlockSize": 32,
            "paged_cache_memory_mb": -1,
        });
        let paged = PagedCacheConfig::from_raw_json_camel_aliases(&raw);
        assert_eq!(paged.use_block_paged_cache, Some(false), "snake wins");
        assert_eq!(paged.paged_block_size, Some(32), "camel fallback");
        assert_eq!(paged.paged_cache_memory_mb, Some(u32::MAX), "as_i64 wrap");
    }

    #[test]
    fn resolve_use_paged_default_applies_caller_default() {
        assert_eq!(
            PagedCacheConfig::resolve_use_paged_default(None, true),
            Some(true)
        );
        assert_eq!(
            PagedCacheConfig::resolve_use_paged_default(Some(false), true),
            Some(false)
        );
    }
}
