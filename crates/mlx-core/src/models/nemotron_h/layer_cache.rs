//! Per-layer cache for NemotronH decoder layers: Mamba2State, a flat KVCache,
//! or nothing (MoE). The MTP stepper snapshots and restores these around each
//! propose/verify cycle.

use crate::array::MxArray;
use crate::transformer::KVCache;

use super::mamba2::Mamba2State;

pub enum NemotronHLayerCache {
    Mamba(Mamba2State),
    Attention(KVCache),
    MoE,
}

/// Point-in-time snapshot of one layer cache (used by the MTP stepper).
#[derive(Clone)]
pub struct NemotronHLayerSnapshot {
    pub(crate) mamba: Option<Mamba2State>,
    pub(crate) kv_keys: Option<MxArray>,
    pub(crate) kv_values: Option<MxArray>,
    pub(crate) kv_offset: i32,
}

impl NemotronHLayerCache {
    pub fn new_mamba(state: Mamba2State) -> Self {
        NemotronHLayerCache::Mamba(state)
    }

    pub fn new_attention() -> Self {
        NemotronHLayerCache::Attention(KVCache::new())
    }

    pub fn new_moe() -> Self {
        NemotronHLayerCache::MoE
    }

    pub fn as_mamba_state(&self) -> Option<&Mamba2State> {
        match self {
            NemotronHLayerCache::Mamba(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_mamba_state_mut(&mut self) -> Option<&mut Mamba2State> {
        match self {
            NemotronHLayerCache::Mamba(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_kv_cache_mut(&mut self) -> Option<&mut KVCache> {
        match self {
            NemotronHLayerCache::Attention(c) => Some(c),
            _ => None,
        }
    }

    /// Shared read-only view of the attention slot, for cursor/offset probes
    /// only. Gated so it cannot pick up a production caller, but wider than
    /// `cfg(test)` so `debug_assert!`s can still use it; production paths
    /// mutate through [`Self::as_kv_cache_mut`].
    #[cfg(any(test, debug_assertions))]
    pub fn as_kv_cache(&self) -> Option<&KVCache> {
        match self {
            NemotronHLayerCache::Attention(c) => Some(c),
            _ => None,
        }
    }

    /// This layer's live attention offset, or `None` for a Mamba/MoE layer.
    ///
    /// Ungated on purpose, unlike [`Self::as_kv_cache`]: `MtpStepper::frontier`
    /// has to name the attention frontier in a RELEASE build too, and it takes
    /// `&self`. Handing back the scalar rather than the cache keeps the
    /// gating intent — no production caller gets a read-only `KVCache` view.
    pub(crate) fn attention_offset(&self) -> Option<i32> {
        match self {
            NemotronHLayerCache::Attention(c) => Some(c.get_offset()),
            NemotronHLayerCache::Mamba(_) | NemotronHLayerCache::MoE => None,
        }
    }

    /// Snapshot this cache (clones refcounted handles - no GPU copy).
    pub fn snapshot(&self) -> NemotronHLayerSnapshot {
        match self {
            NemotronHLayerCache::Mamba(s) => NemotronHLayerSnapshot {
                mamba: Some(s.clone()),
                kv_keys: None,
                kv_values: None,
                kv_offset: 0,
            },
            NemotronHLayerCache::Attention(c) => NemotronHLayerSnapshot {
                mamba: None,
                kv_keys: c.keys_ref().cloned(),
                kv_values: c.values_ref().cloned(),
                kv_offset: c.get_offset(),
            },
            NemotronHLayerCache::MoE => NemotronHLayerSnapshot {
                mamba: None,
                kv_keys: None,
                kv_values: None,
                kv_offset: 0,
            },
        }
    }

    /// Restore this cache from a snapshot.
    pub fn restore(&mut self, snap: NemotronHLayerSnapshot) -> Result<(), String> {
        match self {
            NemotronHLayerCache::Mamba(s) => {
                let m = snap
                    .mamba
                    .ok_or("NemotronH layer cache restore: Mamba snapshot missing")?;
                *s = m;
            }
            NemotronHLayerCache::Attention(c) => {
                let keys = snap
                    .kv_keys
                    .ok_or("NemotronH layer cache restore: KV snapshot missing")?;
                let values = snap
                    .kv_values
                    .ok_or("NemotronH layer cache restore: KV snapshot missing")?;
                c.set_keys(keys);
                c.set_values(values);
                c.set_offset(snap.kv_offset);
            }
            NemotronHLayerCache::MoE => {}
        }
        Ok(())
    }

    /// Collect cache arrays for eval.
    pub fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        match self {
            NemotronHLayerCache::Mamba(s) => {
                out.push(&s.conv);
                out.push(&s.ssm);
            }
            NemotronHLayerCache::Attention(c) => {
                if let Some(k) = c.keys_ref() {
                    out.push(k);
                }
                if let Some(v) = c.values_ref() {
                    out.push(v);
                }
            }
            NemotronHLayerCache::MoE => {}
        }
    }
}
