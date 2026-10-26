//! Residency facts shared with `residencyInfo()` while direct banks fill or
//! partial slots change on the model thread.
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Why full direct residency was refused in favor of partial slots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fallback {
    LoadGate { bank_bytes: u64, budget: u64 },
}

impl std::fmt::Display for Fallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        match self {
            Self::LoadGate { bank_bytes, budget } => write!(
                f,
                "load gate: direct expert banks need {:.1} GiB, the host leaves {:.1} GiB",
                gib(*bank_bytes),
                gib(*budget)
            ),
        }
    }
}

/// What `residencyInfo()` reads from the JS thread while the model thread
/// updates direct-fill progress and live memory accounting.
pub struct Live {
    pub direct: AtomicBool,
    pub direct_bytes: AtomicU64,
    pub direct_layers: AtomicU64,
    pub direct_filled_layers: AtomicU64,
    pub direct_fill_in_progress: AtomicBool,
    pub weight_budget: AtomicU64,
    pub partial_expert_slots: AtomicU64,
    pub bank_budget: AtomicU64,
    pub live_non_reclaimable: AtomicU64,
    pub prefill_chunk_tokens: AtomicU64,
    pub fallback: Mutex<Option<Fallback>>,
}

impl Default for Live {
    fn default() -> Self {
        Self {
            direct: AtomicBool::new(false),
            direct_bytes: AtomicU64::new(0),
            direct_layers: AtomicU64::new(0),
            direct_filled_layers: AtomicU64::new(0),
            direct_fill_in_progress: AtomicBool::new(false),
            weight_budget: AtomicU64::new(0),
            partial_expert_slots: AtomicU64::new(0),
            bank_budget: AtomicU64::new(0),
            live_non_reclaimable: AtomicU64::new(0),
            prefill_chunk_tokens: AtomicU64::new(0),
            fallback: Mutex::new(None),
        }
    }
}

impl Live {
    pub fn set_fallback(&self, fallback: Fallback) {
        *self.fallback.lock().unwrap_or_else(|p| p.into_inner()) = Some(fallback);
    }

    pub fn fallback(&self) -> Option<Fallback> {
        self.fallback
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    pub fn store(&self, plan: &super::memory::Plan) {
        self.direct.store(plan.direct, Ordering::Release);
        self.direct_bytes
            .store(plan.direct_bytes, Ordering::Release);
        self.direct_layers
            .store(plan.direct_layers as u64, Ordering::Release);
        if !plan.direct {
            self.direct_filled_layers.store(0, Ordering::Release);
            self.direct_fill_in_progress.store(false, Ordering::Release);
        }
        self.weight_budget.store(plan.budget, Ordering::Release);
        self.partial_expert_slots
            .store(plan.partial_expert_slots as u64, Ordering::Release);
        self.bank_budget.store(plan.bank_budget, Ordering::Release);
        self.live_non_reclaimable
            .store(plan.live_non_reclaimable, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_gate_reason_names_bank_size_and_budget() {
        let reason = Fallback::LoadGate {
            bank_bytes: 73 << 30,
            budget: 61 << 30,
        };
        assert_eq!(
            reason.to_string(),
            "load gate: direct expert banks need 73.0 GiB, the host leaves 61.0 GiB"
        );
        let live = Live::default();
        assert!(live.fallback().is_none());
        live.set_fallback(reason.clone());
        assert_eq!(live.fallback(), Some(reason));
    }
}
