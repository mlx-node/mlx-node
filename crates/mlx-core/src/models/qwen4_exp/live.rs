//! Residency facts shared with `residencyInfo()` while direct banks fill or
//! partial slots change on the model thread.
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Why the experts take partial slots instead of full direct banks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// The host refused the whole bank set the store plan chose.
    LoadGate { bank_bytes: u64, budget: u64 },
    /// The store plan streams the experts through a capped store.
    StorePlan { bank_bytes: u64, cap: u64 },
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
            Self::StorePlan { bank_bytes, cap } => write!(
                f,
                "store plan: experts stream through a {:.1} GiB store, all banks need {:.1} GiB",
                gib(*cap),
                gib(*bank_bytes)
            ),
        }
    }
}

/// Where the routed experts live, as `residencyInfo().residency` names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    Direct,
    HostGate,
    StorePlan,
    Rows,
}

impl Residency {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::HostGate => "host-gate",
            Self::StorePlan => "store-plan",
            Self::Rows => "rows",
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
    /// Routed experts the slots or banks served without a read since load,
    /// and the ones a slot fill had to read.
    pub slot_hits: AtomicU64,
    pub slot_misses: AtomicU64,
    /// Bytes read from the checkpoint files since load: expert fills and
    /// row imports.
    pub bytes_read: AtomicU64,
    /// Replay passes after a tentative device route missed the slots: each
    /// pass a decode token runs again from its missed layer (on device or
    /// host routes; one token can run several) and each prefill window.
    pub route_replays: AtomicU64,
    /// Decode tokens that ran a tentative device-route pass, and those of
    /// them whose routes missed the slots.
    pub route_tries: AtomicU64,
    pub route_miss_tokens: AtomicU64,
    /// Decode layers replays ran again on device routes after reading the
    /// missed experts, and those they ran again on host routes.
    pub replayed_layers: AtomicU64,
    pub host_tail_layers: AtomicU64,
    /// Times the route guard sent decode tokens to host routes.
    pub route_guard_trips: AtomicU64,
    /// Slot fills since load, the wall time inside them, and the bytes they
    /// asked of the checkpoint files: each wanted range rounded out to
    /// 16 KiB pages, a page two ranges share read once, no bytes between
    /// two ranges.
    pub fill_calls: AtomicU64,
    pub fill_micros: AtomicU64,
    pub fill_read_bytes: AtomicU64,
    /// The share of the fill counters, `slot_misses`, `bytes_read` and
    /// `route_replays` that the decoder's prefill passes moved; decode
    /// fills and MTP verification passes are not in them.
    pub prefill_fill_calls: AtomicU64,
    pub prefill_fill_micros: AtomicU64,
    pub prefill_fill_read_bytes: AtomicU64,
    pub prefill_slot_misses: AtomicU64,
    pub prefill_bytes_read: AtomicU64,
    pub prefill_route_replays: AtomicU64,
    /// Layer-major prefill groups of two or more windows, and their windows.
    pub group_passes: AtomicU64,
    pub group_windows: AtomicU64,
    /// The `HostMemory` split behind `live_non_reclaimable`.
    pub host_wired: AtomicU64,
    pub host_compressor: AtomicU64,
    pub host_anonymous: AtomicU64,
    pub host_purgeable: AtomicU64,
    /// Plan bytes beside the experts when the residency was chosen.
    pub anonymous_need: AtomicU64,
    /// This process's MLX active plus cache bytes when the residency was
    /// chosen.
    pub self_mlx_bytes: AtomicU64,
    /// Times a store the host gate refused at load moved up to whole banks.
    pub promotions: AtomicU64,
    /// Times the gate admitted such a move but the host could not take its
    /// growth, so it stopped before any bank dropped.
    pub promotion_refusals: AtomicU64,
    /// Drafted cycles whose rows the rule set to one: the draft proposed,
    /// the verify took the anchor alone, and the engine saw no proposal.
    pub draft_one_row_cycles: AtomicU64,
    /// Decode steps that ran the rows of 2-4 scheduled turns in one pass.
    pub lane_steps: AtomicU64,
    pub fallback: Mutex<Option<Fallback>>,
    /// Tests hold the direct fill before its first layer until a pass waits
    /// for one.
    #[cfg(test)]
    pub fill_hold: FillHold,
}

/// A test's hold on the direct fill worker.
#[cfg(test)]
#[derive(Default)]
pub struct FillHold {
    held: Mutex<bool>,
    wake: std::sync::Condvar,
}

#[cfg(test)]
impl FillHold {
    pub fn hold(&self) {
        *self.held.lock().unwrap() = true;
    }

    pub fn release(&self) {
        *self.held.lock().unwrap() = false;
        self.wake.notify_all();
    }

    pub fn wait(&self) {
        let mut held = self.held.lock().unwrap();
        while *held {
            held = self.wake.wait(held).unwrap();
        }
    }
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
            slot_hits: AtomicU64::new(0),
            slot_misses: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            route_replays: AtomicU64::new(0),
            route_tries: AtomicU64::new(0),
            route_miss_tokens: AtomicU64::new(0),
            replayed_layers: AtomicU64::new(0),
            host_tail_layers: AtomicU64::new(0),
            route_guard_trips: AtomicU64::new(0),
            fill_calls: AtomicU64::new(0),
            fill_micros: AtomicU64::new(0),
            fill_read_bytes: AtomicU64::new(0),
            prefill_fill_calls: AtomicU64::new(0),
            prefill_fill_micros: AtomicU64::new(0),
            prefill_fill_read_bytes: AtomicU64::new(0),
            prefill_slot_misses: AtomicU64::new(0),
            prefill_bytes_read: AtomicU64::new(0),
            prefill_route_replays: AtomicU64::new(0),
            group_passes: AtomicU64::new(0),
            group_windows: AtomicU64::new(0),
            host_wired: AtomicU64::new(0),
            host_compressor: AtomicU64::new(0),
            host_anonymous: AtomicU64::new(0),
            host_purgeable: AtomicU64::new(0),
            anonymous_need: AtomicU64::new(0),
            self_mlx_bytes: AtomicU64::new(0),
            promotions: AtomicU64::new(0),
            promotion_refusals: AtomicU64::new(0),
            draft_one_row_cycles: AtomicU64::new(0),
            lane_steps: AtomicU64::new(0),
            fallback: Mutex::new(None),
            #[cfg(test)]
            fill_hold: FillHold::default(),
        }
    }
}

impl Live {
    /// Each since-load counter a prefill pass moves, with its prefill share.
    pub fn prefill_shares(&self) -> [(&AtomicU64, &AtomicU64); 6] {
        [
            (&self.fill_calls, &self.prefill_fill_calls),
            (&self.fill_micros, &self.prefill_fill_micros),
            (&self.fill_read_bytes, &self.prefill_fill_read_bytes),
            (&self.slot_misses, &self.prefill_slot_misses),
            (&self.bytes_read, &self.prefill_bytes_read),
            (&self.route_replays, &self.prefill_route_replays),
        ]
    }

    pub fn set_fallback(&self, fallback: Fallback) {
        *self.fallback.lock().unwrap_or_else(|p| p.into_inner()) = Some(fallback);
    }

    pub fn clear_fallback(&self) {
        *self.fallback.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }

    pub fn fallback(&self) -> Option<Fallback> {
        self.fallback
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    pub fn residency(&self) -> Residency {
        if self.direct.load(Ordering::Acquire) {
            return Residency::Direct;
        }
        match self.fallback() {
            Some(Fallback::LoadGate { .. }) => Residency::HostGate,
            Some(Fallback::StorePlan { .. }) => Residency::StorePlan,
            None => Residency::Rows,
        }
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
        let host = plan.host.unwrap_or_default();
        self.host_wired.store(host.wired, Ordering::Release);
        self.host_compressor
            .store(host.compressor, Ordering::Release);
        self.host_anonymous.store(host.anonymous, Ordering::Release);
        self.host_purgeable.store(host.purgeable, Ordering::Release);
        self.anonymous_need
            .store(plan.anonymous_need, Ordering::Release);
        self.self_mlx_bytes
            .store(plan.self_mlx_bytes, Ordering::Release);
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
        assert_eq!(live.residency(), Residency::Rows);
        live.set_fallback(reason.clone());
        assert_eq!(live.fallback(), Some(reason));
        assert_eq!(live.residency(), Residency::HostGate);
        let plan = Fallback::StorePlan {
            bank_bytes: 73 << 30,
            cap: 59 << 30,
        };
        assert_eq!(
            plan.to_string(),
            "store plan: experts stream through a 59.0 GiB store, all banks need 73.0 GiB"
        );
        live.set_fallback(plan);
        assert_eq!(live.residency().as_str(), "store-plan");
        live.direct.store(true, Ordering::Release);
        assert_eq!(live.residency().as_str(), "direct");
    }
}
