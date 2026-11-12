//! The governor's status (Splash `MemoryStatusReporter` and `/status`
//! `memory_governor`), read from the governor alone: it never waits on a
//! model thread.

use napi_derive::napi;

use super::governor::{Counters, Governor, GovernorSnapshot, ResourceWaitStatus, SchedulerStatus};
use super::metrics::{GAP_CAUSES, Metrics, TtftPart};
use crate::engine::latency::LatencyStats;
use crate::engine::scheduler::DECODE_GAP_MS;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GovernorStatus {
    pub snapshot: GovernorSnapshot,
    pub waits: ResourceWaitStatus,
    pub counters: Counters,
    pub metrics: Metrics,
    pub decisions: Vec<String>,
    pub scheduler: Vec<SchedulerStatus>,
}

/// `None` while no model is registered.
pub(crate) fn status(governor: &Governor) -> Option<GovernorStatus> {
    if governor.registered_models() == 0 {
        return None;
    }
    Some(GovernorStatus {
        snapshot: governor.snapshot(),
        waits: governor.resource_waits(),
        counters: governor.counters(),
        metrics: governor.metrics(),
        decisions: governor.decisions(),
        scheduler: governor.schedulers(),
    })
}

#[napi(object)]
pub struct MemoryGovernorHost {
    pub measured: bool,
    pub available: f64,
    pub spare: f64,
    pub spare_reserve: f64,
    pub headroom: f64,
    pub system_pressure: String,
    pub growth_allowed: bool,
}

#[napi(object)]
pub struct MemoryGovernorAccount {
    pub name: String,
    pub current: f64,
}

#[napi(object)]
pub struct MemoryGovernorOwed {
    pub bytes: f64,
    pub transfers: f64,
}

#[napi(object)]
pub struct MemoryGovernorClaims {
    pub waiting: u32,
    pub bytes: f64,
    pub oldest_ms: f64,
}

#[napi(object)]
pub struct MemoryGovernorResourceWaits {
    pub memory: u32,
    pub concurrency: u32,
    pub suspended: u32,
    pub draining: bool,
    pub held_behind_refusal: u32,
    pub oldest_ms: f64,
}

#[napi(object)]
pub struct MemoryGovernorCounters {
    pub replans: f64,
    pub resource_suspensions: f64,
    pub resource_resumptions: f64,
    pub resource_timeouts: f64,
    pub capacity_exhausted: f64,
    pub metal_out_of_memory: f64,
    pub publications_refused: f64,
}

/// Milliseconds by Splash `TtftPart`, summed over the requests' first
/// tokens; Rest is never accumulated.
#[napi(object)]
pub struct MemoryGovernorTtftParts {
    pub own_prefill: f64,
    pub claim_wait: f64,
    pub decode: f64,
    pub other_prefill: f64,
}

#[napi(object)]
pub struct MemoryGovernorGapCause {
    pub gaps: f64,
    pub ms: f64,
}

/// Decode gaps past 500 ms by Splash `GapCause`.
#[napi(object)]
pub struct MemoryGovernorGapCauses {
    pub prefill_unit: MemoryGovernorGapCause,
    pub claim_wait: MemoryGovernorGapCause,
    pub ssd_read: MemoryGovernorGapCause,
    pub pool_release: MemoryGovernorGapCause,
    pub replan: MemoryGovernorGapCause,
    pub other: MemoryGovernorGapCause,
    pub prefill_ms: f64,
}

/// Every gap between two token events of a lane, as log-2 bucket bounds.
#[napi(object)]
pub struct MemoryGovernorGapLatency {
    pub p50: f64,
    pub p99: f64,
    pub max: f64,
}

/// Splash `LatencyStats` in milliseconds.
#[napi(object)]
pub struct MemoryGovernorLatency {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    pub total: f64,
    pub samples: f64,
}

impl From<LatencyStats> for MemoryGovernorLatency {
    fn from(stats: LatencyStats) -> Self {
        Self {
            p50: stats.p50_ms,
            p95: stats.p95_ms,
            p99: stats.p99_ms,
            max: stats.max_ms,
            total: stats.total_ms,
            samples: stats.samples as f64,
        }
    }
}

/// One model's scheduler (Splash `Status.cpp` `writeScheduler`).
#[napi(object)]
pub struct MemoryGovernorScheduler {
    /// The model's `governorModelId` in its context limits.
    pub model_id: f64,
    pub queued: f64,
    pub waiting_resources: f64,
    pub waiting_prefix: f64,
    pub prefilling: f64,
    pub decoding: f64,
    pub prefill_batches: f64,
    pub prefill_rows: f64,
    pub decode_batches: f64,
    /// Decode batches of width `index + 1`.
    pub decode_batches_by_width: Vec<f64>,
    pub decode_gap_target_ms: f64,
    pub contended_prefill_ms: MemoryGovernorLatency,
    pub prefill_units: f64,
    pub bound_cuts: f64,
    pub replans: f64,
}

impl From<SchedulerStatus> for MemoryGovernorScheduler {
    fn from(status: SchedulerStatus) -> Self {
        let SchedulerStatus {
            model_id,
            counters,
            replans,
        } = status;
        Self {
            model_id: model_id as f64,
            queued: counters.queued as f64,
            waiting_resources: counters.waiting_resources as f64,
            waiting_prefix: counters.waiting_prefix as f64,
            prefilling: counters.prefilling as f64,
            decoding: counters.decoding as f64,
            prefill_batches: counters.prefill_batches as f64,
            prefill_rows: counters.prefill_rows as f64,
            decode_batches: counters.decode_batches as f64,
            decode_batches_by_width: counters
                .decode_batches_by_width
                .iter()
                .map(|&batches| batches as f64)
                .collect(),
            decode_gap_target_ms: DECODE_GAP_MS,
            contended_prefill_ms: counters.contended_prefill.into(),
            prefill_units: counters.prefill_units as f64,
            bound_cuts: counters.bound_cuts as f64,
            replans: replans as f64,
        }
    }
}

#[napi(object)]
pub struct MemoryGovernorStatus {
    pub limit: f64,
    pub charged: f64,
    pub headroom: f64,
    /// `critical`, `warning` (also while the host is in its band) or
    /// `normal`.
    pub pressure: String,
    pub denied_reservations: f64,
    pub host: MemoryGovernorHost,
    pub accounts: Vec<MemoryGovernorAccount>,
    pub owed: MemoryGovernorOwed,
    pub claims: MemoryGovernorClaims,
    pub resource_waits: MemoryGovernorResourceWaits,
    pub counters: MemoryGovernorCounters,
    pub ttft_parts_ms: MemoryGovernorTtftParts,
    pub decode_gap_causes: MemoryGovernorGapCauses,
    pub decode_gap_ms: MemoryGovernorGapLatency,
    /// The last 64 decision-log lines, oldest first.
    pub decisions: Vec<String>,
    /// One per model whose scheduler registered.
    pub scheduler: Vec<MemoryGovernorScheduler>,
}

impl From<GovernorStatus> for MemoryGovernorStatus {
    fn from(status: GovernorStatus) -> Self {
        let GovernorStatus {
            snapshot,
            waits,
            counters,
            metrics,
            decisions,
            scheduler,
        } = status;
        let ledger = &snapshot.ledger;
        let cause = |at: usize| MemoryGovernorGapCause {
            gaps: metrics.gaps[at] as f64,
            ms: metrics.gap_seconds[at] * 1000.0,
        };
        let part = |part: TtftPart| metrics.ttft_parts[part as usize] * 1000.0;
        debug_assert_eq!(GAP_CAUSES, 6);
        Self {
            limit: snapshot.limit as f64,
            charged: snapshot.charged as f64,
            headroom: snapshot.headroom as f64,
            pressure: snapshot.pressure.name().to_string(),
            denied_reservations: snapshot.denied_reservations as f64,
            host: MemoryGovernorHost {
                measured: snapshot.host.measured,
                available: snapshot.host.available as f64,
                spare: snapshot.host.spare as f64,
                spare_reserve: snapshot.host.spare_reserve as f64,
                headroom: snapshot.host.headroom as f64,
                system_pressure: snapshot.host.system_pressure.name().to_string(),
                growth_allowed: snapshot.host.growth_allowed,
            },
            accounts: ledger
                .accounts
                .iter()
                .map(|row| MemoryGovernorAccount {
                    name: row.name.clone(),
                    current: row.bytes as f64,
                })
                .collect(),
            owed: MemoryGovernorOwed {
                bytes: ledger.owed_bytes as f64,
                transfers: 0.0,
            },
            claims: MemoryGovernorClaims {
                waiting: ledger.claims_waiting,
                bytes: ledger.claims_bytes as f64,
                oldest_ms: ledger.oldest_waiting_ms,
            },
            resource_waits: MemoryGovernorResourceWaits {
                memory: waits.memory,
                concurrency: waits.concurrency,
                suspended: waits.suspended,
                draining: waits.draining,
                held_behind_refusal: waits.held_behind_refusal,
                oldest_ms: waits.oldest_ms,
            },
            counters: MemoryGovernorCounters {
                replans: counters.replans as f64,
                resource_suspensions: counters.resource_suspensions as f64,
                resource_resumptions: counters.resource_resumptions as f64,
                resource_timeouts: counters.resource_timeouts as f64,
                capacity_exhausted: counters.capacity_exhausted as f64,
                metal_out_of_memory: counters.metal_out_of_memory as f64,
                publications_refused: counters.publications_refused as f64,
            },
            ttft_parts_ms: MemoryGovernorTtftParts {
                own_prefill: part(TtftPart::OwnPrefill),
                claim_wait: part(TtftPart::ClaimWait),
                decode: part(TtftPart::Decode),
                other_prefill: part(TtftPart::OtherPrefill),
            },
            decode_gap_causes: MemoryGovernorGapCauses {
                prefill_unit: cause(0),
                claim_wait: cause(1),
                ssd_read: cause(2),
                pool_release: cause(3),
                replan: cause(4),
                other: cause(5),
                prefill_ms: metrics.gap_prefill_seconds * 1000.0,
            },
            decode_gap_ms: MemoryGovernorGapLatency {
                p50: metrics.gap_histogram.quantile_ms(0.5),
                p99: metrics.gap_histogram.quantile_ms(0.99),
                max: metrics.gap_histogram.maximum_ms(),
            },
            decisions,
            scheduler: scheduler
                .into_iter()
                .map(MemoryGovernorScheduler::from)
                .collect(),
        }
    }
}

/// The memory governor's status (Splash `/status` `memory_governor`), or
/// `null` while no model is registered with it. It reads the governor
/// alone and never waits for a model thread.
#[napi]
pub fn memory_governor_status() -> Option<MemoryGovernorStatus> {
    status(super::governor()).map(MemoryGovernorStatus::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::governor::governor::MemoryAccounts;
    use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
    use crate::engine::governor::host::Pressure;
    use crate::engine::governor::metrics::{Gap, GapCause};

    /// The status reaches JS in milliseconds and bytes, its accounts summing
    /// to charged; with no model registered there is none.
    #[test]
    fn status_converts_for_js() {
        let io = FakeIo::new(stats(6 * G), host(32 * G, Pressure::Normal));
        io.with(|fake| fake.stats.cache = G);
        let governor = Governor::new(io);
        assert!(status(&governor).is_none());
        let _registration = governor.register_model(model(16 * G, 0)).expect("register");
        governor.publish_accounts(
            7,
            MemoryAccounts {
                kv: 2 * G,
                states: G,
                embeddings: 0,
            },
        );
        governor.first_tokens(&[0.5, 0.25, 0.0, 0.0]);
        governor.decode_gap(&Gap {
            seconds: 0.75,
            cause: Some(GapCause::ClaimWait),
            prefill_seconds: 0.0,
        });
        governor.capacity_exhausted();
        let js = MemoryGovernorStatus::from(status(&governor).expect("registered"));
        assert_eq!(js.charged, (7 * G) as f64);
        let names: Vec<&str> = js.accounts.iter().map(|row| row.name.as_str()).collect();
        assert_eq!(names, ["fixed", "kv", "states", "embeddings", "free_pool"]);
        assert_eq!(
            js.accounts.iter().map(|row| row.current).sum::<f64>(),
            js.charged
        );
        assert_eq!(js.ttft_parts_ms.own_prefill, 500.0);
        assert_eq!(js.ttft_parts_ms.claim_wait, 250.0);
        assert_eq!(js.decode_gap_causes.claim_wait.gaps, 1.0);
        assert_eq!(js.decode_gap_causes.claim_wait.ms, 750.0);
        assert_eq!(js.decode_gap_ms.max, 750.0);
        assert_eq!(js.counters.capacity_exhausted, 1.0);
        assert_eq!(js.pressure, "normal");
        if let Some(process) = memory_governor_status() {
            assert_eq!(
                process.accounts.iter().map(|row| row.current).sum::<f64>(),
                process.charged
            );
        }
    }

    /// Splash `Status.cpp` writes one scheduler per engine: each model's
    /// counters are their own entry, read live, and leave with its
    /// registration.
    /// Mutation S1: one entry for all models (the last registration) → red
    /// (len 1).
    #[test]
    fn governor_status_carries_the_cut_counters_per_model() {
        use crate::engine::governor::governor::ModelRegistration;
        use crate::engine::scheduler::SchedulerCounters;
        use std::sync::Arc;
        use std::sync::atomic::Ordering;

        let io = FakeIo::new(stats(6 * G), host(32 * G, Pressure::Normal));
        let governor = Governor::new(io);
        let register = |model_id| {
            governor
                .register_model(ModelRegistration {
                    model_id,
                    ..model(16 * G, 0)
                })
                .expect("register")
        };
        let _first = register(7);
        let second = register(8);
        let first_counters = Arc::new(SchedulerCounters::new(4));
        first_counters.prefill_units.store(7, Ordering::Relaxed);
        first_counters.bound_cuts.store(3, Ordering::Relaxed);
        first_counters.contended_prefill.add(512_000_000);
        first_counters.contended_prefill.add(640_000_000);
        let second_counters = Arc::new(SchedulerCounters::new(4));
        second_counters.prefill_units.store(1, Ordering::Relaxed);
        governor.register_scheduler(7, Arc::clone(&first_counters));
        governor.register_scheduler(8, second_counters);

        let js = MemoryGovernorStatus::from(status(&governor).expect("registered"));
        assert_eq!(js.scheduler.len(), 2);
        let first = &js.scheduler[0];
        assert_eq!(first.model_id, 7.0);
        assert_eq!(first.prefill_units, 7.0);
        assert_eq!(first.bound_cuts, 3.0);
        assert_eq!(first.contended_prefill_ms.samples, 2.0);
        assert_eq!(first.contended_prefill_ms.max, 640.0);
        assert_eq!(first.decode_gap_target_ms, 500.0);
        assert_eq!(first.decode_batches_by_width.len(), 4);
        assert_eq!(js.scheduler[1].prefill_units, 1.0);

        first_counters.prefill_units.store(9, Ordering::Relaxed);
        drop(second);
        let js = MemoryGovernorStatus::from(status(&governor).expect("registered"));
        assert_eq!(js.scheduler.len(), 1);
        assert_eq!(js.scheduler[0].model_id, 7.0);
        assert_eq!(js.scheduler[0].prefill_units, 9.0, "read live");
    }
}
