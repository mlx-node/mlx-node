//! The governor's status (Splash `MemoryStatusReporter` and `/status`
//! `memory_governor`), read from the governor alone: it never waits on a
//! model thread.

use napi_derive::napi;

use super::governor::{Counters, Governor, GovernorSnapshot, ResourceWaitStatus};
use super::metrics::{GAP_CAUSES, Metrics, TtftPart};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GovernorStatus {
    pub snapshot: GovernorSnapshot,
    pub waits: ResourceWaitStatus,
    pub counters: Counters,
    pub metrics: Metrics,
    pub decisions: Vec<String>,
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
}

impl From<GovernorStatus> for MemoryGovernorStatus {
    fn from(status: GovernorStatus) -> Self {
        let GovernorStatus {
            snapshot,
            waits,
            counters,
            metrics,
            decisions,
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
}
