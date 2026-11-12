//! Splash `model/LatencyHistogram.hpp`: latency quantiles for the serving
//! status.

use std::sync::atomic::{AtomicU64, Ordering};

const BUCKETS: usize = 40;

/// Splash `LatencyHistogram`: latencies in log-2 buckets of microseconds;
/// a quantile reads as its bucket's upper bound.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LatencyHistogram {
    buckets: [u64; BUCKETS],
    maximum_seconds: f64,
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self {
            buckets: [0; BUCKETS],
            maximum_seconds: 0.0,
        }
    }
}

impl LatencyHistogram {
    pub(crate) fn add(&mut self, seconds: f64) {
        let micros = (seconds.max(0.0) * 1e6) as u64;
        let bucket = ((u64::BITS - micros.leading_zeros()) as usize).min(BUCKETS - 1);
        self.buckets[bucket] += 1;
        self.maximum_seconds = self.maximum_seconds.max(seconds);
    }

    pub(crate) fn maximum_ms(&self) -> f64 {
        self.maximum_seconds * 1000.0
    }

    pub(crate) fn quantile_ms(&self, q: f64) -> f64 {
        let count: u64 = self.buckets.iter().sum();
        if count == 0 {
            return 0.0;
        }
        let rank = (q * (count - 1) as f64) as u64 + 1;
        let mut seen = 0u64;
        for (bucket, &n) in self.buckets.iter().enumerate() {
            seen += n;
            if seen >= rank {
                return (1u64 << bucket) as f64 / 1000.0;
            }
        }
        self.maximum_ms()
    }
}

/// Splash `LatencyStats`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct LatencyStats {
    pub samples: u64,
    pub total_ms: f64,
    pub max_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

const FINE_BUCKETS: usize = 4096;
const NANOS_PER_MS: u64 = 1_000_000;

/// Splash `FineLatencyHistogram`: latencies in buckets of a millisecond,
/// updated from any thread. Bucket b holds (b - 1, b] ms and reads as b, or
/// as the maximum where that is lower; latencies past the buckets share the
/// last one, which reads as the maximum.
pub(crate) struct FineLatencyHistogram {
    buckets: Box<[AtomicU64]>,
    total_ns: AtomicU64,
    max_ns: AtomicU64,
}

impl FineLatencyHistogram {
    pub(crate) fn new() -> Self {
        Self {
            buckets: (0..FINE_BUCKETS).map(|_| AtomicU64::new(0)).collect(),
            total_ns: AtomicU64::new(0),
            max_ns: AtomicU64::new(0),
        }
    }

    pub(crate) fn add(&self, nanos: u64) {
        let millis = nanos.div_ceil(NANOS_PER_MS);
        let bucket = millis.min(FINE_BUCKETS as u64 - 1) as usize;
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
        self.total_ns.fetch_add(nanos, Ordering::Relaxed);
        self.max_ns.fetch_max(nanos, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self) -> LatencyStats {
        let counts: Vec<u64> = self
            .buckets
            .iter()
            .map(|bucket| bucket.load(Ordering::Relaxed))
            .collect();
        let samples: u64 = counts.iter().sum();
        let max_ms = self.max_ns.load(Ordering::Relaxed) as f64 / 1e6;
        let quantile = |q: f64| {
            if samples == 0 {
                return 0.0;
            }
            let rank = (q * (samples - 1) as f64) as u64 + 1;
            let mut seen = 0u64;
            for (bucket, &count) in counts[..FINE_BUCKETS - 1].iter().enumerate() {
                seen += count;
                if seen >= rank {
                    return (bucket as f64).min(max_ms);
                }
            }
            max_ms
        };
        LatencyStats {
            samples,
            total_ms: self.total_ns.load(Ordering::Relaxed) as f64 / 1e6,
            max_ms,
            p50_ms: quantile(0.50),
            p95_ms: quantile(0.95),
            p99_ms: quantile(0.99),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Splash `FineLatencyHistogram`: a latency reads as its millisecond
    /// rounded up, never above the maximum; past the buckets it reads as
    /// the maximum.
    /// Mutation H1: the bucket is the millisecond rounded down → red (p50
    /// 450.0).
    #[test]
    fn fine_latency_quantiles_match_splash() {
        let histogram = FineLatencyHistogram::new();
        for nanos in [300_000_000, 450_000_001, 520_400_000] {
            histogram.add(nanos);
        }
        let stats = histogram.snapshot();
        assert_eq!(stats.samples, 3);
        assert_eq!(stats.p50_ms, 451.0);
        assert_eq!(stats.max_ms, 520.4);
        assert!((stats.total_ms - 1270.400001).abs() < 1e-9);
        histogram.add(5_000_000_000);
        histogram.add(5_000_000_000);
        let stats = histogram.snapshot();
        assert_eq!(stats.samples, 5);
        assert_eq!(stats.max_ms, 5000.0);
        assert_eq!(stats.p99_ms, 5000.0);
    }
}
