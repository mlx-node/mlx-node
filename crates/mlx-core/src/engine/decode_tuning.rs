//! Decode planning for one loaded model: a device rule for grouped
//! paged-attention partitions and a bounded submission-depth search.
//!
//! Partitions are never timed: every partition count reduces the bf16
//! online-softmax partials over different page subsets, so a timing-based
//! pick made greedy transcripts differ run to run. Only the submission depth,
//! which changes when graphs are submitted and never their arithmetic, is
//! searched on completed production tokens. No chip-name tables, synthetic
//! inputs, extra forward passes, or extra GPU synchronizations. Decisions
//! belong to one loaded model and context scale, and are never persisted as
//! portable hardware defaults.

use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DecodePlan {
    pub early_layers: usize,
    /// None leaves existing routing alone; Some(0) disables grouped attention.
    pub grouped_stripes: Option<u32>,
}

thread_local! {
    static ACTIVE: Cell<Option<DecodePlan>> = const { Cell::new(None) };
}

pub(crate) fn current_plan() -> Option<DecodePlan> {
    ACTIVE.get()
}

/// The plan is captured while building the graph, never read by the GPU's
/// asynchronous evaluator. Restore even on errors or nested model execution.
pub(crate) struct PlanScope(Option<DecodePlan>);

impl PlanScope {
    pub fn enter(plan: DecodePlan) -> Self {
        Self(ACTIVE.replace(Some(plan)))
    }
}

impl Drop for PlanScope {
    fn drop(&mut self) {
        ACTIVE.set(self.0);
    }
}

/// Stage-1 SIMD groups per GPU core that saturate a grouped paged-attention
/// kernel (one SIMD group per query head per partition, each walking its
/// 16-token pages serially). Measured on a 40-core GPU: Gemma4 D512 Hq8
/// decode rose from 64 to 128 partitions (12.8 to 25.6 SIMD groups per core)
/// and was flat at 256, Hq16 rose to 64 (25.6 per core), gained 1% at 128
/// and lost 3% at 256; Muse-Glimmer D128 Hq32 attention at 32 (25.6 per
/// core) was level with generic V2 at 1K, 1.2-2.3x faster from 4K to 60K,
/// and within 6% of the fastest count up to 8K (9-11% behind 128-512 from
/// 16K, under 1.5% of a decode step).
const GROUPED_SIMD_GROUPS_PER_CORE: u32 = 16;

/// GPU cores of the active Metal device (IORegistry `gpu-core-count`; 0 when
/// nothing publishes it).
pub(crate) fn gpu_core_count() -> u32 {
    static CORES: OnceLock<u32> = OnceLock::new();
    *CORES.get_or_init(|| u32::try_from(unsafe { mlx_sys::mlx_gpu_core_count() }).unwrap_or(0))
}

/// Partition count of a grouped paged-attention decode: the smallest power
/// of two whose query-head SIMD groups fill the device, bounded by the
/// 16-token work tiles and `max_stripes` (the kernel's partition limit or a
/// live resource cap, a power of two). A pure function of the context and the
/// device, so every token of a run and every run on one machine reduce
/// attention in the same order. 0 when `max_stripes` admits no partition.
pub(crate) fn grouped_partition_stripes(
    context: u32,
    query_heads: u32,
    gpu_cores: u32,
    max_stripes: u32,
) -> u32 {
    if max_stripes < 4 {
        return 0;
    }
    let saturating = (GROUPED_SIMD_GROUPS_PER_CORE * gpu_cores.max(1))
        .div_ceil(query_heads.max(1))
        .next_power_of_two();
    let tile_bound = 1u32 << context.div_ceil(16).max(1).ilog2();
    saturating.min(tile_bound).clamp(4, max_stripes)
}

#[derive(Debug)]
struct Sweep {
    candidates: Vec<DecodePlan>,
    samples: Vec<Vec<f64>>,
    step: usize,
}

impl Sweep {
    fn new(candidates: Vec<DecodePlan>) -> Self {
        Self {
            samples: vec![Vec::with_capacity(3); candidates.len()],
            candidates,
            step: 0,
        }
    }

    fn index(&self) -> usize {
        let n = self.candidates.len();
        let offset = self.step % n;
        // Reverse alternate rounds to reduce systematic warmup/context drift.
        if (self.step / n).is_multiple_of(2) {
            offset
        } else {
            n - 1 - offset
        }
    }

    fn plan(&self) -> DecodePlan {
        self.candidates[self.index()]
    }

    fn observe(&mut self, seconds: f64) -> Option<DecodePlan> {
        let index = self.index();
        // First use of each candidate warms its graph/pipeline. An invalid
        // sample still advances the finite budget, but cannot qualify a plan.
        if self.step >= self.candidates.len() && seconds.is_finite() && seconds > 0.0 {
            self.samples[index].push(seconds);
        }
        self.step += 1;
        if self.step < 4 * self.candidates.len() {
            return None;
        }
        let mut best = 0;
        for index in 1..self.candidates.len() {
            if self.samples[index].len() != 3 || self.samples[best].len() != 3 {
                continue;
            }
            let incumbent = median(&self.samples[best]);
            let candidate = median(&self.samples[index]);
            // Require a win larger than observed jitter. The 1% floor is a
            // selection stability rule, not a device performance constant.
            let noise = 2.0 * mad(&self.samples[best]).max(mad(&self.samples[index]));
            if incumbent - candidate > noise.max(incumbent * 0.01) {
                best = index;
            }
        }
        Some(self.candidates[best])
    }
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn mad(values: &[f64]) -> f64 {
    let center = median(values);
    median(
        &values
            .iter()
            .map(|value| (value - center).abs())
            .collect::<Vec<_>>(),
    )
}

#[derive(Debug)]
struct ContextPlan {
    bucket: u32,
    plan: DecodePlan,
    sweep: Option<Sweep>,
}

fn submission_candidates(plan: DecodePlan, layers: usize) -> Vec<DecodePlan> {
    let mut candidates = vec![plan];
    let mut depth = 1;
    while depth < layers {
        candidates.push(DecodePlan {
            early_layers: depth,
            ..plan
        });
        depth = depth.saturating_mul(2);
    }
    if layers > 1
        && candidates
            .last()
            .is_none_or(|last| last.early_layers != layers - 1)
    {
        candidates.push(DecodePlan {
            early_layers: layers - 1,
            ..plan
        });
    }
    candidates
}

#[derive(Debug, Default)]
pub(crate) struct DecodeTuning {
    contexts: VecDeque<ContextPlan>,
}

impl DecodeTuning {
    /// Submission-depth search only; the returned plan leaves attention
    /// routing (`grouped_stripes`) to the caller's device rule.
    pub fn begin(&mut self, context: u32, layers: usize, submission: bool) -> DecodePlan {
        let bucket = context
            .max(512)
            .checked_next_power_of_two()
            .unwrap_or(u32::MAX);
        if let Some(index) = self
            .contexts
            .iter()
            .position(|entry| entry.bucket == bucket)
        {
            if let Some(entry) = self.contexts.remove(index) {
                self.contexts.push_front(entry);
            }
        } else {
            let plan = DecodePlan::default();
            self.contexts.push_front(ContextPlan {
                bucket,
                plan,
                sweep: (submission && layers > 1)
                    .then(|| Sweep::new(submission_candidates(plan, layers))),
            });
            self.contexts.truncate(8);
        }
        let entry = &self.contexts[0];
        entry.sweep.as_ref().map_or(entry.plan, Sweep::plan)
    }

    pub fn observe(&mut self, seconds: f64) {
        let Some(entry) = self.contexts.front_mut() else {
            return;
        };
        let Some(sweep) = &mut entry.sweep else {
            return;
        };
        if let Some(plan) = sweep.observe(seconds) {
            entry.plan = plan;
            tracing::info!(target: "mlx_core::decode_tuning", event = "decode_tuned",
                stage = "submission", context_bucket = entry.bucket,
                early_layers = plan.early_layers, samples = ?sweep.samples,
                candidate_early_layers = ?sweep.candidates.iter().map(|p| p.early_layers).collect::<Vec<_>>(),
                "Decode plan selected from completed token timings");
            entry.sweep = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_rule_fills_the_device_within_tiles_and_caps() {
        // (context, query heads, GPU cores, max stripes) -> stripes: the
        // smallest power of two that fills the device, bounded by the
        // 16-token tiles and the caller's cap (Gemma4's 256-partition D512
        // reducer, Muse-Glimmer's live D128 resource limit).
        for (context, heads, cores, cap, expected) in [
            (513, 8, 40, 256, 32),
            (1_024, 8, 40, 256, 64),
            (2_048, 8, 40, 256, 128),
            (20_000, 8, 40, 256, 128),
            (20_000, 16, 40, 256, 64),
            (20_000, 32, 40, 256, 32),
            (20_000, 16, 8, 256, 8),
            (20_000, 32, 8, 256, 4),
            (20_000, 8, 4_096, 256, 256),
            (600, 8, 4_096, 256, 32),
            (20_000, 8, 0, 256, 4),
            (1_000, 32, 40, 32, 32),
            (60_000, 32, 40, 1_024, 32),
            (60_000, 32, 128, 1_024, 64),
            (60_000, 32, 4_096, 1_024, 1_024),
            (60_000, 32, 40, 16, 16),
            (60_000, 32, 40, 0, 0),
        ] {
            assert_eq!(
                grouped_partition_stripes(context, heads, cores, cap),
                expected,
                "context={context} heads={heads} cores={cores} cap={cap}"
            );
        }
    }

    #[test]
    fn adapts_to_different_device_latency_curves_and_layer_counts() {
        for (layers, best_depth) in [(48, 16), (18, 2), (80, 0)] {
            let mut tuner = DecodeTuning::default();
            for _ in 0..128 {
                let plan = tuner.begin(20_000, layers, true);
                assert_eq!(plan.grouped_stripes, None, "routing stays with the caller");
                // Unit-test timing observations, never model benchmark inputs.
                tuner.observe(if plan.early_layers == best_depth {
                    0.02
                } else {
                    0.025
                });
            }
            assert_eq!(
                tuner.begin(20_001, layers, true),
                DecodePlan {
                    early_layers: best_depth,
                    grouped_stripes: None,
                }
            );
        }
    }

    #[test]
    fn jitter_or_invalid_samples_cannot_qualify_a_new_default() {
        for candidate_samples in [[9.0, 10.0, 11.0], [f64::NAN; 3], [0.0; 3]] {
            let baseline = DecodePlan::default();
            let mut sweep = Sweep::new(vec![
                baseline,
                DecodePlan {
                    early_layers: 1,
                    ..baseline
                },
            ]);
            let mut outcome = None;
            for step in 0usize..8 {
                let value = if sweep.index() == 0 {
                    10.0
                } else {
                    candidate_samples[(step / 2).saturating_sub(1)]
                };
                outcome = sweep.observe(value);
            }
            assert_eq!(outcome, Some(baseline));
        }
    }

    #[test]
    fn respects_disabled_capabilities_and_bounds_context_storage() {
        let mut tuner = DecodeTuning::default();
        for exponent in 0..24 {
            assert_eq!(tuner.begin(1 << exponent, 1, false), DecodePlan::default());
        }
        assert_eq!(tuner.contexts.len(), 8);
        let first = tuner.contexts.front().unwrap().bucket;
        tuner.begin(1 << 22, 1, false);
        assert_eq!(tuner.contexts[1].bucket, first);
    }

    #[test]
    fn scoped_plan_restores_prior_state_on_early_return() {
        assert_eq!(current_plan(), None);
        let first = DecodePlan {
            early_layers: 2,
            grouped_stripes: Some(32),
        };
        let _scope = PlanScope::enter(first);
        {
            let _nested = PlanScope::enter(DecodePlan::default());
        }
        assert_eq!(current_plan(), Some(first));
    }
}
