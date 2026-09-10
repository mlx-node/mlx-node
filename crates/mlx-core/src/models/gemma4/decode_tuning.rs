//! Bounded, device-local tuning using completed production decode steps.
//!
//! No chip-name tables, synthetic inputs, extra forward passes, or extra GPU
//! synchronizations. Supported attention partitions and submission depths are
//! searched separately on real tokens. Decisions belong to one loaded model
//! and context scale, and are never persisted as portable hardware defaults.

use std::cell::Cell;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct DecodePlan {
    pub early_layers: usize,
    /// None leaves existing routing alone; Some(0) disables grouped attention.
    pub grouped_stripes: Option<u32>,
}

thread_local! {
    static ACTIVE: Cell<Option<DecodePlan>> = const { Cell::new(None) };
}

pub(super) fn current_plan() -> Option<DecodePlan> {
    ACTIVE.get()
}

/// The plan is captured while building the graph, never read by the GPU's
/// asynchronous evaluator. Restore even on errors or nested model execution.
pub(super) struct PlanScope(Option<DecodePlan>);

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
    submission_pending: bool,
    refine_pending: bool,
    stage: &'static str,
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
    if layers > 1 && candidates.last().unwrap().early_layers != layers - 1 {
        candidates.push(DecodePlan {
            early_layers: layers - 1,
            ..plan
        });
    }
    candidates
}

#[derive(Debug, Default)]
pub(super) struct DecodeTuning {
    contexts: VecDeque<ContextPlan>,
}

impl DecodeTuning {
    pub fn begin(
        &mut self,
        context: u32,
        layers: usize,
        grouped: bool,
        submission: bool,
    ) -> DecodePlan {
        let bucket = context
            .max(512)
            .checked_next_power_of_two()
            .unwrap_or(u32::MAX);
        if let Some(index) = self
            .contexts
            .iter()
            .position(|entry| entry.bucket == bucket)
        {
            let entry = self.contexts.remove(index).unwrap();
            self.contexts.push_front(entry);
        } else {
            let plan = DecodePlan {
                early_layers: 0,
                grouped_stripes: grouped.then_some(0),
            };
            let sweep = if grouped && context > 512 {
                let mut candidates = vec![plan];
                // Kernel supports power-of-two partitions [4,256]. Never
                // launch more partitions than physical 16-token work tiles.
                let work_tiles = context.div_ceil(16);
                for stripes in [4, 8, 16, 32, 64, 128, 256] {
                    if stripes <= work_tiles {
                        candidates.push(DecodePlan {
                            grouped_stripes: Some(stripes),
                            ..plan
                        });
                    }
                }
                Some(Sweep::new(candidates))
            } else if submission && layers > 1 {
                Some(Sweep::new(submission_candidates(plan, layers)))
            } else {
                None
            };
            self.contexts.push_front(ContextPlan {
                bucket,
                plan,
                sweep,
                submission_pending: grouped && context > 512 && submission && layers > 1,
                refine_pending: grouped && context > 512 && submission && layers > 1,
                stage: if grouped && context > 512 {
                    "attention"
                } else {
                    "submission"
                },
            });
            self.contexts.truncate(8);
        }
        let entry = &self.contexts[0];
        entry.sweep.as_ref().map_or(entry.plan, Sweep::plan)
    }

    pub fn observe(&mut self, seconds: f64, layers: usize) {
        let Some(entry) = self.contexts.front_mut() else {
            return;
        };
        let Some(sweep) = &mut entry.sweep else {
            return;
        };
        if let Some(plan) = sweep.observe(seconds) {
            entry.plan = plan;
            let stage = entry.stage;
            tracing::info!(target: "mlx_core::decode_tuning", event = "gemma4_decode_tuned",
                stage, context_bucket = entry.bucket, early_layers = plan.early_layers,
                grouped_stripes = plan.grouped_stripes.unwrap_or(0),
                grouped_tuning = plan.grouped_stripes.is_some(), samples = ?sweep.samples,
                candidate_early_layers = ?sweep.candidates.iter().map(|p| p.early_layers).collect::<Vec<_>>(),
                candidate_stripes = ?sweep.candidates.iter().map(|p| p.grouped_stripes.unwrap_or(0)).collect::<Vec<_>>(),
                "Gemma4 decode plan selected from completed token timings");
            entry.sweep = if entry.submission_pending {
                entry.submission_pending = false;
                entry.stage = "submission";
                Some(Sweep::new(submission_candidates(plan, layers)))
            } else if entry.refine_pending {
                entry.refine_pending = false;
                entry.stage = "attention_refinement";
                // Submission changes CPU/GPU overlap. Recheck neighboring
                // attention partitions under the selected scheduling depth.
                let mut candidates = vec![plan];
                let selected = plan.grouped_stripes.unwrap_or(0);
                for stripes in [4, 8, 16, 32, 64, 128, 256] {
                    if (selected == 0 || stripes == selected / 2 || stripes == selected * 2)
                        && stripes <= entry.bucket.div_ceil(16)
                    {
                        candidates.push(DecodePlan {
                            grouped_stripes: Some(stripes),
                            ..plan
                        });
                    }
                }
                Some(Sweep::new(candidates))
            } else {
                None
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapts_to_different_device_latency_curves_and_layer_counts() {
        for (layers, best_stripes, best_depth) in [(48, 64, 16), (18, 16, 2), (80, 256, 0)] {
            let mut tuner = DecodeTuning::default();
            for _ in 0..128 {
                let plan = tuner.begin(20_000, layers, true, true);
                // Unit-test timing observations, never model benchmark inputs.
                let seconds =
                    0.02 + if plan.grouped_stripes == Some(best_stripes) {
                        0.0
                    } else {
                        0.01
                    } + if plan.early_layers == best_depth {
                        0.0
                    } else {
                        0.005
                    };
                tuner.observe(seconds, layers);
            }
            assert_eq!(
                tuner.begin(20_001, layers, true, true),
                DecodePlan {
                    early_layers: best_depth,
                    grouped_stripes: Some(best_stripes),
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
            assert_eq!(
                tuner.begin(1 << exponent, 1, false, false),
                DecodePlan::default()
            );
        }
        assert_eq!(tuner.contexts.len(), 8);
        let first = tuner.contexts.front().unwrap().bucket;
        tuner.begin(1 << 22, 1, false, false);
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
