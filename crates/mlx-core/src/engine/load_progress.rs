//! What each model load in flight is doing, keyed by the path it was given:
//! reading its weights, or preparing GPU kernels (the warm-up) step by step.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use napi_derive::napi;

const KERNELS: u8 = 1;

/// One load's phase (weights until the warm-up begins) and steps. The
/// model thread writes, any thread reads.
#[derive(Debug, Default)]
pub(crate) struct LoadProgressCell {
    phase: AtomicU8,
    steps: AtomicU32,
    steps_done: AtomicU32,
    /// Every `steps_done` a step left, in order.
    #[cfg(test)]
    pub(crate) done_history: Mutex<Vec<u32>>,
}

impl LoadProgressCell {
    /// A cell no path reads, for loads that do not report.
    pub(crate) fn detached() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// The warm-up starts: `steps` steps, none done.
    pub(crate) fn begin_kernels(&self, steps: u32) {
        self.steps_done.store(0, Ordering::Relaxed);
        self.steps.store(steps, Ordering::Relaxed);
        self.phase.store(KERNELS, Ordering::Relaxed);
    }

    /// One warm-up step ended, run or skipped.
    pub(crate) fn step_done(&self) {
        let _done = self.steps_done.fetch_add(1, Ordering::Relaxed) + 1;
        #[cfg(test)]
        self.done_history
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(_done);
    }

    pub(crate) fn snapshot(&self) -> LoadProgress {
        let kernels = self.phase.load(Ordering::Relaxed) == KERNELS;
        LoadProgress {
            phase: if kernels { "kernels" } else { "weights" }.to_string(),
            steps: if kernels {
                self.steps.load(Ordering::Relaxed)
            } else {
                0
            },
            steps_done: if kernels {
                self.steps_done.load(Ordering::Relaxed)
            } else {
                0
            },
        }
    }
}

/// A model load in flight.
#[napi(object, js_name = "LoadProgress")]
#[derive(Clone, Debug, PartialEq)]
pub struct LoadProgress {
    #[napi(ts_type = "'weights' | 'kernels'")]
    pub phase: String,
    pub steps: u32,
    pub steps_done: u32,
}

type Loads = HashMap<String, Vec<Arc<LoadProgressCell>>>;

fn loads() -> std::sync::MutexGuard<'static, Loads> {
    static LOADS: OnceLock<Mutex<Loads>> = OnceLock::new();
    LOADS
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Removes its load's entry when the load ends, ok or not.
pub(crate) struct LoadProgressGuard {
    path: String,
    cell: Arc<LoadProgressCell>,
}

impl LoadProgressGuard {
    pub(crate) fn cell(&self) -> Arc<LoadProgressCell> {
        Arc::clone(&self.cell)
    }
}

impl Drop for LoadProgressGuard {
    fn drop(&mut self) {
        let mut loads = loads();
        if let Some(cells) = loads.get_mut(&self.path) {
            cells.retain(|cell| !Arc::ptr_eq(cell, &self.cell));
            if cells.is_empty() {
                loads.remove(&self.path);
            }
        }
    }
}

/// Registers a load of `path` in the weights phase.
pub(crate) fn begin(path: &str) -> LoadProgressGuard {
    let cell = LoadProgressCell::detached();
    loads()
        .entry(path.to_string())
        .or_default()
        .push(Arc::clone(&cell));
    LoadProgressGuard {
        path: path.to_string(),
        cell,
    }
}

/// The load of `path` in flight (the oldest when several are); `null` when
/// none is, or its family does not report.
#[napi]
pub fn load_progress(path: String) -> Option<LoadProgress> {
    loads()
        .get(&path)
        .and_then(|cells| cells.first())
        .map(|cell| cell.snapshot())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two loads keep their own counts; a guard's drop removes only its
    /// own entry.
    #[test]
    fn loads_report_by_path_until_they_end() {
        let first = begin("/models/progress-a");
        let second = begin("/models/progress-b");
        assert_eq!(
            load_progress("/models/progress-a".into()).unwrap().phase,
            "weights"
        );
        first.cell().begin_kernels(5);
        first.cell().step_done();
        second.cell().begin_kernels(9);
        let a = load_progress("/models/progress-a".into()).unwrap();
        let b = load_progress("/models/progress-b".into()).unwrap();
        assert_eq!((a.phase.as_str(), a.steps, a.steps_done), ("kernels", 5, 1));
        assert_eq!((b.phase.as_str(), b.steps, b.steps_done), ("kernels", 9, 0));
        let again = begin("/models/progress-a");
        drop(first);
        assert_eq!(
            load_progress("/models/progress-a".into()).unwrap().phase,
            "weights"
        );
        drop(again);
        drop(second);
        assert_eq!(load_progress("/models/progress-a".into()), None);
        assert_eq!(load_progress("/models/progress-b".into()), None);
    }
}
