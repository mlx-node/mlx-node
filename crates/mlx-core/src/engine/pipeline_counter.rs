//! Counter of the Metal pipelines and source libraries the process creates,
//! split by the creating thread's scope: a model thread's init closure is
//! loading, everything else is serving.
//!
//! A diagnostic: it replaces methods of the system Metal device class, so it
//! is installed only in this crate's tests and when `MLX_PIPELINE_COUNTER`
//! is set to a true value. Not installed, [`pipeline_stats`] is `null` and
//! nothing else changes.

use napi_derive::napi;

pub(crate) const SWITCH: &str = "MLX_PIPELINE_COUNTER";

/// Process-wide, cumulative pipeline creation totals.
#[napi(object, js_name = "PipelineStats")]
#[derive(Clone, Debug, PartialEq)]
pub struct PipelineStats {
    /// Pipelines and source libraries created inside a model load.
    pub created_loading: u32,
    /// Pipelines and source libraries created outside any model load.
    pub created_serving: u32,
    /// Milliseconds spent in the loading creations.
    pub loading_ms: f64,
    /// Milliseconds spent in the serving creations.
    pub serving_ms: f64,
}

pub(crate) fn enabled() -> bool {
    if cfg!(test) {
        return true;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| crate::inference_trace::env_flag_enabled(SWITCH))
}

/// Installs the counter once when [`enabled`]; whether it is installed.
#[cfg(target_os = "macos")]
pub(crate) fn install() -> bool {
    if !enabled() {
        return false;
    }
    match mlx_metal::pipeline_counter::install() {
        Ok(()) => true,
        Err(error) => {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    target: "mlx_core::inference",
                    "pipeline counter not installed: {error}"
                );
            });
            false
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn install() -> bool {
    false
}

/// Creations on this thread count as loading until the guard drops.
#[cfg(target_os = "macos")]
pub(crate) fn loading_scope() -> Option<mlx_metal::pipeline_counter::LoadingScope> {
    install().then(mlx_metal::pipeline_counter::loading_scope)
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn loading_scope() -> Option<()> {
    None
}

pub(crate) fn read() -> Option<PipelineStats> {
    #[cfg(target_os = "macos")]
    {
        let stats = mlx_metal::pipeline_counter::stats()?;
        let count = |n: u64| u32::try_from(n).unwrap_or(u32::MAX);
        Some(PipelineStats {
            created_loading: count(stats.created_loading),
            created_serving: count(stats.created_serving),
            loading_ms: stats.loading_ms,
            serving_ms: stats.serving_ms,
        })
    }
    #[cfg(not(target_os = "macos"))]
    None
}

/// The names [`warn_serving`] warned about, for tests.
#[cfg(test)]
pub(crate) static WARNED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The first warm-up of the process ran turns and nothing was counted:
/// the hooks are not on the path pipelines take, so the totals read `null`.
pub(crate) fn check_first_warmup(ran_turns: bool, created: u32) {
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| {
        #[cfg(target_os = "macos")]
        if ran_turns && created == 0 && read().is_some() {
            tracing::warn!(
                target: "mlx_core::inference",
                "pipeline counter saw no creation in the first load warm-up; it reads null"
            );
            mlx_metal::pipeline_counter::declare_dead();
        }
        #[cfg(not(target_os = "macos"))]
        let _ = (ran_turns, created);
    });
}

/// One warning per pipeline created while serving since the last call.
pub(crate) fn warn_serving() {
    #[cfg(target_os = "macos")]
    for created in mlx_metal::pipeline_counter::drain_serving() {
        #[cfg(test)]
        WARNED
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(created.name.clone());
        tracing::warn!(
            target: "mlx_core::inference",
            event = "pipeline_created_while_serving",
            name = %created.name,
            ms = created.ms,
        );
    }
}

/// Pipeline creation totals of this process; `null` while the counter is
/// not installed (it is installed only with `MLX_PIPELINE_COUNTER=1`).
#[napi]
pub fn pipeline_stats() -> Option<PipelineStats> {
    install();
    read()
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    /// W0: a pipeline made through `MetalState::get_pipeline` and one an MLX
    /// op makes are both counted and named, as serving; inside a loading
    /// scope they count as loading and leave no name.
    #[test]
    fn the_counter_sees_both_creation_sites() {
        if !crate::test_support::in_own_process(
            "engine::pipeline_counter::tests::the_counter_sees_both_creation_sites",
        ) {
            return;
        }
        let state = match mlx_paged_attn::metal::MetalState::get() {
            Ok(state) => state,
            Err(error) => {
                eprintln!("skipping the_counter_sees_both_creation_sites: {error}");
                return;
            }
        };
        assert!(install(), "the counter installs in tests");
        let before = read().expect("installed");
        let _ = mlx_metal::pipeline_counter::drain_serving();

        let kernel = mlx_paged_attn::metal::MetalState::reshape_and_cache_kernel_name(
            mlx_paged_attn::metal::MetalDtype::Float32,
            mlx_paged_attn::metal::MetalDtype::Float32,
        );
        state.get_pipeline(&kernel).expect("paged pipeline");
        let after_paged = read().expect("installed");
        assert_eq!(
            after_paged.created_serving - before.created_serving,
            1,
            "one paged pipeline"
        );
        let names = mlx_metal::pipeline_counter::drain_serving();
        assert_eq!(
            names.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            vec![kernel.as_str()]
        );

        let a = crate::array::MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).expect("a");
        let b = crate::array::MxArray::from_float32(&[4.0, 5.0, 6.0], &[3]).expect("b");
        let sum = a.add(&b).expect("add");
        sum.eval();
        let after_mlx = read().expect("installed");
        let mlx_created = after_mlx.created_serving - after_paged.created_serving;
        assert!(mlx_created >= 1, "the MLX op made a pipeline");
        let names = mlx_metal::pipeline_counter::drain_serving();
        assert_eq!(names.len() as u32, mlx_created, "one name each: {names:?}");
        assert!(
            names.iter().all(|c| !c.name.is_empty() && c.ms >= 0.0),
            "{names:?}"
        );
        assert!(after_mlx.serving_ms > before.serving_ms);

        {
            let _loading = loading_scope().expect("installed");
            let kernel = mlx_paged_attn::metal::MetalState::reshape_and_cache_kernel_name(
                mlx_paged_attn::metal::MetalDtype::Float16,
                mlx_paged_attn::metal::MetalDtype::Float16,
            );
            state.get_pipeline(&kernel).expect("paged pipeline");
        }
        let after_loading = read().expect("installed");
        assert_eq!(
            after_loading.created_loading - after_mlx.created_loading,
            1,
            "a creation in a loading scope counts as loading"
        );
        assert_eq!(after_loading.created_serving, after_mlx.created_serving);
        assert!(mlx_metal::pipeline_counter::drain_serving().is_empty());
    }
}
