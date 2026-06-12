//! Process-wide synchronization for the compiled C++ forward paths.
//!
//! The compiled paths share process-global C++ state (the `g_weights()` map,
//! `g_active_model_id`, per-family compiled decode caches), so every family
//! that drives them must serialize on these one-per-process locks. Moved here
//! from `models::qwen3_5::model` so the contract lives in family-neutral code.

/// RwLock protecting the C++ global weight map against concurrent mutation.
/// Write-locked during weight registration (model load), read-locked during
/// compiled inference. This prevents a concurrent model load from swapping
/// weights underneath an in-flight compiled decode, and eliminates the TOCTOU
/// between has_weight() / get_weight() in linear_proj().
pub(crate) static COMPILED_WEIGHTS_RWLOCK: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Process-wide mutex serializing the compiled forward LIFECYCLE (per-turn
/// init / decode / reset) across model instances AND model families.
///
/// Within a single model instance the dedicated model thread serializes calls,
/// but distinct models run on distinct OS threads (one per model, see
/// `model_thread.rs`), so a qwen3.5 compiled decode and an lfm2 compiled decode
/// genuinely run in parallel. They collide on the SAME process-global C++
/// globals: the `g_weights()` map (read by the NOT id-aware `get_weight` /
/// `get_weight_t`), the shared `g_active_model_id` atom, and each family's
/// compiled decode state (`g_*_caches` / `g_*_offset_int`).
///
/// `pub(crate)` and family-agnostic by design: EVERY model that drives a
/// compiled path over the shared registry (qwen3.5 dense + MoE, lfm2, …) MUST
/// serialize its compiled lifecycle on THIS one instance — never a private
/// per-family mutex, which would provide zero mutual exclusion against an
/// in-flight compiled decode from another family.
pub(crate) static COMPILED_LIFECYCLE_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
