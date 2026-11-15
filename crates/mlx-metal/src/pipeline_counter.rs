//! Counts the compute pipelines and source libraries this process creates.
//!
//! [`install`] replaces, on the class of the system default `MTLDevice`, the
//! implementations of the methods every pipeline and source library is born
//! in (MLX's `Device::get_kernel_` / `build_library_` through metal-cpp, and
//! [`crate::Device::new_compute_pipeline_state_with_function`]). Each
//! replacement times the original call and adds it to the loading or the
//! serving totals, by the calling thread's [`loading_scope`]. A creation
//! outside a loading scope also leaves its name in a 64-entry ring that
//! [`drain_serving`] empties. A selector the class lacks is left alone.
//!
//! Nothing is replaced until [`install`] runs.

use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use objc2::ffi;
use objc2::rc::Retained;
use objc2::runtime::{AnyClass, AnyObject, Imp, Sel};
use objc2::{msg_send, sel};
use objc2_foundation::NSString;

const RING_CAPACITY: usize = 64;

static INSTALLED: OnceLock<Result<(), String>> = OnceLock::new();
static CREATED_LOADING: AtomicU64 = AtomicU64::new(0);
static CREATED_SERVING: AtomicU64 = AtomicU64::new(0);
static LOADING_NS: AtomicU64 = AtomicU64::new(0);
static SERVING_NS: AtomicU64 = AtomicU64::new(0);
static SERVING_RING: Mutex<VecDeque<Created>> = Mutex::new(VecDeque::new());
static DEAD: AtomicBool = AtomicBool::new(false);

static ORIGINAL_WITH_FUNCTION: OnceLock<Imp> = OnceLock::new();
static ORIGINAL_WITH_FUNCTION_REFLECTION: OnceLock<Imp> = OnceLock::new();
static ORIGINAL_WITH_DESCRIPTOR: OnceLock<Imp> = OnceLock::new();
static ORIGINAL_WITH_SOURCE: OnceLock<Imp> = OnceLock::new();

thread_local! {
    static LOADING_DEPTH: Cell<u32> = const { Cell::new(0) };
    static HOOK_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Process-wide totals since [`install`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub created_loading: u64,
    pub created_serving: u64,
    pub loading_ms: f64,
    pub serving_ms: f64,
}

/// One creation outside a loading scope.
#[derive(Clone, Debug, PartialEq)]
pub struct Created {
    pub name: String,
    pub ms: f64,
}

/// Creations on this thread count as loading while the scope lives.
#[must_use]
pub struct LoadingScope(());

impl Drop for LoadingScope {
    fn drop(&mut self) {
        LOADING_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

pub fn loading_scope() -> LoadingScope {
    LOADING_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
    LoadingScope(())
}

/// Replace the creation methods once per process. Later calls return the
/// first call's result.
pub fn install() -> Result<(), String> {
    INSTALLED.get_or_init(install_once).clone()
}

/// The hooks were installed but saw creations that must have happened go by
/// uncounted: [`stats`] reads `None` from now on.
pub fn declare_dead() {
    DEAD.store(true, Ordering::Relaxed);
}

/// The totals, `None` until [`install`] succeeded or once [`declare_dead`]
/// ran.
pub fn stats() -> Option<Stats> {
    if !matches!(INSTALLED.get(), Some(Ok(()))) || DEAD.load(Ordering::Relaxed) {
        return None;
    }
    let ms = |ns: &AtomicU64| ns.load(Ordering::Relaxed) as f64 / 1e6;
    Some(Stats {
        created_loading: CREATED_LOADING.load(Ordering::Relaxed),
        created_serving: CREATED_SERVING.load(Ordering::Relaxed),
        loading_ms: ms(&LOADING_NS),
        serving_ms: ms(&SERVING_NS),
    })
}

/// The serving creations recorded since the last drain, oldest first; at
/// most the last 64.
pub fn drain_serving() -> Vec<Created> {
    let mut ring = SERVING_RING
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    ring.drain(..).collect()
}

type WithFunction = unsafe extern "C-unwind" fn(
    *mut AnyObject,
    Sel,
    *mut AnyObject,
    *mut *mut AnyObject,
) -> *mut AnyObject;
type WithOptions = unsafe extern "C-unwind" fn(
    *mut AnyObject,
    Sel,
    *mut AnyObject,
    usize,
    *mut *mut AnyObject,
    *mut *mut AnyObject,
) -> *mut AnyObject;
type WithSource = unsafe extern "C-unwind" fn(
    *mut AnyObject,
    Sel,
    *mut AnyObject,
    *mut AnyObject,
    *mut *mut AnyObject,
) -> *mut AnyObject;

fn install_once() -> Result<(), String> {
    let device =
        objc2_metal::MTLCreateSystemDefaultDevice().ok_or_else(|| "no Metal device".to_string())?;
    let object: &AnyObject = device.as_ref();
    let class = object.class();
    // SAFETY: each hook has the C signature of the selector it replaces and
    // calls the implementation it replaced with the same arguments.
    let hooked = unsafe {
        [
            replace(
                class,
                sel!(newComputePipelineStateWithFunction:error:),
                std::mem::transmute::<WithFunction, Imp>(with_function),
                &ORIGINAL_WITH_FUNCTION,
            ),
            replace(
                class,
                sel!(newComputePipelineStateWithFunction:options:reflection:error:),
                std::mem::transmute::<WithOptions, Imp>(with_function_reflection),
                &ORIGINAL_WITH_FUNCTION_REFLECTION,
            ),
            replace(
                class,
                sel!(newComputePipelineStateWithDescriptor:options:reflection:error:),
                std::mem::transmute::<WithOptions, Imp>(with_descriptor),
                &ORIGINAL_WITH_DESCRIPTOR,
            ),
            replace(
                class,
                sel!(newLibraryWithSource:options:error:),
                std::mem::transmute::<WithSource, Imp>(with_source),
                &ORIGINAL_WITH_SOURCE,
            ),
        ]
    };
    if hooked.iter().any(|hooked| *hooked) {
        Ok(())
    } else {
        Err(format!(
            "{} has none of the pipeline creation methods",
            class.name().to_string_lossy()
        ))
    }
}

/// # Safety
/// `hook` must have the C signature of `selector`'s method.
unsafe fn replace(
    class: &AnyClass,
    selector: Sel,
    hook: Imp,
    original: &'static OnceLock<Imp>,
) -> bool {
    // SAFETY: a live class and selector; the method pointer is null when the
    // class lacks the selector.
    let method = unsafe { ffi::class_getInstanceMethod(class, selector) };
    if method.is_null() {
        return false;
    }
    // SAFETY: `method` is a method of a live class.
    let Some(previous) = (unsafe { ffi::method_getImplementation(method) }) else {
        return false;
    };
    if original.set(previous).is_err() {
        return false;
    }
    // SAFETY: upheld by the caller; the original is stored before the hook
    // can run.
    unsafe { ffi::method_setImplementation(method, hook) };
    true
}

/// Counts once per outermost creation on a thread: an implementation that
/// calls another replaced method of the device is not counted twice.
struct HookDepth {
    outermost: bool,
}

impl HookDepth {
    fn enter() -> Self {
        let outermost = HOOK_DEPTH.with(|depth| {
            let current = depth.get();
            depth.set(current.saturating_add(1));
            current == 0
        });
        Self { outermost }
    }
}

impl Drop for HookDepth {
    fn drop(&mut self) {
        HOOK_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

fn record(elapsed: Duration, name: impl FnOnce() -> String) {
    let ns = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
    if LOADING_DEPTH.with(Cell::get) > 0 {
        CREATED_LOADING.fetch_add(1, Ordering::Relaxed);
        LOADING_NS.fetch_add(ns, Ordering::Relaxed);
        return;
    }
    CREATED_SERVING.fetch_add(1, Ordering::Relaxed);
    SERVING_NS.fetch_add(ns, Ordering::Relaxed);
    let created = Created {
        name: name(),
        ms: ns as f64 / 1e6,
    };
    let mut ring = SERVING_RING
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if ring.len() == RING_CAPACITY {
        ring.pop_front();
    }
    ring.push_back(created);
}

fn timed(call: impl FnOnce() -> *mut AnyObject, name: impl FnOnce() -> String) -> *mut AnyObject {
    let depth = HookDepth::enter();
    let start = Instant::now();
    let result = call();
    let elapsed = start.elapsed();
    if depth.outermost {
        record(elapsed, name);
    }
    result
}

fn function_name(function: *mut AnyObject) -> String {
    // SAFETY: a null or live `MTLFunction`; `name` returns an `NSString`.
    let name: Option<Retained<NSString>> = match unsafe { function.as_ref() } {
        Some(function) => unsafe { msg_send![function, name] },
        None => None,
    };
    name.map_or_else(|| "<unnamed function>".to_string(), |name| name.to_string())
}

fn descriptor_function_name(descriptor: *mut AnyObject) -> String {
    // SAFETY: a null or live `MTLComputePipelineDescriptor`.
    let function: *mut AnyObject = match unsafe { descriptor.as_ref() } {
        Some(descriptor) => unsafe { msg_send![descriptor, computeFunction] },
        None => std::ptr::null_mut(),
    };
    function_name(function)
}

fn original<F: Copy>(slot: &OnceLock<Imp>) -> Option<F> {
    // SAFETY: each slot holds the implementation of the selector whose
    // signature `F` is.
    slot.get()
        .map(|imp| unsafe { std::mem::transmute_copy::<Imp, F>(imp) })
}

unsafe extern "C-unwind" fn with_function(
    this: *mut AnyObject,
    cmd: Sel,
    function: *mut AnyObject,
    error: *mut *mut AnyObject,
) -> *mut AnyObject {
    let Some(call) = original::<WithFunction>(&ORIGINAL_WITH_FUNCTION) else {
        return std::ptr::null_mut();
    };
    timed(
        // SAFETY: the replaced implementation, called with its arguments.
        || unsafe { call(this, cmd, function, error) },
        || function_name(function),
    )
}

unsafe extern "C-unwind" fn with_function_reflection(
    this: *mut AnyObject,
    cmd: Sel,
    function: *mut AnyObject,
    options: usize,
    reflection: *mut *mut AnyObject,
    error: *mut *mut AnyObject,
) -> *mut AnyObject {
    let Some(call) = original::<WithOptions>(&ORIGINAL_WITH_FUNCTION_REFLECTION) else {
        return std::ptr::null_mut();
    };
    timed(
        // SAFETY: the replaced implementation, called with its arguments.
        || unsafe { call(this, cmd, function, options, reflection, error) },
        || function_name(function),
    )
}

unsafe extern "C-unwind" fn with_descriptor(
    this: *mut AnyObject,
    cmd: Sel,
    descriptor: *mut AnyObject,
    options: usize,
    reflection: *mut *mut AnyObject,
    error: *mut *mut AnyObject,
) -> *mut AnyObject {
    let Some(call) = original::<WithOptions>(&ORIGINAL_WITH_DESCRIPTOR) else {
        return std::ptr::null_mut();
    };
    timed(
        // SAFETY: the replaced implementation, called with its arguments.
        || unsafe { call(this, cmd, descriptor, options, reflection, error) },
        || descriptor_function_name(descriptor),
    )
}

unsafe extern "C-unwind" fn with_source(
    this: *mut AnyObject,
    cmd: Sel,
    source: *mut AnyObject,
    options: *mut AnyObject,
    error: *mut *mut AnyObject,
) -> *mut AnyObject {
    let Some(call) = original::<WithSource>(&ORIGINAL_WITH_SOURCE) else {
        return std::ptr::null_mut();
    };
    timed(
        // SAFETY: the replaced implementation, called with its arguments.
        || unsafe { call(this, cmd, source, options, error) },
        || {
            // SAFETY: a null or live `NSString`.
            let length: usize = match unsafe { source.as_ref() } {
                Some(source) => unsafe { msg_send![source, length] },
                None => 0,
            };
            format!("source library ({length} UTF-16 units)")
        },
    )
}
