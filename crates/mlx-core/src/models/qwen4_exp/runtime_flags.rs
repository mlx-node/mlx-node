//! Cache settings during one forward, as the reference caches its feature
//! switches. Refresh between forwards so comparisons can still change flags.
use std::{ffi::CStr, marker::PhantomData, rc::Rc};

pub(super) struct Scope(PhantomData<Rc<()>>);

pub(super) fn scope() -> Scope {
    unsafe { mlx_sys::mlx_qwen4_flags_begin() };
    Scope(PhantomData)
}

impl Drop for Scope {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_qwen4_flags_end() };
    }
}

pub(super) fn is_zero(name: &'static CStr) -> bool {
    unsafe { mlx_sys::mlx_qwen4_flag_equals(name.as_ptr(), c"0".as_ptr()) }
}

pub(super) fn is_one(name: &'static CStr) -> bool {
    unsafe { mlx_sys::mlx_qwen4_flag_equals(name.as_ptr(), c"1".as_ptr()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_caches_and_refreshes() {
        // Environment mutation is isolated from the model/GPU test suite.
        const CHILD: &str = "MLX_QWEN4_SCOPE_TEST_CHILD";
        if std::env::var(CHILD).as_deref() != Ok("1") {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "models::qwen4_exp::runtime_flags::tests::scope_caches_and_refreshes",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("RUST_TEST_THREADS", "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        const NAME: &str = "MLX_QWEN4_SCOPE_TEST_VALUE";
        const KEY: &CStr = c"MLX_QWEN4_SCOPE_TEST_VALUE";
        // This child runs only this test. No GPU work or other environment
        // reader is started; the thread below joins before another mutation.
        unsafe {
            std::env::set_var("MLX_QWEN4_CACHE_FLAGS", "1");
            std::env::remove_var(NAME);
        }
        {
            let _outer = scope();
            assert!(!is_zero(KEY) && !is_one(KEY));
            unsafe { std::env::set_var(NAME, "0") };
            let _nested = scope();
            assert!(!is_zero(KEY) && !is_one(KEY));
        }
        assert!(is_zero(KEY));
        {
            let _outer = scope();
            assert!(is_zero(KEY));
            unsafe { std::env::set_var(NAME, "1") };
            std::thread::spawn(|| {
                let _other = scope();
                assert!(is_one(KEY));
            })
            .join()
            .unwrap();
            assert!(is_zero(KEY));
        }
        let early_return = || -> Result<(), ()> {
            let _scope = scope();
            assert!(is_one(KEY));
            Err(())?;
            Ok(())
        };
        assert!(early_return().is_err());
        unsafe { std::env::set_var(NAME, "invalid") };
        {
            let _scope = scope();
            assert!(!is_zero(KEY) && !is_one(KEY));
        }
        unsafe {
            std::env::set_var("MLX_QWEN4_CACHE_FLAGS", "0");
            std::env::set_var(NAME, "0");
        }
        let _scope = scope();
        assert!(is_zero(KEY));
        unsafe { std::env::set_var(NAME, "1") };
        assert!(is_one(KEY));
    }
}
