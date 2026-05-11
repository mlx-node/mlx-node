use std::fs::OpenOptions;
use std::io::Write;
use std::sync::OnceLock;

pub(crate) fn enabled() -> bool {
    trace_file().is_some()
}

pub(crate) fn env_flag_value_enabled(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

pub(crate) fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| env_flag_value_enabled(&value))
}

pub(crate) fn env_flag_value_or_default(value: Option<&str>, default_when_unset: bool) -> bool {
    value
        .map(env_flag_value_enabled)
        .unwrap_or(default_when_unset)
}

pub(crate) fn env_flag_enabled_or_default(name: &str, default_when_unset: bool) -> bool {
    std::env::var(name)
        .map(|value| env_flag_value_or_default(Some(&value), default_when_unset))
        .unwrap_or(default_when_unset)
}

fn trace_file() -> Option<&'static str> {
    static TRACE_FILE: OnceLock<Option<String>> = OnceLock::new();
    TRACE_FILE
        .get_or_init(|| {
            if !env_flag_enabled("MLX_INFERENCE_TRACE") {
                return None;
            }
            std::env::var("MLX_INFERENCE_TRACE_FILE")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
        .as_deref()
}

pub(crate) fn write(args: std::fmt::Arguments<'_>) {
    let Some(path) = trace_file() else {
        return;
    };
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{args}");
    }
}

pub(crate) fn elapsed_ms(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::{env_flag_value_enabled, env_flag_value_or_default};

    #[test]
    fn env_flag_value_enabled_accepts_only_explicit_truthy_values() {
        for value in ["1", "true", "TRUE", " yes ", "On"] {
            assert!(env_flag_value_enabled(value), "{value:?} should enable");
        }

        for value in ["", "0", "false", "no", "off", "abc", "enabled", "2"] {
            assert!(
                !env_flag_value_enabled(value),
                "{value:?} should not enable"
            );
        }
    }

    #[test]
    fn env_flag_value_or_default_uses_default_only_when_unset() {
        assert!(env_flag_value_or_default(None, true));
        assert!(!env_flag_value_or_default(None, false));
        assert!(env_flag_value_or_default(Some("on"), false));
        assert!(!env_flag_value_or_default(Some("abc"), true));
    }
}
