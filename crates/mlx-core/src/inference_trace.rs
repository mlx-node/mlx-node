use std::fs::OpenOptions;
use std::io::Write;
use std::sync::OnceLock;

pub(crate) fn enabled() -> bool {
    trace_file().is_some()
}

fn trace_file() -> Option<&'static str> {
    static TRACE_FILE: OnceLock<Option<String>> = OnceLock::new();
    TRACE_FILE
        .get_or_init(|| {
            let enabled = match std::env::var("MLX_INFERENCE_TRACE") {
                Ok(value) => matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                ),
                Err(_) => false,
            };
            if !enabled {
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
