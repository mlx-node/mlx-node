use napi_derive::module_init;

const LOG_FILTER_ENV: &str = "MLX_NODE_LOG";
const LOG_FILE_ENV: &str = "MLX_NODE_LOG_FILE";

#[module_init]
pub fn try_init_tracing() {
    use std::fs::OpenOptions;
    use std::path::Path;
    use std::str::FromStr;

    use tracing_subscriber::filter::Targets;
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::util::SubscriberInitExt;

    // Usage without the `regex` feature.
    // <https://github.com/tokio-rs/tracing/issues/1436#issuecomment-918528013>
    let targets = std::env::var(LOG_FILTER_ENV).map_or_else(
        |_| Targets::new(),
        |value| {
            Targets::from_str(&value).unwrap_or_else(|error| {
                eprintln!("invalid {LOG_FILTER_ENV} filter {value:?}: {error}");
                Targets::new()
            })
        },
    );

    let log_path = std::env::var(LOG_FILE_ENV)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());

    if let Some(log_path) = log_path {
        let path = Path::new(&log_path);
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            && let Err(error) = std::fs::create_dir_all(parent)
        {
            eprintln!("failed to create {LOG_FILE_ENV} parent for {log_path:?}: {error}");
        }

        let mut options = OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }

        match options.open(path) {
            Ok(file) => {
                // File mode is intentionally JSON: every normal `tracing`
                // event remains machine-readable for post-run analysis while
                // the default stderr subscriber below keeps its human format.
                // `File`'s MakeWriter implementation clones the descriptor per
                // event; tracing serializes each event into one complete line.
                let init_result = tracing_subscriber::registry()
                    .with(targets)
                    .with(
                        tracing_subscriber::fmt::layer()
                            .json()
                            .flatten_event(true)
                            .with_writer(file)
                            .with_ansi(false)
                            .with_target(true)
                            .with_file(true)
                            .with_line_number(true)
                            .with_thread_ids(true)
                            .with_thread_names(true),
                    )
                    .try_init();
                match init_result {
                    Ok(()) => tracing::info!(
                        target: "mlx_core::inference",
                        event = "tracing_ready",
                        log_file = %log_path,
                        "inference tracing subscriber initialized"
                    ),
                    Err(error) => eprintln!(
                        "failed to initialize {LOG_FILE_ENV}={log_path:?}: {error}; \
                         another global tracing subscriber may already be installed"
                    ),
                }
                return;
            }
            Err(error) => {
                eprintln!("failed to open {LOG_FILE_ENV}={log_path:?}: {error}; logging to stderr");
            }
        }
    }

    if let Err(error) = tracing_subscriber::registry()
        .with(targets)
        .with(tracing_subscriber::fmt::layer())
        .try_init()
    {
        eprintln!("failed to initialize mlx tracing subscriber: {error}");
    }
}
