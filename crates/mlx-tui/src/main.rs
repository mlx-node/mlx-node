//! MLX Training TUI
//!
//! A terminal user interface for monitoring and controlling GRPO training runs.
//! This binary wraps a Node.js training script and provides real-time visualization.
//!
//! Features:
//! - Auto-restart on crash with 5s countdown
//! - Automatically adds --resume flag on restart
//! - Press 'c' to cancel restart countdown

mod app;
mod commands;
mod logging;
mod messages;
mod ui;

use std::io;
use std::process::{ExitStatus, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use clap::Parser;
use color_eyre::eyre::{Result, eyre};
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers, MouseButton,
        MouseEventKind,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures::StreamExt;
use ratatui::{Terminal, backend::CrosstermBackend};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::time::interval;

use tracing::{debug, error};

use app::App;
use commands::{ControlCommand, send_command};
use messages::TrainingMessage;

/// Safely truncate a string for debug logging, respecting UTF-8 character boundaries
fn truncate_for_debug(s: &str, max_chars: usize) -> &str {
    if s.chars().count() <= max_chars {
        s
    } else {
        // Find the byte index of the max_chars-th character
        s.char_indices()
            .nth(max_chars)
            .map(|(idx, _)| &s[..idx])
            .unwrap_or(s)
    }
}

/// Parse log level from stderr output (tracing format or raw)
/// Returns (level, cleaned message)
fn parse_stderr_log_level(line: &str) -> (messages::LogLevel, String) {
    // Tracing format: "2024-01-15T10:30:00.123456Z  INFO mlx_core::grpo::engine: message"
    // or with colors: "\x1b[32m INFO\x1b[0m mlx_core: message"
    let line_upper = line.to_uppercase();

    // Check for log level keywords in the line
    if line_upper.contains(" INFO ")
        || line_upper.contains("INFO:")
        || line_upper.starts_with("INFO ")
    {
        // Extract message after the target (after first colon)
        let message = extract_log_message(line);
        (messages::LogLevel::Info, message)
    } else if line_upper.contains(" DEBUG ")
        || line_upper.contains("DEBUG:")
        || line_upper.starts_with("DEBUG ")
    {
        let message = extract_log_message(line);
        (messages::LogLevel::Debug, message)
    } else if line_upper.contains(" WARN ")
        || line_upper.contains("WARN:")
        || line_upper.starts_with("WARN ")
    {
        let message = extract_log_message(line);
        (messages::LogLevel::Warn, message)
    } else if line_upper.contains(" ERROR ")
        || line_upper.contains("ERROR:")
        || line_upper.starts_with("ERROR ")
    {
        let message = extract_log_message(line);
        (messages::LogLevel::Error, message)
    } else {
        // Unknown format, show as warn (original behavior)
        (messages::LogLevel::Warn, line.to_string())
    }
}

/// Extract the actual log message from tracing format
/// "2024-01-15T10:30:00Z  INFO mlx_core::grpo::engine: Phase 1: Generating..."
/// -> "Phase 1: Generating..."
fn extract_log_message(line: &str) -> String {
    // Strip ANSI color codes
    let stripped = strip_ansi_codes(line);

    // Try to find pattern: "LEVEL target: message"
    // Look for the target module path followed by colon
    if let Some(idx) = stripped.find("mlx_core") {
        // Find the colon after the target
        if let Some(colon_idx) = stripped[idx..].find(": ") {
            let msg_start = idx + colon_idx + 2;
            if msg_start < stripped.len() {
                return stripped[msg_start..].trim().to_string();
            }
        }
    }

    // Fallback: just return the stripped line
    stripped.trim().to_string()
}

/// Strip ANSI escape codes from a string
fn strip_ansi_codes(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip until we hit a letter (end of escape sequence)
            while let Some(&next) = chars.peek() {
                chars.next();
                if next.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Global child process ID for cleanup on panic/signal
static CHILD_PID: AtomicU32 = AtomicU32::new(0);

/// Guard that kills the child process on drop (panic cleanup)
struct ChildGuard;

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let pid = CHILD_PID.load(Ordering::SeqCst);
        if pid != 0 {
            // Kill the child process group synchronously on drop
            #[cfg(unix)]
            {
                // Send SIGTERM to the entire process group (negative pid)
                // This ensures child processes spawned by Node.js are also killed
                unsafe {
                    libc::kill(-(pid as libc::pid_t), libc::SIGTERM);
                }
                // Give it a moment to terminate gracefully
                std::thread::sleep(std::time::Duration::from_millis(100));
                // Force kill the process group if still running
                unsafe {
                    libc::kill(-(pid as libc::pid_t), libc::SIGKILL);
                }
            }
            #[cfg(not(unix))]
            {
                // On Windows, use /T flag to kill the process tree
                if let Ok(mut child) = std::process::Command::new("taskkill")
                    .args(["/F", "/T", "/PID", &pid.to_string()])
                    .spawn()
                {
                    let _ = child.wait();
                }
            }
            CHILD_PID.store(0, Ordering::SeqCst);
        }
    }
}

/// Restart countdown duration in seconds
const RESTART_COUNTDOWN_SECS: u8 = 5;

/// MLX Training TUI - Monitor and control GRPO training
#[derive(Parser, Clone)]
#[command(name = "mlx-train")]
#[command(about = "TUI for monitoring and controlling MLX-Node GRPO training")]
#[command(version)]
struct Cli {
    /// Path to the training script (optional if --db is used)
    #[arg(short, long)]
    script: Option<String>,

    /// Path to SQLite database to browse (standalone mode)
    #[arg(long)]
    db: Option<String>,

    /// Working directory for the training script
    #[arg(short = 'd', long)]
    workdir: Option<String>,

    /// Node.js --import flag(s) to load before the script (e.g., tsx for TypeScript)
    #[arg(short, long, action = clap::ArgAction::Append)]
    import: Vec<String>,

    /// Disable auto-restart on crash
    #[arg(long)]
    no_auto_restart: bool,

    /// Additional arguments to pass to the training script
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<String>,
}

/// Holds references to the spawned child process I/O
struct ChildProcess {
    child: Child,
    stdin: tokio::process::ChildStdin,
    stdout_reader: tokio::io::Lines<BufReader<tokio::process::ChildStdout>>,
    stderr_reader: tokio::io::Lines<BufReader<tokio::process::ChildStderr>>,
}

/// Spawn a new training process
fn spawn_training_process(cli: &Cli, is_restart: bool) -> Result<ChildProcess> {
    debug!(is_restart, "spawn_training_process called");
    let script = cli.script.as_ref().ok_or_else(|| {
        eyre!("No training script specified. Use --script or --db for standalone mode")
    })?;
    debug!(%script, "Script path");

    let mut cmd = Command::new("node");

    // Add --import flags to node (e.g., --import tsx for TypeScript)
    for import in &cli.import {
        cmd.arg("--import").arg(import);
    }

    cmd.arg(script);

    // Add original args
    for arg in &cli.args {
        cmd.arg(arg);
    }

    // Add --resume flag on restart if not already present
    if is_restart && !cli.args.iter().any(|a| a == "--resume" || a == "-r") {
        cmd.arg("--resume");
    }

    // Note: Prompt responses (like training targets) are replayed automatically
    // when the training script sends the same prompt again on restart.
    // See the Prompt message handler in run_event_loop.

    // Set up process group for clean termination (Unix only)
    #[cfg(unix)]
    unsafe {
        cmd.pre_exec(|| {
            // Set process group to self, so we can kill the whole group
            libc::setpgid(0, 0);
            Ok(())
        });
    }

    cmd.env("MLX_TUI_MODE", "1") // Signal to use TUI-compatible output
        .env("MLX_NODE_LOG", "mlx_core=debug") // Enable Rust debug logging for training phases
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(ref workdir) = cli.workdir {
        cmd.current_dir(workdir);
    }

    debug!("Spawning child process...");
    let mut child = cmd.spawn().map_err(|e| {
        debug!(%e, "Failed to spawn");
        eyre!("Failed to spawn training script: {e}")
    })?;

    // Store child PID for cleanup on panic
    if let Some(pid) = child.id() {
        debug!(pid, "Child spawned");
        CHILD_PID.store(pid, Ordering::SeqCst);
    }

    let stdin = child.stdin.take().expect("Failed to get stdin");
    let stdout_pipe = child.stdout.take().expect("Failed to get stdout");
    let stderr_pipe = child.stderr.take().expect("Failed to get stderr");

    let stdout_reader = BufReader::new(stdout_pipe).lines();
    let stderr_reader = BufReader::new(stderr_pipe).lines();

    Ok(ChildProcess {
        child,
        stdin,
        stdout_reader,
        stderr_reader,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize unified logging - hold guard to ensure logs flush before runtime shutdown
    let _log_guard = logging::init();

    // Set up panic hook to log panics
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        error!("PANIC: {}", info);
        // Also try to restore terminal
        let _ = crossterm::terminal::disable_raw_mode();
        let _ = crossterm::execute!(
            std::io::stdout(),
            crossterm::terminal::LeaveAlternateScreen,
            crossterm::event::DisableMouseCapture
        );
        default_hook(info);
    }));

    debug!("=== TUI Starting ===");

    color_eyre::install()?;
    let cli = Cli::parse();
    debug!(script = ?cli.script, "CLI parsed");

    // Validate arguments
    if cli.script.is_none() && cli.db.is_none() {
        return Err(eyre!("Either --script or --db must be provided"));
    }

    // Create the child guard - this ensures child cleanup on panic or early exit
    // It will be dropped last due to LIFO drop order, cleaning up any orphaned child
    let _child_guard = ChildGuard;

    // Setup signal handlers for graceful shutdown
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm = signal(SignalKind::terminate())?;
        let mut sigint = signal(SignalKind::interrupt())?;

        tokio::spawn(async move {
            tokio::select! {
                _ = sigterm.recv() => {},
                _ = sigint.recv() => {},
            }
            // Restore terminal state before exiting
            let _ = crossterm::terminal::disable_raw_mode();
            let _ = crossterm::execute!(
                std::io::stdout(),
                crossterm::terminal::LeaveAlternateScreen,
                crossterm::event::DisableMouseCapture
            );
            // Kill child process group on signal (negative pid kills the group)
            let pid = CHILD_PID.load(Ordering::SeqCst);
            if pid != 0 {
                unsafe {
                    libc::kill(-(pid as libc::pid_t), libc::SIGTERM);
                }
            }
            std::process::exit(130); // Standard exit code for SIGINT
        });
    }

    // Setup terminal
    debug!("Setting up terminal...");
    enable_raw_mode()?;
    debug!("Raw mode enabled");
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    debug!("Alternate screen enabled");
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    debug!("Terminal created");

    let mut app = App::new();
    app.auto_restart_enabled = !cli.no_auto_restart;
    debug!("App created");

    // Handle standalone database mode
    if let (None, Some(db_path)) = (&cli.script, &cli.db) {
        let db_path = db_path.clone();
        app.handle_message(TrainingMessage::Log {
            level: messages::LogLevel::Info,
            message: format!("Opening database: {}", db_path),
        });
        // Open database and get most recent run via spawn_blocking to avoid nested runtime panic
        // (SyncDbReader::open and list_runs use block_on internally)
        match tokio::task::spawn_blocking({
            let path = db_path.clone();
            move || {
                let reader = mlx_db::SyncDbReader::open(&path)?;
                // Get most recent run for standalone browsing
                let runs = reader.block_on(mlx_db::list_runs(reader.pool(), Some(1), None))?;
                let recent_run = runs.into_iter().next();
                Ok::<_, mlx_db::DbError>((reader, recent_run))
            }
        })
        .await
        {
            Ok(Ok((reader, recent_run))) => {
                app.db_reader = Some(reader);
                app.db_path = Some(db_path);
                // Set run_id from most recent run (required for loading generations)
                if let Some(run) = recent_run {
                    let run_name = run.name.clone().unwrap_or_else(|| run.id.clone());
                    app.handle_message(TrainingMessage::Log {
                        level: messages::LogLevel::Info,
                        message: format!("Selected run: {}", run_name),
                    });
                    app.db_run_id = Some(run.id);
                    app.pending_db_action = Some(app::DbAction::RefreshGenerations);
                } else {
                    app.handle_message(TrainingMessage::Log {
                        level: messages::LogLevel::Warn,
                        message: "No training runs found in database".to_string(),
                    });
                }
            }
            Ok(Err(e)) => {
                app.handle_message(TrainingMessage::Log {
                    level: messages::LogLevel::Error,
                    message: format!("Failed to open database: {}", e),
                });
            }
            Err(e) => {
                app.handle_message(TrainingMessage::Log {
                    level: messages::LogLevel::Error,
                    message: format!("Database task error: {}", e),
                });
            }
        }
        // Set standalone mode state
        app.state = app::TrainingState::Complete;
        app.child_exited = true;

        // Run in standalone mode (just viewing DB, no training process)
        let result = run_standalone_mode(&mut terminal, &mut app).await;

        // Drop db_reader via spawn_blocking to avoid nested runtime panic
        if let Some(reader) = app.db_reader.take() {
            let _ = tokio::task::spawn_blocking(move || drop(reader)).await;
        }

        // Cleanup terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        return result;
    }

    // Add initial log
    app.handle_message(TrainingMessage::Log {
        level: messages::LogLevel::Info,
        message: format!(
            "Starting: {} {}",
            cli.script.as_deref().unwrap_or(""),
            cli.args.join(" ")
        ),
    });

    // If --db is also provided, open the database via spawn_blocking
    // (SyncDbReader::open uses block_on internally which can't run in async context)
    // Note: Don't trigger RefreshGenerations here - the training script will send
    // a DatabasePath message with the run_id, which triggers the refresh
    if let Some(ref db_path) = cli.db {
        let path = db_path.clone();
        match tokio::task::spawn_blocking(move || mlx_db::SyncDbReader::open(&path)).await {
            Ok(Ok(reader)) => {
                app.db_reader = Some(reader);
                app.db_path = Some(db_path.clone());
                // Don't set pending_db_action here - wait for DatabasePath message with run_id
                app.handle_message(TrainingMessage::Log {
                    level: messages::LogLevel::Info,
                    message: format!("Opened database: {}", db_path),
                });
            }
            Ok(Err(e)) => {
                app.handle_message(TrainingMessage::Log {
                    level: messages::LogLevel::Error,
                    message: format!("Failed to open database: {}", e),
                });
            }
            Err(e) => {
                app.handle_message(TrainingMessage::Log {
                    level: messages::LogLevel::Error,
                    message: format!("Database task error: {}", e),
                });
            }
        }
    }

    // Outer loop for restart handling
    let result = run_with_restart(&mut terminal, &mut app, &cli).await;

    // Drop db_reader via spawn_blocking to avoid nested runtime panic
    // SyncDbReader contains its own tokio runtime which can't be dropped in async context
    if let Some(reader) = app.db_reader.take() {
        let _ = tokio::task::spawn_blocking(move || drop(reader)).await;
    }

    // Cleanup terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    result
}

/// Run the training process with automatic restart on crash
async fn run_with_restart(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    cli: &Cli,
) -> Result<()> {
    debug!("run_with_restart called");
    let mut is_restart = false;

    loop {
        debug!("Main loop iteration");
        // Spawn the training process
        // Note: Prompt responses are replayed automatically when the script asks again
        let process = match spawn_training_process(cli, is_restart) {
            Ok(p) => p,
            Err(e) => {
                app.handle_message(TrainingMessage::Log {
                    level: messages::LogLevel::Error,
                    message: format!("Failed to spawn process: {e}"),
                });

                // Also allow restart on spawn failure
                if app.auto_restart_enabled {
                    if !run_restart_countdown(terminal, app).await? {
                        wait_for_quit(terminal, app).await?;
                        return Ok(());
                    }
                    // Drop db_reader via spawn_blocking to avoid nested runtime panic
                    if let Some(reader) = app.prepare_for_restart() {
                        let _ = tokio::task::spawn_blocking(move || drop(reader)).await;
                    }
                    is_restart = true;
                    continue;
                } else {
                    app.state = app::TrainingState::Error;
                    app.child_exited = true;
                    wait_for_quit(terminal, app).await?;
                    return Ok(());
                }
            }
        };

        let ChildProcess {
            mut child,
            mut stdin,
            mut stdout_reader,
            mut stderr_reader,
        } = process;

        if is_restart {
            app.handle_message(TrainingMessage::Log {
                level: messages::LogLevel::Info,
                message: format!(
                    "Restarted with --resume flag (restart #{})",
                    app.restart_count
                ),
            });
        }

        // Run the main event loop
        let exit_status = run_event_loop(
            terminal,
            app,
            &mut child,
            &mut stdin,
            &mut stdout_reader,
            &mut stderr_reader,
        )
        .await;

        // Kill child process group if still running and clear the global PID
        // Use process group kill to ensure spawned subprocesses are also terminated
        let pid = CHILD_PID.load(Ordering::SeqCst);
        if pid != 0 {
            #[cfg(unix)]
            unsafe {
                // Kill the entire process group (negative pid)
                libc::kill(-(pid as libc::pid_t), libc::SIGTERM);
            }
            #[cfg(not(unix))]
            {
                // On Windows, use /T flag to kill the process tree
                let _ = std::process::Command::new("taskkill")
                    .args(["/F", "/T", "/PID", &pid.to_string()])
                    .spawn();
            }
        }
        // Also call tokio's kill to clean up the child handle
        let _ = child.kill().await;
        CHILD_PID.store(0, Ordering::SeqCst);

        // Check if user requested quit
        if app.should_quit {
            return Ok(());
        }

        // Handle exit status
        debug!(?exit_status, "Handling exit status from run_event_loop");
        match exit_status {
            Ok(Some(status)) => {
                let code = status.code();
                app.last_exit_code = code;
                debug!(
                    ?code,
                    success = status.success(),
                    "Process exited with status"
                );

                if status.success() {
                    // Clean exit - don't restart
                    app.handle_message(TrainingMessage::Log {
                        level: messages::LogLevel::Info,
                        message: "Training process exited successfully".to_string(),
                    });
                    app.state = app::TrainingState::Complete;
                    wait_for_quit(terminal, app).await?;
                    return Ok(());
                } else {
                    // Non-zero exit - potentially restart
                    debug!(
                        "Adding crash error log. Current log count: {}, error count: {}",
                        app.logs.len(),
                        app.logs
                            .iter()
                            .filter(|e| e.level == messages::LogLevel::Error)
                            .count()
                    );
                    app.handle_message(TrainingMessage::Log {
                        level: messages::LogLevel::Error,
                        message: format!(
                            "Training process crashed (exit code: {})",
                            code.map_or("unknown".to_string(), |c| c.to_string())
                        ),
                    });
                    debug!(
                        "After adding crash error log. Current log count: {}, error count: {}",
                        app.logs.len(),
                        app.logs
                            .iter()
                            .filter(|e| e.level == messages::LogLevel::Error)
                            .count()
                    );

                    if app.auto_restart_enabled {
                        // Start countdown and wait
                        if !run_restart_countdown(terminal, app).await? {
                            // User cancelled restart
                            wait_for_quit(terminal, app).await?;
                            return Ok(());
                        }
                        // Drop db_reader via spawn_blocking to avoid nested runtime panic
                        if let Some(reader) = app.prepare_for_restart() {
                            let _ = tokio::task::spawn_blocking(move || drop(reader)).await;
                        }
                        is_restart = true;
                        continue;
                    } else {
                        app.state = app::TrainingState::Error;
                        wait_for_quit(terminal, app).await?;
                        return Ok(());
                    }
                }
            }
            Ok(None) => {
                // Process ended but child.wait() failed - treat as crash
                // This happens when the child process exits abnormally and wait() fails
                debug!("Ok(None) exit status - child.wait() failed. Adding error log.");
                app.handle_message(TrainingMessage::Log {
                    level: messages::LogLevel::Error,
                    message: "Training process crashed (exit status unavailable)".to_string(),
                });

                // Also trigger restart on unexpected exit
                if app.auto_restart_enabled {
                    if !run_restart_countdown(terminal, app).await? {
                        wait_for_quit(terminal, app).await?;
                        return Ok(());
                    }
                    // Drop db_reader via spawn_blocking to avoid nested runtime panic
                    if let Some(reader) = app.prepare_for_restart() {
                        let _ = tokio::task::spawn_blocking(move || drop(reader)).await;
                    }
                    is_restart = true;
                    continue;
                } else {
                    wait_for_quit(terminal, app).await?;
                    return Ok(());
                }
            }
            Err(e) => {
                app.handle_message(TrainingMessage::Log {
                    level: messages::LogLevel::Error,
                    message: format!("Error: {e}"),
                });

                // Also trigger restart on event loop errors
                if app.auto_restart_enabled {
                    if !run_restart_countdown(terminal, app).await? {
                        wait_for_quit(terminal, app).await?;
                        return Ok(());
                    }
                    // Drop db_reader via spawn_blocking to avoid nested runtime panic
                    if let Some(reader) = app.prepare_for_restart() {
                        let _ = tokio::task::spawn_blocking(move || drop(reader)).await;
                    }
                    is_restart = true;
                    continue;
                } else {
                    app.state = app::TrainingState::Error;
                    wait_for_quit(terminal, app).await?;
                    return Ok(());
                }
            }
        }
    }
}

/// Run the restart countdown, returns true if restart should proceed
async fn run_restart_countdown(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<bool> {
    app.start_restart_countdown(RESTART_COUNTDOWN_SECS);

    let mut ticker = interval(Duration::from_secs(1));
    let mut event_stream = crossterm::event::EventStream::new();

    loop {
        // Render
        terminal.draw(|f| ui::draw(f, app))?;

        tokio::select! {
            biased;

            // Handle keyboard events
            maybe_event = event_stream.next() => {
                if let Some(Ok(Event::Key(key))) = maybe_event {
                    match key.code {
                        KeyCode::Char('c') => {
                            // Cancel restart
                            app.cancel_restart();
                            return Ok(false);
                        }
                        KeyCode::Char('q') | KeyCode::Esc => {
                            // Quit entirely
                            app.should_quit = true;
                            return Ok(false);
                        }
                        KeyCode::Enter => {
                            // Skip countdown and restart now
                            return Ok(true);
                        }
                        _ => {}
                    }
                }
            }

            // Tick the countdown
            _ = ticker.tick() => {
                if app.tick_restart_countdown() {
                    // Countdown reached zero
                    return Ok(true);
                }
            }
        }
    }
}

/// Wait for user to quit (after training completes or errors)
async fn wait_for_quit(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<()> {
    app.child_exited = true;
    let mut event_stream = crossterm::event::EventStream::new();

    loop {
        terminal.draw(|f| ui::draw(f, app))?;

        if let Some(Ok(Event::Key(key))) = event_stream.next().await {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => {
                    app.should_quit = true;
                    return Ok(());
                }
                _ => {}
            }
        }
    }
}

/// Process a pending database action via spawn_blocking
async fn process_db_action(app: &mut App, action: app::DbAction) {
    // We need to clone the data we need for the blocking task
    let Some(reader) = app.db_reader.take() else {
        return;
    };
    let run_id = app.db_run_id.clone();
    let filter = app.db_filter.clone();

    let result = tokio::task::spawn_blocking(move || {
        match action {
            app::DbAction::RefreshGenerations => {
                let Some(run_id) = run_id else {
                    return (reader, Err("No run_id".to_string()));
                };
                let mut filter = filter;
                if filter.limit.is_none() {
                    filter.limit = Some(100);
                }
                match reader.block_on(mlx_db::get_generations_filtered(
                    reader.pool(),
                    &run_id,
                    &filter,
                )) {
                    Ok((gens, total)) => {
                        let records: Vec<_> = gens.into_iter().map(|g| g.generation).collect();
                        (reader, Ok(DbResult::Generations(records, total)))
                    }
                    Err(e) => (reader, Err(format!("Failed to load generations: {e}"))),
                }
            }
            app::DbAction::LoadMetrics => {
                let Some(run_id) = run_id else {
                    return (reader, Err("No run_id for metrics".to_string()));
                };
                // Load all step summaries to restore sparkline history
                match reader.block_on(mlx_db::get_step_summaries(
                    reader.pool(),
                    &run_id,
                    None,
                    None,
                )) {
                    Ok(summaries) => {
                        let metrics: Vec<(f64, f64, f64)> = summaries
                            .into_iter()
                            .map(|s| (s.loss, s.mean_reward, s.std_advantage))
                            .collect();
                        (reader, Ok(DbResult::Metrics(metrics)))
                    }
                    Err(e) => (reader, Err(format!("Failed to load metrics: {e}"))),
                }
            }
            app::DbAction::LoadHistoricalLogs => {
                let Some(run_id) = run_id else {
                    return (reader, Err("No run_id for logs".to_string()));
                };
                // Load logs from database for this run (info level and above)
                match reader.block_on(mlx_db::get_logs(
                    reader.pool(),
                    Some(&run_id),
                    Some("INFO"),
                    500,
                    0,
                )) {
                    Ok(logs) => (reader, Ok(DbResult::HistoricalLogs(logs))),
                    Err(e) => (reader, Err(format!("Failed to load logs: {e}"))),
                }
            }
            app::DbAction::LoadHistoricalSamples => {
                let Some(run_id) = run_id else {
                    return (reader, Err("No run_id for samples".to_string()));
                };
                // Load recent generations from database for samples panel
                match reader.block_on(mlx_db::get_recent_generations(reader.pool(), &run_id, 50)) {
                    Ok(gens) => (reader, Ok(DbResult::HistoricalSamples(gens))),
                    Err(e) => (reader, Err(format!("Failed to load samples: {e}"))),
                }
            }
        }
    })
    .await;

    match result {
        Ok((reader, Ok(db_result))) => {
            app.db_reader = Some(reader);
            match db_result {
                DbResult::Generations(gens, total) => {
                    app.set_generations(gens, total);
                }
                DbResult::Metrics(metrics) => {
                    app.restore_metrics_history(metrics);
                }
                DbResult::HistoricalLogs(logs) => {
                    let count = logs.len();
                    app.restore_historical_logs(logs);
                    if count > 0 {
                        app.add_log(
                            messages::LogLevel::Info,
                            format!("Restored {} historical log entries", count),
                        );
                    }
                    // Chain: now load historical samples
                    if app.resume_state_received
                        && app.samples.is_empty()
                        && app.pending_db_action.is_none()
                    {
                        app.pending_db_action = Some(app::DbAction::LoadHistoricalSamples);
                    }
                }
                DbResult::HistoricalSamples(gens) => {
                    let count = gens.len();
                    app.restore_historical_samples(gens);
                    app.add_log(
                        messages::LogLevel::Info,
                        format!("Restored {} historical samples", count),
                    );
                }
            }
        }
        Ok((reader, Err(e))) => {
            app.db_reader = Some(reader);
            app.add_log(messages::LogLevel::Error, e);
        }
        Err(e) => {
            app.add_log(messages::LogLevel::Error, format!("Task error: {e}"));
        }
    }
}

/// Result type for database operations
enum DbResult {
    Generations(Vec<mlx_db::GenerationRecord>, usize),
    /// Historical metrics for sparkline restoration: Vec<(loss, reward, advantage)>
    Metrics(Vec<(f64, f64, f64)>),
    /// Historical logs for resume
    HistoricalLogs(Vec<mlx_db::LogRecord>),
    /// Historical samples (generations) for resume
    HistoricalSamples(Vec<mlx_db::GenerationRecord>),
}

/// Run in standalone database browsing mode (no training process)
async fn run_standalone_mode(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<()> {
    let mut event_stream = crossterm::event::EventStream::new();

    loop {
        terminal.draw(|f| ui::draw(f, app))?;

        if app.should_quit {
            return Ok(());
        }

        // Process pending DB action via spawn_blocking
        if let Some(action) = app.take_pending_db_action() {
            process_db_action(app, action).await;
        }

        if let Some(Ok(event)) = event_stream.next().await {
            match event {
                Event::Key(key) if handle_standalone_key(key, app) => {
                    return Ok(());
                }
                Event::Mouse(mouse) => {
                    handle_mouse(mouse, app);
                }
                _ => {}
            }
        }
    }
}

/// Handle keyboard input in standalone mode, returns true if should quit
fn handle_standalone_key(key: event::KeyEvent, app: &mut App) -> bool {
    // Handle help overlay
    if app.show_help {
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc | KeyCode::Char('?') => {
                app.show_help = false;
            }
            _ => {}
        }
        return false;
    }

    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => {
            app.should_quit = true;
            return true;
        }

        // Tab navigation
        KeyCode::Tab => app.next_tab(),
        KeyCode::BackTab => app.prev_tab(),
        KeyCode::Char('1') => app.switch_to_tab(app::ActiveTab::Logs),
        KeyCode::Char('2') => app.switch_to_tab(app::ActiveTab::Samples),
        KeyCode::Char('3') => app.switch_to_tab(app::ActiveTab::Config),

        // Scrolling
        KeyCode::Up | KeyCode::Char('k') => app.scroll_up(),
        KeyCode::Down | KeyCode::Char('j') => app.scroll_down(),
        KeyCode::PageUp => app.page_up(),
        KeyCode::PageDown => app.page_down(),
        KeyCode::Char('g') => app.scroll_to_top(),
        KeyCode::Char('G') => app.scroll_to_bottom(),

        // Help
        KeyCode::Char('?') => app.toggle_help(),

        _ => {}
    }

    false
}

async fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    child: &mut Child,
    stdin: &mut tokio::process::ChildStdin,
    stdout_reader: &mut tokio::io::Lines<BufReader<tokio::process::ChildStdout>>,
    stderr_reader: &mut tokio::io::Lines<BufReader<tokio::process::ChildStderr>>,
) -> Result<Option<ExitStatus>> {
    debug!("run_event_loop started");
    let mut event_stream = crossterm::event::EventStream::new();
    let mut frame_count = 0u64;

    loop {
        // Render
        if frame_count == 0 {
            debug!("First frame render...");
        }
        terminal.draw(|f| ui::draw(f, app))?;
        if frame_count == 0 {
            debug!("First frame rendered successfully");
        }
        frame_count += 1;

        // Log all logs added to app
        if !app.logs.is_empty() && frame_count <= 5 {
            for log in app.logs.iter() {
                debug!(level = ?log.level, message = %log.message, "APP LOG");
            }
        }

        if app.should_quit {
            // Try to get exit status if available
            let status = child.try_wait().ok().flatten();
            return Ok(status);
        }

        // Process pending DB action via spawn_blocking
        if let Some(action) = app.take_pending_db_action() {
            process_db_action(app, action).await;
        }

        // Handle events using tokio::select!
        // Once child has exited, return to let outer loop handle restart
        if app.child_exited {
            // Get the exit status
            let status = child.try_wait().ok().flatten();
            return Ok(status);
        }

        // Child is running - handle all events
        tokio::select! {
            biased;

            // Keyboard and mouse events (highest priority for responsive UI)
            maybe_event = event_stream.next() => {
                if let Some(Ok(event)) = maybe_event {
                    match event {
                        Event::Key(key) => {
                            match handle_key(key, app, stdin).await {
                                Ok(true) => {
                                    let status = child.try_wait().ok().flatten();
                                    return Ok(status);
                                }
                                Ok(false) => {}
                                Err(e) => {
                                    app.handle_message(TrainingMessage::Log {
                                        level: messages::LogLevel::Error,
                                        message: format!("Command error: {e}"),
                                    });
                                }
                            }
                        }
                        Event::Mouse(mouse) => {
                            handle_mouse(mouse, app);
                        }
                        _ => {}
                    }
                }
            }

            // Child stdout (JSONL messages)
            maybe_line = stdout_reader.next_line() => {
                match maybe_line {
                    Ok(Some(line)) => {
                        debug!(stdout = %truncate_for_debug(&line, 200), "STDOUT");
                        // Try to parse as JSONL message
                        match serde_json::from_str::<TrainingMessage>(&line) {
                            Ok(msg) => {
                                debug!(msg_type = ?std::mem::discriminant(&msg), "Parsed message successfully");
                                // Handle Prompt specially - check for cached response to auto-reply on restart
                                if let TrainingMessage::Prompt { ref id, .. } = msg
                                    && let Some(cached_value) = app.captured_prompt_responses.get(id).cloned() {
                                        // Auto-reply with cached value from previous run
                                        debug!(%id, %cached_value, "Auto-replying to prompt with cached value");
                                        app.add_log(messages::LogLevel::Info, format!("Auto-replied to '{}' (cached)", id));
                                        let _ = send_command(stdin, ControlCommand::PromptResponse {
                                            id: id.clone(),
                                            value: cached_value,
                                        }).await;
                                        continue; // Don't show prompt UI
                                    }
                                    // No cached response - fall through to handle_message

                                // Handle DatabasePath specially - it needs spawn_blocking
                                // because SyncDbReader::open calls block_on which can't run in async context
                                if let TrainingMessage::DatabasePath { ref path, ref run_id, .. } = msg {
                                    debug!(%path, %run_id, "DatabasePath received");
                                    let path_clone = path.clone();
                                    let run_id_clone = run_id.clone();
                                    let filter = app.db_filter.clone();
                                    // Open database and fetch initial generations in blocking task
                                    match tokio::task::spawn_blocking(move || {
                                        let reader = mlx_db::SyncDbReader::open(&path_clone)?;
                                        // Also fetch initial generations while in blocking context
                                        let mut filter = filter;
                                        if filter.limit.is_none() {
                                            filter.limit = Some(100);
                                        }
                                        let result = reader.block_on(
                                            mlx_db::get_generations_filtered(reader.pool(), &run_id_clone, &filter)
                                        )?;
                                        Ok::<_, mlx_db::DbError>((reader, result))
                                    }).await {
                                        Ok(Ok((reader, (generations_with_tools, total)))) => {
                                            debug!(count = generations_with_tools.len(), total, "Database opened with generations");
                                            // Drop old reader via spawn_blocking to avoid nested runtime panic
                                            if let Some(old_reader) = app.db_reader.take() {
                                                let _ = tokio::task::spawn_blocking(move || drop(old_reader)).await;
                                            }
                                            app.db_reader = Some(reader);
                                            app.db_path = Some(path.clone());
                                            if let TrainingMessage::DatabasePath { run_id, run_name, .. } = &msg {
                                                app.db_run_id = Some(run_id.clone());
                                                if let Some(name) = run_name {
                                                    app.add_log(messages::LogLevel::Info, format!("Database: {name}"));
                                                } else {
                                                    app.add_log(messages::LogLevel::Info, format!("Database run: {run_id}"));
                                                }
                                            }
                                            // Store generations
                                            app.db_total_count = total;
                                            let gen_count = generations_with_tools.len();
                                            app.db_generations = generations_with_tools
                                                .into_iter()
                                                .map(|g| g.generation)
                                                .collect();
                                            app.add_log(messages::LogLevel::Info, format!("Loaded {} generations (total: {})", gen_count, total));
                                            debug!(generations = app.db_generations.len(), "Database setup complete");

                                            // Handle resume state restoration
                                            if app.resume_state_received {
                                                // ResumeState already populated sparklines, so load logs and samples instead
                                                if app.logs.is_empty() && app.pending_db_action.is_none() {
                                                    // Load historical logs first, then samples
                                                    app.pending_db_action = Some(app::DbAction::LoadHistoricalLogs);
                                                }
                                            } else if app.loss_history.is_empty() && app.pending_db_action.is_none() {
                                                // Fallback: load metrics if sparklines are empty
                                                app.pending_db_action = Some(app::DbAction::LoadMetrics);
                                            }
                                        }
                                        Ok(Err(e)) => {
                                            debug!(%e, "Database open error");
                                            app.add_log(messages::LogLevel::Error, format!("DB error: {}", e));
                                        }
                                        Err(e) => {
                                            debug!(%e, "spawn_blocking error");
                                            app.add_log(messages::LogLevel::Error, format!("DB task error: {}", e));
                                        }
                                    }
                                } else {
                                    app.handle_message(msg);
                                }
                            }
                            Err(e) => {
                                // Log the parsing error for debugging
                                debug!(%e, line = %truncate_for_debug(&line, 500), "JSON parse failed");
                                // Non-JSON output, treat as log
                                app.handle_message(TrainingMessage::Log {
                                    level: messages::LogLevel::Info,
                                    message: line.to_string(),
                                });
                            }
                        }
                    }
                    Ok(None) => {
                        debug!("STDOUT closed - child process ended");
                        // Child process ended - drain any remaining stderr before returning
                        app.child_exited = true;

                        // Drain remaining stderr (error output from crash)
                        while let Ok(Some(line)) = stderr_reader.next_line().await {
                            if !line.trim().is_empty() {
                                let (level, message) = parse_stderr_log_level(&line);
                                app.handle_message(TrainingMessage::Log {
                                    level,
                                    message,
                                });
                            }
                        }

                        // Wait for process to fully exit
                        let wait_result = child.wait().await;
                        debug!(?wait_result, "Child wait() result");
                        match &wait_result {
                            Ok(status) => {
                                debug!(?status, success = status.success(), code = ?status.code(), "Child exit status");
                            }
                            Err(e) => {
                                debug!(%e, "child.wait() failed - will return Ok(None)");
                            }
                        }
                        let status = wait_result.ok();
                        return Ok(status);
                    }
                    Err(e) => {
                        debug!(%e, "STDOUT read error");
                        app.handle_message(TrainingMessage::Log {
                            level: messages::LogLevel::Error,
                            message: format!("Read error: {e}"),
                        });
                        app.state = app::TrainingState::Error;
                        app.child_exited = true;

                        // Drain remaining stderr (error output from crash)
                        while let Ok(Some(line)) = stderr_reader.next_line().await {
                            if !line.trim().is_empty() {
                                let (level, message) = parse_stderr_log_level(&line);
                                app.handle_message(TrainingMessage::Log {
                                    level,
                                    message,
                                });
                            }
                        }

                        let status = child.wait().await.ok();
                        return Ok(status);
                    }
                }
            }

            // Child stderr (parse tracing log level or default to warn)
            maybe_line = stderr_reader.next_line() => {
                match maybe_line {
                    Ok(Some(line)) => {
                        debug!(stderr = %truncate_for_debug(&line, 200), "STDERR");
                        // Skip empty lines
                        if !line.trim().is_empty() {
                            // Parse tracing format: "2024-01-15T10:30:00Z  INFO target: message"
                            // or simpler: " INFO ..." or "INFO ..." or "WARN ..." etc.
                            let (level, message) = parse_stderr_log_level(&line);
                            app.handle_message(TrainingMessage::Log {
                                level,
                                message,
                            });
                        }
                    }
                    Ok(None) => {
                        debug!("STDERR closed");
                        // Stderr closed - ignore
                    }
                    Err(_) => {} // Ignore stderr errors
                }
            }
        }
    }
}

/// Handle keyboard input, returns true if should quit
async fn handle_key(
    key: event::KeyEvent,
    app: &mut App,
    stdin: &mut tokio::process::ChildStdin,
) -> Result<bool> {
    // Handle interactive prompt (highest priority - blocks all other input)
    if app.has_active_prompt() {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                app.prompt_select_prev();
            }
            KeyCode::Down | KeyCode::Char('j') => {
                app.prompt_select_next();
            }
            KeyCode::Char(' ') => {
                // Space toggles selection in multi-select mode
                app.prompt_toggle();
            }
            KeyCode::Enter => {
                // Send the selected value back to the training process
                if let Some((id, value)) = app.prompt_confirm() {
                    send_command(stdin, ControlCommand::PromptResponse { id, value }).await?;
                }
            }
            _ => {}
        }
        return Ok(false);
    }

    // Handle quit confirmation popup (second priority)
    if app.show_quit_confirm {
        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                // Confirm quit
                let _ = send_command(stdin, ControlCommand::Stop).await;
                app.should_quit = true;
                return Ok(true);
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                // Cancel quit
                app.show_quit_confirm = false;
            }
            _ => {}
        }
        return Ok(false);
    }

    // Handle help overlay separately (second priority)
    if app.show_help {
        match key.code {
            KeyCode::Char('q') => {
                // Close help and show quit confirmation
                app.show_help = false;
                if !app.child_exited {
                    app.show_quit_confirm = true;
                } else {
                    app.should_quit = true;
                    return Ok(true);
                }
            }
            KeyCode::Esc | KeyCode::Char('?') => {
                // Just close help
                app.show_help = false;
            }
            _ => {}
        }
        return Ok(false);
    }

    // Handle sample detail popup (third priority)
    if app.selected_sample.is_some() {
        match key.code {
            KeyCode::Esc => {
                app.selected_sample = None;
                app.sample_detail_scroll = 0;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                app.sample_detail_scroll = app.sample_detail_scroll.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                app.sample_detail_scroll = app.sample_detail_scroll.saturating_add(1);
            }
            KeyCode::PageUp => {
                app.sample_detail_scroll = app.sample_detail_scroll.saturating_sub(10);
            }
            KeyCode::PageDown => {
                app.sample_detail_scroll = app.sample_detail_scroll.saturating_add(10);
            }
            _ => {}
        }
        return Ok(false);
    }

    // Handle settings popup (fourth priority)
    if app.show_settings {
        match key.code {
            KeyCode::Esc | KeyCode::Char('o') => {
                app.show_settings = false;
            }
            // Quick log level selection
            KeyCode::Char('d') => {
                app.log_level_filter = messages::LogLevel::Debug;
            }
            KeyCode::Char('i') => {
                app.log_level_filter = messages::LogLevel::Info;
            }
            KeyCode::Char('w') => {
                app.log_level_filter = messages::LogLevel::Warn;
            }
            KeyCode::Char('e') => {
                app.log_level_filter = messages::LogLevel::Error;
            }
            KeyCode::Char('l') => {
                app.log_level_filter = app.log_level_filter.next_filter();
            }
            _ => {}
        }
        return Ok(false);
    }

    match (key.code, key.modifiers) {
        // Quit - show confirmation if training is still running
        (KeyCode::Char('q'), _) => {
            if app.child_exited {
                // Training already finished, quit immediately
                app.should_quit = true;
                return Ok(true);
            } else {
                // Show confirmation popup
                app.show_quit_confirm = true;
            }
        }
        (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
            if app.child_exited {
                app.should_quit = true;
                return Ok(true);
            } else {
                app.show_quit_confirm = true;
            }
        }

        // Pause/Resume
        (KeyCode::Char('p'), _) if app.state == app::TrainingState::Running => {
            send_command(stdin, ControlCommand::Pause).await?;
        }
        (KeyCode::Char('r'), _) if app.state == app::TrainingState::Paused => {
            send_command(stdin, ControlCommand::Resume).await?;
        }

        // Save checkpoint
        (KeyCode::Char('s'), _) => {
            send_command(stdin, ControlCommand::SaveCheckpoint).await?;
            app.handle_message(TrainingMessage::Log {
                level: messages::LogLevel::Info,
                message: "Checkpoint save requested...".to_string(),
            });
        }

        // Tab navigation
        (KeyCode::Tab, _) => {
            app.next_tab();
        }
        (KeyCode::BackTab, _) => {
            app.prev_tab();
        }
        // Quick tab switch with 1/2/3/4
        (KeyCode::Char('1'), _) => {
            app.switch_to_tab(app::ActiveTab::Logs);
        }
        (KeyCode::Char('2'), _) => {
            app.switch_to_tab(app::ActiveTab::Samples);
        }
        (KeyCode::Char('3'), _) => {
            app.switch_to_tab(app::ActiveTab::Config);
        }

        // Scrolling
        (KeyCode::Up, _) | (KeyCode::Char('k'), _) => {
            app.scroll_up();
        }
        (KeyCode::Down, _) | (KeyCode::Char('j'), _) => {
            app.scroll_down();
        }
        (KeyCode::PageUp, _) => {
            app.page_up();
        }
        (KeyCode::PageDown, _) => {
            app.page_down();
        }
        // Go to top/bottom (vim-style)
        (KeyCode::Char('g'), _) => {
            app.scroll_to_top();
        }
        (KeyCode::Char('G'), _) => {
            app.scroll_to_bottom();
        }

        // Sample display mode
        (KeyCode::Char('m'), _) => {
            app.cycle_sample_mode();
            send_command(
                stdin,
                ControlCommand::SetSampleDisplay(app.sample_display_mode),
            )
            .await?;
        }

        // Cycle log level filter
        (KeyCode::Char('l'), _) => {
            app.log_level_filter = app.log_level_filter.next_filter();
        }

        // Open settings popup
        (KeyCode::Char('o'), _) => {
            app.show_settings = true;
        }

        // Help
        (KeyCode::Char('?'), _) => {
            app.toggle_help();
        }

        // Enter to open sample detail popup
        (KeyCode::Enter, _)
            if app.active_tab == app::ActiveTab::Samples && !app.samples.is_empty() =>
        {
            let sample_idx = app.sample_scroll as usize;
            if sample_idx < app.samples.len() {
                app.selected_sample = Some(sample_idx);
                app.sample_detail_scroll = 0;
            }
        }

        _ => {}
    }

    Ok(false)
}

/// Handle mouse input for tab selection and scrolling
fn handle_mouse(mouse: event::MouseEvent, app: &mut App) {
    // If quit confirmation is showing, ignore mouse events
    if app.show_quit_confirm {
        return;
    }

    // If sample detail popup is open, handle scrolling for it
    if app.selected_sample.is_some() {
        match mouse.kind {
            MouseEventKind::ScrollUp => {
                app.sample_detail_scroll = app.sample_detail_scroll.saturating_sub(1);
            }
            MouseEventKind::ScrollDown => {
                app.sample_detail_scroll = app.sample_detail_scroll.saturating_add(1);
            }
            // Click outside popup area could close it, but for simplicity just ignore clicks
            _ => {}
        }
        return;
    }

    // If help is showing, ignore mouse events
    if app.show_help {
        return;
    }

    // If settings popup is open, ignore mouse events
    if app.show_settings {
        return;
    }

    // If prompt is active, ignore mouse events (user must respond via keyboard)
    if app.has_active_prompt() {
        return;
    }

    match mouse.kind {
        // Left click to select tab or open sample detail
        MouseEventKind::Down(MouseButton::Left) => {
            // First check if clicking on tab header
            if let Some(tab) = get_tab_at_position(mouse.column, mouse.row, app) {
                app.active_tab = tab;
                return;
            }

            // If on samples tab and clicking in the content area, open sample detail
            if app.active_tab == app::ActiveTab::Samples
                && !app.samples.is_empty()
                && let Some((tabs_x, tabs_y, tabs_width, tabs_height)) = app.tabs_area
            {
                // Check if click is within the tabs content area (below header)
                let content_y = tabs_y + 2; // Tab header is 2 rows
                if mouse.column >= tabs_x
                    && mouse.column < tabs_x + tabs_width
                    && mouse.row >= content_y
                    && mouse.row < tabs_y + tabs_height
                {
                    // Calculate which sample was clicked
                    // Layout: 3 header lines (mode, stats, blank) + 5 lines per sample
                    let header_lines = 3u16;
                    let lines_per_sample = 5u16;
                    let click_row = mouse.row.saturating_sub(content_y);

                    if click_row >= header_lines {
                        // Calculate sample index from click position
                        let sample_offset = (click_row - header_lines) / lines_per_sample;
                        let sample_idx = app.sample_scroll as usize + sample_offset as usize;

                        if sample_idx < app.samples.len() {
                            app.selected_sample = Some(sample_idx);
                            app.sample_detail_scroll = 0;
                        }
                    }
                }
            }
        }
        // Scroll wheel
        MouseEventKind::ScrollUp => {
            app.scroll_up();
        }
        MouseEventKind::ScrollDown => {
            app.scroll_down();
        }
        _ => {}
    }
}

/// Determine which tab (if any) was clicked based on position
fn get_tab_at_position(col: u16, row: u16, app: &App) -> Option<app::ActiveTab> {
    // Use the stored tabs_area from the last render
    let (tabs_x, tabs_y, _tabs_width, _tabs_height) = app.tabs_area?;

    // Tab header is in the first 2 rows of the tabs area (rows tabs_y and tabs_y+1)
    if row < tabs_y || row >= tabs_y + 2 {
        return None;
    }

    // Check if click is within the tabs panel horizontally
    if col < tabs_x {
        return None;
    }

    // Get available tabs based on training type (SFT only shows Logs and Config)
    let available_tabs = app.get_available_tabs();

    // Calculate clickable regions dynamically based on available tabs
    // Each tab has: title + 3 chars for separator/padding
    let relative_col = col - tabs_x;
    let mut offset: u16 = 0;

    for tab in &available_tabs {
        let tab_width: u16 = match tab {
            app::ActiveTab::Logs => 4 + 3,    // "Logs" + separator
            app::ActiveTab::Samples => 7 + 3, // "Samples" + separator
            app::ActiveTab::Config => 6 + 3,  // "Config" + separator
        };

        if relative_col < offset + tab_width {
            return Some(*tab);
        }
        offset += tab_width;
    }

    None
}
