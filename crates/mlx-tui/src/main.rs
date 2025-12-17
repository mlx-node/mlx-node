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
mod messages;
mod ui;

use std::io;
use std::process::{ExitStatus, Stdio};
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

use app::App;
use commands::{ControlCommand, send_command};
use messages::TrainingMessage;

/// Restart countdown duration in seconds
const RESTART_COUNTDOWN_SECS: u8 = 5;

/// MLX Training TUI - Monitor and control GRPO training
#[derive(Parser, Clone)]
#[command(name = "mlx-train")]
#[command(about = "TUI for monitoring and controlling MLX-Node GRPO training")]
#[command(version)]
struct Cli {
    /// Path to the training script
    #[arg(short, long)]
    script: String,

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
    let mut cmd = Command::new("node");

    // Add --import flags to node (e.g., --import tsx for TypeScript)
    for import in &cli.import {
        cmd.arg("--import").arg(import);
    }

    cmd.arg(&cli.script);

    // Add original args
    for arg in &cli.args {
        cmd.arg(arg);
    }

    // Add --resume flag on restart if not already present
    if is_restart && !cli.args.iter().any(|a| a == "--resume" || a == "-r") {
        cmd.arg("--resume");
    }

    cmd.env("MLX_TUI_MODE", "1") // Signal to use TUI-compatible output
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(ref workdir) = cli.workdir {
        cmd.current_dir(workdir);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| eyre!("Failed to spawn training script: {e}"))?;

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
    color_eyre::install()?;
    let cli = Cli::parse();

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    app.auto_restart_enabled = !cli.no_auto_restart;

    // Add initial log
    app.handle_message(TrainingMessage::Log {
        level: messages::LogLevel::Info,
        message: format!("Starting: {} {}", cli.script, cli.args.join(" ")),
    });

    // Outer loop for restart handling
    let result = run_with_restart(&mut terminal, &mut app, &cli).await;

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
    let mut is_restart = false;

    loop {
        // Spawn the training process
        let process = match spawn_training_process(cli, is_restart) {
            Ok(p) => p,
            Err(e) => {
                app.handle_message(TrainingMessage::Log {
                    level: messages::LogLevel::Error,
                    message: format!("Failed to spawn process: {e}"),
                });
                app.state = app::TrainingState::Error;
                app.child_exited = true;
                // Wait for user to quit
                wait_for_quit(terminal, app).await?;
                return Ok(());
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

        // Kill child if still running
        let _ = child.kill().await;

        // Check if user requested quit
        if app.should_quit {
            return Ok(());
        }

        // Handle exit status
        match exit_status {
            Ok(Some(status)) => {
                let code = status.code();
                app.last_exit_code = code;

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
                    app.handle_message(TrainingMessage::Log {
                        level: messages::LogLevel::Error,
                        message: format!(
                            "Training process crashed (exit code: {})",
                            code.map_or("unknown".to_string(), |c| c.to_string())
                        ),
                    });

                    if app.auto_restart_enabled {
                        // Start countdown and wait
                        if !run_restart_countdown(terminal, app).await? {
                            // User cancelled restart
                            wait_for_quit(terminal, app).await?;
                            return Ok(());
                        }
                        // Prepare for restart
                        app.prepare_for_restart();
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
                // Process still running but we exited loop (shouldn't happen)
                app.handle_message(TrainingMessage::Log {
                    level: messages::LogLevel::Warn,
                    message: "Event loop exited unexpectedly".to_string(),
                });
                wait_for_quit(terminal, app).await?;
                return Ok(());
            }
            Err(e) => {
                app.handle_message(TrainingMessage::Log {
                    level: messages::LogLevel::Error,
                    message: format!("Error: {e}"),
                });
                app.state = app::TrainingState::Error;
                wait_for_quit(terminal, app).await?;
                return Ok(());
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

async fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    child: &mut Child,
    stdin: &mut tokio::process::ChildStdin,
    stdout_reader: &mut tokio::io::Lines<BufReader<tokio::process::ChildStdout>>,
    stderr_reader: &mut tokio::io::Lines<BufReader<tokio::process::ChildStderr>>,
) -> Result<Option<ExitStatus>> {
    let mut event_stream = crossterm::event::EventStream::new();

    loop {
        // Render
        terminal.draw(|f| ui::draw(f, app))?;

        if app.should_quit {
            // Try to get exit status if available
            let status = child.try_wait().ok().flatten();
            return Ok(status);
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
                        // Try to parse as JSONL message
                        match serde_json::from_str::<TrainingMessage>(&line) {
                            Ok(msg) => app.handle_message(msg),
                            Err(_) => {
                                // Non-JSON output, treat as log
                                app.handle_message(TrainingMessage::Log {
                                    level: messages::LogLevel::Info,
                                    message: line.to_string(),
                                });
                            }
                        }
                    }
                    Ok(None) => {
                        // Child process ended - wait for actual exit status
                        app.child_exited = true;
                        // Wait for process to fully exit
                        let status = child.wait().await.ok();
                        return Ok(status);
                    }
                    Err(e) => {
                        app.handle_message(TrainingMessage::Log {
                            level: messages::LogLevel::Error,
                            message: format!("Read error: {e}"),
                        });
                        app.state = app::TrainingState::Error;
                        app.child_exited = true;
                        let status = child.wait().await.ok();
                        return Ok(status);
                    }
                }
            }

            // Child stderr (log as warnings/errors)
            maybe_line = stderr_reader.next_line() => {
                match maybe_line {
                    Ok(Some(line)) => {
                        // Skip empty lines
                        if !line.trim().is_empty() {
                            app.handle_message(TrainingMessage::Log {
                                level: messages::LogLevel::Warn,
                                message: line.to_string(),
                            });
                        }
                    }
                    Ok(None) => {
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
    // Handle quit confirmation popup (highest priority)
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
        (KeyCode::Char('p'), _) => {
            if app.state == app::TrainingState::Running {
                send_command(stdin, ControlCommand::Pause).await?;
            }
        }
        (KeyCode::Char('r'), _) => {
            if app.state == app::TrainingState::Paused {
                send_command(stdin, ControlCommand::Resume).await?;
            }
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
        // Quick tab switch with 1/2/3
        (KeyCode::Char('1'), _) => {
            app.active_tab = app::ActiveTab::Logs;
        }
        (KeyCode::Char('2'), _) => {
            app.active_tab = app::ActiveTab::Samples;
        }
        (KeyCode::Char('3'), _) => {
            app.active_tab = app::ActiveTab::Config;
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
        (KeyCode::Enter, _) => {
            if app.active_tab == app::ActiveTab::Samples && !app.samples.is_empty() {
                let sample_idx = app.sample_scroll as usize;
                if sample_idx < app.samples.len() {
                    app.selected_sample = Some(sample_idx);
                    app.sample_detail_scroll = 0;
                }
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
                    let sample_idx = app.sample_scroll as usize;
                    if sample_idx < app.samples.len() {
                        app.selected_sample = Some(sample_idx);
                        app.sample_detail_scroll = 0;
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

    // Tab header is in the first 2 rows of the tabs area
    if row < tabs_y || row > tabs_y + 2 {
        return None;
    }

    // Check if click is within the tabs panel horizontally
    if col < tabs_x {
        return None;
    }

    // Tab labels are rendered with fixed widths at the start of the panel:
    // "Logs" (4 chars) + " | " (3 chars) + "Samples" (7 chars) + " | " (3 chars) + "Config" (6 chars)
    // Approximate clickable regions:
    // Logs:    tabs_x to tabs_x + 6
    // Samples: tabs_x + 6 to tabs_x + 18
    // Config:  tabs_x + 18 to tabs_x + 28
    let relative_col = col - tabs_x;

    if relative_col < 8 {
        Some(app::ActiveTab::Logs)
    } else if relative_col < 18 {
        Some(app::ActiveTab::Samples)
    } else if relative_col < 28 {
        Some(app::ActiveTab::Config)
    } else {
        None
    }
}
