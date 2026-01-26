//! UI rendering module
//!
//! Contains all the UI components and the main draw function.

mod config;
mod confirm_quit;
mod header;
mod help;
mod logs;
mod metrics;
mod progress;
mod prompt;
mod sample_detail;
mod samples;
mod settings;

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
};

use crate::app::{ActiveTab, App};

/// Draw the entire UI
pub fn draw(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Header
            Constraint::Min(10),   // Main content
            Constraint::Length(1), // Footer
        ])
        .split(f.area());

    // Header bar
    header::draw(f, app, chunks[0]);

    // Main content - split horizontally
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(55), // Left: Metrics panel
            Constraint::Percentage(45), // Right: Tabs panel
        ])
        .split(chunks[1]);

    // Left side: Metrics + Progress + Stats
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(15), // Metrics with sparklines (4 lines per metric)
            Constraint::Length(5),  // Progress bars
            Constraint::Min(5),     // Stats (3 lines + borders)
        ])
        .split(main_chunks[0]);

    metrics::draw(f, app, left_chunks[0]);
    progress::draw(f, app, left_chunks[1]);
    draw_stats(f, app, left_chunks[2]);

    // Right side: Tabbed content
    // Store the tabs area for mouse click detection
    let tabs_area = main_chunks[1];
    app.tabs_area = Some((tabs_area.x, tabs_area.y, tabs_area.width, tabs_area.height));
    draw_tabs(f, app, tabs_area);

    // Footer with keybindings
    draw_footer(f, app, chunks[2]);

    // Sample detail popup if visible (before help so help can appear on top)
    if let Some(sample_idx) = app.selected_sample {
        sample_detail::draw(f, app, sample_idx);
    }

    // Settings popup if visible
    if app.show_settings {
        settings::draw(f, app);
    }

    // Help overlay if visible
    if app.show_help {
        help::draw(f);
    }

    // Quit confirmation popup
    if app.show_quit_confirm {
        confirm_quit::draw(f);
    }

    // Interactive prompt popup (highest priority - blocks everything)
    if let Some(ref active_prompt) = app.active_prompt {
        prompt::draw(f, active_prompt);
    }
}

/// Draw the tabbed content panel
fn draw_tabs(f: &mut Frame, app: &App, area: Rect) {
    use ratatui::{
        style::{Color, Modifier, Style},
        text::{Line, Span},
        widgets::{Block, Borders, Tabs},
    };

    // Get available tabs based on training type (SFT hides Samples and DB)
    let available_tabs = app.get_available_tabs();

    // Create tab titles using ActiveTab::title()
    let titles: Vec<Line> = available_tabs
        .iter()
        .map(|tab| {
            let style = if *tab == app.active_tab {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            Line::from(Span::styled(tab.title(), style))
        })
        .collect();

    // Find the index of the active tab in the available tabs
    let selected_index = available_tabs
        .iter()
        .position(|t| *t == app.active_tab)
        .unwrap_or(0);

    let tabs = Tabs::new(titles)
        .select(selected_index)
        .style(Style::default().fg(Color::White))
        .highlight_style(Style::default().fg(Color::Cyan));

    // Split for tabs header and content
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(1)])
        .split(area);

    let tabs_block = Block::default()
        .borders(Borders::TOP | Borders::LEFT | Borders::RIGHT)
        .border_style(Style::default().fg(Color::DarkGray));

    f.render_widget(tabs.block(tabs_block), chunks[0]);

    // Render active tab content
    let content_block = Block::default()
        .borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = content_block.inner(chunks[1]);
    f.render_widget(content_block, chunks[1]);

    match app.active_tab {
        ActiveTab::Logs => logs::draw(f, app, inner),
        ActiveTab::Samples => samples::draw(f, app, inner),
        ActiveTab::Config => config::draw(f, app, inner),
    }
}

/// Draw stats panel
fn draw_stats(f: &mut Frame, app: &App, area: Rect) {
    use ratatui::{
        style::{Color, Style},
        text::{Line, Span},
        widgets::{Block, Borders, Paragraph},
    };

    let tokens_str = format_tokens(app.total_tokens);
    let tokens_per_sec = format!("{:.0}/s", app.tokens_per_sec());
    let speed_str = format!("{:.0}ms/step", app.ms_per_step());
    let gen_train = format!(
        "(gen:{:.0}ms train:{:.0}ms)",
        app.generation_time_ms, app.training_time_ms
    );

    // First two lines are common to both SFT and GRPO
    let line1 = Line::from(vec![
        Span::raw("Tokens: "),
        Span::styled(tokens_str, Style::default().fg(Color::Cyan)),
        Span::styled(
            format!(" ({tokens_per_sec})"),
            Style::default().fg(Color::DarkGray),
        ),
        Span::raw(" | Elapsed: "),
        Span::styled(app.elapsed_str(), Style::default().fg(Color::Cyan)),
        Span::raw(" | ETA: "),
        Span::styled(app.eta_str(), Style::default().fg(Color::Yellow)),
    ]);

    // Memory display (convert MB to GB)
    let memory_str = if app.peak_memory_mb > 0.0 {
        format!(
            "{:.1}GB peak, {:.1}GB active",
            app.peak_memory_mb / 1024.0,
            app.active_memory_mb / 1024.0
        )
    } else {
        "-".to_string()
    };

    let line2 = Line::from(vec![
        Span::raw("Speed: "),
        Span::styled(speed_str, Style::default().fg(Color::Yellow)),
        Span::raw(" "),
        Span::styled(gen_train, Style::default().fg(Color::DarkGray)),
        Span::raw(" | Memory: "),
        Span::styled(memory_str, Style::default().fg(Color::Magenta)),
    ]);

    // Third line differs based on training type
    let line3 = if app.training_type == "sft" {
        // SFT: Show best and average loss
        let best_loss_str = if app.best_loss < f64::INFINITY {
            format!("{:.4}", app.best_loss)
        } else {
            "-".to_string()
        };

        let avg_loss_str = if app.loss_count > 0 {
            format!("{:.4}", app.avg_loss())
        } else {
            "-".to_string()
        };

        Line::from(vec![
            Span::raw("Loss: "),
            Span::styled("Best ", Style::default().fg(Color::DarkGray)),
            Span::styled(best_loss_str, Style::default().fg(Color::Green)),
            Span::styled(" | Avg ", Style::default().fg(Color::DarkGray)),
            Span::styled(avg_loss_str, Style::default().fg(Color::Cyan)),
        ])
    } else {
        // GRPO: Show best and average reward
        let best_reward_str = if app.best_reward > f64::NEG_INFINITY {
            format!("{:.2}", app.best_reward)
        } else {
            "-".to_string()
        };

        let avg_reward_str = if app.reward_count > 0 {
            format!("{:.2}", app.avg_reward())
        } else {
            "-".to_string()
        };

        let std_reward_str = if app.reward_count > 0 && app.current_std_reward > 0.0 {
            format!("±{:.2}", app.current_std_reward)
        } else {
            String::new()
        };

        Line::from(vec![
            Span::raw("Reward: "),
            Span::styled("Best ", Style::default().fg(Color::DarkGray)),
            Span::styled(best_reward_str, Style::default().fg(Color::Green)),
            Span::styled(" | Avg ", Style::default().fg(Color::DarkGray)),
            Span::styled(avg_reward_str, Style::default().fg(Color::Cyan)),
            Span::styled(
                format!(" {}", std_reward_str),
                Style::default().fg(Color::DarkGray),
            ),
        ])
    };

    let lines = vec![line1, line2, line3];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title("Stats");
    let paragraph = Paragraph::new(lines).block(block);
    f.render_widget(paragraph, area);
}

/// Draw footer with keybindings
fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    use ratatui::{
        style::{Color, Modifier, Style},
        text::{Line, Span},
        widgets::Paragraph,
    };

    let key_style = Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::BOLD);
    let desc_style = Style::default().fg(Color::Gray);
    let sep_style = Style::default().fg(Color::DarkGray);

    let mut spans = vec![];

    // Pause/Resume based on state
    if app.state == crate::app::TrainingState::Paused {
        spans.extend(vec![
            Span::styled("[r]", key_style),
            Span::styled(" Resume", desc_style),
        ]);
    } else if app.state == crate::app::TrainingState::Running {
        spans.extend(vec![
            Span::styled("[p]", key_style),
            Span::styled(" Pause", desc_style),
        ]);
    }

    spans.extend(vec![
        Span::styled(" │ ", sep_style),
        Span::styled("[s]", key_style),
        Span::styled(" Save", desc_style),
        Span::styled(" │ ", sep_style),
        Span::styled("[Tab]", key_style),
        Span::styled(" Tabs", desc_style),
        Span::styled(" │ ", sep_style),
        Span::styled("[↑↓]", key_style),
        Span::styled(" Scroll", desc_style),
        Span::styled(" │ ", sep_style),
        Span::styled("[m]", key_style),
        Span::styled(" Mode", desc_style),
        Span::styled(" │ ", sep_style),
        Span::styled("[l]", key_style),
        Span::styled(" Log", desc_style),
        Span::styled(" │ ", sep_style),
        Span::styled("[?]", key_style),
        Span::styled(" Help", desc_style),
        Span::styled(" │ ", sep_style),
        Span::styled("[q]", key_style),
        Span::styled(" Quit", desc_style),
    ]);

    let paragraph = Paragraph::new(Line::from(spans));
    f.render_widget(paragraph, area);
}

/// Format token count with K/M suffix
fn format_tokens(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        format!("{:.1}M", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1_000 {
        format!("{:.1}K", tokens as f64 / 1_000.0)
    } else {
        tokens.to_string()
    }
}
