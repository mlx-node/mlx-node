//! Settings popup component

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Flex, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
};

use crate::app::App;
use crate::messages::LogLevel;

/// Draw the settings popup
pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();

    // Center the popup
    let popup_width = 45.min(area.width.saturating_sub(4));
    let popup_height = 14.min(area.height.saturating_sub(4));

    let popup_area = centered_rect(popup_width, popup_height, area);

    // Clear the background
    f.render_widget(Clear, popup_area);

    let key_style = Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::BOLD);
    let desc_style = Style::default().fg(Color::White);
    let section_style = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let selected_style = Style::default()
        .fg(Color::Green)
        .add_modifier(Modifier::BOLD);
    let unselected_style = Style::default().fg(Color::DarkGray);

    // Build log level options
    let levels = [
        (LogLevel::Debug, "d", "Debug+ (all logs)"),
        (LogLevel::Info, "i", "Info+ (info and above)"),
        (LogLevel::Warn, "w", "Warn+ (warnings and errors)"),
        (LogLevel::Error, "e", "Error (errors only)"),
    ];

    let mut lines = vec![
        Line::from(Span::styled("─── Log Level Filter ───", section_style)),
        Line::from(""),
    ];

    for (level, key, desc) in levels {
        let is_selected = app.log_level_filter == level;
        let indicator = if is_selected { "●" } else { "○" };
        let style = if is_selected {
            selected_style
        } else {
            unselected_style
        };

        lines.push(Line::from(vec![
            Span::styled(format!("  [{key}] "), key_style),
            Span::styled(indicator, style),
            Span::styled(" ", Style::default()),
            Span::styled(desc, if is_selected { desc_style } else { style }),
        ]));
    }

    lines.extend(vec![
        Line::from(""),
        Line::from(Span::styled("─── Hints ───", section_style)),
        Line::from(""),
        Line::from(vec![
            Span::styled("  [l]       ", key_style),
            Span::styled("Cycle log filter (anywhere)", desc_style),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  [Esc]     ", key_style),
            Span::styled("Close settings", desc_style),
        ]),
    ]);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Settings ")
        .title_alignment(Alignment::Center);

    let paragraph = Paragraph::new(lines).block(block);

    f.render_widget(paragraph, popup_area);
}

/// Create a centered rect
fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(height),
            Constraint::Min(0),
        ])
        .flex(Flex::Center)
        .split(area);

    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(width),
            Constraint::Min(0),
        ])
        .flex(Flex::Center)
        .split(vertical[1]);

    horizontal[1]
}
