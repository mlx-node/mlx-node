//! Help overlay component

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Flex, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
};

/// Draw the help overlay
pub fn draw(f: &mut Frame) {
    let area = f.area();

    // Center the help popup
    let popup_width = 50.min(area.width.saturating_sub(4));
    let popup_height = 28.min(area.height.saturating_sub(4));

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

    let lines = vec![
        Line::from(Span::styled("─── Controls ───", section_style)),
        Line::from(""),
        Line::from(vec![
            Span::styled("  p         ", key_style),
            Span::styled("Pause training", desc_style),
        ]),
        Line::from(vec![
            Span::styled("  r         ", key_style),
            Span::styled("Resume training", desc_style),
        ]),
        Line::from(vec![
            Span::styled("  s         ", key_style),
            Span::styled("Save checkpoint now", desc_style),
        ]),
        Line::from(vec![
            Span::styled("  q / Esc   ", key_style),
            Span::styled("Quit (stops training)", desc_style),
        ]),
        Line::from(""),
        Line::from(Span::styled("─── Navigation ───", section_style)),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Tab/1/2/3 ", key_style),
            Span::styled("Switch tabs", desc_style),
        ]),
        Line::from(vec![
            Span::styled("  ↑/↓ j/k   ", key_style),
            Span::styled("Scroll content", desc_style),
        ]),
        Line::from(vec![
            Span::styled("  g / G     ", key_style),
            Span::styled("Go to top/bottom", desc_style),
        ]),
        Line::from(vec![
            Span::styled("  PgUp/PgDn ", key_style),
            Span::styled("Page up/down", desc_style),
        ]),
        Line::from(""),
        Line::from(Span::styled("─── Display ───", section_style)),
        Line::from(""),
        Line::from(vec![
            Span::styled("  m         ", key_style),
            Span::styled("Cycle sample display mode", desc_style),
        ]),
        Line::from(vec![
            Span::styled("  l         ", key_style),
            Span::styled("Cycle log level filter", desc_style),
        ]),
        Line::from(vec![
            Span::styled("  o         ", key_style),
            Span::styled("Open settings", desc_style),
        ]),
        Line::from(vec![
            Span::styled("  ?         ", key_style),
            Span::styled("Toggle this help", desc_style),
        ]),
        Line::from(""),
        Line::from(Span::styled("─── Auto-Restart ───", section_style)),
        Line::from(""),
        Line::from(vec![
            Span::styled("  c         ", key_style),
            Span::styled("Cancel restart countdown", desc_style),
        ]),
        Line::from(vec![
            Span::styled("  Enter     ", key_style),
            Span::styled("Skip countdown, restart now", desc_style),
        ]),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Help ")
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
