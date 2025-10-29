//! Quit confirmation popup component
//!
//! Shows a confirmation dialog when the user tries to quit during training.

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Flex, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
};

/// Draw the quit confirmation popup
pub fn draw(f: &mut Frame) {
    let area = f.area();

    // Create centered popup area (width, height in characters)
    let popup_area = centered_rect(50, 9, area);

    // Clear background
    f.render_widget(Clear, popup_area);

    // Build content
    let lines = vec![
        Line::from(""),
        Line::from(vec![Span::styled(
            "⚠ Quit training?",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "This will stop the training process and ",
            Style::default().fg(Color::White),
        )]),
        Line::from(vec![Span::styled(
            "lose any unsaved progress.",
            Style::default().fg(Color::White),
        )]),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "[y]",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" Quit  ", Style::default().fg(Color::Gray)),
            Span::styled(
                "[n/Esc]",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" Cancel", Style::default().fg(Color::Gray)),
        ]),
    ];

    let content = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow))
                .title(" Confirm Quit ")
                .title_alignment(Alignment::Center),
        )
        .alignment(Alignment::Center);

    f.render_widget(content, popup_area);
}

/// Create a centered rect with fixed dimensions
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
