//! Sample detail popup component
//!
//! Shows full content of a sample (prompt + completion) in a centered popup
//! with scrolling support.

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Flex, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};

use crate::app::App;

/// Draw the sample detail popup
pub fn draw(f: &mut Frame, app: &App, sample_idx: usize) {
    let area = f.area();

    // Get the sample
    let Some(sample) = app.samples.get(sample_idx) else {
        return;
    };

    // Create centered popup area (~80% of screen)
    let popup_area = centered_rect(80, 80, area);

    // Clear background
    f.render_widget(Clear, popup_area);

    // Build content
    let mut lines: Vec<Line> = vec![];

    // Header
    let reward_color = if sample.reward > 0.5 {
        Color::Green
    } else if sample.reward > 0.0 {
        Color::Yellow
    } else {
        Color::Red
    };

    lines.push(Line::from(vec![
        Span::styled("Sample #", Style::default().fg(Color::DarkGray)),
        Span::styled(
            sample.index.to_string(),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" | Reward: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{:.4}", sample.reward),
            Style::default().fg(reward_color),
        ),
        Span::styled(" | Tokens: ", Style::default().fg(Color::DarkGray)),
        Span::styled(sample.tokens.to_string(), Style::default().fg(Color::Cyan)),
    ]));
    lines.push(Line::from(""));

    // Prompt section
    lines.push(Line::from(vec![Span::styled(
        "━━━ Prompt ━━━",
        Style::default()
            .fg(Color::Blue)
            .add_modifier(Modifier::BOLD),
    )]));
    for line in sample.prompt.lines() {
        lines.push(Line::from(Span::styled(
            line.to_string(),
            Style::default().fg(Color::Blue),
        )));
    }
    if sample.prompt.is_empty() {
        lines.push(Line::from(Span::styled(
            "(empty)",
            Style::default().fg(Color::DarkGray),
        )));
    }
    lines.push(Line::from(""));

    // Completion section
    lines.push(Line::from(vec![Span::styled(
        "━━━ Completion ━━━",
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
    )]));
    for line in sample.completion.lines() {
        lines.push(Line::from(Span::styled(
            line.to_string(),
            Style::default().fg(Color::White),
        )));
    }
    if sample.completion.is_empty() {
        lines.push(Line::from(Span::styled(
            "(empty)",
            Style::default().fg(Color::DarkGray),
        )));
    }

    // Create paragraph with scroll
    let content = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(" Sample Detail (ESC to close, ↑↓ to scroll) ")
                .title_alignment(Alignment::Center),
        )
        .wrap(Wrap { trim: false })
        .scroll((app.sample_detail_scroll, 0));

    f.render_widget(content, popup_area);
}

/// Create a centered rect using percentage of the area
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_height = area.height * percent_y / 100;
    let popup_width = area.width * percent_x / 100;

    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(popup_height),
            Constraint::Min(0),
        ])
        .flex(Flex::Center)
        .split(area);

    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(popup_width),
            Constraint::Min(0),
        ])
        .flex(Flex::Center)
        .split(vertical[1]);

    horizontal[1]
}
