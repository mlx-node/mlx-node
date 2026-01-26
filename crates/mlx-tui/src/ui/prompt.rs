//! Interactive prompt overlay component
//!
//! Displays a selection prompt from the training process and
//! allows the user to navigate and select an option.
//! Supports both single-select (radio) and multi-select (checkbox) modes.

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Flex, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
};

use crate::app::ActivePrompt;

/// Draw the interactive prompt overlay
pub fn draw(f: &mut Frame, prompt: &ActivePrompt) {
    let area = f.area();

    // Calculate popup size based on content
    let max_choice_len = prompt
        .choices
        .iter()
        .map(|c| c.label.len() + c.description.as_ref().map_or(0, |d| d.len() + 3))
        .max()
        .unwrap_or(20);

    let popup_width = (max_choice_len + 12).max(prompt.message.len() + 6).min(70) as u16;
    let popup_height = (prompt.choices.len() + 6).min(20) as u16;

    let popup_area = centered_rect(popup_width, popup_height, area);

    // Clear the background
    f.render_widget(Clear, popup_area);

    let cursor_style = Style::default()
        .fg(Color::Black)
        .bg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let normal_style = Style::default().fg(Color::White);
    let checked_style = Style::default().fg(Color::Green);
    let hint_style = Style::default().fg(Color::DarkGray);
    let desc_style = Style::default().fg(Color::Yellow);

    let mut lines = vec![
        Line::from(Span::styled(
            &prompt.message,
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
    ];

    // Render each choice
    for (i, choice) in prompt.choices.iter().enumerate() {
        let is_cursor = i == prompt.cursor;
        let is_checked = prompt.selected.get(i).copied().unwrap_or(false);

        let prefix = if prompt.multi_select {
            // Checkbox style for multi-select
            if is_checked { "[x] " } else { "[ ] " }
        } else {
            // Arrow style for single-select
            if is_cursor { "▶ " } else { "  " }
        };

        // Determine style based on cursor and selection state
        let style = if is_cursor {
            cursor_style
        } else if is_checked && prompt.multi_select {
            checked_style
        } else {
            normal_style
        };

        let mut spans = vec![Span::styled(format!("{}{}", prefix, choice.label), style)];

        // Add description if present
        if let Some(ref desc) = choice.description {
            spans.push(Span::styled(format!(" - {}", desc), desc_style));
        }

        lines.push(Line::from(spans));
    }

    // Add navigation hint
    lines.push(Line::from(""));
    let hint = if prompt.multi_select {
        "↑↓ navigate  Space toggle  ⏎ confirm"
    } else {
        "↑↓ navigate  ⏎ select"
    };
    lines.push(Line::from(Span::styled(hint, hint_style)));

    let title = if prompt.multi_select {
        " Select Options (multi) "
    } else {
        " Select Option "
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(title)
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
