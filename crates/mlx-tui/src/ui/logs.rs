//! Log viewer component

use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{List, ListItem},
};

use crate::app::App;

/// Draw the log viewer
pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let available_width = area.width.saturating_sub(2) as usize; // Account for borders
    let visible_height = area.height.saturating_sub(1) as usize; // Reserve 1 line for header

    // Filter logs by level
    let filtered_logs: Vec<_> = app
        .logs
        .iter()
        .filter(|entry| entry.level >= app.log_level_filter)
        .collect();

    // Header line showing filter level
    let filter_line = Line::from(vec![
        Span::styled("Filter: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            app.log_level_filter.filter_name(),
            Style::default().fg(Color::Cyan),
        ),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "[l]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" cycle", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!(" │ {} entries", filtered_logs.len()),
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    let mut items: Vec<ListItem> = vec![ListItem::new(filter_line)];

    if filtered_logs.is_empty() {
        items.push(ListItem::new(Line::from(Span::styled(
            "No logs matching filter...",
            Style::default().fg(Color::DarkGray),
        ))));
        let list = List::new(items);
        f.render_widget(list, area);
        return;
    }

    let total_logs = filtered_logs.len();

    // Calculate visible range
    let scroll = app.log_scroll as usize;
    let start = scroll.min(total_logs.saturating_sub(visible_height));
    let end = (start + visible_height).min(total_logs);

    for entry in filtered_logs.iter().skip(start).take(end - start) {
        let time = entry.timestamp.format("%H:%M:%S");
        let level_style = Style::default().fg(entry.level.color());
        let prefix = format!("[{}] {} ", time, entry.level.prefix());
        let prefix_len = prefix.len();

        // If message is short enough, display on one line
        if prefix_len + entry.message.len() <= available_width {
            items.push(ListItem::new(Line::from(vec![
                Span::styled(
                    format!("[{time}]"),
                    Style::default().fg(ratatui::style::Color::DarkGray),
                ),
                Span::raw(" "),
                Span::styled(entry.level.prefix().to_string(), level_style),
                Span::raw(" "),
                Span::raw(entry.message.clone()),
            ])));
        } else {
            // Wrap long messages across multiple lines
            let msg_width = available_width.saturating_sub(prefix_len).max(10);

            // First line with timestamp and level
            let first_chunk: String = entry.message.chars().take(msg_width).collect();
            items.push(ListItem::new(Line::from(vec![
                Span::styled(
                    format!("[{time}]"),
                    Style::default().fg(ratatui::style::Color::DarkGray),
                ),
                Span::raw(" "),
                Span::styled(entry.level.prefix().to_string(), level_style),
                Span::raw(" "),
                Span::raw(first_chunk),
            ])));

            // Continuation lines (indented)
            let indent = " ".repeat(prefix_len);
            let remaining: String = entry.message.chars().skip(msg_width).collect();
            let cont_width = available_width.saturating_sub(2).max(10);

            for chunk in remaining.chars().collect::<Vec<_>>().chunks(cont_width) {
                let chunk_str: String = chunk.iter().collect();
                items.push(ListItem::new(Line::from(vec![
                    Span::styled(
                        indent.clone(),
                        Style::default().fg(ratatui::style::Color::DarkGray),
                    ),
                    Span::raw(chunk_str),
                ])));
            }
        }
    }

    let list = List::new(items);
    f.render_widget(list, area);
}
