//! Generation samples viewer component

use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{List, ListItem, Paragraph},
};

use crate::app::App;

/// Draw the samples viewer
pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    if app.samples.is_empty() {
        let empty = Paragraph::new("No samples yet. Samples will appear as training progresses.")
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(empty, area);
        return;
    }

    // Calculate sample stats
    let total_samples = app.samples.len();
    let (best_reward, worst_reward, avg_reward) = if !app.samples.is_empty() {
        let rewards: Vec<f64> = app.samples.iter().map(|s| s.reward).collect();
        let best = rewards.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let worst = rewards.iter().copied().fold(f64::INFINITY, f64::min);
        let avg = rewards.iter().sum::<f64>() / rewards.len() as f64;
        (best, worst, avg)
    } else {
        (0.0, 0.0, 0.0)
    };

    // Show mode indicator at top with hint
    let mode_line = Line::from(vec![
        Span::styled("Mode: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            app.sample_display_mode.display_name(),
            Style::default().fg(Color::Cyan),
        ),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled("Enter", Style::default().fg(Color::Yellow)),
        Span::styled("/", Style::default().fg(Color::DarkGray)),
        Span::styled("Click", Style::default().fg(Color::Yellow)),
        Span::styled(" to view full", Style::default().fg(Color::DarkGray)),
    ]);

    // Stats summary line
    let stats_line = Line::from(vec![
        Span::styled("Best:", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{:.2}", best_reward),
            Style::default().fg(Color::Green),
        ),
        Span::styled(" Worst:", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{:.2}", worst_reward),
            Style::default().fg(Color::Red),
        ),
        Span::styled(" Avg:", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{:.2}", avg_reward),
            Style::default().fg(Color::Cyan),
        ),
        Span::styled(
            format!(" │ {} samples", total_samples),
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    let visible_height = area.height.saturating_sub(3) as usize; // Account for extra header line

    // Calculate visible range
    let scroll = app.sample_scroll as usize;
    let start = scroll.min(total_samples.saturating_sub(1));

    // Build sample items with proper line breaks
    let mut all_lines: Vec<Line> = vec![mode_line, stats_line, Line::from("")];

    for sample in app.samples.iter().skip(start) {
        // Check if we have room for at least the header
        if all_lines.len() >= visible_height {
            break;
        }

        // Color reward based on value
        let reward_color = if sample.reward > 0.5 {
            Color::Green
        } else if sample.reward > 0.0 {
            Color::Yellow
        } else {
            Color::Red
        };

        // Sample header
        all_lines.push(Line::from(vec![
            Span::styled("━━━ ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("Sample #{}", sample.index),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" ━━━", Style::default().fg(Color::DarkGray)),
        ]));

        // Reward and tokens
        all_lines.push(Line::from(vec![
            Span::styled("Reward: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.4}", sample.reward),
                Style::default().fg(reward_color),
            ),
            Span::styled(" | Tokens: ", Style::default().fg(Color::DarkGray)),
            Span::styled(sample.tokens.to_string(), Style::default().fg(Color::Cyan)),
        ]));

        // Prompt - show first line or truncate if too long
        let prompt_preview = sample.prompt.lines().next().unwrap_or("");
        let prompt_display = if prompt_preview.len() > 60 {
            format!("{}...", &prompt_preview[..60])
        } else if sample.prompt.lines().count() > 1 {
            format!("{}...", prompt_preview)
        } else {
            prompt_preview.to_string()
        };
        all_lines.push(Line::from(vec![
            Span::styled("Prompt: ", Style::default().fg(Color::DarkGray)),
            Span::styled(prompt_display, Style::default().fg(Color::Blue)),
        ]));

        // Output - show first line only (click/Enter for full view)
        let output_preview = sample.completion.lines().next().unwrap_or("");
        let output_display = if output_preview.len() > 80 {
            format!("{}...", &output_preview[..80])
        } else if sample.completion.lines().count() > 1 {
            format!("{}...", output_preview)
        } else {
            output_preview.to_string()
        };
        all_lines.push(Line::from(vec![
            Span::styled("Output: ", Style::default().fg(Color::DarkGray)),
            Span::styled(output_display, Style::default().fg(Color::White)),
        ]));

        // Blank line between samples
        all_lines.push(Line::from(""));
    }

    let items: Vec<ListItem> = all_lines.into_iter().map(ListItem::new).collect();
    let list = List::new(items);
    f.render_widget(list, area);
}
