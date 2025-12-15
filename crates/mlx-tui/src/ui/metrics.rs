//! Metrics panel with sparklines

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    symbols,
    widgets::{Block, Borders, Paragraph, Sparkline},
};
use std::collections::VecDeque;

use crate::app::App;

/// Draw the metrics panel with sparklines
pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title("Metrics");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(4),
            Constraint::Length(4),
        ])
        .split(inner);

    // For loss, down is good (green arrow), up is bad (red arrow)
    let loss_trend = App::trend_indicator(app.current_loss, app.prev_loss);
    let loss_trend_color = if loss_trend == "↓" {
        Color::Green
    } else if loss_trend == "↑" {
        Color::Red
    } else {
        Color::DarkGray
    };

    // For reward, up is good (green arrow), down is bad (red arrow)
    let reward_trend = App::trend_indicator(app.current_reward, app.prev_reward);
    let reward_trend_color = if reward_trend == "↑" {
        Color::Green
    } else if reward_trend == "↓" {
        Color::Red
    } else {
        Color::DarkGray
    };

    // For advantage, neutral coloring
    let adv_trend = App::trend_indicator(app.current_advantage, app.prev_advantage);

    draw_metric_row(
        f,
        "Loss",
        app.current_loss,
        loss_trend,
        loss_trend_color,
        &app.loss_history,
        Color::Red,
        chunks[0],
    );
    draw_metric_row(
        f,
        "Reward",
        app.current_reward,
        reward_trend,
        reward_trend_color,
        &app.reward_history,
        Color::Green,
        chunks[1],
    );
    draw_metric_row(
        f,
        "Advant.",
        app.current_advantage,
        adv_trend,
        Color::DarkGray,
        &app.advantage_history,
        Color::Blue,
        chunks[2],
    );
}

/// Draw a single metric row with label, value, trend indicator, and sparkline
#[allow(clippy::too_many_arguments)]
fn draw_metric_row(
    f: &mut Frame,
    label: &str,
    current: f64,
    trend: &str,
    trend_color: Color,
    history: &VecDeque<f64>,
    color: Color,
    area: Rect,
) {
    use ratatui::text::{Line, Span};

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(20), // Label + value + trend
            Constraint::Min(10),    // Sparkline
        ])
        .split(area);

    // Label, current value, and trend indicator
    let spans = vec![
        Span::styled(format!("{:8} ", label), Style::default().fg(color)),
        Span::styled(format!("{:>7.4}", current), Style::default().fg(color)),
        Span::raw(" "),
        Span::styled(trend, Style::default().fg(trend_color)),
    ];
    let label_widget = Paragraph::new(Line::from(spans));
    f.render_widget(label_widget, chunks[0]);

    // Sparkline
    if !history.is_empty() {
        let data = normalize_for_sparkline(history);
        let sparkline = Sparkline::default()
            .data(&data)
            .style(Style::default().fg(color))
            .bar_set(symbols::bar::NINE_LEVELS);
        f.render_widget(sparkline, chunks[1]);
    }
}

/// Normalize values to 0-100 range for sparkline
fn normalize_for_sparkline(data: &VecDeque<f64>) -> Vec<u64> {
    if data.is_empty() {
        return vec![];
    }

    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(0.001); // Avoid division by zero

    data.iter()
        .map(|&v| (((v - min) / range) * 100.0) as u64)
        .collect()
}
