//! Metrics panel with charts

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style, Stylize},
    symbols,
    text::Span,
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph},
};
use std::collections::VecDeque;

use crate::app::App;

/// Draw the metrics panel with charts
pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title("Metrics");
    let inner = block.inner(area);
    f.render_widget(block, area);

    // For SFT training, show Loss chart
    // For GRPO training, show Loss/Reward chart
    if app.training_type == "sft" {
        draw_sft_metrics(f, app, inner);
    } else {
        draw_grpo_metrics(f, app, inner);
    }
}

/// Draw SFT metrics (Loss chart + current values)
fn draw_sft_metrics(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Current values row
            Constraint::Min(8),    // Chart
        ])
        .split(area);

    // Current values with trend indicators
    draw_current_values_sft(f, app, chunks[0]);

    // Loss chart
    draw_loss_chart(f, app, chunks[1]);
}

/// Draw GRPO metrics (Loss/Reward chart + current values)
fn draw_grpo_metrics(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Current values row
            Constraint::Min(8),    // Chart
        ])
        .split(area);

    // Current values with trend indicators
    draw_current_values_grpo(f, app, chunks[0]);

    // Loss/Reward chart
    draw_loss_reward_chart(f, app, chunks[1]);
}

/// Draw current SFT values row
fn draw_current_values_sft(f: &mut Frame, app: &App, area: Rect) {
    use ratatui::text::Line;

    let loss_trend = App::trend_indicator(app.current_loss, app.prev_loss);
    let loss_trend_color = trend_color_lower_better(loss_trend);

    let ppl_trend = App::trend_indicator(app.current_perplexity, app.prev_perplexity);
    let ppl_trend_color = trend_color_lower_better(ppl_trend);

    let line = Line::from(vec![
        Span::styled("Loss ", Style::default().fg(Color::Red)),
        Span::styled(
            format!("{:>7.4}", app.current_loss),
            Style::default().fg(Color::Red),
        ),
        Span::styled(
            format!(" {} ", loss_trend),
            Style::default().fg(loss_trend_color),
        ),
        Span::raw("  "),
        Span::styled("Perplx ", Style::default().fg(Color::Yellow)),
        Span::styled(
            format!("{:>7.2}", app.current_perplexity),
            Style::default().fg(Color::Yellow),
        ),
        Span::styled(
            format!(" {} ", ppl_trend),
            Style::default().fg(ppl_trend_color),
        ),
    ]);

    let para = Paragraph::new(line);
    f.render_widget(para, area);
}

/// Draw current GRPO values row
fn draw_current_values_grpo(f: &mut Frame, app: &App, area: Rect) {
    use ratatui::text::Line;

    let loss_trend = App::trend_indicator(app.current_loss, app.prev_loss);
    let loss_trend_color = trend_color_lower_better(loss_trend);

    let reward_trend = App::trend_indicator(app.current_reward, app.prev_reward);
    let reward_trend_color = trend_color_higher_better(reward_trend);

    let adv_trend = App::trend_indicator(app.current_std_advantage, app.prev_std_advantage);
    let adv_trend_color = trend_color_higher_better(adv_trend);

    let line = Line::from(vec![
        Span::styled("Loss ", Style::default().fg(Color::Red)),
        Span::styled(
            format!("{:>7.4}", app.current_loss),
            Style::default().fg(Color::Red),
        ),
        Span::styled(
            loss_trend.to_string(),
            Style::default().fg(loss_trend_color),
        ),
        Span::raw("  "),
        Span::styled("Reward ", Style::default().fg(Color::Green)),
        Span::styled(
            format!("{:>6.3}", app.current_reward),
            Style::default().fg(Color::Green),
        ),
        Span::styled(
            reward_trend.to_string(),
            Style::default().fg(reward_trend_color),
        ),
        Span::raw("  "),
        Span::styled("Adv.Std ", Style::default().fg(Color::Blue)),
        Span::styled(
            format!("{:>6.3}", app.current_std_advantage),
            Style::default().fg(Color::Blue),
        ),
        Span::styled(adv_trend.to_string(), Style::default().fg(adv_trend_color)),
    ]);

    let para = Paragraph::new(line);
    f.render_widget(para, area);
}

/// Draw loss-only chart for SFT
fn draw_loss_chart(f: &mut Frame, app: &App, area: Rect) {
    if app.loss_history.is_empty() {
        return;
    }

    let loss_data = to_chart_data(&app.loss_history);
    let (min_loss, max_loss) = get_bounds(&app.loss_history);

    let datasets = vec![
        Dataset::default()
            .name("Loss")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Red))
            .data(&loss_data),
    ];

    let x_max = loss_data.len().max(1) as f64;
    let x_labels = make_x_labels(x_max);
    let y_labels = make_y_labels(min_loss, max_loss);

    let chart = Chart::new(datasets)
        .x_axis(
            Axis::default()
                .style(Style::default().fg(Color::DarkGray))
                .bounds([0.0, x_max])
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .title("Loss".red())
                .style(Style::default().fg(Color::DarkGray))
                .bounds([min_loss, max_loss])
                .labels(y_labels),
        );

    f.render_widget(chart, area);
}

/// Draw combined loss/reward chart for GRPO
fn draw_loss_reward_chart(f: &mut Frame, app: &App, area: Rect) {
    if app.loss_history.is_empty() && app.reward_history.is_empty() {
        return;
    }

    // Normalize both to 0-1 range for dual display
    let loss_data = normalize_to_chart_data(&app.loss_history, true); // Invert so down is good
    let reward_data = normalize_to_chart_data(&app.reward_history, false);

    let x_max = loss_data.len().max(reward_data.len()).max(1) as f64;

    let mut datasets = vec![];

    if !loss_data.is_empty() {
        datasets.push(
            Dataset::default()
                .name("Loss ↓")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Red))
                .data(&loss_data),
        );
    }

    if !reward_data.is_empty() {
        datasets.push(
            Dataset::default()
                .name("Reward ↑")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Green))
                .data(&reward_data),
        );
    }

    let x_labels = make_x_labels(x_max);

    let chart = Chart::new(datasets)
        .x_axis(
            Axis::default()
                .title("Step".dark_gray())
                .style(Style::default().fg(Color::DarkGray))
                .bounds([0.0, x_max])
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .title("Normalized")
                .style(Style::default().fg(Color::DarkGray))
                .bounds([0.0, 1.0])
                .labels(vec![
                    Span::styled("0%", Style::default().fg(Color::DarkGray)),
                    Span::styled("50%", Style::default().fg(Color::DarkGray)),
                    Span::styled("100%", Style::default().fg(Color::DarkGray)),
                ]),
        )
        .legend_position(Some(ratatui::widgets::LegendPosition::TopRight));

    f.render_widget(chart, area);
}

/// Convert VecDeque to chart data points (x, y)
fn to_chart_data(data: &VecDeque<f64>) -> Vec<(f64, f64)> {
    data.iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v))
        .collect()
}

/// Normalize data to 0-1 range and convert to chart data points
/// If `invert` is true, high values become low (for loss where lower is better)
fn normalize_to_chart_data(data: &VecDeque<f64>, invert: bool) -> Vec<(f64, f64)> {
    if data.is_empty() {
        return vec![];
    }

    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(0.001);

    data.iter()
        .enumerate()
        .map(|(i, &v)| {
            let normalized = (v - min) / range;
            let y = if invert { 1.0 - normalized } else { normalized };
            (i as f64, y)
        })
        .collect()
}

/// Get min/max bounds with some padding
fn get_bounds(data: &VecDeque<f64>) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 1.0);
    }

    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(0.001);
    let padding = range * 0.1;

    ((min - padding).max(0.0), max + padding)
}

/// Create X-axis labels
fn make_x_labels(max: f64) -> Vec<Span<'static>> {
    let mid = (max / 2.0) as i64;
    vec![
        Span::styled("0", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{}", mid), Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{}", max as i64),
            Style::default().fg(Color::DarkGray),
        ),
    ]
}

/// Create Y-axis labels
fn make_y_labels(min: f64, max: f64) -> Vec<Span<'static>> {
    let mid = (min + max) / 2.0;
    vec![
        Span::styled(format!("{:.2}", min), Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{:.2}", mid), Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{:.2}", max), Style::default().fg(Color::DarkGray)),
    ]
}

/// Get trend color where lower values are better (loss, perplexity)
fn trend_color_lower_better(trend: &str) -> Color {
    match trend {
        "↓" => Color::Green,
        "↑" => Color::Red,
        _ => Color::DarkGray,
    }
}

/// Get trend color where higher values are better (reward, accuracy)
fn trend_color_higher_better(trend: &str) -> Color {
    match trend {
        "↑" => Color::Green,
        "↓" => Color::Red,
        _ => Color::DarkGray,
    }
}
