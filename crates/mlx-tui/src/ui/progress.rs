//! Progress bars component

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    widgets::{Block, Borders, Gauge},
};

use crate::app::App;

/// Draw progress bars for epoch and step
pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title("Progress");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1)])
        .split(inner);

    // Epoch progress
    let epoch_progress = app.epoch_progress();
    let epoch_label = format!(
        "Epoch {}/{} ({:.0}%)",
        app.current_epoch,
        app.total_epochs,
        epoch_progress * 100.0
    );
    let epoch_gauge = Gauge::default()
        .ratio(epoch_progress)
        .label(epoch_label)
        .gauge_style(Style::default().fg(Color::Cyan).bg(Color::DarkGray));
    f.render_widget(epoch_gauge, chunks[0]);

    // Step progress within epoch
    let step_progress = app.step_progress();
    let step_label = format!(
        "Step {}/{} ({:.0}%)",
        app.step_in_epoch,
        app.total_steps_in_epoch,
        step_progress * 100.0
    );
    let step_gauge = Gauge::default()
        .ratio(step_progress)
        .label(step_label)
        .gauge_style(Style::default().fg(Color::Yellow).bg(Color::DarkGray));
    f.render_widget(step_gauge, chunks[1]);
}
