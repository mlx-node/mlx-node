//! Header bar component

use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
};

use crate::app::App;

/// Draw the header bar
pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let model_name = if app.model_name.is_empty() {
        "Loading..."
    } else {
        &app.model_name
    };

    let epoch_info = if app.total_epochs > 0 {
        format!("Epoch {}/{}", app.current_epoch, app.total_epochs)
    } else {
        "Epoch -/-".to_string()
    };

    let step_info = if app.total_steps_in_epoch > 0 {
        format!("Step {}/{}", app.step_in_epoch, app.total_steps_in_epoch)
    } else {
        format!("Step {}", app.current_step)
    };

    let state = app.state;
    let state_style = Style::default()
        .fg(state.color())
        .add_modifier(Modifier::BOLD);

    let spans = vec![
        Span::raw(" "), // Left padding
        Span::styled(
            "@mlx-node/trl",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled(model_name, Style::default().fg(Color::White)),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled(epoch_info, Style::default().fg(Color::White)),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled(step_info, Style::default().fg(Color::White)),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled(state.icon(), state_style),
        Span::raw(" "),
        Span::styled(state.display(), state_style),
    ];

    let paragraph = Paragraph::new(Line::from(spans));
    f.render_widget(paragraph, area);
}
