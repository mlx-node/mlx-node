//! Configuration display component

use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::{List, ListItem, Paragraph},
};

use crate::app::App;

/// Draw the config display
pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let Some(config) = &app.config else {
        let empty = Paragraph::new("Waiting for training config...")
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(empty, area);
        return;
    };

    let mut items: Vec<ListItem> = Vec::new();

    // Helper function to create config items
    fn make_item(label: &str, value: String) -> ListItem<'static> {
        ListItem::new(Line::from(vec![
            Span::styled(
                format!("{:24}", label),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(value, Style::default().fg(Color::Cyan)),
        ]))
    }

    fn section_header(title: &str) -> ListItem<'static> {
        ListItem::new(Line::from(Span::styled(
            title.to_string(),
            Style::default().fg(Color::Yellow),
        )))
    }

    // Training parameters
    items.push(section_header("─── Training ───"));

    if let Some(lr) = config.learning_rate {
        items.push(make_item("Learning Rate", format!("{:.2e}", lr)));
    }
    if let Some(bs) = config.batch_size {
        items.push(make_item("Batch Size", bs.to_string()));
    }
    if let Some(epochs) = config.num_epochs {
        items.push(make_item("Epochs", epochs.to_string()));
    }
    if let Some(accum) = config.gradient_accumulation_steps {
        items.push(make_item("Gradient Accumulation", accum.to_string()));
    }

    items.push(ListItem::new(Line::from("")));

    // GRPO parameters
    items.push(section_header("─── GRPO ───"));

    if let Some(gs) = config.group_size {
        items.push(make_item("Group Size", gs.to_string()));
    }
    if let Some(clip) = config.clip_epsilon {
        items.push(make_item("Clip Epsilon", format!("{:.2}", clip)));
    }
    if let Some(ref loss_type) = config.loss_type {
        items.push(make_item("Loss Type", loss_type.clone()));
    }

    items.push(ListItem::new(Line::from("")));

    // Generation parameters
    items.push(section_header("─── Generation ───"));

    if let Some(tokens) = config.max_new_tokens {
        items.push(make_item("Max New Tokens", tokens.to_string()));
    }
    if let Some(temp) = config.temperature {
        items.push(make_item("Temperature", format!("{:.2}", temp)));
    }

    // Extra fields
    if !config.extra.is_empty() {
        items.push(ListItem::new(Line::from("")));
        items.push(section_header("─── Other ───"));

        for (key, value) in &config.extra {
            let value_str = match value {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                serde_json::Value::Bool(b) => b.to_string(),
                _ => format!("{}", value),
            };
            items.push(make_item(key, value_str));
        }
    }

    let list = List::new(items);
    f.render_widget(list, area);
}
