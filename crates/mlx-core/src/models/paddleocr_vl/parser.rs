/**
 * PaddleOCR-VL Output Parser
 *
 * Parses structured table tokens from VLM output into a document structure,
 * then formats it according to user preferences (plain text, markdown, etc.)
 *
 * Table tokens used by PaddleOCR-VL:
 * - <nl>   : newline / row separator
 * - <lcel> : left cell boundary
 * - <ecel> : empty cell / cell end
 * - <fcel> : first cell in row
 * - <bcel> : blank/bottom cell
 * - <rcel> : right cell boundary
 */
use napi_derive::napi;
use unicode_width::UnicodeWidthStr;

/// A single cell in a table
#[napi(object)]
#[derive(Debug, Clone)]
pub struct TableCell {
    pub content: String,
    pub is_empty: bool,
}

/// A row in a table
#[napi(object)]
#[derive(Debug, Clone)]
pub struct TableRow {
    pub cells: Vec<TableCell>,
}

/// A table structure
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Table {
    pub rows: Vec<TableRow>,
}

/// A text paragraph
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Paragraph {
    pub content: String,
}

/// Document element type
#[napi(string_enum)]
#[derive(Debug, Clone, PartialEq)]
pub enum ElementType {
    Table,
    Paragraph,
}

/// Document element - either a table or paragraph
#[napi(object)]
#[derive(Debug, Clone)]
pub struct DocumentElement {
    pub element_type: ElementType,
    /// Table data (only present if element_type is Table)
    pub table: Option<Table>,
    /// Paragraph data (only present if element_type is Paragraph)
    pub paragraph: Option<Paragraph>,
}

/// Parsed document structure
#[napi(object)]
#[derive(Debug, Clone)]
pub struct ParsedDocument {
    pub elements: Vec<DocumentElement>,
}

/// Output format options
#[napi(string_enum)]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum OutputFormat {
    /// Raw output with minimal processing
    Raw,
    /// Plain text with aligned columns
    Plain,
    /// Markdown tables
    #[default]
    Markdown,
    /// HTML tables
    Html,
}

/// Parser configuration
#[napi(object)]
#[derive(Debug, Clone)]
pub struct ParserConfig {
    /// Output format (default: 'markdown')
    pub format: Option<OutputFormat>,
    /// Whether to trim whitespace from cells (default: true)
    pub trim_cells: Option<bool>,
    /// Whether to collapse empty rows (default: true)
    pub collapse_empty_rows: Option<bool>,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            format: Some(OutputFormat::Markdown),
            trim_cells: Some(true),
            collapse_empty_rows: Some(true),
        }
    }
}

/// Parse VLM output into structured document
fn parse_vlm_output_internal(text: &str) -> ParsedDocument {
    let mut elements: Vec<DocumentElement> = Vec::new();

    // Split by <nl> to get rows/lines
    let parts: Vec<&str> = text.split("<nl>").collect();

    let mut current_table: Option<Table> = None;
    let mut current_paragraph = String::new();

    for part in parts {
        // Check if this part contains table cell markers
        let has_table_tokens = part.contains("<lcel>")
            || part.contains("<fcel>")
            || part.contains("<ecel>")
            || part.contains("<rcel>")
            || part.contains("<bcel>");

        if has_table_tokens {
            // If we have pending paragraph text, save it
            let trimmed = current_paragraph.trim();
            if !trimmed.is_empty() {
                elements.push(DocumentElement {
                    element_type: ElementType::Paragraph,
                    table: None,
                    paragraph: Some(Paragraph {
                        content: trimmed.to_string(),
                    }),
                });
                current_paragraph.clear();
            }

            // Start a new table if needed
            if current_table.is_none() {
                current_table = Some(Table { rows: Vec::new() });
            }

            // Parse this row
            let row = parse_table_row(part);
            if let Some(ref mut table) = current_table {
                table.rows.push(row);
            }
        } else {
            // This is regular text
            // If we have a table in progress, save it
            if let Some(table) = current_table.take() {
                elements.push(DocumentElement {
                    element_type: ElementType::Table,
                    table: Some(table),
                    paragraph: None,
                });
            }

            // Add to current paragraph
            let trimmed = part.trim();
            if !trimmed.is_empty() {
                if !current_paragraph.is_empty() {
                    current_paragraph.push('\n');
                }
                current_paragraph.push_str(trimmed);
            }
        }
    }

    // Save any remaining content
    if let Some(table) = current_table {
        elements.push(DocumentElement {
            element_type: ElementType::Table,
            table: Some(table),
            paragraph: None,
        });
    }
    let trimmed = current_paragraph.trim();
    if !trimmed.is_empty() {
        elements.push(DocumentElement {
            element_type: ElementType::Paragraph,
            table: None,
            paragraph: Some(Paragraph {
                content: trimmed.to_string(),
            }),
        });
    }

    ParsedDocument { elements }
}

/// All recognized cell marker tags
const CELL_MARKERS: [&str; 5] = ["<fcel>", "<ecel>", "<lcel>", "<rcel>", "<bcel>"];

/// Check if the string at given byte position starts with a cell marker
/// Returns the marker if found, None otherwise
fn find_marker_at(text: &str, pos: usize) -> Option<&'static str> {
    let remaining = &text[pos..];
    CELL_MARKERS
        .into_iter()
        .find(|&marker| remaining.starts_with(marker))
        .map(|v| v as _)
}

/// Parse a single table row from text with cell markers
fn parse_table_row(text: &str) -> TableRow {
    let mut cells: Vec<TableCell> = Vec::new();
    let mut pending_content = String::new();
    let mut byte_pos = 0;

    while byte_pos < text.len() {
        // Check for cell markers at current byte position
        if let Some(marker) = find_marker_at(text, byte_pos) {
            match marker {
                "<fcel>" | "<lcel>" => {
                    // First cell or left cell - start new cell, save pending if any
                    if !pending_content.trim().is_empty() {
                        cells.push(TableCell {
                            content: pending_content.clone(),
                            is_empty: false,
                        });
                    }
                    pending_content.clear();
                }
                "<ecel>" => {
                    // Empty cell - save pending and add empty cell
                    if !pending_content.trim().is_empty() {
                        cells.push(TableCell {
                            content: pending_content.clone(),
                            is_empty: false,
                        });
                    }
                    pending_content.clear();
                    cells.push(TableCell {
                        content: String::new(),
                        is_empty: true,
                    });
                }
                "<rcel>" => {
                    // Right cell - end of cell, save content
                    if !pending_content.trim().is_empty() {
                        cells.push(TableCell {
                            content: pending_content.clone(),
                            is_empty: false,
                        });
                    }
                    pending_content.clear();
                }
                "<bcel>" => {
                    // Blank cell
                    if !pending_content.trim().is_empty() {
                        cells.push(TableCell {
                            content: pending_content.clone(),
                            is_empty: false,
                        });
                    }
                    pending_content.clear();
                    cells.push(TableCell {
                        content: String::new(),
                        is_empty: true,
                    });
                }
                _ => unreachable!(),
            }
            byte_pos += marker.len();
            continue;
        }

        // Regular character - get the next UTF-8 character and advance by its byte length
        // Safety: byte_pos is always at a valid UTF-8 character boundary because:
        // 1. We start at 0 (valid)
        // 2. We only advance by marker.len() (ASCII markers, always valid boundaries)
        // 3. We advance by char.len_utf8() (always lands on next char boundary)
        let remaining = &text[byte_pos..];
        if let Some(ch) = remaining.chars().next() {
            pending_content.push(ch);
            byte_pos += ch.len_utf8();
        } else {
            // Should not happen with valid UTF-8, but handle gracefully
            break;
        }
    }

    // Handle any remaining content
    if !pending_content.trim().is_empty() {
        cells.push(TableCell {
            content: pending_content.clone(),
            is_empty: false,
        });
    }

    TableRow { cells }
}

/// Format parsed document according to config
fn format_document_internal(doc: &ParsedDocument, config: &ParserConfig) -> String {
    let format = config.format.clone().unwrap_or(OutputFormat::Markdown);
    let trim_cells = config.trim_cells.unwrap_or(true);
    let collapse_empty_rows = config.collapse_empty_rows.unwrap_or(true);

    match format {
        OutputFormat::Raw => format_raw(doc),
        OutputFormat::Plain => format_plain(doc, trim_cells, collapse_empty_rows),
        OutputFormat::Markdown => format_markdown(doc, trim_cells, collapse_empty_rows),
        OutputFormat::Html => format_html(doc, trim_cells, collapse_empty_rows),
    }
}

/// Raw format - just join elements
fn format_raw(doc: &ParsedDocument) -> String {
    doc.elements
        .iter()
        .map(|el| {
            if el.element_type == ElementType::Paragraph {
                el.paragraph
                    .as_ref()
                    .map(|p| p.content.clone())
                    .unwrap_or_default()
            } else {
                el.table
                    .as_ref()
                    .map(|t| {
                        t.rows
                            .iter()
                            .map(|row| {
                                row.cells
                                    .iter()
                                    .map(|c| c.content.as_str())
                                    .collect::<Vec<_>>()
                                    .join("\t")
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default()
            }
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Plain text format
fn format_plain(doc: &ParsedDocument, trim_cells: bool, collapse_empty_rows: bool) -> String {
    let mut parts: Vec<String> = Vec::new();

    for el in &doc.elements {
        if el.element_type == ElementType::Paragraph {
            if let Some(p) = &el.paragraph {
                parts.push(p.content.clone());
            }
        } else if let Some(table) = &el.table {
            // Filter rows if needed
            let rows: Vec<&TableRow> = if collapse_empty_rows {
                table
                    .rows
                    .iter()
                    .filter(|r| r.cells.iter().any(|c| !c.is_empty))
                    .collect()
            } else {
                table.rows.iter().collect()
            };

            if rows.is_empty() {
                continue;
            }

            // Calculate column widths
            let col_count = rows.iter().map(|r| r.cells.len()).max().unwrap_or(0);
            let mut col_widths: Vec<usize> = vec![0; col_count];

            for row in &rows {
                for (i, cell) in row.cells.iter().enumerate() {
                    let content = if trim_cells {
                        cell.content.trim()
                    } else {
                        &cell.content
                    };
                    if i < col_widths.len() {
                        col_widths[i] = col_widths[i].max(UnicodeWidthStr::width(content));
                    }
                }
            }

            // Format rows
            for row in rows {
                let cells: Vec<String> = row
                    .cells
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let content = if trim_cells {
                            c.content.trim()
                        } else {
                            &c.content
                        };
                        let width = col_widths.get(i).copied().unwrap_or(0);
                        format!("{:width$}", content, width = width)
                    })
                    .collect();
                parts.push(cells.join("  "));
            }
        }
    }

    parts.join("\n\n")
}

/// Markdown format
fn format_markdown(doc: &ParsedDocument, trim_cells: bool, collapse_empty_rows: bool) -> String {
    let mut parts: Vec<String> = Vec::new();

    for el in &doc.elements {
        if el.element_type == ElementType::Paragraph {
            if let Some(p) = &el.paragraph {
                parts.push(p.content.clone());
            }
        } else if let Some(table) = &el.table {
            // Filter rows if needed
            let rows: Vec<&TableRow> = if collapse_empty_rows {
                table
                    .rows
                    .iter()
                    .filter(|r| r.cells.iter().any(|c| !c.is_empty))
                    .collect()
            } else {
                table.rows.iter().collect()
            };

            if rows.is_empty() {
                continue;
            }

            let col_count = rows.iter().map(|r| r.cells.len()).max().unwrap_or(0);

            // Build markdown table
            let mut table_lines: Vec<String> = Vec::new();

            for (row_idx, row) in rows.iter().enumerate() {
                let mut cells: Vec<String> = Vec::new();

                for i in 0..col_count {
                    let cell = row.cells.get(i);
                    let content = cell
                        .map(|c| {
                            if trim_cells {
                                c.content.trim()
                            } else {
                                &c.content
                            }
                        })
                        .unwrap_or("");
                    // Escape pipe characters in content
                    cells.push(content.replace('|', "\\|"));
                }

                table_lines.push(format!("| {} |", cells.join(" | ")));

                // Add header separator after first row
                if row_idx == 0 {
                    let separator = vec!["---"; col_count].join(" | ");
                    table_lines.push(format!("| {} |", separator));
                }
            }

            parts.push(table_lines.join("\n"));
        }
    }

    parts.join("\n\n")
}

/// HTML format
fn format_html(doc: &ParsedDocument, trim_cells: bool, collapse_empty_rows: bool) -> String {
    let mut parts: Vec<String> = Vec::new();

    for el in &doc.elements {
        if el.element_type == ElementType::Paragraph {
            if let Some(p) = &el.paragraph {
                parts.push(format!("<p>{}</p>", escape_html(&p.content)));
            }
        } else if let Some(table) = &el.table {
            let rows: Vec<&TableRow> = if collapse_empty_rows {
                table
                    .rows
                    .iter()
                    .filter(|r| r.cells.iter().any(|c| !c.is_empty))
                    .collect()
            } else {
                table.rows.iter().collect()
            };

            if rows.is_empty() {
                continue;
            }

            let mut table_lines: Vec<String> = vec!["<table>".to_string()];

            for (row_idx, row) in rows.iter().enumerate() {
                let tag = if row_idx == 0 { "th" } else { "td" };

                table_lines.push("  <tr>".to_string());
                for cell in &row.cells {
                    let content = if trim_cells {
                        cell.content.trim()
                    } else {
                        &cell.content
                    };
                    table_lines.push(format!("    <{tag}>{}</{tag}>", escape_html(content)));
                }
                table_lines.push("  </tr>".to_string());
            }

            table_lines.push("</table>".to_string());
            parts.push(table_lines.join("\n"));
        }
    }

    parts.join("\n\n")
}

fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ============== NAPI Exports ==============

/// Parse VLM output into structured document
#[napi]
pub fn parse_vlm_output(text: String) -> ParsedDocument {
    parse_vlm_output_internal(&text)
}

/// Format parsed document according to config
#[napi]
pub fn format_document(doc: ParsedDocument, config: Option<ParserConfig>) -> String {
    let cfg = config.unwrap_or_default();
    format_document_internal(&doc, &cfg)
}

/// Parse and format PaddleOCR-VL response in one step
///
/// Convenience function that parses the VLM output and formats it
/// according to the specified configuration.
///
/// # Arguments
/// * `text` - Raw VLM output containing table tokens
/// * `config` - Optional parser configuration (format, trim_cells, etc.)
///
/// # Returns
/// * Formatted string in the requested format (markdown, plain, html, raw)
///
/// # Example
/// ```typescript
/// import { parsePaddleResponse } from '@mlx-node/core';
///
/// // Parse and format as markdown (default)
/// const markdown = parsePaddleResponse(vlmResult.text);
///
/// // Parse and format as HTML
/// const html = parsePaddleResponse(vlmResult.text, { format: 'html' });
///
/// // Parse and format as plain text
/// const plain = parsePaddleResponse(vlmResult.text, { format: 'plain' });
/// ```
#[napi]
pub fn parse_paddle_response(text: String, config: Option<ParserConfig>) -> String {
    let cfg = config.unwrap_or_default();
    let doc = parse_vlm_output_internal(&text);
    format_document_internal(&doc, &cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_text() {
        let doc = parse_vlm_output_internal("Hello World");
        assert_eq!(doc.elements.len(), 1);
        assert_eq!(doc.elements[0].element_type, ElementType::Paragraph);
        assert_eq!(
            doc.elements[0].paragraph.as_ref().unwrap().content,
            "Hello World"
        );
    }

    #[test]
    fn test_parse_text_with_newlines() {
        let doc = parse_vlm_output_internal("Line 1<nl>Line 2<nl>Line 3");
        assert_eq!(doc.elements.len(), 1);
        assert_eq!(doc.elements[0].element_type, ElementType::Paragraph);
        assert_eq!(
            doc.elements[0].paragraph.as_ref().unwrap().content,
            "Line 1\nLine 2\nLine 3"
        );
    }

    #[test]
    fn test_parse_table_with_cells() {
        let doc =
            parse_vlm_output_internal("<fcel>Header 1<ecel><ecel><nl><fcel>Row 1<ecel><ecel>");
        assert_eq!(doc.elements.len(), 1);
        assert_eq!(doc.elements[0].element_type, ElementType::Table);

        let table = doc.elements[0].table.as_ref().unwrap();
        assert_eq!(table.rows.len(), 2);
    }

    #[test]
    fn test_format_markdown() {
        let doc = parse_vlm_output_internal("<fcel>Name<ecel><ecel><nl><fcel>Alice<ecel><ecel>");
        let cfg = ParserConfig {
            format: Some(OutputFormat::Markdown),
            ..Default::default()
        };
        let formatted = format_document_internal(&doc, &cfg);

        assert!(formatted.contains("|"));
        assert!(formatted.contains("Name"));
        assert!(formatted.contains("Alice"));
        assert!(formatted.contains("---"));
    }

    #[test]
    fn test_format_html() {
        let doc = parse_vlm_output_internal("<fcel>Name<ecel><ecel><nl><fcel>Alice<ecel><ecel>");
        let cfg = ParserConfig {
            format: Some(OutputFormat::Html),
            ..Default::default()
        };
        let formatted = format_document_internal(&doc, &cfg);

        assert!(formatted.contains("<table>"));
        assert!(formatted.contains("<th>"));
        assert!(formatted.contains("<td>"));
        assert!(formatted.contains("</table>"));
    }

    #[test]
    fn test_format_plain() {
        let doc = parse_vlm_output_internal("<fcel>Name<ecel><ecel><nl><fcel>Alice<ecel><ecel>");
        let cfg = ParserConfig {
            format: Some(OutputFormat::Plain),
            ..Default::default()
        };
        let formatted = format_document_internal(&doc, &cfg);

        assert!(!formatted.contains("|"));
        assert!(formatted.contains("Name"));
        assert!(formatted.contains("Alice"));
    }

    #[test]
    fn test_parse_paddle_response() {
        let text = "<fcel>Header<ecel><ecel><nl><fcel>Data<ecel><ecel>";
        let result = parse_paddle_response(
            text.to_string(),
            Some(ParserConfig {
                format: Some(OutputFormat::Markdown),
                ..Default::default()
            }),
        );

        assert!(result.contains("|"));
        assert!(result.contains("Header"));
        assert!(result.contains("Data"));
    }

    #[test]
    fn test_escape_html() {
        assert_eq!(escape_html("<script>"), "&lt;script&gt;");
        assert_eq!(escape_html("a & b"), "a &amp; b");
        assert_eq!(escape_html("\"quote\""), "&quot;quote&quot;");
    }

    #[test]
    fn test_handle_mixed_content() {
        let doc = parse_vlm_output_internal("Title<nl><fcel>Cell 1<ecel><ecel><nl>Footer text");
        // Should have at least 2 elements (paragraph before table, table, possibly paragraph after)
        assert!(doc.elements.len() >= 2);
    }

    #[test]
    fn test_format_raw() {
        let doc = parse_vlm_output_internal("Simple text<nl>More text");
        let cfg = ParserConfig {
            format: Some(OutputFormat::Raw),
            ..Default::default()
        };
        let formatted = format_document_internal(&doc, &cfg);

        assert!(formatted.contains("Simple text"));
        assert!(formatted.contains("More text"));
    }

    #[test]
    fn test_parse_and_format_default_markdown() {
        let text = "<fcel>A<ecel><ecel>";
        // Default config should use markdown
        let result = parse_paddle_response(text.to_string(), None);
        assert!(result.contains("|"));
    }

    #[test]
    fn test_real_world_ocr_output() {
        let text = "Trunch Parish Council<lcel><nl><ecel><ecel><nl>\
            <fcel>BANK RECONCILIATION AS AT 31ST OCTOBER 2019<lcel><nl>\
            <fcel>Account:<ecel><nl>\
            <fcel>BANK STATEMENT BALANCE £14,389.43<ecel>";

        let doc = parse_vlm_output_internal(text);

        // Should have at least one element
        assert!(!doc.elements.is_empty());

        // When formatted as markdown, should contain key text
        let cfg = ParserConfig {
            format: Some(OutputFormat::Markdown),
            ..Default::default()
        };
        let md = format_document_internal(&doc, &cfg);
        assert!(md.contains("Trunch Parish Council"));
        assert!(md.contains("BANK RECONCILIATION"));
    }

    #[test]
    fn test_utf8_multibyte_characters() {
        // Test with Chinese characters (3 bytes each in UTF-8)
        // This was the original bug: using byte indices with .chars().nth(i) corrupted text
        let text = "你好<fcel>中文<ecel><fcel>测试<rcel>";
        let row = parse_table_row(text);

        // Cells created:
        // 1. "你好" (content before <fcel>)
        // 2. "中文" (content before <ecel>)
        // 3. empty (from <ecel>)
        // 4. "测试" (content before <rcel>)
        assert_eq!(row.cells.len(), 4);
        assert_eq!(row.cells[0].content, "你好");
        assert!(!row.cells[0].is_empty);
        assert_eq!(row.cells[1].content, "中文");
        assert!(!row.cells[1].is_empty);
        assert!(row.cells[2].is_empty); // <ecel> creates empty cell
        assert_eq!(row.cells[3].content, "测试");
        assert!(!row.cells[3].is_empty);

        // Test full document parsing with UTF-8
        let doc_text = "<fcel>名前<ecel><nl><fcel>田中<ecel>";
        let doc = parse_vlm_output_internal(doc_text);

        assert_eq!(doc.elements.len(), 1);
        assert_eq!(doc.elements[0].element_type, ElementType::Table);

        let table = doc.elements[0].table.as_ref().unwrap();
        assert_eq!(table.rows.len(), 2);
        assert_eq!(table.rows[0].cells[0].content, "名前");
        assert_eq!(table.rows[1].cells[0].content, "田中");

        // Test markdown formatting preserves UTF-8
        let cfg = ParserConfig {
            format: Some(OutputFormat::Markdown),
            ..Default::default()
        };
        let md = format_document_internal(&doc, &cfg);
        assert!(md.contains("名前"));
        assert!(md.contains("田中"));
    }

    #[test]
    fn test_utf8_mixed_scripts() {
        // Test with mixed scripts: Japanese, emoji, and ASCII
        let text = "<fcel>こんにちは🌸<ecel><fcel>Hello世界<rcel>";
        let row = parse_table_row(text);

        // Cells created:
        // 1. "こんにちは🌸" (content before <ecel>)
        // 2. empty (from <ecel>)
        // 3. "Hello世界" (content before <rcel>)
        assert_eq!(row.cells.len(), 3);
        assert_eq!(row.cells[0].content, "こんにちは🌸");
        assert!(!row.cells[0].is_empty);
        assert!(row.cells[1].is_empty); // <ecel> creates empty cell
        assert_eq!(row.cells[2].content, "Hello世界");
        assert!(!row.cells[2].is_empty);
    }
}
