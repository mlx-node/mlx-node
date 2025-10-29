use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Row, RowAccessor};
use serde_json::json;

struct ColumnIndices {
    question: usize,
    answer: usize,
    ty: Option<usize>,
}

fn resolve_indices(row: &Row) -> std::result::Result<ColumnIndices, String> {
    let mut question = None;
    let mut answer = None;
    let mut ty = None;

    for (idx, (name, _)) in row.get_column_iter().enumerate() {
        match name.as_str() {
            "question" => question = Some(idx),
            "answer" => answer = Some(idx),
            "type" => ty = Some(idx),
            _ => {}
        }
    }

    Ok(ColumnIndices {
        question: question.ok_or_else(|| "Missing required column 'question'".to_string())?,
        answer: answer.ok_or_else(|| "Missing required column 'answer'".to_string())?,
        ty,
    })
}

fn convert_impl(input_path: &str, output_path: &str) -> std::result::Result<(), String> {
    let input = Path::new(input_path);
    if !input.exists() {
        return Err(format!("Parquet file does not exist: {input_path}"));
    }

    let reader = File::open(input)
        .map_err(|err| format!("Failed to open Parquet file {input_path}: {err}"))?;

    let reader = SerializedFileReader::new(reader)
        .map_err(|err| format!("Failed to load Parquet metadata from {input_path}: {err}"))?;

    let rows = reader
        .get_row_iter(None)
        .map_err(|err| format!("Failed to iterate rows from {input_path}: {err}"))?;

    let output_file = File::create(output_path)
        .map_err(|err| format!("Failed to create JSONL file {output_path}: {err}"))?;

    let mut writer = BufWriter::new(output_file);
    let mut col_indices: Option<ColumnIndices> = None;

    for (index, row_result) in rows.enumerate() {
        let row = row_result
            .map_err(|err| format!("Failed to read row {} from {input_path}: {err}", index + 1))?;

        if col_indices.is_none() {
            col_indices = Some(resolve_indices(&row)?);
        }
        let indices = col_indices.as_ref().unwrap();

        let question = row
            .get_string(indices.question)
            .map_err(|err| format!("Row {} column 'question' error: {err}", index + 1))?
            .to_owned();

        let answer = row
            .get_string(indices.answer)
            .map_err(|err| format!("Row {} column 'answer' error: {err}", index + 1))?
            .to_owned();

        let mut payload = serde_json::Map::with_capacity(3);
        payload.insert("question".to_string(), json!(question));
        payload.insert("answer".to_string(), json!(answer));

        if let Some(ty_idx) = indices.ty
            && let Ok(kind) = row.get_string(ty_idx)
        {
            payload.insert("type".to_string(), json!(kind));
        }

        serde_json::to_writer(&mut writer, &payload)
            .map_err(|err| format!("Failed to serialize row {}: {err}", index + 1))?;
        writer
            .write_all(b"\n")
            .map_err(|err| format!("Failed to write newline for row {}: {err}", index + 1))?;
    }

    writer
        .flush()
        .map_err(|err| format!("Failed to flush JSONL file {output_path}: {err}"))?;

    Ok(())
}

#[napi]
pub fn convert_parquet_to_jsonl(input_path: String, output_path: String) -> Result<()> {
    convert_impl(&input_path, &output_path).map_err(Error::from_reason)
}
