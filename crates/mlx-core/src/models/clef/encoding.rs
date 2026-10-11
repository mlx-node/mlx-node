//! Fragment-for-fragment port of Cloudflare's text CLEF record encoder.
use napi::{Error, Result};
use serde::{
    Deserialize, Deserializer,
    de::{MapAccess, Visitor},
};
use serde_json::Value;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    pub state: Value,
    #[serde(rename = "model")]
    pub _model: Option<String>,
    #[serde(deserialize_with = "ordered_questions")]
    pub questions: Vec<(String, Question)>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Question {
    #[serde(rename = "type")]
    pub kind: String,
    pub instructions: Option<Value>,
    pub criteria: Option<Value>,
}

// Value's sorted object map must NOT reorder the question sequence. Parsing
// raw HTTP JSON through this visitor also preserves integer-looking keys.
fn ordered_questions<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<Vec<(String, Question)>, D::Error> {
    struct Questions;
    impl<'de> Visitor<'de> for Questions {
        type Value = Vec<(String, Question)>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a nonempty question object")
        }
        fn visit_map<M: MapAccess<'de>>(
            self,
            mut map: M,
        ) -> std::result::Result<Self::Value, M::Error> {
            let mut entries = Vec::new();
            while let Some((id, question)) = map.next_entry::<String, Question>()? {
                if entries.iter().any(|(key, _)| key == &id) {
                    return Err(serde::de::Error::custom("duplicate question ID"));
                }
                entries.push((id, question));
                if entries.len() > 256 {
                    return Err(serde::de::Error::custom(
                        "at most 256 questions are supported",
                    ));
                }
            }
            Ok(entries)
        }
    }
    d.deserialize_map(Questions)
}

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct EncodedQuestion {
    pub id: String,
    pub kind: u32,
    pub span: (usize, usize),
    pub option_spans: Vec<(usize, usize)>,
    pub option_ids: Vec<String>,
}
#[derive(Debug, serde::Serialize)]
pub(crate) struct EncodedRecord {
    pub input_ids: Vec<u32>,
    pub questions: Vec<EncodedQuestion>,
}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::from_reason(format!("Invalid CLEF request: {}", message.into()))
}

impl Question {
    fn options(&self) -> Result<Vec<(String, Value)>> {
        match self.kind.as_str() {
            "noul" => {
                let criteria = match self.criteria.as_ref() {
                    None | Some(Value::Null) => None,
                    Some(Value::Object(map)) if map.keys().all(|k| k == "true" || k == "false") => {
                        Some(map)
                    }
                    _ => {
                        return Err(invalid(
                            "noul criteria must contain only true/false descriptions",
                        ));
                    }
                };
                Ok([
                    ("true", "The proposition is true or the answer is yes."),
                    ("false", "The proposition is false or the answer is no."),
                ]
                .into_iter()
                .map(|(id, default)| {
                    (
                        id.to_string(),
                        criteria
                            .and_then(|c| c.get(id))
                            .cloned()
                            .unwrap_or(Value::String(default.into())),
                    )
                })
                .collect())
            }
            "choice" => match self.criteria.as_ref().and_then(Value::as_object) {
                Some(map) if (1..=255).contains(&map.len()) => {
                    let mut options: Vec<_> =
                        map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                    options.sort_by(|a, b| a.0.cmp(&b.0));
                    Ok(options)
                }
                _ => Err(invalid("choice requires 1 to 255 options")),
            },
            "score" => match self.criteria.as_ref().and_then(Value::as_array) {
                Some(levels) if (2..=10).contains(&levels.len()) => Ok(levels
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i.to_string(), v.clone()))
                    .collect()),
                _ => Err(invalid("score requires 2 to 10 levels")),
            },
            _ => Err(invalid("question type must be noul, choice or score")),
        }
    }
}

// Python json.dumps uses Unicode code point ordering, compact separators and
// repr-style floating point notation. Keep that contract independent of JS's
// object enumeration and JSON.stringify number formatting.
fn json(value: &Value) -> String {
    match value {
        Value::Array(values) => format!(
            "[{}]",
            values.iter().map(json).collect::<Vec<_>>().join(",")
        ),
        Value::Object(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            format!(
                "{{{}}}",
                entries
                    .into_iter()
                    // serde_json cannot fail to serialize a &str.
                    .map(|(k, v)| {
                        let key = serde_json::to_string(k).unwrap_or_else(|_| format!("\"{k}\""));
                        format!("{key}:{}", json(v))
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
        Value::Number(n) if n.is_f64() => {
            let v = n.as_f64().unwrap_or_default();
            if v != 0.0 && (v.abs() < 1e-4 || v.abs() >= 1e16) {
                let s = format!("{v:e}");
                match s
                    .split_once('e')
                    .and_then(|(m, e)| e.parse::<i32>().ok().map(|exponent| (m, exponent)))
                {
                    Some((mantissa, exponent)) => format!("{mantissa}e{exponent:+03}"),
                    None => s,
                }
            } else {
                let s = format!("{v}");
                if s.contains('.') { s } else { format!("{s}.0") }
            }
        }
        _ => serde_json::to_string(value).unwrap_or_else(|_| "null".to_string()),
    }
}
fn render(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| json(value))
}

/// `tokenize` encodes every fragment in ONE call (the tokenizer's batch API)
/// and returns one id vector per fragment. Fragments stay independent —
/// BPE may not merge across a boundary — so the token stream is identical to
/// per-fragment `encode` calls, just without paying full pipeline overhead
/// for each of the ~5Q + 3·Σoptions fragments.
pub(crate) fn encode(
    request: &Request,
    tokenize: impl Fn(&[String]) -> Result<Vec<Vec<u32>>>,
    max_length: usize,
) -> Result<EncodedRecord> {
    if request.questions.is_empty() {
        return Err(invalid("questions must not be empty"));
    }
    // Fragment texts plus the mark identifying each span's fragment index.
    let mut frags: Vec<String> = vec!["\n\nSCHEMA FIELDS:\n".into()];
    let mut instruction_marks = Vec::with_capacity(request.questions.len());
    let mut option_marks = Vec::with_capacity(request.questions.len());
    let mut question_options = Vec::with_capacity(request.questions.len());
    for (index, (id, question)) in request.questions.iter().enumerate() {
        frags.push(format!(
            "\nFIELD {}\nID: {}\nTYPE: {}\nINSTRUCTION: ",
            index + 1,
            id,
            question.kind
        ));
        let instruction = question
            .instructions
            .as_ref()
            .filter(|v| !v.is_null() && v.as_str() != Some(""));
        instruction_marks.push(frags.len());
        frags.push(instruction.map(render).unwrap_or_else(|| id.clone()));
        frags.push("\nALLOWED OPTIONS:\n".into());
        let options = question.options()?;
        let mut marks = Vec::with_capacity(options.len());
        let mut option_ids = Vec::with_capacity(options.len());
        for (i, (option_id, description)) in options.into_iter().enumerate() {
            frags.push(format!("OPTION {}: ", i + 1));
            let mut semantics = serde_json::Map::new();
            semantics.insert("option_id".into(), Value::String(option_id.clone()));
            if !description.is_null() {
                semantics.insert("description".into(), description);
            }
            marks.push(frags.len());
            frags.push(json(&Value::Object(semantics)));
            frags.push("\n".into());
            option_ids.push(option_id);
        }
        frags.push("END FIELD\n".into());
        option_marks.push(marks);
        question_options.push(option_ids);
    }
    let prefix_mark = frags.len();
    frags.push(
        "<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n".into(),
    );
    let suffix_mark = frags.len();
    frags.push(
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
            .into(),
    );
    let state_mark = frags.len();
    frags.push(render(&request.state));

    let encoded = tokenize(&frags)?;
    if encoded.len() != frags.len() {
        return Err(Error::from_reason(
            "CLEF tokenizer returned a mismatched batch",
        ));
    }
    // Schema token lengths in emission order; marks below prefix_mark are schema.
    let mut schema_len = 0usize;
    let mut mark_offsets = vec![0usize; prefix_mark];
    for (i, ids) in encoded.iter().enumerate().take(prefix_mark) {
        mark_offsets[i] = schema_len;
        schema_len += ids.len();
    }
    if schema_len > max_length {
        return Err(invalid("schema exceeds the input token limit"));
    }
    let mut questions = Vec::with_capacity(request.questions.len());
    for (i, (id, question)) in request.questions.iter().enumerate() {
        let im = instruction_marks[i];
        let span = (mark_offsets[im], mark_offsets[im] + encoded[im].len());
        if span.0 == span.1 {
            return Err(invalid("question instruction produces no tokens"));
        }
        let option_spans = option_marks[i]
            .iter()
            .map(|&m| (mark_offsets[m], mark_offsets[m] + encoded[m].len()))
            .collect();
        questions.push(EncodedQuestion {
            id: id.clone(),
            kind: match question.kind.as_str() {
                "noul" => 0,
                "choice" => 1,
                _ => 2,
            },
            span,
            option_spans,
            option_ids: std::mem::take(&mut question_options[i]),
        });
    }
    let mut prefix = encoded[prefix_mark].clone();
    let suffix = &encoded[suffix_mark];
    let mut state = encoded[state_mark].clone();
    let fixed = prefix.len() + schema_len + suffix.len();
    if fixed > max_length {
        return Err(invalid(format!(
            "schema requires {fixed} tokens before state; maximum is {max_length}"
        )));
    }
    state.truncate(max_length - fixed);
    let offset = prefix.len() + state.len();
    for q in &mut questions {
        q.span.0 += offset;
        q.span.1 += offset;
        for span in &mut q.option_spans {
            span.0 += offset;
            span.1 += offset;
        }
    }
    prefix.extend(state);
    for ids in &encoded[..prefix_mark] {
        prefix.extend_from_slice(ids);
    }
    prefix.extend_from_slice(suffix);
    Ok(EncodedRecord {
        input_ids: prefix,
        questions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordering_rendering_and_truncation() {
        let char_tokens = |frags: &[String]| {
            Ok(frags
                .iter()
                .map(|s| s.chars().map(u32::from).collect())
                .collect::<Vec<Vec<u32>>>())
        };
        let request: Request = serde_json::from_str(r#"{"state":{"z":1e-7,"a":"你好"},"questions":{"10":{"type":"choice","criteria":{"z":null,"a":"yes"}},"2":{"type":"noul"}}}"#).unwrap();
        let encoded = encode(&request, char_tokens, 16384).unwrap();
        assert_eq!(encoded.questions[0].id, "10");
        assert_eq!(encoded.questions[1].id, "2");
        assert_eq!(encoded.questions[0].option_ids, ["a", "z"]);
        assert_eq!(json(&request.state), r#"{"a":"你好","z":1e-07}"#);
        let short = encode(&request, char_tokens, encoded.input_ids.len() - 3).unwrap();
        assert_eq!(short.input_ids.len(), encoded.input_ids.len() - 3);
        assert_eq!(short.questions[0].span.0, encoded.questions[0].span.0 - 3);
        assert!(encode(&request, char_tokens, 10).is_err());
    }
    #[test]
    fn duplicate_questions_and_media_fail_closed() {
        assert!(
            serde_json::from_str::<Request>(
                r#"{"state":"","questions":{"x":{"type":"noul"},"x":{"type":"noul"}}}"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<Request>(r#"{"state":"","questions":{},"images":[]}"#).is_err()
        );
    }
}

#[cfg(test)]
mod reference_tests {
    use super::*;
    #[test]
    fn bpe_fragments_match_cloudflare_reference_exactly() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../__test__/fixtures/clef");
        let tokenizer = tokenizers::Tokenizer::from_file(root.join("tokenizer.json")).unwrap();
        let fixtures: Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("encoding.json")).unwrap())
                .unwrap();
        for fixture in fixtures.as_array().unwrap() {
            let request: Request = serde_json::from_value(fixture["request"].clone()).unwrap();
            let actual = encode(
                &request,
                |frags| {
                    Ok(tokenizer
                        .encode_batch(frags.to_vec(), false)
                        .unwrap()
                        .iter()
                        .map(|e| e.get_ids().to_vec())
                        .collect())
                },
                fixture["max_length"].as_u64().unwrap() as usize,
            )
            .unwrap();
            let actual = serde_json::to_value(actual).unwrap();
            assert_eq!(actual["input_ids"], fixture["input_ids"]);
            assert_eq!(actual["questions"], fixture["questions"]);
        }
    }
}
