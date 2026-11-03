//! Schema-driven tool-argument type coercion.
//!
//! Port of vLLM's `ParserEngine._fix_arg_types` / `coerce_to_schema_type`.
//! Templates that write argument values as bare text (Qwen3.5 `<parameter=…>`,
//! K2 `xml`, Muse-Glimmer `<atem:parameter>`) render `700` and `"700"`
//! identically, so only the request's tool schema can say which one the model
//! meant. Parsers stay schema-blind; this runs once at the NAPI boundary, where
//! `ChatConfig.tools` is still in scope.
//!
//! Deliberate differences from vLLM:
//! - A property with no recognizable type is left exactly as the parser
//!   produced it (vLLM treats it as `string`).
//! - An `object` / `array` candidate must parse to that container kind (vLLM
//!   accepts any JSON there).
//! - A typed `null` stays `null` (vLLM turns it into the string `"null"` under
//!   a string schema, which breaks `Option` fields in strict deserializers).

use std::sync::Arc;

use serde_json::{Map, Number, Value};

use super::ToolCallResult;
use crate::engine::types::{ChatConfig, ChatStreamChunk};
use crate::tokenizer::ToolDefinition;

/// Each tool's `parameters.properties` schema, keyed by tool name.
#[derive(Debug)]
pub(crate) struct ToolArgSchemas {
    by_name: std::collections::HashMap<String, Map<String, Value>>,
}

impl ToolArgSchemas {
    /// `None` when the request carries no tool with a usable `properties`
    /// schema, so schema-less requests skip coercion entirely.
    pub(crate) fn from_config(config: &ChatConfig) -> Option<Arc<Self>> {
        Self::from_tools(config.tools.as_deref()?).map(Arc::new)
    }

    fn from_tools(tools: &[ToolDefinition]) -> Option<Self> {
        let mut by_name = std::collections::HashMap::new();
        for tool in tools {
            let function = &tool.function;
            let Some(properties) = function
                .parameters
                .as_ref()
                .and_then(|p| p.properties.as_deref())
            else {
                continue;
            };
            match serde_json::from_str::<Value>(properties) {
                Ok(Value::Object(props)) if !props.is_empty() => {
                    by_name.entry(function.name.clone()).or_insert(props);
                }
                Ok(_) => {}
                Err(e) => tracing::debug!(
                    tool = %function.name,
                    error = %e,
                    "skipping tool-argument coercion: properties is not valid JSON"
                ),
            }
        }
        (!by_name.is_empty()).then_some(Self { by_name })
    }

    /// Coerce the arguments of every successfully parsed call whose tool name
    /// is known. Keys absent from the schema, or absent from a present
    /// `raw_argument_text`, are left untouched.
    pub(crate) fn coerce_calls(&self, calls: &mut [ToolCallResult]) {
        for call in calls {
            if call.status != "ok" {
                continue;
            }
            let Some(props) = self.by_name.get(&call.name) else {
                continue;
            };
            let Value::Object(args) = &mut call.arguments else {
                continue;
            };
            let raw_text = call.raw_argument_text.as_ref();
            for (key, value) in args.iter_mut() {
                let Some(prop) = props.get(key).filter(|p| p.is_object()) else {
                    continue;
                };
                match (raw_text.map(|m| m.get(key)), schema_types(prop)) {
                    (Some(None), _) => {}
                    (Some(Some(raw)), Some(types)) => {
                        *value = coerce_str(raw, types);
                        if value.is_object() || value.is_array() {
                            coerce_value(value, prop);
                        }
                    }
                    _ => coerce_value(value, prop),
                }
            }
        }
    }
}

/// [`ToolArgSchemas::coerce_calls`] for an optional schema set.
pub(crate) fn coerce_tool_calls(schemas: Option<&ToolArgSchemas>, calls: &mut [ToolCallResult]) {
    if let Some(schemas) = schemas {
        schemas.coerce_calls(calls);
    }
}

/// Coerce the tool calls of a streamed chunk. Only `Ok` chunks carrying
/// `tool_calls` (the terminal chunk) are touched.
pub(crate) fn coerce_chunk(
    schemas: Option<&ToolArgSchemas>,
    chunk: &mut napi::Result<ChatStreamChunk>,
) {
    if let Ok(chunk) = chunk
        && let Some(calls) = chunk.tool_calls.as_mut()
    {
        coerce_tool_calls(schemas, calls);
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
struct TypeSet(u8);

impl TypeSet {
    const NULL: u8 = 1;
    const INTEGER: u8 = 1 << 1;
    const NUMBER: u8 = 1 << 2;
    const BOOLEAN: u8 = 1 << 3;
    const OBJECT: u8 = 1 << 4;
    const ARRAY: u8 = 1 << 5;
    const STRING: u8 = 1 << 6;

    fn has(self, bit: u8) -> bool {
        self.0 & bit != 0
    }

    fn insert_name(&mut self, name: &str) {
        let name = name.trim().to_lowercase();
        self.0 |= match name.as_str() {
            "null" => Self::NULL,
            "integer" | "int" | "int32" | "int64" | "uint" | "uint32" | "uint64" | "long"
            | "short" | "unsigned" => Self::INTEGER,
            "number" | "float" | "float32" | "float64" | "double" => Self::NUMBER,
            "boolean" | "bool" => Self::BOOLEAN,
            "object" | "dict" => Self::OBJECT,
            "array" | "arr" | "list" | "sequence" => Self::ARRAY,
            "string" | "str" | "text" | "varchar" | "char" | "enum" => Self::STRING,
            _ => 0,
        };
    }

    fn insert_value_kind(&mut self, value: &Value) {
        self.0 |= match value {
            Value::Null => Self::NULL,
            Value::Bool(_) => Self::BOOLEAN,
            Value::Number(n) if n.is_i64() || n.is_u64() => Self::INTEGER,
            Value::Number(_) => Self::NUMBER,
            Value::String(_) => Self::STRING,
            Value::Array(_) => Self::ARRAY,
            Value::Object(_) => Self::OBJECT,
        };
    }
}

/// Types a property schema admits, from `type`, `enum` member kinds and
/// `anyOf` / `oneOf` / `allOf`. `None` when nothing recognizable is declared
/// (`$ref`-only, untyped, or unknown type names).
fn schema_types(schema: &Value) -> Option<TypeSet> {
    fn collect(schema: &Value, set: &mut TypeSet) {
        let Some(schema) = schema.as_object() else {
            return;
        };
        match schema.get("type") {
            Some(Value::String(t)) => set.insert_name(t),
            Some(Value::Array(ts)) => ts
                .iter()
                .filter_map(Value::as_str)
                .for_each(|t| set.insert_name(t)),
            _ => {}
        }
        if let Some(Value::Array(members)) = schema.get("enum") {
            members.iter().for_each(|m| set.insert_value_kind(m));
        }
        for key in ["anyOf", "oneOf", "allOf"] {
            if let Some(Value::Array(choices)) = schema.get(key) {
                choices.iter().for_each(|c| collect(c, set));
            }
        }
    }
    let mut set = TypeSet::default();
    collect(schema, &mut set);
    (set.0 != 0).then_some(set)
}

/// `Number.MAX_SAFE_INTEGER`. NAPI hands JS a `BigInt` for any integer beyond
/// it, and `JSON.stringify` throws on `BigInt`, failing the finished turn.
const MAX_SAFE_INTEGER: i64 = (1 << 53) - 1;

fn parse_integer(raw: &str) -> Option<Value> {
    raw.trim()
        .parse::<i64>()
        .ok()
        .filter(|n| n.unsigned_abs() <= MAX_SAFE_INTEGER as u64)
        .map(Value::from)
}

fn parse_number(raw: &str) -> Option<Value> {
    if let Some(n) = parse_integer(raw) {
        return Some(n);
    }
    if is_integer_literal(raw) {
        return None;
    }
    let f = raw.trim().parse::<f64>().ok().filter(|f| f.is_finite())?;
    if f.fract() == 0.0 && f.abs() <= MAX_SAFE_INTEGER as f64 {
        return Some(Value::from(f as i64));
    }
    Number::from_f64(f).map(Value::Number)
}

/// Coerce one bare-text value to the first type in the vLLM priority order
/// null > integer > number > boolean > object > array > string that accepts
/// it. If none does, the text is JSON-parsed, else kept verbatim.
fn coerce_str(raw: &str, types: TypeSet) -> Value {
    if types.has(TypeSet::NULL) && raw.to_lowercase() == "null" {
        return Value::Null;
    }
    if types.has(TypeSet::INTEGER)
        && let Some(n) = parse_integer(raw)
    {
        return n;
    }
    if types.has(TypeSet::NUMBER)
        && let Some(n) = parse_number(raw)
    {
        return n;
    }
    if types.has(TypeSet::BOOLEAN) {
        match raw.trim().to_lowercase().as_str() {
            "true" | "1" => return Value::Bool(true),
            "false" | "0" => return Value::Bool(false),
            _ => {}
        }
    }
    if types.has(TypeSet::OBJECT)
        && let Ok(v @ Value::Object(_)) = serde_json::from_str::<Value>(raw)
    {
        return v;
    }
    if types.has(TypeSet::ARRAY)
        && let Ok(v @ Value::Array(_)) = serde_json::from_str::<Value>(raw)
    {
        return v;
    }
    if types.has(TypeSet::STRING) {
        return Value::String(raw.to_string());
    }
    match serde_json::from_str::<Value>(raw) {
        Ok(Value::Number(_)) if is_integer_literal(raw) && parse_integer(raw).is_none() => {
            Value::String(raw.to_string())
        }
        Ok(v) => v,
        Err(_) => Value::String(raw.to_string()),
    }
}

fn is_integer_literal(raw: &str) -> bool {
    let t = raw.trim();
    let digits = t.strip_prefix(['-', '+']).unwrap_or(t);
    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
}

/// Coerce an already-typed value against its property schema, recursing
/// through `properties` and `items`.
fn coerce_value(value: &mut Value, schema: &Value) {
    let Some(schema_obj) = schema.as_object() else {
        return;
    };
    match value {
        Value::Null => {}
        Value::String(s) => {
            if let Some(types) = schema_types(schema) {
                *value = coerce_str(s, types);
            }
        }
        Value::Object(map) => {
            if let Some(Value::Object(props)) = schema_obj.get("properties") {
                coerce_map(map, props);
            }
        }
        Value::Array(items) => {
            if let Some(item_schema @ Value::Object(_)) = schema_obj.get("items") {
                for item in items {
                    coerce_value(item, item_schema);
                }
            }
        }
        Value::Number(_) | Value::Bool(_) => {
            if let Some(types) = schema_types(schema) {
                let coerced = coerce_str(&value.to_string(), types);
                if coerced != *value && (types.has(TypeSet::STRING) || !coerced.is_string()) {
                    *value = coerced;
                }
            }
        }
    }
}

fn coerce_map(args: &mut Map<String, Value>, props: &Map<String, Value>) {
    for (key, value) in args.iter_mut() {
        if let Some(prop) = props.get(key).filter(|p| p.is_object()) {
            coerce_value(value, prop);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::*;
    use crate::engine::types::ChatResult;
    use crate::tokenizer::{FunctionDefinition, FunctionParameters};

    fn tool(name: &str, properties: Option<&str>) -> ToolDefinition {
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: name.to_string(),
                description: None,
                parameters: Some(FunctionParameters {
                    r#type: "object".to_string(),
                    properties: properties.map(str::to_string),
                    required: None,
                }),
            },
        }
    }

    fn schemas(tools: &[ToolDefinition]) -> Arc<ToolArgSchemas> {
        ToolArgSchemas::from_config(&ChatConfig {
            tools: Some(tools.to_vec()),
            ..Default::default()
        })
        .expect("schemas")
    }

    fn types(schema: Value) -> TypeSet {
        schema_types(&schema).expect("typed schema")
    }

    #[test]
    fn coerce_str_follows_vllm_priority() {
        let int = types(json!({"type": "integer"}));
        assert_eq!(coerce_str("300000", int), json!(300000));
        assert_eq!(coerce_str(" 700\n", int), json!(700));
        assert_eq!(coerce_str("+5", int), json!(5));
        assert_eq!(
            coerce_str("18446744073709551615", int),
            json!("18446744073709551615")
        );
        assert_eq!(coerce_str("abc", int), json!("abc"));
        assert_eq!(coerce_str("1.5", int), json!(1.5));
        assert_eq!(
            coerce_str("123456789012345678901234567890", int),
            json!("123456789012345678901234567890")
        );

        let int_or_null = types(json!({"type": ["integer", "null"]}));
        assert_eq!(coerce_str("700", int_or_null), json!(700));
        assert_eq!(coerce_str("null", int_or_null), Value::Null);
        assert_eq!(coerce_str("NULL", int_or_null), Value::Null);
        let str_or_null = types(json!({"type": ["string", "null"]}));
        assert_eq!(coerce_str("null", str_or_null), Value::Null);
        assert_eq!(coerce_str("700", str_or_null), json!("700"));

        let num = types(json!({"type": "number"}));
        assert_eq!(coerce_str("1.5", num), json!(1.5));
        assert_eq!(coerce_str("1.0", num), json!(1));
        assert_eq!(coerce_str("42", num), json!(42));
        assert_eq!(coerce_str("inf", num), json!("inf"));
        assert_eq!(coerce_str("NaN", num), json!("NaN"));

        let boolean = types(json!({"type": "boolean"}));
        assert_eq!(coerce_str("true", boolean), json!(true));
        assert_eq!(coerce_str("True", boolean), json!(true));
        assert_eq!(coerce_str("1", boolean), json!(true));
        assert_eq!(coerce_str("FALSE", boolean), json!(false));
        assert_eq!(coerce_str("0", boolean), json!(false));
        assert_eq!(coerce_str("maybe", boolean), json!("maybe"));

        let string = types(json!({"type": "string"}));
        assert_eq!(coerce_str("700", string), json!("700"));
        assert_eq!(coerce_str("{\"a\": 1}", string), json!("{\"a\": 1}"));

        let object = types(json!({"type": "object"}));
        assert_eq!(coerce_str("{\"a\": 1}", object), json!({"a": 1}));
        assert_eq!(coerce_str("[1]", object), json!([1]));
        assert_eq!(coerce_str("not json", object), json!("not json"));
        let array = types(json!({"type": "array"}));
        assert_eq!(coerce_str("[1, \"x\"]", array), json!([1, "x"]));
        let array_or_string = types(json!({"type": ["array", "string"]}));
        assert_eq!(coerce_str("5", array_or_string), json!("5"));
    }

    #[test]
    fn coerce_str_never_produces_an_integer_outside_js_safe_range() {
        let int = types(json!({"type": "integer"}));
        let num = types(json!({"type": "number"}));
        let enum_int = types(json!({"enum": [1, 2]}));
        let boolean = types(json!({"type": "boolean"}));
        let object = types(json!({"type": "object"}));
        let int_or_string = types(json!({"type": ["integer", "string"]}));
        for set in [int, num, enum_int] {
            assert_eq!(
                coerce_str("9007199254740991", set),
                json!(9007199254740991_i64)
            );
            assert_eq!(
                coerce_str("-9007199254740991", set),
                json!(-9007199254740991_i64)
            );
        }
        for raw in [
            "9007199254740992",
            "9007199254740993",
            "-9007199254740992",
            "-9223372036854775808",
            "18446744073709551615",
            "+9007199254740993",
        ] {
            for set in [int, num, enum_int, boolean, object, int_or_string] {
                assert_eq!(coerce_str(raw, set), json!(raw), "{raw}");
            }
        }
        let big_float = coerce_str("1e17", num);
        assert!(big_float.as_f64() == Some(1e17) && !big_float.is_i64());
        assert_eq!(
            coerce_str("9007199254740990.0", num),
            json!(9007199254740990_i64)
        );
    }

    #[test]
    fn coerce_value_keeps_a_parser_produced_unsafe_integer() {
        let mut v = json!(9007199254740993_u64);
        coerce_value(&mut v, &json!({"type": "integer"}));
        assert_eq!(v, json!(9007199254740993_u64));
        let mut v = json!(9007199254740993_u64);
        coerce_value(&mut v, &json!({"type": "string"}));
        assert_eq!(v, json!("9007199254740993"));
    }

    #[test]
    fn schema_types_reads_aliases_enum_and_combinators() {
        let int = types(json!({"type": "integer"}));
        assert_eq!(types(json!({"type": "int"})), int);
        assert_eq!(types(json!({"type": " Int64 "})), int);
        assert_eq!(
            types(json!({"type": "bool"})),
            types(json!({"type": "boolean"}))
        );
        assert_eq!(types(json!({"enum": [1, 2]})), int);
        assert_eq!(
            types(json!({"enum": ["a", null]})),
            types(json!({"type": ["string", "null"]}))
        );
        assert_eq!(
            types(json!({"anyOf": [{"type": "integer"}, {"type": "null"}]})),
            types(json!({"type": ["integer", "null"]}))
        );
        assert_eq!(
            coerce_str("true", types(json!({"type": "bool"}))),
            json!(true)
        );
        assert_eq!(coerce_str("9", types(json!({"type": "int"}))), json!(9));

        assert_eq!(schema_types(&json!({"$ref": "#/$defs/x"})), None);
        assert_eq!(schema_types(&json!({"description": "untyped"})), None);
        assert_eq!(schema_types(&json!({"type": "mystery"})), None);
        assert_eq!(schema_types(&json!("integer")), None);
    }

    #[test]
    fn coerce_value_handles_typed_parser_output() {
        let mut v = json!("700");
        coerce_value(&mut v, &json!({"type": "integer"}));
        assert_eq!(v, json!(700));

        let mut v = json!(5);
        coerce_value(&mut v, &json!({"type": "string"}));
        assert_eq!(v, json!("5"));

        let mut v = json!(true);
        coerce_value(&mut v, &json!({"type": "string"}));
        assert_eq!(v, json!("true"));

        let mut v = json!(1);
        coerce_value(&mut v, &json!({"type": "boolean"}));
        assert_eq!(v, json!(true));

        let mut v = json!(1.5);
        coerce_value(&mut v, &json!({"type": "integer"}));
        assert_eq!(v, json!(1.5));

        let mut v = json!(1.0);
        coerce_value(&mut v, &json!({"type": "number"}));
        assert_eq!(v, json!(1));

        let mut v = Value::Null;
        coerce_value(&mut v, &json!({"type": "string"}));
        assert_eq!(v, Value::Null);

        let mut v = json!("42");
        coerce_value(&mut v, &json!({"$ref": "#/$defs/x"}));
        assert_eq!(v, json!("42"));
        let mut v = json!(42);
        coerce_value(&mut v, &json!({"description": "untyped"}));
        assert_eq!(v, json!(42));
    }

    #[test]
    fn coerce_value_recurses_into_properties_and_items() {
        let schema = json!({
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "opts": {
                    "type": "object",
                    "properties": {"verbose": {"type": "boolean"}, "name": {"type": "string"}}
                },
                "ids": {"type": "array", "items": {"type": "integer"}},
                "edits": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"line": {"type": "integer"}}}
                }
            }
        });
        let mut v = json!({
            "limit": "10",
            "opts": {"verbose": "true", "name": 7},
            "ids": ["1", "2", "x"],
            "edits": [{"line": "3", "text": "keep"}],
            "extra": "5"
        });
        coerce_value(&mut v, &schema);
        assert_eq!(
            v,
            json!({
                "limit": 10,
                "opts": {"verbose": true, "name": "7"},
                "ids": [1, 2, "x"],
                "edits": [{"line": 3, "text": "keep"}],
                "extra": "5"
            })
        );
    }

    fn raw_call(name: &str, args: Value, raw: &[(&str, &str)]) -> ToolCallResult {
        ToolCallResult::ok(name.to_string(), args, String::new()).with_raw_argument_text(
            raw.iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<HashMap<_, _>>(),
        )
    }

    #[test]
    fn coerce_calls_uses_raw_text_for_bare_text_formats() {
        let s = schemas(&[tool(
            "write",
            Some(
                r#"{"content": {"type": "string"}, "meta": {"type": "object", "properties": {"n": {"type": "integer"}}},
                    "n": {"type": ["integer", "null"]}, "free": {"description": "untyped"}}"#,
            ),
        )]);
        let mut calls = vec![raw_call(
            "write",
            json!({"content": {"a": 1}, "meta": "{\"n\": \"2\"}", "n": "null", "free": "9"}),
            &[
                ("content", "{\"a\": 1}"),
                ("meta", "{\"n\": \"2\"}"),
                ("n", "null"),
                ("free", "9"),
            ],
        )];
        s.coerce_calls(&mut calls);
        assert_eq!(
            calls[0].arguments,
            json!({"content": "{\"a\": 1}", "meta": {"n": 2}, "n": null, "free": "9"})
        );
    }

    #[test]
    fn coerce_calls_skips_unknown_and_non_ok_calls() {
        let s = schemas(&[tool("read_file", Some(r#"{"limit": {"type": "integer"}}"#))]);
        let mut calls = vec![
            raw_call("other_tool", json!({"limit": "1"}), &[("limit", "1")]),
            raw_call("read_file", json!({"unlisted": "1"}), &[("unlisted", "1")]),
            ToolCallResult::parse_error(
                "read_file".to_string(),
                "{\"limit\": \"1\"".to_string(),
                "bad".to_string(),
                String::new(),
            ),
        ];
        let before: Vec<Value> = calls.iter().map(|c| c.arguments.clone()).collect();
        s.coerce_calls(&mut calls);
        let after: Vec<Value> = calls.iter().map(|c| c.arguments.clone()).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn from_config_needs_a_usable_schema() {
        assert!(ToolArgSchemas::from_config(&ChatConfig::default()).is_none());
        let none = |tools: Vec<ToolDefinition>| {
            ToolArgSchemas::from_config(&ChatConfig {
                tools: Some(tools),
                ..Default::default()
            })
            .is_none()
        };
        assert!(none(vec![]));
        assert!(none(vec![tool("a", None)]));
        assert!(none(vec![tool("a", Some("{not json"))]));
        assert!(none(vec![tool("a", Some("{}"))]));

        let s = schemas(&[
            tool("broken", Some("{not json")),
            tool("dup", Some(r#"{"x": {"type": "integer"}}"#)),
            tool("dup", Some(r#"{"x": {"type": "string"}}"#)),
        ]);
        let mut calls = vec![raw_call("dup", json!({"x": "3"}), &[("x", "3")])];
        s.coerce_calls(&mut calls);
        assert_eq!(calls[0].arguments, json!({"x": 3}));
    }

    #[test]
    fn coerce_chunk_touches_only_ok_tool_call_chunks() {
        let s = schemas(&[tool("t", Some(r#"{"n": {"type": "integer"}}"#))]);
        let chunk = |tool_calls: Option<Vec<ToolCallResult>>| ChatStreamChunk {
            text: "x".to_string(),
            done: tool_calls.is_some(),
            finish_reason: None,
            tool_calls,
            thinking: None,
            thinking_enabled: None,
            num_tokens: None,
            prompt_tokens: None,
            reasoning_tokens: None,
            raw_text: None,
            public_raw_text: None,
            text_authoritative: None,
            cached_tokens: None,
            performance: None,
            is_reasoning: None,
        };

        let mut done = Ok(chunk(Some(vec![raw_call(
            "t",
            json!({"n": "4"}),
            &[("n", "4")],
        )])));
        coerce_chunk(Some(&s), &mut done);
        assert_eq!(
            done.unwrap().tool_calls.unwrap()[0].arguments,
            json!({"n": 4})
        );

        let mut text = Ok(chunk(None));
        coerce_chunk(Some(&s), &mut text);
        assert!(text.unwrap().tool_calls.is_none());

        let mut err: napi::Result<ChatStreamChunk> = Err(napi::Error::from_reason("boom"));
        coerce_chunk(Some(&s), &mut err);
        assert_eq!(err.unwrap_err().reason, "boom");

        let mut untouched = Ok(chunk(Some(vec![raw_call(
            "t",
            json!({"n": "4"}),
            &[("n", "4")],
        )])));
        coerce_chunk(None, &mut untouched);
        assert_eq!(
            untouched.unwrap().tool_calls.unwrap()[0].arguments,
            json!({"n": "4"})
        );
    }

    #[test]
    fn chat_session_call_result_coerces_tool_calls() {
        let s = schemas(&[tool(
            "get_command_or_subagent_output",
            Some(r#"{"timeout_ms": {"type": "integer"}}"#),
        )]);
        let (tx, rx) = tokio::sync::oneshot::channel();
        let call = crate::engine::types::ChatSessionCall {
            cancelled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            result_rx: std::sync::Mutex::new(Some(rx)),
            tool_schemas: Some(s),
        };
        let result = ChatResult {
            text: String::new(),
            tool_calls: vec![raw_call(
                "get_command_or_subagent_output",
                json!({"timeout_ms": "300000"}),
                &[("timeout_ms", "300000")],
            )],
            thinking: None,
            thinking_enabled: false,
            num_tokens: 0,
            prompt_tokens: 0,
            reasoning_tokens: 0,
            finish_reason: "tool_calls".to_string(),
            raw_text: String::new(),
            public_raw_text: None,
            cached_tokens: 0,
            performance: None,
        };
        tx.send(Ok(result)).unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let out = runtime.block_on(call.result()).unwrap();
        assert_eq!(out.tool_calls[0].arguments, json!({"timeout_ms": 300000}));
    }
}
