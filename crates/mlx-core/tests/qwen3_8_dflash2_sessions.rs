//! G1: deterministic 4-history harness for DFlash2 prefix reuse across sessions.
//!
//! Four recorded agent sessions (A = main, B/C/D = subagents) run through a
//! fresh `chat_session_start` per turn with a random cache owner, as the server
//! does, in the order `A1 B1 C1 D1 A2 B2 C2 D2 B3 A3 D3 C3 A4 B4 C4 D4`.
//! Each session then runs alone in its own process (serial reference).
//! Each turn's hit/miss fields come from its `dflash2_prefix` event. The test
//! runs every pass as a child process of itself, one model load at a time,
//! with `MLX_NODE_LOG=mlx_core::inference=info` and `MLX_NODE_LOG_FILE` set,
//! and reads each turn's event from that JSON log (the format of a server
//! `inference.log`), kept next to the report.
//!
//! ```shell
//! MLX_TEST_QWEN38_TARGET_PATH=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
//! MLX_TEST_QWEN38_DFLASH2_PATH=/path/to/qwen3.8-27b-dflash2 \
//! MLX_DFLASH2_GATE_OUT=/path/to/gates \
//! cargo test -p mlx-core --release --test qwen3_8_dflash2_sessions -- \
//!   --ignored --nocapture --test-threads=1
//! ```
//!
//! - `MLX_DFLASH2_SESSIONS_DIR`: the recorded sessions directory. Unset, the
//!   harness reads `~/.mlx-agent/sessions/%2FUsers%2Fbrooklyn%2Fworkspace%2Fgithub%2Fsplash`
//!   when it exists and otherwise falls back to synthetic text with the same
//!   sharing shape (B/C/D share one header; A shares 8 chars with them).
//! - `MLX_G1_STAGE` (default `S0`): the report goes to
//!   `$MLX_DFLASH2_GATE_OUT/<stage>/g1[-ar][-own][-restart].json` (without
//!   `MLX_DFLASH2_GATE_OUT`, under cargo's test tmpdir).
//! - `MLX_G1_MODE=ar`: the same schedule with no DFlash2 companion loaded.
//! - `MLX_G1_HISTORY=own` (report-only): earlier assistant rows carry the
//!   model's own outputs from the same pass instead of the recorded text.
//! - `MLX_G1_RESTART=1`: turn 1 of every session runs in one process and turn
//!   2 in a second process.
//!
//! Turn k of a session is every recorded row before its k-th assistant row.
//! The tool results answering one assistant row share a 4096-token budget.
//! Every assistant row keeps its reasoning: sessions render with
//! `preserve_thinking`, and the Qwen3.8 template keeps reasoning after the last
//! real user query anyway, so no server request strips it. The `rerender`
//! class comes from the recorded rows instead: they differ from the model's
//! own T=0 output, so each serial non-first turn diverges inside the previous
//! generation.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use mlx_core::engine::types::{ChatConfig, ChatResult};
use mlx_core::models::qwen3_5::model::{Qwen3_5Model, Qwen35LoadOptions};
use mlx_core::tokenizer::{
    ChatMessage, FunctionDefinition, FunctionParameters, ToolCall, ToolDefinition,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const SCHEDULE: &str = "A1 B1 C1 D1 A2 B2 C2 D2 B3 A3 D3 C3 A4 B4 C4 D4";
const SESSION_NAMES: [char; 4] = ['A', 'B', 'C', 'D'];
const TURNS: usize = 4;
const TOOL_STEP_TOKENS: usize = 4096;
const MAX_NEW_TOKENS: i32 = 64;
const SHARED_SYSTEM_CHARS: usize = 8;
const RECORDED_SESSION_IDS: [&str; 4] = [
    "01a185b5-5748-7721-a722-4cfc60df99a7",
    "01a185c3-418e-7ea1-92d5-f35e9a11f1ef",
    "01a185c3-418e-7ea1-92d5-f36748dac80a",
    "01a185c3-418e-7ea1-92d5-f37b419c4bc9",
];
const DEFAULT_SESSIONS_DIR: &str =
    ".mlx-agent/sessions/%2FUsers%2Fbrooklyn%2Fworkspace%2Fgithub%2Fsplash";
const TEST_NAME: &str = "qwen38_dflash2_sessions_g1";
const PHASE_ENV: &str = "MLX_G1_PHASE";
const PROBE_ENV: &str = "MLX_G1_LOG_PROBE";

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Row {
    User(String),
    Assistant {
        content: String,
        reasoning: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    ToolResult {
        call_id: String,
        content: String,
    },
}

struct Session {
    name: char,
    system: String,
    tools: Vec<ToolDefinition>,
    rows: Vec<Row>,
}

impl Session {
    fn assistant_rows(&self) -> usize {
        self.rows
            .iter()
            .filter(|row| matches!(row, Row::Assistant { .. }))
            .count()
    }
}

/// What the model said on an earlier turn, fed back in `MLX_G1_HISTORY=own`.
#[derive(Clone)]
struct OwnReply {
    content: String,
    reasoning: Option<String>,
}

fn text_of(value: &Value, at: &str) -> Result<String, String> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(blocks) => blocks
            .iter()
            .map(
                |block| match (block["type"].as_str(), block["text"].as_str()) {
                    (Some("text"), Some(text)) => Ok(text),
                    _ => Err(format!("{at}: unsupported content block {block}")),
                },
            )
            .collect::<Result<Vec<_>, _>>()
            .map(|parts| parts.concat()),
        other => Err(format!("{at}: unsupported content {other}")),
    }
}

/// Parse a Pi `chat_history.jsonl`. A `reasoning` row belongs to the
/// assistant row that follows it; the `system` row is replaced by the
/// session's `system_prompt.txt`.
fn parse_history(text: &str) -> Result<Vec<Row>, String> {
    let mut rows = Vec::new();
    let mut reasoning: Option<String> = None;
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let at = format!("line {}", index + 1);
        let row: Value = serde_json::from_str(line).map_err(|error| format!("{at}: {error}"))?;
        match row["type"].as_str() {
            Some("system") => {}
            Some("user") => rows.push(Row::User(text_of(&row["content"], &at)?)),
            Some("reasoning") => {
                let summary = row["summary"]
                    .as_array()
                    .ok_or_else(|| format!("{at}: reasoning row without summary"))?;
                let text = summary
                    .iter()
                    .map(|part| {
                        part["text"]
                            .as_str()
                            .ok_or_else(|| format!("{at}: reasoning summary without text"))
                    })
                    .collect::<Result<String, _>>()?;
                reasoning.get_or_insert_with(String::new).push_str(&text);
            }
            Some("assistant") => {
                let tool_calls = match row["tool_calls"].as_array() {
                    None => Vec::new(),
                    Some(calls) => calls
                        .iter()
                        .map(|call| {
                            Ok(ToolCall {
                                id: call["id"].as_str().map(str::to_string),
                                name: call["name"]
                                    .as_str()
                                    .ok_or_else(|| format!("{at}: tool call without name"))?
                                    .to_string(),
                                arguments: match &call["arguments"] {
                                    Value::String(arguments) => arguments.clone(),
                                    arguments => arguments.to_string(),
                                },
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?,
                };
                rows.push(Row::Assistant {
                    content: text_of(&row["content"], &at)?,
                    reasoning: reasoning.take().filter(|text| !text.is_empty()),
                    tool_calls,
                });
            }
            Some("tool_result") => rows.push(Row::ToolResult {
                call_id: row["tool_call_id"]
                    .as_str()
                    .ok_or_else(|| format!("{at}: tool_result without tool_call_id"))?
                    .to_string(),
                content: text_of(&row["content"], &at)?,
            }),
            other => return Err(format!("{at}: unsupported row type {other:?}")),
        }
    }
    Ok(rows)
}

/// Map `tool_definitions.json` the way the server's Anthropic mapper does:
/// `properties` becomes a JSON string, `required` stays a list.
fn parse_tools(text: &str) -> Result<Vec<ToolDefinition>, String> {
    let definitions: Vec<Value> = serde_json::from_str(text).map_err(|error| error.to_string())?;
    definitions
        .iter()
        .map(|definition| {
            let function = &definition["function"];
            let schema = &function["parameters"];
            Ok(ToolDefinition {
                r#type: "function".to_string(),
                function: FunctionDefinition {
                    name: function["name"]
                        .as_str()
                        .ok_or_else(|| format!("tool without name: {definition}"))?
                        .to_string(),
                    description: function["description"].as_str().map(str::to_string),
                    parameters: Some(FunctionParameters {
                        r#type: schema["type"].as_str().unwrap_or("object").to_string(),
                        properties: Some(
                            schema
                                .get("properties")
                                .cloned()
                                .unwrap_or_else(|| json!({}))
                                .to_string(),
                        ),
                        required: schema["required"].as_array().map(|required| {
                            required
                                .iter()
                                .filter_map(|name| name.as_str().map(str::to_string))
                                .collect()
                        }),
                    }),
                },
            })
        })
        .collect()
}

fn load_recorded_sessions(dir: &Path) -> Result<Vec<Session>, String> {
    RECORDED_SESSION_IDS
        .iter()
        .zip(SESSION_NAMES)
        .map(|(id, name)| {
            let session_dir = dir.join(id);
            let read = |file: &str| {
                let path = session_dir.join(file);
                std::fs::read_to_string(&path)
                    .map_err(|error| format!("{}: {error}", path.display()))
            };
            Ok(Session {
                name,
                system: read("system_prompt.txt")?,
                tools: parse_tools(&read("tool_definitions.json")?)?,
                rows: parse_history(&read("chat_history.jsonl")?)?,
            })
        })
        .collect()
}

fn synthetic_sessions() -> Vec<Session> {
    let main_system = format!(
        "You are the lead coding agent for this workspace. {}",
        "Plan the work, delegate research to subagents, and verify every claim before you report it. "
            .repeat(40)
    );
    let sub_system = format!(
        "You are a focused worker subagent. {}",
        "Complete the delegated task directly, use the tools, and report precise findings with file references. "
            .repeat(40)
    );
    let tools = |prefix: &str, count: usize| -> Vec<ToolDefinition> {
        (0..count)
            .map(|index| ToolDefinition {
                r#type: "function".to_string(),
                function: FunctionDefinition {
                    name: format!("{prefix}_tool_{index}"),
                    description: Some(format!(
                        "{prefix} tool {index}: {}",
                        "reads, searches and edits files in the workspace. ".repeat(8)
                    )),
                    parameters: Some(FunctionParameters {
                        r#type: "object".to_string(),
                        properties: Some(
                            json!({"path": {"type": "string", "description": "absolute path"}})
                                .to_string(),
                        ),
                        required: Some(vec!["path".to_string()]),
                    }),
                },
            })
            .collect()
    };
    let rows = |name: char, task: &str, tool: &str| -> Vec<Row> {
        let mut rows = vec![Row::User(format!(
            "{task} {}",
            "Report the control flow with file and line references. ".repeat(6)
        ))];
        for step in 0..TURNS {
            let call_id = format!("call_{name}_{step}");
            rows.push(Row::Assistant {
                content: format!("Step {step}: reading the next file."),
                reasoning: Some(format!(
                    "I need to inspect part {step} of the task before answering. {}",
                    "Check the callers and the error paths first. ".repeat(4)
                )),
                tool_calls: vec![ToolCall {
                    id: Some(call_id.clone()),
                    name: format!("{tool}_tool_0"),
                    arguments: json!({"path": format!("/workspace/{name}/{step}.rs")}).to_string(),
                }],
            });
            rows.push(Row::ToolResult {
                call_id,
                content: format!("fn step_{step}() {{ /* {task} */ }}\n").repeat(40),
            });
        }
        rows
    };
    vec![
        Session {
            name: 'A',
            system: main_system,
            tools: tools("main", 6),
            rows: rows('A', "Coordinate the study of the whole project.", "main"),
        },
        Session {
            name: 'B',
            system: sub_system.clone(),
            tools: tools("sub", 4),
            rows: rows('B', "Study the HTTP server layer.", "sub"),
        },
        Session {
            name: 'C',
            system: sub_system.clone(),
            tools: tools("sub", 4),
            rows: rows('C', "Study the native runtime layer.", "sub"),
        },
        Session {
            name: 'D',
            system: sub_system,
            tools: tools("sub", 4),
            rows: rows('D', "Study the installer and the desktop app.", "sub"),
        },
    ]
}

/// The premise every later G1 check leans on: junctions exist only among
/// B/C/D, and every session has enough recorded turns.
fn check_sharing_shape(sessions: &[Session]) -> Result<(), String> {
    let tools_json = |session: &Session| serde_json::to_string(&session.tools).unwrap();
    let [main, first_sub, subs @ ..] = sessions else {
        return Err(format!("expected 4 sessions, got {}", sessions.len()));
    };
    for sub in subs {
        if sub.system != first_sub.system || tools_json(sub) != tools_json(first_sub) {
            return Err(format!(
                "subagent {} does not share {}'s system prompt and tools",
                sub.name, first_sub.name
            ));
        }
    }
    let shared = main
        .system
        .chars()
        .zip(first_sub.system.chars())
        .take_while(|(left, right)| left == right)
        .count();
    if shared > SHARED_SYSTEM_CHARS || tools_json(main) == tools_json(first_sub) {
        return Err(format!(
            "main shares {shared} system chars (max {SHARED_SYSTEM_CHARS}) or its tools with the subagents"
        ));
    }
    for session in sessions {
        if session.assistant_rows() < TURNS {
            return Err(format!(
                "session {} has {} assistant rows, needs {TURNS}",
                session.name,
                session.assistant_rows()
            ));
        }
    }
    Ok(())
}

fn message(role: &str, content: String) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
        content,
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: None,
        thinking_enabled: None,
        images: None,
        audio: None,
    }
}

/// The request for turn `turn` (1-based): every row before the session's
/// `turn`-th assistant row. Tool results follow their assistant row in its
/// declared call order (the server's canonical order) and share
/// `TOOL_STEP_TOKENS`.
fn turn_messages(
    session: &Session,
    turn: usize,
    own: &[Option<OwnReply>],
    truncate: &dyn Fn(&str, usize) -> String,
) -> Result<Vec<ChatMessage>, String> {
    let end = session
        .rows
        .iter()
        .enumerate()
        .filter(|(_, row)| matches!(row, Row::Assistant { .. }))
        .nth(turn - 1)
        .map(|(index, _)| index)
        .ok_or_else(|| format!("session {} has no assistant row {turn}", session.name))?;
    let rows = &session.rows[..end];
    let mut messages = vec![message("system", session.system.clone())];
    let mut assistant_index = 0;
    let mut index = 0;
    while index < rows.len() {
        match &rows[index] {
            Row::User(text) => {
                messages.push(message("user", text.clone()));
                index += 1;
            }
            Row::ToolResult { call_id, .. } => {
                return Err(format!(
                    "session {}: tool result {call_id} does not follow an assistant row",
                    session.name
                ));
            }
            Row::Assistant {
                content,
                reasoning,
                tool_calls,
            } => {
                let (content, reasoning) = match own.get(assistant_index).and_then(Option::as_ref) {
                    Some(reply) => (reply.content.clone(), reply.reasoning.clone()),
                    None => (content.clone(), reasoning.clone()),
                };
                messages.push(ChatMessage {
                    tool_calls: (!tool_calls.is_empty()).then(|| tool_calls.clone()),
                    reasoning_content: reasoning,
                    ..message("assistant", content)
                });
                assistant_index += 1;
                index += 1;
                let start = index;
                while matches!(rows.get(index), Some(Row::ToolResult { .. })) {
                    index += 1;
                }
                let mut results: Vec<(&String, &String)> = rows[start..index]
                    .iter()
                    .filter_map(|row| match row {
                        Row::ToolResult { call_id, content } => Some((call_id, content)),
                        _ => None,
                    })
                    .collect();
                results.sort_by_key(|(call_id, _)| {
                    tool_calls
                        .iter()
                        .position(|call| call.id.as_deref() == Some(call_id.as_str()))
                        .unwrap_or(usize::MAX)
                });
                let budget = TOOL_STEP_TOKENS / results.len().max(1);
                for (call_id, content) in results {
                    messages.push(ChatMessage {
                        tool_call_id: Some(call_id.clone()),
                        ..message("tool", truncate(content, budget))
                    });
                }
            }
        }
    }
    Ok(messages)
}

/// `SCHEDULE` as (session index, 1-based turn) pairs.
fn schedule() -> Vec<(usize, usize)> {
    SCHEDULE
        .split_whitespace()
        .map(|label| {
            let mut chars = label.chars();
            let session = chars
                .next()
                .and_then(|name| SESSION_NAMES.iter().position(|&known| known == name))
                .unwrap_or_else(|| panic!("schedule label {label} names no session"));
            let turn = chars
                .as_str()
                .parse()
                .unwrap_or_else(|_| panic!("schedule label {label} has no turn"));
            (session, turn)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Event capture
// ---------------------------------------------------------------------------

type PrefixEvent = HashMap<String, String>;

/// The inference-log filter a G1 child process runs with.
const LOG_FILTER: &str = "mlx_core::inference=info";

/// `dflash2_prefix` events in JSON inference-log lines (the
/// `MLX_NODE_LOG_FILE` format, fields flattened onto the line).
fn prefix_events(log: &str) -> Vec<PrefixEvent> {
    log.lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|line| line["target"] == "mlx_core::inference" && line["event"] == "dflash2_prefix")
        .filter_map(|line| match line {
            Value::Object(fields) => Some(
                fields
                    .into_iter()
                    .map(|(name, value)| {
                        let value = match value {
                            Value::String(text) => text,
                            other => other.to_string(),
                        };
                        (name, value)
                    })
                    .collect(),
            ),
            _ => None,
        })
        .collect()
}

/// Reads the inference log from where the previous call stopped.
///
/// mlx-core installs its global tracing subscriber from a ctor
/// (`tracing.rs`, `#[module_init]`) before the test harness starts, so a test
/// cannot install its own layer; the harness runs in a child process whose
/// subscriber writes `mlx_core::inference` events to this file instead.
struct LogTail {
    path: PathBuf,
    offset: usize,
}

impl LogTail {
    fn new(path: PathBuf) -> LogTail {
        let offset = std::fs::metadata(&path).map_or(0, |meta| meta.len() as usize);
        LogTail { path, offset }
    }

    fn take(&mut self) -> Vec<PrefixEvent> {
        let bytes = std::fs::read(&self.path)
            .unwrap_or_else(|error| panic!("read {}: {error}", self.path.display()));
        let fresh = &bytes[self.offset.min(bytes.len())..];
        let complete = fresh
            .iter()
            .rposition(|&byte| byte == b'\n')
            .map_or(0, |end| end + 1);
        self.offset += complete;
        prefix_events(&String::from_utf8_lossy(&fresh[..complete]))
    }
}

/// Run this test binary's `test` in a child process whose inference log goes
/// to `log`.
fn run_logged_child(test: &str, log: &Path, envs: &[(&str, String)]) -> std::process::ExitStatus {
    if let Some(dir) = log.parent() {
        std::fs::create_dir_all(dir)
            .unwrap_or_else(|error| panic!("create {}: {error}", dir.display()));
    }
    std::fs::remove_file(log).ok();
    std::process::Command::new(std::env::current_exe().expect("test binary path"))
        .args([
            test,
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("MLX_NODE_LOG", LOG_FILTER)
        .env("MLX_NODE_LOG_FILE", log)
        .envs(envs.iter().map(|(name, value)| (*name, value.as_str())))
        .status()
        .unwrap_or_else(|error| panic!("spawn {test}: {error}"))
}

#[derive(Debug)]
struct TurnTrace {
    prompt_tokens: u64,
    cached_tokens: u64,
    prefilled_tokens: u64,
    gen_suffix_len: u64,
    replay_point: i64,
    hit_kind: String,
    hit_tokens: u64,
    miss_reason: String,
    shadow_hit: u64,
    ideal_tokens: u64,
    diverge_at: i64,
    ttft_ms: f64,
}

fn event_field<T: std::str::FromStr>(
    event: &PrefixEvent,
    label: &str,
    name: &str,
) -> Result<T, String> {
    let raw = event
        .get(name)
        .ok_or_else(|| format!("{label}: dflash2_prefix has no `{name}`"))?;
    raw.parse()
        .map_err(|_| format!("{label}: dflash2_prefix `{name}` = {raw:?} does not parse"))
}

/// The trace-completeness self-check: one event per turn, every field
/// present, and a classified reason.
fn turn_trace(label: &str, events: &[PrefixEvent]) -> Result<TurnTrace, String> {
    let [event] = events else {
        return Err(format!(
            "{label}: expected one dflash2_prefix event, got {}",
            events.len()
        ));
    };
    let trace = TurnTrace {
        prompt_tokens: event_field(event, label, "prompt_tokens")?,
        cached_tokens: event_field(event, label, "cached_tokens")?,
        prefilled_tokens: event_field(event, label, "prefilled_tokens")?,
        gen_suffix_len: event_field(event, label, "gen_suffix_len")?,
        replay_point: event_field(event, label, "replay_point")?,
        hit_kind: event_field(event, label, "hit_kind")?,
        hit_tokens: event_field(event, label, "hit_tokens")?,
        miss_reason: event_field(event, label, "miss_reason")?,
        shadow_hit: event_field(event, label, "shadow_hit")?,
        ideal_tokens: event_field(event, label, "ideal_tokens")?,
        diverge_at: event_field(event, label, "diverge_at")?,
        ttft_ms: event_field(event, label, "ttft_ms")?,
    };
    if trace.miss_reason == "unknown" {
        return Err(format!("{label}: miss reason is unknown: {trace:?}"));
    }
    Ok(trace)
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    DFlash2,
    Ar,
}

struct Options {
    mode: Mode,
    own_history: bool,
    restart: bool,
    stage: String,
    out_dir: PathBuf,
}

impl Options {
    fn from_env() -> Options {
        let var = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
        let mode = match var("MLX_G1_MODE").as_deref() {
            None | Some("dflash2") => Mode::DFlash2,
            Some("ar") => Mode::Ar,
            Some(other) => panic!("MLX_G1_MODE must be dflash2 or ar, got {other}"),
        };
        let own_history = match var("MLX_G1_HISTORY").as_deref() {
            None | Some("recorded") => false,
            Some("own") => true,
            Some(other) => panic!("MLX_G1_HISTORY must be recorded or own, got {other}"),
        };
        let restart = var("MLX_G1_RESTART").as_deref() == Some("1");
        assert!(
            !(restart && own_history),
            "MLX_G1_RESTART=1 cannot carry own outputs across processes"
        );
        let stage = var("MLX_G1_STAGE").unwrap_or_else(|| "S0".to_string());
        let out_dir = var("MLX_DFLASH2_GATE_OUT")
            .map_or_else(
                || PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("g1"),
                PathBuf::from,
            )
            .join(&stage);
        Options {
            mode,
            own_history,
            restart,
            stage,
            out_dir,
        }
    }

    fn report_name(&self) -> String {
        let mut name = String::from("g1");
        if self.mode == Mode::Ar {
            name.push_str("-ar");
        }
        if self.own_history {
            name.push_str("-own");
        }
        if self.restart {
            name.push_str("-restart");
        }
        name
    }
}

fn sessions_source() -> (String, Vec<Session>) {
    let dir = match std::env::var_os("MLX_DFLASH2_SESSIONS_DIR") {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            assert!(
                dir.is_dir(),
                "MLX_DFLASH2_SESSIONS_DIR {} is not a directory",
                dir.display()
            );
            Some(dir)
        }
        None => std::env::var_os("HOME")
            .map(|home| PathBuf::from(home).join(DEFAULT_SESSIONS_DIR))
            .filter(|dir| dir.is_dir()),
    };
    let (source, sessions) = match dir {
        Some(dir) => (
            dir.display().to_string(),
            load_recorded_sessions(&dir).unwrap_or_else(|error| panic!("{error}")),
        ),
        None => ("synthetic".to_string(), synthetic_sessions()),
    };
    check_sharing_shape(&sessions).unwrap_or_else(|error| panic!("{source}: {error}"));
    (source, sessions)
}

fn token_truncator(target: &Path) -> impl Fn(&str, usize) -> String {
    let dir = if target.is_file() {
        target.parent().unwrap_or(Path::new("."))
    } else {
        target
    };
    let path = dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&path)
        .unwrap_or_else(|error| panic!("load {}: {error}", path.display()));
    move |text, budget| {
        let encoding = tokenizer
            .encode(text, false)
            .unwrap_or_else(|error| panic!("encode tool result: {error}"));
        let ids = encoding.get_ids();
        if ids.len() <= budget {
            return text.to_string();
        }
        tokenizer
            .decode(&ids[..budget], false)
            .unwrap_or_else(|error| panic!("decode tool result: {error}"))
    }
}

fn sha256_hex(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

struct Harness<'a> {
    model: &'a Qwen3_5Model,
    options: &'a Options,
    sessions: &'a [Session],
    truncate: &'a dyn Fn(&str, usize) -> String,
    log: LogTail,
    errors: Vec<String>,
}

impl Harness<'_> {
    async fn run_turn(
        &mut self,
        pass: &str,
        session_index: usize,
        turn: usize,
        own: &[Option<OwnReply>],
    ) -> (Value, OwnReply) {
        let session = &self.sessions[session_index];
        let label = format!("{}{turn}", session.name);
        let messages = turn_messages(session, turn, own, self.truncate)
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        self.log.take();
        let result = self
            .model
            .chat_session_start(
                messages,
                Some(ChatConfig {
                    cache_owner_id: Some(uuid::Uuid::new_v4().to_string()),
                    max_new_tokens: Some(MAX_NEW_TOKENS),
                    temperature: Some(0.0),
                    report_performance: Some(true),
                    reuse_cache: Some(true),
                    enable_mtp: Some(self.options.mode == Mode::DFlash2),
                    tools: Some(session.tools.clone()),
                    ..ChatConfig::default()
                }),
            )
            .await
            .unwrap_or_else(|error| panic!("{pass} {label}: chat turn failed: {error}"));
        let events = self.log.take();
        let trace = if self.options.mode == Mode::DFlash2 || !events.is_empty() {
            match turn_trace(&format!("{pass} {label}"), &events) {
                Ok(trace) => Some(trace),
                Err(error) => {
                    self.errors.push(error);
                    None
                }
            }
        } else {
            None
        };
        let record = turn_record(pass, &label, session_index, turn, &result, trace.as_ref());
        eprintln!("[g1] {}", summary_line(&record));
        let reply = OwnReply {
            content: result.text.clone(),
            reasoning: result.thinking.clone(),
        };
        (record, reply)
    }

    async fn run_order(&mut self, pass: &str, order: &[(usize, usize)]) -> Vec<Value> {
        let mut own: Vec<Vec<Option<OwnReply>>> = vec![vec![None; TURNS]; self.sessions.len()];
        let mut records = Vec::with_capacity(order.len());
        for &(session, turn) in order {
            let history = if self.options.own_history {
                own[session].clone()
            } else {
                Vec::new()
            };
            let (record, reply) = self.run_turn(pass, session, turn, &history).await;
            own[session][turn - 1] = Some(reply);
            records.push(record);
        }
        records
    }
}

fn turn_record(
    pass: &str,
    label: &str,
    session: usize,
    turn: usize,
    result: &ChatResult,
    trace: Option<&TurnTrace>,
) -> Value {
    let perf = result.performance.as_ref();
    let cycles = perf.and_then(|perf| perf.mtp_cycles);
    let c_ms_per_cycle = perf.zip(cycles).and_then(|(perf, cycles)| {
        (cycles > 0 && perf.decode_tokens_per_second > 0.0).then(|| {
            result.num_tokens as f64 / perf.decode_tokens_per_second * 1000.0 / cycles as f64
        })
    });
    let prompt_tokens = trace.map_or(u64::from(result.prompt_tokens), |t| t.prompt_tokens);
    let hit_tokens = trace.map_or(u64::from(result.cached_tokens), |t| t.hit_tokens);
    json!({
        "pass": pass,
        "label": label,
        "session": SESSION_NAMES[session].to_string(),
        "turn": turn,
        "prompt_tokens": prompt_tokens,
        "result_prompt_tokens": result.prompt_tokens,
        "result_cached_tokens": result.cached_tokens,
        "hit_kind": trace.map(|t| t.hit_kind.clone()),
        "hit_tokens": hit_tokens,
        "prefilled_tokens": trace.map_or(prompt_tokens.saturating_sub(hit_tokens), |t| t.prefilled_tokens),
        "miss_reason": trace.map(|t| t.miss_reason.clone()),
        "shadow_hit": trace.map(|t| t.shadow_hit),
        "ideal_tokens": trace.map(|t| t.ideal_tokens),
        "ideal_prefill": trace.map(|t| t.prompt_tokens.saturating_sub(t.ideal_tokens)),
        "event_cached_tokens": trace.map(|t| t.cached_tokens),
        "replay_point": trace.map(|t| t.replay_point),
        "gen_suffix_len": trace.map(|t| t.gen_suffix_len),
        "diverge_at": trace.map(|t| t.diverge_at),
        "ttft_ms": trace.map(|t| t.ttft_ms),
        "perf_ttft_ms": perf.map(|perf| perf.ttft_ms),
        "num_tokens": result.num_tokens,
        "finish_reason": result.finish_reason,
        "output_sha256": sha256_hex(&result.raw_text),
        "mtp_cycles": cycles,
        "a_tokens_per_cycle": perf.and_then(|perf| perf.mtp_mean_accepted_tokens_total),
        "c_ms_per_cycle": c_ms_per_cycle,
    })
}

fn summary_line(record: &Value) -> String {
    format!(
        "{} {} prompt={} hit={}:{} prefilled={} reason={} shadow={} ideal_prefill={} ttft={}ms A={} C={}",
        record["pass"].as_str().unwrap_or("?"),
        record["label"].as_str().unwrap_or("?"),
        record["prompt_tokens"],
        record["hit_kind"].as_str().unwrap_or("-"),
        record["hit_tokens"],
        record["prefilled_tokens"],
        record["miss_reason"].as_str().unwrap_or("-"),
        record["shadow_hit"],
        record["ideal_prefill"],
        record["ttft_ms"],
        record["a_tokens_per_cycle"],
        record["c_ms_per_cycle"],
    )
}

/// Mark each interleaved turn against the serial run of the same turn. In
/// own-history mode the comparison stops at a session's first turn whose
/// resume point differs from the serial run's.
fn compare_with_serial(interleaved: &mut [Value], serial: &[Value], own_history: bool) {
    let serial_by_label: HashMap<&str, &Value> = serial
        .iter()
        .filter_map(|record| record["label"].as_str().map(|label| (label, record)))
        .collect();
    let mut diverged: HashMap<String, bool> = HashMap::new();
    let mut by_session: Vec<&mut Value> = interleaved.iter_mut().collect();
    by_session.sort_by_key(|record| {
        (
            record["session"].as_str().unwrap_or_default().to_string(),
            record["turn"].as_u64(),
        )
    });
    for record in by_session {
        let Some(reference) = record["label"]
            .as_str()
            .and_then(|label| serial_by_label.get(label))
        else {
            continue;
        };
        let same_resume = record["hit_tokens"] == reference["hit_tokens"];
        let same_output = record["output_sha256"] == reference["output_sha256"];
        let session = record["session"].as_str().unwrap_or_default().to_string();
        let compared = !(own_history && diverged.get(&session).copied().unwrap_or(false));
        if own_history && !same_resume {
            diverged.insert(session, true);
        }
        record["serial_same_resume_point"] = json!(same_resume);
        record["serial_output_equal"] = json!(compared.then_some(same_output));
    }
}

fn pass_summary(records: &[Value]) -> Value {
    let sum = |key: &str| records.iter().filter_map(|r| r[key].as_u64()).sum::<u64>();
    let mut reasons: BTreeMap<String, u64> = BTreeMap::new();
    for record in records {
        let reason = record["miss_reason"].as_str().unwrap_or("n/a").to_string();
        *reasons.entry(reason).or_default() += 1;
    }
    let count = |reason: &str| reasons.get(reason).copied().unwrap_or(0);
    let first_round = SESSION_NAMES.len();
    let later: Vec<&Value> = records.iter().skip(first_round).collect();
    let cross_session_with_shadow = later
        .iter()
        .filter(|r| {
            r["miss_reason"] == "cross_session"
                && r["shadow_hit"].as_u64().unwrap_or(0) * 10
                    >= r["prompt_tokens"].as_u64().unwrap_or(0) * 9
        })
        .count();
    json!({
        "turns": records.len(),
        "prompt_tokens": sum("prompt_tokens"),
        "hit_tokens": sum("hit_tokens"),
        "prefilled_tokens": sum("prefilled_tokens"),
        "ideal_prefill": sum("ideal_prefill"),
        "reasons": reasons,
        "unknown": count("unknown"),
        "state_existed_unused": count("state_existed_unused"),
        "after_first_round": later.len(),
        "after_first_round_cross_session_shadow_ge_90pct": cross_session_with_shadow,
        "serial_output_equal": records.iter().filter(|r| r["serial_output_equal"] == true).count(),
        "serial_output_compared": records.iter().filter(|r| r["serial_output_equal"].is_boolean()).count(),
    })
}

fn write_report(path: &Path, report: &Value) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .unwrap_or_else(|error| panic!("create {}: {error}", dir.display()));
    }
    std::fs::write(path, serde_json::to_string_pretty(report).unwrap())
        .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
    eprintln!("[g1] wrote {}", path.display());
}

/// One child process of the harness. Every serial session runs alone in its
/// own process, so nothing another session left in the model (ledger, caches,
/// stored states) can reach it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Interleaved,
    Serial(usize),
    Restart(usize),
}

impl Phase {
    fn name(self) -> String {
        match self {
            Phase::Interleaved => "interleaved".to_string(),
            Phase::Serial(session) => format!("serial-{}", SESSION_NAMES[session]),
            Phase::Restart(turn) => format!("restart-{turn}"),
        }
    }

    fn parse(name: &str) -> Phase {
        let all = (0..SESSION_NAMES.len()).map(Phase::Serial).chain([
            Phase::Interleaved,
            Phase::Restart(1),
            Phase::Restart(2),
        ]);
        all.into_iter()
            .find(|phase| phase.name() == name)
            .unwrap_or_else(|| panic!("unknown G1 phase {name}"))
    }

    fn pass(self) -> &'static str {
        match self {
            Phase::Interleaved => "interleaved",
            Phase::Serial(_) => "serial",
            Phase::Restart(_) => "restart",
        }
    }

    fn order(self) -> Vec<(usize, usize)> {
        match self {
            Phase::Interleaved => schedule(),
            Phase::Serial(session) => (1..=TURNS).map(|turn| (session, turn)).collect(),
            Phase::Restart(turn) => schedule()
                .into_iter()
                .filter(|&(_, scheduled)| scheduled == turn)
                .collect(),
        }
    }

    fn file(self, options: &Options, extension: &str) -> PathBuf {
        options.out_dir.join(format!(
            "{}.{}.{extension}",
            options.report_name(),
            self.name()
        ))
    }
}

/// The parent process loads nothing: it runs each phase in a logged child,
/// one model load at a time, and assembles the report.
fn run_parent(options: &Options) {
    let phases: Vec<Phase> = if options.restart {
        vec![Phase::Restart(1), Phase::Restart(2)]
    } else {
        std::iter::once(Phase::Interleaved)
            .chain((0..SESSION_NAMES.len()).map(Phase::Serial))
            .collect()
    };
    let mut failed = Vec::new();
    let mut parts = Vec::new();
    for phase in phases {
        let part = phase.file(options, "json");
        std::fs::remove_file(&part).ok();
        let status = run_logged_child(
            TEST_NAME,
            &phase.file(options, "inference.log"),
            &[(PHASE_ENV, phase.name())],
        );
        if !status.success() {
            failed.push(format!("{}: {status}", phase.name()));
        }
        match std::fs::read_to_string(&part) {
            Ok(text) => {
                parts.push((
                    phase,
                    serde_json::from_str::<Value>(&text).expect("phase JSON"),
                ));
                std::fs::remove_file(&part).ok();
            }
            Err(_) => {
                failed.push(format!("{}: wrote no report", phase.name()));
                break;
            }
        }
    }
    if let Some((_, first)) = parts.first() {
        let mut report = first["header"].clone();
        let records = |pass: &str| -> Vec<Value> {
            parts
                .iter()
                .filter(|(phase, _)| phase.pass() == pass)
                .flat_map(|(_, part)| part["turns"].as_array().cloned().unwrap_or_default())
                .collect()
        };
        let trace_errors: Vec<Value> = parts
            .iter()
            .flat_map(|(_, part)| part["trace_errors"].as_array().cloned().unwrap_or_default())
            .collect();
        report["inference_logs"] = json!(
            parts
                .iter()
                .map(|(_, part)| part["inference_log"].clone())
                .collect::<Vec<_>>()
        );
        if options.restart {
            let turns = records("restart");
            report["restart_summary"] = pass_summary(&turns);
            report["restart"] = json!(turns);
        } else {
            let mut interleaved = records("interleaved");
            let serial = records("serial");
            compare_with_serial(&mut interleaved, &serial, options.own_history);
            report["interleaved_summary"] = pass_summary(&interleaved);
            report["serial_summary"] = pass_summary(&serial);
            report["interleaved"] = json!(interleaved);
            report["serial"] = json!(serial);
        }
        report["trace_errors"] = json!(trace_errors);
        write_report(
            &options
                .out_dir
                .join(format!("{}.json", options.report_name())),
            &report,
        );
        for key in ["interleaved_summary", "serial_summary", "restart_summary"] {
            if !report[key].is_null() {
                eprintln!("[g1] {key}: {}", report[key]);
            }
        }
    }
    assert!(
        failed.is_empty(),
        "G1 phases failed:\n{}",
        failed.join("\n")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires real Qwen3.8 target and DFlash2 checkpoints"]
async fn qwen38_dflash2_sessions_g1() {
    let options = Options::from_env();
    let Some(phase) = std::env::var(PHASE_ENV)
        .ok()
        .map(|name| Phase::parse(&name))
    else {
        return run_parent(&options);
    };
    let log = PathBuf::from(
        std::env::var_os("MLX_NODE_LOG_FILE").expect("G1 child runs with MLX_NODE_LOG_FILE"),
    );
    let target =
        std::env::var("MLX_TEST_QWEN38_TARGET_PATH").expect("set MLX_TEST_QWEN38_TARGET_PATH");
    let draft = (options.mode == Mode::DFlash2).then(|| {
        std::env::var("MLX_TEST_QWEN38_DFLASH2_PATH").expect("set MLX_TEST_QWEN38_DFLASH2_PATH")
    });
    let (source, sessions) = sessions_source();
    let truncate = token_truncator(Path::new(&target));
    let model = Qwen3_5Model::load(
        target.clone(),
        Some(Qwen35LoadOptions {
            draft_model_path: draft.clone(),
            kv_format: Some("int8".to_string()),
            max_memory_bytes: None,
            max_context_tokens: None,
            decode_share: None,
            cache_disk: None,
            persistent_cache: None,
            cache_dir: None,
        }),
    )
    .await
    .expect("load Qwen3.8 target");
    if options.mode == Mode::DFlash2 {
        assert!(model.has_mtp_weights(), "DFlash2 companion must be loaded");
    }

    let mut harness = Harness {
        model: &model,
        options: &options,
        sessions: &sessions,
        truncate: &truncate,
        log: LogTail::new(log.clone()),
        errors: Vec::new(),
    };
    let records = harness.run_order(phase.pass(), &phase.order()).await;
    let errors = std::mem::take(&mut harness.errors);
    write_report(
        &phase.file(&options, "json"),
        &json!({
            "header": {
                "stage": options.stage,
                "mode": if options.mode == Mode::DFlash2 { "dflash2" } else { "ar" },
                "history": if options.own_history { "own" } else { "recorded" },
                "sessions": source,
                "target": target,
                "draft": draft,
                "schedule": SCHEDULE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "tool_step_tokens": TOOL_STEP_TOKENS,
            },
            "inference_log": log.display().to_string(),
            "turns": records,
            "trace_errors": errors,
        }),
    );
    model
        .shutdown()
        .expect("Qwen3.8 model thread must shut down cleanly");
    assert!(
        errors.is_empty(),
        "incomplete G1 trace:\n{}",
        errors.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Model-free checks of the harness itself
// ---------------------------------------------------------------------------

fn char_truncate(text: &str, budget: usize) -> String {
    text.chars().take(budget).collect()
}

fn complete_event(miss_reason: &str) -> PrefixEvent {
    [
        ("event", "dflash2_prefix"),
        ("prompt_tokens", "120"),
        ("cached_tokens", "0"),
        ("prefilled_tokens", "120"),
        ("gen_suffix_len", "3"),
        ("replay_point", "112"),
        ("hit_kind", "none"),
        ("hit_tokens", "0"),
        ("miss_reason", miss_reason),
        ("shadow_hit", "96"),
        ("ideal_tokens", "96"),
        ("diverge_at", "-1"),
        ("ttft_ms", "12.5"),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value.to_string()))
    .collect()
}

/// Emits probe events when spawned by
/// `log_capture_reads_prefix_events_from_the_model_thread`.
#[test]
#[ignore = "helper process for log_capture_reads_prefix_events_from_the_model_thread"]
fn emit_probe_prefix_events() {
    if std::env::var_os(PROBE_ENV).is_none() {
        return;
    }
    std::thread::spawn(|| {
        tracing::info!(
            target: "mlx_core::inference",
            event = "dflash2_prefix",
            prompt_tokens = 120usize,
            replay_point = -1i64,
            hit_kind = "none",
            miss_reason = "cross_session",
            ttft_ms = 12.5f64,
            "DFlash2 prefix"
        );
        tracing::info!(
            target: "mlx_core::inference",
            event = "decode_profile_summary",
            prompt_tokens = 7usize,
            "not a prefix event"
        );
        tracing::info!(
            target: "mlx_core::other",
            event = "dflash2_prefix",
            prompt_tokens = 9usize,
            "wrong target"
        );
    })
    .join()
    .unwrap();
}

#[test]
fn log_capture_reads_prefix_events_from_the_model_thread() {
    let log = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("g1")
        .join(format!("probe-{}.inference.log", std::process::id()));
    let status = run_logged_child(
        "emit_probe_prefix_events",
        &log,
        &[(PROBE_ENV, "1".to_string())],
    );
    assert!(status.success(), "probe child failed: {status}");
    let foreign = json!({"target": "mlx_core::other", "event": "dflash2_prefix"}).to_string();
    assert!(
        prefix_events(&foreign).is_empty(),
        "other targets are not prefix events"
    );
    let mut tail = LogTail::new(log.clone());
    assert!(
        tail.take().is_empty(),
        "a new tail starts at the end of the log"
    );
    let events = LogTail {
        path: log.clone(),
        offset: 0,
    }
    .take();
    std::fs::remove_file(&log).ok();
    assert_eq!(events.len(), 1, "{events:?}");
    let probe = &events[0];
    assert_eq!(probe["prompt_tokens"], "120");
    assert_eq!(probe["replay_point"], "-1");
    assert_eq!(probe["hit_kind"], "none");
    assert_eq!(probe["miss_reason"], "cross_session");
    assert_eq!(probe["ttft_ms"], "12.5");
}

#[test]
fn turn_trace_requires_one_classified_event() {
    let trace = turn_trace("A1", &[complete_event("cross_session")]).unwrap();
    assert_eq!(trace.miss_reason, "cross_session");
    assert_eq!(trace.prompt_tokens - trace.ideal_tokens, 24);
    assert_eq!(trace.replay_point, 112);
    assert!(turn_trace("A1", &[]).is_err());
    assert!(
        turn_trace(
            "A1",
            &[complete_event("none"), complete_event("cross_session")]
        )
        .is_err()
    );
    let unknown = turn_trace("A1", &[complete_event("unknown")]).unwrap_err();
    assert!(unknown.contains("unknown"), "{unknown}");
    let mut missing = complete_event("none");
    missing.remove("shadow_hit");
    assert!(turn_trace("A1", &[missing]).is_err());
}

#[test]
fn schedule_interleaves_rounds_and_keeps_session_order() {
    let order = schedule();
    assert_eq!(order.len(), SESSION_NAMES.len() * TURNS);
    for (round, chunk) in order.chunks(SESSION_NAMES.len()).enumerate() {
        let mut sessions: Vec<usize> = chunk.iter().map(|&(session, _)| session).collect();
        sessions.sort_unstable();
        assert_eq!(
            sessions,
            [0, 1, 2, 3],
            "round {round} must hold every session"
        );
        assert!(
            chunk.iter().all(|&(_, turn)| turn == round + 1),
            "round {round} must hold turn {}",
            round + 1
        );
    }
    let third: Vec<usize> = order[2 * SESSION_NAMES.len()..3 * SESSION_NAMES.len()]
        .iter()
        .map(|&(session, _)| session)
        .collect();
    assert_ne!(third, [0, 1, 2, 3], "round 3 must reorder the sessions");
}

#[test]
fn synthetic_sessions_have_the_recorded_sharing_shape() {
    let sessions = synthetic_sessions();
    check_sharing_shape(&sessions).unwrap();
    let mut broken = synthetic_sessions();
    broken[3].system.push('!');
    assert!(check_sharing_shape(&broken).is_err());
    let mut joined = synthetic_sessions();
    joined[0].system = joined[1].system.clone();
    assert!(check_sharing_shape(&joined).is_err());
}

#[test]
fn turn_messages_follow_recorded_rows() {
    let history = [
        json!({"type": "system", "content": "ignored"}),
        json!({"type": "user", "content": [{"type": "text", "text": "reminder"}]}),
        json!({"type": "user", "content": [{"type": "text", "text": "task "}, {"type": "text", "text": "one"}]}),
        json!({"type": "reasoning", "id": "", "summary": [{"type": "summary_text", "text": "think-1"}]}),
        json!({"type": "assistant", "content": "", "tool_calls": [
            {"id": "a", "name": "read", "arguments": "{\"p\":1}"},
            {"id": "b", "name": "read", "arguments": "{\"p\":2}"},
        ]}),
        json!({"type": "tool_result", "tool_call_id": "b", "content": "0123456789bbbb"}),
        json!({"type": "tool_result", "tool_call_id": "a", "content": "aaaa"}),
        json!({"type": "reasoning", "id": "", "summary": [{"type": "summary_text", "text": "think-2"}]}),
        json!({"type": "assistant", "content": "two", "tool_calls": [{"id": "c", "name": "grep", "arguments": "{}"}]}),
        json!({"type": "tool_result", "tool_call_id": "c", "content": "cccc"}),
        json!({"type": "reasoning", "id": "", "summary": [{"type": "summary_text", "text": "think-3"}]}),
        json!({"type": "assistant", "content": "three", "tool_calls": [{"id": "d", "name": "grep", "arguments": "{}"}]}),
        json!({"type": "tool_result", "tool_call_id": "d", "content": "dddd"}),
        json!({"type": "assistant", "content": "four"}),
    ]
    .iter()
    .map(Value::to_string)
    .collect::<Vec<_>>()
    .join("\n");
    let session = Session {
        name: 'B',
        system: "sys".to_string(),
        tools: Vec::new(),
        rows: parse_history(&history).unwrap(),
    };
    assert_eq!(session.assistant_rows(), 4);
    let roles = |messages: &[ChatMessage]| -> Vec<String> {
        messages.iter().map(|m| m.role.clone()).collect()
    };

    let first = turn_messages(&session, 1, &[], &char_truncate).unwrap();
    assert_eq!(roles(&first), ["system", "user", "user"]);
    assert_eq!(first[0].content, "sys");
    assert_eq!(first[2].content, "task one");

    let second = turn_messages(&session, 2, &[], &char_truncate).unwrap();
    assert_eq!(
        roles(&second),
        ["system", "user", "user", "assistant", "tool", "tool"]
    );
    assert_eq!(second[3].reasoning_content.as_deref(), Some("think-1"));
    assert_eq!(second[3].tool_calls.as_ref().map(Vec::len), Some(2));
    assert_eq!(second[4].tool_call_id.as_deref(), Some("a"));
    assert_eq!(second[5].tool_call_id.as_deref(), Some("b"));
    let halves = turn_messages(&session, 2, &[], &|text: &str, budget: usize| {
        format!("{budget}:{}", &text[..2])
    })
    .unwrap();
    assert_eq!(halves[5].content, format!("{}:01", TOOL_STEP_TOKENS / 2));

    let third = turn_messages(&session, 3, &[], &char_truncate).unwrap();
    assert_eq!(third.len(), 8);
    let reasoning: Vec<Option<&str>> = third
        .iter()
        .filter(|m| m.role == "assistant")
        .map(|m| m.reasoning_content.as_deref())
        .collect();
    assert_eq!(reasoning, [Some("think-1"), Some("think-2")]);

    let fourth = turn_messages(&session, 4, &[], &char_truncate).unwrap();
    let reasoning: Vec<Option<&str>> = fourth
        .iter()
        .filter(|m| m.role == "assistant")
        .map(|m| m.reasoning_content.as_deref())
        .collect();
    assert_eq!(
        reasoning,
        [Some("think-1"), Some("think-2"), Some("think-3")]
    );

    let own = [
        None,
        Some(OwnReply {
            content: "own-two".to_string(),
            reasoning: Some("own-think".to_string()),
        }),
    ];
    let with_own = turn_messages(&session, 4, &own, &char_truncate).unwrap();
    let assistants: Vec<(&str, Option<&str>)> = with_own
        .iter()
        .filter(|m| m.role == "assistant")
        .map(|m| (m.content.as_str(), m.reasoning_content.as_deref()))
        .collect();
    assert_eq!(
        assistants,
        [
            ("", Some("think-1")),
            ("own-two", Some("own-think")),
            ("three", Some("think-3"))
        ]
    );
    assert!(turn_messages(&session, 5, &[], &char_truncate).is_err());
}

#[test]
fn phases_round_trip_and_split_the_schedule() {
    let serial: Vec<Phase> = (0..SESSION_NAMES.len()).map(Phase::Serial).collect();
    for phase in
        serial
            .iter()
            .copied()
            .chain([Phase::Interleaved, Phase::Restart(1), Phase::Restart(2)])
    {
        assert!(Phase::parse(&phase.name()) == phase, "{}", phase.name());
    }
    let mut serial_turns: Vec<(usize, usize)> =
        serial.iter().flat_map(|phase| phase.order()).collect();
    let mut interleaved = Phase::Interleaved.order();
    serial_turns.sort_unstable();
    interleaved.sort_unstable();
    assert_eq!(
        serial_turns, interleaved,
        "serial passes run every scheduled turn once"
    );
    for session in 0..SESSION_NAMES.len() {
        let turns: Vec<usize> = Phase::Serial(session)
            .order()
            .iter()
            .map(|&(owner, turn)| {
                assert_eq!(owner, session);
                turn
            })
            .collect();
        assert_eq!(turns, [1, 2, 3, 4]);
    }
    assert_eq!(Phase::Restart(2).order(), [(0, 2), (1, 2), (2, 2), (3, 2)]);
}
