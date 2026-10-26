//! Offline partial-capacity replay for the P3.0-c route-trace schema.
//!
//! P3.0-c captured no valid real-model traces because its real runs were
//! memory-blocked. The checked-in expectations below are synthetic unit
//! traces, not real-model capacity evidence.

use super::route_plan;
use napi::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::io::BufRead;

const TARGET_SOURCES: [&str; 2] = ["decode_device_commit", "verify_device_commit"];
const PREFILL_SOURCES: [&str; 1] = ["prefill_device_commit"];
const HOST_SOURCES: [&str; 2] = ["host_read", "batch_host_read"];

fn err(message: impl std::fmt::Display) -> Error {
    Error::from_reason(message.to_string())
}

#[derive(Debug, Deserialize)]
struct TraceHeader {
    kind: String,
    schema_version: u64,
    mode: String,
    prompt_tokens: String,
    layer_count: usize,
    expert_count: usize,
    top_k: usize,
}

#[derive(Debug, Deserialize)]
struct TraceEvent {
    kind: String,
    sequence: u64,
    pass: u64,
    entry: u64,
    history_len: usize,
    source: String,
    layer: usize,
    rows: usize,
    top_k: usize,
    expert_ids: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Prefill,
    Target,
}

#[derive(Debug)]
struct Event {
    layer: usize,
    rows: usize,
    expert_ids: Vec<u32>,
    phase: Phase,
}

#[derive(Debug)]
struct Trace {
    mode: String,
    prompt_tokens: usize,
    layers: usize,
    experts: usize,
    events: Vec<Event>,
}

fn phase_for(source: &str, history_len: usize, prompt_tokens: usize) -> Result<Phase> {
    if TARGET_SOURCES.contains(&source) {
        return Ok(Phase::Target);
    }
    if PREFILL_SOURCES.contains(&source) {
        return Ok(Phase::Prefill);
    }
    if HOST_SOURCES.contains(&source) {
        return Ok(if history_len > prompt_tokens {
            Phase::Target
        } else {
            Phase::Prefill
        });
    }
    Err(err(format!("unknown route source: {source}")))
}

fn read_trace(reader: &mut impl BufRead, output_tokens: usize) -> Result<Trace> {
    if output_tokens == 0 {
        return Err(err("route replay requires at least one output token"));
    }
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Err(err("empty route trace"));
    }
    let header: TraceHeader = serde_json::from_str(&line).map_err(err)?;
    if header.kind != "qwen4_exp_expert_route_trace" {
        return Err(err("wrong route trace header kind"));
    }
    if header.schema_version != 1 {
        return Err(err(format!(
            "unsupported route trace schema: {}",
            header.schema_version
        )));
    }
    if !matches!(header.mode.as_str(), "ar" | "mtp") {
        return Err(err(format!("unknown route trace mode: {}", header.mode)));
    }
    if header.layer_count == 0 || header.expert_count == 0 || header.top_k == 0 {
        return Err(err("route trace dimensions must be positive"));
    }
    let prompt_tokens = header
        .prompt_tokens
        .parse::<usize>()
        .map_err(|_| err("header.prompt_tokens is not a nonnegative integer"))?;
    let layers = header.layer_count;
    let experts = header.expert_count;
    let top_k = header.top_k;
    let mut events = Vec::new();
    let mut pass_entries: BTreeMap<(String, u64), Vec<usize>> = BTreeMap::new();
    let mut source_rows: BTreeMap<String, usize> = BTreeMap::new();
    let mut expected_sequence = 0u64;
    let mut previous_pass = None;
    let mut previous_entry = 0u64;
    let mut line_number = 1usize;
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        line_number += 1;
        if !line.ends_with('\n') {
            return Err(err(format!(
                "route trace line {line_number}: unterminated event"
            )));
        }
        if line.trim().is_empty() {
            return Err(err(format!("route trace line {line_number}: blank event")));
        }
        let event: TraceEvent = serde_json::from_str(&line)
            .map_err(|error| err(format!("route trace line {line_number}: {error}")))?;
        if event.kind != "routes" {
            return Err(err(format!(
                "route trace line {line_number}: wrong event kind"
            )));
        }
        if event.sequence != expected_sequence {
            return Err(err(format!("route trace line {line_number}: sequence gap")));
        }
        expected_sequence += 1;
        if previous_pass.is_some_and(|previous| event.pass < previous) {
            return Err(err("route trace pass ordering went backwards"));
        }
        if previous_pass != Some(event.pass) {
            previous_entry = 0;
        }
        if event.entry != previous_entry {
            return Err(err(format!(
                "route trace line {line_number}: entry ordering gap"
            )));
        }
        previous_pass = Some(event.pass);
        previous_entry = event.entry + 1;
        if event.layer >= layers {
            return Err(err(format!(
                "route trace line {line_number}: invalid layer"
            )));
        }
        if event.top_k != top_k {
            return Err(err(format!(
                "route trace line {line_number}: inconsistent top_k"
            )));
        }
        let expected_ids = event
            .rows
            .checked_mul(event.top_k)
            .ok_or_else(|| err("route trace row width overflow"))?;
        if event.expert_ids.len() != expected_ids {
            return Err(err(format!(
                "route trace line {line_number}: row/top/id length mismatch"
            )));
        }
        if event
            .expert_ids
            .iter()
            .any(|&expert| expert as usize >= experts)
        {
            return Err(err(format!(
                "route trace line {line_number}: invalid expert id"
            )));
        }
        let phase = phase_for(&event.source, event.history_len, prompt_tokens)?;
        *source_rows.entry(event.source.clone()).or_default() += event.rows;
        pass_entries
            .entry((event.source.clone(), event.pass))
            .or_default()
            .push(event.layer);
        events.push(Event {
            layer: event.layer,
            rows: event.rows,
            expert_ids: event.expert_ids,
            phase,
        });
    }
    if events.is_empty() {
        return Err(err("route trace has no route events"));
    }
    for ((source, pass), entry_layers) in pass_entries {
        if TARGET_SOURCES.contains(&source.as_str())
            && (entry_layers.len() != layers || {
                let mut sorted = entry_layers;
                sorted.sort_unstable();
                sorted != (0..layers).collect::<Vec<_>>()
            })
        {
            return Err(err(format!("incomplete target pass {pass} for {source}")));
        }
    }
    let prefill_rows = events
        .iter()
        .filter(|event| event.phase == Phase::Prefill)
        .try_fold(0usize, |rows, event| rows.checked_add(event.rows))
        .ok_or_else(|| err("route trace prefill row overflow"))?;
    let expected_prefill = prompt_tokens
        .checked_mul(layers)
        .ok_or_else(|| err("route trace prompt row overflow"))?;
    if prefill_rows != expected_prefill {
        return Err(err(format!(
            "prefill route coverage {prefill_rows} differs from prompt x layers {expected_prefill}"
        )));
    }
    let target_layer_rows = events
        .iter()
        .filter(|event| event.phase == Phase::Target)
        .try_fold(0usize, |rows, event| rows.checked_add(event.rows))
        .ok_or_else(|| err("route trace target row overflow"))?;
    if target_layer_rows == 0 || !target_layer_rows.is_multiple_of(layers) {
        return Err(err(
            "route trace target rows do not form nonempty whole token rows",
        ));
    }
    let decode_rows = source_rows
        .get("decode_device_commit")
        .copied()
        .unwrap_or(0);
    let verify_rows = source_rows
        .get("verify_device_commit")
        .copied()
        .unwrap_or(0);
    if header.mode == "ar" {
        if decode_rows != (output_tokens - 1) * layers || verify_rows != 0 {
            return Err(err(
                "AR route rows are inconsistent with output tokens or contain verify rows",
            ));
        }
    } else if verify_rows == 0 {
        return Err(err("MTP route trace has no verify rows"));
    }
    Ok(Trace {
        mode: header.mode,
        prompt_tokens,
        layers,
        experts,
        events,
    })
}

#[derive(Debug)]
struct LruLayer {
    occupants: Vec<Option<u32>>,
    mapping: Vec<Option<usize>>,
    ages: Vec<u64>,
    tick: u64,
}

#[derive(Default, Debug)]
struct Access {
    hits: usize,
    misses: usize,
    unique: usize,
    bypassed: usize,
}

impl LruLayer {
    fn new(capacity: usize, experts: usize) -> Self {
        Self {
            occupants: vec![None; capacity],
            mapping: vec![None; experts],
            ages: vec![0; capacity],
            tick: 0,
        }
    }

    fn access(&mut self, ids: &[u32]) -> Result<Access> {
        let wanted: HashSet<u32> = ids.iter().copied().collect();
        let unique = wanted.len();
        if unique > self.occupants.len() {
            return Ok(Access {
                unique,
                bypassed: 1,
                ..Access::default()
            });
        }
        let mut missing: Vec<u32> = wanted
            .iter()
            .filter(|&&expert| self.mapping[expert as usize].is_none())
            .copied()
            .collect();
        missing.sort_unstable();
        let hits = unique - missing.len();
        for batch in missing.chunks(8) {
            let mut chosen = Vec::new();
            for &expert in batch {
                let mut best = None;
                for (slot, &occupant) in self.occupants.iter().enumerate() {
                    if chosen.contains(&slot)
                        || occupant.is_some_and(|occupant| wanted.contains(&occupant))
                    {
                        continue;
                    }
                    let age = if occupant.is_none() {
                        0
                    } else {
                        self.ages[slot] + 1
                    };
                    let key = (age, slot);
                    if best.is_none_or(|(_, best_key): (usize, (u64, usize))| key < best_key) {
                        best = Some((slot, key));
                    }
                }
                let slot = best
                    .map(|(slot, _)| slot)
                    .ok_or_else(|| err("LRU slot reservation exhausted"))?;
                chosen.push(slot);
                if let Some(old) = self.occupants[slot] {
                    self.mapping[old as usize] = None;
                }
                self.occupants[slot] = Some(expert);
                self.mapping[expert as usize] = Some(slot);
            }
        }
        for &expert in ids {
            let slot = self.mapping[expert as usize]
                .ok_or_else(|| err("LRU requested expert has no slot"))?;
            self.tick = self
                .tick
                .checked_add(1)
                .ok_or_else(|| err("LRU recency overflow"))?;
            self.ages[slot] = self.tick;
        }
        Ok(Access {
            hits,
            misses: missing.len(),
            unique,
            bypassed: 0,
        })
    }
}

#[derive(Default, Debug)]
struct Totals {
    trace_events: usize,
    events: usize,
    rows: usize,
    assignments: usize,
    unique: usize,
    hits: usize,
    misses: usize,
    bypassed: usize,
}

impl Totals {
    fn add(&mut self, access: Access) {
        self.events += 1;
        self.unique += access.unique;
        self.hits += access.hits;
        self.misses += access.misses;
        self.bypassed += access.bypassed;
    }
}

fn prefill_slot_calls(ids: &[u32], experts: usize, capacity: usize) -> Result<Vec<Vec<u32>>> {
    let order = route_plan::expert_order(ids, experts).map_err(err)?;
    if order.len() <= capacity {
        return Ok(vec![ids.to_vec()]);
    }
    let groups = route_plan::assignment_groups(ids, &order, experts, capacity).map_err(err)?;
    Ok(groups
        .into_iter()
        .map(|group| {
            group
                .into_iter()
                .map(|assignment| ids[assignment])
                .collect()
        })
        .collect())
}

#[derive(Debug, Serialize, PartialEq)]
struct ReplayResult {
    capacity: usize,
    policy: &'static str,
    prefill_trace_events: usize,
    prefill_events: usize,
    prefill_layer_rows: usize,
    prefill_rows: usize,
    prefill_unique_requests: usize,
    prefill_hits: usize,
    prefill_misses: usize,
    prefill_miss_bytes: u64,
    prefill_over_capacity_events: usize,
    target_events: usize,
    target_rows: usize,
    target_assignments: usize,
    target_unique_requests: usize,
    target_hits: usize,
    target_misses: usize,
    target_over_capacity_events: usize,
    output_tokens: usize,
    slot_cost_bytes: u64,
    target_miss_bytes: u64,
    target_miss_bytes_per_output_token: f64,
    target_miss_mib_per_output_token: f64,
    target_miss_bytes_per_target_row: f64,
    target_miss_mib_per_target_row: f64,
    total_miss_bytes: u64,
    total_miss_bytes_per_output_token: f64,
    total_miss_mib_per_output_token: f64,
}

fn replay(
    trace: &Trace,
    capacity: usize,
    include_prefill: bool,
    output_tokens: usize,
    slot_cost: u64,
) -> Result<ReplayResult> {
    if capacity == 0 || output_tokens == 0 {
        return Err(err("replay capacity and output tokens must be positive"));
    }
    let mut states: Vec<LruLayer> = (0..trace.layers)
        .map(|_| LruLayer::new(capacity, trace.experts))
        .collect();
    let mut prefill = Totals::default();
    let mut target = Totals::default();
    for event in &trace.events {
        if event.phase == Phase::Prefill && !include_prefill {
            continue;
        }
        let calls = if event.phase == Phase::Prefill {
            prefill.trace_events += 1;
            prefill.rows += event.rows;
            prefill_slot_calls(&event.expert_ids, trace.experts, capacity)?
        } else {
            vec![event.expert_ids.clone()]
        };
        for ids in calls {
            let access = states[event.layer].access(&ids)?;
            let bucket = if event.phase == Phase::Prefill {
                &mut prefill
            } else {
                &mut target
            };
            bucket.assignments += ids.len();
            if event.phase == Phase::Target {
                bucket.rows += event.rows;
            }
            bucket.add(access);
        }
    }
    if target.rows == 0 || !target.rows.is_multiple_of(trace.layers) {
        return Err(err("replay selected no whole target token rows"));
    }
    let target_rows = target.rows / trace.layers;
    let target_miss_bytes = (target.misses as u64)
        .checked_mul(slot_cost)
        .ok_or_else(|| err("target miss-byte overflow"))?;
    let prefill_miss_bytes = (prefill.misses as u64)
        .checked_mul(slot_cost)
        .ok_or_else(|| err("prefill miss-byte overflow"))?;
    let total_miss_bytes = target_miss_bytes
        .checked_add(prefill_miss_bytes)
        .ok_or_else(|| err("total miss-byte overflow"))?;
    let output = output_tokens as f64;
    let rows = target_rows as f64;
    let mib = (1024 * 1024) as f64;
    Ok(ReplayResult {
        capacity,
        policy: if include_prefill {
            "target_after_prefill"
        } else {
            "target_cold"
        },
        prefill_trace_events: prefill.trace_events,
        prefill_events: prefill.events,
        prefill_layer_rows: prefill.rows,
        prefill_rows: prefill.rows / trace.layers,
        prefill_unique_requests: prefill.unique,
        prefill_hits: prefill.hits,
        prefill_misses: prefill.misses,
        prefill_miss_bytes,
        prefill_over_capacity_events: prefill.bypassed,
        target_events: target.events,
        target_rows,
        target_assignments: target.assignments,
        target_unique_requests: target.unique,
        target_hits: target.hits,
        target_misses: target.misses,
        target_over_capacity_events: target.bypassed,
        output_tokens,
        slot_cost_bytes: slot_cost,
        target_miss_bytes,
        target_miss_bytes_per_output_token: target_miss_bytes as f64 / output,
        target_miss_mib_per_output_token: target_miss_bytes as f64 / output / mib,
        target_miss_bytes_per_target_row: target_miss_bytes as f64 / rows,
        target_miss_mib_per_target_row: target_miss_bytes as f64 / rows / mib,
        total_miss_bytes,
        total_miss_bytes_per_output_token: total_miss_bytes as f64 / output,
        total_miss_mib_per_output_token: total_miss_bytes as f64 / output / mib,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};

    const SLOT_COST: u64 = 8 * 1024 * 1024;

    /// One layer, two prompt rows and two AR decode rows. This is deliberately
    /// synthetic: no valid P3.0-c real trace exists to substitute for it.
    fn synthetic_trace() -> String {
        let header = serde_json::json!({
            "kind": "qwen4_exp_expert_route_trace",
            "schema_version": 1,
            "request": "synthetic-ar-r1",
            "mode": "ar",
            "case": "synthetic",
            "model_path": "synthetic.gguf",
            "inputs_path": "synthetic-inputs.json",
            "prompt_tokens": "2",
            "prompt_json_sha256": "synthetic",
            "prompt_u32le_sha256": "synthetic",
            "git_head": "synthetic",
            "process_id": 1,
            "layer_count": 1,
            "expert_count": 4,
            "top_k": 2,
        });
        let events = [
            serde_json::json!({"kind":"routes","sequence":0,"pass":0,"entry":0,"source":"prefill_device_commit","history_len":2,"layer":0,"rows":2,"top_k":2,"expert_ids":[0,1,2,3]}),
            serde_json::json!({"kind":"routes","sequence":1,"pass":1,"entry":0,"source":"decode_device_commit","history_len":2,"layer":0,"rows":1,"top_k":2,"expert_ids":[0,2]}),
            serde_json::json!({"kind":"routes","sequence":2,"pass":2,"entry":0,"source":"decode_device_commit","history_len":3,"layer":0,"rows":1,"top_k":2,"expert_ids":[1,3]}),
        ];
        std::iter::once(header)
            .chain(events)
            .map(|value| value.to_string() + "\n")
            .collect()
    }

    fn synthetic() -> Trace {
        read_trace(&mut BufReader::new(Cursor::new(synthetic_trace())), 3).unwrap()
    }

    #[test]
    fn synthetic_partial_replay_matches_the_p3_0_c_tool() {
        let trace = synthetic();
        assert_eq!((trace.mode.as_str(), trace.prompt_tokens), ("ar", 2));
        let cold = replay(&trace, 2, false, 3, SLOT_COST).unwrap();
        let warm = replay(&trace, 2, true, 3, SLOT_COST).unwrap();
        // Expected by replay_lru.py's LruLayer/prefill_slot_calls/replay on
        // this same synthetic trace: 4 cold misses, 3 warmed target misses,
        // and 4 prefill misses at two slots with an 8 MiB slot charge.
        assert_eq!(cold.target_misses, 4);
        assert_eq!(cold.target_miss_bytes, 4 * SLOT_COST);
        assert_eq!(cold.target_miss_mib_per_output_token, 32.0 / 3.0);
        assert_eq!(warm.prefill_misses, 4);
        assert_eq!(warm.target_misses, 3);
        assert_eq!(warm.target_miss_bytes, 3 * SLOT_COST);
        assert_eq!(warm.target_miss_mib_per_output_token, 8.0);
        assert_eq!(warm.total_miss_bytes, 7 * SLOT_COST);
        assert_eq!(warm.total_miss_mib_per_output_token, 56.0 / 3.0);
        assert_eq!(warm.prefill_events, 2);
        assert_eq!(warm.prefill_rows, 2);
        assert_eq!(warm.target_rows, 2);
        assert_eq!(warm.target_assignments, 4);
        let three = replay(&trace, 3, true, 3, SLOT_COST).unwrap();
        assert_eq!(three.target_misses, 2);
        assert_eq!(three.target_miss_mib_per_output_token, 16.0 / 3.0);
        assert_eq!(three.total_miss_mib_per_output_token, 16.0);
        eprintln!(
            "synthetic replay (not real-model evidence): capacity=2 cold={} MiB/token, after-prefill target={} MiB/token, total={} MiB/token; capacity=3 target={} MiB/token, total={} MiB/token",
            cold.target_miss_mib_per_output_token,
            warm.target_miss_mib_per_output_token,
            warm.total_miss_mib_per_output_token,
            three.target_miss_mib_per_output_token,
            three.total_miss_mib_per_output_token,
        );
    }

    #[test]
    fn trace_parser_enforces_schema_order_and_source_phases() {
        let invalid = synthetic_trace().replacen(
            "\"source\":\"decode_device_commit\"",
            "\"source\":\"unknown\"",
            1,
        );
        assert!(read_trace(&mut BufReader::new(Cursor::new(invalid)), 3).is_err());
        let invalid = synthetic_trace().replacen("\"sequence\":2", "\"sequence\":3", 1);
        assert!(read_trace(&mut BufReader::new(Cursor::new(invalid)), 3).is_err());
        let invalid = synthetic_trace().replacen("\"rows\":1", "\"rows\":2", 1);
        assert!(read_trace(&mut BufReader::new(Cursor::new(invalid)), 3).is_err());
        let unterminated = synthetic_trace().trim_end().to_owned();
        assert!(read_trace(&mut BufReader::new(Cursor::new(unterminated)), 3).is_err());

        let host = synthetic_trace().replace("prefill_device_commit", "host_read");
        let trace = read_trace(&mut BufReader::new(Cursor::new(host)), 3).unwrap();
        assert_eq!(trace.events[0].phase, Phase::Prefill);
        assert_eq!(phase_for("batch_host_read", 2, 2).unwrap(), Phase::Prefill);
        assert_eq!(phase_for("batch_host_read", 3, 2).unwrap(), Phase::Target);
    }

    /// Opt-in replay of a future valid P3.0-c capture. Expected output may be
    /// the tool's lru-results.json; QWEN4_EXP_REPLAY_LABEL selects its capture.
    #[test]
    #[ignore = "requires a valid real P3.0-c route trace; none was captured"]
    fn real_trace_partial_replay() {
        let path = std::env::var("QWEN4_EXP_ROUTE_TRACE").expect("QWEN4_EXP_ROUTE_TRACE");
        let output_tokens = std::env::var("QWEN4_EXP_REAL_STEPS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(256);
        let slot_cost = std::env::var("QWEN4_EXP_REPLAY_SLOT_COST")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(SLOT_COST);
        let capacities = std::env::var("QWEN4_EXP_REPLAY_CAPACITIES")
            .unwrap_or_else(|_| "128,192,256,384,512".into());
        let capacities: Vec<usize> = capacities
            .split(',')
            .map(|value| value.trim().parse().expect("invalid replay capacity"))
            .collect();
        let file = std::fs::File::open(&path).unwrap();
        let trace = read_trace(&mut BufReader::new(file), output_tokens).unwrap();
        let expected: Option<serde_json::Value> = std::env::var("QWEN4_EXP_REPLAY_EXPECTED")
            .ok()
            .map(|path| serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap());
        let label = std::env::var("QWEN4_EXP_REPLAY_LABEL").ok();
        for include_prefill in [false, true] {
            for &capacity in &capacities {
                let result =
                    replay(&trace, capacity, include_prefill, output_tokens, slot_cost).unwrap();
                let actual = serde_json::to_value(&result).unwrap();
                println!("{}", serde_json::to_string(&actual).unwrap());
                if let Some(expected) = &expected {
                    let policy = if include_prefill {
                        "target_after_prefill"
                    } else {
                        "target_cold"
                    };
                    let rows = expected["replays"].as_array().unwrap();
                    let expected: Vec<&serde_json::Value> = rows
                        .iter()
                        .filter(|row| {
                            row["capacity"] == capacity
                                && row["policy"] == policy
                                && label.as_ref().is_none_or(|label| row["label"] == *label)
                        })
                        .collect();
                    assert_eq!(expected.len(), 1, "expected replay row is not unique");
                    for (key, value) in actual.as_object().unwrap() {
                        assert_eq!(&expected[0][key], value, "replay mismatch for {key}");
                    }
                }
            }
        }
    }
}
