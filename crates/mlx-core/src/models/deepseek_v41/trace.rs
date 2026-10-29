//! Route trace (test harness, `DSV41_ROUTE_TRACE=<file>`): JSONL in the
//! `qwen4_exp/route_replay.rs` schema v1 (a header line, then one `routes`
//! event per layer of every call; `pass` = call, `entry` = layer order).

use std::io::Write;

use serde_json::json;

use super::moe::Routes;

pub struct RouteTrace {
    out: std::io::BufWriter<std::fs::File>,
    sequence: u64,
    pass: u64,
    prompt_tokens: usize,
}

impl RouteTrace {
    /// Opens `$DSV41_ROUTE_TRACE` when set (truncates it).
    pub fn from_env(
        prompt_tokens: usize,
        layers: usize,
        experts: usize,
        top_k: usize,
    ) -> std::io::Result<Option<Self>> {
        let Some(p) = std::env::var_os("DSV41_ROUTE_TRACE") else {
            return Ok(None);
        };
        let mut out = std::io::BufWriter::new(std::fs::File::create(p)?);
        let head = json!({
            // the kind `qwen4_exp/route_replay.rs` reads: the same schema and
            // replay tool serve both families
            "kind": "qwen4_exp_expert_route_trace",
            "schema_version": 1,
            "mode": "ar",
            "prompt_tokens": prompt_tokens.to_string(),
            "layer_count": layers,
            "expert_count": experts,
            "top_k": top_k,
        });
        writeln!(out, "{head}")?;
        Ok(Some(Self {
            out,
            sequence: 0,
            pass: 0,
            prompt_tokens,
        }))
    }

    /// One call's routes (layer order); `history_len` = tokens after the call.
    pub fn call(&mut self, routes: &[Routes], history_len: usize) -> std::io::Result<()> {
        let source = if history_len <= self.prompt_tokens {
            "prefill_device_commit"
        } else {
            "decode_device_commit"
        };
        for (layer, r) in routes.iter().enumerate() {
            let ev = json!({
                "kind": "routes",
                "sequence": self.sequence,
                "pass": self.pass,
                "entry": layer,
                "history_len": history_len,
                "source": source,
                "layer": layer,
                "rows": r.rows(),
                "top_k": r.k,
                "expert_ids": r.ids,
            });
            writeln!(self.out, "{ev}")?;
            self.sequence += 1;
        }
        self.pass += 1;
        self.out.flush()
    }
}
